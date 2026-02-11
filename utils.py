import torch
import torch.nn.functional as F
import numpy as np
import copy
from torchvision import transforms
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
)
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
    BaseOutput,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (
    retrieve_timesteps,
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from diffusers.models import (
    AsymmetricAutoencoderKL,
    AutoencoderKL,
    ImageProjection,
    UNet2DConditionModel,
)
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    LoraLoaderMixin,
    TextualInversionLoaderMixin,
)
import PIL


def rearrange_3(tensor, f):
    F, D, C = tensor.size()
    return torch.reshape(tensor, (F // f, f, D, C))


def rearrange_4(tensor):
    B, F, D, C = tensor.size()
    return torch.reshape(tensor, (B * F, D, C))


class CrossFrameAttnProcessor:
    """
    Cross frame attention processor. Each frame attends the first frame.

    Args:
        batch_size: The number that represents actual batch size, other than the frames.
            For example, calling unet with a single prompt and num_images_per_prompt=1, batch_size should be equal to
            2, due to classifier-free guidance.
    """

    def __init__(self, batch_size=2):
        self.batch_size = batch_size

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Cross Frame Attention
        if not is_cross_attention:
            video_length = key.size()[0] // self.batch_size
            first_frame_index = [0] * video_length

            # rearrange keys to have batch and frames in the 1st and 2nd dims respectively
            key = rearrange_3(key, video_length)
            key = key[:, first_frame_index]
            # rearrange values to have batch and frames in the 1st and 2nd dims respectively
            value = rearrange_3(value, video_length)
            value = value[:, first_frame_index]

            # rearrange back to original shape
            key = rearrange_4(key)
            value = rearrange_4(value)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def coords_grid(batch, ht, wd, device):
    # Adapted from https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py
    coords = torch.meshgrid(
        torch.arange(ht, device=device), torch.arange(wd, device=device)
    )
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def warp_single_latent(latent, reference_flow):
    """
    Warp latent of a single frame with given flow

    Args:
        latent: latent code of a single frame
        reference_flow: flow which to warp the latent with

    Returns:
        warped: warped latent
    """
    _, _, H, W = reference_flow.size()
    _, _, h, w = latent.size()
    coords0 = coords_grid(1, H, W, device=latent.device).to(latent.dtype)

    coords_t0 = coords0 + reference_flow
    coords_t0[:, 0] /= W
    coords_t0[:, 1] /= H

    coords_t0 = coords_t0 * 2.0 - 1.0
    coords_t0 = F.interpolate(coords_t0, size=(h, w), mode="bilinear")
    coords_t0 = torch.permute(coords_t0, (0, 2, 3, 1))

    warped = F.grid_sample(latent, coords_t0, mode="nearest", padding_mode="reflection")
    return warped


def create_motion_field(
    motion_field_strength_x, motion_field_strength_y, frame_ids, device, dtype
):
    """
    Create translation motion field

    Args:
        motion_field_strength_x: motion strength along x-axis
        motion_field_strength_y: motion strength along y-axis
        frame_ids: indexes of the frames the latents of which are being processed.
            This is needed when we perform chunk-by-chunk inference
        device: device
        dtype: dtype

    Returns:

    """
    seq_length = len(frame_ids)
    reference_flow = torch.zeros((seq_length, 2, 512, 512), device=device, dtype=dtype)
    for fr_idx in range(seq_length):
        reference_flow[fr_idx, 0, :, :] = motion_field_strength_x * (frame_ids[fr_idx])
        reference_flow[fr_idx, 1, :, :] = motion_field_strength_y * (frame_ids[fr_idx])
    return reference_flow


def create_motion_field_z_translation(
    motion_strength, frame_ids, device, dtype, height=512, width=512
):
    """
    Create motion field simulating camera movement along the Z-axis (forward/backward).

    Args:
        motion_strength: float, control strength of forward/backward movement (>0 = forward)
        frame_ids: list of frame indices
        device: torch device
        dtype: torch dtype
        height, width: size of the flow field

    Returns:
        reference_flow: (seq_length, 2, H, W) tensor of flow vectors
    """
    seq_length = len(frame_ids)
    coords = coords_grid(1, height, width, device=device)[0]  # (2, H, W)

    center_x = width / 2
    center_y = height / 2

    # Create a vector pointing from center to each pixel
    x_offset = coords[0] - center_x  # (H, W)
    y_offset = coords[1] - center_y

    # Normalize to unit vector field
    norm = torch.sqrt(x_offset ** 2 + y_offset ** 2 + 1e-8)
    unit_x = x_offset / norm
    unit_y = y_offset / norm

    reference_flow = torch.zeros((seq_length, 2, height, width), device=device, dtype=dtype)

    for fr_idx in range(seq_length):
        scale = motion_strength * frame_ids[fr_idx]
        reference_flow[fr_idx, 0] = unit_x * scale
        reference_flow[fr_idx, 1] = unit_y * scale

    return reference_flow


def create_motion_field_and_warp_latents_xy(
    latents, motion_field_strength_x=12, motion_field_strength_y=12, video_lenght=2
):
    motion_field = create_motion_field(
        motion_field_strength_x=motion_field_strength_x,
        motion_field_strength_y=motion_field_strength_y,
        frame_ids=list(range(video_lenght))[1:],
        device=latents.device,
        dtype=latents.dtype,
    )
    warped_latents = latents.clone().detach()
    for i in range(len(warped_latents)):
        warped_latents[i] = warp_single_latent(latents[i][None], motion_field[i][None])
    return warped_latents


def create_motion_field_and_warp_latents_z(
    latents, motion_field_strength_z=12, video_lenght=2
):
    motion_field = create_motion_field_z_translation(
        motion_strength=motion_field_strength_z,
        frame_ids=list(range(video_lenght))[1:],
        device=latents.device,
        dtype=latents.dtype,
    )
    warped_latents = latents.clone().detach()
    for i in range(len(warped_latents)):
        warped_latents[i] = warp_single_latent(latents[i][None], motion_field[i][None])
    return warped_latents


def max_warp_latents(latents, x=0, y=0, z=0):
    max_loss, max_x, max_y, max_z = 0, 0, 0, 0
    
    for k in np.arange(-z, z + 1):
        if k < z // 2 and k > -z // 2:
            continue
        warped_latents = create_motion_field_and_warp_latents_z(
            latents, motion_field_strength_z=k
        )
        loss = torch.abs(warped_latents - latents).mean().item()
        if loss > max_loss:
            max_loss = loss
            max_z = k
            warped_latents_z = warped_latents
            
    if max_z != 0 :
        latents = warped_latents_z

    for i in np.arange(-x, x + 1):
        if i < x // 2 and i > -x // 2:
            continue
        warped_latents = create_motion_field_and_warp_latents_xy(
            latents, motion_field_strength_x=i, motion_field_strength_y=0,
        )
        loss = torch.abs(warped_latents - latents).mean().item()
        if loss > max_loss:
            max_loss = loss
            max_x = i
            warped_latents_x = warped_latents

    max_loss = 0
    for j in np.arange(-y, y + 1):
        if j < y // 2 and j > -y // 2:
            continue
        warped_latents = create_motion_field_and_warp_latents_xy(
            latents, motion_field_strength_x=0, motion_field_strength_y=j
        )
        loss = torch.abs(warped_latents - latents).mean().item()
        if loss > max_loss:
            max_loss = loss
            max_y = j
            warped_latents_x_y = warped_latents

    print("max_x, max_y, max_z", max_x, max_y, max_z)
    return max_x, max_y, max_z
    # return warped_latents_x_y


def generate_signed_noise(latents, a=1.0):
    z = torch.randn(latents.shape, dtype=latents.dtype, device=latents.device)
    sign_mask = torch.sign(latents)
    z = torch.abs(z) * sign_mask * a
    return z


class MyStableDiffusionPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            image_encoder,
            requires_safety_checker,
        )
        self.unet.set_attn_processor((CrossFrameAttnProcessor(batch_size=2)))  # frame attention

    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        xyz=[40, 40, 0],
        warped_latents_timestep=None,
        noise_a=1.0,
        custom_timesteps=None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        if custom_timesteps is not None:
            timesteps = torch.tensor(custom_timesteps, device=timesteps.device, dtype=timesteps.dtype)
            num_inference_steps = len(custom_timesteps)
            self.scheduler.timesteps = timesteps
            self.scheduler.num_inference_steps = num_inference_steps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels

        # current frame + next frame
        if num_images_per_prompt == 2:
            latents = self.prepare_latents(
                batch_size * 1,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
            
            # warped_latents = create_motion_field_and_warp_latents(latents, motion_field_strength_x=xy[0], motion_field_strength_y=xy[1])
            # warped_latents = max_warp_latents(latents, x=xy[0], y=xy[1])
            
            warped_latents = latents.detach()
            max_x, max_y, max_z = xyz[0], xyz[1], xyz[2]
            max_x, max_y, max_z = max_warp_latents(warped_latents, x=xyz[0], y=xyz[1], z=xyz[2])
            
            warped_latents = create_motion_field_and_warp_latents_z(
                warped_latents, motion_field_strength_z=max_z
            )
            
            warped_latents = create_motion_field_and_warp_latents_xy(
                warped_latents, motion_field_strength_x=max_x, motion_field_strength_y=max_y
            )

            warped_latents_noise = torch.randn(
                latents.shape,
                device=latents.device,
                dtype=latents.dtype,
            )
            warped_latents = self.scheduler.add_noise(
                warped_latents, warped_latents_noise, warped_latents_timestep
            )

            latents = torch.cat([latents, warped_latents])
        else:
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )


# ============================================================
# Tree-Ring Watermark Related Classes and Functions
# ============================================================

class ModifiedStableDiffusionPipelineOutput(BaseOutput):
    """Output class for modified stable diffusion pipeline."""
    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]
    init_latents: Optional[torch.FloatTensor]


class ModifiedStableDiffusionPipeline(StableDiffusionPipeline):
    """
    Modified Stable Diffusion Pipeline for Tree-Ring watermarking.
    Exactly matches the reference implementation from TreeRingRand.
    """
    
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        image_encoder=None,
        requires_safety_checker: bool = True,
    ):
        super(ModifiedStableDiffusionPipeline, self).__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            image_encoder,
            requires_safety_checker,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        watermarking_gamma: float = None,
        watermarking_delta: float = None,
        watermarking_mask: Optional[torch.BoolTensor] = None,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        self.check_inputs(prompt, height, width, callback_steps)

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        text_embeddings = torch.concat(
            [
                self._encode_prompt(
                    p,
                    device,
                    num_images_per_prompt,
                    do_classifier_free_guidance,
                    negative_prompt,
                )
                for p in prompt
            ]
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        init_latents = copy.deepcopy(latents)

        # Watermarking mask
        if watermarking_gamma is not None:
            watermarking_mask = (
                torch.rand(latents.shape, device=device) < watermarking_gamma
            )

        # Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if watermarking_mask is not None:
                    latents[watermarking_mask] += watermarking_delta * torch.sign(
                        latents[watermarking_mask]
                    )

                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample

                # Perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # Compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                # Call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Post-processing: decode latents to images (returns Tensor)
        image = self.decode_latents(latents)
        image, has_nsfw_concept = self.run_safety_checker(
            image, device, text_embeddings.dtype
        )

        # Convert to PIL if requested
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return image, None

    def decode_latents(self, latents):
        """Decode latents to image tensor (N, C, H, W) with values in [0, 1]."""
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        return (image / 2 + 0.5).clamp(0, 1)

    @torch.inference_mode()
    def decode_image(self, latents: torch.FloatTensor, **kwargs):
        scaled_latents = 1 / 0.18215 * latents
        image = [
            self.vae.decode(scaled_latents[i : i + 1]).sample
            for i in range(len(latents))
        ]
        image = torch.cat(image, dim=0)
        return image

    @torch.inference_mode()
    def torch_to_numpy(self, image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return image

    @torch.inference_mode()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents


def backward_ddim(x_t, alpha_t, alpha_tm1, eps_xt):
    """From noise to image."""
    return (
        alpha_tm1**0.5
        * (
            (alpha_t**-0.5 - alpha_tm1**-0.5) * x_t
            + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
        )
        + x_t
    )


def forward_ddim(x_t, alpha_t, alpha_tp1, eps_xt):
    """From image to noise, it's the same as backward_ddim."""
    return backward_ddim(x_t, alpha_t, alpha_tp1, eps_xt)


class InversableStableDiffusionPipeline(ModifiedStableDiffusionPipeline):
    """
    Inversable Stable Diffusion Pipeline for Tree-Ring watermarking.
    Supports both forward and backward diffusion for watermark encoding and detection.
    """
    
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        image_encoder=None,
        requires_safety_checker: bool = True,
    ):
        super(InversableStableDiffusionPipeline, self).__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            image_encoder,
            requires_safety_checker,
        )
        from functools import partial
        self.forward_diffusion = partial(self.backward_diffusion, reverse_process=True)

    def get_random_latents(
        self, latents=None, height=512, width=512, generator=None, batch_size=1
    ):
        """Generate random latent vectors."""
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        device = self._execution_device

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            self.text_encoder.dtype,
            device,
            generator,
            latents,
        )
        return latents

    @torch.inference_mode()
    def get_text_embedding(self, prompt):
        """Get text embeddings for a prompt."""
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
        return text_embeddings

    @torch.inference_mode()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        """Encode an image to latent space."""
        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents

    @torch.inference_mode()
    def backward_diffusion(
        self,
        use_old_emb_i=25,
        text_embeddings=None,
        old_text_embeddings=None,
        new_text_embeddings=None,
        latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        reverse_process: bool = False,
        **kwargs,
    ):
        """
        Perform backward (or forward) diffusion process.
        
        Args:
            reverse_process: If True, performs forward diffusion (image to noise).
        """
        do_classifier_free_guidance = guidance_scale > 1.0
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        latents = latents * self.scheduler.init_noise_sigma

        if old_text_embeddings is not None and new_text_embeddings is not None:
            prompt_to_prompt = True
        else:
            prompt_to_prompt = False

        for i, t in enumerate(
            self.progress_bar(
                timesteps_tensor if not reverse_process else reversed(timesteps_tensor)
            )
        ):
            if prompt_to_prompt:
                if i < use_old_emb_i:
                    text_embeddings = old_text_embeddings
                else:
                    text_embeddings = new_text_embeddings

            # Expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            prev_timestep = (
                t
                - self.scheduler.config.num_train_timesteps
                // self.scheduler.num_inference_steps
            )
            
            # Call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

            # DDIM step
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.final_alpha_cumprod
            )
            if reverse_process:
                alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
            latents = backward_ddim(
                x_t=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
            )
        return latents

    @torch.inference_mode()
    def decode_image(self, latents: torch.FloatTensor, **kwargs):
        scaled_latents = 1 / 0.18215 * latents
        image = [
            self.vae.decode(scaled_latents[i : i + 1]).sample
            for i in range(len(latents))
        ]
        image = torch.cat(image, dim=0)
        return image

    @torch.inference_mode()
    def torch_to_numpy(self, image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return image

    def get_decoder_transforms(self):
        """Get transforms for decoding images."""
        return transforms.Compose(
            [
                transforms.Resize(512, antialias=None),
                transforms.CenterCrop(512),
                transforms.Lambda(lambda x: 2.0 * x - 1.0),
            ]
        )

    def channel_idx(self):
        """Get channel index for watermarking."""
        return 3


def create_inversable_stable_diffusion(checkpoint, device):
    """
    Create an InversableStableDiffusionPipeline from a checkpoint.
    
    Args:
        checkpoint: Path to the model checkpoint or HuggingFace model ID
        device: Device to load the model on
    
    Returns:
        InversableStableDiffusionPipeline instance
    """
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        checkpoint,
        subfolder="scheduler",
    )
    return InversableStableDiffusionPipeline.from_pretrained(
        checkpoint,
        scheduler=scheduler,
    )


class TreeRingWatermark:
    """
    Tree-Ring Watermark system for generating and detecting watermarks.
    
    This class implements the Tree-Ring watermark which embeds watermarks
    in the frequency domain of latent representations.
    
    Exactly matches the reference implementation from TREERINGRAND.
    """
    
    def __init__(
        self,
        checkpoint="stabilityai/stable-diffusion-2-1-base",
        device="cuda",
        image_size=512,
        watermark_channel=3,
        ring_radius=10,
        batch_size=1,  # Small batch size to avoid CUDA OOM when multiple pipelines are loaded
        coco_root="./data/coco2017/val2017",
        coco_instance_ann="./data/coco2017/annotations/instances_val2017.json",
        coco_caption_ann="./data/coco2017/annotations/captions_val2017.json",
        watermark_path=None,  # Path to save/load watermark pattern for reproducibility
    ):
        """
        Initialize the Tree-Ring watermark system.
        
        Args:
            checkpoint: Path to Stable Diffusion checkpoint
            device: Device to run on
            image_size: Size of generated images
            watermark_channel: Channel index for watermark embedding (default: 3)
            ring_radius: Radius of the ring pattern (default: 10)
            batch_size: Batch size for processing
            coco_root: Path to COCO images directory
            coco_instance_ann: Path to COCO instance annotations
            coco_caption_ann: Path to COCO caption annotations
            watermark_path: Path to save/load watermark pattern (for reproducibility)
        """
        self.device = device
        self.image_size = image_size
        self.watermark_channel = watermark_channel
        self.ring_radius = ring_radius
        self.batch_size = batch_size
        self.watermark_path = watermark_path
        
        # Initialize the diffusion pipeline (encoder and decoder are the same)
        self.pipe = create_inversable_stable_diffusion(checkpoint, device)
        self.pipe = self.pipe.to(device)
        self.pipe.safety_checker = None
        self.pipe.unet.requires_grad_(False)
        self.pipe.set_progress_bar_config(leave=False)
        self.pipe.set_progress_bar_config(disable=True)
        
        # Encoder and decoder reference (same pipeline)
        self.encoder = self.pipe
        self.decoder = self.pipe
        
        # Get decoder transforms
        self.resizer = transforms.Resize((image_size, image_size), antialias=None)
        self.img_transforms = self.pipe.get_decoder_transforms()
        
        # Detection threshold (matching reference implementation)
        self.acceptance_thresh = 71
        
        # Generation parameters (matching reference implementation)
        self.encoder_kwargs = {
            "num_images_per_prompt": 1,
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "height": image_size,
            "width": image_size,
            "output_type": "numpy",  # Note: reference returns tensor, handled in post_process_raw
            "return_dict": False,
        }
        self.decoder_kwargs = {
            "guidance_scale": 1,
            "num_inference_steps": 50,
        }
        
        # Initialize watermark pattern (with save/load support for reproducibility)
        self._init_watermark(watermark_path)
        
        # Initialize COCO dataset paths
        self.coco_root = coco_root
        self.coco_instance_ann = coco_instance_ann
        self.coco_caption_ann = coco_caption_ann
        self.coco_initialized = False
        self.iters = 0  # Iterator for batch processing
    
    def _init_watermark(self, watermark_path=None):
        """
        Initialize the watermark pattern and mask.
        
        If watermark_path is provided and exists, load from file.
        Otherwise, generate new pattern and save to file (if path provided).
        This ensures reproducibility across runs (matching reference implementation).
        """
        import pickle
        import os
        
        if watermark_path is not None and os.path.exists(watermark_path):
            # Load existing watermark pattern (matching reference implementation)
            print(f"Loading watermark pattern from: {watermark_path}")
            with open(watermark_path, "rb") as f:
                watermarking_mask_np, pattern_np = pickle.load(f)
            self.watermarking_mask = torch.from_numpy(watermarking_mask_np).to(self.device)
            # Pattern is complex (FFT result), need to handle complex dtype
            pattern = torch.from_numpy(pattern_np).to(self.device)
        else:
            # Generate new watermark pattern
            print("Generating new watermark pattern...")
            latent_shape = self.encoder.get_random_latents().shape
            self.watermarking_mask = self.get_watermarking_mask(latent_shape)
            pattern = self.get_watermarking_pattern()
            
            # Save for reproducibility (matching reference implementation)
            if watermark_path is not None:
                # Create directory if it has a parent directory
                parent_dir = os.path.dirname(watermark_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                print(f"Saving watermark pattern to: {watermark_path}")
                with open(watermark_path, "wb") as f:
                    pickle.dump(
                        (
                            self.watermarking_mask.detach().cpu().numpy(),
                            pattern.detach().cpu().numpy(),
                        ),
                        f,
                    )
        
        watermark = pattern[self.watermarking_mask].view(1, -1)
        self.watermark_length = watermark.shape[-1]
        self.watermark = watermark.to(self.device)
    
    def get_watermarking_mask(self, shape):
        """Create watermarking mask for frequency domain (matching reference)."""
        watermarking_mask = torch.zeros(shape, dtype=torch.bool).to(self.device)
        np_mask = self.circle_mask(shape[-1], r=self.ring_radius)
        torch_mask = torch.tensor(np_mask).to(self.device)
        watermarking_mask[:, self.encoder.channel_idx()] = torch_mask
        return watermarking_mask

    def get_watermarking_pattern(self):
        """Generate the watermark pattern in frequency domain (matching reference)."""
        gt_init = self.encoder.get_random_latents()
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]  # Use first element for all (matching reference)
        return gt_patch
    
    @staticmethod
    def circle_mask(size=64, r=10):
        """Create a circular mask."""
        x0 = y0 = size // 2
        y, x = np.ogrid[:size, :size]
        return ((x - x0) ** 2 + (y[::-1] - y0) ** 2) <= r**2
    
    def _init_coco(self):
        """Initialize COCO dataset (lazy loading)."""
        if self.coco_initialized:
            return
        
        try:
            from pycocotools.coco import COCO
            from torchvision.datasets import CocoDetection
            
            # Create COCO API instance for captions
            self.coco_caps = COCO(self.coco_caption_ann)
            
            # Use CocoDetection to load image data
            self.coco_det = CocoDetection(
                root=self.coco_root,
                annFile=self.coco_instance_ann,
                transform=None
            )
            
            self.coco_initialized = True
            print(f"COCO dataset initialized: {len(self.coco_det)} images available")
        except Exception as e:
            print(f"Warning: Failed to initialize COCO dataset: {e}")
            print("You can still use generate_watermarked_image() with custom prompts.")
            self.coco_initialized = False
    
    def reset_iterator(self):
        """Reset the COCO dataset iterator to the beginning."""
        self.iters = 0

    def get_raw_images(self, num_images):
        """
        Get raw prompts and latents from COCO dataset (matching reference).
        
        Returns:
            tuple: (init_latents, prompts, image_size)
        """
        self._init_coco()
        
        if not self.coco_initialized:
            raise RuntimeError("COCO dataset not initialized. Check dataset paths.")
        
        image_ids = self.coco_det.ids
        prompts = []
        
        for image_id in image_ids[self.iters * num_images : (self.iters + 1) * num_images]:
            ann_ids = self.coco_caps.getAnnIds(imgIds=image_id)
            annotations = self.coco_caps.loadAnns(ann_ids)
            caption = annotations[0]['caption'] if annotations else ""
            prompts.append(caption)
        
        init_latents_w = torch.concat(
            [self.encoder.get_random_latents() for _ in range(num_images)]
        )
        
        return (init_latents_w, prompts, self.image_size)

    def post_process_raw(self, x):
        """Post-process raw images with batch processing to avoid OOM."""
        init_latents_w, prompts, image_size = x
        
        # Process in batches to avoid CUDA OOM
        all_images = []
        n_batch = int(np.ceil(len(init_latents_w) / self.batch_size))
        
        for step in range(n_batch):
            latents_i = init_latents_w[
                step * self.batch_size : (step + 1) * self.batch_size
            ].to(self.device)
            prompts_i = prompts[step * self.batch_size : (step + 1) * self.batch_size]
            
            with torch.no_grad():
                images_batch = self.encoder(prompts_i, latents=latents_i, **self.encoder_kwargs)[0]
            
            # Convert to tensor if it's numpy
            if isinstance(images_batch, np.ndarray):
                images_batch = torch.from_numpy(images_batch).permute(0, 3, 1, 2).to(self.device)
            else:
                images_batch = images_batch.to(self.device)
            
            all_images.append(images_batch)
        
        images = torch.concat(all_images)
        return self.resizer(images)

    def _encode_batch(self, x_batch, msg_batch):
        """Encode watermark into a batch of latents (matching reference)."""
        init_latents_w, prompts, image_size = x_batch
        init_latents_w_fft = torch.fft.fftshift(
            torch.fft.fft2(init_latents_w), dim=(-1, -2)
        ).to(torch.complex64)
        
        mask = (
            self.watermarking_mask
            if len(self.watermarking_mask.shape) == len(init_latents_w_fft.shape)
            else self.watermarking_mask.unsqueeze(0)
        )
        mask = mask.repeat(
            init_latents_w_fft.shape[0], *([1] * (len(init_latents_w_fft.shape) - 1))
        )
        assert mask.shape == init_latents_w_fft.shape
        
        init_latents_w_fft[mask] = (
            self.watermark.clone().repeat(init_latents_w.shape[0], 1).view(-1)
        )
        init_latents_w = torch.fft.ifft2(
            torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))
        ).real
        
        return self.post_process_raw((init_latents_w, prompts, self.image_size))

    def encode(self, x, with_grad=False):
        """Encode watermark (matching reference)."""
        init_latents_w, prompts, orig_size = x
        encoded = []
        n_batch = int(np.ceil(len(init_latents_w) / self.batch_size))

        for step in range(n_batch):
            latents_i = init_latents_w[
                step * self.batch_size : (step + 1) * self.batch_size
            ].to(self.device)
            prompts_i = prompts[step * self.batch_size : (step + 1) * self.batch_size]
            imgs = (latents_i, prompts_i, orig_size)
            msg_batch = (
                self.watermark.repeat(latents_i.shape[0], 1)
                if self.watermark is not None
                else None
            )
            if not with_grad:
                with torch.no_grad():
                    encoded_image_batch = self._encode_batch(imgs, msg_batch)
            else:
                encoded_image_batch = self._encode_batch(imgs, msg_batch)
            encoded.append(encoded_image_batch)
        
        encoded = torch.concat(encoded).view(-1, 3, self.image_size, self.image_size)
        return transforms.Resize((orig_size, orig_size), antialias=None)(encoded).to(
            init_latents_w.device
        )

    def get_watermarked_images(self, num_images):
        """
        Generate watermarked images using COCO dataset (matching reference interface).
        
        Args:
            num_images: Number of images to generate
        
        Returns:
            tuple: (original_images, watermarked_images, captions)
                - Both image tensors are shape (N, 3, H, W) with values in [0, 1]
        """
        raw_images = self.get_raw_images(num_images)
        captions = raw_images[1]
        self.iters += 1
        return self.post_process_raw(raw_images), self.encode(raw_images), captions

    def generate_watermarked_images_from_coco(self, num_images, start_idx=None):
        """
        Generate watermarked images using COCO dataset prompts.
        
        Args:
            num_images: Number of images to generate
            start_idx: Starting index in COCO dataset
        
        Returns:
            tuple: (original_images, watermarked_images, captions)
                - Images are tensors of shape (N, 3, H, W) with values in [0, 1]
        """
        if start_idx is not None:
            self.iters = start_idx
        
        return self.get_watermarked_images(num_images)

    def _decode_batch_raw(self, x):
        """Decode batch to extract watermark pattern (matching reference)."""
        embeddings = torch.concat(
            [self.decoder.get_text_embedding("") for _ in range(len(x))]
        )
        decoded = self.decoder.get_image_latents(
            self.img_transforms(x).to(embeddings.dtype).to(self.device), sample=False
        )
        decoded = self.decoder.forward_diffusion(
            latents=decoded,
            text_embeddings=embeddings,
            **self.decoder_kwargs,
        )
        decoded = torch.fft.fftshift(torch.fft.fft2(decoded), dim=(-1, -2))
        mask = (
            self.watermarking_mask
            if len(self.watermarking_mask.shape) == len(decoded.shape)
            else self.watermarking_mask.unsqueeze(0)
        )
        mask = mask.repeat(x.shape[0], *([1] * (len(decoded.shape) - 1)))
        return decoded[mask].view(x.shape[0], -1)

    def _decode_batch(self, x_batch, msg_batch):
        """Decode batch (matching reference)."""
        return self._decode_batch_raw(x_batch)

    def err(self, x_batch, msg_batch):
        """Compute error between extracted and expected watermark."""
        return torch.abs(x_batch - msg_batch).mean(-1)

    def stats(self, imgs, decoded, msg_batch):
        """Compute detection statistics (matching reference)."""
        return self.err(decoded, msg_batch)

    def is_detected(self, accs):
        """Determine if watermark is detected (matching reference)."""
        return accs < self.acceptance_thresh

    def detect_watermark(self, image):
        """
        Detect watermark in an image.
        
        Args:
            image: Input image (PIL Image, numpy array, or torch tensor)
        
        Returns:
            tuple: (detection_score, is_watermarked)
        """
        # Convert image to tensor if needed
        if isinstance(image, PIL.Image.Image):
            image = transforms.ToTensor()(image)
        if isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = torch.from_numpy(image).permute(2, 0, 1).float()
            else:
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Ensure proper shape and device
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = self.resizer(image.to(self.device))
        
        # Decode to extract watermark
        msg_batch = self.watermark.repeat(image.shape[0], 1)
        with torch.no_grad():
            decoded = self._decode_batch(image, msg_batch)
        
        # Compute score
        score = self.stats(image, decoded, msg_batch).detach().cpu().numpy()
        
        # Determine if watermark is detected
        is_watermarked = self.is_detected(score)
        
        return score, is_watermarked

    def __call__(self, x):
        """
        Detect watermark in batch of images (matching reference interface).
        
        Args:
            x: Batch of images tensor (N, 3, H, W)
        
        Returns:
            numpy array of detection scores
        """
        accs = np.zeros((0,), dtype=np.float32)
        n_batch = int(np.ceil(len(x) / self.batch_size))
        
        for step in range(n_batch):
            imgs = self.resizer(
                x[step * self.batch_size : (step + 1) * self.batch_size].to(self.device)
            )
            msg_batch = (
                self.watermark.repeat(imgs.shape[0], 1)
                if self.watermark is not None
                else None
            )
            with torch.no_grad():
                decoded = self._decode_batch(imgs, msg_batch)
            accs = np.concatenate(
                (
                    accs,
                    self.stats(imgs, decoded, msg_batch)
                    .detach()
                    .cpu()
                    .numpy()
                    .round(2),
                )
            )
        return accs

    def compute_detection_rate(self, images, threshold=None):
        """
        Compute watermark detection rate for a batch of images.
        
        Args:
            images: Tensor of images (N, 3, H, W) or list of images
            threshold: Optional custom threshold. If None, uses self.acceptance_thresh
        
        Returns:
            tuple: (detection_rate, scores)
        """
        if isinstance(images, list):
            # Convert list to tensor
            tensor_list = []
            for img in images:
                if isinstance(img, PIL.Image.Image):
                    img = transforms.ToTensor()(img)
                elif isinstance(img, np.ndarray):
                    if img.max() <= 1.0:
                        img = torch.from_numpy(img).permute(2, 0, 1).float()
                    else:
                        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                tensor_list.append(img)
            images = torch.stack(tensor_list)
        
        scores = self(images)
        if threshold is None:
            detection_rate = np.mean(self.is_detected(scores))
        else:
            detection_rate = np.mean(scores < threshold)
        return detection_rate, scores

    def compute_threshold_at_fpr(self, unwatermarked_scores, target_fpr=0.01):
        """
        Compute detection threshold at a specific False Positive Rate (FPR).
        
        For Tree-Ring watermark, a LOWER score indicates the presence of watermark.
        FPR = P(detected | no watermark) = P(score < threshold | no watermark)
        
        This method matches the internal project's implementation:
        min_orig = sorted(orig_values)[cnt//100-1]
        
        Args:
            unwatermarked_scores: Detection scores from unwatermarked (original) images
            target_fpr: Target false positive rate (default: 0.01 = 1%)
        
        Returns:
            float: Threshold value where FPR equals target_fpr
        """
        # Match internal project: sorted(orig_values)[cnt//100-1]
        # For 100 images with 1% FPR: index = 100 // 100 - 1 = 0 (minimum value)
        n = len(unwatermarked_scores)
        k = max(0, int(n * target_fpr) - 1)  # Index for threshold
        sorted_scores = np.sort(unwatermarked_scores)
        threshold = sorted_scores[k]
        return threshold
    
    def compute_tpr_at_fpr(self, watermarked_scores, unwatermarked_scores, target_fpr=0.01):
        """
        Compute True Positive Rate (TPR) at a specific False Positive Rate (FPR).
        
        This is the standard metric for watermark detection evaluation:
        - FPR = P(detected | no watermark)
        - TPR = P(detected | watermark present)
        
        Args:
            watermarked_scores: Detection scores from watermarked images
            unwatermarked_scores: Detection scores from unwatermarked images
            target_fpr: Target false positive rate (default: 0.01 = 1%)
        
        Returns:
            tuple: (tpr, threshold, fpr_actual)
                - tpr: True Positive Rate at the given FPR
                - threshold: The threshold used
                - fpr_actual: Actual FPR achieved (should be close to target_fpr)
        """
        # Compute threshold at target FPR
        threshold = self.compute_threshold_at_fpr(unwatermarked_scores, target_fpr)
        
        # Compute TPR: percentage of watermarked images correctly detected
        tpr = np.mean(watermarked_scores < threshold)
        
        # Verify actual FPR
        fpr_actual = np.mean(unwatermarked_scores < threshold)
        
        return tpr, threshold, fpr_actual
    
    def evaluate_with_dynamic_threshold(
        self,
        orig_scores,
        watermarked_scores,
        attacked_scores,
        target_fpr=0.01
    ):
        """
        Evaluate detection rates using dynamic threshold based on FPR.
        
        This method computes the threshold at a specific FPR using unwatermarked (original)
        images, then evaluates TPR for watermarked and attacked images.
        
        Args:
            orig_scores: Detection scores from original (unwatermarked) images
            watermarked_scores: Detection scores from watermarked images
            attacked_scores: Detection scores from attacked images
            target_fpr: Target false positive rate (default: 0.01 = 1%)
        
        Returns:
            dict: Contains threshold, FPR, and TPR values
        """
        # Compute threshold at target FPR using original (unwatermarked) images
        threshold = self.compute_threshold_at_fpr(orig_scores, target_fpr)
        
        # Compute detection rates using the dynamic threshold
        fpr_orig = np.mean(orig_scores < threshold)  # Should be ~target_fpr
        tpr_watermarked = np.mean(watermarked_scores < threshold)
        tpr_attacked = np.mean(attacked_scores < threshold)
        
        return {
            "threshold": threshold,
            "target_fpr": target_fpr,
            "fpr_actual": fpr_orig,
            "tpr_watermarked": tpr_watermarked,
            "tpr_attacked": tpr_attacked,
            "avg_score_orig": np.mean(orig_scores),
            "avg_score_watermarked": np.mean(watermarked_scores),
            "avg_score_attacked": np.mean(attacked_scores),
        }

    def load_img(self, path, image_size=256):
        """Load image from path as tensor."""
        return (
            transforms.Resize((image_size, image_size), antialias=None)(
                transforms.ToTensor()(PIL.Image.open(path).convert("RGB"))
            )
            .view(1, 3, image_size, image_size)
            .to(self.device)
        )

    def save(self, image, path):
        """Save tensor image to path."""
        transforms.ToPILImage()(
            self.resizer(image.view(1, 3, image.shape[-1], image.shape[-1]))
            .detach()
            .cpu()
            .squeeze()
        ).save(path)
