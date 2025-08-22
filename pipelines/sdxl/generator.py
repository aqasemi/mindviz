import torch
from typing import Optional, List
from diffusers import DiffusionPipeline


class SDXLGenerator:
    def __init__(
        self,
        model_id: str = "stabilityai/sdxl-turbo",
        ip_adapter_repo: str = "h94/IP-Adapter",
        ip_adapter_weight: str = "ip-adapter_sdxl_vit-h.bin",
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        ip_adapter_scale: float = 1.0,
    ) -> None:
        self.device = device
        self.dtype = torch_dtype

        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            variant="fp16",
        )
        pipe.to(device)

        # Load IP-Adapter
        pipe.load_ip_adapter(
            ip_adapter_repo,
            subfolder="sdxl_models",
            weight_name=ip_adapter_weight,
            torch_dtype=self.dtype,
        )
        pipe.set_ip_adapter_scale(ip_adapter_scale)

        self.pipe = pipe

    def _call_with_added_kwargs(
        self,
        prompt: str,
        image_embeds: torch.Tensor,
        num_inference_steps: int,
        guidance_scale: float,
        generator: Optional[torch.Generator],
        height: Optional[int],
        width: Optional[int],
        negative_prompt: Optional[str],
    ):
        pipe = self.pipe
        device = pipe._execution_device

        # 1) Encode prompt to get token-level and pooled embeddings
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=guidance_scale > 1.0,
            negative_prompt=negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            clip_skip=getattr(pipe, "clip_skip", None),
        )

        # 2) Prepare time ids
        if height is None:
            height = pipe.default_sample_size * pipe.vae_scale_factor
        if width is None:
            width = pipe.default_sample_size * pipe.vae_scale_factor

        text_encoder_projection_dim = (
            int(pooled_prompt_embeds.shape[-1]) if pipe.text_encoder_2 is None else pipe.text_encoder_2.config.projection_dim
        )
        add_time_ids = pipe._get_add_time_ids(
            original_size=(height, width),
            crops_coords_top_left=(0, 0),
            target_size=(height, width),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        # 3) Build added conditions dict with image_embeds
        added = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids, "image_embeds": image_embeds}

        # 4) Call pipeline using explicit embeddings and added conditions
        result = pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=height,
            width=width,
            added_cond_kwargs=added,
        )
        return result.images

    @torch.no_grad()
    def generate(
        self,
        image_embeds: torch.Tensor,
        prompt: Optional[str] = "",
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        generator: Optional[torch.Generator] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        negative_prompt: Optional[str] = None,
    ):
        image_embeds = image_embeds.to(device=self.device, dtype=self.dtype)
        # Try modern diffusers API first; fall back to alternative arg names
        call_kwargs = dict(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
        )

        # Attempt with ip_adapter_embeds
        try:
            result = self.pipe(ip_adapter_image=None, ip_adapter_embeds=image_embeds, **call_kwargs)
            return result.images
        except TypeError:
            pass
        except ValueError as e:
            last_err = e

        # Attempt with image_embeds (older/newer variants)
        try:
            result = self.pipe(image_embeds=image_embeds, **call_kwargs)
            return result.images
        except TypeError:
            pass
        except ValueError as e:
            last_err = e

        # Attempt by explicitly passing added cond kwargs (API variants)
        try:
            result = self.pipe(added_cond_kwargs={"image_embeds": image_embeds}, **call_kwargs)
            return result.images
        except TypeError:
            pass
        except ValueError as e:
            last_err = e

        # Another naming used in some versions
        try:
            result = self.pipe(added_conditions={"image_embeds": image_embeds}, **call_kwargs)
            return result.images
        except TypeError:
            pass
        except ValueError as e:
            last_err = e

        # Try explicit embeddings path that constructs added_cond kwargs
        try:
            return self._call_with_added_kwargs(
                prompt=prompt,
                image_embeds=image_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
            )
        except Exception as e:
            last_err = e

        # If nothing works, raise a clear error
        raise RuntimeError(
            (
                "Failed to pass IP-Adapter embeddings to SDXL pipeline. "
                "Tried ip_adapter_embeds, image_embeds, added_cond_kwargs, added_conditions. "
                "Also attempted explicit added_cond construction. Please upgrade diffusers or adapt argument names."
            ) + (f" Last error: {last_err}" if 'last_err' in locals() else "")
        )
