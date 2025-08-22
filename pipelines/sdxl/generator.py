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

        # If neither works, raise a clear error
        raise RuntimeError(
            (
                "Failed to pass IP-Adapter embeddings to SDXL pipeline. "
                "Tried ip_adapter_embeds, image_embeds, added_cond_kwargs, added_conditions. "
                "Please upgrade diffusers or adapt argument names."
            ) + (f" Last error: {last_err}" if 'last_err' in locals() else "")
        )
