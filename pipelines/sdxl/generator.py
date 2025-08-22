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

        # A baseline pipeline without IP-Adapter for comparison
        base_pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            variant="fp16",
        )
        base_pipe.to(device)
        self.base_pipe = base_pipe

    def _format_ip_adapter_embeds(self, image_embeds: torch.Tensor, guidance_scale: float) -> list:
        """Format precomputed IP-Adapter embeddings for diffusers SDXL pipeline.

        Expects input shape [batch, dim]. Returns a list with one tensor shaped
        [batch, 1, dim] when guidance_scale <= 1, otherwise [2*batch, 1, dim]
        where the first half are zeros for unconditional CFG.
        """
        if image_embeds.ndim == 2:
            image_embeds = image_embeds.unsqueeze(1)  # [B, 1, D]

        if guidance_scale is not None and guidance_scale > 1.0:
            zeros = torch.zeros_like(image_embeds)
            image_embeds = torch.cat([zeros, image_embeds], dim=0)

        # diffusers expects a list with length equal to the number of IP-Adapters
        return [image_embeds]

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

        # Prepare ip_adapter_image_embeds as expected by diffusers
        ip_adapter_image_embeds = self._format_ip_adapter_embeds(image_embeds, guidance_scale)

        # Call the pipeline; it will build added_cond_kwargs with text/time and our image embeds
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=height,
            width=width,
            ip_adapter_image_embeds=ip_adapter_image_embeds,
        )
        return result.images

    @torch.no_grad()
    def generate_no_adapter(
        self,
        prompt: Optional[str] = "",
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        generator: Optional[torch.Generator] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        negative_prompt: Optional[str] = None,
    ):
        result = self.base_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=height,
            width=width,
        )
        return result.images


 
