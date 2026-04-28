from typing import Callable, Union

import numpy as np
import torch
import torchvision.transforms.functional as TVF
from PIL import Image

try:
    from transformers import Qwen2VLImageProcessor
except ImportError:  # pragma: no cover
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from transformers.feature_extraction_utils import BatchFeature


# ---------------------------------------------------------------------------
# DINOv3 preprocessor
# ---------------------------------------------------------------------------

def create_dinov3_processor() -> Callable[[Union[Image.Image, np.ndarray]], torch.Tensor]:
    """Create a DINOv3 image preprocessor.

    Resizes the input to 512x512 and applies ImageNet normalization.
    Compatible with the logic in ``qwen2vl_multi_encoder_task_encoder.py``.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    size = [512, 512]
    rescale_factor = 1.0 / 255.0

    def preprocess(image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype("uint8"))
        arr = np.array(image)
        # [C, H, W], float, range [0, 255]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
        # Rescale
        tensor = tensor * rescale_factor
        # Resize (bilinear, antialias=True)
        tensor = TVF.resize(tensor, size, interpolation=TVF.InterpolationMode.BILINEAR, antialias=True)
        # Normalize
        tensor = TVF.normalize(tensor, mean=mean, std=std)
        # Return [1, C, H, W]
        return tensor.unsqueeze(0)

    return preprocess


# ---------------------------------------------------------------------------
# SigLIP preprocessor
# ---------------------------------------------------------------------------

def create_siglip_processor() -> Callable[[Union[Image.Image, np.ndarray]], torch.Tensor]:
    """Create a SigLIP image preprocessor.

    Resizes the input to 384x384 and applies SigLIP normalization
    (mean=0.5, std=0.5, fused with 255 rescale).
    Compatible with the logic in ``qwen2vl_multi_encoder_task_encoder.py``.
    """
    fused_mean = [127.5, 127.5, 127.5]
    fused_std = [127.5, 127.5, 127.5]
    size = [384, 384]

    def preprocess(image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype("uint8"))
        # PIL -> uint8 tensor [C, H, W], range [0, 255]
        tensor = TVF.pil_to_tensor(image)
        # Resize (bilinear, antialias=True) on uint8 tensor
        tensor = TVF.resize(tensor, size, interpolation=TVF.InterpolationMode.BILINEAR, antialias=True)
        # Normalize (fused: equivalent to /255 then (x-mean)/std)
        tensor = TVF.normalize(tensor.float(), mean=fused_mean, std=fused_std)
        # Return [1, C, H, W]
        return tensor.unsqueeze(0)

    return preprocess


# ---------------------------------------------------------------------------
# Innovator-VL image processor
# ---------------------------------------------------------------------------

class InnovatorVLImageProcessor(Qwen2VLImageProcessor):
    """Custom image processor for Innovator-VL.

    Extends ``Qwen2VLImageProcessor`` with additional SigLIP and DINOv3
    pixel streams required by ``InnovatorVl_ForConditionalGeneration``.
    """

    model_input_names = [
        "pixel_values",
        "image_grid_thw",
        "pixel_values_images_siglip",
        "pixel_values_images_dinov3",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.siglip_processor = create_siglip_processor()
        self.dinov3_processor = create_dinov3_processor()

    def preprocess(self, images, **kwargs) -> BatchFeature:
        """Preprocess images for Innovator-VL.

        In addition to the standard Qwen2-VL pixel values and grid info,
        this processor also generates ``pixel_values_images_siglip`` and
        ``pixel_values_images_dinov3`` for the hybrid vision encoder.
        """
        # Standard Qwen2-VL preprocessing (pixel_values, image_grid_thw)
        outputs = super().preprocess(images, **kwargs)

        # Auxiliary SigLIP / DINOv3 pixel streams
        if images is not None:
            pil_images = self._to_pil_list(images)
            if pil_images:
                siglip_pixels = [self.siglip_processor(img) for img in pil_images]
                dinov3_pixels = [self.dinov3_processor(img) for img in pil_images]
                outputs["pixel_values_images_siglip"] = torch.cat(siglip_pixels, dim=0)
                outputs["pixel_values_images_dinov3"] = torch.cat(dinov3_pixels, dim=0)

        return outputs

    def _to_pil_list(self, images):
        """Convert various image input formats to a list of RGB PIL Images."""
        if not isinstance(images, (list, tuple)):
            # batch tensor / array -> list of single images
            if isinstance(images, (np.ndarray, torch.Tensor)) and images.ndim == 4:
                images = [images[i] for i in range(images.shape[0])]
            else:
                images = [images]

        pil_images = []
        for img in images:
            if isinstance(img, Image.Image):
                pil_img = img.convert("RGB")
            elif isinstance(img, np.ndarray):
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = img.transpose(1, 2, 0)
                if img.dtype in (np.float32, np.float64):
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img).convert("RGB")
            elif isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = img.transpose(1, 2, 0)
                if img.dtype in (np.float32, np.float64):
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img).convert("RGB")
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
            pil_images.append(pil_img)

        return pil_images


__all__ = [
    "create_dinov3_processor",
    "create_siglip_processor",
    "InnovatorVLImageProcessor",
]
