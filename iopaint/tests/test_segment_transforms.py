import torch

from iopaint.plugins.segment_anything.utils.transforms import ResizeLongestSide
from iopaint.plugins.segment_anything2.utils.transforms import SAM2Transforms


def test_sam2_postprocess_masks_restores_original_size():
    # low-res mask 固定 256×256，不插值回原图会让前端把 4K 图当成 256px 掩膜拉伸
    transforms = SAM2Transforms(resolution=1024, mask_threshold=0.0)
    masks = torch.rand(1, 3, 256, 256)
    out = transforms.postprocess_masks(masks, (240, 320))
    assert out.shape == (1, 3, 240, 320)


def test_sam_apply_image_torch_resizes_height_width_not_batch_channels():
    transforms = ResizeLongestSide(1024)
    out = transforms.apply_image_torch(torch.zeros(1, 3, 100, 200))
    assert out.shape == (1, 3, 512, 1024)
