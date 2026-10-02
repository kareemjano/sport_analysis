import torch
from torchvision.transforms import v2


def image_transform(image_size):
    """SAM3 input preprocessing: uint8 RGB image tensor -> normalized square float image."""
    return v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(image_size, image_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
