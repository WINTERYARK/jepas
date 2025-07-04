from typing import Optional

from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms

from configs import get_image_dataset_config
from .image_transforms import make_transforms

dataset_config = get_image_dataset_config()


class HFImageDataset(TorchDataset):
    """Dataset wrapper for HuggingFace image datasets."""

    def __init__(
        self,
        dataset_id: str,
        subset_name: Optional[str],
        split: str,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()
        self.dataset = load_dataset(dataset_id, name=subset_name, split=split)
        self.transform = transform or transforms.Compose(make_transforms())

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        image = sample.get("image")
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            # Some datasets return dict with 'bytes'
            image = Image.open(image).convert("RGB")
        return self.transform(image)
