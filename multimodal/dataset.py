from typing import Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset


class VQADataset(Dataset):
    """Dataset wrapping the VQAv2 data from Hugging Face.

    Each item is a tuple of transformed image tensor and the raw text string.
    """

    def __init__(
        self,
        split: str,
        image_transform: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()

        self.dataset = load_dataset("lmms-lab/VQAv2", split=split)
        self.image_transform = image_transform or transforms.Compose(
            [transforms.ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[transforms.Tensor, str]:
        sample = self.dataset[idx]
        image = sample.get("image")
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        if self.image_transform:
            image = self.image_transform(image)
        question = sample.get("question") or sample.get("text")
        return image, question
