from PIL import Image
import torch
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    """Custom Dataset"""
    def __init__(self, images_path: list, images_class: list, transform=None):
        """
        Initialize the dataset
        :param images_path: List of image paths
        :param images_class: List of corresponding class labels
        :param transform: Data preprocessing
        """
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        # Automatically extract class names and sort them to ensure consistent order
        self.classes = sorted(list(set(images_class)))

    def __len__(self):
        """Return the size of the dataset"""
        return len(self.images_path)

    def __getitem__(self, item):
        """Get image and label by index"""
        img = Image.open(self.images_path[item])
        # Ensure the image is in RGB mode
        if img.mode != 'RGB':
            raise ValueError(f"Image: {self.images_path[item]} isn't RGB mode.")
        label = self.images_class[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels