from torch.utils.data import Dataset
from PIL import Image

class CustomTestDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Open image
        if self.transform:
            image = self.transform(image)  # Apply the transformations
        return image, -1  # -1 is a placeholder since there are no labels for testing
