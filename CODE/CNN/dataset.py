import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CancerDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name)
        if "no" in self.image_files[idx]:
            label = 0
        elif "me" in self.image_files[idx]:
            label = 1
        else:
            label = 2
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_name

# Define transforms for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to match the input size expected by the CNN
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create DataLoader
def get_data_loader(image_dir, batch_size=32, shuffle=True, num_workers=4):
    dataset = CancerDataset(image_dir=image_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader
