import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CancerDataset(Dataset):
    """
    Custom Dataset class for loading cancer images.

    Attributes:
        image_dir (str): Directory containing image files.
        transform (torchvision.transforms.Compose): Transformations to apply to the images.
        image_files (list): List of image file names in the directory.
    """
    def __init__(self, image_dir, transform=None):
        """
        Initializes the dataset with image directory and transformations.

        Parameters:
            image_dir (str): Directory containing the images.
            transform (torchvision.transforms.Compose): Transformations to apply to the images.
        """
        self.image_dir = image_dir
        self.transform = transform
        # List all files in the directory
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        """
        Returns the total number of images.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Retrieves an image and its label by index.

        Parameters:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: (image, label, img_name)
                - image (PIL.Image or torch.Tensor): Transformed image.
                - label (int): Label of the image (0: no tumor, 1: meningioma, 2: pituitary).
                - img_name (str): File path of the image.
        """
        img_name = os.path.join(self.image_dir, self.image_files[idx])  # Get image file path
        image = Image.open(img_name)  # Open image file
        
        # Determine the label based on the file name
        if "no" in self.image_files[idx]:  # notumor
            label = 0
        elif "me" in self.image_files[idx]: # meningioma
            label = 1
        else:                               # pituitary
            label = 2
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_name  # Return the image, label, and file path

# Define transforms for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to match the input size expected by the CNN
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
])

# Create DataLoader
def get_data_loader(image_dir, batch_size=32, shuffle=True, num_workers=4):
    """
    Creates a DataLoader for the cancer dataset.

    Parameters:
        image_dir (str): Directory containing the images.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    dataset = CancerDataset(image_dir=image_dir, transform=transform)  # Initialize the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)  # Create DataLoader
    return data_loader