import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_CANCER_DETECTOR(nn.Module):
    """
    Convolutional Neural Network for Cancer Detection.

    This class defines a CNN model with three convolutional layers followed by three
    fully connected layers for classification.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        pool (nn.MaxPool2d): Max pooling layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer (output layer).
    """

    def __init__(self, channel_size):
        """
        Initializes the CNN model.

        Parameters:
            channel_size (int): Number of input channels (e.g., 3 for RGB images).
        """
        super(CNN_CANCER_DETECTOR, self).__init__()
        
        # First convolutional layer: input channels = channel_size, output channels = 32
        self.conv1 = nn.Conv2d(in_channels=channel_size, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Second convolutional layer: input channels = 32, output channels = 64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Third convolutional layer: input channels = 64, output channels = 128
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # First fully connected layer: input features = 128 * 16 * 16, output features = 512
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        
        # Second fully connected layer: input features = 512, output features = 128
        self.fc2 = nn.Linear(512, 128)
        
        # Third fully connected layer: input features = 128, output features = 3 (number of classes)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        
        # Apply first convolutional layer, ReLU activation, and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Apply second convolutional layer, ReLU activation, and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Apply third convolutional layer, ReLU activation, and max pooling
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the tensor to (batch_size, 128 * 16 * 16) for fully connected layers
        x = x.view(-1, 128 * 16 * 16)
        
        # Apply first fully connected layer and ReLU activation
        x = F.relu(self.fc1(x))
        
        # Apply second fully connected layer and ReLU activation
        x = F.relu(self.fc2(x))
        
        # Apply third fully connected layer (output layer)
        x = self.fc3(x)
        
        return x