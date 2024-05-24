import os
import torch
import cv2
import matplotlib.pyplot as plt
from cnn import CNN_CANCER_DETECTOR
from dataset import get_data_loader
from grad_cam import GradCAM
import sys

def preprocess_image(image):
    """
    Preprocesses a tensor image for Grad-CAM visualization.
    
    Parameters:
        image (torch.Tensor): Input image tensor.
    
    Returns:
        torch.Tensor: Preprocessed image tensor.
        numpy.ndarray: Original image as a numpy array.
    """
    # Convert tensor to numpy array and transpose to (H, W, C) format
    img = image.numpy().transpose(1, 2, 0)
    # Convert back to tensor, add batch dimension, and normalize
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    img_tensor = img_tensor / 255.0
    return img_tensor, img

def save_cam_image(overlayed_img, save_path):
    """
    Saves the Grad-CAM overlayed image to disk.
    
    Parameters:
        overlayed_img (numpy.ndarray): Grad-CAM overlayed image.
        save_path (str): Path to save the image.
    """
    # Save the image in RGB format
    cv2.imwrite(save_path, cv2.cvtColor(overlayed_img, cv2.COLOR_RGB2BGR))

def generate_and_save_gradcam(model, dataloader, target_layer, output_dir):
    """
    Generates and saves Grad-CAM visualizations for images in the dataloader.
    Parameters:
        model (torch.nn.Module): The CNN model.
        dataloader (torch.utils.data.DataLoader): Dataloader for test images.
        target_layer (str): Target layer for Grad-CAM.
        output_dir (str): Directory to save the Grad-CAM images.
    """
    grad_cam = GradCAM(model, target_layer)
    
    for i, (images, labels, paths) in enumerate(dataloader):
        for img, path in zip(images, paths):
            # Preprocess image
            input_image, original_image = preprocess_image(img)
            input_image = input_image.to(next(model.parameters()).device)
            
            # Generate Grad-CAM overlay
            overlayed_img = grad_cam(input_image, original_image)
            
            # Save Grad-CAM image
            img_name = os.path.basename(path)
            save_path = os.path.join(output_dir, img_name)
            save_cam_image(overlayed_img, save_path)
            print(f"Saved Grad-CAM image for {img_name} at {save_path}")

def main(model_path, test_image_dir, target_layer, output_dir, batch_size=32):
    """
    Main function to load model, create dataloader, and generate Grad-CAM visualizations.
    
    Parameters:
        model_path (str): Path to the model checkpoint.
        test_image_dir (str): Directory containing test images.
        target_layer (str): Target layer for Grad-CAM.
        output_dir (str): Directory to save the Grad-CAM images.
        batch_size (int, optional): Batch size for dataloader. Defaults to 32.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = CNN_CANCER_DETECTOR(channel_size=3).to(device)
    model.load_state_dict(torch.load(model_path))

    # Create test dataloader
    test_loader = get_data_loader(image_dir=test_image_dir, batch_size=batch_size, shuffle=False)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save Grad-CAM visualizations
    generate_and_save_gradcam(model, test_loader, target_layer, output_dir)

if __name__ == "__main__":
    # Change working directory and set environment variable
    os.chdir(r"C:\Users\maron\OneDrive\Bureau\PROJECT") # Adapt this to the location of the PROJECT directory
    sys.path.append("CODE\CNN")
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' 

    # Define paths and parameters
    model_path = r"LOGS_CNN\checkpoints\best_model.pth"
    test_image_dir = r"DATA\processed\test"
    target_layer = "conv3"
    output_dir = r"LOGS_CNN\grad_cam_c3_test_results"
    
    # Execute main function
    main(model_path, test_image_dir, target_layer, output_dir)
