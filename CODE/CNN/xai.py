import os
import torch
import cv2
import matplotlib.pyplot as plt
from cnn import CNN_CANCER_DETECTOR
from dataset import get_data_loader
from grad_cam import GradCAM
import sys

# Example usage
def preprocess_image(image):
    img = image.numpy().transpose(1, 2, 0)
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    img_tensor = img_tensor / 255.0
    return img_tensor, img

def save_cam_image(overlayed_img, save_path):
    cv2.imwrite(save_path, cv2.cvtColor(overlayed_img, cv2.COLOR_RGB2BGR))

def generate_and_save_gradcam(model, dataloader, target_layer, output_dir):
    grad_cam = GradCAM(model, target_layer)
    
    for i, (images, labels, paths) in enumerate(dataloader):
        for img, path in zip(images, paths):
            input_image, original_image = preprocess_image(img)
            input_image = input_image.to(next(model.parameters()).device)
            
            overlayed_img = grad_cam(input_image, original_image)
            
            img_name = os.path.basename(path)
            save_path = os.path.join(output_dir, img_name)
            save_cam_image(overlayed_img, save_path)
            print(f"Saved Grad-CAM image for {img_name} at {save_path}")

def main(model_path, test_image_dir, target_layer, output_dir, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = CNN_CANCER_DETECTOR(channel_size=3).to(device)
    model.load_state_dict(torch.load(model_path))

    
    # Create test dataloader
    test_loader = get_data_loader(image_dir=test_image_dir, batch_size=batch_size, shuffle=False)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save Grad-CAM visualizations
    generate_and_save_gradcam(model, test_loader, target_layer, output_dir)

if __name__ == "__main__":
    os.chdir(r"C:\Users\maron\OneDrive\Bureau\PROJECT") # Adapt this to the location of the PROJECT directory
    sys.path.append("CODE\CNN")
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'

    model_path = r"LOGS_CNN\checkpoints\best_model.pth"
    test_image_dir = r"DATA\processed\test"
    target_layer = "conv3"
    output_dir = r"LOGS_CNN\grad_cam_c3_test_results"
    
    main(model_path, test_image_dir, target_layer, output_dir)
