import torch
import numpy as np
import cv2

class GradCAM:
    """
    GradCAM class to generate Grad-CAM visualizations for a given model and target layer.

    Attributes:
        model (torch.nn.Module): The neural network model.
        target_layer (str): The name of the target layer to visualize.
        gradients (torch.Tensor): Gradients computed during backpropagation.
        activations (torch.Tensor): Activations from the forward pass.

    Methods:
        _register_hooks(): Registers hooks to capture gradients and activations.
        generate_cam(input_image, class_idx=None): Generates the Grad-CAM heatmap for a given input image.
        overlay_cam(img, cam, alpha=0.4): Overlays the Grad-CAM heatmap on the original image.
        __call__(input_image, original_image, class_idx=None): Generates and overlays the Grad-CAM heatmap on the original image.
    """
    
    def __init__(self, model, target_layer):
        """
        Initializes the GradCAM object and registers hooks.
        
        Parameters:
            model (torch.nn.Module): The neural network model.
            target_layer (str): The name of the target layer to visualize.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()  # Register hooks to capture gradients and activations

    def _register_hooks(self):
        """
        Registers hooks to capture gradients and activations from the target layer.
        """
        def backward_hook(module, grad_in, grad_out):
            # Capture gradients from the backward pass
            self.gradients = grad_out[0]

        def forward_hook(module, input, output):
            # Capture activations from the forward pass
            self.activations = output

        # Get the target layer from the model
        target_layer = dict([*self.model.named_modules()])[self.target_layer]
        # Register backward hook to capture gradients
        target_layer.register_backward_hook(backward_hook)
        # Register forward hook to capture activations
        target_layer.register_forward_hook(forward_hook)

    def generate_cam(self, input_image, class_idx=None):
        """
        Generates the Grad-CAM heatmap for a given input image.
        
        Parameters:
            input_image (torch.Tensor): The input image tensor of shape (1, C, H, W).
            class_idx (int, optional): The class index for which to generate the heatmap. If None, uses the predicted class.

        Returns:
            cam (np.ndarray): The generated Grad-CAM heatmap of shape (H, W).
        """
        self.model.eval()  # Set the model to evaluation mode
        output = self.model(input_image)  # Perform a forward pass

        if class_idx is None:
            # If class index is not provided, use the class with the highest score
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()  # Zero the gradients
        class_score = output[:, class_idx]  # Get the score for the target class
        class_score.backward()  # Perform backpropagation to get gradients

        # Get the gradients and activations
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        # Compute the weights for each channel
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        # Compute the weighted sum of the activations
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # Apply ReLU to the CAM to keep only positive values
        cam = np.maximum(cam, 0)
        # Resize the CAM to the input image size
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))
        # Normalize the CAM to range [0, 1]
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        return cam

    def overlay_cam(self, img, cam, alpha=0.4):
        """
        Overlays the Grad-CAM heatmap on the original image.
        
        Parameters:
            img (np.ndarray): The original image of shape (H, W, C).
            cam (np.ndarray): The Grad-CAM heatmap of shape (H, W).
            alpha (float): The transparency factor for the heatmap overlay.

        Returns:
            overlayed_img (np.ndarray): The image with the Grad-CAM heatmap overlay of shape (H, W, C).
        """
        # Apply color map to the CAM
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        # Overlay the heatmap on the original image
        overlayed_img = heatmap * alpha + np.float32(img) / 255
        # Normalize the overlayed image to range [0, 1]
        overlayed_img = overlayed_img / np.max(overlayed_img)
        return np.uint8(255 * overlayed_img)

    def __call__(self, input_image, original_image, class_idx=None):
        """
        Generates and overlays the Grad-CAM heatmap on the original image.
        
        Parameters:
            input_image (torch.Tensor): The input image tensor of shape (1, C, H, W).
            original_image (np.ndarray): The original image of shape (H, W, C).
            class_idx (int, optional): The class index for which to generate the heatmap. If None, uses the predicted class.

        Returns:
            overlayed_img (np.ndarray): The image with the Grad-CAM heatmap overlay of shape (H, W, C).
        """
        # Generate the Grad-CAM heatmap
        cam = self.generate_cam(input_image, class_idx)
        # Overlay the heatmap on the original image
        overlayed_img = self.overlay_cam(original_image, cam)
        return overlayed_img
