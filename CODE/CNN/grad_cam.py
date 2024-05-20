import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        def forward_hook(module, input, output):
            self.activations = output

        target_layer = dict([*self.model.named_modules()])[self.target_layer]
        target_layer.register_backward_hook(backward_hook)
        target_layer.register_forward_hook(forward_hook)

    def generate_cam(self, input_image, class_idx=None):
        self.model.eval()
        output = self.model(input_image)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        class_score = output[:, class_idx]
        class_score.backward()

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        return cam

    def overlay_cam(self, img, cam, alpha=0.4):
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        overlayed_img = heatmap * alpha + np.float32(img) / 255
        overlayed_img = overlayed_img / np.max(overlayed_img)
        return np.uint8(255 * overlayed_img)

    def __call__(self, input_image, original_image, class_idx=None):
        cam = self.generate_cam(input_image, class_idx)
        overlayed_img = self.overlay_cam(original_image, cam)
        return overlayed_img
