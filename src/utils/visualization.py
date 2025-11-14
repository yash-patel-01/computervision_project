"""
Utility functions for visualizing bounding boxes and predictions.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image


# Color map for different classes
COLORS = {
    0: (128, 128, 128),  # Background (gray) - not used
    1: (0, 255, 0),      # Player (green)
    2: (255, 0, 0)       # Ball (red in BGR -> blue in RGB)
}

CLASS_NAMES = {
    0: 'background',
    1: 'player',
    2: 'ball'
}


def denormalize_image(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize a normalized image tensor for visualization.
    
    Args:
        image_tensor (torch.Tensor): Normalized image tensor [C, H, W]
        mean (list): Mean used for normalization
        std (list): Std used for normalization
    
    Returns:
        numpy.ndarray: Denormalized image [H, W, C] in range [0, 255]
    """
    image = image_tensor.clone()
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    image = image.numpy().transpose(1, 2, 0)
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


def draw_bounding_boxes(image, boxes, labels, scores=None, thickness=2):
    """
    Draw bounding boxes on an image.
    
    Args:
        image (numpy.ndarray): Image array [H, W, C]
        boxes (numpy.ndarray or torch.Tensor): Bounding boxes [N, 4] in [x_min, y_min, x_max, y_max]
        labels (numpy.ndarray or torch.Tensor): Class labels [N]
        scores (numpy.ndarray or torch.Tensor): Confidence scores [N] (optional)
        thickness (int): Line thickness
    
    Returns:
        numpy.ndarray: Image with bounding boxes
    """
    # Convert to numpy if needed
    if torch.is_tensor(image):
        if image.shape[0] == 3:  # [C, H, W]
            image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
    
    image = image.copy()
    
    if torch.is_tensor(boxes):
        boxes = boxes.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if scores is not None and torch.is_tensor(scores):
        scores = scores.cpu().numpy()
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x_min, y_min, x_max, y_max = map(int, box)
        
        # Get color for this class
        color = COLORS.get(int(label), (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # Draw label
        label_text = CLASS_NAMES.get(int(label), f'class_{label}')
        if scores is not None:
            label_text = f'{label_text}: {scores[i]:.2f}'
        
        # Draw text background
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            image,
            (x_min, y_min - text_height - 5),
            (x_min + text_width, y_min),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            image,
            label_text,
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return image


def visualize_predictions(image, predictions, save_path=None, show=True):
    """
    Visualize model predictions on an image.
    
    Args:
        image (numpy.ndarray or torch.Tensor): Input image
        predictions (dict): Predictions with 'boxes', 'labels', 'scores'
        save_path (str): Path to save the visualization (optional)
        show (bool): Whether to display the image
    """
    # Draw bounding boxes
    vis_image = draw_bounding_boxes(
        image,
        predictions['boxes'],
        predictions['labels'],
        predictions.get('scores')
    )
    
    if show or save_path:
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_image)
        plt.axis('off')
        plt.title('Football Tracking Predictions')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        if show:
            plt.show()
        
        plt.close()
    
    return vis_image


def visualize_batch(images, targets, predictions=None, max_images=4):
    """
    Visualize a batch of images with ground truth and/or predictions.
    
    Args:
        images (list of tensors): Batch of images
        targets (list of dicts): Ground truth targets
        predictions (list of dicts): Model predictions (optional)
        max_images (int): Maximum number of images to visualize
    """
    n_images = min(len(images), max_images)
    fig, axes = plt.subplots(n_images, 2 if predictions else 1, figsize=(15, 5 * n_images))
    
    if n_images == 1:
        axes = [axes] if predictions else [[axes]]
    elif not predictions:
        axes = [[ax] for ax in axes]
    
    for i in range(n_images):
        image = images[i]
        target = targets[i]
        
        # Denormalize image
        if image.max() <= 1.0:
            vis_image = denormalize_image(image)
        else:
            vis_image = image.permute(1, 2, 0).numpy().astype(np.uint8)
        
        # Ground truth
        gt_image = draw_bounding_boxes(
            vis_image,
            target['boxes'],
            target['labels']
        )
        
        ax_idx = 0 if not predictions else 0
        axes[i][ax_idx].imshow(gt_image)
        axes[i][ax_idx].set_title('Ground Truth')
        axes[i][ax_idx].axis('off')
        
        # Predictions
        if predictions:
            pred_image = draw_bounding_boxes(
                vis_image,
                predictions[i]['boxes'],
                predictions[i]['labels'],
                predictions[i].get('scores')
            )
            axes[i][1].imshow(pred_image)
            axes[i][1].set_title('Predictions')
            axes[i][1].axis('off')
    
    plt.tight_layout()
    plt.show()


def save_video_with_predictions(video_path, predictions_list, output_path, fps=30):
    """
    Save a video with bounding box predictions overlaid.
    
    Args:
        video_path (str): Path to input video
        predictions_list (list): List of predictions for each frame
        output_path (str): Path to save output video
        fps (int): Frames per second
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx < len(predictions_list):
            pred = predictions_list[frame_idx]
            frame = draw_bounding_boxes(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                pred['boxes'],
                pred['labels'],
                pred.get('scores')
            )
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
