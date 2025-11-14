"""
Inference script for football tracking model.
Supports both image and video inference.
"""

import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from models.detector import FootballDetector
from utils.visualization import draw_bounding_boxes, visualize_predictions


def get_inference_transforms():
    """Get transforms for inference."""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def load_model(checkpoint_path, num_classes=3, device='cuda'):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        num_classes (int): Number of classes
        device (str): Device to load model on
    
    Returns:
        model: Loaded model in eval mode
    """
    model = FootballDetector(num_classes=num_classes, device=device)
    model.load(checkpoint_path)
    model.eval_mode()
    return model


def predict_image(model, image_path, output_path=None, confidence_threshold=0.5):
    """
    Run inference on a single image.
    
    Args:
        model: Trained model
        image_path (str): Path to input image
        output_path (str): Path to save output image (optional)
        confidence_threshold (float): Confidence threshold for predictions
    
    Returns:
        predictions (dict): Model predictions
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # Apply transforms
    transforms = get_inference_transforms()
    transformed = transforms(image=image_np)
    image_tensor = transformed['image'].to(model.device)
    
    # Run inference
    predictions = model.predict([image_tensor], confidence_threshold=confidence_threshold)[0]
    
    # Visualize results
    if output_path:
        vis_image = draw_bounding_boxes(
            image_np,
            predictions['boxes'].cpu().numpy(),
            predictions['labels'].cpu().numpy(),
            predictions['scores'].cpu().numpy()
        )
        Image.fromarray(vis_image).save(output_path)
        print(f'Saved result to {output_path}')
    
    return predictions


def predict_video(model, video_path, output_path, confidence_threshold=0.5, skip_frames=1):
    """
    Run inference on a video.
    
    Args:
        model: Trained model
        video_path (str): Path to input video
        output_path (str): Path to save output video
        confidence_threshold (float): Confidence threshold for predictions
        skip_frames (int): Process every Nth frame (1 = process all frames)
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f'Could not open video: {video_path}')
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f'Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames')
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Get transforms
    transforms = get_inference_transforms()
    
    frame_idx = 0
    predictions_cache = None
    
    pbar = tqdm(total=total_frames, desc='Processing video')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference on selected frames
        if frame_idx % skip_frames == 0:
            # Apply transforms
            transformed = transforms(image=frame_rgb)
            image_tensor = transformed['image'].to(model.device)
            
            # Run inference
            predictions = model.predict([image_tensor], confidence_threshold=confidence_threshold)[0]
            predictions_cache = predictions
        else:
            # Use cached predictions
            predictions = predictions_cache
        
        # Draw bounding boxes if we have predictions
        if predictions is not None:
            frame_rgb = draw_bounding_boxes(
                frame_rgb,
                predictions['boxes'].cpu().numpy(),
                predictions['labels'].cpu().numpy(),
                predictions['scores'].cpu().numpy()
            )
        
        # Convert back to BGR
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f'Saved output video to {output_path}')


def predict_frames(model, frames_dir, output_dir, confidence_threshold=0.5):
    """
    Run inference on a directory of frames.
    
    Args:
        model: Trained model
        frames_dir (str): Directory containing input frames
        output_dir (str): Directory to save output frames
        confidence_threshold (float): Confidence threshold for predictions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    print(f'Found {len(image_files)} frames')
    
    # Get transforms
    transforms = get_inference_transforms()
    
    for image_file in tqdm(image_files, desc='Processing frames'):
        # Load image
        image_path = os.path.join(frames_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Apply transforms
        transformed = transforms(image=image_np)
        image_tensor = transformed['image'].to(model.device)
        
        # Run inference
        predictions = model.predict([image_tensor], confidence_threshold=confidence_threshold)[0]
        
        # Draw bounding boxes
        vis_image = draw_bounding_boxes(
            image_np,
            predictions['boxes'].cpu().numpy(),
            predictions['labels'].cpu().numpy(),
            predictions['scores'].cpu().numpy()
        )
        
        # Save result
        output_path = os.path.join(output_dir, image_file)
        Image.fromarray(vis_image).save(output_path)
    
    print(f'Saved results to {output_dir}')


def main(args):
    """Main inference function."""
    # Load model
    print('Loading model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = load_model(args.checkpoint, num_classes=args.num_classes, device=device)
    print('Model loaded successfully!')
    
    # Run inference based on input type
    if args.image:
        print(f'Running inference on image: {args.image}')
        predictions = predict_image(
            model,
            args.image,
            args.output,
            confidence_threshold=args.confidence_threshold
        )
        print(f'Detected {len(predictions["boxes"])} objects')
        
    elif args.video:
        print(f'Running inference on video: {args.video}')
        predict_video(
            model,
            args.video,
            args.output,
            confidence_threshold=args.confidence_threshold,
            skip_frames=args.skip_frames
        )
        
    elif args.frames_dir:
        print(f'Running inference on frames in: {args.frames_dir}')
        predict_frames(
            model,
            args.frames_dir,
            args.output,
            confidence_threshold=args.confidence_threshold
        )
    
    else:
        print('Error: Must specify --image, --video, or --frames-dir')
        return
    
    print('Inference completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference with football tracking model')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num-classes', type=int, default=3,
                        help='Number of classes (including background)')
    
    # Input/output parameters
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to input video')
    parser.add_argument('--frames-dir', type=str, default=None,
                        help='Directory containing input frames')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save output')
    
    # Inference parameters
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                        help='Confidence threshold for predictions')
    parser.add_argument('--skip-frames', type=int, default=1,
                        help='Process every Nth frame in video (1 = all frames)')
    
    args = parser.parse_args()
    main(args)
