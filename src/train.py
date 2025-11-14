"""
Training script for football tracking model.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from data.dataset import FootballTrackingDataset, get_train_transforms, get_val_transforms, collate_fn
from models.detector import FootballDetector
from utils.visualization import visualize_batch


def train_one_epoch(model, dataloader, optimizer, device, epoch, writer=None):
    """
    Train the model for one epoch.
    
    Args:
        model: The detection model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        writer: TensorBoard writer
    
    Returns:
        Average loss for the epoch
    """
    model.train_mode()
    epoch_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, targets) in enumerate(pbar):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model.forward(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Track loss
        epoch_loss += losses.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': losses.item(),
            'loss_classifier': loss_dict['loss_classifier'].item(),
            'loss_box_reg': loss_dict['loss_box_reg'].item(),
            'loss_objectness': loss_dict['loss_objectness'].item(),
            'loss_rpn_box_reg': loss_dict['loss_rpn_box_reg'].item()
        })
        
        # Log to tensorboard
        if writer:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/total_loss', losses.item(), global_step)
            for loss_name, loss_value in loss_dict.items():
                writer.add_scalar(f'train/{loss_name}', loss_value.item(), global_step)
    
    return epoch_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device, epoch, writer=None):
    """
    Evaluate the model on validation set.
    
    Args:
        model: The detection model
        dataloader: Validation data loader
        device: Device to evaluate on
        epoch: Current epoch number
        writer: TensorBoard writer
    
    Returns:
        Average loss for validation set
    """
    model.train_mode()  # Keep in training mode to get losses
    epoch_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Validation')
    for images, targets in pbar:
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model.forward(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        epoch_loss += losses.item()
        pbar.set_postfix({'val_loss': losses.item()})
    
    avg_loss = epoch_loss / len(dataloader)
    
    if writer:
        writer.add_scalar('val/total_loss', avg_loss, epoch)
    
    return avg_loss


def main(args):
    """Main training function."""
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create datasets
    print('Loading datasets...')
    train_dataset = FootballTrackingDataset(
        root_dir=args.train_data_dir,
        annotation_file=args.train_annotations,
        transforms=get_train_transforms()
    )
    
    val_dataset = FootballTrackingDataset(
        root_dir=args.val_data_dir,
        annotation_file=args.val_annotations,
        transforms=get_val_transforms()
    )
    
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Val dataset size: {len(val_dataset)}')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Create model
    print('Creating model...')
    model = FootballDetector(num_classes=args.num_classes, device=device)
    
    # Create optimizer
    params = [p for p in model.model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir else None
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    print('Starting training...')
    best_val_loss = float('inf')
    
    for epoch in range(args.start_epoch, args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, writer)
        print(f'Train Loss: {train_loss:.4f}')
        
        # Validate
        val_loss = evaluate(model, val_loader, device, epoch, writer)
        print(f'Val Loss: {val_loss:.4f}')
        
        # Update learning rate
        lr_scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'model_epoch_{epoch + 1}.pth'
            )
            model.save(checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            model.save(best_model_path)
            print(f'Saved best model: {best_model_path}')
    
    # Save final model
    final_model_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    model.save(final_model_path)
    print(f'Saved final model: {final_model_path}')
    
    if writer:
        writer.close()
    
    print('Training completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train football tracking model')
    
    # Data parameters
    parser.add_argument('--train-data-dir', type=str, required=True,
                        help='Directory containing training images')
    parser.add_argument('--train-annotations', type=str, required=True,
                        help='Path to training annotations file')
    parser.add_argument('--val-data-dir', type=str, required=True,
                        help='Directory containing validation images')
    parser.add_argument('--val-annotations', type=str, required=True,
                        help='Path to validation annotations file')
    
    # Model parameters
    parser.add_argument('--num-classes', type=int, default=3,
                        help='Number of classes (including background)')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch (for resuming)')
    parser.add_argument('--learning-rate', type=float, default=0.005,
                        help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay')
    parser.add_argument('--lr-step-size', type=int, default=10,
                        help='Learning rate step size')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Learning rate decay factor')
    
    # Other parameters
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='runs',
                        help='Directory for tensorboard logs')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (YAML)')
    
    args = parser.parse_args()
    main(args)
