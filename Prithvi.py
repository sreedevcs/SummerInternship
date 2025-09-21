#!/usr/bin/env python3
"""
Prithvi Foundation Model for Cloud/Shadow Segmentation
Complete training pipeline for Cloud, No-cloud, and Shadow detection
"""

# =======
# IMPORTS AND SETUP
# =======
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import argparse
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import rasterio
import random
import gc
from terratorch import BACKBONE_REGISTRY 
from tqdm import tqdm

# Set random seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ===============================
# 1. DATASET CLASS
# ===============================

class BinaryCloudDataset(Dataset):
    def __init__(self, images_dir, masks_dir, channel_means, channel_stds, target_class):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.channel_means = np.array(channel_means)
        self.channel_stds = np.array(channel_stds)
        self.target_class = target_class
        
        # Get all image files
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.tif')])
        self.pairs = []
        
        for img_file in image_files:
            # Handle both original and augmented files
            if '_shadowaug_' in img_file:
                mask_file = img_file.replace('.tif', '_mask.tif')
            else:
                base, idx = img_file.rsplit('_', 1)
                mask_file = base + '_mask_' + idx
            
            mask_path = os.path.join(masks_dir, mask_file)
            img_path = os.path.join(images_dir, img_file)
            
            if os.path.exists(mask_path):
                self.pairs.append((img_path, mask_path))
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        # Load image (RGB)
        with rasterio.open(img_path) as src:
            image = np.stack([src.read(1), src.read(2), src.read(3)], axis=0)
            image = np.nan_to_num(image, nan=0.0).astype(np.float32)
        
        # Normalize
        for i in range(3):
            image[i] = (image[i] - self.channel_means[i]) / self.channel_stds[i]
        
        # Load mask
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.uint8)
        
        # Create binary mask for target class
        binary_mask = (mask == self.target_class).astype(np.float32)
        
        return torch.tensor(image, dtype=torch.float32), torch.tensor(binary_mask, dtype=torch.float32)

# ===============================
# 2. MODEL ARCHITECTURE
# ===============================

class SimpleSegmentationDecoder(nn.Module):
    def __init__(self, input_dim=1024, num_classes=1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
    
    def forward(self, features):
        if isinstance(features, list):
            x = features[-1]
        else:
            x = features
        
        # Handle CLS token removal
        if x.shape[1] == 197:
            x = x[:, 1:, :]
        
        B, N, C = x.shape
        H = W = int(N**0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.decoder(x)
        return x

def get_prithvi_model():
    encoder = BACKBONE_REGISTRY.build("terratorch_prithvi_eo_v2_300", pretrained=True)
    
    # Modify for 3 channels
    original_proj = encoder.patch_embed.proj
    new_proj = nn.Conv3d(
        in_channels=3,
        out_channels=original_proj.out_channels,
        kernel_size=original_proj.kernel_size,
        stride=original_proj.stride,
        padding=original_proj.padding,
        bias=original_proj.bias is not None
    )
    
    with torch.no_grad():
        new_proj.weight.data = original_proj.weight.data[:, :3, :, :, :]
        if new_proj.bias is not None:
            new_proj.bias.data = original_proj.bias.data
    
    encoder.patch_embed.proj = new_proj
    decoder = SimpleSegmentationDecoder(input_dim=1024, num_classes=1)
    
    class PrithviSegmentation(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
        
        def forward(self, x):
            if x.ndim == 4:
                x = x.unsqueeze(2)
            features = self.encoder(x)
            out = self.decoder(features)
            return out
    
    return PrithviSegmentation(encoder, decoder)

# ===============================
# 3. LOSS & METRICS
# ===============================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def iou_loss(inputs, targets, smooth=1e-6):
    inputs = torch.sigmoid(inputs)
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou

def compute_metrics(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    target_tp = ((y_pred == 1) & (y_true == 1)).sum()
    target_fp = ((y_pred == 1) & (y_true == 0)).sum()
    target_fn = ((y_pred == 0) & (y_true == 1)).sum()
    target_tn = ((y_pred == 0) & (y_true == 0)).sum()
    
    bg_tp = target_tn
    bg_fp = target_fn
    bg_fn = target_fp
    bg_tn = target_tp
    
    def safe_divide(a, b):
        return a / (b + 1e-8)
    
    target_precision = safe_divide(target_tp, target_tp + target_fp)
    target_recall = safe_divide(target_tp, target_tp + target_fn)
    target_f1 = safe_divide(2 * target_precision * target_recall, target_precision + target_recall)
    target_accuracy = safe_divide(target_tp + target_tn, target_tp + target_fp + target_fn + target_tn)
    target_iou = safe_divide(target_tp, target_tp + target_fp + target_fn)
    
    bg_precision = safe_divide(bg_tp, bg_tp + bg_fp)
    bg_recall = safe_divide(bg_tp, bg_tp + bg_fn)
    bg_f1 = safe_divide(2 * bg_precision * bg_recall, bg_precision + bg_recall)
    bg_accuracy = safe_divide(bg_tp + bg_tn, bg_tp + bg_fp + bg_fn + bg_tn)
    bg_iou = safe_divide(bg_tp, bg_tp + bg_fp + bg_fn)
    
    return {
        'Target IoU': float(target_iou), 'Background IoU': float(bg_iou),
        'Target F1': float(target_f1), 'Background F1': float(bg_f1),
        'Target Accuracy': float(target_accuracy), 'Background Accuracy': float(bg_accuracy),
        'Target Precision': float(target_precision), 'Background Precision': float(bg_precision),
        'Target Recall': float(target_recall), 'Background Recall': float(bg_recall)
    }

# ===============================
# 4. TRAINING FUNCTION
# ===============================

def train_prithvi_model(train_loader, val_loader, class_name, focal_params, device='cuda', n_epochs=1):
    model = get_prithvi_model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    focal_criterion = FocalLoss(alpha=focal_params['alpha'], gamma=focal_params['gamma'])
    
    best_combined_iou = 0
    patience_counter = 0
    patience = 8
    
    history = {
        'train_loss': [], 'val_loss': [], 'val_combined_iou': [],
        'val_target_iou': [], 'val_bg_iou': [], 'val_target_f1': [], 'val_bg_f1': [],
        'val_target_acc': [], 'val_bg_acc': [], 'val_target_prec': [], 'val_bg_prec': [],
        'val_target_rec': [], 'val_bg_rec': []
    }
    
    print(f"\n🚀 Training Prithvi binary model for {class_name}")
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss, batches = 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs).squeeze(1)
            loss_f = focal_criterion(outputs, labels)
            loss_i = iou_loss(outputs, labels)
            loss = loss_f + loss_i
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batches += 1
        
        train_loss /= max(1, batches)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss, val_preds, val_labels = 0, [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs).squeeze(1)
                loss = focal_criterion(outputs, labels) + iou_loss(outputs, labels)
                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                val_preds.append((probs > 0.5).cpu().numpy())
                val_labels.append(labels.cpu().numpy())
        
        val_loss /= max(1, len(val_loader))
        history['val_loss'].append(val_loss)
        
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        metrics = compute_metrics(val_labels, val_preds)
        
        # Update history
        history['val_combined_iou'].append((metrics['Target IoU'] + metrics['Background IoU']) / 2)
        history['val_target_iou'].append(metrics['Target IoU'])
        history['val_bg_iou'].append(metrics['Background IoU'])
        history['val_target_f1'].append(metrics['Target F1'])
        history['val_bg_f1'].append(metrics['Background F1'])
        history['val_target_acc'].append(metrics['Target Accuracy'])
        history['val_bg_acc'].append(metrics['Background Accuracy'])
        history['val_target_prec'].append(metrics['Target Precision'])
        history['val_bg_prec'].append(metrics['Background Precision'])
        history['val_target_rec'].append(metrics['Target Recall'])
        history['val_bg_rec'].append(metrics['Background Recall'])
        
        scheduler.step(metrics['Target IoU'] + metrics['Background IoU'])
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} Val Loss={val_loss:.4f}")
        
        print(f"\n🎯 {class_name} CLASS METRICS:")
        print(f"   IoU: {metrics['Target IoU']:.4f}")
        print(f"   Accuracy: {metrics['Target Accuracy']:.4f}")
        print(f"   Precision: {metrics['Target Precision']:.4f}")
        print(f"   Recall: {metrics['Target Recall']:.4f}")
        print(f"   F1 Score: {metrics['Target F1']:.4f}")
        
        print(f"\n🌫️ BACKGROUND CLASS METRICS:")
        print(f"   IoU: {metrics['Background IoU']:.4f}")
        print(f"   Accuracy: {metrics['Background Accuracy']:.4f}")
        print(f"   Precision: {metrics['Background Precision']:.4f}")
        print(f"   Recall: {metrics['Background Recall']:.4f}")
        print(f"   F1 Score: {metrics['Background F1']:.4f}")
        
        combined_iou = (metrics['Target IoU'] + metrics['Background IoU']) / 2
        if combined_iou > best_combined_iou:
            torch.save(model.state_dict(), f"best_prithvi_{class_name.lower()}.pth")
            print(f"✅ Saved best {class_name} model at epoch {epoch+1}")
            best_combined_iou = combined_iou
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > patience:
                print("⏹️ Early stopping due to no improvement!")
                break
        
        gc.collect()
        torch.cuda.empty_cache()
    
    return model, history


# =======
# MAIN EXECUTION WITH COMMAND LINE ARGUMENTS
# =======
def main():
    parser = argparse.ArgumentParser(description='Train Prithvi models for satellite image segmentation')
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='Path to dataset root directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: cuda, cpu, or auto (default: auto)')
    parser.add_argument('--classes', nargs='+', default=['CLOUD', 'NOCLOUD', 'SHADOW'],
                       choices=['CLOUD', 'NOCLOUD', 'SHADOW'],
                       help='Classes to train (default: all three)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for models and logs (default: ./outputs)')
    
    args = parser.parse_args()
    
    # Set up device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds
    set_seeds()
    
    # Dataset statistics (update these with your actual dataset statistics)
    means = [0.3813302780979631, 0.4006466170724529, 0.4106628602303568]
    stds = [0.2407521328203374, 0.2612074618803232, 0.20322935788977547]
    
    # Verify dataset structure
    required_dirs = [
        os.path.join(args.dataset_root, 'train', 'images'),
        os.path.join(args.dataset_root, 'train', 'masks'),
        os.path.join(args.dataset_root, 'val', 'images'),
        os.path.join(args.dataset_root, 'val', 'masks')
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Required directory not found: {dir_path}")
    
    print("✅ Dataset structure verified")
    
    # Filter classes to train
    class_configs = {
        'CLOUD': {
            'target_class': 1,
            'focal_params': {'alpha': 0.9, 'gamma': 2.0},
            'batch_size': 16
        },
        'NOCLOUD': {
            'target_class': 0,
            'focal_params': {'alpha': 0.8, 'gamma': 2.0},
            'batch_size': 16
        },
        'SHADOW': {
            'target_class': 2,
            'focal_params': {'alpha': 0.9, 'gamma': 3.0},
            'batch_size': 16
        }
    }
    
    # Filter configurations based on user selection
    filtered_configs = {k: v for k, v in class_configs.items() if k in args.classes}
    
    # Train models
    results = {}
    
    for class_name, config in filtered_configs.items():
        print(f"\n{'='*50}")
        print(f"TRAINING {class_name} MODEL")
        print(f"{'='*50}")
        
        try:
            # Create datasets
            train_dataset = BinaryCloudDataset(
                os.path.join(args.dataset_root, 'train', 'images'),
                os.path.join(args.dataset_root, 'train', 'masks'),
                means, stds, target_class=config['target_class']
            )
            
            val_dataset = BinaryCloudDataset(
                os.path.join(args.dataset_root, 'val', 'images'),
                os.path.join(args.dataset_root, 'val', 'masks'),
                means, stds, target_class=config['target_class']
            )
            
            print(f"{class_name} Training: {len(train_dataset)} samples")
            print(f"{class_name} Validation: {len(val_dataset)} samples")
            
            if len(train_dataset) == 0 or len(val_dataset) == 0:
                print(f"⚠️ Warning: No samples found for {class_name}. Skipping...")
                continue
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config['batch_size'], 
                shuffle=True, 
                num_workers=2
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=config['batch_size'], 
                shuffle=False, 
                num_workers=2
            )
            
            # Train model
            model, history = train_prithvi_model(
                train_loader, 
                val_loader, 
                class_name, 
                config['focal_params'], 
                device, 
                args.epochs
            )
            
            # Save outputs to specified directory
            model_path = os.path.join(args.output_dir, f"best_prithvi_{class_name.lower()}.pth")
            history_path = os.path.join(args.output_dir, f"prithvi_{class_name.lower()}_history.json")
            
            # Move saved model to output directory
            if os.path.exists(f"best_prithvi_{class_name.lower()}.pth"):
                os.rename(f"best_prithvi_{class_name.lower()}.pth", model_path)
            
            # Save history
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)
            
            results[class_name] = {
                'model_path': model_path,
                'history': history,
                'best_iou': max(history['val_combined_iou']) if history['val_combined_iou'] else 0.0
            }
            
            print(f"🎉 {class_name} model training completed!")
            print(f"📁 Model saved to: {model_path}")
            print(f"📊 History saved to: {history_path}")
            
        except Exception as e:
            print(f"❌ Error training {class_name} model: {str(e)}")
            continue
        finally:
            # Clean up memory
            if 'model' in locals():
                del model
            if 'train_dataset' in locals():
                del train_dataset, val_dataset, train_loader, val_loader
            gc.collect()
            torch.cuda.empty_cache()
    
    # Print final summary
    print(f"\n🎊 TRAINING COMPLETED! 🎊")
    print(f"\n📊 FINAL SUMMARY:")
    for class_name, result in results.items():
        print(f"{class_name}: Best Combined IoU = {result['best_iou']:.4f}")
        print(f"  Model: {result['model_path']}")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "training_summary.json")
    summary = {
        'args': vars(args),
        'results': {k: {'best_iou': v['best_iou'], 'model_path': v['model_path']} 
                   for k, v in results.items()}
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
