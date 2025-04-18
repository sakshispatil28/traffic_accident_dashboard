import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import video as video_models
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import transforms
import albumentations as A
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
import gc
import torch.cuda.amp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision.models import efficientnet_v2_l
from torchvision.models import ResNet50_Weights
import json
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Clean up memory
gc.collect()
torch.cuda.empty_cache()

# Set high precision for better accuracy
torch.set_float32_matmul_precision('high')

class Config:
    """Configuration settings for the model."""
    # Paths
    DATA_ROOT = "data/traffic_videos/"
    OUTPUT_DIR = "output/"
    CHECKPOINT_DIR = "checkpoints/"
    
    # Video processing
    VIDEO_FRAMES = 16  # Number of frames to sample per video
    FRAME_SIZE = (224, 224)  # Increased resolution for better feature extraction
    CLIP_DURATION = 3.0  # in seconds
    
    # Training
    BATCH_SIZE = 8  # Adjusted for most GPUs
    LEARNING_RATE = 2e-4
    EPOCHS = 100
    WEIGHT_DECAY = 5e-5
    
    # Model
    DROPOUT = 0.5
    HIDDEN_DIM = 1024
    
    # RCNN Parameters
    DETECTION_THRESHOLD = 0.6
    VEHICLE_CLASSES = [2, 3, 4, 6, 7, 8]  # COCO classes: car, motorcycle, bus, truck
    
    # Augmentations
    AUGMENTATION_PROB = 0.7
    
    # Optimization
    MIXED_PRECISION = True
    NUM_WORKERS = 4
    
    # Multi-model fusion
    USE_ENSEMBLE = True
    
    # K-fold cross validation
    USE_KFOLD = True
    NUM_FOLDS = 5
    
    # Focal loss parameters
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Seed
    SEED = 42


def set_seed(seed):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VideoDataset(Dataset):
    """Dataset for loading video clips with efficient memory usage."""
    def __init__(self, video_paths, labels=None, transform=None, clip_len=16, frame_size=(224, 224), mode='train'):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.clip_len = clip_len
        self.frame_size = frame_size
        self.mode = mode
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = self._load_video(video_path)
        
        # Apply transformations
        if self.transform:
            frames = self._apply_transform(frames)
        
        # Convert to tensor
        frames_tensor = torch.FloatTensor(frames)
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)  # [C, T, H, W]
        
        if self.labels is not None:
            label = self.labels[idx]
            return frames_tensor, torch.tensor(label, dtype=torch.long)
        else:
            return frames_tensor
    
    def _load_video(self, video_path):
        """Load video and extract frames efficiently."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"⚠️ Warning: Could not open video: {video_path}")
            return np.zeros((self.clip_len, *self.frame_size, 3))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Skip videos with too few frames
        if total_frames < self.clip_len:
            cap.release()
            return np.zeros((self.clip_len, *self.frame_size, 3))
        
        # Sample frames strategically - uniform sampling with slight randomness
        if self.mode == 'train':
            # Add randomness to frame selection during training
            start_idx = random.randint(0, max(0, total_frames - self.clip_len))
            indices = np.linspace(start_idx, start_idx + total_frames * 0.8, self.clip_len, dtype=int)
            indices = np.clip(indices, 0, total_frames - 1)
        else:
            # For validation/test, sample uniformly
            indices = np.linspace(0, total_frames - 1, self.clip_len, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, self.frame_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # If reading fails, add a black frame
                frames.append(np.zeros((*self.frame_size, 3), dtype=np.uint8))
        
        cap.release()
        return np.array(frames)
    
    def _apply_transform(self, frames):
        """Apply transformations to all frames."""
        transformed_frames = []
        
        if self.mode == 'train':
            # Apply the same transformation to all frames in the clip
            transform = self.transform() if callable(self.transform) else self.transform
            
            for frame in frames:
                frame = (frame * 255).astype(np.uint8)
                aug = transform(image=frame)
                transformed_frames.append(aug['image'])
        else:
            # For validation/test, just normalize
            for frame in frames:
                transformed_frames.append(self.transform(image=frame)['image'])
        
        return np.array(transformed_frames) / 255.0  # Normalize to 0-1


class ObjectDetector:
    """Enhanced object detector using FasterRCNN with ResNet50 FPN V2."""
    def __init__(self, threshold=0.7, vehicle_classes=[2, 3, 4, 6, 7, 8], device='cuda'):
        self.model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
        self.model.to(device)
        self.model.eval()
        self.threshold = threshold
        self.vehicle_classes = vehicle_classes
        self.device = device
        
    def detect_vehicles(self, frame):
        """Detect vehicles in a single frame."""
        # Convert numpy array to tensor
        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            predictions = self.model([img_tensor])
        
        # Filter detections by confidence and class
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Keep only vehicle classes above threshold
        mask = (scores >= self.threshold) & np.isin(labels, self.vehicle_classes)
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        return boxes, scores, labels
    
    def analyze_video_clip(self, frames):
        """Advanced analysis of a sequence of frames and extract vehicle features."""
        num_frames = len(frames)
        vehicle_counts = np.zeros(num_frames)
        motion_features = np.zeros((num_frames, 10))  # Enhanced feature set
        
        prev_boxes = None
        
        for i, frame in enumerate(frames):
            boxes, scores, labels = self.detect_vehicles(frame)
            
            # Count vehicles
            vehicle_counts[i] = len(boxes)
            
            # Extract motion features
            if len(boxes) > 0:
                # Average box size
                widths = boxes[:, 2] - boxes[:, 0]
                heights = boxes[:, 3] - boxes[:, 1]
                box_sizes = widths * heights
                avg_size = np.mean(box_sizes)
                
                # Size variance (indicates mixed vehicle types)
                size_variance = np.var(box_sizes) if len(box_sizes) > 1 else 0
                
                # Distribution of vehicle positions
                x_centers = (boxes[:, 0] + boxes[:, 2]) / 2
                y_centers = (boxes[:, 1] + boxes[:, 3]) / 2
                
                # Spatial distribution
                x_std = np.std(x_centers) if len(x_centers) > 1 else 0
                y_std = np.std(y_centers) if len(y_centers) > 1 else 0
                
                # Motion features if previous frame exists
                size_change, position_change, angle_change = 0, 0, 0
                
                if prev_boxes is not None and len(prev_boxes) > 0:
                    # Size change
                    prev_widths = prev_boxes[:, 2] - prev_boxes[:, 0]
                    prev_heights = prev_boxes[:, 3] - prev_boxes[:, 1]
                    prev_avg_size = np.mean(prev_widths * prev_heights)
                    size_change = avg_size - prev_avg_size
                    
                    # Position change
                    if len(boxes) > 0 and len(prev_boxes) > 0:
                        curr_centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, 
                                                       (boxes[:, 1] + boxes[:, 3]) / 2))
                        prev_centers = np.column_stack(((prev_boxes[:, 0] + prev_boxes[:, 2]) / 2, 
                                                       (prev_boxes[:, 1] + prev_boxes[:, 3]) / 2))
                        
                        # Calculate position changes
                        min_boxes = min(len(curr_centers), len(prev_centers))
                        if min_boxes > 0:
                            # Calculate optical flow-like features
                            position_changes = np.linalg.norm(curr_centers[:min_boxes] - prev_centers[:min_boxes], axis=1)
                            position_change = np.mean(position_changes)
                            
                            # Calculate angle changes
                            if min_boxes > 1:
                                vectors_curr = curr_centers[1:min_boxes] - curr_centers[0:min_boxes-1]
                                vectors_prev = prev_centers[1:min_boxes] - prev_centers[0:min_boxes-1]
                                
                                # Normalize vectors
                                norms_curr = np.linalg.norm(vectors_curr, axis=1)
                                norms_prev = np.linalg.norm(vectors_prev, axis=1)
                                
                                # Avoid division by zero
                                valid_idx = (norms_curr > 0) & (norms_prev > 0)
                                if np.any(valid_idx):
                                    unit_vectors_curr = vectors_curr[valid_idx] / norms_curr[valid_idx, np.newaxis]
                                    unit_vectors_prev = vectors_prev[valid_idx] / norms_prev[valid_idx, np.newaxis]
                                    
                                    # Calculate dot products
                                    dot_products = np.sum(unit_vectors_curr * unit_vectors_prev, axis=1)
                                    dot_products = np.clip(dot_products, -1.0, 1.0)  # Avoid numerical issues
                                    
                                    # Calculate angle changes
                                    angles = np.arccos(dot_products)
                                    angle_change = np.mean(angles)
                
                # Store all features
                motion_features[i, 0] = avg_size
                motion_features[i, 1] = size_change
                motion_features[i, 2] = position_change
                motion_features[i, 3] = angle_change
                motion_features[i, 4] = size_variance
                motion_features[i, 5] = x_std
                motion_features[i, 6] = y_std
                motion_features[i, 7] = len(boxes)  # Number of vehicles
                motion_features[i, 8] = np.mean(scores) if len(scores) > 0 else 0  # Average confidence
                motion_features[i, 9] = np.max(scores) if len(scores) > 0 else 0   # Max confidence
                
            prev_boxes = boxes
        
        # Calculate temporal features
        if num_frames > 1:
            vehicle_count_change = np.diff(vehicle_counts)
            vehicle_count_change = np.append(vehicle_count_change, vehicle_count_change[-1])
            
            # Create a feature vector
            features = np.column_stack((
                vehicle_counts,
                vehicle_count_change,
                motion_features
            ))
        else:
            features = np.column_stack((
                vehicle_counts,
                np.zeros_like(vehicle_counts),
                motion_features
            ))
        
        return features

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, inputs, targets):
        # Convert class targets to one-hot
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Calculate BCE loss
        bce_loss = self.bce(inputs, targets_one_hot)
        
        # Calculate focal weights
        pt = torch.exp(-bce_loss)
        focal_weights = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_weights = torch.ones_like(inputs) * self.alpha
        alpha_weights = torch.where(targets_one_hot == 1, alpha_weights, 1 - alpha_weights)
        
        # Apply weights to BCE loss
        focal_loss = alpha_weights * focal_weights * bce_loss
        
        # Reduce based on reduction type
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MixupTransform:
    """Mixup augmentation for videos."""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, batch_x, batch_y):
        """Apply mixup to a batch of videos."""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * batch_x + (1 - lam) * batch_x[index]
        y_a, y_b = batch_y, batch_y[index]
        
        return mixed_x, y_a, y_b, lam


class AccidentDetectionModel(nn.Module):
    """Advanced video-based accident detection model combining 3D ResNet, attention & object features."""
    def __init__(self, num_classes=2, dropout=0.5, hidden_dim=1024):
        super(AccidentDetectionModel, self).__init__()
        
        # Use a more advanced video backbone - X3D or SlowFast
        self.video_backbone = video_models.r3d_18(pretrained=True)
        
        # Replace the final fully connected layer
        backbone_out_features = self.video_backbone.fc.in_features
        self.video_backbone.fc = nn.Identity()
        
        # Image feature extraction (for key frames)
        self.img_backbone = efficientnet_v2_l(weights='DEFAULT')
        self.img_backbone.classifier = nn.Identity()
        img_features = 1280  # EfficientNetV2-L feature size
        
        # Motion features input size (enhanced set)
        motion_features_size = 12  # Based on features extracted by ObjectDetector
        
        # Self-attention for temporal modeling
        self.temporal_attention = nn.MultiheadAttention(backbone_out_features, num_heads=8, dropout=0.1)
        
        # Feature fusion with gating mechanism
        self.fusion_gate = nn.Sequential(
            nn.Linear(1804, 1804),
            nn.Sigmoid()
        )
        
        # Main fusion layers
        self.fusion_fc = nn.Sequential(
            nn.Linear(1804, 1024),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, video_input, motion_features=None, key_frames=None):
        # Extract video features
        video_features = self.video_backbone(video_input)
        
        # Apply self-attention for temporal modeling
        # Reshape for attention (seq_len, batch, hidden)
        batch_size = video_features.size(0)
        
        # Extract image features from key frames if provided
        img_features = torch.zeros(batch_size, 1280).to(video_input.device)
        if key_frames is not None:
            img_features = self.img_backbone(key_frames)
        
        if motion_features is not None:
            # Aggregate motion features across time dimension with attention weights
            motion_features_agg = torch.mean(motion_features, dim=1)
            
            # Additional motion statistics - min, max, std
            motion_stats = torch.cat([
                torch.min(motion_features, dim=1)[0],
                torch.max(motion_features, dim=1)[0],
                torch.std(motion_features, dim=1)
            ], dim=1)
            
            # Concatenate all features
            print(f"Video Features shape: {video_features.shape}")  
            print(f"Image Features shape: {img_features.shape}")  
            print(f"Motion Features shape: {motion_features_agg.shape}")  

            combined_features = torch.cat([video_features, img_features, motion_features_agg], dim=1)
            
            print(f"Combined Features shape (before fusion_gate): {combined_features.shape}")  
            
            # Apply gating mechanism
            gates = self.fusion_gate(combined_features)
            print(f"Gates shape: {gates.shape}")  
            print(f"Combined Features shape: {combined_features.shape}")

            gated_features = gates * combined_features
            
            # Final classification with gated features
            output = self.fusion_fc(gated_features)
        else:
            # If no motion features, just use video and image features
            combined_features = torch.cat([video_features, img_features], dim=1)
            output = self.fusion_fc(combined_features)
        
        return output


class SlowFastAccidentModel(nn.Module):
    """Implementation of SlowFast network for accident detection."""
    def __init__(self, num_classes=2, dropout=0.5):
        super(SlowFastAccidentModel, self).__init__()
        
        # Load pretrained SlowFast model
        self.slowfast = video_models.slowfast_r50(pretrained=True)
        
        # Get feature dimensions
        slow_features = self.slowfast.classifier[0].in_features
        
        # Replace classifier
        self.slowfast.classifier = nn.Identity()
        
        # Create our classifier
        self.classifier = nn.Sequential(
            nn.Linear(slow_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.slowfast(x)
        return self.classifier(features)


def get_train_transform():
    """Create advanced training augmentations."""
    return A.Compose([
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            A.ToGray(p=0.2),
            A.RandomBrightnessContrast(p=0.8),
        ], p=0.7),
        A.OneOf([
            A.GaussianBlur(p=0.3),
            A.MotionBlur(p=0.3),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.RandomRain(p=0.1),  # For weather variation
            A.RandomShadow(p=0.2),  # For shadow effects similar to accidents
        ], p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def create_transforms():
    """Create data transformations."""
    train_transform = get_train_transform
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def prepare_data(config):
    """Prepare datasets and dataloaders with a balanced approach."""
    accident_videos = [str(p) for p in Path(config.DATA_ROOT).glob("**/accident/*.mp4")]
    non_accident_videos = [str(p) for p in Path(config.DATA_ROOT).glob("**/non_accident/*.mp4")]
    
    if not accident_videos or not non_accident_videos:
        # Alternative structure
        all_videos = list(Path(config.DATA_ROOT).glob("**/*.mp4"))
        
        # Try to infer classes from filenames
        accident_videos = [str(p) for p in all_videos if 'accident' in p.stem.lower()]
        non_accident_videos = [str(p) for p in all_videos if 'accident' not in p.stem.lower()]
    
    print(f"Found {len(accident_videos)} accident videos and {len(non_accident_videos)} non-accident videos")
    
    if len(accident_videos) == 0 or len(non_accident_videos) == 0:
        raise ValueError(f"Could not find properly organized videos in {config.DATA_ROOT}. "
                        f"Expected either accident/non_accident folders or filenames containing 'accident'")
    
    # Create balanced dataset
    if len(accident_videos) > len(non_accident_videos):
        # Sample accident videos to match non-accident count
        accident_videos = random.sample(accident_videos, len(non_accident_videos))
    elif len(non_accident_videos) > len(accident_videos):
        # Sample non-accident videos to match accident count
        non_accident_videos = random.sample(non_accident_videos, len(accident_videos))
    
    # Combine videos and create labels
    all_videos = accident_videos + non_accident_videos
    labels = [1] * len(accident_videos) + [0] * len(non_accident_videos)
    
    # Create transforms
    train_transform, val_transform = create_transforms()
    
    if config.USE_KFOLD:
        return prepare_kfold_data(all_videos, labels, train_transform, val_transform, config)
    else:
        return prepare_train_val_data(all_videos, labels, train_transform, val_transform, config)
    
def prepare_train_val_data(all_videos, labels, train_transform, val_transform, config):
    """Prepare train/val split."""
        # Split into train and validation
    train_videos, val_videos, train_labels, val_labels = train_test_split(
        all_videos, labels, test_size=0.2, random_state=config.SEED, stratify=labels
    )
    
    # Create datasets
    train_dataset = VideoDataset(
        train_videos, train_labels, 
        transform=train_transform,
        clip_len=config.VIDEO_FRAMES,
        frame_size=config.FRAME_SIZE,
        mode='train'
    )
    
    val_dataset = VideoDataset(
        val_videos, val_labels,
        transform=val_transform,
        clip_len=config.VIDEO_FRAMES,
        frame_size=config.FRAME_SIZE,
        mode='val'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, None, train_dataset, None


def prepare_kfold_data(all_videos, labels, train_transform, val_transform, config):
    """Prepare data for k-fold cross-validation."""
    # Initialize k-fold
    kfold = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED)
    
    # Convert to numpy arrays for indexing
    videos_array = np.array(all_videos)
    labels_array = np.array(labels)
    
    # Create folds
    fold_datasets = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(videos_array, labels_array)):
        # Get train/val data for this fold
        fold_train_videos = videos_array[train_idx].tolist()
        fold_train_labels = labels_array[train_idx].tolist()
        fold_val_videos = videos_array[val_idx].tolist()
        fold_val_labels = labels_array[val_idx].tolist()
        
        # Create datasets
        train_dataset = VideoDataset(
            fold_train_videos, fold_train_labels, 
            transform=train_transform,
            clip_len=config.VIDEO_FRAMES,
            frame_size=config.FRAME_SIZE,
            mode='train'
        )
        
        val_dataset = VideoDataset(
            fold_val_videos, fold_val_labels,
            transform=val_transform,
            clip_len=config.VIDEO_FRAMES,
            frame_size=config.FRAME_SIZE,
            mode='val'
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )
        
        fold_datasets.append((train_loader, val_loader, train_dataset, val_dataset))
    
    # Return first fold by default (full set of folds is returned for k-fold training)
    return fold_datasets[0][0], fold_datasets[0][1], None, fold_datasets[0][2], fold_datasets, None

def generate_motion_features(frames, object_detector):
    """Generate motion features for a batch of video frames."""
    batch_size, C, T, H, W = frames.shape
    
    # Process each video in the batch
    batch_features = []
    
    for i in range(batch_size):
        # Get frames for this video and convert to numpy for OpenCV
        video_frames = frames[i].permute(1, 2, 3, 0).cpu().numpy()
        
        # Extract motion features
        features = object_detector.analyze_video_clip(video_frames)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).to(frames.device)
        batch_features.append(features_tensor)
    
    # Stack features from all videos in batch
    batch_features = torch.stack(batch_features)
    
    return batch_features


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config, fold=None):
    """Train the model with advanced training techniques."""
    device = config.DEVICE
    model.to(device)
    
    # Initialize mixup
    mixup = MixupTransform(alpha=0.2)
    
    # Create TensorBoard writer
    log_dir = Path(config.OUTPUT_DIR) / f"logs/fold_{fold}" if fold is not None else Path(config.OUTPUT_DIR) / "logs_new"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Initialize object detector for motion features
    object_detector = ObjectDetector(
        threshold=config.DETECTION_THRESHOLD, 
        vehicle_classes=config.VEHICLE_CLASSES, 
        device=device
    )
    
    # Initialize metrics tracking
    best_val_auc = 0.0
    best_epoch = 0
    
    # Create checkpoint directory
    ckpt_dir = Path(config.CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if config.MIXED_PRECISION else None
    
    # Main training loop
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        
        for batch_idx, (videos, targets) in enumerate(train_pbar):
            videos, targets = videos.to(device), targets.to(device)
            
            # Apply mixup augmentation with 50% probability
            use_mixup = (random.random() < 0.5)
            if use_mixup:
                videos, targets_a, targets_b, lam = mixup(videos, targets)
            
            # Generate motion features
            with torch.no_grad():
                motion_features = generate_motion_features(videos, object_detector)
            
            # Mixed precision training
            if config.MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    outputs = model(videos, motion_features)
                    
                    # Calculate loss
                    if use_mixup:
                        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                    else:
                        loss = criterion(outputs, targets)
                
                # Backward and optimize with gradient scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward pass
                outputs = model(videos, motion_features)
                
                # Calculate loss
                if use_mixup:
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    loss = criterion(outputs, targets)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            # Update scheduler
            scheduler.step(epoch + batch_idx / len(train_loader))
            
            # Update metrics
            train_loss += loss.item()
            
            # Calculate accuracy only if not using mixup
            if not use_mixup:
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f"{train_loss/(batch_idx+1):.4f}",
                    'acc': f"{100.*train_correct/train_total:.2f}%"
                })
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total if train_total > 0 else 0
        
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, criterion, object_detector, config)
        val_loss, val_acc, val_auc, val_f1, val_precision, val_recall = val_metrics
        
        # Scheduler step (if using epoch-based scheduler)
        if hasattr(scheduler, 'step') and not isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.EPOCHS} - {epoch_time:.2f}s - "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}% - "
            f"Val AUC: {val_auc:.4f} - Val F1: {val_f1:.4f}")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('AUC/val', val_auc, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        writer.add_scalar('Precision/val', val_precision, epoch)
        writer.add_scalar('Recall/val', val_recall, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save the best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            
            # Construct checkpoint path
            if fold is not None:
                ckpt_path = ckpt_dir / f"best_model_fold_{fold}.pth"
            else:
                ckpt_path = ckpt_dir / "best_model.pth"
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'config': {k: v for k, v in vars(config).items() if not k.startswith('__')}
            }, ckpt_path)
            
            print(f"🔥 New best model saved at epoch {epoch+1} with AUC: {val_auc:.4f}")
    
    print(f"\nTraining completed. Best model was at epoch {best_epoch+1} with AUC: {best_val_auc:.4f}")
    writer.close()
    
    return best_val_auc, best_epoch
    
def evaluate_model(model, dataloader, criterion, object_detector, config):
    """Evaluate the model on validation/test data."""
    device = config.DEVICE
    model.eval()
    
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_targets = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for videos, targets in tqdm(dataloader, desc="Validating"):
            videos, targets = videos.to(device), targets.to(device)
            
            # Generate motion features
            motion_features = generate_motion_features(videos, object_detector)
            
            # Forward pass
            if config.MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    outputs = model(videos, motion_features)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(videos, motion_features)
                loss = criterion(outputs, targets)
            
            # Calculate metrics
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
            
            # Save predictions and targets for overall metrics
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    # Calculate aggregate metrics
    val_loss = val_loss / len(dataloader)
    val_acc = 100. * val_correct / val_total
    
    # Convert to numpy arrays
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    
    # Calculate other metrics
    val_auc = roc_auc_score(all_targets, all_probs)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='binary'
    )
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Print detailed metrics
    print(f"Validation Results:")
    print(f"Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
    print(f"AUC: {val_auc:.4f}, F1: {val_f1:.4f}")
    print(f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
    print(f"Specificity: {specificity:.4f}, Sensitivity: {sensitivity:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    return val_loss, val_acc, val_auc, val_f1, val_precision, val_recall

def train_kfold(model_class, train_val_folds, criterion, config):
    """Train model using k-fold cross-validation."""
    fold_metrics = []
    
    for fold, (train_loader, val_loader, train_dataset, val_dataset) in enumerate(train_val_folds):
        print(f"\n{'='*40}\nTraining Fold {fold+1}/{len(train_val_folds)}\n{'='*40}")
        
        # Initialize a new model for this fold
        model = model_class(num_classes=2, dropout=config.DROPOUT, hidden_dim=config.HIDDEN_DIM)
        
        # Initialize optimizer and scheduler for this fold
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Cosine annealing with warm restarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the restart interval after each restart
            eta_min=config.LEARNING_RATE * 0.01  # Min learning rate
        )
        
        # Train the model for this fold
        best_auc, best_epoch = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, config, fold=fold
        )
        
        fold_metrics.append({
            'fold': fold,
            'best_auc': best_auc,
            'best_epoch': best_epoch
        })
    
    # Print summary of all folds
    print("\nK-Fold Cross-Validation Results:")
    for metrics in fold_metrics:
        print(f"Fold {metrics['fold']+1}: Best AUC = {metrics['best_auc']:.4f} at epoch {metrics['best_epoch']+1}")
    
    # Calculate average AUC across folds
    avg_auc = sum(m['best_auc'] for m in fold_metrics) / len(fold_metrics)
    print(f"\nAverage AUC across all folds: {avg_auc:.4f}")
    
    return fold_metrics


def ensemble_inference(models, video_path, object_detector, config):
    """Run inference with an ensemble of models."""
    # Load video and preprocess
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset with single video
    dataset = VideoDataset(
        [video_path], None, 
        transform=transform,
        clip_len=config.VIDEO_FRAMES,
        frame_size=config.FRAME_SIZE,
        mode='test'
    )
    
    # Get video frames
    frames = dataset[0]
    frames = frames.unsqueeze(0).to(config.DEVICE)  # Add batch dimension
    
    # Generate motion features
    motion_features = generate_motion_features(frames, object_detector)
    
    # Run inference with each model
    all_probs = []
    for model in models:
        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=config.MIXED_PRECISION):
                outputs = model(frames, motion_features)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_probs.append(probs)
    
    # Average probabilities from all models
    avg_prob = np.mean(all_probs, axis=0)[0]
    prediction = int(avg_prob >= 0.5)
    
    return prediction, avg_prob


def predict_video(model, video_path, object_detector, config):
    """Make a prediction on a single video."""
    # Set model to evaluation mode
    model.eval()
    
    # Preprocess video
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset with single video
    dataset = VideoDataset(
        [video_path], None, 
        transform=transform,
        clip_len=config.VIDEO_FRAMES,
        frame_size=config.FRAME_SIZE,
        mode='test'
    )
    
    # Get video frames
    frames = dataset[0]
    frames = frames.unsqueeze(0).to(config.DEVICE)  # Add batch dimension
    
    # Generate motion features
    motion_features = generate_motion_features(frames, object_detector)
    
    # Make prediction
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=config.MIXED_PRECISION):
            outputs = model(frames, motion_features)
            probs = torch.softmax(outputs, dim=1)
            
    # Get prediction and probability
    pred_class = torch.argmax(probs, dim=1).item()
    accident_prob = probs[0, 1].item()
    
    return pred_class, accident_prob


def load_model(model_path, model_class, config):
    """Load a trained model from checkpoint."""
    # Initialize model
    model = model_class(num_classes=2, dropout=config.DROPOUT, hidden_dim=config.HIDDEN_DIM)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    model = model.to(config.DEVICE)
    
    return model
def load_ensemble_models(fold_paths, model_class, config):
    """Load models from different folds for ensemble prediction."""
    models = []
    
    for path in fold_paths:
        model = load_model(path, model_class, config)
        models.append(model)
    
    return models
def visualize_results(video_path, pred_class, accident_prob, save_path=None):
    """Visualize prediction results with key frames from the video."""
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video details
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    # Sample key frames
    num_frames = 4
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # Create figure
    fig, axes = plt.subplots(1, num_frames, figsize=(16, 4))
    
    # Extract and display frames
    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axes[i].imshow(frame)
            axes[i].set_title(f"Frame {idx}")
            axes[i].axis('off')
    
    # Add prediction information
    plt.suptitle(f"Prediction: {'ACCIDENT' if pred_class else 'NO ACCIDENT'} (Confidence: {accident_prob:.2%})", 
                fontsize=16, y=0.95)
    plt.figtext(0.5, 0.01, f"Video: {Path(video_path).name} | Duration: {duration:.2f}s | Frames: {total_frames}", 
                ha='center', fontsize=12)
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()
    cap.release()
def visualize_attention(model, video_path, object_detector, config, save_path=None):
    """Visualize attention weights on key frames of the video."""
    # Load and preprocess video
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset with single video
    dataset = VideoDataset(
        [video_path], None, 
        transform=transform,
        clip_len=config.VIDEO_FRAMES,
        frame_size=config.FRAME_SIZE,
        mode='test'
    )
    
    # Get original frames (before normalization) for visualization
    cap = cv2.VideoCapture(video_path)
    original_frames = []
    for i in range(config.VIDEO_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // config.VIDEO_FRAMES))
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, config.FRAME_SIZE)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frames.append(frame)
    cap.release()
    
    # Get video frames tensor
    frames = dataset[0]
    frames = frames.unsqueeze(0).to(config.DEVICE)  # Add batch dimension
    
    # Generate motion features
    motion_features = generate_motion_features(frames, object_detector)
    
    # Get prediction with attention weights
    model.eval()
    
    # We need to modify the forward pass to capture attention weights
    # This is a simplified version assuming we have access to attention weights
    with torch.no_grad():
        outputs = model(frames, motion_features)
        
    # For visualization, we'll assume we have attention weights
    # In practice, you'd need to modify the model to return these
    # This is just a placeholder for demonstration
    attention_weights = torch.ones(config.VIDEO_FRAMES) / config.VIDEO_FRAMES
    
    # Plot frames with attention overlay
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (frame, weight) in enumerate(zip(original_frames[:8], attention_weights[:8])):
        axes[i].imshow(frame)
        axes[i].set_title(f"Attention: {weight.item():.4f}")
        axes[i].axis('off')
    
    plt.suptitle(f"Attention Visualization for {Path(video_path).name}", fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()
def create_confusion_matrix_plot(y_true, y_pred, save_path=None):
    """Create and save confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=['No Accident', 'Accident'],
                yticklabels=['No Accident', 'Accident'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    # Save or display
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()
def create_roc_curve_plot(y_true, y_prob, save_path=None):
    """Create and save ROC curve visualization."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    # Save or display
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()
def create_precision_recall_curve(y_true, y_prob, save_path=None):
    """Create and save precision-recall curve visualization."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    # Save or display
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()
def evaluate_test_data(model, test_loader, object_detector, config, output_dir):
    """Comprehensive evaluation on test data with visualizations."""
    device = config.DEVICE
    model.eval()
    
    all_targets = []
    all_probs = []
    all_preds = []
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for videos, targets in tqdm(test_loader, desc="Evaluating Test Data"):
            videos, targets = videos.to(device), targets.to(device)
            
            # Generate motion features
            motion_features = generate_motion_features(videos, object_detector)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=config.MIXED_PRECISION):
                outputs = model(videos, motion_features)
                probs = torch.softmax(outputs, dim=1)[:, 1]
            
            # Get predictions
            preds = (probs >= 0.5).int()
            
            # Save results
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # Convert to numpy arrays
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    auc_score = roc_auc_score(all_targets, all_probs)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary')
    
    # Print results
    print("\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc_score:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Create visualizations
    create_confusion_matrix_plot(all_targets, all_preds, save_path=output_dir / "confusion_matrix.png")
    create_roc_curve_plot(all_targets, all_probs, save_path=output_dir / "roc_curve.png")
    create_precision_recall_curve(all_targets, all_probs, save_path=output_dir / "pr_curve.png")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'true': all_targets,
        'pred': all_preds,
        'prob': all_probs
    })
    results_df.to_csv(output_dir / "test_results.csv", index=False)
    
    # Save metrics to JSON
    metrics = {
        'accuracy': float(accuracy),
        'auc': float(auc_score),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    }
    
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics
def real_time_detection(video_path, model, object_detector, config, output_path=None, display=True):
    """Run accident detection in (simulated) real-time and visualize results."""
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Setup output video if needed
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize frame buffer
    frame_buffer = []
    buffer_size = config.VIDEO_FRAMES
    prediction_history = []
    
    # Initialize normalization transform
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Set model to evaluation mode
    model.eval()
    
    # Process video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame for model input
        model_frame = cv2.resize(frame, config.FRAME_SIZE)
        model_frame_rgb = cv2.cvtColor(model_frame, cv2.COLOR_BGR2RGB)
        
        # Normalize frame
        norm_frame = transform(image=model_frame_rgb)['image']
        
        # Add to buffer
        frame_buffer.append(norm_frame)
        
        # Keep buffer at fixed size
        if len(frame_buffer) > buffer_size:
            frame_buffer.pop(0)
        
        # Make prediction once we have enough frames
        if len(frame_buffer) == buffer_size:
            # Convert buffer to tensor
            frames_tensor = np.array(frame_buffer)
            frames_tensor = torch.FloatTensor(frames_tensor).permute(3, 0, 1, 2).unsqueeze(0).to(config.DEVICE)
            
            # Generate motion features
            motion_features = generate_motion_features(frames_tensor, object_detector)
            
            # Get prediction
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=config.MIXED_PRECISION):
                    outputs = model(frames_tensor, motion_features)
                    probs = torch.softmax(outputs, dim=1)
                    accident_prob = probs[0, 1].item()
            
            # Add to prediction history
            prediction_history.append(accident_prob)
            
            # Use moving average for stability
            if len(prediction_history) > 5:
                prediction_history.pop(0)
            
            avg_prob = sum(prediction_history) / len(prediction_history)
            
            # Display prediction on frame
            text = f"Accident Probability: {avg_prob:.2%}"
            color = (0, 0, 255) if avg_prob > 0.5 else (0, 255, 0)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Add vehicle detections
            boxes, scores, labels = object_detector.detect_vehicles(np.array(frame_buffer[-1]))
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.astype(int)
                # Scale coordinates to original frame size
                x1 = int(x1 * width / config.FRAME_SIZE[0])
                y1 = int(y1 * height / config.FRAME_SIZE[1])
                x2 = int(x2 * width / config.FRAME_SIZE[0])
                y2 = int(y2 * height / config.FRAME_SIZE[1])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Write frame to output video
        if out is not None:
            out.write(frame)
        
        # Display frame
        if display:
            cv2.imshow('Traffic Accident Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
def main():
    """Main function to run the traffic accident detection system."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Traffic Accident Detection System')
    parser.add_argument('--data', type=str, default='data/traffic_videos/', help='Path to data directory')
    parser.add_argument('--output', type=str, default='output/', help='Path to output directory')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'predict'], help='Operation mode')
    parser.add_argument('--model', type=str, default='standard', choices=['standard', 'slowfast'], help='Model type')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for testing/prediction')
    parser.add_argument('--video', type=str, help='Path to video file for prediction')
    parser.add_argument('--kfold', action='store_true', help='Use k-fold cross-validation')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble of models')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    config.DATA_ROOT = args.data
    config.OUTPUT_DIR = args.output
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    config.SEED = args.seed
    config.USE_KFOLD = args.kfold
    config.USE_ENSEMBLE = args.ensemble
    config.MIXED_PRECISION = args.mixed_precision
    
    # Set up directories
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    set_seed(config.SEED)
    
    # Initialize model class based on argument
    if args.model == 'standard':
        model_class = AccidentDetectionModel
    else:
        model_class = SlowFastAccidentModel
    
    # Initialize object detector
    object_detector = ObjectDetector(
        threshold=config.DETECTION_THRESHOLD,
        vehicle_classes=config.VEHICLE_CLASSES,
        device=config.DEVICE
    )
    
    # Mode-specific operations
    if args.mode == 'train':
        # Prepare data
        if config.USE_KFOLD:
            train_loader, val_loader, _, train_dataset, fold_datasets = prepare_data(config)
        else:
            train_loader, val_loader, _, train_dataset, _ = prepare_data(config)
        
        # Create model
        model = model_class(num_classes=2, dropout=config.DROPOUT, hidden_dim=config.HIDDEN_DIM)
        
        # Initialize optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Initialize scheduler
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=config.LEARNING_RATE * 0.01
        )
        
        # Initialize loss function
        criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
        
        # Train model
        if config.USE_KFOLD:
            train_kfold(model_class, fold_datasets, criterion, config)
        else:
            train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config)
            
    elif args.mode == 'test':
        # Check if checkpoint is provided
        if not args.checkpoint:
            print("Error: Checkpoint path must be provided for test mode")
            return
        
        # Prepare data
        _, val_loader, _, _, _ = prepare_data(config)
        
        # Load model
        model = load_model(args.checkpoint, model_class, config)
        
        # Evaluate model
        evaluate_test_data(model, val_loader, object_detector, config, output_dir)
        
    elif args.mode == 'predict':
        # Check if video path is provided
        if not args.video:
            print("Error: Video path must be provided for predict mode")
            return
        
        # Check if checkpoint is provided
        if not args.checkpoint:
            print("Error: Checkpoint path must be provided for predict mode")
            return
        
        # Load model
        if config.USE_ENSEMBLE:
            # Find all checkpoints for ensemble
            checkpoint_paths = list(checkpoint_dir.glob("best_model_fold_*.pth"))
            if not checkpoint_paths:
                print("Error: No fold checkpoints found for ensemble")
                return
            
            # Load ensemble models
            models = load_ensemble_models(checkpoint_paths, model_class, config)
            
            # Make prediction
            pred_class, prob = ensemble_inference(models, args.video, object_detector, config)
        else:
            # Load single model
            model = load_model(args.checkpoint, model_class, config)
            
            # Make prediction
            pred_class, prob = predict_video(model, args.video, object_detector, config)
        
        # Print prediction
        print(f"Prediction: {'ACCIDENT' if pred_class else 'NO ACCIDENT'}")
        print(f"Probability: {prob:.2%}")
        
        # Visualize results if requested
        if args.visualize:
            if config.USE_ENSEMBLE:
                # Use first model for visualization
                model = models[0]
            
            # Visualize prediction
            visualize_results(args.video, pred_class, prob, save_path=output_dir / "prediction.png")
            
            # Run real-time detection
            real_time_detection(
                args.video, 
                model, 
                object_detector, 
                config, 
                output_path=output_dir / "detection_output.mp4",
                display=True
            )
if __name__ == "__main__":
    main()