"""
Training Script for DeepSceneLoc
Week 5 - Model Training (Krishan Yadav - Lead)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
import json
import time
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt


class Trainer:
    """Handles model training with checkpointing and logging"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        save_dir: str = 'models/checkpoints',
        log_dir: str = 'logs',
        multi_gpu: bool = True,
        use_amp: bool = True,
    ):
        """
        Initialize trainer

        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to train on ('cuda' or 'cpu')
            save_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            multi_gpu: Wrap model in nn.DataParallel when >1 CUDA device is visible
            use_amp: Mixed-precision (autocast + GradScaler) on CUDA. T4 tensor
                cores make this ~2-3x faster than plain FP32.
        """
        self.model = model.to(device)
        self.use_dp = (
            multi_gpu
            and torch.device(device).type == 'cuda'
            and torch.cuda.device_count() > 1
        )
        if self.use_dp:
            print(f"  [DataParallel] Training across {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp and torch.device(device).type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        
        # Create directories
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass (AMP autocast)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            # Backward pass (AMP-safe scaling)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass (AMP autocast)
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100 * correct / total:.2f}%"
                })
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(
        self,
        num_epochs: int,
        start_epoch: int = 1,
        save_frequency: int = 5,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
    ):
        """
        Train the model
        
        Args:
            num_epochs: Number of epochs to train
            start_epoch: Epoch number to start from when resuming
            save_frequency: Save checkpoint every N epochs
            early_stopping_patience: Stop if validation accuracy does not improve
                by at least ``early_stopping_min_delta`` for this many epochs.
                If None, early stopping is disabled.
            early_stopping_min_delta: Minimum validation accuracy improvement
                required to reset patience counter.
        """
        print("="*60)
        print("Starting Training")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Epochs: {start_epoch} to {start_epoch + num_epochs - 1}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("="*60)
        
        start_time = time.time()
        
        end_epoch = start_epoch + num_epochs
        for epoch in range(start_epoch, end_epoch):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{end_epoch - 1}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # Save best model
            if val_acc > (self.best_val_acc + early_stopping_min_delta):
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"  [OK] Best model saved (Val Acc: {val_acc:.2f}%)")
            else:
                self.epochs_no_improve += 1
            
            # Save periodic checkpoint
            if epoch % save_frequency == 0:
                self.save_checkpoint(epoch, is_best=False)

            if (
                early_stopping_patience is not None
                and self.epochs_no_improve >= early_stopping_patience
            ):
                print(
                    f"  [STOP] Early stopping triggered after {self.epochs_no_improve} "
                    f"epochs without val improvement >= {early_stopping_min_delta:.4f}."
                )
                print("-"*60)
                break
            
            print("-"*60)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        
        # Save training history
        self.save_history()
        
        # Plot training curves
        self.plot_training_curves()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        model_to_save = self.model.module if self.use_dp else self.model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        if is_best:
            path = self.save_dir / 'best_model.pth'
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history to JSON"""
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\nTraining history saved to {history_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        save_path = self.log_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
        plt.close()


def setup_training(config: Dict):
    """
    Setup training from configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured trainer
    """
    from src.models.model import create_model
    from src.preprocessing.pipeline import create_dataloaders
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4),
        image_size=config.get('image_size', 224),
        augment_train=config.get('augment_train', True)
    )
    
    # Create model
    print("Creating model...")
    model = create_model(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        pretrained=config.get('pretrained', True)
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup loss function
    criterion = nn.CrossEntropyLoss()
    
    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # Setup scheduler
    scheduler = StepLR(
        optimizer,
        step_size=config.get('step_size', 7),
        gamma=config.get('gamma', 0.1)
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=config.get('save_dir', 'models/checkpoints'),
        log_dir=config.get('log_dir', 'logs')
    )
    
    return trainer


if __name__ == "__main__":
    """Example training setup"""
    
    # Example configuration
    config = {
        'data_dir': 'data/processed',
        'model_name': 'resnet50',
        'num_classes': 5,
        'batch_size': 32,
        'num_epochs': 20,
        'learning_rate': 0.001,
        'step_size': 7,
        'gamma': 0.1,
        'num_workers': 4,
        'image_size': 224,
        'augment_train': True,
        'pretrained': True,
        'save_dir': 'models/checkpoints',
        'log_dir': 'logs'
    }
    
    print("Training Configuration:")
    print(json.dumps(config, indent=2))
    print("\nTo start training:")
    print("  trainer = setup_training(config)")
    print("  trainer.train(num_epochs=20)")
