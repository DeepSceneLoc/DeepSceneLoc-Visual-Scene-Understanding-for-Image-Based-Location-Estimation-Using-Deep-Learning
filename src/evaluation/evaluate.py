"""
Evaluation Module
Handles model evaluation and metrics calculation
Week 6 - Model Evaluation (Krishan Yadav - Lead)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Tuple, List
from tqdm import tqdm
import json
from pathlib import Path


class ModelEvaluator:
    """Handles model evaluation on test set"""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        class_names: List[str],
        device: str = 'cuda'
    ):
        """
        Initialize evaluator
        
        Args:
            model: Trained model to evaluate
            test_loader: Test data loader
            class_names: List of class names
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []
    
    def evaluate(self, use_tta: bool = True) -> Dict:
        """
        Evaluate model on test set with optional Test-Time Augmentation (TTA).
        
        Args:
            use_tta: If True, uses Horizontal Flip TTA to improve accuracy.
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        self.model.eval()
        
        print("="*60)
        print("Evaluating Model on Test Set")
        print("="*60)
        
        # Collect predictions
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Evaluating")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                if use_tta:
                    # Test-Time Augmentation: Original + Horizontal Flip
                    inputs_flipped = torch.flip(inputs, dims=[3])
                    outputs_flipped = self.model(inputs_flipped)
                    
                    prob_orig = torch.softmax(outputs, dim=1)
                    prob_flipped = torch.softmax(outputs_flipped, dim=1)
                    
                    # Average probabilities
                    probabilities = (prob_orig + prob_flipped) / 2.0
                else:
                    probabilities = torch.softmax(outputs, dim=1)
                    
                _, predicted = torch.max(probabilities, 1)
                
                # Store results
                self.all_predictions.extend(predicted.cpu().numpy())
                self.all_labels.extend(labels.cpu().numpy())
                self.all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        self.all_predictions = np.array(self.all_predictions)
        self.all_labels = np.array(self.all_labels)
        self.all_probabilities = np.array(self.all_probabilities)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Print results
        self._print_results(metrics)
        
        return metrics
    
    def _calculate_metrics(self) -> Dict:
        """Calculate all evaluation metrics"""
        
        # Overall accuracy
        overall_acc = accuracy_score(self.all_labels, self.all_predictions)
        
        # Per-class metrics
        precision = precision_score(
            self.all_labels,
            self.all_predictions,
            average=None,
            zero_division=0
        )
        
        recall = recall_score(
            self.all_labels,
            self.all_predictions,
            average=None,
            zero_division=0
        )
        
        f1 = f1_score(
            self.all_labels,
            self.all_predictions,
            average=None,
            zero_division=0
        )
        
        # Macro averages
        macro_precision = precision_score(
            self.all_labels,
            self.all_predictions,
            average='macro',
            zero_division=0
        )
        
        macro_recall = recall_score(
            self.all_labels,
            self.all_predictions,
            average='macro',
            zero_division=0
        )
        
        macro_f1 = f1_score(
            self.all_labels,
            self.all_predictions,
            average='macro',
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        
        # Per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        # Compile metrics
        metrics = {
            'overall_accuracy': float(overall_acc),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'per_class_metrics': {},
            'confusion_matrix': cm.tolist(),
            'total_samples': len(self.all_labels)
        }
        
        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics['per_class_metrics'][class_name] = {
                'accuracy': float(per_class_acc[i]),
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(cm[i].sum())
            }
        
        return metrics
    
    def _print_results(self, metrics: Dict):
        """Print evaluation results"""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
        
        print("\n" + "-"*60)
        print("Per-Class Performance:")
        print("-"*60)
        
        # Print table header
        print(f"{'Class':<15} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Support':<10}")
        print("-"*60)
        
        # Print per-class metrics
        for class_name, class_metrics in metrics['per_class_metrics'].items():
            print(
                f"{class_name:<15} "
                f"{class_metrics['accuracy']:.4f}  "
                f"{class_metrics['precision']:.4f}  "
                f"{class_metrics['recall']:.4f}  "
                f"{class_metrics['f1_score']:.4f}  "
                f"{class_metrics['support']:<10}"
            )
        
        print("="*60)
    
    def save_results(self, save_path: str):
        """Save evaluation results to JSON"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Evaluate if not done yet
        if len(self.all_predictions) == 0:
            metrics = self.evaluate()
        else:
            metrics = self._calculate_metrics()
        
        # Save to JSON
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nResults saved to {save_path}")
    
    def get_misclassified_samples(self, top_k: int = 10) -> List[Dict]:
        """
        Get information about misclassified samples
        
        Args:
            top_k: Number of worst predictions to return
            
        Returns:
            List of dictionaries with misclassification info
        """
        misclassified = []
        
        for i in range(len(self.all_predictions)):
            if self.all_predictions[i] != self.all_labels[i]:
                true_label = self.all_labels[i]
                pred_label = self.all_predictions[i]
                confidence = self.all_probabilities[i][pred_label]
                
                misclassified.append({
                    'index': i,
                    'true_label': self.class_names[true_label],
                    'predicted_label': self.class_names[pred_label],
                    'confidence': float(confidence),
                    'true_prob': float(self.all_probabilities[i][true_label])
                })
        
        # Sort by confidence (most confident wrong predictions first)
        misclassified.sort(key=lambda x: x['confidence'], reverse=True)
        
        return misclassified[:top_k]
    
    def print_confusion_analysis(self):
        """Print analysis of confusion matrix"""
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        
        print("\n" + "="*60)
        print("CONFUSION ANALYSIS")
        print("="*60)
        
        # Find most confused pairs
        n_classes = len(self.class_names)
        confusions = []
        
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm[i, j] > 0:
                    confusions.append({
                        'true': self.class_names[i],
                        'predicted': self.class_names[j],
                        'count': int(cm[i, j]),
                        'percentage': float(cm[i, j] / cm[i].sum() * 100)
                    })
        
        # Sort by count
        confusions.sort(key=lambda x: x['count'], reverse=True)
        
        print(f"\nTop Confusions:")
        print("-"*60)
        print(f"{'True Label':<15} {'Predicted As':<15} {'Count':<10} {'% of True':<10}")
        print("-"*60)
        
        for conf in confusions[:10]:
            print(
                f"{conf['true']:<15} "
                f"{conf['predicted']:<15} "
                f"{conf['count']:<10} "
                f"{conf['percentage']:.2f}%"
            )


def load_model_and_evaluate(
    model_path: str,
    model_name: str,
    test_loader: DataLoader,
    class_names: List[str],
    device: str = 'cuda'
) -> Dict:
    """
    Load a trained model and evaluate it
    
    Args:
        model_path: Path to model checkpoint
        model_name: Name of model architecture
        test_loader: Test data loader
        class_names: List of class names
        device: Device to run on
        
    Returns:
        Evaluation metrics dictionary
    """
    from src.models.model import create_model
    
    # Create model
    model = create_model(model_name, num_classes=len(class_names), pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from {model_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'unknown')}")
    
    # Create evaluator
    evaluator = ModelEvaluator(model, test_loader, class_names, device)
    
    # Evaluate
    metrics = evaluator.evaluate()
    
    # Show confusion analysis
    evaluator.print_confusion_analysis()
    
    # Show worst predictions
    print("\n" + "="*60)
    print("TOP 10 MISCLASSIFIED SAMPLES")
    print("="*60)
    misclassified = evaluator.get_misclassified_samples(top_k=10)
    for i, sample in enumerate(misclassified, 1):
        print(f"\n{i}. True: {sample['true_label']}, Predicted: {sample['predicted_label']}")
        print(f"   Confidence: {sample['confidence']:.4f}, True Prob: {sample['true_prob']:.4f}")
    
    return metrics


if __name__ == "__main__":
    """Example evaluation"""
    print("="*60)
    print("DeepSceneLoc - Model Evaluation")
    print("Week 6 - Krishan Yadav (Lead)")
    print("="*60)
    
    print("\nTo evaluate a model:")
    print("  from src.evaluation.evaluate import load_model_and_evaluate")
    print("  metrics = load_model_and_evaluate(")
    print("      model_path='models/checkpoints/best_model.pth',")
    print("      model_name='resnet50',")
    print("      test_loader=test_loader,")
    print("      class_names=['Coastal', 'Forest', 'Mountain', 'Rural', 'Urban']")
    print("  )")
