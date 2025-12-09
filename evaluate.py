import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm

from model import ViT
from config import ViTConfig
from dataset import get_dataloaders


def load_model(checkpoint_path, device='cuda'):
    config = ViTConfig()
    model = ViT(
        config.img_size,
        config.patch_size,
        config.in_channels,
        config.num_classes,
        config.emb_dim,
        config.mlp_dim,
        config.drop_rate,
        config.num_heads,
        config.depth
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint


def get_predictions(model, loader, device):
    """Get all predictions and true labels"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Getting predictions"):
            x = x.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    plt.close()


def plot_per_class_accuracy(y_true, y_pred, classes):
    """Plot per-class accuracy"""
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(classes)), per_class_acc)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.ylim(0, 1)
    
    # Color bars by performance
    for i, bar in enumerate(bars):
        if per_class_acc[i] < 0.5:
            bar.set_color('red')
        elif per_class_acc[i] < 0.7:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    plt.tight_layout()
    plt.savefig('per_class_accuracy.png', dpi=300, bbox_inches='tight')
    print("Per-class accuracy saved as 'per_class_accuracy.png'")
    plt.close()


def analyze_misclassifications(y_true, y_pred, classes):
    """Find most common misclassifications"""
    mistakes = []
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label != pred_label:
            mistakes.append((classes[true_label], classes[pred_label]))
    
    from collections import Counter
    most_common = Counter(mistakes).most_common(10)
    
    print("\n=== Top 10 Most Common Misclassifications ===")
    for (true_class, pred_class), count in most_common:
        print(f"{true_class:10s} -> {pred_class:10s}: {count:4d} times")


def main():
    config = ViTConfig()
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    # CIFAR-10 classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Load model
    print("Loading model...")
    model, checkpoint = load_model(config.save_path, device)
    print(f"Model trained for {checkpoint['epoch']+1} epochs")
    print(f"Training accuracy: {checkpoint['train_acc']:.4f}")
    print(f"Test accuracy: {checkpoint['test_acc']:.4f}")
    
    # Load test data
    print("\nLoading test data...")
    _, test_loader = get_dataloaders(config.batch_size, config.img_size, config.num_workers)
    
    # Get predictions
    print("\nEvaluating model...")
    y_pred, y_true, y_probs = get_predictions(model, test_loader, device)
    
    # Overall accuracy
    accuracy = (y_pred == y_true).mean()
    print(f"\n=== Overall Test Accuracy: {accuracy:.4f} ===")
    
    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, classes)
    
    # Plot per-class accuracy
    print("Generating per-class accuracy plot...")
    plot_per_class_accuracy(y_true, y_pred, classes)
    
    # Analyze misclassifications
    analyze_misclassifications(y_true, y_pred, classes)
    
    # Top-k accuracy
    top5_correct = 0
    for i, true_label in enumerate(y_true):
        top5_preds = np.argsort(y_probs[i])[-5:]
        if true_label in top5_preds:
            top5_correct += 1
    
    top5_acc = top5_correct / len(y_true)
    print(f"\n=== Top-5 Accuracy: {top5_acc:.4f} ===")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()