import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm 

from model import ViT
from config import ViTConfig
from dataset import get_dataloaders


def train_epoch(model,loader, optimizer, criterion, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0

    for x,y in tqdm(loader, desc="Training", leave=False):
        x,y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)



def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.inference_mode():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
    return correct / len(loader.dataset)


def save_checkpoint(model, optimizer, epoch, train_loss, train_acc, test_acc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_acc': test_acc,
    }, path)
    print(f"Model saved to '{path}'")



def main():
    config = ViTConfig()
    
    # Setup
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    train_loader, test_loader = get_dataloaders(
        config.batch_size, 
        config.img_size,
        config.num_workers
    )
    
    # Model
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
    ).to(device)
    
    # Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs
    )
    
    # Training loop
    best_test_acc = 0.0
    for epoch in tqdm(range(config.epochs), desc="Epochs"):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, device
        )
        test_acc = evaluate(model, test_loader, device)
        
        print(f"Epoch {epoch+1}/{config.epochs}: "
              f"Loss={train_loss:.4f}, "
              f"Train Acc={train_acc:.4f}, "
              f"Test Acc={test_acc:.4f}")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_checkpoint(
                model, optimizer, epoch, train_loss, train_acc, test_acc,
                config.save_path
            )
    
    print(f"\nTraining complete! Best test accuracy: {best_test_acc:.4f}")


if __name__ == "__main__":
    main()