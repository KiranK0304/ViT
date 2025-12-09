import torch
from model import ViT
from config import ViTConfig


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
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']+1} epochs")
    print(f"Best test accuracy: {checkpoint['test_acc']:.4f}")
    
    return model


if __name__ == "__main__":
    config = ViTConfig()
    model = load_model(config.save_path)
    print("Model ready for inference!")