class ViTConfig:
    # Model architecture
    img_size = 32
    patch_size = 4
    in_channels = 3
    num_classes = 10
    emb_dim = 256
    mlp_dim = 512
    drop_rate = 0.1
    num_heads = 8
    depth = 6
    
    # Training
    batch_size = 128
    learning_rate = 3e-4
    weight_decay = 0.05
    epochs = 10
    
    # System
    device = 'cuda'
    num_workers = 4
    save_path = 'checkpoints/vit_cifar10.pth'