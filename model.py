from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
import torch.nn as nn
import torch



learning_rate = 3e-4
num_classes = 10
batch_size = 128
drop_rate = 0.1
in_channels = 3
patch_size = 4
emb_dim = 256
img_size = 32
num_heads = 8
mlp_dim = 512
epochs = 10
depth = 6
device = 'cuda'

class PatchEmbedding(nn.Module):

    def __init__(self, img_size, patch_size, in_channels, embd_dim):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels = in_channels,
                              out_channels = embd_dim,
                              kernel_size = patch_size,
                              stride = patch_size,)
        self.cls_token = nn.Parameter(torch.randn(1,1,embd_dim))
        self.pos_embd = nn.Parameter(torch.randn(1,1+num_patches, embd_dim))

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        cls_token = self.cls_token.expand(B,-1,-1)
        x = torch.cat((cls_token,x), dim=1)
        x = x + self.pos_embd
        
        return x

    
class MLP(nn.Module):

    def __init__(self, in_feature, hidden_features, drop_rate):
        super().__init__()
        self.fc1 = nn.Linear(in_feature, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_feature)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self,x):
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        
        return x
    

class TransformerEncoderLayer(nn.Module):

    def __init__(self, embd_dim, num_heads, mlp_dim, drop_rate):
        super().__init__()
        self.ln1 = nn.LayerNorm(embd_dim)
        self.attn =  nn.MultiheadAttention(embed_dim=embd_dim,
                                           num_heads=num_heads,
                                           dropout=drop_rate,
                                           batch_first=True)
        self.ln2 = nn.LayerNorm(embd_dim)
        self.mlp = MLP(in_feature=embd_dim,
                       hidden_features=mlp_dim,
                       drop_rate=drop_rate)
        
    def forward(self,x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x
    

class ViT(nn.Module):

    def __init__(self, img_size, patch_size, in_channels, num_classes,
                 emb_dim, mlp_dim, drop_rate, num_heads, depth):
        super().__init__()
        self.patch_emb = PatchEmbedding(img_size=img_size,
                                        patch_size=patch_size,
                                        in_channels=in_channels,
                                        embd_dim=emb_dim)
        self.enc = nn.Sequential(*[TransformerEncoderLayer(emb_dim, num_heads,mlp_dim,drop_rate) for _ in range(depth)])
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self,x):
        x = self.patch_emb(x)
        x = self.enc(x)
        x = self.norm(x)
        cls_token = x[:,0]
        return self.head(cls_token)
    
    
def get_dataloaders(batch_size, img_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5))
    ])

    train_dataset = datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=transform_test
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(model,loader, optimizer, criterion, scheduler=None):
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

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.inference_mode():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
    return correct / len(loader.dataset)

model = ViT(    
            img_size, 
            patch_size, 
            in_channels, 
            num_classes,
            emb_dim, 
            mlp_dim, 
            drop_rate, 
            num_heads, 
            depth
        )
model = model.to(device)

train_loader, test_loader = get_dataloaders(batch_size=128, img_size=32)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=0.05)
criterion = nn.CrossEntropyLoss()        

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=epochs
    )

for epoch in tqdm(range(epochs), desc="Epochs"):
    train_loss, train_acc = train(model,train_loader,optimizer,criterion,scheduler)
    test_acc = evaluate(model,test_loader)
    
    print(f"Epoch {epoch+1}/{epochs}: "
          f"Loss={train_loss:.4f}, "
          f"Train Acc={train_acc:.4f}, "
          f"Test Acc={test_acc:.4f}")

