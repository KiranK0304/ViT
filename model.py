import torch.nn.functional as F
import torch.nn as nn
import torch


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
    
    