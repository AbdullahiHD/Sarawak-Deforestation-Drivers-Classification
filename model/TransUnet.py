import torch
import torch.nn as nn
from einops import rearrange
from torchvision.models import resnet50


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads, dim_head)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerBlock(dim, heads, dim_head, mlp_dim, dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransResNet(nn.Module):
    def __init__(
        self,
        img_dim,
        in_channels,
        out_channels,
        embed_dim=2048,
        depth=12,
        heads=16,
        mlp_dim=4096,
        dropout=0.1,
        num_classes=4,
    ):
        super().__init__()

        # ResNet encoder
        self.encoder = resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])

        # Transformer
        self.transformer = Transformer(embed_dim, depth, heads, 128, mlp_dim, dropout)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # Encoder
        features = self.encoder(x)

        # Reshape and apply Transformer
        b, c, h, w = features.shape
        features = features.flatten(2).transpose(1, 2)
        features = self.transformer(features)
        features = features.transpose(1, 2).view(b, c, h, w)

        # Decoder
        segmentation = self.decoder(features)

        # Classification
        classification = self.classifier(features)

        return segmentation, classification


def create_TransResNet(img_dim=160, in_channels=3, out_channels=1, num_classes=4):
    return TransResNet(img_dim, in_channels, out_channels, num_classes=num_classes)


def load_TransResNet(path, img_dim=160, in_channels=3, out_channels=1, num_classes=4):
    model = create_TransResNet(img_dim, in_channels, out_channels, num_classes)
    model.load_state_dict(torch.load(path))
    return model
