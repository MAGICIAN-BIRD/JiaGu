import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch.multiprocessing as mp

# --- 1. 数据预处理（修改图像尺寸）---
transform = transforms.Compose([
    transforms.Resize((112, 112)),  # 修改为112×112
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# --- 2. 自定义数据集 ---
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        # 创建标签映射字典
        self.label_to_idx = {}
        self.idx_to_label = {}
        labels = set(self.classes)  # 直接将文件夹名称作为标签
        sorted_labels = sorted(labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        class_name = self.classes[target]
        label = class_name  # 使用文件夹名称作为标签
        return img, self.label_to_idx[label]


# --- 3. 数据加载器（适当增加batch_size）---
dataset = CustomImageFolder(root=r'Y:\py-torch\甲骨文切割\JiaGuData_large_tapian', transform=transform)

# 数据集划分：训练集80%、验证集10%、测试集10%
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)


# --- 4. 设备设置 ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 5. 优化后的ViT实现（适配新尺寸）---
class VisionTransformer(nn.Module):
    def __init__(self, img_size=112,  # 修改为112
                 patch_size=16,
                 in_channels=3,
                 num_classes=len(train_data.dataset.label_to_idx),
                 dim=512,  # 减小嵌入维度
                 depth=6,  # 减少层数
                 heads=8,  # 减少注意力头
                 mlp_dim=1024,
                 dropout=0.1):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * (patch_size ** 2)

        # Patch Embedding
        self.patch_embedding = nn.Linear(self.patch_dim, dim)

        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Positional Embedding
        self.positional_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, dim) * 0.02
        )

        # Transformer
        self.transformer = nn.Sequential(*[
            TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])

        # Classification Head
        self.classification_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        batch_size, c, h, w = x.shape

        # 分块处理
        patch_size = h // int(self.num_patches ** 0.5)
        x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        x = x.contiguous().view(batch_size, self.num_patches, -1)

        # 投影和拼接
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positional_embedding

        # Transformer处理
        x = self.transformer(x)

        # 分类头
        return self.classification_head(x[:, 0])


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = MultiHeadSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == '__main__':
    mp.freeze_support()

    # --- 初始化模型 ---
    model = VisionTransformer(
        img_size=112,
        patch_size=16,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024
    ).to(device)

    # --- 优化配置 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.03)  # 调整学习率

    # --- 混合精度训练 ---
    scaler = torch.cuda.amp.GradScaler()

    # --- 训练循环 ---
    epochs = 50
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                # 混合精度前向
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                # 混合精度反向
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # 统计指标
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{correct / total:.2%}"
                })

        # 验证模型
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2%}")

        # 保存最佳模型
        epoch_acc = correct / total
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), f'Y:\\py-torch\甲骨文切割\\transformer\\拓片预训练模型\\best_vit_112px_acc{epoch_acc:.2f}.pth')
            print(f"New best model saved with accuracy {epoch_acc:.2%}")

    # 使用测试集评估最佳模型
    model.load_state_dict(torch.load(f'Y:\\py-torch\甲骨文切割\\transformer\\拓片预训练模型\\best_vit_112px_acc{best_acc:.2f}.pth'))
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc:.2%}")

    print("Training complete!")
