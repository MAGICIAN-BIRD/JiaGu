import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm
import torch.multiprocessing as mp
import numpy as np

# --- 1. 数据预处理（修改图像尺寸）---
transform = transforms.Compose([
    transforms.Resize((224,224)),  # 修改为112×112
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


# --- 3. 数据加载器 ---
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


# --- 5. 加载预训练的 ViT 模型并修改分类头 ---
def load_pretrained_vit(num_classes):
    # 加载预训练的 ViT 模型
    model = models.vit_l_32(weights="IMAGENET1K_V1")  # 使用最新的权重加载 ViT 模型
    # 获取当前分类头的输入特征数
    in_features = model.heads[0].in_features  # 获取输入特征数

    # 修改分类头以适应你的任务
    model.heads = nn.Linear(in_features, num_classes)

    return model


# --- 6. 进程保护 ---
if __name__ == '__main__':
    mp.freeze_support()  # 确保进程保护

    # 创建模型
    model = load_pretrained_vit(num_classes=len(train_data.dataset.label_to_idx)).to(device)

    # --- 优化配置 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.heads.parameters(), lr=3e-4, weight_decay=0.03)  # 只训练分类头

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
            torch.save(model.state_dict(), f'best_vit_{epoch_acc:.2f}.pth')
            print(f"New best model saved with accuracy {epoch_acc:.2f}")

    # 使用测试集评估最佳模型
    model.load_state_dict(torch.load(f'best_vit_{best_acc:.2f}.pth'))
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
