import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),  # Convert to grayscale with 3 channels (RGB)
    transforms.GaussianBlur(kernel_size=5, sigma=1.0),  # Apply Gaussian blur
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = r"Y:\py-torch\甲骨文切割\JiaGuData_large_moben"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# 划分数据集
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

# 自定义数据集类
class ImageSequenceDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = img.permute(1, 2, 0)  # (H, W, C)
        img = img.reshape(224, -1)  # (224, 224*3=672)
        return img, label

# 定义RNN模型（修正输入维度问题）
class RNNClassifier(nn.Module):
    def __init__(self, input_size=672, hidden_size=128, num_classes=2, num_layers=2):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,  # 修正为实际输入维度672
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x形状: (batch_size, seq_len=224, input_size=672)
        out, _ = self.rnn(x)  # 删除不必要的permute操作
        out = out[:, -1, :]   # 取最后一个时间步
        out = self.fc(out)
        return out

def main():
    # 创建数据加载器（使用划分后的数据集）
    train_loader = DataLoader(
        ImageSequenceDataset(train_dataset),
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        ImageSequenceDataset(val_dataset),
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        ImageSequenceDataset(test_dataset),
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 初始化模型
    model = RNNClassifier(num_classes=len(dataset.classes)).to(device)

    # 定义损失函数和优化器（添加梯度裁剪）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 训练和评估
    num_epochs = 50
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 添加梯度裁剪
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"Train | Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating", ncols=100):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        val_loss /= len(val_loader)
        print(f"Val   | Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "RNN_best_model.pth")
            print(f"New best model saved with val acc {val_acc:.2f}%")

    # 测试最佳模型
    model.load_state_dict(torch.load("RNN_best_model.pth"))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100. * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == '__main__':
    main()
