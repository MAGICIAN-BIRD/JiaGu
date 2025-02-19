import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import time
import os

# 数据集路径：你可以修改这里的路径，指向你的数据集
data_dir = r"path_to_u_data" # 修改为你的数据集地址

# 设置设备：如果有GPU，使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理流程：包括图像大小调整、标准化等
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像大小调整为VGG-16输入大小(112)
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # VGG-16的标准化
])

# 加载数据集并应用数据增强
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# 按照6:4比例划分数据集
train_size = int(0.6 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载预训练的VGG-16模型
model = models.vgg19(pretrained=True)

# 冻结预训练模型的卷积层，只训练最后的全连接层
for param in model.parameters():
    param.requires_grad = False

# 修改最后的全连接层，适应你的数据集类别数
num_classes = len(dataset.classes)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

# 将模型移动到GPU或CPU
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)  # 优化器


# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs=30):
    model.train()  # 设置为训练模式
    start_time = time.time()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")


# 评估模型
def evaluate_model(model, test_loader):
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


# 执行训练和评估
train_model(model, train_loader, criterion, optimizer, num_epochs=30)  # 你可以调整训练的epoch数
evaluate_model(model, test_loader)
