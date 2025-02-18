import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm  # 导入 tqdm 用于进度条可视化
import os
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练模块

# 设置 GPU 设备，如果 GPU 可用则使用 GPU，否则使用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# 数据增强和预处理
train_transform = transforms.Compose([
    transforms.Resize((64,64)),  # 修改图片的大小为 112x112
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(15),  # 随机旋转，角度范围为 -15 到 15 度
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),  # 随机裁剪并缩放到指定大小
    transforms.ToTensor(),  # 转换为 Tensor 格式，适合网络输入
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化（基于预训练模型）
])

val_transform = transforms.Compose([
    transforms.Resize((64,64)),  # 修改图片的大小为 112x112
    transforms.ToTensor(),  # 转换为 Tensor 格式，适合网络输入
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化（基于预训练模型）
])

test_transform = transforms.Compose([
    transforms.Resize((64,64)),  # 修改图片的大小为 112x112
    transforms.ToTensor(),  # 转换为 Tensor 格式，适合网络输入
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化（基于预训练模型）
])

# 加载数据集并应用数据增强（仅对训练集应用增强，验证集和测试集只进行标准化处理）
data_dir = r"Y:\py-torch\甲骨文切割\新版甲骨文数据集切割图片-摹本"  # 修改为你的数据集路径
train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)
test_dataset = datasets.ImageFolder(root=data_dir, transform=test_transform)

# 获取每个类别中的图片数量
class_counts = defaultdict(int)
for _, label in train_dataset.imgs:
    class_counts[label] += 1

# 重新定义训练集、验证集和测试集的划分
train_samples = []
val_samples = []
test_samples = []

# 遍历每个类别，按比例划分样本，特殊处理单张图片的类别
for label, count in class_counts.items():
    indices = [i for i, (_, lbl) in enumerate(train_dataset.imgs) if lbl == label]

    if count == 1:  # 如果某个类别只有一张图片，将它全部划分到测试集
        test_samples.extend(indices)
    else:
        # 按照 7:1:2 划分训练集、验证集和测试集
        train_split = int(0.7 * count)  # 70% 训练集
        val_split = int(0.1 * count)  # 10% 验证集
        test_split = count - train_split - val_split  # 剩下的部分为测试集

        # 手动划分数据
        train_samples.extend(indices[:train_split])
        val_samples.extend(indices[train_split:train_split+val_split])
        test_samples.extend(indices[train_split+val_split:])

# 创建新的数据集对象，确保每个类别的图片都分配到训练集、验证集和测试集中
train_dataset = Subset(train_dataset, train_samples)
val_dataset = Subset(val_dataset, val_samples)
test_dataset = Subset(test_dataset, test_samples)

# 使用 DataLoader 加载数据，指定 batch_size 和 shuffle 参数，开启四线程加速
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=6)
val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False, num_workers=6)

# 使用预训练的 ResNet-18 模型
model = models.resnet18(weights="IMAGENET1K_V1")

# 修改最后一层（全连接层），使其适应我们的分类数（即子文件夹的数量）
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.dataset.classes))  # 分类数为子文件夹的数量

# 将模型转移到 GPU 或 CPU
model = model.to(device)

# 定义损失函数（交叉熵损失）和优化器（Adam）
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 混合精度训练的Scaler
scaler = GradScaler()

# 训练模型
num_epochs = 10
# 设置训练轮数
if __name__ == "__main__":  # 添加这个保护语句，确保 multiprocessing 正常工作
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        correct = 0
        total = 0

        # 使用 tqdm 显示进度条，训练过程中迭代训练数据
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据转移到 GPU/CPU

            # 前向传播
            optimizer.zero_grad()  # 清空之前的梯度
            with autocast():  # 混合精度训练
                outputs = model(inputs)  # 获得模型输出
                loss = criterion(outputs, labels)  # 计算损失

            # 反向传播
            scaler.scale(loss).backward()  # 混合精度训练的反向传播
            scaler.step(optimizer)  # 更新权重
            scaler.update()  # 更新scaler

            # 统计训练集的损失和准确度
            running_loss += loss.item()  # 累加损失
            _, predicted = torch.max(outputs, 1)  # 获取预测结果
            total += labels.size(0)  # 总样本数
            correct += (predicted == labels).sum().item()  # 计算正确的预测数

        # 输出训练的损失和准确率
        epoch_loss = running_loss / len(train_loader)  # 平均损失
        epoch_acc = correct / total * 100  # 训练集的准确率

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # 验证集评估
        model.eval()  # 设置模型为评估模式，关闭 dropout 和 batch normalization
        val_correct = 0
        val_total = 0
        with torch.no_grad():  # 在验证阶段，不计算梯度
            for inputs, labels in tqdm(val_loader, desc="Validation", ncols=100):
                inputs, labels = inputs.to(device), labels.to(device)  # 将数据转移到 GPU/CPU
                outputs = model(inputs)  # 获得模型输出
                _, predicted = torch.max(outputs, 1)  # 获取预测结果
                val_total += labels.size(0)  # 总样本数
                val_correct += (predicted == labels).sum().item()  # 计算正确的预测数

        val_acc = val_correct / val_total * 100
        print(f"Validation Accuracy: {val_acc:.2f}%")

    # 测试模型
    model.eval()  # 设置模型为评估模式，关闭 dropout 和 batch normalization
    correct = 0
    total = 0

    # 在测试集上进行推理并显示进度条
    with torch.no_grad():  # 在推理阶段，不计算梯度
        for inputs, labels in tqdm(test_loader, desc="Testing", ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据转移到 GPU/CPU
            outputs = model(inputs)  # 获得模型输出
            _, predicted = torch.max(outputs, 1)  # 获取预测结果
            total += labels.size(0)  # 总样本数
            correct += (predicted == labels).sum().item()  # 计算正确的预测数

    # 输出测试集的准确率
    test_acc = correct / total * 100
    print(f"Test Accuracy: {test_acc:.2f}%")
