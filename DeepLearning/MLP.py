import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# 判断是否有可用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# 1. 数据预处理与加载
# -------------------------------

# 定义图像预处理：这里将图像调整为 28x28，转换为灰度（如果你希望使用彩色图像，可去掉Grayscale转换），转换为Tensor，并进行归一化
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 根据需要调整图像尺寸
    transforms.Grayscale(num_output_channels=1),  # 转换为灰度图（如果不需要，可注释此行）
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 均值和标准差，可根据数据集进行调整
])

# 加载数据集，注意：此处根目录为"need_摹本"，该文件夹下每个子文件夹即为一个类别
dataset = ImageFolder(root=r'Y:\py-torch\甲骨文切割\need_手写', transform=transform)

# -------------------------------
# 2. 划分训练集、验证集、测试集 (6:2:2)
# -------------------------------
dataset_size = len(dataset)
train_size = int(0.6 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = dataset_size - train_size - val_size  # 保证总数正确

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# 创建 DataLoader（可根据内存情况调整 batch_size）
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------
# 3. 定义 MLP 模型
# -------------------------------
# 由于图像尺寸为28x28，且为灰度图，所以输入特征数为 28*28 = 784
input_size = 28 * 28
hidden_size = 256  # 隐藏层单元数，可根据需要调整
num_classes = len(dataset.classes)  # 类别数，由文件夹个数决定


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),  # 将输入图像展开成一维向量
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)


model = MLP(input_size, hidden_size, num_classes).to(device)
print(model)

# -------------------------------
# 4. 定义损失函数和优化器
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# 5. 模型训练
# -------------------------------
num_epochs = 20  # 训练周期数，可根据数据集和实际效果调整

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # 将数据移动到GPU（如果可用）
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # 清零梯度
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / train_size
    train_acc = correct / total

    # 验证过程
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    val_loss /= val_size
    val_acc = correct_val / total_val

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# -------------------------------
# 6. 在测试集上评估模型
# -------------------------------
model.eval()
correct_test = 0
total_test = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()
test_acc = correct_test / total_test
print(f"Test Accuracy: {test_acc:.4f}")
