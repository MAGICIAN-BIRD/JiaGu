import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms, models
import time
import os
from tqdm import tqdm  # 引入进度条模块
from PIL import Image, ImageFilter
import pandas as pd  # 用于保存为CSV文件
from torch.cuda.amp import GradScaler, autocast  # 混合精度训练


# 数据集路径：你可以修改这里的路径，指向你的数据集
data_dir = r"Y:\py-torch\甲骨文切割\need_tapian"  # 修改为你的数据集地址

# 设置设备：如果有GPU，使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 定义高斯滤波函数
def gaussian_blur(image):
    return image.filter(ImageFilter.GaussianBlur(radius=2))

# 定义数据预处理流程：包括图像灰度化、高斯滤波、图像大小调整、标准化等
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像大小调整为224x224
    transforms.Grayscale(num_output_channels=3),  # 转为灰度图，并保持3通道
    transforms.Lambda(gaussian_blur),  # 使用普通函数进行高斯滤波
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # VGG-16的标准化
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # 转为灰度图，并保持3通道
    transforms.Lambda(gaussian_blur),  # 使用普通函数进行高斯滤波
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 自定义数据集类，修改数据划分逻辑
class CustomImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.imgs = self._make_dataset()

    def _make_dataset(self):
        imgs = []
        for cls in self.classes:
            cls_path = os.path.join(self.data_dir, cls)
            img_paths = [os.path.join(cls_path, fname) for fname in os.listdir(cls_path)]
            if len(img_paths) <= 1:
                imgs.extend([(path, self.class_to_idx[cls]) for path in img_paths])
            elif len(img_paths) < 5:
                half = len(img_paths) // 2
                imgs.extend([(path, self.class_to_idx[cls]) for path in img_paths[:half]])
                imgs.extend([(path, self.class_to_idx[cls]) for path in img_paths[half:]])
            else:
                imgs.extend([(path, self.class_to_idx[cls]) for path in img_paths])
        return imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# 加载数据集
dataset = CustomImageFolder(root_dir=data_dir, transform=transform_train)

# 按照6:2:2比例划分数据集
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# 创建数据加载器，并使用12个进程加载数据
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# 加载预训练的VGG-16模型
model = models.vgg16(pretrained=True)

# 冻结一些卷积层，仅冻结前几个层
for param in model.features[:10].parameters():
    param.requires_grad = False

# 修改最后的全连接层，适应你的数据集类别数
num_classes = len(dataset.classes)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

# 将模型移动到GPU或CPU
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

# 混合精度训练的Scaler
scaler = GradScaler()

# 训练模型并保存每轮测试结果为CSV
def train_model(model, train_loader, criterion, optimizer, num_epochs=30):
    model.train()  # 设置为训练模式
    start_time = time.time()

    results = []
    best_val_acc = 0.0  # 初始化最好的验证集准确率
    best_model_weights = None  # 用来保存最优模型权重

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)

            # 混合精度训练
            optimizer.zero_grad()
            with autocast():  # 开始自动混合精度训练
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # 反向传播与优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # 保存每一轮次的验证结果
        val_loss, val_acc = evaluate_model(model, val_loader)
        results.append([epoch + 1, epoch_loss, epoch_acc, val_loss, val_acc])

        # 如果当前模型在验证集上的精度更高，则保存该模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = model.state_dict()  # 保存模型权重

        # 更新学习率
        scheduler.step()

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # 保存训练过程中的所有结果为CSV
    filename = f"training_results_{int(time.time())}.csv"
    df = pd.DataFrame(results, columns=["Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"])
    df.to_csv(filename, index=False)
    print(f"Training results saved to {filename}")

    # 保存最优模型权重
    torch.save(best_model_weights, 'best_model.pth')
    print("Best model saved to best_model.pth")

def evaluate_model(model, val_loader):
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating", ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return val_loss, accuracy

def _test_model(model, test_loader):
    model.eval()

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    print("Loaded best model for testing.")

    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

    filename = f"test_results_{int(time.time())}.csv"
    df = pd.DataFrame({"Test Loss": [test_loss], "Test Accuracy": [accuracy]})
    df.to_csv(filename, index=False)
    print(f"Test results saved to {filename}")

# 执行训练、验证和测试
train_model(model, train_loader, criterion, optimizer, num_epochs=30)
_test_model(model, test_loader)
