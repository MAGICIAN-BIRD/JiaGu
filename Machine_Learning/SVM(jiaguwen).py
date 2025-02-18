import os
import time
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_data(directory):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        # 尝试调整以下参数以提高精确度
        transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.05, 2.5)),  # 更小的核和sigma值
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(root=directory, transform=transform)
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    images = []
    labels = []
    for data, label in data_loader:
        images.extend(data.view(data.size(0), -1).numpy())
        labels.extend(label.numpy())

    return images, labels

train_images, train_labels = load_data(r'Y:\py-torch\甲骨文切割\train_拓片')
test_images, test_labels = load_data(r'Y:\py-torch\甲骨文切割\test_拓片')
print("完成加载")

train_images = np.array(train_images)
test_images = np.array(test_images)

# 数据标准化
scaler = StandardScaler()
train_images = scaler.fit_transform(train_images)
test_images = scaler.transform(test_images)

clf = SVC(kernel='rbf')  # 尝试使用rbf核
print("已选择分类器")

X_train_part, X_val, y_train_part, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

start_time = time.time()
clf.fit(X_train_part, y_train_part)
train_time = time.time() - start_time

predicted_val = clf.predict(X_val)
val_accuracy = accuracy_score(y_val, predicted_val)

start_time = time.time()
predicted = clf.predict(test_images)
test_time = time.time() - start_time

accuracy = accuracy_score(test_labels, predicted)

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Training time: {train_time:.2f} seconds")
print(f"Testing time: {test_time:.2f} seconds")
print(f"Test Accuracy: {accuracy:.4f}")