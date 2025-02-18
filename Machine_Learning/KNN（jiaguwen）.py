import os  # 导入os模块，用于处理文件和目录路径
import time  # 导入time模块，用于测量时间
from sklearn.neighbors import KNeighborsClassifier  # 从sklearn库导入KNeighborsClassifier类，用于KNN分类
from sklearn.metrics import accuracy_score  # 从sklearn库导入accuracy_score函数，用于计算准确率
from torchvision import datasets, transforms  # 从torchvision库导入datasets和transforms模块，用于加载和预处理数据集
from torch.utils.data import DataLoader  # 从torch.utils.data导入DataLoader类，用于加载数据
import numpy as np  # 导入numpy库，并简写为np，用于处理数组

# 数据预处理
transform = transforms.Compose([  # 创建一个transform，用于对数据集进行预处理
    transforms.Resize((224, 224)),  # 将图片大小调整为224x224
    transforms.ToTensor(),  # 将图片转换为Tensor格式
])


# 加载数据集
def load_dataset(root_dir, transform=transform):  # 定义一个函数，用于加载数据集
    data = datasets.ImageFolder(root=root_dir, transform=transform)  # 使用ImageFolder加载数据集，并应用预处理
    return data  # 返回加载的数据集


# 训练KNN模型
def train_knn(train_loader):  # 定义一个函数，用于训练KNN模型
    features = []  # 初始化一个空列表，用于存储特征
    labels = []  # 初始化一个空列表，用于存储标签
    for images, labels_batch in train_loader:  # 遍历训练数据加载器
        features.extend(images.view(images.size(0), -1).numpy())  # 将图片数据展平并转换为numpy数组，然后添加到特征列表中
        labels.extend(labels_batch.numpy())  # 将标签转换为numpy数组，然后添加到标签列表中
    features = np.array(features)  # 将特征列表转换为numpy数组
    labels = np.array(labels)  # 将标签列表转换为numpy数组

    knn = KNeighborsClassifier(n_neighbors=5)  # 创建一个KNN分类器，设置邻居数为5
    start_time = time.time()  # 记录训练开始时间
    knn.fit(features, labels)  # 使用特征和标签训练KNN分类器
    end_time = time.time()  # 记录训练结束时间
    print(f"Training time: {end_time - start_time:.2f} seconds")  # 打印训练时间
    return knn  # 返回训练好的KNN分类器


# 测试KNN模型
def test_knn(model, test_loader):  # 定义一个函数，用于测试KNN模型
    features = []  # 初始化一个空列表，用于存储特征
    labels = []  # 初始化一个空列表，用于存储标签（实际上这个列表在后续没有被使用，可以省略）
    for images, labels_batch in test_loader:  # 遍历测试数据加载器
        features.extend(images.view(images.size(0), -1).numpy())  # 将图片数据展平并转换为numpy数组，然后添加到特征列表中
    features = np.array(features)  # 将特征列表转换为numpy数组

    start_time = time.time()  # 记录测试开始时间
    predictions = model.predict(features)  # 使用KNN分类器对特征进行预测
    end_time = time.time()  # 记录测试结束时间
    print(f"Testing time: {end_time - start_time:.2f} seconds")  # 打印测试时间

    labels = np.array([label for _, label in test_loader.dataset])  # 从测试数据加载器中获取所有标签，并转换为numpy数组
    accuracy = accuracy_score(labels, predictions)  # 计算准确率
    print(f"Accuracy: {accuracy:.2f}")  # 打印准确率


# 主函数
def main():  # 定义主函数
    train_dir = r"C:\Users\何Sir\Desktop\need\train01"  # 训练数据集目录
    test_dir = r"C:\Users\何Sir\Desktop\need\test01"  # 测试数据集目录

    train_data = load_dataset(train_dir)  # 加载训练数据集
    test_data = load_dataset(test_dir)  # 加载测试数据集

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)  # 创建训练数据加载器，设置批次大小为32，并启用打乱
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)  # 创建测试数据加载器，设置批次大小为32，不启用打乱

    model = train_knn(train_loader)  # 训练KNN模型
    test_knn(model, test_loader)  # 测试KNN模型


if __name__ == "__main__":  # 如果此脚本作为主程序运行
    main()  # 调用主函数