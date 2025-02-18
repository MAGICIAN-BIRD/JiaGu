import os  # 导入os模块，用于操作系统相关功能，如文件路径操作
import time  # 导入time模块，用于测量程序执行时间
from sklearn.decomposition import PCA  # 从sklearn库导入PCA类，用于降维
from sklearn.neighbors import KNeighborsClassifier  # 从sklearn库导入KNeighborsClassifier类，用于KNN分类
from sklearn.metrics import accuracy_score  # 从sklearn库导入accuracy_score函数，用于计算准确率
from torchvision import datasets, transforms  # 从torchvision库导入datasets和transforms模块，用于加载和预处理数据集
from torch.utils.data import DataLoader  # 从torch.utils.data模块导入DataLoader类，用于加载数据
import numpy as np  # 导入numpy库，用于处理多维数组

# 数据预处理
transform = transforms.Compose([  # 使用Compose组合多个预处理步骤
    transforms.Resize((224, 224)),  # 将图片大小调整为224x224
    transforms.ToTensor(),  # 将图片转换为Tensor格式
])


# 加载数据集
def load_dataset(root_dir, transform=transform):  # 定义一个函数，用于加载数据集
    data = datasets.ImageFolder(root=root_dir, transform=transform)  # 使用ImageFolder加载数据集，并应用预处理
    return data  # 返回加载的数据集


# 训练KNN模型
def train_knn(train_features, train_labels, n_components=50):  # 定义一个函数，用于训练KNN模型
    pca = PCA(n_components=n_components)  # 创建PCA对象，设置降维后的特征数量为50
    train_features_pca = pca.fit_transform(train_features)  # 使用PCA对数据进行降维
    knn = KNeighborsClassifier(n_neighbors=5)  # 创建KNN模型，设置邻居数为5
    start_time = time.time()  # 记录训练开始时间
    knn.fit(train_features_pca, train_labels)  # 训练KNN模型
    end_time = time.time()  # 记录训练结束时间
    print(f"Training time: {end_time - start_time:.2f} seconds")  # 打印训练时间
    return knn, pca  # 返回训练好的KNN模型和PCA对象


# 测试KNN模型
def test_knn(model, pca, test_features, test_labels):  # 定义一个函数，用于测试KNN模型
    test_features_pca = pca.transform(test_features)  # 使用PCA对测试数据进行降维
    start_time = time.time()  # 记录测试开始时间
    predictions = model.predict(test_features_pca)  # 使用KNN模型进行预测
    end_time = time.time()  # 记录测试结束时间
    print(f"Testing time: {end_time - start_time:.2f} seconds")  # 打印测试时间
    accuracy = accuracy_score(test_labels, predictions)  # 计算准确率
    print(f"Accuracy: {accuracy:.2f}")  # 打印准确率


# 主函数
def main():  # 定义主函数
    train_dir = r"C:\Users\何Sir\Desktop\need\train01"  # 训练集目录
    test_dir = r"C:\Users\何Sir\Desktop\need\test01"  # 测试集目录
    train_data = load_dataset(train_dir)  # 加载训练集
    test_data = load_dataset(test_dir)  # 加载测试集
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)  # 创建训练数据加载器
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)  # 创建测试数据加载器

    # 提取特征和标签
    train_features = []  # 初始化训练特征列表
    train_labels = []  # 初始化训练标签列表
    for images, labels in train_loader:  # 遍历训练数据加载器
        train_features.extend(images.view(images.size(0), -1).numpy())  # 提取特征并添加到列表
        train_labels.extend(labels.numpy())  # 提取标签并添加到列表
    train_features = np.array(train_features)  # 将训练特征列表转换为numpy数组
    train_labels = np.array(train_labels)  # 将训练标签列表转换为numpy数组

    test_features = []  # 初始化测试特征列表
    test_labels = []  # 初始化测试标签列表
    for images, labels in test_loader:  # 遍历测试数据加载器
        test_features.extend(images.view(images.size(0), -1).numpy())  # 提取特征并添加到列表
        test_labels.extend(labels.numpy())  # 提取标签并添加到列表
    test_features = np.array(test_features)  # 将测试特征列表转换为numpy数组
    test_labels = np.array(test_labels)  # 将测试标签列表转换为numpy数组

    # 训练模型
    model, pca = train_knn(train_features, train_labels, n_components=50)  # 训练KNN模型

    # 测试模型
    test_knn(model, pca, test_features, test_labels)  # 测试KNN模型


if __name__ == "__main__":  # 如果直接运行此脚本
    main()  # 调用主函数