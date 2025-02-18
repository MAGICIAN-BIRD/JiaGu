import os
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from PIL import Image


def load_image(img_path):
    """加载并预处理图像"""
    try:
        img = Image.open(img_path)
        img = img.resize((64, 64))  # 将所有图像调整为64x64
        img = np.array(img)
        # 如果图像是彩色的，转换为灰度图
        if img.ndim == 3:
            img = img.mean(axis=2).astype(np.uint8)
        return img.flatten()  # 将图像展平为一维数组
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None


def get_label(img_path):
    """从文件路径中提取标签"""
    label = img_path.split('_')[0]
    if label.isdigit():
        return int(label)
    else:
        # 如果标签不是数字，使用标签映射字典
        return label_mapping.get(label, None)


label_mapping = {'class_1': 0, 'class_2': 1, 'class_3': 2}


def load_dataset(path):
    images = []
    labels = []
    if not os.path.exists(path):
        print(f"Path {path} does not exist.")
        return None, None
    for img_path in os.listdir(path):
        img = load_image(os.path.join(path, img_path))
        if img is None:
            continue
        label = get_label(img_path)
        if label is None:
            print(f"Warning: No label found for {img_path}")
            continue
        images.append(img)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


# Load training and testing datasets
train_images, train_labels = load_dataset(r"Y:\py-torch\甲骨文切割\test_拓片")
test_images, test_labels = load_dataset(r"Y:\py-torch\甲骨文切割\train_拓片")

# Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
start_time = time.time()
clf.fit(train_images, train_labels)
print(f"Training time: {time.time() - start_time} seconds")

# Predict on test set
start_time = time.time()
y_pred = clf.predict(test_images)
print(f"Testing time: {time.time() - start_time} seconds")

# Evaluate the classifier
print("Accuracy:", accuracy_score(test_labels, y_pred))
print("Classification Report:\n", classification_report(test_labels, y_pred))