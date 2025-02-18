import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def load_data_and_labels(directory):
    texts = []
    labels = []
    for label_id, subdir in enumerate(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            for filename in os.listdir(os.path.join(directory, subdir)):
                with open(os.path.join(directory, subdir, filename), 'r', encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read())
                labels.append(label_id)
    return texts, labels


# 加载数据
train_texts, train_labels = load_data_and_labels(r"Y:\py-torch\甲骨文切割\train_拓片")
test_texts, test_labels = load_data_and_labels(r"Y:\py-torch\甲骨文切割\test_拓片")

# 向量化
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# 决策树分类器
clf = DecisionTreeClassifier()

# 记录训练时间
start_time = time.time()
clf.fit(X_train, train_labels)  # 直接使用稀疏矩阵进行训练
train_time = time.time() - start_time

# 记录测试时间
start_time = time.time()
predicted = clf.predict(X_test)  # 直接使用稀疏矩阵进行预测
test_time = time.time() - start_time

# 准确率
accuracy = accuracy_score(test_labels, predicted)

print(f"Training time: {train_time:.2f} seconds")
print(f"Testing time: {test_time:.2f} seconds")
print(f"Accuracy: {accuracy:.4f}")