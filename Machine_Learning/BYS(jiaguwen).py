import os
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def preprocess_text(text):
    # 简单的文本预处理：去除标点符号和换行符，转为小写
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text


def load_data_and_labels(directory):
    texts = []
    labels = []
    for label_id, subdir in enumerate(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            for filename in os.listdir(os.path.join(directory, subdir)):
                with open(os.path.join(directory, subdir, filename), 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    text = preprocess_text(text)
                    texts.append(text)
                labels.append(label_id)
    return texts, labels


# 加载数据
train_texts, train_labels = load_data_and_labels('train_拓片')
test_texts, test_labels = load_data_and_labels('test_拓片')

# 向量化
vectorizer = TfidfVectorizer(max_df=0.7, min_df=3, max_features=15000, ngram_range=(1, 3))
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# 朴素贝叶斯分类器
clf = MultinomialNB()

# 记录训练时间
start_time = time.time()
clf.fit(X_train, train_labels)
train_time = time.time() - start_time

# 记录测试时间
start_time = time.time()
predicted = clf.predict(X_test)
test_time = time.time() - start_time

# 准确率
accuracy = accuracy_score(test_labels, predicted)

print(f"Training time: {train_time:.2f} seconds")
print(f"Testing time: {test_time:.2f} seconds")
print(f"Accuracy: {accuracy:.4f}")