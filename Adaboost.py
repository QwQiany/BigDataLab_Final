import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

import argparse
from contextlib import contextmanager
import sys
import os
import subprocess

@contextmanager
def suppress_stderr():
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')  # 把 stderr 重定向到 /dev/null 来隐藏错误输出
    try:
        yield
    finally:
        sys.stderr = original_stderr

def main():
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

    with suppress_stderr():
        parser = argparse.ArgumentParser(
            description='Train AdaBoost classifier on student data',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
        # 模型参数
        parser.add_argument('--estimator', type=lambda x: eval(x) if x else None, default=None)
        parser.add_argument('--n_estimators', type=int, default=50)
        parser.add_argument('--learning_rate', type=float, default=1.0)
        parser.add_argument('--algorithm', type=str, choices=['SAMME'], default='SAMME')
        parser.add_argument('--random_state', type=lambda x: int(x) if x else None, default=None)
        
        # 数据参数
        # parser.add_argument('--data_path', type=str, required=True, 
        #                   help='Path to the data file')
        # parser.add_argument('--dataset_type', type=str, choices=['3class', '2class'], default='3class',
        #                   help='Type of dataset: 3class for 3 GPA categories, 2class for 2 GPA categories')
        parser.add_argument('--test_size', type=float, default=0.2, 
                          help='Test set size ratio')
        
        args = parser.parse_args()
        
        # 读取数据
        data = pd.read_csv('processed_happiness_data.csv')
        
        # 处理特征和标签
        # 第一列是国家名字(忽略)，第二列是标签，剩下的列都是归一化好的数据
        y = data.iloc[:, 1]
        X = data.iloc[:, 2:]
        
        # 数据拆分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42)
        
        # 创建AdaBoost分类器
        adaboost = AdaBoostClassifier(**{k: v for k, v in vars(args).items() 
                                       if k in ['estimator', 'n_estimators', 
                                               'learning_rate', 'algorithm', 'random_state']})
        
        # 在训练集上训练AdaBoost模型
        adaboost.fit(X_train, y_train)
        
        # 在测试集上进行预测
        y_pred = adaboost.predict(X_test)
        
        # 统计结果
        num_good = sum(y_pred == 2)
        num_medium = sum(y_pred == 1)
        num_poor = sum(y_pred == 0)
        
        print("Number of samples with good Happiness:", num_good)
        print("Number of samples with medium Happiness:", num_medium)
        print("Number of samples with poor Happiness:", num_poor)
        
        # 绘制三分类条形图
        categories = ['Poor', 'Medium', 'Good']
        counts = [num_poor, num_medium, num_good]
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        classification = classification_report(y_test, y_pred, digits=4)
        
        print("Accuracy:", accuracy)
        print("Classification Report:\n", classification)
        
        # 绘制条形图
        plt.bar(categories, counts)
        plt.xlabel('GPA Categories')
        plt.ylabel('Number of Students')
        plt.title('Distribution of GPA Categories')
        plt.show()

if __name__ == "__main__":
    main()