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
        parser.add_argument('--base_estimator', type=lambda x: eval(x) if x else None, default=None)
        parser.add_argument('--n_estimators', type=int, default=50)
        parser.add_argument('--learning_rate', type=float, default=1.0)
        parser.add_argument('--algorithm', type=str, choices=['SAMME', 'SAMME.R'], default='SAMME.R')
        parser.add_argument('--random_state', type=lambda x: int(x) if x else None, default=None)
        
        # 数据参数
        parser.add_argument('--data_path', type=str, required=True, 
                          help='Path to the data file')
        parser.add_argument('--dataset_type', type=str, choices=['3class', '2class'], default='3class',
                          help='Type of dataset: 3class for 3 GPA categories, 2class for 2 GPA categories')
        parser.add_argument('--test_size', type=float, default=0.2, 
                          help='Test set size ratio')
        
        args = parser.parse_args()
        
        # 读取数据
        data = pd.read_csv(args.data_path)
        
        # 根据数据集类型处理特征和标签
        if args.dataset_type == '3class':
            # 三分类数据集处理
            X = data.iloc[:, 1:]  # 特征
            y = data.iloc[:, 0]   # 标签
        else:
            # 二分类数据集处理
            y = data.iloc[:, 0]   # 标签
            X = data.drop(columns=["label", "BH"])  # 特征，去掉标签列和BH列
        
        # 数据拆分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42)
        
        # 创建AdaBoost分类器
        adaboost = AdaBoostClassifier(**{k: v for k, v in vars(args).items() 
                                       if k in ['base_estimator', 'n_estimators', 
                                               'learning_rate', 'algorithm', 'random_state']})
        
        # 在训练集上训练AdaBoost模型
        adaboost.fit(X_train, y_train)
        
        # 在测试集上进行预测
        y_pred = adaboost.predict(X_test)
        
        # 根据数据集类型统计结果
        if args.dataset_type == '3class':
            # 三分类统计
            num_good = sum(y_pred == 2)
            num_medium = sum(y_pred == 1)
            num_poor = sum(y_pred == 0)
            
            print("Number of students with good GPA:", num_good)
            print("Number of students with medium GPA:", num_medium)
            print("Number of students with poor GPA:", num_poor)
            
            # 绘制三分类条形图
            categories = ['Poor', 'Medium', 'Good']
            counts = [num_poor, num_medium, num_good]
        else:
            # 二分类统计
            num_good = sum(y_pred == 1)
            num_poor = sum(y_pred == 0)
            
            print("Number of students with good GPA:", num_good)
            print("Number of students with poor GPA:", num_poor)
            
            # 绘制二分类条形图
            categories = ['Poor', 'Good']
            counts = [num_poor, num_good]
        
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