# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import argparse
from contextlib import contextmanager
import sys
import os

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
    
    parser = argparse.ArgumentParser(
        description='Train SVM classifier on student data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # 模型参数
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--kernel', type=str, default='rbf', choices=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])
    parser.add_argument('--degree', type=int, default=3)
    parser.add_argument('--gamma', type=str, default='scale', choices=['scale', 'auto'])
    parser.add_argument('--coef0', type=float, default=0.0)
    parser.add_argument('--shrinking', type=bool, default=True)
    parser.add_argument('--probability', type=bool, default=False)
    parser.add_argument('--tol', type=float, default=1e-3)
    parser.add_argument('--cache_size', type=float, default=200)
    parser.add_argument('--class_weight', type=lambda x: eval(x) if x else None, default=None)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--max_iter', type=int, default=-1)
    parser.add_argument('--decision_function_shape', type=str, default='ovr')
    parser.add_argument('--break_ties', type=bool, default=False)
    
    # 数据参数
    # parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    # parser.add_argument('--dataset_type', type=str, choices=['3class', '2class'], default='3class',
    #                    help='Type of dataset: 3class for 3 GPA categories, 2class for 2 GPA categories')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size ratio')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for splitting')
    
    args = parser.parse_args()

    with suppress_stderr():
        # 读取数据
        data = pd.read_csv('processed_happiness_data.csv')
        
        # 处理特征和标签
        # 第一列是国家名字(忽略)，第二列是标签，剩下的列都是归一化好的数据
        y = data.iloc[:, 1]
        X = data.iloc[:, 2:]
        
        # 数据拆分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values, 
            test_size=args.test_size, 
            random_state=args.random_state
        )
        
        # 创建SVM分类器
        svm_classifier = SVC(
            C=args.C,
            kernel=args.kernel,
            degree=args.degree,
            gamma=args.gamma,
            coef0=args.coef0,
            shrinking=args.shrinking,
            probability=args.probability,
            tol=args.tol,
            cache_size=args.cache_size,
            class_weight=args.class_weight,
            verbose=args.verbose,
            max_iter=args.max_iter,
            decision_function_shape=args.decision_function_shape,
            break_ties=args.break_ties
        )
        
        # 训练SVM模型
        svm_classifier.fit(X_train, y_train)
        
        # 在测试集上进行预测
        y_pred = svm_classifier.predict(X_test)
        
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
        
        # 计算准确率和其他评估指标
        accuracy = accuracy_score(y_test, y_pred)
        classification = classification_report(y_test, y_pred, digits=4)
        
        print("SVM Accuracy:", accuracy)
        print("SVM Classification Report:\n", classification)
        
        # 绘制条形图
        plt.bar(categories, counts)
        plt.xlabel('GPA Categories')
        plt.ylabel('Number of Students')
        plt.title('Distribution of Predicted GPA Categories (SVM)')
        
        # 在条形上显示数值
        for i, count in enumerate(counts):
            plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        plt.show()

if __name__ == "__main__":
    main()