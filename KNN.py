import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

    parser = argparse.ArgumentParser(
        description='Train KNN classifier on student data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 模型参数
    parser.add_argument('--n_neighbors', type=int, default=3)
    parser.add_argument('--weights', type=str, default='uniform')
    parser.add_argument('--algorithm', type=str, default='auto')
    parser.add_argument('--leaf_size', type=int, default=30)
    parser.add_argument('--p', type=int, default=2)
    parser.add_argument('--metric', type=str, default='minkowski')
    parser.add_argument('--metric_params', type=lambda x: eval(x) if x else None, default=None)
    parser.add_argument('--n_jobs', type=int, default=None)

    # 数据参数
    parser.add_argument(
        '--data_path',
        type=str,
        default='processed_happiness_data.csv',
        help='Path to the data file (default: processed_happiness_data.csv in current dir)')
    parser.add_argument('--dataset_type', type=str, choices=['3class', '2class'], default='3class',
                        help='Type of dataset: 3class for 3 GPA categories, 2class for 2 GPA categories')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size ratio')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for splitting')

    args = parser.parse_args()

    # Suppress errors during the critical block, including plt.show()
    with suppress_stderr():
        # 读取数据
        data = pd.read_csv(args.data_path)

        # ========== 重要修改开始 ==========
        # 根据数据集类型处理特征和标签
        if args.dataset_type == '3class':
            # 三分类数据集处理 - 修改标签列为第二列（label列）
            y = data.iloc[:, 1]  # 标签（第二列）
            # 特征：去掉第一列（Country name）和第二列（label）
            X = data.drop(data.columns[[0, 1]], axis=1)
        else:
            # 二分类数据集处理 - 修改标签列为第二列（label列）
            y = data.iloc[:, 1]  # 标签（第二列）
            # 特征：去掉第一列（Country name）和第二列（label）
            X = data.drop(data.columns[[0, 1]], axis=1)
        # ========== 重要修改结束 ==========

        # 数据拆分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values,
            test_size=args.test_size,
            random_state=args.random_state
        )

        # 创建KNN分类器
        knn = KNeighborsClassifier(
            n_neighbors=args.n_neighbors,
            weights=args.weights,
            algorithm=args.algorithm,
            leaf_size=args.leaf_size,
            p=args.p,
            metric=args.metric,
            metric_params=args.metric_params,
            n_jobs=args.n_jobs
        )

        # 在训练集上训练KNN模型
        knn.fit(X_train, y_train)

        # 在测试集上进行预测
        y_pred = knn.predict(X_test)

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
        plt.title('Distribution of Predicted GPA Categories')

        # 在条形上显示数值
        for i, count in enumerate(counts):
            plt.text(i, count + 0.1, str(count), ha='center', va='bottom')

        plt.show()


if __name__ == "__main__":
    main()