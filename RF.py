import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
        description='Train Random Forest classifier on student data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # 模型参数
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--criterion', type=str, default='gini', choices=['gini', 'entropy'])
    parser.add_argument('--max_depth', type=lambda x: int(x) if x != 'None' else None, default=None)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--min_samples_leaf', type=float, default=1)
    parser.add_argument('--min_weight_fraction_leaf', type=float, default=0.0)
    parser.add_argument('--max_features', type=lambda x: int(x) if x.isdigit() else x, choices=['sqrt', 'log2', None], default=None)
    parser.add_argument('--max_leaf_nodes', type=lambda x: int(x) if x != 'None' else None, default=None)
    parser.add_argument('--min_impurity_decrease', type=float, default=0.0)
    parser.add_argument('--bootstrap', type=bool, default=True)
    parser.add_argument('--oob_score', type=bool, default=False)
    parser.add_argument('--n_jobs', type=lambda x: int(x) if x else None, default=None)
    parser.add_argument('--random_state', type=lambda x: int(x) if x else None, default=None)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--warm_start', type=bool, default=False)
    parser.add_argument('--class_weight', type=lambda x: eval(x) if x else None, default=None)
    parser.add_argument('--ccp_alpha', type=float, default=0.0)
    parser.add_argument('--max_samples', type=lambda x: float(x) if x != 'None' else None, default=None)
    
    # 数据参数
    # parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    # parser.add_argument('--dataset_type', type=str, choices=['3class', '2class'], default='3class',
    #                    help='Type of dataset: 3class for 3 GPA categories, 2class for 2 GPA categories')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size ratio')
    parser.add_argument('--random_state_split', type=int, default=42, help='Random state for data splitting')
    
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
            X, y, 
            test_size=args.test_size, 
            random_state=args.random_state_split
        )
        
        # 创建随机森林分类器
        random_forest = RandomForestClassifier(
            n_estimators=args.n_estimators,
            criterion=args.criterion,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            min_weight_fraction_leaf=args.min_weight_fraction_leaf,
            max_features=args.max_features,
            max_leaf_nodes=args.max_leaf_nodes,
            min_impurity_decrease=args.min_impurity_decrease,
            bootstrap=args.bootstrap,
            oob_score=args.oob_score,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            verbose=args.verbose,
            warm_start=args.warm_start,
            class_weight=args.class_weight,
            ccp_alpha=args.ccp_alpha,
            max_samples=args.max_samples
        )
        
        # 在训练集上训练随机森林模型
        random_forest.fit(X_train, y_train)
        
        # 在测试集上进行预测
        y_pred = random_forest.predict(X_test)
        
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
        
        # 如果启用了OOB评分，则显示
        if args.oob_score:
            print(f"OOB Score: {random_forest.oob_score_:.4f}")
        
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