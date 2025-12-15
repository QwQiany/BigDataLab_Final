import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot as plt
import warnings
import argparse
import torch
import os

warnings.filterwarnings("ignore", message="Device used : cpu")
warnings.filterwarnings("ignore", message="No early stopping will be performed, last training weights will be used")
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    parser = argparse.ArgumentParser(description="TabNet Parameters", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # 数据集类型参数
    parser.add_argument('--dataset_type', type=str, choices=['2class', '3class'], required=True,
                       help='Type of dataset: 2class for binary classification, 3class for 3-class classification')
    
    # TabNet 参数
    parser.add_argument('--n_d', type=int, default=8, help='Dimension of the feature representation (n_d)')
    parser.add_argument('--n_a', type=int, default=8, help='Dimension of the attention embedding (n_a)')
    parser.add_argument('--n_steps', type=int, default=3, help='Number of steps in TabNet')
    parser.add_argument('--gamma', type=float, default=1.3, help='Gamma parameter for feature reusage in TabNet')
    parser.add_argument('--cat_emb_dim', type=int, default=1, help='Dimension of the embeddings for categorical features')
    parser.add_argument('--lambda_sparse', type=float, default=1e-3, help='Sparse regularization lambda')
    parser.add_argument('--momentum', type=float, default=0.02, help='Momentum for the batch normalization')
    parser.add_argument('--max_epochs', type=int, default=200, help='Maximum number of epochs for training')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--virtual_batch_size', type=int, default=128, help='Virtual batch size for large datasets')
    parser.add_argument('--learning_rate', type=float, default=2e-2, help='Learning rate for optimizer')
    parser.add_argument('--mask_type', type=str, default='entmax', choices=['entmax', 'sparsemax'], help='Masking function for feature selection')
    parser.add_argument('--optimizer_fn', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer function')
    parser.add_argument('--clip_value', type=float, default=None, help='Clip value for gradient clipping')

    # Parse arguments
    args = parser.parse_args()

    # 根据数据集类型设置文件路径
    if args.dataset_type == '2class':
        csv_file_labels = 'data/student_2_labels.csv'
        csv_file_features = 'data/features_2class.csv'
        train_idx = 'data/train_idx_2class.csv'
        test_idx = 'data/test_idx_2class.csv'
        dataset_name = "2-Class"
    else:  # 3class
        csv_file_labels = 'data/student_features_min_oversampler.csv'
        csv_file_features = 'data/features.csv'
        train_idx = 'data/train_idx.csv'
        test_idx = 'data/test_idx.csv'
        dataset_name = "3-Class"

    # 检查文件是否存在
    for file_path in [csv_file_labels, csv_file_features, train_idx, test_idx]:
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 不存在!")
            print("请先运行文件创建脚本生成所需的文件。")
            return

    # 使用 pandas 读取 CSV 文件
    data = pd.read_csv(csv_file_labels)
    
    # 获取标签数据
    if args.dataset_type == '2class':
        labels = data["label"].to_numpy().reshape((-1,))
    else:  # 3class
        labels = data.iloc[:, 0].to_numpy().reshape((-1,))
    
    features = pd.read_csv(csv_file_features, header=None)
    idx_train = pd.read_csv(train_idx, header=None)
    idx_train = idx_train.iloc[:, 0].tolist()
    idx_train = [x for x in idx_train]
    idx_test = pd.read_csv(test_idx, header=None)
    idx_test = idx_test.iloc[:, 0].tolist()
    idx_test = [x for x in idx_test]

    # 提取训练集和测试集数据
    train_data = features.iloc[idx_train].values.astype(np.float32)
    train_labels = labels[idx_train].astype(np.int64)
    test_data = features.iloc[idx_test].values.astype(np.float32)
    test_labels = labels[idx_test].astype(np.int64)

    print(f"使用 {dataset_name} 数据集")
    print(f"训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")
    print(f"特征维度: {train_data.shape[1]}")

    # 创建 TabNet 分类器
    tabnet = TabNetClassifier(
        device_name='cuda' if torch.cuda.is_available() else 'cpu',
        n_d=args.n_d,
        n_a=args.n_a,
        n_steps=args.n_steps,
        gamma=args.gamma,
        lambda_sparse=args.lambda_sparse,
        cat_emb_dim=args.cat_emb_dim,
        optimizer_params=dict(lr=args.learning_rate),
        mask_type=args.mask_type,
        clip_value=args.clip_value
    )

    # 使用训练数据进行训练
    print("训练 TabNet 模型...")
    tabnet.fit(
        train_data,
        train_labels,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        virtual_batch_size=args.virtual_batch_size,
    )

    # 在测试集上进行预测
    y_pred = tabnet.predict(test_data)

    # 根据数据集类型统计结果
    if args.dataset_type == '2class':
        # 二分类统计
        num_good = sum(y_pred == 1)
        num_poor = sum(y_pred == 0)
        
        print("Number of students with good GPA:", num_good)
        print("Number of students with poor GPA:", num_poor)
        
        # 绘制二分类条形图
        categories = ['Poor', 'Good']
        counts = [num_poor, num_good]
    else:
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

    # 计算准确率
    accuracy = accuracy_score(test_labels, y_pred)
    classification = classification_report(test_labels, y_pred, digits=4)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification)

    # 绘制条形图
    plt.bar(categories, counts)
    plt.xlabel('GPA Categories')
    plt.ylabel('Number of Students')
    plt.title(f'Distribution of GPA Categories (TabNet - {dataset_name})')
    
    # 在条形上显示数值
    for i, count in enumerate(counts):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
    
    plt.show()

if __name__ == "__main__":
    main()