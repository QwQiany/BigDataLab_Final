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

import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
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
    
    # 数据集类型参数 (Retained for compatibility but logic adapted)
    parser.add_argument('--dataset_type', type=str, choices=['2class', '3class'], default='3class',
                       help='Type of dataset: 2class or 3class (Logic adapted for Happiness Data)')
    
    # TabNet 参数
    parser.add_argument('--n_d', type=int, default=8, help='Dimension of the feature representation (n_d)')
    parser.add_argument('--n_a', type=int, default=8, help='Dimension of the attention embedding (n_a)')
    parser.add_argument('--n_steps', type=int, default=3, help='Number of steps in TabNet')
    parser.add_argument('--gamma', type=float, default=1.3, help='Gamma parameter for feature reusage in TabNet')
    parser.add_argument('--cat_emb_dim', type=int, default=1, help='Dimension of the embeddings for categorical features')
    parser.add_argument('--lambda_sparse', type=float, default=1e-3, help='Sparse regularization lambda')
    parser.add_argument('--momentum', type=float, default=0.02, help='Momentum for the batch normalization')
    parser.add_argument('--max_epochs', type=int, default=500, help='Maximum number of epochs for training')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--virtual_batch_size', type=int, default=8, help='Virtual batch size for large datasets')
    parser.add_argument('--learning_rate', type=float, default=2e-2, help='Learning rate for optimizer')
    parser.add_argument('--mask_type', type=str, default='entmax', choices=['entmax', 'sparsemax'], help='Masking function for feature selection')
    parser.add_argument('--optimizer_fn', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer function')
    parser.add_argument('--clip_value', type=float, default=None, help='Clip value for gradient clipping')

    # Parse arguments
    args = parser.parse_args()

    # 读取数据 (Hardcoded for processed_happiness_data.csv)
    data_path = "processed_happiness_data.csv"
    print(f"Loading data from {data_path}...")
    
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} not found!")
        return

    # 使用 pandas 读取 CSV 文件
    data = pd.read_csv(data_path)
    
    # 数据预处理
    # 第一列是国家名字(忽略), 第二列是标签, 剩下的列是特征
    labels = data.iloc[:, 1].values
    features = data.iloc[:, 2:].values

    # 转换为 Numpy 数组并确保类型正确
    X = features.astype(np.float32)
    y = labels.astype(np.int64)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Dataset Shape: {data.shape}")
    print(f"Train Set Size: {len(X_train)}")
    print(f"Test Set Size: {len(X_test)}")
    print(f"Feature Dimension: {X_train.shape[1]}")

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
        clip_value=args.clip_value,
        verbose=10
    )

    # 使用训练数据进行训练
    print("Training TabNet model...")
    tabnet.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)], # Use test set for early stopping/monitoring
        eval_name=['test'],
        eval_metric=['accuracy'],
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        virtual_batch_size=args.virtual_batch_size,
    )

    # 在测试集上进行预测
    y_pred = tabnet.predict(X_test)

    # 统计预测结果分布
    unique_classes = np.unique(y)
    categories = [f'Class {c}' for c in unique_classes]
    counts = []
    
    print("\nPredicted Distribution:")
    for c in unique_classes:
        count = np.sum(y_pred == c)
        print(f"Number of samples in Class {c}: {count}")
        counts.append(count)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    classification = classification_report(y_test, y_pred, digits=4)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification)

    # 绘制条形图
    plt.figure(figsize=(8, 6))
    plt.bar(categories, counts)
    plt.xlabel('Happiness Categories')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Predicted Happiness Categories (TabNet)')
    
    # 在条形上显示数值
    for i, count in enumerate(counts):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
    
    # plt.show() # Prevent blocking if running headless

if __name__ == "__main__":
    main()