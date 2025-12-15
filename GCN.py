import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors
import argparse
import matplotlib.pyplot as plt

class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train GCN model on student data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    parser.add_argument('--dataset_type', type=str, choices=['3class', '2class'], default='3class',
                       help='Type of dataset: 3class for 3 GPA categories, 2class for 2 GPA categories')
    parser.add_argument('--k', type=int, default=10, help='Number of neighbors for KNN graph')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size ratio')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for splitting')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 读取数据并根据数据集类型处理
    student_feature = pd.read_csv(args.data_path)
    
    if args.dataset_type == '3class':
        # 三分类数据集处理
        features = student_feature.iloc[:, 1:].astype(np.float32).values
        labels = student_feature.iloc[:, 0].values
    else:
        # 二分类数据集处理
        features = student_feature.drop(columns=["label", "BH"]).astype(np.float32).values
        labels = student_feature["label"].values
    
    # 转换为Tensor并移动到设备
    features = torch.tensor(features, dtype=torch.float32).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    
    # 构建KNN图
    knn = NearestNeighbors(n_neighbors=args.k, metric='euclidean')
    knn.fit(features.cpu().numpy())
    distances, indices = knn.kneighbors(features.cpu().numpy())
    
    # 转换为PyG的edge_index格式
    edge_index = []
    for i in range(features.shape[0]):
        for j in indices[i]:
            edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    print(f"Edge index shape: {edge_index.shape}")
    
    # 划分训练集和测试集索引
    train_index, test_index = train_test_split(
        np.arange(len(features)), 
        test_size=args.test_size, 
        random_state=args.random_state
    )
    train_index = torch.tensor(train_index, dtype=torch.long).to(device)
    test_index = torch.tensor(test_index, dtype=torch.long).to(device)
    
    # 初始化模型和优化器
    input_dim = features.shape[1]
    output_dim = len(torch.unique(labels))
    model = GCNModel(
        in_channels=input_dim, 
        hidden_channels=args.hidden_dim, 
        out_channels=output_dim
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练模型
    train_losses = []
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        train_output = model(features, edge_index)
        train_loss = nn.CrossEntropyLoss()(train_output[train_index], labels[train_index])
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss.item():.4f}")
    
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # 在测试集上评估模型
    model.eval()
    with torch.no_grad():
        test_output = model(features, edge_index)
        predicted_labels = torch.argmax(test_output[test_index], dim=1)
        test_accuracy = accuracy_score(labels[test_index].cpu(), predicted_labels.cpu())
        
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        classification_rep = classification_report(
            labels[test_index].cpu(), 
            predicted_labels.cpu(), 
            digits=4
        )
        print("Classification Report:\n", classification_rep)
    
    # 统计预测结果分布
    if args.dataset_type == '3class':
        num_good = sum(predicted_labels.cpu() == 2)
        num_medium = sum(predicted_labels.cpu() == 1)
        num_poor = sum(predicted_labels.cpu() == 0)
        
        print("\nPredicted Distribution:")
        print("Number of students with good GPA:", num_good)
        print("Number of students with medium GPA:", num_medium)
        print("Number of students with poor GPA:", num_poor)
        
        # 绘制三分类条形图
        categories = ['Poor', 'Medium', 'Good']
        counts = [num_poor, num_medium, num_good]
    else:
        num_good = sum(predicted_labels.cpu() == 1)
        num_poor = sum(predicted_labels.cpu() == 0)
        
        print("\nPredicted Distribution:")
        print("Number of students with good GPA:", num_good)
        print("Number of students with poor GPA:", num_poor)
        
        # 绘制二分类条形图
        categories = ['Poor', 'Good']
        counts = [num_poor, num_good]
    
    # 绘制预测分布条形图
    plt.figure(figsize=(8, 6))
    plt.bar(categories, counts)
    plt.xlabel('GPA Categories')
    plt.ylabel('Number of Students')
    plt.title('Distribution of Predicted GPA Categories')
    plt.show()

if __name__ == "__main__":
    main()