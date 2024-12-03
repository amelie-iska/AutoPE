# train_model.py
# 有错误 这个不能处理连续值
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
import torch.nn as nn
import esm
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score, log_loss, brier_score_loss
)
from sklearn.metrics import roc_curve, precision_recall_curve
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import json

from model import ProteinClassifier

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def find_best_threshold(y_true, y_scores):
    """
    寻找最佳阈值，使 F1 分数最大化
    """
    thresholds = np.linspace(0, 1, num=100)
    best_threshold = 0.0
    best_f1 = 0.0
    for threshold in thresholds:
        y_pred = y_scores 
        # print(y_pred,y_true)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, batch_converter):
        self.data = list(zip(sequences, labels))
        self.batch_converter = batch_converter

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, label = self.data[idx]
        data = [("protein", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        return batch_tokens.squeeze(0), torch.tensor(label, dtype=torch.float32)

# def load_data(config):
#     """
#     加载并预处理数据
#     """
#     # 读取数据
#     df = pd.read_excel(config['save_path'])

#     # 分离甜和不甜的数据
#     sweet_df = df[df['Sweetness'] == '甜']
#     not_sweet_df = df[df['Sweetness'] == '不甜']

#     print(f"原始甜的样本数量：{len(sweet_df)}, 原始不甜的样本数量：{len(not_sweet_df)}")

#     # 上采样甜的样本
#     sweet_df_upsampled = sweet_df.sample(n=len(not_sweet_df), replace=True, random_state=42)

#     # 合并并打乱数据集
#     df_balanced = pd.concat([sweet_df_upsampled, not_sweet_df]).sample(frac=1, random_state=42).reset_index(drop=True)

#     # 提取序列和标签
#     sequences = df_balanced['Mutations sequence'].tolist()
#     labels = df_balanced['Sweetness'].apply(lambda x: 1 if x == '甜' else 0).tolist()

#     # 检查平衡后的标签分布
#     label_counts = pd.Series(labels).value_counts()
#     print("平衡后的标签分布：")
#     print(label_counts)

#     return sequences, labels

def load_data(config):
    df = pd.read_excel(config['save_path'])
    
    # 提取序列和连续数值标签
    sequences = df['Mutations sequence'].tolist()
    # 将 'Sweetness' 列转换为数值类型的标签
    # 假设 'Sweetness' 列已经是数值类型；如果不是，需要进行转换
    labels = df['value'].astype(float).tolist()
    
    return sequences, labels

def train_model(config):
    # 加载预训练模型
    n_gpus = torch.cuda.device_count()
    devices = [torch.device(f'cuda:{i}') for i in range(n_gpus)]
    print(f"Using devices: {devices}")

    model_path = config['model_path']
    pretrained_model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    batch_converter = alphabet.get_batch_converter()
    pretrained_model = pretrained_model.to(device)
    
    pretrained_model.eval()  # 设置为评估模式

    # 创建模型实例
    model = ProteinClassifier(
        pretrained_model=pretrained_model,
        # devices=devices,
        embedding_dim=1280,
        hidden_size=config.get('hidden_size', 256),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.1)
    )

    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs for training.")
    #     model = nn.DataParallel(model)
    # else:
    #     print("Using a single GPU or CPU for training.")

    # 冻结预训练模型的参数
    for param in pretrained_model.parameters():
        param.requires_grad = False

    # 加载数据
    sequences, labels = load_data(config)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.4, random_state=42
    )

    # 创建数据集和数据加载器
    train_dataset = ProteinDataset(X_train, y_train, batch_converter)
    val_dataset = ProteinDataset(X_val, y_val, batch_converter)

    batch_size = config.get('batch_size', 16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get('lr', 1e-4),
        weight_decay=config.get('weight_decay', 1e-5)
    )

    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get('scheduler_step_size', 10),
        gamma=config.get('scheduler_gamma', 0.1)
    )

    num_epochs = config.get('num_epochs', 50)
    best_f1 = 0.0
    best_epoch = 0

    for epoch in tqdm(range(num_epochs)):
        # 训练阶段
        model.train().to(device)
        total_loss = 0.0
        for batch_tokens, labels_batch in train_loader:
            batch_tokens = batch_tokens.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(batch_tokens, mask=batch_tokens != alphabet.padding_idx)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels_batch.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for batch_tokens, labels_batch in val_loader:
                batch_tokens = batch_tokens.to(device)
                labels_batch = labels_batch.to(device)

                outputs = model(batch_tokens, mask=batch_tokens != alphabet.padding_idx)
                loss = criterion(outputs, labels_batch)

                val_loss += loss.item() * labels_batch.size(0)
                all_labels.extend(labels_batch.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)

        # 计算评估指标
        all_labels = np.array(all_labels)
        all_outputs = np.array(all_outputs)
        all_probs = torch.sigmoid(torch.tensor(all_outputs)).numpy()

        # best_threshold = find_best_threshold(all_labels, all_probs)
        preds = all_probs

        accuracy = accuracy_score(all_labels, preds)
        precision = precision_score(all_labels, preds, zero_division=0)
        recall = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)
        roc_auc = roc_auc_score(all_labels, all_probs)
        pr_auc = average_precision_score(all_labels, all_probs)
        tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        mcc = matthews_corrcoef(all_labels, preds)
        kappa = cohen_kappa_score(all_labels, preds)
        logloss = log_loss(all_labels, all_probs, labels=[0, 1])
        brier = brier_score_loss(all_labels, all_probs)
        srcc, _ = spearmanr(all_labels, all_probs)

        # 更新学习率
        scheduler.step()

        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_model.pth')

        # 打印结果
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f} | PR AUC: {pr_auc:.4f} | SRCC: {srcc:.4f}")
        print(f"Specificity: {specificity:.4f} | MCC: {mcc:.4f} | Cohen's Kappa: {kappa:.4f}")
        print(f"Log Loss: {logloss:.4f} | Brier Score: {brier:.4f}")
        # print(f"Best Threshold: {best_threshold:.4f}")
        print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print("-" * 50)

        # 保存结果到文件
        results = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'srcc': srcc,
            'mcc': mcc,
            'kappa': kappa,
            'log_loss': logloss,
            'brier_score': brier,
            # 'best_threshold': best_threshold,
            'confusion_matrix': {'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn)}
        }
        if (epoch+1)%10 == 0:

            with open(f'results_epoch_{epoch+1}.json', 'w') as f:
                json.dump(results, f, indent=4)

            # 绘制并保存 ROC 曲线
        print('label:', all_labels)
        print('probs:',all_probs)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        print('fpr and tpr',fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (Epoch {epoch+1})')
        plt.legend(loc="lower right")
        plt.savefig(f'./roc_fig/{num_epochs}/roc_curve_epoch_{epoch+1}.png')
        plt.close()

        # 绘制并保存 PR 曲线
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
        plt.figure()
        plt.plot(recall_curve, precision_curve, color='red', lw=2, label=f'PR curve (area = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (Epoch {epoch+1})')
        plt.legend(loc="lower left")
        plt.savefig(f'./pr_fig/{num_epochs}/pr_curve_epoch_{epoch+1}.png')
        plt.close()

    print(f"Training complete. Best F1 Score: {best_f1:.4f} at epoch {best_epoch}")

def main():
    config = {
        'save_path': '/home/netzone22/mydisk/yungeng/ray/突变库数据1.03_with_mutated_sequences.xlsx',  # 请修改为您的数据路径
        'model_path': '/home/netzone22/mydisk/yungeng/esm_models/esm2_t33_650M_UR50D.pt',        # 请修改为您的模型路径
        'lr': 1e-4,
        'dropout': 0.3,
        'num_epochs': 30,
        'batch_size': 4,
        'hidden_size': 256,
        'num_layers': 2,
        'weight_decay': 1e-5,
        'scheduler_step_size': 10,
        'scheduler_gamma': 0.1
    }

    train_model(config)

if __name__ == "__main__":
    main()