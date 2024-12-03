# train_model.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["RAY_TEMP_DIR"] = "/home/netzone22/mydisk"  # 修改为您的路径

import torch
import torch.nn as nn
import esm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
import ray
from ray.tune.schedulers import ASHAScheduler
from ray import tune,train
from model import Model

# 初始化 Ray
ray.init(ignore_reinit_error=True, _temp_dir="your_path")  # 修改为您的路径
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def find_best_threshold(y_true, y_pred_proba):
    best_threshold = 0.0
    best_f1 = 0.0
    thresholds = np.linspace(y_pred_proba.min(), y_pred_proba.max(), num=100)
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold

# 自定义数据集类
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, batch_converter):
        self.sequences = sequences
        self.labels = labels
        self.batch_converter = batch_converter

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        data = [("sequence", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        return batch_tokens.squeeze(0), label

def load_data(config):
    """
    加载并预处理数据
    """
    # 读取数据
    df = pd.read_excel(config['save_path'])

    # 分离甜和不甜的数据
    sweet_df = df[df['value'] > 0.5]
    not_sweet_df = df[df['value'] < 0.5]

    print(f"原始甜的样本数量：{len(sweet_df)}, 原始不甜的样本数量：{len(not_sweet_df)}")

    # 上采样甜的样本
    sweet_df_upsampled = sweet_df.sample(n=len(not_sweet_df), replace=True, random_state=42)

    # 合并并打乱数据集
    df_balanced = pd.concat([sweet_df_upsampled, not_sweet_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    # 提取序列和标签
    sequences = df_balanced['Mutations sequence'].tolist()
    labels = df_balanced['value'].apply(lambda x: 1 if x >0.5  else 0).tolist()

    # 检查平衡后的标签分布
    label_counts = pd.Series(labels).value_counts()
    print("平衡后的标签分布：")
    print(label_counts)

    return sequences, labels
    
def train_model(config, checkpoint_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = '/home/netzone22/mydisk/yungeng/esm_models/esm2_t33_650M_UR50D.pt'  # 修改为您的模型路径
    pretrained_model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    pretrained_model.eval()
    batch_converter = alphabet.get_batch_converter()
    pretrained_model = pretrained_model.to(device)

    model = Model(pretrained_model=pretrained_model, embedding_dim=1280, output_dim=1, self_attention_layers=True, repr_layers=33, dropout=float(config["dropout"]))
    model = model.to(device)
    model.train()

    sequences, labels = load_data(config)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )


    # 创建数据集
    train_dataset = SequenceDataset(X_train, y_train, batch_converter)
    val_dataset = SequenceDataset(X_val, y_val, batch_converter)

    # 定义损失函数、优化器和调度器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    batch_size = int(config.get("batch_size", 8))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    accumulation_steps = int(config.get("accumulation_steps", 1))  # 梯度累积步数

    # 初始化
    best_val_f1 = 0.0
    best_epoch = 0
    for epoch in range(int(config["num_epochs"])):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for i, (batch_tokens, labels_batch) in enumerate(train_loader):
            batch_tokens, labels_batch = batch_tokens.to(device), labels_batch.to(device)
            labels_batch = labels_batch.float()

            outputs, _ = model(batch_tokens, mask=batch_tokens != alphabet.padding_idx)
            loss = criterion(outputs, labels_batch)
            loss = loss / accumulation_steps  # 梯度累积

            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps  # 累积损失

        # 验证模型
        model.eval()
        val_loss = 0.0
        all_y_true = []
        all_y_pred_proba = []
        with torch.no_grad():
            for batch_tokens, labels_batch in val_loader:
                batch_tokens, labels_batch = batch_tokens.to(device), labels_batch.to(device)
                labels_batch = labels_batch.float()

                outputs, _ = model(batch_tokens, mask=batch_tokens != alphabet.padding_idx)
                loss = criterion(outputs, labels_batch)

                val_loss += loss.item()

                outputs_np = torch.sigmoid(outputs).cpu().numpy()
                all_y_pred_proba.extend(outputs_np)
                all_y_true.extend(labels_batch.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        scheduler.step(val_loss)

        # 找到最佳阈值
        all_y_pred_proba = np.array(all_y_pred_proba)
        best_threshold = find_best_threshold(all_y_true, all_y_pred_proba)
        all_y_pred = (all_y_pred_proba >= best_threshold).astype(int)

        accuracy = accuracy_score(all_y_true, all_y_pred)
        precision = precision_score(all_y_true, all_y_pred, zero_division=0)
        recall = recall_score(all_y_true, all_y_pred, zero_division=0)
        f1 = f1_score(all_y_true, all_y_pred, zero_division=0)

        print(f"Epoch [{epoch+1}/{int(config['num_epochs'])}]")
        print(f"Training Loss: {total_loss/len(train_loader):.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Best Threshold: {best_threshold:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Predictions distribution: {np.bincount(all_y_pred)}")
        print(f"True labels distribution: {np.bincount(np.array(all_y_true, dtype=int))}")

        # 打印混淆矩阵
        cm = confusion_matrix(all_y_true, all_y_pred)
        print('Confusion Matrix:')
        print(cm)
        print("-" * 50)

        # 保存最佳模型
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_epoch = epoch + 1
            if checkpoint_dir is not None:
                checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            else:
                checkpoint_dir = './'
                checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()},
                       checkpoint_path)

        # 每个 epoch 结束后报告
        # tune.report(loss=val_loss, accuracy=accuracy, precision=precision, recall=recall, f1=f1, best_threshold=best_threshold)
        train.report({'loss': val_loss, 'accuracy':accuracy, 'precision':precision, 'recall':recall, 'f1':f1, 'best_threshold':best_threshold})

    print(f"Best F1 Score: {best_val_f1:.4f} at epoch {best_epoch}")

def main_sweet():
    config = {
        'save_path': '/home/netzone22/mydisk/yungeng/ray/突变库数据1.03_with_mutated_sequences.xlsx',  # 修改为您的数据路径
        'cpu_per_trial': '4',
        'gpus_per_trial': '1',
        'num_samples': 50,
        'lr': tune.loguniform(1e-6, 1e-3),
        'dropout': tune.uniform(0.01, 0.2),
        'num_epochs': 100,
        'batch_size': tune.choice([8, 16]),
        'accumulation_steps': 4
    }

    scheduler = ASHAScheduler(
        metric="f1",
        mode="max",
        max_t=50,
        grace_period=10,
        reduction_factor=2
    )

    result = tune.run(
        train_model,
        resources_per_trial={"cpu": int(config["cpu_per_trial"]), "gpu": int(config["gpus_per_trial"])},
        config=config,
        num_samples=int(config['num_samples']),
        scheduler=scheduler,
        storage_path='/home/netzone22/mydisk'  # 修改为您的路径
    )

    best_trial = result.get_best_trial("f1", "max", "last")
    print(f"Best trial found at {best_trial}")
    print(f"Best trial final validation F1 score: {best_trial.last_result['f1']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

if __name__ == "__main__":
    main_sweet()