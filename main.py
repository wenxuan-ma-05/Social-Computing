"""
融合BERT+GAT的高级计算分析框架
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import dgl
import dgl.nn.pytorch as dglnn
import networkx as nx
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, ndcg_score
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ===================== 全局配置与设备设置 =====================
CURRENT_DIR = os.path.dirname(__file__)
RESULT_DIR = os.path.join(CURRENT_DIR, "result")
DATA_DIR = os.path.join(CURRENT_DIR, "data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{DEVICE}")

# 创建结果文件夹
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

# BERT配置
BERT_MODEL_NAME = "bert-base-chinese"
BERT_MAX_LEN = 128
BERT_LEARNING_RATE = 2e-5

# GAT配置
GAT_HIDDEN_DIM = [256, 128]
GAT_HEADS = [4, 2]
GAT_LEARNING_RATE = 1e-4

# 训练配置
BATCH_SIZE = 32
EPOCHS = 20
PATIENCE = 10  # 早停耐心值
WEIGHT_DECAY = 1e-4

# 事件配置
EVENT_TYPES = {
    "公共安全类": "预处理后_公共安全类数据.xlsx",
    "民生政策类": "预处理后_民生政策类数据.xlsx"
}

# ===================== 数据预处理与数据集构建 =====================
class YuQingDataset(Dataset):
    def __init__(self, df, tokenizer, event_type):
        self.df = df
        self.tokenizer = tokenizer
        self.event_type = event_type
        self.scaler = MinMaxScaler()
        
        # 文本编码
        self.text_encodings = self.tokenizer(
            df["清洗后内容"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=BERT_MAX_LEN,
            return_tensors="pt"
        )
        
        # 结构特征标准化
        structure_features = df[["加权度中心性", "介数中心性", "紧密中心性", "聚类系数"]].values
        self.structure_features = torch.tensor(self.scaler.fit_transform(structure_features), dtype=torch.float32)
        
        # 传播特征标准化
        propagation_features = df[["传播延迟", "扩散范围", "互动频率"]].values
        self.propagation_features = torch.tensor(self.scaler.fit_transform(propagation_features), dtype=torch.float32)
        
        # 关键节点标注处理（无标注时自动生成）
        if "关键节点标注" not in self.df.columns or self.df["关键节点标注"].isnull().all():
            self.df["关键节点标注"] = 0
            top_threshold = self.df["加权度中心性"].quantile(0.9)
            self.df.loc[self.df["加权度中心性"] >= top_threshold, "关键节点标注"] = 1
        
        # 情感标签处理（无标注时默认填充）
        if "情感标签" not in self.df.columns:
            self.df["情感标签"] = 1  # 1代表中性
        
        self.labels = torch.tensor(self.df["关键节点标注"].values, dtype=torch.long)
        self.emotion_labels = torch.tensor(self.df["情感标签"].values, dtype=torch.long)
        self.user_ids = self.df["用户ID"].tolist()
        self.forward_source_ids = self.df["转发来源ID"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "input_ids": self.text_encodings["input_ids"][idx],
            "attention_mask": self.text_encodings["attention_mask"][idx],
            "structure_features": self.structure_features[idx],
            "propagation_features": self.propagation_features[idx],
            "labels": self.labels[idx],
            "emotion_labels": self.emotion_labels[idx],
            "user_id": self.user_ids[idx],
            "forward_source_id": self.forward_source_ids[idx]
        }

def preprocess_advanced_data(df, event_type, tokenizer):
    """高级数据预处理：文本清洗+特征工程+网络构建"""
    # 1. 高级文本清洗
    df["清洗后内容"] = df["内容"].str.replace(r'[^\u4e00-\u9fa5\w\s]', '', regex=True)
    df["清洗后内容"] = df["清洗后内容"].str.strip().fillna("无内容")
    
    # 2. 基础时间特征计算
    df["发布时间"] = pd.to_datetime(df["发布时间"], errors='coerce').fillna(datetime.now())
    if "事件发生时间" not in df.columns:
        df["事件发生时间"] = df["发布时间"].min()
    if "最新互动时间" not in df.columns:
        df["最新互动时间"] = df["发布时间"]
    
    # 3. 传播特征计算（前置计算，避免时序错误）
    df["传播延迟"] = (df["发布时间"] - pd.to_datetime(df["事件发生时间"])).dt.total_seconds() / 3600
    df["传播延迟"] = df["传播延迟"].fillna(0)
    
    # 扩散范围计算（除数保护）
    max_repost = df["转发数"].max() if df["转发数"].max() > 0 else 1
    df["扩散范围"] = df["转发数"].fillna(0) / max_repost
    
    # 互动频率计算（除数保护）
    time_diff = (df["最新互动时间"] - df["发布时间"]).dt.total_seconds()
    time_diff = time_diff.where(time_diff > 0, 1)
    df["互动频率"] = (df["转发数"].fillna(0) + df["评论数"].fillna(0) + df["点赞数"].fillna(0)) / time_diff * 3600
    df["互动频率"] = df["互动频率"].fillna(0)
    
    # 4. 网络构建与结构特征计算
    G = build_weighted_hetero_network(df, event_type)
    structure_metrics = calculate_network_metrics(G)
    df = pd.merge(df, structure_metrics, on="用户ID", how="left")
    
    # 5. 缺失值填充
    df = df.fillna(df.median(numeric_only=True))
    
    # 6. 账号类型补充
    if "账号类型" not in df.columns:
        df["账号类型"] = "普通用户"
    
    return df, G

def build_weighted_hetero_network(df, event_type):
    """构建加权异构社会网络"""
    G = nx.DiGraph()
    
    # 添加节点（用户+话题）
    users = df["用户ID"].unique()
    topics = df["关键词"].unique() if "关键词" in df.columns else ["默认话题"]
    G.add_nodes_from(users, node_type="user")
    G.add_nodes_from(topics, node_type="topic")
    
    # 计算边权重
    df["互动强度"] = (df["转发数"].fillna(0) * 0.4 + df["评论数"].fillna(0) * 0.3 + df["点赞数"].fillna(0) * 0.3)
    max_interact = df["互动强度"].max() if df["互动强度"].max() > 0 else 1
    df["互动强度"] = df["互动强度"] / max_interact
    df["时间衰减"] = np.exp(-0.02 * df["传播延迟"].fillna(0))
    
    # 节点类型权重
    user_type_weight = {"媒体": 0.8, "政府": 0.7, "意见领袖": 0.6, "普通用户": 0.3}
    df["节点类型权重"] = df["账号类型"].map(user_type_weight).fillna(0.3)
    
    # 添加用户-用户边
    for _, row in df.iterrows():
        if pd.notna(row["转发来源ID"]) and row["转发来源ID"] in users:
            weight = row["互动强度"] * row["时间衰减"] * row["节点类型权重"]
            G.add_edge(row["转发来源ID"], row["用户ID"], weight=weight, edge_type="repost")
    
    # 添加用户-话题边
    topic_col = "关键词" if "关键词" in df.columns else 0
    for _, row in df.iterrows():
        topic = row[topic_col] if topic_col != 0 else "默认话题"
        G.add_edge(row["用户ID"], topic, weight=row["互动强度"], edge_type="topic_association")
    
    return G

def calculate_network_metrics(G):
    """计算加权网络结构指标"""
    user_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type", "") == "user"]
    metrics = []
    
    for node in user_nodes:
        # 加权度中心性
        weighted_degree = G.degree(node, weight="weight")
        total_weighted_degree = sum(G.degree(n, weight="weight") for n in user_nodes)
        weighted_degree_centrality = weighted_degree / total_weighted_degree if total_weighted_degree > 0 else 0
        
        # 介数中心性（加权）
        betweenness = nx.betweenness_centrality(G, k=min(100, len(user_nodes)), weight="weight")
        
        # 紧密中心性（加权）
        try:
            closeness = nx.closeness_centrality(G, distance="weight")
        except:
            closeness = {node: 0}
        
        # 聚类系数（加权）
        clustering = nx.clustering(G, weight="weight")
        
        metrics.append({
            "用户ID": node,
            "加权度中心性": weighted_degree_centrality,
            "介数中心性": betweenness.get(node, 0),
            "紧密中心性": closeness.get(node, 0),
            "聚类系数": clustering.get(node, 0)
        })
    
    return pd.DataFrame(metrics)

# ===================== 深度学习模型定义 =====================
class BERTFeatureExtractor(nn.Module):
    """基于BERT的文本特征提取模型"""
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        self.dropout = nn.Dropout(0.1)
        self.emotion_classifier = nn.Linear(768, 3)  # 情感分类（负向、中性、正向）
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
        cls_embedding = self.dropout(cls_embedding)
        emotion_logits = self.emotion_classifier(cls_embedding)
        return cls_embedding, emotion_logits

class GATNetwork(nn.Module):
    """图注意力网络模型"""
    def __init__(self, in_dim, hidden_dims, heads):
        super().__init__()
        self.layers = nn.ModuleList()
        # 输入层
        self.layers.append(dglnn.GATConv(in_dim, hidden_dims[0], heads[0], activation=nn.LeakyReLU()))
        # 隐藏层
        for i in range(1, len(hidden_dims)):
            self.layers.append(dglnn.GATConv(hidden_dims[i-1]*heads[i-1], hidden_dims[i], heads[i], activation=nn.LeakyReLU()))
        # 输出层
        self.layers.append(dglnn.GATConv(hidden_dims[-1]*heads[-1], 1, 1, activation=None))
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, g, features):
        h = features
        for layer in self.layers[:-1]:
            h = layer(g, h)
            h = h.flatten(1)  # 多头注意力拼接
            h = self.dropout(h)
        # 输出层
        out = self.layers[-1](g, h)
        return out.squeeze(), h  # 节点得分与增强特征

class BERTGATFusionModel(nn.Module):
    """BERT+GAT融合模型"""
    def __init__(self):
        super().__init__()
        self.bert_extractor = BERTFeatureExtractor()
        self.structure_encoder = nn.Linear(4, 64)
        self.propagation_encoder = nn.Linear(3, 32)
        self.fusion_encoder = nn.Linear(768 + 64 + 32 + 3, 512)  # 768(BERT)+64(结构)+32(传播)+3(情感)
        self.gat_network = GATNetwork(512, GAT_HIDDEN_DIM, GAT_HEADS)
        self.classifier = nn.Linear(GAT_HIDDEN_DIM[-1]*GAT_HEADS[-1], 2)  # 二分类（关键节点/非关键节点）
    
    def forward(self, g, batch_data):
        # BERT特征提取
        bert_embedding, emotion_logits = self.bert_extractor(
            batch_data["input_ids"].to(DEVICE),
            batch_data["attention_mask"].to(DEVICE)
        )
        
        # 结构特征编码
        structure_feat = self.structure_encoder(batch_data["structure_features"].to(DEVICE))
        structure_feat = nn.ReLU()(structure_feat)
        
        # 传播特征编码
        propagation_feat = self.propagation_encoder(batch_data["propagation_features"].to(DEVICE))
        propagation_feat = nn.ReLU()(propagation_feat)
        
        # 情感特征（one-hot）
        emotion_feat = nn.Softmax(dim=1)(emotion_logits)
        
        # 特征融合
        fused_feat = torch.cat([bert_embedding, structure_feat, propagation_feat, emotion_feat], dim=1)
        fused_feat = self.fusion_encoder(fused_feat)
        fused_feat = nn.ReLU()(fused_feat)
        
        # GAT网络
        node_scores, enhanced_feat = self.gat_network(g, fused_feat)
        
        # 分类
        logits = self.classifier(enhanced_feat)
        
        return logits, node_scores, emotion_logits

# ===================== 模型训练与验证 =====================
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, patience):
    """模型训练（含早停策略）"""
    model.to(DEVICE)
    best_val_f1 = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # 构建批次真实子图（基于转发关系）
            batch_user_ids = batch["user_id"]
            src_nodes = []
            dst_nodes = []
            for idx, src_id in enumerate(batch["forward_source_id"]):
                if pd.notna(src_id) and src_id in batch_user_ids:
                    src_nodes.append(idx)
                    dst_nodes.append(batch_user_ids.index(src_id))
            
            # 构建图并补充自环
            g = dgl.graph((src_nodes, dst_nodes), num_nodes=len(batch_user_ids)).to(DEVICE)
            g = dgl.add_self_loop(g)
            
            logits, _, emotion_logits = model(g, batch)
            
            # 损失计算（分类损失+情感损失）
            cls_loss = criterion(logits, batch["labels"].to(DEVICE))
            emotion_criterion = nn.CrossEntropyLoss()
            emotion_loss = emotion_criterion(emotion_logits, batch["emotion_labels"].to(DEVICE))
            total_loss = cls_loss + 0.1 * emotion_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_labels.extend(batch["labels"].numpy())
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # 构建验证批次图
                batch_user_ids = batch["user_id"]
                src_nodes = []
                dst_nodes = []
                for idx, src_id in enumerate(batch["forward_source_id"]):
                    if pd.notna(src_id) and src_id in batch_user_ids:
                        src_nodes.append(idx)
                        dst_nodes.append(batch_user_ids.index(src_id))
                
                g = dgl.graph((src_nodes, dst_nodes), num_nodes=len(batch_user_ids)).to(DEVICE)
                g = dgl.add_self_loop(g)
                
                logits, _, emotion_logits = model(g, batch)
                
                cls_loss = criterion(logits, batch["labels"].to(DEVICE))
                emotion_loss = emotion_criterion(emotion_logits, batch["emotion_labels"].to(DEVICE))
                total_loss = cls_loss + 0.1 * emotion_loss
                
                val_loss += total_loss.item()
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(batch["labels"].numpy())
        
        # 计算指标
        train_f1 = f1_score(train_labels, train_preds, average="binary", zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, average="binary", zero_division=0)
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        
        print(f"Epoch {epoch+1}/{epochs} | 训练损失：{train_loss_avg:.4f} | 验证损失：{val_loss_avg:.4f}")
        print(f"训练F1：{train_f1:.4f} | 验证F1：{val_f1:.4f}")
        
        # 早停策略
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(RESULT_DIR, "best_fusion_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发，最佳验证F1：{best_val_f1:.4f}")
                break
    
    # 绘制损失曲线
    plt.plot(train_losses, label="训练损失")
    plt.plot(val_losses, label="验证损失")
    plt.xlabel("Epoch")
    plt.ylabel("损失值")
    plt.legend()
    plt.savefig(os.path.join(RESULT_DIR, "训练损失曲线.png"))
    plt.close()
    
    return best_val_f1

def evaluate_model(model, test_loader):
    """模型评估（多维度指标）"""
    model.to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(RESULT_DIR, "best_fusion_model.pth")))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 构建评估批次图
            batch_user_ids = batch["user_id"]
            src_nodes = []
            dst_nodes = []
            for idx, src_id in enumerate(batch["forward_source_id"]):
                if pd.notna(src_id) and src_id in batch_user_ids:
                    src_nodes.append(idx)
                    dst_nodes.append(batch_user_ids.index(src_id))
            
            g = dgl.graph((src_nodes, dst_nodes), num_nodes=len(batch_user_ids)).to(DEVICE)
            g = dgl.add_self_loop(g)
            
            logits, node_scores, _ = model(g, batch)
            
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(batch["labels"].numpy())
            all_scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    
    # 计算指标
    precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    map_score = average_precision_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0
    ndcg = ndcg_score([all_labels], [all_scores]) if len(set(all_labels)) > 1 else 0
    
    print("\n模型评估结果：")
    print(f"准确率：{precision:.4f}")
    print(f"召回率：{recall:.4f}")
    print(f"F1分数：{f1:.4f}")
    print(f"MAP：{map_score:.4f}")
    print(f"NDCG@10：{ndcg:.4f}")
    
    # 保存评估结果
    eval_results = pd.DataFrame({
        "指标": ["准确率", "召回率", "F1分数", "MAP", "NDCG@10"],
        "数值": [precision, recall, f1, map_score, ndcg]
    })
    eval_results.to_excel(os.path.join(RESULT_DIR, "模型评估结果.xlsx"), index=False)
    
    return eval_results

# ===================== 主流程 =====================
def main():
    start_time = datetime.now()
    print(f"【高级社会计算分析框架启动】{start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 加载数据
    event_name = "公共安全类"  # 可切换为"民生政策类"
    df_raw = pd.read_excel(os.path.join(DATA_DIR, EVENT_TYPES[event_name]))
    
    # 2. 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    # 3. 高级数据预处理
    df_processed, G = preprocess_advanced_data(df_raw, event_name, tokenizer)
    
    # 4. 构建数据集与数据加载器
    dataset = YuQingDataset(df_processed, tokenizer, event_name)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 5. 初始化模型、优化器、损失函数
    model = BERTGATFusionModel()
    optimizer = optim.AdamW([
        {"params": model.bert_extractor.parameters(), "lr": BERT_LEARNING_RATE},
        {"params": model.gat_network.parameters(), "lr": GAT_LEARNING_RATE},
        {"params": model.structure_encoder.parameters(), "lr": 1e-4},
        {"params": model.propagation_encoder.parameters(), "lr": 1e-4},
        {"params": model.fusion_encoder.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-4}
    ], weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    # 6. 模型训练
    print("\n开始模型训练...")
    best_val_f1 = train_model(model, train_loader, val_loader, optimizer, criterion, EPOCHS, PATIENCE)
    
    # 7. 模型评估
    print("\n开始模型评估...")
    eval_results = evaluate_model(model, test_loader)
    
    # 8. 关键节点识别与保存
    print("\n开始关键节点识别...")
    model.load_state_dict(torch.load(os.path.join(RESULT_DIR, "best_fusion_model.pth")))
    model.eval()
    
    all_user_ids = []
    all_scores = []
    all_preds = []
    
    # 构建用户ID-账号类型映射字典
    user_type_dict = dict(zip(df_processed["用户ID"], df_processed["账号类型"]))
    
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False):
            # 构建推理批次图
            batch_user_ids = batch["user_id"]
            src_nodes = []
            dst_nodes = []
            for idx, src_id in enumerate(batch["forward_source_id"]):
                if pd.notna(src_id) and src_id in batch_user_ids:
                    src_nodes.append(idx)
                    dst_nodes.append(batch_user_ids.index(src_id))
            
            g = dgl.graph((src_nodes, dst_nodes), num_nodes=len(batch_user_ids)).to(DEVICE)
            g = dgl.add_self_loop(g)
            
            logits, node_scores, _ = model(g, batch)
            
            all_user_ids.extend(batch["user_id"])
            all_scores.extend(node_scores.cpu().numpy())
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    
    # 保存关键节点列表
    key_nodes_df = pd.DataFrame({
        "用户ID": all_user_ids,
        "节点重要性得分": all_scores,
        "是否关键节点": all_preds,
        "账号类型": [user_type_dict.get(uid, "未知") for uid in all_user_ids]
    })
    key_nodes_df = key_nodes_df.sort_values("节点重要性得分", ascending=False)
    key_nodes_df.to_excel(os.path.join(RESULT_DIR, f"{event_name}_关键节点识别结果.xlsx"), index=False)
    
    # 保存网络结构文件（用于可视化）
    nx.write_gexf(G, os.path.join(RESULT_DIR, f"{event_name}_社会网络结构.gexf"))
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    print(f"\n【高级社会计算分析完成】耗时：{duration:.2f}分钟")
    print(f"结果文件已保存至：{RESULT_DIR}")

if __name__ == "__main__":
    main()