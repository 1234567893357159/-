import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import joblib

# 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# -------------------------- 1. 数据加载与预处理（删除分型相关） --------------------------
# 1.1 加载数据（仅保留target标签，无需构建disease_type）
df = pd.read_csv("heart_clean_no_negative.csv")  # 你的数据路径
print(f"原始数据形状：{df.shape}")
print(f"患病标签（target）分布：\n{df['target'].value_counts()}")  # target=1患病，0未患病

# 1.2 分离特征与标签（仅保留“是否患病”标签y）
X = df.drop("target", axis=1)  # 13个输入特征
y = df["target"].values  # 仅一个标签：是否患病（二分类，0/1）

# 1.3 划分训练集/测试集（按患病标签分层抽样）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\n训练集：{X_train.shape[0]}样本，测试集：{X_test.shape[0]}样本")

# 1.4 特征预处理（与原逻辑一致，仅处理特征）
numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features)
    ])

X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
X_test_processed = preprocessor.transform(X_test).astype(np.float32)
print(f"\n预处理后特征维度：{X_train_processed.shape[1]}（原始13→编码后扩展）")

# 1.5 自定义Dataset（仅传入“是否患病”标签）
class HeartDataset(Dataset):
    def __init__(self, features, label):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.long)  # 分类标签用long型
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.label[idx]  # 仅返回特征和患病标签

# 创建DataLoader（无分型相关数据）
train_dataset = HeartDataset(X_train_processed, y_train)
test_dataset = HeartDataset(X_test_processed, y_test)
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# -------------------------- 2. 模型结构（删除分型输出层，仅保留患病预测） --------------------------
class HeartDiseaseModel(nn.Module):
    def __init__(self, input_dim):
        super(HeartDiseaseModel, self).__init__()
        # 共享隐藏层（保留原结构，提取特征）
        self.hidden1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.hidden2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)
        
        # 仅保留“是否患病”输出层（2个神经元：未患病/患病）
        self.output = nn.Linear(32, 2)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # 隐藏层计算（与原逻辑一致）
        x = self.hidden1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.hidden2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # 仅输出患病预测的logits和概率
        logits = self.output(x)  # 用于计算损失
        prob = self.softmax(logits)  # 用于预测概率（[未患病概率, 患病概率]）
        return logits, prob

# 初始化模型（输入维度不变）
input_dim = X_train_processed.shape[1]
model = HeartDiseaseModel(input_dim)
print(f"\n模型结构：\n{model}")

# 设备指定（与原逻辑一致）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"\n训练设备：{device}")


# -------------------------- 3. 训练逻辑（删除分型损失，仅优化患病预测） --------------------------
# 3.1 损失函数（仅用二分类交叉熵）
criterion = nn.CrossEntropyLoss()  # 仅针对“是否患病”标签

# 3.2 优化器（与原逻辑一致）
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

# 3.3 训练参数（仅记录患病预测的损失和准确率）
epochs = 30
train_history = {
    "loss": [],  # 仅训练损失
    "acc": []    # 仅训练准确率
}
val_history = {
    "loss": [],  # 仅验证损失
    "acc": []    # 仅验证准确率
}


# 3.4 训练1轮函数（删除分型相关计算）
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    for batch_idx, (features, label) in enumerate(loader):
        features, label = features.to(device), label.to(device)
        total_samples += features.size(0)
        
        # 梯度清零→前向传播→损失计算→反向传播
        optimizer.zero_grad()
        logits, prob = model(features)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        
        # 累计损失和正确数
        total_loss += loss.item() * features.size(0)
        pred = torch.argmax(prob, dim=1)  # 预测类别（0/1）
        correct += (pred == label).sum().item()
    
    # 计算平均指标
    avg_loss = total_loss / total_samples
    avg_acc = correct / total_samples
    return avg_loss, avg_acc


# 3.5 评估函数（删除分型相关计算）
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        # 正确：enumerate返回 (批量索引, (特征, 标签))，需先接收索引
        for batch_idx, (features, label) in enumerate(loader):
            # 此时features和label都是PyTorch张量，可正常调用.to(device)
            features, label = features.to(device), label.to(device)
            total_samples += features.size(0)  # 累加样本数（每个批次的样本数）
            
            # 前向传播（仅输出logits和概率）
            logits, prob = model(features)
            
            # 计算损失
            loss = criterion(logits, label)
            total_loss += loss.item() * features.size(0)  # 按批次大小加权累加损失
            
            # 计算正确预测数
            pred = torch.argmax(prob, dim=1)  # 取概率最大的类别（0=未患病，1=患病）
            correct += (pred == label).sum().item()  # 累加正确数
    
    avg_loss = total_loss / total_samples
    avg_acc = correct / total_samples
    return avg_loss, avg_acc


# 执行训练（仅打印患病预测指标）
print("\n" + "="*60)
print("开始训练（共{}轮，仅预测是否患病）".format(epochs))
for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    
    scheduler.step()
    
    # 记录历史
    train_history["loss"].append(train_loss)
    train_history["acc"].append(train_acc)
    val_history["loss"].append(val_loss)
    val_history["acc"].append(val_acc)
    
    # 每5轮打印结果
    if epoch % 5 == 0 or epoch == 1:
        print(f"\n第{epoch:2d}轮 | 训练损失：{train_loss:.4f} | 验证损失：{val_loss:.4f}")
        print(f"        | 训练准确率：{train_acc:.4f} | 验证准确率：{val_acc:.4f}")

print("\n" + "="*60)
print("训练完成！")


# -------------------------- 4. 模型评估（仅分析“是否患病”预测结果） --------------------------
# 获取测试集预测结果（仅概率和类别）
def get_test_predictions(model, loader, device):
    model.eval()
    all_prob = []  # 患病概率：[未患病, 患病]
    all_pred = []  # 预测类别：0/1
    all_true = []  # 真实类别：0/1
    
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for batch_idx, (features, label) in enumerate(loader):
            # 1. 数据移至设备（CPU/GPU），前向传播获取概率（此时prob是PyTorch张量）
            features = features.to(device)
            _, prob = model(features)  # prob: Tensor类型，形状为[batch_size, 2]
            
            # 2. 先对张量计算预测类别（用torch.argmax()，此时仍为Tensor）
            pred_tensor = torch.argmax(prob, dim=1)  # 按维度1取最大值索引，形状为[batch_size]
            
            # 3. 再将Tensor转为numpy数组（移至CPU后转换）
            prob_np = prob.cpu().numpy()  # 概率转为numpy
            pred_np = pred_tensor.cpu().numpy()  # 预测类别转为numpy
            label_np = label.cpu().numpy()  # 真实类别转为numpy
            
            # 4. 累计结果（扩展到列表中）
            all_prob.extend(prob_np)  # 注意：extend用于扩展数组，append用于添加单个元素
            all_pred.extend(pred_np)
            all_true.extend(label_np)
    
    return np.array(all_prob), np.array(all_pred), np.array(all_true)

# 评估指标计算
prob, pred, true = get_test_predictions(model, test_loader, device)
print("\n" + "="*60)
print("【是否患病预测评估结果】")
acc = accuracy_score(true, pred)
recall = recall_score(true, pred)  # 重点关注：避免漏诊
f1 = f1_score(true, pred)
auc = roc_auc_score(true, prob[:, 1])  # 用“患病概率”列计算AUC

print(f"准确率（Accuracy）：{acc:.4f}")
print(f"召回率（Recall，避免漏诊）：{recall:.4f}")
print(f"F1分数（综合精度与召回）：{f1:.4f}")
print(f"AUC值（概率区分能力）：{auc:.4f}")

# 混淆矩阵（仅显示未患病/患病）
plt.figure(figsize=(8, 6))
cm = confusion_matrix(true, pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["未患病", "患病"], yticklabels=["未患病", "患病"])
plt.title("是否患病预测混淆矩阵（PyTorch模型）")
plt.xlabel("预测标签")
plt.ylabel("真实标签")
save_path = "./是否患病预测混淆矩阵.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"\n混淆矩阵已保存到：{save_path}")
plt.show()


# -------------------------- 5. 模型保存与新样本预测（仅输出患病概率） --------------------------
# 5.1 保存模型（仅保留必要文件）
torch.save(model.state_dict(), "heart_disease_binary_weights.pth")  # 二分类权重
joblib.dump(preprocessor, "heart_preprocessor_binary.pkl")  # 预处理流水线
print("\n" + "="*60)
print("模型文件保存完成：")
print("1. 二分类模型权重：heart_disease_binary_weights.pth")
print("2. 特征预处理流水线：heart_preprocessor_binary.pkl")

# 5.2 新样本预测（仅输出是否患病概率）
def predict_disease(new_data, preprocessor, model_path, input_dim, device):
    # 预处理新样本
    new_data_processed = preprocessor.transform(new_data).astype(np.float32)
    new_tensor = torch.tensor(new_data_processed, dtype=torch.float32).to(device)
    
    # 加载模型
    model = HeartDiseaseModel(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 预测
    with torch.no_grad():
        _, prob = model(new_tensor)
    
    # 解析结果（仅关注患病概率）
    prob = prob.cpu().numpy()[0]
    no_disease_prob = round(prob[0], 4)  # 未患病概率
    disease_prob = round(prob[1], 4)     # 患病概率
    pred_result = "患病" if disease_prob > 0.5 else "未患病"  # 概率>0.5判定为患病
    
    # 打印结果
    print("\n【是否患病预测结果】")
    print(f"未患病概率：{no_disease_prob}")
    print(f"患病概率：{disease_prob}")
    print(f"预测结论：{pred_result}")
    
    return disease_prob, pred_result

# 构造新样本（13个特征，与原逻辑一致）
new_sample = pd.DataFrame({
    "age": [55], "sex": [1], "cp": [2], "trestbps": [135], "chol": [260],
    "fbs": [1], "restecg": [1], "thalach": [145], "exang": [0],
    "oldpeak": [1.8], "slope": [1], "ca": [1], "thal": [2]
})

# 调用预测函数
predict_disease(
    new_data=new_sample,
    preprocessor=preprocessor,
    model_path="heart_disease_binary_weights.pth",
    input_dim=input_dim,
    device=device
)


# -------------------------- 6. 训练历史可视化（仅显示患病预测的损失和准确率） --------------------------
def plot_training_history(train_hist, val_hist, epochs):
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 8))
    
    # 损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, train_hist["loss"], label="训练损失", color="red")
    plt.plot(epochs_range, val_hist["loss"], label="验证损失", color="darkred", linestyle="--")
    plt.xlabel("训练轮次（Epochs）")
    plt.ylabel("损失值（Cross Entropy）")
    plt.title("PyTorch二分类模型训练损失变化")
    plt.legend(prop={'family':'SimHei'})
    
    # 准确率曲线
    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, train_hist["acc"], label="训练准确率", color="blue")
    plt.plot(epochs_range, val_hist["acc"], label="验证准确率", color="darkblue", linestyle="--")
    plt.xlabel("训练轮次（Epochs）")
    plt.ylabel("准确率（Accuracy）")
    plt.title("PyTorch二分类模型训练准确率变化")
    plt.legend(prop={'family':'SimHei'})
    
    plt.tight_layout()
    save_path = "./二分类模型训练历史.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n训练历史图已保存到：{save_path}")
    plt.show()

# 绘制训练历史
plot_training_history(train_history, val_history, epochs)