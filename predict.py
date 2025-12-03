# -------------------------- 独立预测程序：仅判断“是否患病” --------------------------
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib  # 加载预处理流水线
from typing import Tuple  # 类型提示，增强代码可读性

# -------------------------- 1. 必须与训练时一致的模型结构 --------------------------
# 注意：模型结构需和训练代码中的HeartDiseaseModel完全相同，否则加载权重会报错
class HeartDiseaseModel(nn.Module):
    def __init__(self, input_dim: int):
        super(HeartDiseaseModel, self).__init__()
        # 隐藏层（与训练时一致）
        self.hidden1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.hidden2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)
        
        # 仅“是否患病”输出层（2个神经元：未患病/患病）
        self.output = nn.Linear(32, 2)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.hidden1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.hidden2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        logits = self.output(x)
        prob = self.softmax(logits)  # 输出概率：[未患病概率, 患病概率]
        return logits, prob

# -------------------------- 2. 核心预测函数 --------------------------
def predict_heart_disease(new_sample: pd.DataFrame) -> None:
    """
    输入新样本特征，预测是否患病并打印结果
    Args:
        new_sample: DataFrame，1行数据，必须包含13个特征（顺序不限）
                    特征列表：age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    """
    # -------------------------- 步骤1：配置文件路径 --------------------------
    # 确保以下两个文件与predict.py在同一文件夹
    PREPROCESSOR_PATH = "heart_preprocessor_binary.pkl"  # 训练时保存的预处理流水线
    MODEL_WEIGHTS_PATH = "heart_disease_binary_weights.pth"  # 训练时保存的模型权重
    INPUT_DIM = 19  # 预处理后的特征维度（训练时固定为19，无需修改）
    
    # -------------------------- 步骤2：加载预处理流水线 --------------------------
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print(f"✅ 成功加载预处理流水线：{PREPROCESSOR_PATH}")
    except FileNotFoundError:
        print(f"❌ 错误：未找到预处理文件 {PREPROCESSOR_PATH}")
        print("请将预处理文件放在当前文件夹，或修改PREPROCESSOR_PATH路径")
        return
    
    # -------------------------- 步骤3：加载模型权重 --------------------------
    # 初始化模型（结构与训练时一致）
    model = HeartDiseaseModel(input_dim=INPUT_DIM)
    # 选择预测设备（CPU，无需GPU）
    device = torch.device("cpu")
    
    try:
        # 加载权重并移至CPU（map_location确保在CPU上运行）
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True))
        model.to(device)
        model.eval()  # 切换为评估模式（关闭Dropout，确保预测稳定）
        print(f"✅ 成功加载模型权重：{MODEL_WEIGHTS_PATH}")
    except FileNotFoundError:
        print(f"❌ 错误：未找到模型权重文件 {MODEL_WEIGHTS_PATH}")
        print("请将模型权重文件放在当前文件夹，或修改MODEL_WEIGHTS_PATH路径")
        return
    except RuntimeError:
        print("❌ 错误：模型权重与当前模型结构不匹配！")
        print("请确保predict.py中的HeartDiseaseModel与训练代码完全一致")
        return
    
    # -------------------------- 步骤4：预处理新样本 --------------------------
    # 检查新样本是否包含所有必要特征
    required_features = ["age", "sex", "cp", "trestbps", "chol", "fbs", 
                         "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    missing_features = [feat for feat in required_features if feat not in new_sample.columns]
    
    if missing_features:
        print(f"❌ 错误：新样本缺少必要特征：{', '.join(missing_features)}")
        print(f"必须包含的13个特征：{', '.join(required_features)}")
        return
    
    # 用训练好的流水线预处理新样本（避免数据泄露）
    try:
        new_sample_processed = preprocessor.transform(new_sample).astype(np.float32)
        print("✅ 新样本预处理完成")
    except Exception as e:
        print(f"❌ 新样本预处理失败：{str(e)}")
        print("请检查新样本特征值是否为有效数字（如age为整数，chol为正数）")
        return
    
    # -------------------------- 步骤5：模型预测 --------------------------
    # 转换为PyTorch张量（适配模型输入）
    new_sample_tensor = torch.tensor(new_sample_processed, dtype=torch.float32).to(device)
    
    # 禁用梯度计算（预测时无需反向传播）
    with torch.no_grad():
        _, prob = model(new_sample_tensor)
    
    # -------------------------- 步骤6：解析并打印结果 --------------------------
    # 转换为numpy数组，提取概率
    prob_np = prob.cpu().numpy()[0]  # [未患病概率, 患病概率]
    no_disease_prob = round(prob_np[0] * 100, 2)  # 未患病概率（百分比，保留2位小数）
    disease_prob = round(prob_np[1] * 100, 2)    # 患病概率（百分比，保留2位小数）
    
    # 预测结论（概率>50%判定为患病）
    pred_result = "患病" if disease_prob > 50 else "未患病"
    
    # 格式化输出
    print("\n" + "="*60)
    print("                    心脏病是否患病预测结果")
    print("="*60)
    print(f"未患病概率：{no_disease_prob}%")
    print(f"患病概率：{disease_prob}%")
    print(f"预测结论：{pred_result}")
    print("="*60)
    print("⚠️  提示：该结果仅为模型预测，不能替代专业医疗诊断")
    print("="*60)

# -------------------------- 3. 示例：构造新样本并预测 --------------------------
if __name__ == "__main__":
    # -------------------------- 请在这里修改新样本特征 --------------------------
    # 特征说明（参考值范围）：
    # age: 年龄（20-80，如55）
    # sex: 性别（1=男性，0=女性）
    # cp: 胸痛类型（0=典型心绞痛，1=非典型心绞痛，2=非心绞痛，3=无症状）
    # trestbps: 静息血压（90-200 mm Hg，如135）
    # chol: 血清胆固醇（100-400 mg/dl，如260）
    # fbs: 空腹血糖（1=＞120mg/dl，0=≤120mg/dl）
    # restecg: 静息心电图（0=正常，1=ST-T异常，2=左心室肥大）
    # thalach: 最大心率（70-200，如145）
    # exang: 运动诱发心绞痛（1=是，0=否）
    # oldpeak: ST段压低（0.0-6.0，如1.8）
    # slope: ST段斜率（0=上坡，1=平坦，2=下坡）
    # ca: 血管数量（0-3，如1）
    # thal: 地中海贫血（1=正常，2=固定缺陷，3=可逆缺陷）
    new_sample_data = pd.DataFrame({
        "age": [55],       # 年龄：55岁
        "sex": [1],        # 性别：男性
        "cp": [2],         # 胸痛类型：非心绞痛
        "trestbps": [135], # 静息血压：135 mm Hg
        "chol": [260],     # 血清胆固醇：260 mg/dl
        "fbs": [1],        # 空腹血糖：＞120mg/dl
        "restecg": [1],    # 静息心电图：ST-T异常
        "thalach": [145],  # 最大心率：145
        "exang": [0],      # 运动诱发心绞痛：否
        "oldpeak": [1.8],  # ST段压低：1.8
        "slope": [1],      # ST段斜率：平坦
        "ca": [1],         # 血管数量：1
        "thal": [2]        # 地中海贫血：固定缺陷
    })

    # 调用预测函数
    predict_heart_disease(new_sample_data)