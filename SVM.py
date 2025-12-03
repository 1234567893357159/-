import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline  # 用Pipeline整合预处理+模型，避免数据泄露
from sklearn.svm import SVC  # 支持向量机分类器
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
import joblib  # 保存SVM模型和预处理流水线

# 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# -------------------------- 1. 数据加载与预处理 --------------------------
def load_and_preprocess_data(data_path: str):
    """加载数据并划分训练集/测试集（与PyTorch模型预处理逻辑一致）"""
    # 加载数据（仅保留“是否患病”标签）
    df = pd.read_csv(data_path)
    print(f"原始数据形状：{df.shape}")
    print(f"患病标签（target）分布：\n{df['target'].value_counts()}\n")

    # 分离特征（X）和标签（y：是否患病，0=未患病，1=患病）
    X = df.drop("target", axis=1)
    y = df["target"].values

    # 划分训练集/测试集（分层抽样，与PyTorch模型参数一致：test_size=0.3，random_state=42）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"训练集：{X_train.shape[0]}样本，测试集：{X_test.shape[0]}样本")

    # 定义特征预处理逻辑（与PyTorch模型完全一致，确保公平对比）
    numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]  # 数值特征
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]  # 分类特征

    preprocessor = ColumnTransformer(
        transformers=[
            # 数值特征：标准化（消除量纲影响，SVM对特征尺度敏感，必须做标准化）
            ("num", StandardScaler(), numeric_features),
            # 分类特征：独热编码（drop="first"避免多重共线性）
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features)
        ])

    return X_train, X_test, y_train, y_test, preprocessor


# -------------------------- 2. 训练SVM模型 --------------------------
def train_svm_model(X_train, y_train, preprocessor):
    """用Pipeline整合“预处理+SVM”，避免数据泄露，训练模型"""
    # 构建Pipeline：先预处理，再训练SVM（SVM参数经过基础调优）
    # kernel='rbf'：径向基核函数（适合非线性数据，心脏病特征与患病的关系非完全线性）
    # C=1.0：正则化强度（控制过拟合，C越小正则化越强）
    # gamma='scale'：核函数系数（自动根据特征尺度调整，避免手动调参）
    svm_pipeline = Pipeline([
        ("preprocessor", preprocessor),  # 第一步：特征预处理
        ("svm", SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))  # 第二步：SVM分类
    ])

    # 训练模型（Pipeline会自动对训练集做预处理，无需手动调用preprocessor.transform）
    print("\n开始训练SVM模型...")
    svm_pipeline.fit(X_train, y_train)
    print("SVM模型训练完成！\n")

    # 保存模型（后续可直接加载预测，无需重新训练）
    joblib.dump(svm_pipeline, "svm_heart_disease_pipeline.pkl")
    print("✅ SVM模型已保存为：svm_heart_disease_pipeline.pkl")

    return svm_pipeline


# -------------------------- 3. 评估模型性能 --------------------------
def evaluate_svm_model(svm_pipeline, X_test, y_test):
    """评估SVM模型在测试集上的性能，输出核心指标和可视化"""
    # 1. 预测测试集结果（概率和类别）
    y_pred = svm_pipeline.predict(X_test)  # 预测类别（0/1）
    y_pred_proba = svm_pipeline.predict_proba(X_test)[:, 1]  # 预测患病概率（取第二类概率）

    # 2. 计算核心评估指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)  # 召回率（重点关注漏诊）
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 3. 打印评估结果
    print("="*60)
    print("                    SVM模型测试集评估结果")
    print("="*60)
    print(f"准确率（Accuracy）：{accuracy:.4f}")
    print(f"召回率（Recall，避免漏诊）：{recall:.4f}")
    print(f"F1分数（综合精度与召回）：{f1:.4f}")
    print(f"AUC值（概率区分能力）：{auc:.4f}")
    print("\n分类报告（详细指标）：")
    print(classification_report(y_test, y_pred, target_names=["未患病", "患病"]))
    print("="*60)

    # 4. 绘制混淆矩阵（直观查看漏诊/误诊情况）
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["未患病", "患病"], yticklabels=["未患病", "患病"])
    plt.title("SVM模型：是否患病预测混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.tight_layout()
    plt.savefig("svm_confusion_matrix.png", dpi=300)
    print(f"\n✅ 混淆矩阵已保存为：svm_confusion_matrix.png")
    plt.show()

    # 5. 绘制ROC曲线（直观查看AUC效果）
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC曲线（AUC = {auc:.4f}）")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")  # 随机猜测线
    plt.xlabel("假阳性率（误诊率）")
    plt.ylabel("真阳性率（召回率）")
    plt.title("SVM模型：ROC曲线")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("svm_roc_curve.png", dpi=300)
    print(f"✅ ROC曲线已保存为：svm_roc_curve.png")
    plt.show()

    return accuracy, recall, f1, auc


# -------------------------- 4. 新样本预测 --------------------------
def predict_new_sample(svm_model_path: str, new_sample: pd.DataFrame):
    """加载已保存的SVM模型，预测新样本是否患病"""
    # 加载模型
    try:
        svm_pipeline = joblib.load(svm_model_path)
        print(f"\n✅ 成功加载SVM模型：{svm_model_path}")
    except FileNotFoundError:
        print(f"❌ 错误：未找到模型文件 {svm_model_path}")
        return

    # 预测新样本
    y_pred = svm_pipeline.predict(new_sample)  # 预测类别
    y_pred_proba = svm_pipeline.predict_proba(new_sample)  # 预测概率（[未患病概率, 患病概率]）
    no_disease_prob = round(y_pred_proba[0][0], 4)
    disease_prob = round(y_pred_proba[0][1], 4)
    pred_result = "患病" if y_pred[0] == 1 else "未患病"

    # 打印预测结果
    print("\n" + "="*60)
    print("                    SVM模型新样本预测结果")
    print("="*60)
    print(f"未患病概率：{no_disease_prob}")
    print(f"患病概率：{disease_prob}")
    print(f"预测结论：{pred_result}")
    print("="*60)
    print("⚠️  提示：该结果仅为模型预测，不能替代专业医疗诊断")
    print("="*60)

    return no_disease_prob, disease_prob, pred_result


# -------------------------- 主函数：整合全流程 --------------------------
if __name__ == "__main__":
    # 1. 加载并预处理数据（数据路径替换为你的heart_clean_no_negative.csv路径）
    data_path = "heart_clean_no_negative.csv"
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(data_path)

    # 2. 训练SVM模型
    svm_model = train_svm_model(X_train, y_train, preprocessor)

    # 3. 评估模型性能
    evaluate_svm_model(svm_model, X_test, y_test)

    # 4. 示例：预测新样本（与PyTorch模型的示例样本一致，方便对比）
    new_sample = pd.DataFrame({
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
    predict_new_sample("svm_heart_disease_pipeline.pkl", new_sample)