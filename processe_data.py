import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------- 1. 加载数据并指定数值型特征 --------------------------
# 加载数据，保留原始索引（用于后续异常样本追溯）
df = pd.read_csv("heart.csv")
df_with_index = df.reset_index(drop=False).rename(columns={"index": "original_index"})

# 定义需要检测负数异常的数值型特征（均为生理指标，理论上无负数）
numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

# 查看处理前的基础信息与核心统计量（重点：均值、方差、标准差）
print("="*60)
print("【处理前】数据基础信息与统计分析：")
print(f"总样本数：{df.shape[0]}，总特征数：{df.shape[1]}")

# 计算并输出处理前的核心统计量（均值、方差、标准差、最小值）
pre_stats = df[numeric_features].agg([
    "count",  # 有效样本数
    "mean",   # 均值
    "var",    # 方差
    "std",    # 标准差
    "min",    # 最小值（判断是否存在负数）
    "max"     # 最大值（辅助参考）
]).round(2)
print("\n【处理前】各数值特征核心统计量（含均值、方差）：")
print(pre_stats)

# 提前检查是否存在负数（避免后续无异常时白跑流程）
negative_check = {feat: (df[feat] < 0).sum() for feat in numeric_features}
print("\n【处理前】各特征负数异常值初步统计：")
for feat, neg_count in negative_check.items():
    print(f"{feat}：{neg_count} 个负数（最小值：{df[feat].min():.2f}）")
has_negative = any(neg_count > 0 for neg_count in negative_check.values())
if not has_negative:
    print("\n⚠️  注意：所有数值特征均无负数异常值，后续流程仅展示统计对比逻辑！")


# -------------------------- 2. 可视化负数异常值（含最小值标注） --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建子图：左为数值分布直方图（标注最小值），右为是否存在负数的判断图
plt.figure(figsize=(18, 8))
for i, feature in enumerate(numeric_features):
    # 子图1：直方图（标注最小值，直观看是否接近负数）
    plt.subplot(2, 6, i+1)
    sns.histplot(df[feature], bins=10, color="#1f77b4", alpha=0.7)
    min_val = df[feature].min()
    # 在直方图上标注最小值
    plt.axvline(x=min_val, color="red", linestyle="--", label=f"最小值：{min_val:.2f}")
    plt.title(f"{feature} 分布（最小值标注）", fontsize=10)
    plt.legend(fontsize=8)
    
    # 子图2：判断是否存在负数（0为界限，红色=负数区域，绿色=非负区域）
    plt.subplot(2, 6, i+7)
    # 绘制背景色：负数区域红色，非负区域绿色
    plt.axhspan(0, 1, xmin=0, xmax=0.1, color="red", alpha=0.3, label="负数区域")
    plt.axhspan(0, 1, xmin=0.1, xmax=1, color="green", alpha=0.3, label="非负区域")
    # 绘制当前特征的最小值位置
    plt.scatter(x=min_val, y=0.5, color="blue", s=50, zorder=3, label=f"最小值：{min_val:.2f}")
    plt.xlabel(feature, fontsize=9)
    plt.yticks([])  # 隐藏y轴刻度（仅作判断用）
    plt.title(f"{feature} 是否含负数", fontsize=10)
    plt.legend(fontsize=8)

plt.tight_layout()
plt.savefig("负数异常值检测图.png", dpi=300, bbox_inches="tight")
plt.show()


# -------------------------- 3. 检测并删除负数异常值+记录异常样本 --------------------------
def remove_negative_outliers_with_stats(data, features):
    """
    检测并删除数值特征中的负数异常值，记录异常样本，输出统计对比
    参数：
        data: 带原始索引的DataFrame
        features: 数值型特征列表
    返回：
        clean_data: 去除负数后的DataFrame（恢复原始格式）
        negative_records: 负数异常样本详情（无异常则为空）
        post_stats: 处理后的核心统计量
    """
    clean_data = data.copy()
    negative_records = []  # 存储负数异常样本
    
    # 遍历特征，删除负数样本并记录
    for feature in features:
        # 筛选当前特征的负数样本
        neg_samples = clean_data[clean_data[feature] < 0].copy()
        if len(neg_samples) > 0:
            # 记录异常样本信息（原始索引、异常特征、异常值）
            neg_samples["abnormal_feature"] = feature
            neg_samples["abnormal_value"] = neg_samples[feature]
            neg_samples["abnormal_type"] = "负数异常"  # 标记异常类型
            negative_records.append(neg_samples[
                ["original_index", "abnormal_feature", "abnormal_value", "abnormal_type"] + features + ["target"]
            ])
            
            # 删除负数样本
            clean_data = clean_data[clean_data[feature] >= 0]
            print(f"\n{feature}：删除 {len(neg_samples)} 个负数异常样本（最小值：{neg_samples[feature].min():.2f}）")
        else:
            print(f"\n{feature}：无负数异常样本（最小值：{clean_data[feature].min():.2f}）")
    
    # 整理异常样本记录（去重，避免同一样本因多特征负数被重复记录）
    if negative_records:
        negative_records = pd.concat(negative_records, ignore_index=True).drop_duplicates(subset="original_index")
    else:
        negative_records = pd.DataFrame(
            columns=["original_index", "abnormal_feature", "abnormal_value", "abnormal_type"] + features + ["target"]
        )
    
    # 计算处理后的核心统计量
    post_stats = clean_data.drop(columns=["original_index"])[features].agg([
        "count", "mean", "var", "std", "min", "max"
    ]).round(2)
    
    # 恢复干净数据的原始格式（删除original_index列）
    clean_data = clean_data.drop(columns=["original_index"])
    
    return clean_data, negative_records, post_stats

# 执行负数异常值处理
df_clean, negative_df, post_stats = remove_negative_outliers_with_stats(df_with_index, numeric_features)


# -------------------------- 4. 统计对比与结果保存 --------------------------
print("\n" + "="*60)
print("【处理前后】核心统计量对比（重点：均值、方差变化）：")
# 合并处理前后的统计量，方便对比
stats_comparison = pd.concat([
    pre_stats.rename(columns=lambda x: f"处理前_{x}"),
    post_stats.rename(columns=lambda x: f"处理后_{x}")
], axis=1)
print(stats_comparison)

# 计算统计量变化率（直观看均值、方差的变化幅度）
change_stats = pd.DataFrame()
for feat in numeric_features:
    pre_mean = pre_stats.loc["mean", feat]
    post_mean = post_stats.loc["mean", feat]
    pre_var = pre_stats.loc["var", feat]
    post_var = post_stats.loc["var", feat]
    
    # 变化率（避免除以0的情况）
    mean_change = ((post_mean - pre_mean) / pre_mean * 100) if pre_mean != 0 else 0
    var_change = ((post_var - pre_var) / pre_var * 100) if pre_var != 0 else 0
    
    change_stats[feat] = [
        f"{pre_mean:.2f} → {post_mean:.2f}",  # 均值变化
        f"{pre_var:.2f} → {post_var:.2f}",    # 方差变化
        f"{mean_change:.2f}%",                # 均值变化率
        f"{var_change:.2f}%"                  # 方差变化率
    ]
change_stats.index = ["均值变化", "方差变化", "均值变化率", "方差变化率"]
print("\n【处理前后】均值与方差变化详情（含变化率）：")
print(change_stats)

# 保存结果文件
# 1. 负数异常样本（无异常则为空白文件，保持格式一致）
negative_df.to_csv("heart_negative_outliers.csv", index=False, encoding="utf-8")
# 2. 处理后的干净数据
df_clean.to_csv("heart_clean_no_negative.csv", index=False, encoding="utf-8")
# 3. 处理前后统计对比表（方便作业报告引用）
stats_comparison.to_csv("heart_stats_comparison.csv", encoding="utf-8")

print("\n" + "="*60)
print("文件保存完成：")
print(f"1. 负数异常样本记录：heart_negative_outliers.csv（共{len(negative_df)}条记录）")
print(f"2. 去除负数后的干净数据：heart_clean_no_negative.csv（共{len(df_clean)}条样本）")
print(f"3. 处理前后统计对比表：heart_stats_comparison.csv（含均值、方差对比）")

# 预览异常样本（若存在）
if len(negative_df) > 0:
    print("\n【负数异常样本】前5条预览：")
    print(negative_df[["original_index", "abnormal_feature", "abnormal_value", "target"]].head())
else:
    print("\n⚠️  无负数异常样本，heart_negative_outliers.csv为空白文件（仅含表头）")