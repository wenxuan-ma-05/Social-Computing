import pandas as pd
import os
from datetime import datetime

# 1. 相对路径
current_dir = os.path.dirname(__file__)  # 获取代码所在目录
data_dir = os.path.join(current_dir, "data")  # 拼接data文件夹路径

# 2. 基础校验：确保data文件夹存在+有符合命名规则的文件
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"自动创建data文件夹：{data_dir}")

# 筛选文件：仅保留“weibo_xxx.xlsx”格式（完全匹配你的命名规则）
file_list = [f for f in os.listdir(data_dir) if f.startswith("weibo_") and f.endswith(".xlsx")]
if not file_list:
    print("错误：data文件夹内未找到'weibo_关键词.xlsx'格式的文件，请确认数据存放路径")
else:
    all_dfs = []
    # 3. 定义两类事件的关键词（与你的爬取关键词完全对应）
    event_category = {
        "公共安全类": ["地铁故障停运", "工厂爆炸事故", "某地突发暴雨", "燃气泄漏", "景区游客被困"],
        "民生政策类": ["医保缴费标准调整", "城乡居民养老保险上调", "公积金提取新规", "义务教育就近入学新政", "跨省异地就医直接结算"]
    }

    # 4. 批量处理每个Excel文件
    for file in file_list:
        # 从文件名提取关键词：weibo_关键词.xlsx → 关键词
        keyword = file.replace("weibo_", "").replace(".xlsx", "")
        file_path = os.path.join(data_dir, file)  # 单个文件的相对路径
        df = pd.read_excel(file_path)
        print(f"正在处理：{file}（关键词：{keyword}，原始数据量：{len(df)}条）")

        # 5. 空值处理（按列优先级处理，保留最大有效数据）
        # 核心列：微博ID为空 → 直接删除（无唯一标识的数据无效）
        df = df.dropna(subset=["微博id"], axis=0)
        # 文本列：内容/链接为空 → 填充默认值
        df["内容"] = df["内容"].fillna("无文本内容")
        df["微博链接"] = df["微博链接"].fillna("无链接")
        # 数值列：转发数/评论数为空 → 填充0（避免后续计算报错）
        df[["转发数", "评论数"]] = df[["转发数", "评论数"]].fillna(0).astype(int)
        # 时间列：发布时间为空 → 用“爬取日期+未知”标注
        crawl_date = datetime.now().strftime("%Y%m%d")
        df["发布时间"] = df["发布时间"].fillna(f"{crawl_date}_未知时间")

        # 6. 自动匹配事件类型（基于提取的关键词）
        if keyword in event_category["公共安全类"]:
            df["事件大类"] = "公共安全类"
        elif keyword in event_category["民生政策类"]:
            df["事件大类"] = "民生政策类"
        else:
            df["事件大类"] = "未分类"
            print(f"警告：{file}的关键词'{keyword}'未匹配到已知事件类型，需手动补充")
        
        # 新增“关键词”列（便于后续追溯原始爬取关键词）
        df["关键词"] = keyword
        all_dfs.append(df)

    # 7. 合并并输出预处理结果（均保存到data文件夹）
    total_df = pd.concat(all_dfs, ignore_index=True)
    # 输出1：总数据（包含所有关键词的预处理结果）
    total_output = os.path.join(data_dir, "预处理后_总数据.xlsx")
    total_df.to_excel(total_output, index=False)

    # 输出2：公共安全类数据（单独拆分）
    ps_df = total_df[total_df["事件大类"] == "公共安全类"].reset_index(drop=True)
    ps_output = os.path.join(data_dir, "预处理后_公共安全类数据.xlsx")
    ps_df.to_excel(ps_output, index=False)

    # 输出3：民生政策类数据（单独拆分）
    policy_df = total_df[total_df["事件大类"] == "民生政策类"].reset_index(drop=True)
    policy_output = os.path.join(data_dir, "预处理后_民生政策类数据.xlsx")
    policy_df.to_excel(policy_output, index=False)

    # 8. 处理结果汇总（清晰展示数据分布）
    print("\n" + "="*60)
    print("数据预处理完成！输出文件：")
    print(f"1. 总数据：{total_output}（总条数：{len(total_df)}）")
    print(f"2. 公共安全类：{ps_output}（条数：{len(ps_df)}，关键词：{ps_df['关键词'].unique().tolist()}）")
    print(f"3. 民生政策类：{policy_output}（条数：{len(policy_df)}，关键词：{policy_df['关键词'].unique().tolist()}）")
    
    # 提醒未分类数据（若有）
    unclassified = total_df[total_df["事件大类"] == "未分类"]
    if len(unclassified) > 0:
        print(f"\n⚠️  存在未分类数据（条数：{len(unclassified)}），涉及关键词：{unclassified['关键词'].unique().tolist()}")
        print("请在event_category字典中补充对应关键词的事件类型")
    print("="*60)