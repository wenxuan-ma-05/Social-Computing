# 基于 BERT-GAT 融合模型的舆情关键节点识别系统

## 简介
本项目面向舆情分析领域，实现了从微博舆情数据爬取、预处理到基于 BERT-GAT 融合模型的关键节点识别全流程。系统能够自动爬取公共安全、民生政策类舆情数据，通过多维度特征融合（文本语义、网络结构、传播特征）精准识别舆情传播中的核心节点，输出可视化分析结果与量化评估指标，可为舆情监测、引导与治理提供技术支撑

## 功能

- **数据爬取**: 自动打开微博登录页面，支持cookie保存和加载，避免重复登录，支持自定义关键词、爬取页数，输出结构化 Excel 数据
- **数据预处理**: 完成文本清洗、特征工程、数据分类，生成模型可直接读取的标准化数据集
- **关键节点识别**: 融合 BERT 文本语义特征与 GAT 图注意力网络，实现舆情网络关键节点精准识别
- **结果分析**: 输出模型评估指标（准确率、F1 分数等）、训练曲线、关键节点列表、社会网络图谱等多维度结果。

## 安装

### 前置需求
- Python 3.12.3+
- Chrome浏览器
- CUDA、GPU

### 环境安装
1. 克隆仓库
```bash
git clone https://github.com/your-username/Social-Computing.git
cd Social-Computing
```

2. 安装依赖
```bash
pip install -r requirements.txt
pip install dgl==1.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers==4.28.0
```


## 使用方法

### 数据爬取
1. 打开 `weibo_search.py` 文件，在最后修改搜索关键词:
```python
if __name__ == "__main__":
    keyword = "你的搜索关键词"  # 替换为你要搜索的关键词
```

2. 运行代码:
```bash
python weibo_search.py
```

3. 首次运行会自动弹出 Chrome 浏览器，手动完成微博登录（登录后保存 cookies，后续无需重复登录）；

4. 爬取完成后，数据会保存至项目目录下的data文件夹，文件名为weibo_关键词.xlsx

### 数据预处理

1. 打开data preprocessing.py，确认事件类型映射（可根据需求补充关键词）:
```python
# 事件类型与关键词映射（可自定义扩展）
event_category = {
    "公共安全类": ["地铁故障", "火灾", "交通事故", "安全预警"],
    "民生政策类": ["养老保险", "医保缴费", "公积金", "教育政策"]
}
```

2. 运行代码:
```bash
python data preprocessing.py
```

3. 结果说明:
data文件夹会生成 3 个预处理文件：
预处理后_总数据.xlsx：所有关键词的合并数据；
预处理后_公共安全类数据.xlsx：分类后的公共安全类数据；
预处理后_民生政策类数据.xlsx：分类后的民生政策类数据。

### 关键节点识别

运行代码：
```bash
python python main.py
```
