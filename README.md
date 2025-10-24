# BCI EEG Processing

一键流水线：采集 → 预处理 → 特征 → 情绪分类 → 存储

## 运行
1. 安装依赖（Windows / cmd）：
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
2. 运行主程序（无设备将使用模拟数据）：
```
python main.py
```

## 目录
见项目结构注释。

## 环境变量（可选）
- EMOTIV_CLIENT_ID
- EMOTIV_CLIENT_SECRET
