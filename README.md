# Repository Name

**proposed structure**

```
project_name/
│
├── data/               # 数据文件夹（原始图像/标签等）
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/             # 存放模型结构（比如ResNet变体、自定义网络）
│   ├── resnet_backbone.py
│   ├── classifier_head.py
│   └── __init__.py
│
├── utils/              # 通用工具函数
│   ├── logger.py       # 打印日志、保存日志
│   ├── metric.py       # 准确率、召回率、F1等计算
│   ├── data_loader.py  # 加载和预处理数据
│   └── __init__.py
│
├── misc/               # 杂项工具
│   ├── resize_helper.py # 特别定制的resize策略
│   ├── debug_tools.py   # 小型调试脚本
│   └── __init__.py
│
├── config/             # 配置文件（超参数设置等）
│   ├── config.yaml
│   └── __init__.py
│
├── train.py            # 训练主程序
├── validate.py         # 验证/测试程序
├── predict.py          # 推理脚本（可选）
│
├── requirements.txt    # 依赖包列表
├── README.md           # 项目说明
└── .gitignore          # Git忽略文件
```

