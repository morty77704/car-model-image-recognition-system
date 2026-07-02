# 车型图像识别系统

一个基于 FastAPI + PyTorch + 原生前端实现的课程项目，用于上传车辆图片并由 3 个不同架构的模型并行识别车型，前端展示多模型结果对比、共识结果，以及管理员端的反馈统计与分模型看板。

## 项目简介

本项目面向“车型识别”场景，核心目标包括：

- 使用多个深度学习模型对同一张车辆图片并行推理
- 对比不同模型的预测结果、置信度和延迟
- 通过多数投票生成共识结果
- 收集用户反馈，支持后续数据修正与模型优化
- 在管理员端分别查看各个模型的反馈表现

## 功能特性

- 三模型并行识别
  - EfficientNet-B3
  - ResNet50
  - MobileNetV3-Large
- 前端上传图片并展示：
  - 多模型共识结果
  - 单模型预测结果
  - Top-3 预测
  - 推理耗时
- 用户反馈采集：
  - 正确 / 有误反馈
  - 正确车型补充
  - 备注说明
- 管理员端看板：
  - 反馈总量统计
  - 各模型单独统计
  - 分模型反馈准确率
  - 平均置信度
  - 最近错误样本
  - 按模型筛选反馈明细

## 项目结构

```text
.
├─ backend/                 后端服务、模型管理、训练脚本
│  ├─ main.py               FastAPI 入口
│  ├─ model_manager.py      多模型加载与推理管理
│  ├─ config.py             模型与路径配置
│  ├─ requirements.txt      Python 依赖
│  ├─ train_resnet50.py     ResNet50 训练脚本
│  ├─ train_mobilenet.py    MobileNetV3 训练脚本
│  └─ classes.txt           类别名称
├─ frontend/                前端页面与样式脚本
│  ├─ index.html            识别页
│  ├─ admin.html            管理员页
│  ├─ app.js                识别页逻辑
│  ├─ admin.js              管理员页逻辑
│  └─ style.css             公共样式
├─ docs/                    项目文档
├─ 启动.py                  一键启动脚本
├─ 启动.bat                 Windows 启动脚本
└─ 停止.bat                 Windows 停止脚本
```

## 技术栈

- 后端：FastAPI、Uvicorn
- 深度学习：PyTorch、torchvision
- 图像处理：Pillow
- 前端：HTML、CSS、JavaScript
- 数据分析/训练辅助：scikit-learn、pandas、scipy、matplotlib

## 运行环境

- Python 3.10 及以上
- Windows 环境下可直接使用项目附带的启动脚本

## 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

## 启动方式

### 方式一：使用一键启动脚本

在项目根目录执行：

```bash
python 启动.py
```

或者直接双击：

- `启动.bat`

### 方式二：手动启动后端

```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8080
```

启动后访问：

- 识别页：[http://127.0.0.1:8080/](http://127.0.0.1:8080/)
- 管理员页：[http://127.0.0.1:8080/admin](http://127.0.0.1:8080/admin)

## API 概览

- `GET /api/health`：服务健康检查与模型状态
- `GET /api/models`：获取模型列表与基础信息
- `POST /api/recognize`：上传图片并进行多模型识别
- `POST /api/feedback`：提交用户反馈
- `GET /api/admin/feedback`：管理员端反馈列表与分模型统计

## 模型说明

项目中使用 3 个不同架构的车型识别模型进行对比：

1. EfficientNet-B3：兼顾精度与效率
2. ResNet50：经典残差网络，稳定性较好
3. MobileNetV3-Large：轻量级模型，便于后续部署优化

系统会同时调用多个模型，对结果进行展示，并通过共识投票得到最终推荐结果。

## 关于数据集与权重文件

由于数据集、训练日志、缓存文件和模型权重体积较大，且部分权重文件超过 GitHub 普通仓库单文件限制，本仓库默认**不包含**以下大文件内容：

- 数据集图片
- 训练生成的模型权重 `.pt`
- 本地缓存与日志
- 用户反馈数据

如果你需要完整运行项目，请在本地自行准备并放置：

- `backend/data/`
- `backend/data_smoke/`
- `backend/weights/`
- `backend/weights_smoke/`

并确保 `backend/config.py` 中的模型路径配置正确。

## 管理员端说明

管理员端用于观察项目上线后的反馈质量，当前支持：

- 查看全部反馈总览
- 分别查看每个模型的表现
- 统计每个模型的反馈数量、正确数、错误数
- 查看反馈准确率与平均置信度
- 查看最近错误样本，辅助后续模型优化

这部分很适合继续扩展为：

- 推理历史统计
- 模型耗时趋势
- 高频误识别类别分析
- 错误样本回流训练

## 后续可优化方向

- 增加推理历史日志，而不只依赖用户反馈
- 将反馈数据改为 SQLite/MySQL 存储
- 支持错误样本图片回看
- 增加模型性能趋势图表
- 支持 Docker 化部署
- 支持模型热更新

## 项目说明

这是一个偏课程/实验性质的多模型车型识别系统实现，适合作为：

- 课程设计
- 深度学习项目展示
- 多模型推理系统练习
- 管理员反馈闭环设计示例

