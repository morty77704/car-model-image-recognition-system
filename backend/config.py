"""
车型图像识别系统 — 配置文件
三个组员，三个模型，三路并行推理
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# 三模型配置
# ============================================================

MODELS_CONFIG = [
    {
        "id": "model_1",
        "name": "组员1 - EfficientNet-B3",
        "architecture": "efficientnet_b3",
        "weight_path": str(BASE_DIR / "weights" / "efficientnet_b3.pt"),
        "input_size": 300,
        "member": "组员1",
        "description": "EfficientNet-B3 + Stanford Cars 196类微调，验证准确率 90.25%",
    },
    {
        "id": "model_2",
        "name": "组员2 - ResNet50",
        "architecture": "resnet50",
        "weight_path": str(BASE_DIR / "weights" / "resnet50.pt"),
        "input_size": 224,
        "member": "组员2",
        "description": "ResNet50 + Stanford Cars 196类微调",
    },
    {
        "id": "model_3",
        "name": "组员3 - MobileNetV3",
        "architecture": "mobilenet_v3_large",
        "weight_path": str(BASE_DIR / "weights" / "mobilenet_v3.pt"),
        "input_size": 224,
        "member": "组员3",
        "description": "MobileNetV3-Large + Stanford Cars 196类微调",
    },
]

CLASSES_FILE = str(BASE_DIR / "classes.txt")
FEEDBACK_FILE = str(BASE_DIR / "feedback.jsonl")
HOST = "0.0.0.0"
PORT = 8080
