"""
模型管理器 — 加载、管理、推理三个车型识别模型
支持架构: EfficientNet-B3 / ResNet50 / MobileNetV3-Large
"""
import io
import time
import base64
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

from config import MODELS_CONFIG, CLASSES_FILE

logger = logging.getLogger(__name__)


# ============================================================
# 模型构建函数（每种架构一个）
# ============================================================

def build_efficientnet_b3(num_classes: int, pretrained: bool = False) -> nn.Module:
    """EfficientNet-B3，input_size=300"""
    model = models.efficientnet_b3(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_resnet50(num_classes: int, pretrained: bool = False) -> nn.Module:
    """ResNet50，input_size=224"""
    model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_mobilenet_v3_large(num_classes: int, pretrained: bool = False) -> nn.Module:
    """MobileNetV3-Large，input_size=224"""
    model = models.mobilenet_v3_large(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.classifier[3].in_features
    model.classifier = nn.Sequential(
        model.classifier[0],   # 保持原有结构
        model.classifier[1],
        model.classifier[2],
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


BUILDERS = {
    "efficientnet_b3": build_efficientnet_b3,
    "resnet50": build_resnet50,
    "mobilenet_v3_large": build_mobilenet_v3_large,
}

# ImageNet 归一化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# 类别名解析
# ============================================================

def parse_class_name(class_name: str) -> dict:
    """将 'Audi TTS Coupe 2012' 拆解为 brand, model, year"""
    parts = class_name.strip().split()
    if len(parts) >= 3 and parts[-1].isdigit():
        return {
            "brand": parts[0],
            "model": " ".join(parts[1:-1]),
            "year": parts[-1],
        }
    return {"brand": class_name, "model": "", "year": ""}


# ============================================================
# 单个模型包装器
# ============================================================

class CarModel:
    """单个车型识别模型"""

    def __init__(self, config: dict, classes: list[str]):
        self.id = config["id"]
        self.name = config["name"]
        self.member = config["member"]
        self.architecture = config["architecture"]
        self.description = config["description"]
        self.input_size = config["input_size"]
        self.weight_path = config["weight_path"]
        self.classes = classes
        self.loaded = False
        self.val_acc = None
        self._model: Optional[nn.Module] = None
        self._transform = None

    def load(self) -> bool:
        """加载模型权重，返回是否成功"""
        weight_file = Path(self.weight_path)
        if not weight_file.exists():
            logger.warning(f"[{self.name}] 权重文件不存在: {self.weight_path}")
            return False

        try:
            checkpoint = torch.load(weight_file, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.error(f"[{self.name}] 加载权重失败: {e}")
            return False

        # 确定类别数
        num_classes = len(self.classes)
        if "class_names" in checkpoint and len(checkpoint["class_names"]) != num_classes:
            logger.warning(
                f"[{self.name}] 权重类别数({len(checkpoint['class_names'])})"
                f"与classes.txt({num_classes})不一致，使用权重中的类别"
            )

        # 构建模型
        builder = BUILDERS.get(self.architecture)
        if builder is None:
            logger.error(f"[{self.name}] 不支持的架构: {self.architecture}")
            return False

        try:
            self._model = builder(num_classes=num_classes, pretrained=False)
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.eval()
        except Exception as e:
            logger.error(f"[{self.name}] 模型构建/加载失败: {e}")
            return False

        # 推理变换
        self._transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        self.loaded = True
        self.val_acc = checkpoint.get("val_acc", None)

        acc_str = f"val_acc={self.val_acc:.4f}" if self.val_acc else "N/A"
        logger.info(f"[{self.name}] 加载成功 | {acc_str}")
        return True

    def infer(self, image_bytes: bytes) -> dict:
        """对单张图片推理，返回结果字典"""
        t0 = time.time()

        if not self.loaded:
            return {
                "model_id": self.id,
                "model_name": self.name,
                "member": self.member,
                "description": self.description,
                "status": "unavailable",
                "error": "模型未加载",
                "latency_ms": 0,
            }

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = self._transform(img).unsqueeze(0)

            with torch.no_grad():
                logits = self._model(tensor)
                probs = torch.softmax(logits, dim=1)

                # Top-3
                top3_values, top3_indices = probs.topk(min(3, len(self.classes)), dim=1)

            latency_ms = round((time.time() - t0) * 1000, 1)

            # Top-1
            top1_class = self.classes[top3_indices[0, 0].item()]
            top1_conf = round(top3_values[0, 0].item(), 4)
            result = parse_class_name(top1_class)
            result["confidence"] = top1_conf

            # Top-3
            top3 = []
            for i in range(len(top3_indices[0])):
                cls = parse_class_name(self.classes[top3_indices[0, i].item()])
                cls["confidence"] = round(top3_values[0, i].item(), 4)
                top3.append(cls)

            return {
                "model_id": self.id,
                "model_name": self.name,
                "member": self.member,
                "description": self.description,
                "status": "ok",
                "brand": result.get("brand", ""),
                "model": result.get("model", ""),
                "year": result.get("year", ""),
                "confidence": top1_conf,
                "top3": top3,
                "latency_ms": latency_ms,
            }

        except Exception as e:
            logger.error(f"[{self.name}] 推理失败: {e}", exc_info=True)
            return {
                "model_id": self.id,
                "model_name": self.name,
                "member": self.member,
                "description": self.description,
                "status": "error",
                "error": str(e),
                "latency_ms": round((time.time() - t0) * 1000, 1),
            }


# ============================================================
# 模型管理器
# ============================================================

class ModelManager:
    """管理所有模型，提供统一推理接口"""

    def __init__(self):
        self.classes: list[str] = []
        self.models: list[CarModel] = []

    def initialize(self) -> dict:
        """加载类别和所有可用模型，返回状态报告"""
        report = {"classes_count": 0, "models": []}

        # 加载类别名
        classes_path = Path(CLASSES_FILE)
        if classes_path.exists():
            with open(classes_path, "r", encoding="utf-8") as f:
                self.classes = [line.strip() for line in f if line.strip()]
            report["classes_count"] = len(self.classes)
            logger.info(f"加载 {len(self.classes)} 个类别")
        else:
            logger.error(f"类别文件不存在: {CLASSES_FILE}")
            return report

        # 加载模型
        for cfg in MODELS_CONFIG:
            car_model = CarModel(cfg, self.classes)
            success = car_model.load()
            self.models.append(car_model)
            report["models"].append({
                "id": cfg["id"],
                "name": cfg["name"],
                "loaded": success,
                "val_acc": car_model.val_acc,
            })

        loaded_count = sum(1 for m in self.models if m.loaded)
        logger.info(f"模型加载完成: {loaded_count}/{len(self.models)} 可用")

        return report

    def recognize(self, image_base64: str) -> dict:
        """对一张图片调用所有可用模型推理，返回汇总结果"""
        import hashlib

        # 解码图片（只解码一次，所有模型共享）
        try:
            # 去掉可能的 data:image/...;base64, 前缀
            b64_data = image_base64
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            image_bytes = base64.b64decode(b64_data)
        except Exception as e:
            return {
                "status": "error",
                "error": f"图片解码失败: {e}",
                "results": [],
            }

        image_id = hashlib.md5(image_bytes).hexdigest()[:12]

        # 所有模型并行推理（这里顺序执行，因为 Python GIL —
        # 实际模型推理在 PyTorch 内部是并行的，且延迟很低）
        results = []
        for m in self.models:
            r = m.infer(image_bytes)
            results.append(r)

        # 计算共识（多数投票）
        consensus = self._compute_consensus(results)

        total_latency = sum(r.get("latency_ms", 0) for r in results)

        return {
            "status": "ok",
            "image_id": image_id,
            "results": results,
            "consensus": consensus,
            "total_latency_ms": round(total_latency, 1),
        }

    def _compute_consensus(self, results: list[dict]) -> dict:
        """从多个模型结果中计算共识"""
        votes = {}
        for r in results:
            if r.get("status") != "ok":
                continue
            key = f"{r.get('brand','')}+{r.get('model','')}+{r.get('year','')}"
            if key not in votes:
                votes[key] = {"count": 0, "confidences": [], "result": r}
            votes[key]["count"] += 1
            votes[key]["confidences"].append(r.get("confidence", 0))

        if not votes:
            return {"brand": "", "model": "", "year": "", "agree_count": 0,
                    "avg_confidence": 0, "total_models": len(results)}

        best = max(votes.values(), key=lambda v: (v["count"], sum(v["confidences"])))
        confs = best["confidences"]

        return {
            "brand": best["result"].get("brand", ""),
            "model": best["result"].get("model", ""),
            "year": best["result"].get("year", ""),
            "agree_count": best["count"],
            "total_models": sum(1 for r in results if r.get("status") == "ok"),
            "avg_confidence": round(sum(confs) / len(confs), 4) if confs else 0,
        }

    def get_status(self) -> dict:
        """返回所有模型的状态"""
        return {
            "classes_count": len(self.classes),
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "member": m.member,
                    "architecture": m.architecture,
                    "description": m.description,
                    "loaded": m.loaded,
                    "val_acc": m.val_acc,
                }
                for m in self.models
            ],
        }
