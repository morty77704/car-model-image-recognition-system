"""
车型图像识别系统 - FastAPI 后端服务
三模型并行推理 + 静态前端 + 用户反馈 + 管理员统计
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import FEEDBACK_FILE
from model_manager import ModelManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
)
logger = logging.getLogger("server")

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app = FastAPI(title="车型图像识别系统", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ModelManager()
load_report = manager.initialize()

logger.info("=" * 50)
logger.info(f"类别数: {load_report['classes_count']}")
for model in load_report["models"]:
    status = "OK" if model["loaded"] else "FAIL"
    acc = f" acc={model['val_acc']:.4f}" if model["val_acc"] is not None else ""
    logger.info(f"  {status} {model['name']}{acc}")
logger.info("=" * 50)


class RecognizeRequest(BaseModel):
    image: str


class FeedbackRequest(BaseModel):
    image_id: str
    model_id: str | None = None
    is_correct: bool
    predicted_label: str | None = None
    confidence: float | None = None
    correct_label: str | None = None
    comment: str | None = None


def _load_feedback_records() -> list[dict]:
    feedback_path = Path(FEEDBACK_FILE)
    records = []

    if feedback_path.exists():
        with open(feedback_path, "r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                content = line.strip().lstrip("\ufeff")
                if not content:
                    continue
                try:
                    item = json.loads(content)
                    item["id"] = line_no
                    records.append(item)
                except json.JSONDecodeError:
                    logger.warning(f"跳过无法解析的反馈记录 line={line_no}")

    records.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
    return records


def _build_admin_feedback_payload() -> dict:
    records = _load_feedback_records()
    models = manager.get_status().get("models", [])

    model_stats: dict[str, dict] = {}
    for model in models:
        model_stats[model["id"]] = {
            "model_id": model["id"],
            "model_name": model["name"],
            "member": model.get("member"),
            "architecture": model.get("architecture"),
            "description": model.get("description"),
            "loaded": model.get("loaded", False),
            "val_acc": model.get("val_acc"),
            "feedback_total": 0,
            "correct": 0,
            "wrong": 0,
            "accuracy_by_feedback": None,
            "avg_confidence": None,
            "recent_feedback_at": None,
            "recent_error_examples": [],
        }

    for item in records:
        model_id = item.get("model_id")
        if not model_id:
            continue

        if model_id not in model_stats:
            model_stats[model_id] = {
                "model_id": model_id,
                "model_name": model_id,
                "member": None,
                "architecture": None,
                "description": None,
                "loaded": False,
                "val_acc": None,
                "feedback_total": 0,
                "correct": 0,
                "wrong": 0,
                "accuracy_by_feedback": None,
                "avg_confidence": None,
                "recent_feedback_at": None,
                "recent_error_examples": [],
            }

        stat = model_stats[model_id]
        stat["feedback_total"] += 1

        if item.get("is_correct") is True:
            stat["correct"] += 1
        elif item.get("is_correct") is False:
            stat["wrong"] += 1

        if stat["recent_feedback_at"] is None:
            stat["recent_feedback_at"] = item.get("timestamp")

        confidence = item.get("confidence")
        if isinstance(confidence, (int, float)):
            stat.setdefault("_confidence_values", []).append(float(confidence))

        if item.get("is_correct") is False and len(stat["recent_error_examples"]) < 5:
            stat["recent_error_examples"].append({
                "timestamp": item.get("timestamp"),
                "image_id": item.get("image_id"),
                "predicted_label": item.get("predicted_label"),
                "correct_label": item.get("correct_label"),
                "comment": item.get("comment"),
            })

    for stat in model_stats.values():
        total = stat["feedback_total"]
        if total:
            stat["accuracy_by_feedback"] = round(stat["correct"] / total, 4)

        values = stat.pop("_confidence_values", [])
        if values:
            stat["avg_confidence"] = round(sum(values) / len(values), 4)

    total = len(records)
    correct_count = sum(1 for item in records if item.get("is_correct") is True)
    wrong_count = sum(1 for item in records if item.get("is_correct") is False)

    return {
        "status": "ok",
        "data": records,
        "stats": {
            "total": total,
            "correct": correct_count,
            "wrong": wrong_count,
            "accuracy_by_feedback": round(correct_count / total, 4) if total else None,
            "models_with_feedback": sum(
                1 for stat in model_stats.values() if stat["feedback_total"] > 0
            ),
        },
        "model_stats": list(model_stats.values()),
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "车型图像识别系统",
        **manager.get_status(),
    }


@app.get("/api/models")
def list_models():
    return manager.get_status()


@app.post("/api/recognize")
def recognize(req: RecognizeRequest):
    if not req.image:
        raise HTTPException(400, "图片数据为空")

    logger.info(f"收到识别请求，图片长度: {len(req.image)} 字符")
    result = manager.recognize(req.image)

    ok_count = sum(1 for item in result["results"] if item.get("status") == "ok")
    logger.info(
        "识别完成 | image_id=%s | 可用模型=%s/%s | 共识=%s %s (%s/%s)",
        result["image_id"],
        ok_count,
        len(result["results"]),
        result["consensus"].get("brand", "?"),
        result["consensus"].get("model", "?"),
        result["consensus"].get("agree_count"),
        result["consensus"].get("total_models"),
    )

    return result


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "image_id": req.image_id,
        "model_id": req.model_id,
        "is_correct": req.is_correct,
        "predicted_label": req.predicted_label,
        "confidence": req.confidence,
        "correct_label": req.correct_label,
        "comment": req.comment,
    }

    feedback_path = Path(FEEDBACK_FILE)
    with open(feedback_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(feedback_entry, ensure_ascii=False) + "\n")

    logger.info(f"反馈已记录 correct={req.is_correct}, image={req.image_id}")
    return {"status": "ok", "message": "感谢反馈"}


@app.get("/api/admin/feedback")
def list_feedback():
    return _build_admin_feedback_payload()


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/admin")
def admin_index():
    return FileResponse(str(FRONTEND_DIR / "admin.html"))


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    from config import HOST, PORT

    logger.info(f"启动服务: http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
