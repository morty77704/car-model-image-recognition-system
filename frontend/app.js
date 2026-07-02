/**
 * 车型图像识别系统 — 前端逻辑
 * 上传图片 → 三模型并行识别 → 结果对比展示
 */

// ============================================================
// DOM 引用
// ============================================================
const uploadArea = document.getElementById("uploadArea");
const fileInput = document.getElementById("fileInput");
const previewArea = document.getElementById("previewArea");
const previewImg = document.getElementById("previewImg");
const btnRecognize = document.getElementById("btnRecognize");
const btnClear = document.getElementById("btnClear");
const statusBar = document.getElementById("statusBar");
const consensusSection = document.getElementById("consensusSection");
const consensusCard = document.getElementById("consensusCard");
const resultsSection = document.getElementById("resultsSection");
const resultsGrid = document.getElementById("resultsGrid");
const modelStatusGrid = document.getElementById("modelStatusGrid");

let currentImageBase64 = null;
let currentImageId = null;

// ============================================================
// 初始化
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
    setupUpload();
    loadSystemStatus();
});

// ============================================================
// 上传功能
// ============================================================
function setupUpload() {
    // 点击上传
    uploadArea.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", handleFileSelect);

    // 拖拽上传
    uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadArea.classList.add("drag-over");
    });
    uploadArea.addEventListener("dragleave", () => {
        uploadArea.classList.remove("drag-over");
    });
    uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadArea.classList.remove("drag-over");
        const file = e.dataTransfer.files[0];
        if (file) processFile(file);
    });

    // 按钮
    btnRecognize.addEventListener("click", startRecognition);
    btnClear.addEventListener("click", resetUpload);

    // 粘贴上传
    document.addEventListener("paste", (e) => {
        const item = e.clipboardData?.items?.[0];
        if (item && item.type.startsWith("image/")) {
            const file = item.getAsFile();
            processFile(file);
        }
    });
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) processFile(file);
}

function processFile(file) {
    if (!file.type.startsWith("image/")) {
        showStatus("请上传图片文件", "error");
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        currentImageBase64 = e.target.result;
        previewImg.src = currentImageBase64;
        uploadArea.style.display = "none";
        previewArea.style.display = "block";
        statusBar.style.display = "none";
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    currentImageBase64 = null;
    currentImageId = null;
    previewImg.src = "";
    previewArea.style.display = "none";
    uploadArea.style.display = "block";
    statusBar.style.display = "none";
    consensusSection.style.display = "none";
    resultsSection.style.display = "none";
    fileInput.value = "";
}

// ============================================================
// 识别
// ============================================================
async function startRecognition() {
    if (!currentImageBase64) return;

    // UI 状态
    btnRecognize.disabled = true;
    showStatus("正在识别中，三个模型并行推理...", "loading");
    consensusSection.style.display = "none";
    resultsSection.style.display = "none";

    try {
        const resp = await fetch("/api/recognize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: currentImageBase64 }),
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "识别失败");
        }

        const data = await resp.json();
        currentImageId = data.image_id;

        // 渲染结果
        renderConsensus(data.consensus);
        renderResults(data.results, data.total_latency_ms);

        statusBar.style.display = "none";
        consensusSection.style.display = "block";
        resultsSection.style.display = "block";

    } catch (err) {
        showStatus(`识别失败: ${err.message}`, "error");
    } finally {
        btnRecognize.disabled = false;
    }
}

// ============================================================
// 渲染共识
// ============================================================
function renderConsensus(consensus) {
    const { brand, model, year, agree_count, total_models, avg_confidence } = consensus;

    if (!brand) {
        consensusCard.innerHTML = `
            <div class="consensus-result" style="color:var(--text-secondary)">暂无共识</div>
            <div class="consensus-meta">可用模型未达成一致结果</div>
        `;
        return;
    }

    let agreeClass = "none";
    if (agree_count === total_models) agreeClass = "full";
    else if (agree_count >= total_models / 2) agreeClass = "majority";

    const agreeText = agree_count === total_models
        ? "全部一致" : `${agree_count}/${total_models} 模型一致`;

    consensusCard.innerHTML = `
        <div class="consensus-result">${brand} ${model} (${year})</div>
        <div class="consensus-meta">多模型共识结果</div>
        <div class="consensus-agreement ${agreeClass}">${agreeText}</div>
        <div class="consensus-avg-conf">平均置信度: ${(avg_confidence * 100).toFixed(1)}%</div>
    `;
}

// ============================================================
// 渲染三模型对比
// ============================================================
function renderResults(results, totalLatency) {
    let html = "";

    for (const r of results) {
        if (r.status === "ok") {
            html += renderModelCard(r);
        } else {
            html += renderErrorCard(r);
        }
    }

    // 总耗时提示
    html += `<div style="grid-column:1/-1;text-align:center;color:var(--text-secondary);font-size:0.85rem;padding-top:4px;">
        总推理耗时: ${totalLatency.toFixed(0)}ms
    </div>`;

    resultsGrid.innerHTML = html;

    // 绑定反馈按钮
    document.querySelectorAll(".btn-feedback").forEach(btn => {
        btn.addEventListener("click", () => handleFeedback(btn));
    });
}

function renderModelCard(r) {
    const confPct = (r.confidence * 100).toFixed(1);
    const predictedLabel = `${r.brand} ${r.model} ${r.year}`.trim();
    const confClass = r.confidence >= 0.8 ? "confidence-high"
        : r.confidence >= 0.5 ? "confidence-medium"
        : "confidence-low";

    // Top-3 列表
    let top3HTML = '<div class="top3-title">Top-3 预测</div>';
    if (r.top3) {
        r.top3.forEach((item, i) => {
            const isTop = i === 0;
            top3HTML += `
                <div class="top3-item ${isTop ? 'highlight' : ''}">
                    <span>
                        <span class="top3-rank">#${i + 1}</span>
                        ${item.brand} ${item.model} (${item.year})
                    </span>
                    <span class="top3-conf">${(item.confidence * 100).toFixed(1)}%</span>
                </div>`;
        });
    }

    return `
        <div class="model-card">
            <div class="model-card-header">
                <div>
                    <div class="model-name">${r.model_name}</div>
                    <div class="model-member">${r.member}</div>
                </div>
                <span class="model-status ok">正常</span>
            </div>

            <div class="model-prediction">
                <div class="prediction-label">${r.brand} ${r.model}</div>
                <div class="prediction-detail">${r.year}年款</div>
            </div>

            <div class="confidence-section">
                <div class="confidence-header">
                    <span>置信度</span>
                    <span class="confidence-value">${confPct}%</span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill ${confClass}"
                         style="width:${confPct}%"></div>
                </div>
            </div>

            <div class="model-top3">${top3HTML}</div>

            <div class="model-latency">⏱ ${r.latency_ms.toFixed(0)}ms</div>

            <div class="feedback-section">
                <button class="btn btn-small btn-success btn-feedback"
                        data-model="${r.model_id}"
                        data-predicted="${escapeAttr(predictedLabel)}"
                        data-confidence="${r.confidence}"
                        data-correct="true">✓ 正确</button>
                <button class="btn btn-small btn-danger btn-feedback"
                        data-model="${r.model_id}"
                        data-predicted="${escapeAttr(predictedLabel)}"
                        data-confidence="${r.confidence}"
                        data-correct="false">✗ 有误</button>
            </div>
            <div class="feedback-message" aria-live="polite"></div>
        </div>`;
}

function renderErrorCard(r) {
    const statusLabel = r.status === "unavailable" ? "未就绪" : "异常";
    const statusClass = r.status === "unavailable" ? "unavailable" : "error";
    return `
        <div class="model-card">
            <div class="model-card-header">
                <div>
                    <div class="model-name">${r.model_name}</div>
                    <div class="model-member">${r.member}</div>
                </div>
                <span class="model-status ${statusClass}">${statusLabel}</span>
            </div>
            <div class="model-error">${r.error || "模型未加载，请等待组员提供训练好的权重文件"}</div>
        </div>`;
}

// ============================================================
// 反馈
// ============================================================
async function handleFeedback(btn) {
    const modelId = btn.dataset.model;
    const isCorrect = btn.dataset.correct === "true";
    const predictedLabel = btn.dataset.predicted || "";
    const confidence = Number(btn.dataset.confidence || 0);

    let correctLabel = "";
    let comment = "";
    if (!isCorrect) {
        correctLabel = window.prompt("请输入正确车型（例如：BMW 530Li 2023）：", predictedLabel) || "";
        if (!correctLabel.trim()) return;
        comment = window.prompt("可选：补充错误原因或备注：", "") || "";
    }

    try {
        const resp = await fetch("/api/feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                image_id: currentImageId,
                model_id: modelId,
                is_correct: isCorrect,
                predicted_label: predictedLabel,
                confidence,
                correct_label: correctLabel.trim() || null,
                comment: comment.trim() || null,
            }),
        });
        if (!resp.ok) throw new Error("反馈接口返回异常");

        // 高亮按钮
        const parent = btn.parentElement;
        parent.querySelectorAll(".btn-feedback").forEach(b => b.style.opacity = "0.4");
        btn.style.opacity = "1";
        btn.style.transform = "scale(1.05)";
        const message = parent.parentElement.querySelector(".feedback-message");
        if (message) message.textContent = isCorrect ? "已记录：识别正确" : "已记录：识别有误";

    } catch (err) {
        console.error("反馈提交失败:", err);
        alert(`反馈提交失败：${err.message}`);
    }
}

// ============================================================
// 系统状态
// ============================================================
async function loadSystemStatus() {
    try {
        const resp = await fetch("/api/health");
        const data = await resp.json();
        renderSystemStatus(data.models);
    } catch (err) {
        modelStatusGrid.innerHTML = `
            <div class="loading-status">无法获取模型状态，请确认服务已启动</div>`;
    }
}

function renderSystemStatus(models) {
    if (!models || models.length === 0) {
        modelStatusGrid.innerHTML = `
            <div class="loading-status">暂无模型信息</div>`;
        return;
    }

    let html = "";
    for (const m of models) {
        const dotClass = m.loaded ? "loaded" : "not-loaded";
        const detail = m.loaded && m.val_acc != null
            ? `准确率: ${(m.val_acc * 100).toFixed(1)}%`
            : m.loaded ? "已加载" : "权重文件缺失";

        html += `
            <div class="status-item">
                <div class="status-dot ${dotClass}"></div>
                <div class="status-info">
                    <div class="name">${m.name}</div>
                    <div class="detail">${detail}</div>
                </div>
            </div>`;
    }

    modelStatusGrid.innerHTML = html;
}

// ============================================================
// 工具
// ============================================================
function showStatus(msg, type) {
    statusBar.textContent = msg;
    statusBar.className = `status-bar ${type}`;
}

function escapeAttr(value) {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/"/g, "&quot;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}
