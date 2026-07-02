const statTotal = document.getElementById("statTotal");
const statCorrect = document.getElementById("statCorrect");
const statWrong = document.getElementById("statWrong");
const statAccuracy = document.getElementById("statAccuracy");
const statModelsWithFeedback = document.getElementById("statModelsWithFeedback");
const modelOverviewGrid = document.getElementById("modelOverviewGrid");
const modelFilterGroup = document.getElementById("modelFilterGroup");
const feedbackTableBody = document.getElementById("feedbackTableBody");
const btnRefresh = document.getElementById("btnRefresh");

const state = {
    records: [],
    modelStats: [],
    selectedModelId: "all",
};

document.addEventListener("DOMContentLoaded", () => {
    btnRefresh.addEventListener("click", loadFeedback);
    modelFilterGroup.addEventListener("click", onFilterClick);
    loadFeedback();
});

async function loadFeedback() {
    btnRefresh.disabled = true;
    try {
        const resp = await fetch("/api/admin/feedback");
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const payload = await resp.json();

        state.records = payload.data || [];
        state.modelStats = payload.model_stats || [];

        renderStats(payload.stats || {});
        renderModelOverview(state.modelStats);
        renderModelFilters(state.modelStats);
        renderFeedbackTable();
    } catch (err) {
        modelOverviewGrid.innerHTML = `
            <div class="empty-state error-text">加载模型统计失败：${escapeHtml(err.message)}</div>
        `;
        feedbackTableBody.innerHTML = `
            <tr><td colspan="8" class="empty-cell error-text">加载反馈失败：${escapeHtml(err.message)}</td></tr>
        `;
    } finally {
        btnRefresh.disabled = false;
    }
}

function renderStats(stats) {
    statTotal.textContent = stats.total ?? 0;
    statCorrect.textContent = stats.correct ?? 0;
    statWrong.textContent = stats.wrong ?? 0;
    statModelsWithFeedback.textContent = stats.models_with_feedback ?? 0;
    statAccuracy.textContent = typeof stats.accuracy_by_feedback === "number"
        ? `${(stats.accuracy_by_feedback * 100).toFixed(1)}%`
        : "-";
}

function renderModelOverview(modelStats) {
    if (!modelStats.length) {
        modelOverviewGrid.innerHTML = `<div class="empty-state">暂无模型信息</div>`;
        return;
    }

    modelOverviewGrid.innerHTML = modelStats.map((item) => {
        const accuracyText = formatPercent(item.accuracy_by_feedback);
        const confidenceText = formatPercent(item.avg_confidence);
        const statusClass = item.loaded ? "is-loaded" : "is-unavailable";
        const statusText = item.loaded ? "已加载" : "未加载";
        const recentErrors = item.recent_error_examples || [];

        return `
            <article class="admin-model-card" data-model-card="${escapeHtml(item.model_id)}">
                <div class="admin-model-card-head">
                    <div>
                        <div class="admin-model-name">${escapeHtml(item.model_name || item.model_id)}</div>
                        <div class="admin-model-meta">
                            ${escapeHtml(item.member || "未标注负责人")}
                            <span class="admin-divider">|</span>
                            ${escapeHtml(item.architecture || "未知架构")}
                        </div>
                    </div>
                    <span class="admin-model-badge ${statusClass}">${statusText}</span>
                </div>

                <p class="admin-model-desc">${escapeHtml(item.description || "暂无模型描述")}</p>

                <div class="admin-model-metrics">
                    <div class="admin-metric">
                        <span>反馈数</span>
                        <strong>${item.feedback_total ?? 0}</strong>
                    </div>
                    <div class="admin-metric">
                        <span>反馈准确率</span>
                        <strong>${accuracyText}</strong>
                    </div>
                    <div class="admin-metric">
                        <span>平均置信度</span>
                        <strong>${confidenceText}</strong>
                    </div>
                    <div class="admin-metric">
                        <span>验证集准确率</span>
                        <strong>${formatPercent(item.val_acc)}</strong>
                    </div>
                </div>

                <div class="admin-model-substats">
                    <span>正确 ${item.correct ?? 0}</span>
                    <span>错误 ${item.wrong ?? 0}</span>
                    <span>最近反馈 ${escapeHtml(formatTime(item.recent_feedback_at))}</span>
                </div>

                <div class="admin-model-errors">
                    <div class="admin-errors-title">最近错误样本</div>
                    ${renderRecentErrors(recentErrors)}
                </div>
            </article>
        `;
    }).join("");
}

function renderRecentErrors(errors) {
    if (!errors.length) {
        return `<div class="admin-error-empty">暂无错误样本</div>`;
    }

    return errors.map((item) => `
        <div class="admin-error-item">
            <div class="admin-error-main">
                <span class="admin-error-label">${escapeHtml(item.predicted_label || "-")}</span>
                <span class="admin-error-arrow">→</span>
                <span>${escapeHtml(item.correct_label || "-")}</span>
            </div>
            <div class="admin-error-sub">
                <span>${escapeHtml(formatTime(item.timestamp))}</span>
                <span>${escapeHtml(item.comment || "无备注")}</span>
            </div>
        </div>
    `).join("");
}

function renderModelFilters(modelStats) {
    const buttons = [
        `<button class="btn btn-secondary ${state.selectedModelId === "all" ? "is-active" : ""}" data-model-filter="all" type="button">全部模型</button>`,
        ...modelStats.map((item) => {
            const activeClass = state.selectedModelId === item.model_id ? "is-active" : "";
            return `
                <button class="btn btn-secondary ${activeClass}" data-model-filter="${escapeHtml(item.model_id)}" type="button">
                    ${escapeHtml(item.model_name || item.model_id)}
                </button>
            `;
        }),
    ];

    modelFilterGroup.innerHTML = buttons.join("");
}

function renderFeedbackTable() {
    const records = getFilteredRecords();

    if (!records.length) {
        const text = state.selectedModelId === "all"
            ? "暂无用户反馈"
            : "当前模型暂无反馈记录";
        feedbackTableBody.innerHTML = `
            <tr><td colspan="8" class="empty-cell">${text}</td></tr>
        `;
        return;
    }

    feedbackTableBody.innerHTML = records.map((item) => {
        const feedbackClass = item.is_correct ? "tag-success" : "tag-danger";
        const feedbackText = item.is_correct ? "正确" : "有误";
        return `
            <tr>
                <td>${escapeHtml(formatTime(item.timestamp))}</td>
                <td>${escapeHtml(findModelName(item.model_id))}</td>
                <td><code>${escapeHtml(item.image_id || "-")}</code></td>
                <td>${escapeHtml(item.predicted_label || "-")}</td>
                <td>${formatPercent(item.confidence)}</td>
                <td><span class="feedback-tag ${feedbackClass}">${feedbackText}</span></td>
                <td>${escapeHtml(item.correct_label || "-")}</td>
                <td>${escapeHtml(item.comment || "-")}</td>
            </tr>
        `;
    }).join("");
}

function getFilteredRecords() {
    if (state.selectedModelId === "all") return state.records;
    return state.records.filter((item) => item.model_id === state.selectedModelId);
}

function onFilterClick(event) {
    const button = event.target.closest("[data-model-filter]");
    if (!button) return;

    state.selectedModelId = button.dataset.modelFilter || "all";
    renderModelFilters(state.modelStats);
    renderFeedbackTable();
}

function findModelName(modelId) {
    if (!modelId) return "-";
    const found = state.modelStats.find((item) => item.model_id === modelId);
    return found?.model_name || modelId;
}

function formatPercent(value) {
    if (typeof value !== "number") return "-";
    return `${(value * 100).toFixed(1)}%`;
}

function formatTime(value) {
    if (!value) return "-";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString("zh-CN", { hour12: false });
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
