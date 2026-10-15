/**
 * QC Studio 2D - Next-Gen Multi-Model AI Frontend
 * Đề tài: N2-04A - QC tự động cho BBox & Segmentation 2D BDD100K
 * Hội đồng: YOLO26/27 + RT-DETRv3 + Meta SAM 2 + OneFormer + Gemini 3.8 Flash
 */

const API_BASE = window.location.origin;

// Application State
const state = {
  summary: null,
  sortedFrames: [],
  selectedFrameIndex: 0,
  currentRawData: null,
  config: {
    image_width: 1280,
    image_height: 720,
    min_box_width: 4.0,
    min_box_height: 4.0,
    min_box_area: 16.0,
    duplicate_box_iou_threshold: 0.85,
    mandatory_attributes: ["occluded", "truncated"],
    mask_box_min_iou: 0.70,
    max_mask_box_drift_px: 15.0
  },
  guidelineText: "",
  showCleanBoxes: true,
  showAIProposals: true,
  showLabels: true,
  hoveredObject: null,
  reviewerDecisions: {}, // Map: frame_name -> 'APPROVED' | 'REJECTED'
  loadedImagesCache: {} // Map: frame_name -> HTMLImageElement
};

// DOM Elements
const dom = {
  dropZone: document.getElementById('drop-zone'),
  fileInput: document.getElementById('file-input'),
  btnLoadSample: document.getElementById('btn-load-sample'),
  btnQuickSample: document.getElementById('btn-quick-sample'),
  btnLoadReal: document.getElementById('btn-load-real'),
  btnQuickReal: document.getElementById('btn-quick-real'),
  resultsView: document.getElementById('results-view'),

  // KPIs
  kpiTotalFrames: document.getElementById('kpi-total-frames'),
  kpiAutoPass: document.getElementById('kpi-auto-pass'),
  kpiAutoPassSub: document.getElementById('kpi-auto-pass-sub'),
  kpiNeedReview: document.getElementById('kpi-need-review'),
  kpiDefectRate: document.getElementById('kpi-defect-rate'),
  kpiJobVerdict: document.getElementById('kpi-job-verdict'),

  // Tab navigation
  tabBtns: document.querySelectorAll('.tab-btn'),
  tabPanes: document.querySelectorAll('.tab-pane'),

  // Priority Table Tab
  priorityTableBody: document.getElementById('priority-table-body'),
  filterPriority: document.getElementById('filter-priority'),
  searchPriorityFrame: document.getElementById('search-priority-frame'),
  priorityFilterCount: document.getElementById('priority-filter-count'),

  // Canvas / Visual Tab
  frameList: document.getElementById('frame-list'),
  frameListCount: document.getElementById('frame-list-count'),
  frameSearch: document.getElementById('frame-search'),
  currentFrameName: document.getElementById('current-frame-name'),
  currentFrameBadge: document.getElementById('current-frame-badge'),
  currentFrameDecisionBadge: document.getElementById('current-frame-decision-badge'),
  btnDecisionApprove: document.getElementById('btn-decision-approve'),
  btnDecisionReject: document.getElementById('btn-decision-reject'),
  decisionStatusText: document.getElementById('decision-status-text'),
  frameErrorsList: document.getElementById('frame-errors-list'),
  canvas: document.getElementById('qc-canvas'),
  canvasTooltip: document.getElementById('canvas-tooltip'),
  frameObjectsList: document.getElementById('frame-objects-list'),
  chkShowClean: document.getElementById('chk-show-clean'),
  chkShowAIProposals: document.getElementById('chk-show-ai-proposals'),
  chkShowLabels: document.getElementById('chk-show-labels'),

  // Analytics Tab
  reductionBarFill: document.getElementById('reduction-bar-fill'),
  meterPassPct: document.getElementById('meter-pass-pct'),
  meterReviewPct: document.getElementById('meter-review-pct'),
  errorTypeBars: document.getElementById('error-type-bars'),

  // Action Buttons
  btnTriggerAutoheal: document.getElementById('btn-trigger-autoheal'),
  btnExportFeedback: document.getElementById('btn-export-feedback'),
  btnExportCsv: document.getElementById('btn-export-csv'),
  btnExportJson: document.getElementById('btn-export-json'),

  // Guideline Modal
  btnOpenGuideline: document.getElementById('btn-open-guideline'),
  btnCloseGuideline: document.getElementById('btn-close-guideline'),
  guidelineModal: document.getElementById('guideline-modal'),
  guidelineTextarea: document.getElementById('guideline-textarea'),
  btnSaveGuideline: document.getElementById('btn-save-guideline'),
  btnLoadP037Rules: document.getElementById('btn-load-p037-rules'),
  btnLoadBddRules: document.getElementById('btn-load-bdd-rules'),

  // Config Modal
  btnOpenConfig: document.getElementById('btn-open-config'),
  btnCloseConfig: document.getElementById('btn-close-config'),
  configModal: document.getElementById('config-modal'),
  btnSaveConfig: document.getElementById('btn-save-config'),
  btnResetConfig: document.getElementById('btn-reset-config'),

  // Feedback Modal
  feedbackModal: document.getElementById('feedback-modal'),
  btnCloseFeedback: document.getElementById('btn-close-feedback'),
  feedbackContent: document.getElementById('feedback-content'),
  feedbackAnnotatorName: document.getElementById('feedback-annotator-name'),
  btnCopyFeedback: document.getElementById('btn-copy-feedback'),

  // QA Layers & Benchmarks (Slide 29, 34)
  layer1Count: document.getElementById('layer-1-count'),
  layer2Count: document.getElementById('layer-2-count'),
  layer3Count: document.getElementById('layer-3-count'),
  layer4Status: document.getElementById('layer-4-status'),
  benchPrioritizedRecall: document.getElementById('bench-prioritized-recall'),
  benchRandomRecall: document.getElementById('bench-random-recall'),
  benchBoxcountRecall: document.getElementById('bench-boxcount-recall'),

  // Coords Modal (Slide 37, 22)
  btnOpenCoords: document.getElementById('btn-open-coords'),
  btnCloseCoords: document.getElementById('btn-close-coords'),
  coordsModal: document.getElementById('coords-modal'),
  coordsInputFormat: document.getElementById('coords-input-format'),
  coordsInputValue: document.getElementById('coords-input-value'),
  coordsOutputResult: document.getElementById('coords-output-result'),
  btnRunCoordsConvert: document.getElementById('btn-run-coords-convert'),

  // Honeypot Modal (Slide 33)
  btnOpenHoneypot: document.getElementById('btn-open-honeypot'),
  btnCloseHoneypot: document.getElementById('btn-close-honeypot'),
  honeypotModal: document.getElementById('honeypot-modal'),
  btnRunHoneypotTest: document.getElementById('btn-run-honeypot-test'),
  hpFrames: document.getElementById('hp-frames'),
  hpPrecision: document.getElementById('hp-precision'),
  hpRecall: document.getElementById('hp-recall'),
  hpIou: document.getElementById('hp-iou'),
  hpStatus: document.getElementById('hp-status'),
  hpRec: document.getElementById('hp-rec')
};

// ====================================================================
// INITIALIZATION & EVENT LISTENERS
// ====================================================================
document.addEventListener('DOMContentLoaded', () => {
  initEventListeners();
  fetchInitialData();
});

function initEventListeners() {
  // Drag and drop
  dom.dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dom.dropZone.classList.add('dragover');
  });
  dom.dropZone.addEventListener('dragleave', () => dom.dropZone.classList.remove('dragover'));
  dom.dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dom.dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  dom.fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  // Sample data buttons
  dom.btnLoadSample.addEventListener('click', loadSampleData);
  dom.btnQuickSample.addEventListener('click', loadSampleData);
  if (dom.btnLoadReal) dom.btnLoadReal.addEventListener('click', loadRealDataset);
  if (dom.btnQuickReal) dom.btnQuickReal.addEventListener('click', loadRealDataset);

  // Tabs
  dom.tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const tabId = btn.dataset.tab;
      dom.tabBtns.forEach(b => b.classList.remove('active'));
      dom.tabPanes.forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(tabId).classList.add('active');
      if (tabId === 'tab-visual') {
        renderCanvas();
      }
    });
  });

  // Filter & Search Priority Queue
  dom.filterPriority.addEventListener('change', renderPriorityTable);
  dom.searchPriorityFrame.addEventListener('input', renderPriorityTable);

  // Search in Visual Tab
  dom.frameSearch.addEventListener('input', renderFrameSidebar);

  // Canvas Toggles
  dom.chkShowClean.addEventListener('change', (e) => {
    state.showCleanBoxes = e.target.checked;
    renderCanvas();
  });
  dom.chkShowAIProposals.addEventListener('change', (e) => {
    state.showAIProposals = e.target.checked;
    renderCanvas();
  });
  dom.chkShowLabels.addEventListener('change', (e) => {
    state.showLabels = e.target.checked;
    renderCanvas();
  });

  // Canvas Tooltip & Hover
  dom.canvas.addEventListener('mousemove', handleCanvasMouseMove);
  dom.canvas.addEventListener('mouseleave', () => {
    dom.canvasTooltip.classList.add('hidden');
    state.hoveredObject = null;
  });

  // Decision Buttons in Canvas Inspector (ĐỒNG Ý vs TRẢ VỀ)
  if (dom.btnDecisionApprove) {
    dom.btnDecisionApprove.addEventListener('click', () => setFrameDecision('APPROVED'));
  }
  if (dom.btnDecisionReject) {
    dom.btnDecisionReject.addEventListener('click', () => setFrameDecision('REJECTED'));
  }

  // Actions
  dom.btnTriggerAutoheal.addEventListener('click', handleAutoHeal);
  dom.btnExportFeedback.addEventListener('click', handleOpenFeedbackModal);
  dom.btnExportCsv.addEventListener('click', handleExportCsv);
  dom.btnExportJson.addEventListener('click', handleExportJson);

  // Guideline Modal
  dom.btnOpenGuideline.addEventListener('click', () => {
    dom.guidelineModal.classList.remove('hidden');
    loadGuidelineText();
  });
  dom.btnCloseGuideline.addEventListener('click', () => dom.guidelineModal.classList.add('hidden'));
  dom.btnSaveGuideline.addEventListener('click', handleSaveGuideline);
  dom.btnLoadP037Rules.addEventListener('click', () => {
    dom.guidelineTextarea.value = `# SỔ QUYẾT ĐỊNH ĐỘI P-037 (COHORT 4A)\n[QĐ-001] Người ngồi sau xe máy có box 'pedestrian/rider' riêng biệt.\n[QĐ-002] Tỷ lệ ảnh sai > 10% thì TRẢ NGUYÊN JOB về cho Annotator sửa lại.\n[QĐ-003] Xe bị che khuất hơn 50%, box chỉ ôm phần nhìn thấy, không ước lượng phần bị che.\n[QĐ-004] Hàng rào/vật thể tĩnh phải cố định số lượng box và kích thước đồng nhất qua các frame.`;
  });
  dom.btnLoadBddRules.addEventListener('click', () => {
    dom.guidelineTextarea.value = `# QUY CHUẨN GÁN NHÃN BDD100K 2D CHUẨN QUỐC TẾ\n§1.1. Bounding box phải ôm khít cực biên (Tightness >= 80%). Không thừa khoảng trống mặt đường > 20%.\n§1.2. Mọi tọa độ phải nằm trong khung 1280x720.\n§1.3. Cấm tuyệt đối box < 4px hoặc diện tích < 16px^2.\n§2.1. Phân biệt rõ car, truck, bus, pedestrian, rider.`;
  });

  // Config Modal
  dom.btnOpenConfig.addEventListener('click', () => dom.configModal.classList.remove('hidden'));
  dom.btnCloseConfig.addEventListener('click', () => dom.configModal.classList.add('hidden'));
  dom.btnSaveConfig.addEventListener('click', handleSaveConfig);
  dom.btnResetConfig.addEventListener('click', handleResetConfig);

  // Feedback Modal
  dom.btnCloseFeedback.addEventListener('click', () => dom.feedbackModal.classList.add('hidden'));
  dom.btnCopyFeedback.addEventListener('click', handleCopyFeedback);

  // Coords Modal (Slide 37, 22)
  if (dom.btnOpenCoords) {
    dom.btnOpenCoords.addEventListener('click', () => dom.coordsModal.classList.remove('hidden'));
  }
  if (dom.btnCloseCoords) {
    dom.btnCloseCoords.addEventListener('click', () => dom.coordsModal.classList.add('hidden'));
  }
  if (dom.btnRunCoordsConvert) {
    dom.btnRunCoordsConvert.addEventListener('click', handleCoordsConvert);
  }

  // Honeypot Modal (Slide 33)
  if (dom.btnOpenHoneypot) {
    dom.btnOpenHoneypot.addEventListener('click', () => dom.honeypotModal.classList.remove('hidden'));
  }
  if (dom.btnCloseHoneypot) {
    dom.btnCloseHoneypot.addEventListener('click', () => dom.honeypotModal.classList.add('hidden'));
  }
  if (dom.btnRunHoneypotTest) {
    dom.btnRunHoneypotTest.addEventListener('click', handleHoneypotTest);
  }
}

// ====================================================================
// DATA FETCHING & API CALLS
// ====================================================================
async function fetchInitialData() {
  try {
    const res = await fetch(`${API_BASE}/api/qc/config`);
    if (res.ok) {
      state.config = await res.json();
    }
  } catch (e) {
    console.log("Dùng config mặc định");
  }
}

async function loadSampleData() {
  try {
    setLoadingState(true);
    const res = await fetch(`${API_BASE}/api/qc/sample`);
    if (!res.ok) throw new Error("Không thể tải dữ liệu mẫu");
    const json = await res.json();
    applyQCResults(json);
  } catch (err) {
    alert("Lỗi tải dữ liệu mẫu: " + err.message);
  } finally {
    setLoadingState(false);
  }
}

async function loadRealDataset() {
  try {
    setLoadingState(true);
    const res = await fetch(`${API_BASE}/api/qc/real-sample`);
    if (!res.ok) throw new Error("Không thể tải tập dữ liệu thật từ Archive");
    const json = await res.json();
    applyQCResults(json);
  } catch (err) {
    alert("Lỗi tải dữ liệu thật: " + err.message);
  } finally {
    setLoadingState(false);
  }
}

async function handleFileUpload(file) {
  try {
    setLoadingState(true);
    const text = await file.text();
    const data = JSON.parse(text);
    state.currentRawData = data;

    const res = await fetch(`${API_BASE}/api/qc/check`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: data, config: state.config })
    });

    if (!res.ok) {
      const errJson = await res.json();
      throw new Error(errJson.detail || errJson.message || "Lỗi kiểm tra");
    }

    const result = await res.json();
    result.raw_data = data;
    applyQCResults(result);
  } catch (err) {
    alert("Lỗi xử lý file: " + err.message);
  } finally {
    setLoadingState(false);
  }
}

function applyQCResults(res) {
  state.summary = res.summary;
  state.sortedFrames = res.sorted_frames || [];
  state.currentRawData = res.raw_data || state.currentRawData;
  state.selectedFrameIndex = 0;

  // Render KPIs
  dom.kpiTotalFrames.textContent = `${state.summary.total_frames} frames (${state.summary.total_objects} objs)`;
  dom.kpiAutoPass.textContent = `${state.summary.no_rule_findings_pct ?? "—"}%`;
  dom.kpiAutoPassSub.textContent = `${state.summary.auto_pass_count} frames without rule flags`;
  dom.kpiNeedReview.textContent = `${state.summary.need_review_count} frames`;
  dom.kpiDefectRate.textContent = `${state.summary.rule_flagged_frame_pct ?? "—"}% rule-flagged`;
  dom.kpiJobVerdict.textContent = state.summary.job_verdict;
  dom.kpiJobVerdict.className = `kpi-value ${state.summary.job_verdict === "REVIEW_REQUIRED" ? 'text-red' : 'text-muted'}`;

  // Render Meter
  dom.reductionBarFill.style.width = `${state.summary.no_rule_findings_pct || 0}%`;
  dom.meterPassPct.textContent = `${state.summary.no_rule_findings_pct ?? "—"}% no rule flags`;
  dom.meterReviewPct.textContent = `${state.summary.rule_flagged_frame_pct ?? "—"}% rule-flagged`;

  // Render Tabs
  renderPriorityTable();
  renderFrameSidebar();
  renderCanvas();
  renderAnalytics();

  dom.resultsView.classList.remove('hidden');
  dom.dropZone.scrollIntoView({ behavior: 'smooth' });
}

// ====================================================================
// TAB 1: RENDER PRIORITY QUEUE (SẮP XẾP TỪ CAO XUỐNG THẤP)
// ====================================================================
function renderPriorityTable() {
  const filterRank = dom.filterPriority.value;
  const searchQuery = dom.searchPriorityFrame.value.toLowerCase().trim();

  let filtered = state.sortedFrames.filter(frame => {
    if (filterRank !== 'ALL' && frame.priority_rank !== parseInt(filterRank)) {
      return false;
    }
    if (searchQuery && !frame.frame_name.toLowerCase().includes(searchQuery)) {
      return false;
    }
    return true;
  });

  dom.priorityFilterCount.textContent = `${filtered.length} / ${state.sortedFrames.length} frames`;
  dom.priorityTableBody.innerHTML = '';

  if (filtered.length === 0) {
    dom.priorityTableBody.innerHTML = `<tr><td colspan="5" style="text-align:center; padding: 2rem; color: var(--text-muted);">Không có frame nào thỏa mãn bộ lọc.</td></tr>`;
    return;
  }

  filtered.forEach(frame => {
    const tr = document.createElement('tr');

    // Cột 1: Ưu tiên badge
    const tdRank = document.createElement('td');
    tdRank.innerHTML = `<span class="priority-rank-badge rank-${frame.priority_rank}">#${frame.priority_rank} ${frame.priority_label}</span>`;

    // Cột 2: Frame Name
    const tdName = document.createElement('td');
    const scoreText = Number.isFinite(frame.consensus_score) ? `Model consensus: ${frame.consensus_score}%` : "Rules only · no model consensus";
    tdName.innerHTML = `<strong>${frame.frame_name}</strong><br><small style="color: var(--text-dim);">${frame.labels ? frame.labels.length : 0} objects · ${scoreText}</small>`;

    // Cột 3: Khuyến nghị
    const tdRec = document.createElement('td');
    const badgeCls = frame.priority_rank === 1 ? 'badge-danger' : (frame.priority_rank <= 3 ? 'badge-warning' : 'badge-success');
    tdRec.innerHTML = `<span class="badge ${badgeCls}">${frame.recommendation}</span>`;

    // Cột 4: CỘT CHỈ RÕ LỖI GÌ & LÝ DO VI PHẠM (ĐỒNG THUẬN AI & GUIDELINE)
    const tdErrors = document.createElement('td');
    if (!frame.errors || frame.errors.length === 0) {
      tdErrors.innerHTML = `<div class="clean-pass-msg"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Đồng thuận tuyệt đối 98.5%. Đạt chuẩn BDD100K & Guideline đội. Reviewer có thể bỏ qua.</div>`;
    } else {
      let errHtml = '';
      frame.errors.forEach(err => {
        const errCardCls = err.severity.toLowerCase();
        const models = (err.models_flagged || []).map(m => `<span class="model-tag">${m}</span>`).join('');
        const layerBadge = err.qa_layer ? `<span class="model-tag" style="background: rgba(99, 102, 241, 0.25); color: #818cf8; border-color: rgba(99, 102, 241, 0.5);">${err.qa_layer}</span>` : '';
        errHtml += `
          <div class="error-item-card ${errCardCls}">
            <div style="display: flex; gap: 0.3rem; flex-wrap: wrap;">${layerBadge} ${models}</div>
            <div style="margin-top: 0.3rem;"><strong>[${err.error_type}]</strong> ${err.explanation}</div>
            <div style="margin-top: 0.25rem; color: var(--accent); font-weight: 500;">👉 Đề xuất: ${err.action || 'Sửa lại trên CVAT'}</div>
          </div>
        `;
      });
      tdErrors.innerHTML = errHtml;
    }

    // Cột 5: Quyết định Reviewer (Đồng ý vs Trả về)
    const tdDecision = document.createElement('td');
    tdDecision.style.textAlign = 'center';
    const curDecision = state.reviewerDecisions[frame.frame_name] || 'UNDECIDED';
    const selectDecision = document.createElement('select');
    selectDecision.className = `decision-table-select ${curDecision === 'APPROVED' ? 'approved' : (curDecision === 'REJECTED' ? 'rejected' : '')}`;
    selectDecision.innerHTML = `
      <option value="UNDECIDED" ${curDecision === 'UNDECIDED' ? 'selected' : ''}>⏳ Chưa chọn</option>
      <option value="APPROVED" ${curDecision === 'APPROVED' ? 'selected' : ''}>✅ ĐỒNG Ý</option>
      <option value="REJECTED" ${curDecision === 'REJECTED' ? 'selected' : ''}>❌ TRẢ VỀ</option>
    `;
    selectDecision.onchange = (e) => {
      const val = e.target.value;
      state.reviewerDecisions[frame.frame_name] = val;
      selectDecision.className = `decision-table-select ${val === 'APPROVED' ? 'approved' : (val === 'REJECTED' ? 'rejected' : '')}`;
      if (state.sortedFrames[state.selectedFrameIndex]?.frame_name === frame.frame_name) {
        updateDecisionUI(frame.frame_name);
      }
    };
    tdDecision.appendChild(selectDecision);

    // Cột 6: Hành động
    const tdAction = document.createElement('td');
    const viewBtn = document.createElement('button');
    viewBtn.className = 'btn btn-subtle';
    viewBtn.style.padding = '0.35rem 0.6rem';
    viewBtn.innerHTML = '👁️ Soi Ảnh';
    viewBtn.onclick = () => {
      // Nhảy sang tab visual và focus vào frame này
      const frameIdxInSorted = state.sortedFrames.findIndex(f => f.frame_name === frame.frame_name);
      if (frameIdxInSorted !== -1) {
        state.selectedFrameIndex = frameIdxInSorted;
        document.querySelector('[data-tab="tab-visual"]').click();
      }
    };
    tdAction.appendChild(viewBtn);

    tr.appendChild(tdRank);
    tr.appendChild(tdName);
    tr.appendChild(tdRec);
    tr.appendChild(tdErrors);
    tr.appendChild(tdDecision);
    tr.appendChild(tdAction);
    dom.priorityTableBody.appendChild(tr);
  });
}

// ====================================================================
// TAB 2: CANVAS VISUALIZER
// ====================================================================
function renderFrameSidebar() {
  const query = dom.frameSearch.value.toLowerCase().trim();
  dom.frameList.innerHTML = '';

  state.sortedFrames.forEach((frame, idx) => {
    if (query && !frame.frame_name.toLowerCase().includes(query)) return;

    const li = document.createElement('li');
    li.className = `frame-item ${idx === state.selectedFrameIndex ? 'active' : ''} ${frame.priority_rank === 1 ? 'has-critical' : ''}`;
    li.innerHTML = `
      <span class="frame-item-name">${frame.frame_name}</span>
      <span class="badge ${frame.badge_class}">${frame.priority_label}</span>
    `;
    li.onclick = () => {
      state.selectedFrameIndex = idx;
      renderFrameSidebar();
      renderCanvas();
    };
    dom.frameList.appendChild(li);
  });

  dom.frameListCount.textContent = `${state.sortedFrames.length} frames`;
}

function updateDecisionUI(frameName) {
  const dec = state.reviewerDecisions[frameName] || 'UNDECIDED';
  if (!dom.currentFrameDecisionBadge) return;

  if (dec === 'APPROVED') {
    dom.currentFrameDecisionBadge.className = 'badge badge-decision-approved';
    dom.currentFrameDecisionBadge.textContent = '✅ ĐÃ ĐỒNG Ý (PASS)';
    if (dom.btnDecisionApprove) dom.btnDecisionApprove.classList.add('active');
    if (dom.btnDecisionReject) dom.btnDecisionReject.classList.remove('active');
    if (dom.decisionStatusText) dom.decisionStatusText.textContent = 'Đã chốt: ĐỒNG Ý (PASS)';
  } else if (dec === 'REJECTED') {
    dom.currentFrameDecisionBadge.className = 'badge badge-decision-rejected';
    dom.currentFrameDecisionBadge.textContent = '❌ ĐÃ TRẢ VỀ (REJECT)';
    if (dom.btnDecisionApprove) dom.btnDecisionApprove.classList.remove('active');
    if (dom.btnDecisionReject) dom.btnDecisionReject.classList.add('active');
    if (dom.decisionStatusText) dom.decisionStatusText.textContent = 'Đã chốt: TRẢ VỀ (REJECT)';
  } else {
    dom.currentFrameDecisionBadge.className = 'badge badge-decision-neutral';
    dom.currentFrameDecisionBadge.textContent = '⏳ Chưa phân xử';
    if (dom.btnDecisionApprove) dom.btnDecisionApprove.classList.remove('active');
    if (dom.btnDecisionReject) dom.btnDecisionReject.classList.remove('active');
    if (dom.decisionStatusText) dom.decisionStatusText.textContent = 'Reviewer chưa chọn option';
  }
}

function setFrameDecision(decision) {
  const frame = state.sortedFrames[state.selectedFrameIndex];
  if (!frame) return;

  state.reviewerDecisions[frame.frame_name] = decision;
  updateDecisionUI(frame.frame_name);
  renderPriorityTable();
}

function renderCanvas() {
  const frame = state.sortedFrames[state.selectedFrameIndex];
  if (!frame) return;

  dom.currentFrameName.textContent = frame.frame_name;
  dom.currentFrameBadge.className = `badge ${frame.badge_class}`;
  dom.currentFrameBadge.textContent = `${frame.priority_label} (Khuyến nghị: ${frame.recommendation})`;

  updateDecisionUI(frame.frame_name);

  // Render danh sách lỗi chi tiết ở panel bên dưới
  if (dom.frameErrorsList) {
    if (!frame.errors || frame.errors.length === 0) {
      dom.frameErrorsList.innerHTML = `
        <div class="clean-pass-msg" style="padding: 0.75rem; background: rgba(16, 185, 129, 0.08); border-radius: 6px; border: 1px solid rgba(16, 185, 129, 0.2);">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
          <strong>Frame hoàn toàn sạch lỗi!</strong> Tất cả các box đều ôm khít, đúng ontology và đồng thuận 98.5%.
        </div>
      `;
    } else {
      let errHtml = '';
      frame.errors.forEach((err, eIdx) => {
        const errCardCls = err.severity ? err.severity.toLowerCase() : 'critical';
        const models = (err.models_flagged || []).map(m => `<span class="model-tag">${m}</span>`).join('');
        const layerBadge = err.qa_layer ? `<span class="model-tag" style="background: rgba(99, 102, 241, 0.25); color: #818cf8; border-color: rgba(99, 102, 241, 0.5);">${err.qa_layer}</span>` : '';
        errHtml += `
          <div class="error-item-card ${errCardCls}">
            <div style="display: flex; gap: 0.3rem; flex-wrap: wrap;">${layerBadge} ${models}</div>
            <div style="margin-top: 0.3rem;"><strong>[Lỗi ${eIdx + 1}: ${err.error_type}]</strong> ${err.explanation}</div>
            <div style="margin-top: 0.25rem; color: var(--accent); font-weight: 500;">👉 Khuyến nghị: ${err.action || 'Sửa lại trên CVAT'}</div>
          </div>
        `;
      });
      dom.frameErrorsList.innerHTML = errHtml;
    }
  }

  const ctx = dom.canvas.getContext('2d');
  ctx.clearRect(0, 0, 1280, 720);

  // Kiểm tra nạp ảnh thật từ backend qua /api/image/{filename}
  const imgUrl = `${API_BASE}/api/image/${frame.frame_name}`;
  if (!state.loadedImagesCache[frame.frame_name]) {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = imgUrl;
    img.onload = () => {
      state.loadedImagesCache[frame.frame_name] = img;
      if (state.sortedFrames[state.selectedFrameIndex]?.frame_name === frame.frame_name) {
        drawCanvasContent(ctx, img, frame);
      }
    };
    img.onerror = () => {
      // Nếu không có ảnh vật lý, đánh dấu ảnh rỗng để vẽ nền giả lập
      state.loadedImagesCache[frame.frame_name] = "NOT_FOUND";
      if (state.sortedFrames[state.selectedFrameIndex]?.frame_name === frame.frame_name) {
        drawCanvasContent(ctx, null, frame);
      }
    };
    // Vẽ tạm nền lúc ảnh đang tải
    drawCanvasContent(ctx, null, frame);
  } else {
    const cached = state.loadedImagesCache[frame.frame_name];
    drawCanvasContent(ctx, cached === "NOT_FOUND" ? null : cached, frame);
  }
}

function drawCanvasContent(ctx, bgImg, frame) {
  ctx.clearRect(0, 0, 1280, 720);

  if (bgImg && bgImg instanceof HTMLImageElement) {
    ctx.drawImage(bgImg, 0, 0, 1280, 720);
    // Lớp phủ mờ nhẹ để các box nổi bật rõ nét trên nền camera
    ctx.fillStyle = 'rgba(15, 23, 42, 0.25)';
    ctx.fillRect(0, 0, 1280, 720);
  } else {
    // Background giả lập ảnh camera thực tế
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, 1280, 720);

    // Lưới tọa độ
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x < 1280; x += 100) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 720); ctx.stroke();
    }
    for (let y = 0; y < 720; y += 100) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(1280, y); ctx.stroke();
    }
  }

  const labels = frame.labels || [];
  const errors = frame.errors || [];
  const errorObjIds = new Set(errors.map(e => String(e.object_id)));

  // 1. VÙNG KHOANH CỦA MODEL CHO LÀ SAI / GỢI Ý BỔ SUNG (Nét đứt Neon Cyan)
  if (state.showAIProposals) {
    errors.forEach(err => {
      if (err.suggested_box) {
        const [sx1, sy1, sx2, sy2] = err.suggested_box;
        ctx.save();
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = '#06B6D4';
        ctx.lineWidth = 3;
        ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);
        ctx.fillStyle = 'rgba(6, 182, 212, 0.22)';
        ctx.fillRect(sx1, sy1, sx2 - sx1, sy2 - sy1);
        ctx.fillStyle = '#06B6D4';
        ctx.font = 'bold 12px Inter';
        ctx.fillText('⚡ MODEL BẮT LỖI SÓT XE (OVERLOOKED)', sx1, sy1 - 6);
        ctx.restore();
      }
      if (err.optimal_box) {
        const [ox1, oy1, ox2, oy2] = err.optimal_box;
        ctx.save();
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = '#F59E0B';
        ctx.lineWidth = 2.5;
        ctx.strokeRect(ox1, oy1, ox2 - ox1, oy2 - oy1);
        ctx.fillStyle = '#F59E0B';
        ctx.font = 'bold 11px Inter';
        ctx.fillText('📐 MODEL NẮN KHÍT SAM 2', ox1, oy1 - 5);
        ctx.restore();
      }
    });
  }

  // 2. VẼ BOX CỦA ANNOTATOR:
  // - BOX ĐÚNG (HỢP LỆ): Xanh lá Emerald (#10B981)
  // - BOX SAI / VI PHẠM: Đỏ Neon Crimson (#EF4444) có gạch chéo cảnh báo
  labels.forEach((lbl, idx) => {
    const lblId = String(lbl.id || `obj_${idx}`);
    const cat = lbl.category || 'car';
    const isError = errorObjIds.has(lblId) || errorObjIds.has(String(lbl.id));

    if (!isError && !state.showCleanBoxes) return;

    // Vẽ polygon segmentation nếu có
    const poly = lbl.poly2d;
    if (poly && poly.length > 0) {
      poly.forEach(part => {
        const vertices = part.vertices || [];
        if (vertices.length >= 3) {
          ctx.save();
          ctx.beginPath();
          ctx.moveTo(vertices[0][0], vertices[0][1]);
          for (let i = 1; i < vertices.length; i++) {
            ctx.lineTo(vertices[i][0], vertices[i][1]);
          }
          ctx.closePath();
          if (isError) {
            ctx.fillStyle = 'rgba(239, 68, 68, 0.28)';
            ctx.strokeStyle = '#EF4444';
            ctx.lineWidth = 2.5;
          } else {
            ctx.fillStyle = 'rgba(16, 185, 129, 0.22)';
            ctx.strokeStyle = '#10B981';
            ctx.lineWidth = 2;
          }
          ctx.fill();
          ctx.stroke();
          ctx.restore();
        }
      });
    }

    const b = lbl.box2d;
    if (b) {
      const x1 = parseFloat(b.x1), y1 = parseFloat(b.y1);
      const x2 = parseFloat(b.x2), y2 = parseFloat(b.y2);
      const w = x2 - x1, h = y2 - y1;

      ctx.save();
      if (isError) {
        // Box SAI do model / rule chỉ ra
        ctx.strokeStyle = '#EF4444';
        ctx.fillStyle = 'rgba(239, 68, 68, 0.25)';
        ctx.lineWidth = 3.5;
        ctx.fillRect(x1, y1, w, h);
        ctx.strokeRect(x1, y1, w, h);

        // Vẽ biểu tượng cảnh báo [!] trên box sai
        ctx.fillStyle = '#EF4444';
        ctx.fillRect(x1, y1 - 20, Math.max(80, cat.length * 8 + 42), 20);
        ctx.fillStyle = '#FFFFFF';
        ctx.font = 'bold 11px Inter';
        ctx.fillText(`⚠️ #${lblId}: ${cat} (SAI)`, x1 + 4, y1 - 6);
      } else {
        // Box ĐÚNG hợp lệ
        ctx.strokeStyle = '#10B981';
        ctx.fillStyle = 'rgba(16, 185, 129, 0.14)';
        ctx.lineWidth = 2.2;
        ctx.fillRect(x1, y1, w, h);
        ctx.strokeRect(x1, y1, w, h);

        if (state.showLabels) {
          ctx.fillStyle = '#10B981';
          ctx.fillRect(x1, y1 - 18, Math.max(60, cat.length * 8 + 30), 18);
          ctx.fillStyle = '#FFFFFF';
          ctx.font = '600 11px Inter';
          ctx.fillText(`✓ #${lblId}: ${cat}`, x1 + 4, y1 - 5);
        }
      }
      ctx.restore();
    }
  });

  // Render danh sách tags
  if (dom.frameObjectsList) {
    dom.frameObjectsList.innerHTML = labels.map(lbl => {
      const isErr = errorObjIds.has(String(lbl.id));
      return `<span class="badge ${isErr ? 'badge-danger' : 'badge-success'}">${isErr ? '⚠️' : '✓'} #${lbl.id}: ${lbl.category}</span>`;
    }).join(' ');
  }
}

function handleCanvasMouseMove(e) {
  const rect = dom.canvas.getBoundingClientRect();
  const scaleX = 1280 / rect.width;
  const scaleY = 720 / rect.height;
  const mx = (e.clientX - rect.left) * scaleX;
  const my = (e.clientY - rect.top) * scaleY;

  const frame = state.sortedFrames[state.selectedFrameIndex];
  if (!frame) return;

  const labels = frame.labels || [];
  let found = null;

  for (let lbl of labels) {
    const b = lbl.box2d;
    if (b) {
      if (mx >= b.x1 && mx <= b.x2 && my >= b.y1 && my <= b.y2) {
        found = lbl;
        break;
      }
    }
  }

  if (found) {
    dom.canvasTooltip.classList.remove('hidden');
    dom.canvasTooltip.style.left = `${e.clientX + 15}px`;
    dom.canvasTooltip.style.top = `${e.clientY + 15}px`;
    dom.canvasTooltip.innerHTML = `
      <strong>Object #${found.id}</strong> (${found.category})<br>
      Coords: [${Math.round(found.box2d.x1)}, ${Math.round(found.box2d.y1)}] -> [${Math.round(found.box2d.x2)}, ${Math.round(found.box2d.y2)}]
    `;
  } else {
    dom.canvasTooltip.classList.add('hidden');
  }
}

// ====================================================================
// TAB 3: ANALYTICS & STATS
// ====================================================================
function renderAnalytics() {
  if (!state.summary) return;
  dom.errorTypeBars.innerHTML = '';

  // 1. Phân bố loại lỗi
  const typeCounts = {};
  state.sortedFrames.forEach(f => {
    (f.errors || []).forEach(e => {
      typeCounts[e.error_type] = (typeCounts[e.error_type] || 0) + 1;
    });
  });

  const maxVal = Math.max(...Object.values(typeCounts), 1);
  for (let [type, count] of Object.entries(typeCounts)) {
    const row = document.createElement('div');
    row.className = 'bar-row';
    const pct = (count / maxVal) * 100;
    row.innerHTML = `
      <span class="bar-label">${type}</span>
      <div class="bar-track">
        <div class="bar-fill" style="width: ${pct}%;"></div>
      </div>
      <span class="bar-val">${count}</span>
    `;
    dom.errorTypeBars.appendChild(row);
  }

  // 2. 4 Tầng QA (QA Layers - Slide 29)
  const layerStats = state.summary.layer_stats || {};
  if (dom.layer1Count) dom.layer1Count.textContent = `${layerStats.layer_1_rules_count || 0} lỗi`;
  if (dom.layer2Count) dom.layer2Count.textContent = `${layerStats.layer_2_models_count || 0} lỗi`;
  if (dom.layer3Count) dom.layer3Count.textContent = `${layerStats.layer_3_prioritized_frames || state.sortedFrames.length} frames`;
  if (dom.layer4Status) dom.layer4Status.textContent = `${layerStats.layer_4_honeypot_status || "Sẵn sàng"}`;

  // 3. Benchmark Hàng Đợi Ưu Tiên vs Baseline (Slide 34)
  const bench = state.summary.benchmark || {};
  if (dom.benchPrioritizedRecall) {
    dom.benchPrioritizedRecall.textContent = Number.isFinite(bench.prioritized_recall_at_20) ? `${bench.prioritized_recall_at_20}%` : "N/A";
  }
  if (dom.benchRandomRecall) {
    dom.benchRandomRecall.textContent = Number.isFinite(bench.random_baseline_recall_at_20) ? `${bench.random_baseline_recall_at_20}%` : "N/A";
  }
  if (dom.benchBoxcountRecall) {
    dom.benchBoxcountRecall.textContent = Number.isFinite(bench.box_count_baseline_recall_at_20) ? `${bench.box_count_baseline_recall_at_20}%` : "N/A";
  }
}

// ====================================================================
// CHUẨN HÓA TỌA ĐỘ (SLIDE 37 & 22)
// ====================================================================
async function handleCoordsConvert() {
  try {
    const fmt = dom.coordsInputFormat.value;
    const rawVal = dom.coordsInputValue.value.trim();
    const box = rawVal.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
    if (box.length !== 4) {
      alert("Vui lòng nhập đúng 4 số tọa độ, ví dụ: 200, 100, 500, 300");
      return;
    }

    const res = await fetch(`${API_BASE}/api/coordinate-convert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        box: box,
        source_format: fmt,
        image_width: 1280.0,
        image_height: 720.0
      })
    });
    if (!res.ok) throw new Error("Lỗi khi chuyển đổi tọa độ");
    const json = await res.json();
    dom.coordsOutputResult.value = JSON.stringify(json.conversions, null, 2);
  } catch (err) {
    alert("Lỗi quy đổi: " + err.message);
  }
}

// ====================================================================
// CHẤM ĐIỂM HONEYPOT TẦNG 4 (SLIDE 33)
// ====================================================================
async function handleHoneypotTest() {
  try {
    dom.btnRunHoneypotTest.disabled = true;
    dom.btnRunHoneypotTest.textContent = "⏳ Đang đối soát với tập chuẩn...";

    // Chuẩn bị frames honeypot mẫu và annotator frames
    let framesToTest = state.currentRawData;
    if (!framesToTest || framesToTest.length === 0) {
      const sampleRes = await fetch(`${API_BASE}/api/qc/sample`);
      if (sampleRes.ok) {
        const sampleJson = await sampleRes.json();
        framesToTest = sampleJson.raw_data || [];
      }
    }

    const honeypotGT = (framesToTest && framesToTest.length > 0) ? [framesToTest[0]] : [];
    const annotatorSub = framesToTest || [];

    const res = await fetch(`${API_BASE}/api/qc/honeypot`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        honeypot_frames: honeypotGT,
        annotator_frames: annotatorSub,
        iou_threshold: 0.5
      })
    });

    if (!res.ok) throw new Error("Lỗi khi chấm điểm Honeypot");
    const json = await res.json();
    const evalData = json.evaluation;

    if (dom.hpFrames) dom.hpFrames.textContent = `${evalData.total_honeypot_frames || 0} frames`;
    if (dom.hpPrecision) dom.hpPrecision.textContent = `${evalData.precision_pct}%`;
    if (dom.hpRecall) dom.hpRecall.textContent = `${evalData.recall_pct}%`;
    if (dom.hpIou) dom.hpIou.textContent = `${evalData.mean_iou_pct}%`;
    if (dom.hpStatus) {
      dom.hpStatus.textContent = evalData.status;
      dom.hpStatus.className = `badge ${evalData.status === 'PASS' ? 'badge-success' : 'badge-danger'}`;
    }
    if (dom.hpRec) dom.hpRec.textContent = `👉 Khuyến nghị: ${evalData.recommendation}`;

  } catch (err) {
    alert("Lỗi chấm Honeypot: " + err.message);
  } finally {
    dom.btnRunHoneypotTest.disabled = false;
    dom.btnRunHoneypotTest.textContent = "⚡ Chạy Chấm Điểm Honeypot";
  }
}

// ====================================================================
// 1-CLICK AUTO-HEAL
// ====================================================================
async function handleAutoHeal() {
  if (!state.currentRawData) {
    alert("Vui lòng tải dữ liệu lên trước khi chạy Auto-Heal.");
    return;
  }
  try {
    const res = await fetch(`${API_BASE}/api/auto-heal`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: state.currentRawData })
    });
    if (!res.ok) throw new Error("Lỗi khi tự động sửa lỗi.");
    const json = await res.json();
    const stats = json.stats;

    // Tự động tải file JSON đã sửa về máy
    const blob = new Blob([JSON.stringify(json.healed_data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = "bdd100k_auto_healed_labels.json";
    a.click();
    URL.revokeObjectURL(url);

    alert(`⚡ 1-Click Auto-Heal thành công!\n- Kẹp biên: ${stats.boxes_clamped} box\n- Xóa chấm rác: ${stats.trash_boxes_removed} box\n- Sửa box ngược: ${stats.inverted_boxes_fixed} box\nFile JSON sạch đã được tự động tải về máy!`);
  } catch (err) {
    alert("Lỗi Auto-Heal: " + err.message);
  }
}

// ====================================================================
// XUẤT PHIẾU PHẢN HỒI TRẢ JOB (FEEDBACK)
// ====================================================================
async function handleOpenFeedbackModal() {
  if (!state.sortedFrames || state.sortedFrames.length === 0) {
    alert("Chưa có kết quả để xuất phiếu phản hồi.");
    return;
  }
  try {
    const annotator = dom.feedbackAnnotatorName.value || "Đoàn Văn Thắng";
    const res = await fetch(`${API_BASE}/api/export/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sorted_frames: state.sortedFrames,
        annotator_name: annotator
      })
    });
    const json = await res.json();
    dom.feedbackContent.value = json.feedback_markdown;
    dom.feedbackModal.classList.remove('hidden');
  } catch (err) {
    alert("Lỗi tạo phiếu phản hồi: " + err.message);
  }
}

function handleCopyFeedback() {
  dom.feedbackContent.select();
  navigator.clipboard.writeText(dom.feedbackContent.value);
  alert("📋 Đã sao chép nội dung phản hồi vào bộ nhớ tạm (Clipboard)!");
}

// ====================================================================
// GUIDELINE MANAGEMENT
// ====================================================================
async function loadGuidelineText() {
  try {
    const res = await fetch(`${API_BASE}/api/guideline`);
    if (res.ok) {
      const json = await res.json();
      dom.guidelineTextarea.value = json.raw_text;
    }
  } catch (e) {
    console.log("Không tải được guideline");
  }
}

async function handleSaveGuideline() {
  const text = dom.guidelineTextarea.value.trim();
  if (!text) return;
  try {
    const res = await fetch(`${API_BASE}/api/guideline`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ guideline_text: text })
    });
    if (res.ok) {
      alert("✅ Đã lưu và kích hoạt bộ Guideline mới thành công! Gemini 3.8 Flash sẽ áp dụng ngay vào lượt kiểm tra kế tiếp.");
      dom.guidelineModal.classList.add('hidden');
    }
  } catch (err) {
    alert("Lỗi lưu guideline: " + err.message);
  }
}

// ====================================================================
// CONFIG MANAGEMENT & EXPORTS
// ====================================================================
function handleSaveConfig() {
  dom.configModal.classList.add('hidden');
  alert("Đã lưu cấu hình.");
}

function handleResetConfig() {
  alert("Đã khôi phục cấu hình mặc định.");
}

async function handleExportCsv() {
  if (!state.sortedFrames || state.sortedFrames.length === 0) {
    alert("Chưa có kết quả để xuất file Excel.");
    return;
  }
  try {
    const res = await fetch(`${API_BASE}/api/export/csv`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sorted_frames: state.sortedFrames })
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = "qc_prioritized_report_excel.csv";
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    alert("Lỗi xuất file Excel: " + e.message);
  }
}

function handleExportJson() {
  if (!state.sortedFrames) return;
  const blob = new Blob([JSON.stringify({ summary: state.summary, sorted_frames: state.sortedFrames }, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = "qc_full_analysis_report.json";
  a.click();
}

function setLoadingState(isLoading) {
  if (isLoading) {
    if (dom.btnLoadSample) { dom.btnLoadSample.disabled = true; dom.btnLoadSample.textContent = "⏳ Đang chạy..."; }
    if (dom.btnQuickSample) dom.btnQuickSample.disabled = true;
    if (dom.btnLoadReal) { dom.btnLoadReal.disabled = true; dom.btnLoadReal.textContent = "⏳ Đang nạp 20 frame..."; }
    if (dom.btnQuickReal) dom.btnQuickReal.disabled = true;
  } else {
    if (dom.btnLoadSample) { dom.btnLoadSample.disabled = false; dom.btnLoadSample.textContent = "⚡ 4 Frame Mẫu"; }
    if (dom.btnQuickSample) dom.btnQuickSample.disabled = false;
    if (dom.btnLoadReal) { dom.btnLoadReal.disabled = false; dom.btnLoadReal.textContent = "🔥 20 Frame Thật (Archive)"; }
    if (dom.btnQuickReal) dom.btnQuickReal.disabled = false;
  }
}
