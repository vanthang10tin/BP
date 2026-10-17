/**
 * QC Studio 2D - Auto-QC Frontend
 * Đề tài: N2-04A - QC tự động cho BBox & Segmentation 2D BDD100K
 * Kiến trúc: Rule Engine Hình học/Ontology BDD100K + Local YOLO26 Detector + Gemini VLM
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
  showErrorBoxes: true,
  showAIProposals: true,
  showGeminiProposals: true,
  showLabels: true,
  selectedIssueIndex: null,
  selectedObjectId: null,
  highlightedBox: null, // [x1, y1, x2, y2]
  hoveredObject: null,
  modelsInfo: null,
  reviewerDecisions: {}, // Map: frame_name -> 'APPROVED' | 'REJECTED'
  loadedImagesCache: {} // Map: frame_name -> HTMLImageElement
};
let selectedHoneypotReference = null;

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
  btnPrevFrame: document.getElementById('btn-prev-frame'),
  btnNextFrame: document.getElementById('btn-next-frame'),
  currentFrameName: document.getElementById('current-frame-name'),
  currentFrameBadge: document.getElementById('current-frame-badge'),
  currentFrameDecisionBadge: document.getElementById('current-frame-decision-badge'),
  btnDecisionApprove: document.getElementById('btn-decision-approve'),
  btnDecisionReject: document.getElementById('btn-decision-reject'),
  decisionStatusText: document.getElementById('decision-status-text'),
  frameErrorsCount: document.getElementById('frame-errors-count'),
  frameErrorsList: document.getElementById('frame-errors-list'),
  canvasContainer: document.getElementById('canvas-container'),
  canvas: document.getElementById('qc-canvas'),
  canvasTooltip: document.getElementById('canvas-tooltip'),
  frameObjectsCount: document.getElementById('frame-objects-count'),
  frameObjectsList: document.getElementById('frame-objects-list'),

  // Layer Toggle Pills (Vừa là Legend vừa Bật/Tắt 1 chạm)
  pillToggleClean: document.getElementById('pill-toggle-clean'),
  pillToggleErrors: document.getElementById('pill-toggle-errors'),
  pillToggleAi: document.getElementById('pill-toggle-ai'),
  pillToggleGemini: document.getElementById('pill-toggle-gemini'),
  pillToggleLabels: document.getElementById('pill-toggle-labels'),

  geminiStatusBadge: document.getElementById('gemini-status-badge'),
  btnInspectGemini: document.getElementById('btn-inspect-gemini'),

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
  hpRec: document.getElementById('hp-rec'),
  honeypotGoldFile: document.getElementById('honeypot-gold-file')
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

  // Layer Pills (Nhãn Chuẩn, Box Vi Phạm, Gợi ý Model, Gemini 3 Bên, Tên ID)
  initLayerPills();

  // Frame Navigator (Arrow buttons [◀] và [▶])
  if (dom.btnPrevFrame) {
    dom.btnPrevFrame.addEventListener('click', () => selectFrame(state.selectedFrameIndex - 1));
  }
  if (dom.btnNextFrame) {
    dom.btnNextFrame.addEventListener('click', () => selectFrame(state.selectedFrameIndex + 1));
  }

  // Keyboard navigation for frames ([ or Left, ] or Right)
  document.addEventListener('keydown', handleGlobalKeydown);

  if (dom.btnInspectGemini) {
    dom.btnInspectGemini.addEventListener('click', handleGeminiInspectCurrentFrame);
  }
  if (dom.geminiStatusBadge) {
    dom.geminiStatusBadge.addEventListener('click', handleGeminiKeyPrompt);
  }

  // Canvas Click (Nhảy tới issue hoặc đối tượng tương ứng)
  dom.canvas.addEventListener('click', handleCanvasClick);

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
    dom.guidelineTextarea.value = `# Ví dụ guideline minh họa — không phải chính sách hoặc tiêu chuẩn chính thức\n# Thay nội dung này bằng guideline đã được chủ dự án phê duyệt.\n- Reviewer xác minh mọi candidate với ảnh gốc và quy tắc của dự án.\n- Không tự động chấp nhận hoặc từ chối job chỉ dựa trên cờ của QC.`;
  });
  dom.btnLoadBddRules.addEventListener('click', () => {
    dom.guidelineTextarea.value = `# Tham số QC mặc định của ứng dụng — cần xác nhận với guideline dự án\n- Box phải nằm trong kích thước ảnh thực tế.\n- Ngưỡng box nhỏ mặc định: width < 4 px, height < 4 px hoặc area < 16 px².\n- Ngưỡng box trùng cùng class mặc định: IoU >= 0.85.\n# Đây là cấu hình phần mềm, không phải tiêu chuẩn BDD100K chính thức.`;
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
  if (dom.honeypotGoldFile) {
    dom.honeypotGoldFile.addEventListener('change', async (event) => {
      selectedHoneypotReference = null;
      const file = event.target.files?.[0];
      if (!file) return;
      try {
        const parsed = JSON.parse(await file.text());
        const frames = Array.isArray(parsed) ? parsed : (parsed.frames || parsed.data || []);
        if (!Array.isArray(frames) || !frames.length) throw new Error('File không có danh sách frame.');
        selectedHoneypotReference = frames;
        if (dom.hpRec) dom.hpRec.textContent = `Đã nạp ${frames.length} frame từ ${file.name}. Chỉ tiếp tục nếu đây là tập độc lập đã adjudicate.`;
      } catch (error) {
        if (dom.hpRec) dom.hpRec.textContent = `Không đọc được tập đối chiếu: ${error.message}`;
      }
    });
  }
}

function initLayerPills() {
  const pills = [
    { el: dom.pillToggleClean, key: 'showCleanBoxes' },
    { el: dom.pillToggleErrors, key: 'showErrorBoxes' },
  ];

  pills.forEach(({ el, key }) => {
    if (!el) return;
    el.addEventListener('click', () => {
      state[key] = !state[key];
      el.classList.toggle('active', state[key]);
      renderCanvas();
    });
  });
}

function handleGlobalKeydown(e) {
  const activeEl = document.activeElement;
  const tag = activeEl ? activeEl.tagName.toLowerCase() : '';
  if (tag === 'input' || tag === 'textarea' || tag === 'select' || activeEl?.isContentEditable) {
    return;
  }

  const visualTab = document.getElementById('tab-visual');
  if (!visualTab || !visualTab.classList.contains('active')) return;

  if (e.key === 'ArrowLeft' || e.key === '[') {
    e.preventDefault();
    selectFrame(state.selectedFrameIndex - 1);
  } else if (e.key === 'ArrowRight' || e.key === ']') {
    e.preventDefault();
    selectFrame(state.selectedFrameIndex + 1);
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

  try {
    const resModels = await fetch(`${API_BASE}/api/models`);
    if (resModels.ok) {
      const data = await resModels.json();
      state.modelsInfo = data;
      updateGeminiBadgeStatus(data.gemini);
    }
  } catch (e) {
    console.warn("Không đọc được trạng thái model", e);
  }
}

function updateGeminiBadgeStatus(geminiConfig) {
  if (!dom.geminiStatusBadge) return;
  if (geminiConfig && geminiConfig.configured) {
    dom.geminiStatusBadge.style.background = 'rgba(16, 185, 129, 0.2)';
    dom.geminiStatusBadge.style.color = '#34d399';
    dom.geminiStatusBadge.style.border = '1px solid rgba(16, 185, 129, 0.5)';
    dom.geminiStatusBadge.textContent = `🤖 Gemini VLM: Sẵn sàng (${geminiConfig.model})`;
    dom.geminiStatusBadge.title = `Key máy chủ đã nạp. Model: ${geminiConfig.model}. Bấm để cập nhật key nếu cần.`;
  } else {
    dom.geminiStatusBadge.style.background = 'rgba(245, 158, 11, 0.2)';
    dom.geminiStatusBadge.style.color = '#fbbf24';
    dom.geminiStatusBadge.style.border = '1px solid rgba(245, 158, 11, 0.5)';
    dom.geminiStatusBadge.textContent = '🤖 Gemini VLM: Chưa có Key';
    dom.geminiStatusBadge.title = 'Bấm để nhập key hoặc cấu hình GEMINI_API_KEY trong .env máy chủ.';
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
    let frames = null;
    let format = "unknown";

    // 1. Dùng backend flexible parser trước (hỗ trợ .zip, .xml, .csv, .txt, .json)
    try {
      const formData = new FormData();
      formData.append("file", file);
      const parseRes = await fetch(`${API_BASE}/api/dataset/parse-flexible`, {
        method: 'POST',
        body: formData
      });
      if (parseRes.ok) {
        const parsed = await parseRes.json();
        frames = parsed.frames || [];
        format = parsed.format || "custom";
        if (parsed.images && Object.keys(parsed.images).length > 0) {
          state.customImages = Object.assign(state.customImages || {}, parsed.images);
          for (const [fname, dataUri] of Object.entries(parsed.images)) {
            const img = new Image();
            img.src = dataUri;
            state.loadedImagesCache[fname] = img;
          }
        }
      }
    } catch (e) {
      console.warn("Flexible parser fallback:", e);
    }

    // 2. Fallback sang đọc JSON nếu flexible parser chưa nạp được frames
    if (!frames || frames.length === 0) {
      const text = await file.text();
      const parsedJson = JSON.parse(text);
      frames = Array.isArray(parsedJson) ? parsedJson : (parsedJson.frames || parsedJson.data || [parsedJson]);
    }

    state.currentRawData = frames;

    // 3. Gửi frames đã chuẩn hóa tới API kiểm định QC
    const res = await fetch(`${API_BASE}/api/qc/check`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: frames, config: state.config })
    });

    if (!res.ok) {
      const errJson = await res.json();
      throw new Error(errJson.detail || errJson.message || "Lỗi kiểm tra");
    }

    const result = await res.json();
    result.raw_data = frames;
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
  dom.kpiAutoPassSub.textContent = `${state.summary.auto_pass_count} frames without findings from enabled checks`;
  dom.kpiNeedReview.textContent = `${state.summary.need_review_count} frames`;
  dom.kpiDefectRate.textContent = `${state.summary.rule_flagged_frame_pct ?? "—"}% rule-flagged`;
  dom.kpiJobVerdict.textContent = state.summary.job_verdict;
  dom.kpiJobVerdict.className = `kpi-value ${state.summary.job_verdict === "REVIEW_REQUIRED" ? 'text-red' : 'text-muted'}`;

  // Render Meter
  dom.reductionBarFill.style.width = `${state.summary.no_rule_findings_pct || 0}%`;
  dom.meterPassPct.textContent = `${state.summary.no_rule_findings_pct ?? "—"}% no candidate findings`;
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

  const fragment = document.createDocumentFragment();

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
      tdErrors.innerHTML = `<div class="clean-pass-msg"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Không có candidate từ các kiểm tra đang bật. Điều này chưa xác minh nhãn đúng.</div>`;
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
    selectDecision.setAttribute('data-frame', frame.frame_name);
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
        selectFrame(frameIdxInSorted);
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
    fragment.appendChild(tr);
  });

  dom.priorityTableBody.appendChild(fragment);
}

function updateRowDecision(frameName, decision) {
  const rowSelect = dom.priorityTableBody ? dom.priorityTableBody.querySelector(`select[data-frame="${CSS.escape(frameName)}"]`) : null;
  if (rowSelect) {
    rowSelect.value = decision;
    rowSelect.className = `decision-table-select ${decision === 'APPROVED' ? 'approved' : (decision === 'REJECTED' ? 'rejected' : '')}`;
  }
}

// ====================================================================
// TAB 2: CANVAS VISUALIZER
// ====================================================================
function renderFrameSidebar() {
  const query = dom.frameSearch.value.toLowerCase().trim();
  dom.frameList.innerHTML = '';
  const fragment = document.createDocumentFragment();

  state.sortedFrames.forEach((frame, idx) => {
    if (query && !frame.frame_name.toLowerCase().includes(query)) return;

    const li = document.createElement('li');
    li.className = `frame-item ${idx === state.selectedFrameIndex ? 'active' : ''} ${frame.priority_rank === 1 ? 'has-critical' : ''}`;
    li.setAttribute('data-frame-idx', idx);
    li.innerHTML = `
      <span class="frame-item-name">${frame.frame_name}</span>
      <span class="badge ${frame.badge_class}">${frame.priority_label}</span>
    `;
    li.onclick = () => {
      selectFrame(idx);
    };
    fragment.appendChild(li);
  });

  dom.frameList.appendChild(fragment);
  dom.frameListCount.textContent = `${state.sortedFrames.length} frames`;
}

function selectFrame(idx) {
  if (!state.sortedFrames || state.sortedFrames.length === 0) return;
  if (idx < 0) idx = 0;
  if (idx >= state.sortedFrames.length) idx = state.sortedFrames.length - 1;

  state.selectedFrameIndex = idx;
  state.selectedIssueIndex = null;
  state.selectedObjectId = null;
  state.highlightedBox = null;

  if (dom.frameList) {
    dom.frameList.querySelectorAll('.frame-item').forEach(el => {
      el.classList.toggle('active', Number(el.getAttribute('data-frame-idx')) === idx);
    });
    const activeItem = dom.frameList.querySelector(`.frame-item[data-frame-idx="${idx}"]`);
    if (activeItem) {
      activeItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  renderCanvas();
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
  updateRowDecision(frame.frame_name, decision);
}

// ====================================================================
// ĐIỀU HƯỚNG 2 CHIỀU (BOX ⟷ ISSUE FOCUS)
// ====================================================================
function focusIssue(issueIdx, shouldScroll = true) {
  const frame = state.sortedFrames[state.selectedFrameIndex];
  if (!frame) return;

  const errors = frame.errors || frame.issues || [];
  const err = errors[issueIdx];
  if (!err) return;

  state.selectedIssueIndex = issueIdx;
  state.selectedObjectId = err.object_id != null ? String(err.object_id) : null;

  // Xác định tọa độ box để vẽ autofocus spotlight reticle
  let targetBox = null;
  if (err.object_id != null) {
    const rawOid = String(err.object_id);
    const cleanOid = rawOid.replace(/^obj_/, '').replace(/^h_/, '');
    const matchedLabel = (frame.labels || []).find((l, idx) => {
      const lid = String(l.id != null ? l.id : '');
      const cleanLid = lid.replace(/^obj_/, '').replace(/^h_/, '');
      return lid === rawOid || cleanLid === cleanOid || cleanLid === rawOid || String(idx) === cleanOid || String(idx) === rawOid;
    });
    if (matchedLabel && matchedLabel.box2d) {
      targetBox = [
        parseFloat(matchedLabel.box2d.x1),
        parseFloat(matchedLabel.box2d.y1),
        parseFloat(matchedLabel.box2d.x2),
        parseFloat(matchedLabel.box2d.y2)
      ];
    }
  }

  if (!targetBox && err.suggested_box && err.suggested_box.length === 4) {
    targetBox = [...err.suggested_box];
  }
  if (!targetBox && err.optimal_box && err.optimal_box.length === 4) {
    targetBox = [...err.optimal_box];
  }
  if (!targetBox && err.box_normalized && err.box_normalized.length === 4) {
    const [px1, py1, px2, py2] = err.box_normalized;
    targetBox = [(px1 * 1280) / 1000, (py1 * 720) / 1000, (px2 * 1280) / 1000, (py2 * 720) / 1000];
  }
  if (!targetBox && (frame.gemini_proposals || []).length > 0) {
    const p = frame.gemini_proposals[0];
    if (p.box_normalized) {
      const [px1, py1, px2, py2] = p.box_normalized;
      targetBox = [(px1 * 1280) / 1000, (py1 * 720) / 1000, (px2 * 1280) / 1000, (py2 * 720) / 1000];
    }
  }

  state.highlightedBox = targetBox;

  // Highlight card trong danh sách lỗi
  if (dom.frameErrorsList) {
    dom.frameErrorsList.querySelectorAll('.error-item-card, .gemini-arbitration-card').forEach(c => c.classList.remove('active-focus'));
    const targetCard = dom.frameErrorsList.querySelector(`[data-issue-idx="${issueIdx}"]`);
    if (targetCard) {
      targetCard.classList.add('active-focus');
      if (shouldScroll) {
        targetCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }

  // Highlight badge object tương ứng nếu có
  if (dom.frameObjectsList && state.selectedObjectId) {
    const rawOid = String(state.selectedObjectId);
    const cleanOid = rawOid.replace(/^obj_/, '').replace(/^h_/, '');
    dom.frameObjectsList.querySelectorAll('.badge').forEach(b => {
      const bid = String(b.getAttribute('data-obj-id') || '');
      const cleanBid = bid.replace(/^obj_/, '').replace(/^h_/, '');
      const isMatch = bid === rawOid || cleanBid === cleanOid || cleanBid === rawOid;
      b.classList.toggle('active-focus', isMatch);
    });
  }

  // Vẽ lại canvas để render autofocus reticle
  renderCanvas(true);
}

function focusObject(objId, shouldScroll = true) {
  const frame = state.sortedFrames[state.selectedFrameIndex];
  if (!frame) return;

  const strId = String(objId);
  const cleanId = strId.replace(/^obj_/, '').replace(/^h_/, '');
  state.selectedObjectId = strId;

  // Kiểm tra xem object này có nằm trong danh sách vi phạm không
  const errors = frame.errors || frame.issues || [];
  const errIdx = errors.findIndex((err, idx) => {
    const rawOid = String(err.object_id != null ? err.object_id : '');
    const cleanOid = rawOid.replace(/^obj_/, '').replace(/^h_/, '');
    return rawOid === strId || cleanOid === cleanId || cleanOid === strId || rawOid === cleanId || String(idx) === strId;
  });

  if (errIdx >= 0) {
    focusIssue(errIdx, shouldScroll);
    return;
  }

  // Nếu không có lỗi, tìm box sạch của object
  const lbl = (frame.labels || []).find((l, idx) => {
    const lid = String(l.id != null ? l.id : '');
    const cleanLid = lid.replace(/^obj_/, '').replace(/^h_/, '');
    return lid === strId || cleanLid === cleanId || String(idx) === strId || String(idx) === cleanId;
  });
  if (lbl && lbl.box2d) {
    state.selectedIssueIndex = null;
    state.highlightedBox = [
      parseFloat(lbl.box2d.x1),
      parseFloat(lbl.box2d.y1),
      parseFloat(lbl.box2d.x2),
      parseFloat(lbl.box2d.y2)
    ];
  } else {
    state.highlightedBox = null;
  }

  // Bỏ active ở list lỗi
  if (dom.frameErrorsList) {
    dom.frameErrorsList.querySelectorAll('.error-item-card, .gemini-arbitration-card').forEach(c => c.classList.remove('active-focus'));
  }

  // Highlight badge trong danh sách đối tượng
  if (dom.frameObjectsList) {
    dom.frameObjectsList.querySelectorAll('.badge').forEach(b => {
      const bid = String(b.getAttribute('data-obj-id') || '');
      const cleanBid = bid.replace(/^obj_/, '').replace(/^h_/, '');
      const isMatch = bid === strId || cleanBid === cleanId;
      b.classList.toggle('active-focus', isMatch);
      if (isMatch && shouldScroll) {
        b.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    });
  }

  renderCanvas(true);
}

function handleCanvasClick(e) {
  const frame = state.sortedFrames[state.selectedFrameIndex];
  if (!frame) return;

  const rect = dom.canvas.getBoundingClientRect();
  const scaleX = 1280 / rect.width;
  const scaleY = 720 / rect.height;
  const mx = (e.clientX - rect.left) * scaleX;
  const my = (e.clientY - rect.top) * scaleY;

  const errors = frame.errors || frame.issues || [];
  const labels = frame.labels || [];
  const proposals = frame.gemini_proposals || [];

  function isInside(box, x, y, pad = 6) {
    if (!box || box.length < 4) return false;
    return x >= (box[0] - pad) && x <= (box[2] + pad) && y >= (box[1] - pad) && y <= (box[3] + pad);
  }

  // 1. Kiểm tra click trúng các vùng có Issue (Màu Đỏ)
  if (state.showErrorBoxes !== false) {
    // 1a. Lỗi sót xe / gợi ý của Model
    for (let i = 0; i < errors.length; i++) {
      const err = errors[i];
      const box = err.suggested_box || err.optimal_box;
      if (box && isInside(box, mx, my, 8)) {
        focusIssue(i, true);
        return;
      }
    }

    // 1b. Vùng phân xử Gemini
    for (let pIdx = 0; pIdx < proposals.length; pIdx++) {
      const prop = proposals[pIdx];
      if (prop.box_normalized && prop.box_normalized.length === 4) {
        const [px1, py1, px2, py2] = prop.box_normalized;
        const x1 = (px1 * 1280) / 1000;
        const y1 = (py1 * 720) / 1000;
        const x2 = (px2 * 1280) / 1000;
        const y2 = (py2 * 720) / 1000;
        if (isInside([x1, y1, x2, y2], mx, my, 8)) {
          const errIdx = errors.findIndex(err => err.qa_layer && err.qa_layer.includes("Gemini"));
          if (errIdx >= 0) {
            focusIssue(errIdx, true);
          } else {
            state.highlightedBox = [x1, y1, x2, y2];
            state.selectedIssueIndex = null;
            state.selectedObjectId = null;
            renderCanvas(true);
          }
          return;
        }
      }
    }
  }

  // 2. Kiểm tra click trúng Box Annotator (cả vi phạm và sạch)
  for (let lblIdx = 0; lblIdx < labels.length; lblIdx++) {
    const lbl = labels[lblIdx];
    const b = lbl.box2d;
    if (b) {
      const x1 = parseFloat(b.x1), y1 = parseFloat(b.y1);
      const x2 = parseFloat(b.x2), y2 = parseFloat(b.y2);
      if (isInside([x1, y1, x2, y2], mx, my, 8)) {
        const rawLid = String(lbl.id != null ? lbl.id : '');
        const cleanLid = rawLid.replace(/^obj_/, '').replace(/^h_/, '');
        const errIdx = errors.findIndex((err, eIdx) => {
          if (!err) return false;
          const rawOid = String(err.object_id != null ? err.object_id : '');
          const cleanOid = rawOid.replace(/^obj_/, '').replace(/^h_/, '');
          if (rawOid === rawLid || cleanOid === cleanLid || cleanOid === String(lblIdx) || rawOid === String(lblIdx)) return true;
          if (err.suggested_box && isInside(err.suggested_box, mx, my, 8)) return true;
          return false;
        });

        if (errIdx >= 0) {
          if (state.showErrorBoxes !== false) {
            focusIssue(errIdx, true);
            return;
          }
        } else {
          if (state.showCleanBoxes !== false) {
            focusObject(lbl.id != null ? lbl.id : lblIdx, true);
            return;
          }
        }
      }
    }
  }

  // Nếu click ra ngoài các box: bỏ highlight
  state.highlightedBox = null;
  state.selectedIssueIndex = null;
  state.selectedObjectId = null;
  if (dom.frameErrorsList) {
    dom.frameErrorsList.querySelectorAll('.error-item-card, .gemini-arbitration-card').forEach(el => el.classList.remove('active-focus'));
  }
  if (dom.frameObjectsList) {
    dom.frameObjectsList.querySelectorAll('.badge').forEach(el => el.classList.remove('active-focus'));
  }
  renderCanvas(true);
}

function renderCanvas(skipDomRebuild = false) {
  const frame = state.sortedFrames[state.selectedFrameIndex];
  if (!frame) return;

  dom.currentFrameName.textContent = frame.frame_name;
  dom.currentFrameBadge.className = `badge ${frame.badge_class}`;
  dom.currentFrameBadge.textContent = `${frame.priority_label} (Khuyến nghị: ${frame.recommendation})`;

  updateDecisionUI(frame.frame_name);

  const errors = frame.errors || [];
  const labels = frame.labels || [];
  const errorObjIds = new Set(errors.map(e => String(e.object_id)));

  if (dom.frameErrorsCount) dom.frameErrorsCount.textContent = errors.length;
  if (dom.frameObjectsCount) dom.frameObjectsCount.textContent = labels.length;

  if (!skipDomRebuild) {
    // Render danh sách lỗi chi tiết ở panel bên dưới
    if (dom.frameErrorsList) {
      if (errors.length === 0) {
        dom.frameErrorsList.innerHTML = `
          <div class="clean-pass-msg" style="padding: 0.75rem; background: rgba(16, 185, 129, 0.08); border-radius: 6px; border: 1px solid rgba(16, 185, 129, 0.2);">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
            <strong>Không phát hiện vi phạm rule.</strong> Các kiểm tra tự động (hình học, ontology, thuộc tính) đều đạt.
          </div>
        `;
      } else {
        let errHtml = '';
        errors.forEach((err, eIdx) => {
          const isSelected = state.selectedIssueIndex === eIdx;
          if (err.qa_layer && err.qa_layer.includes("Gemini")) {
            // Render Card Phân Xử 3 Bên Gemini VLM
            errHtml += `
              <div class="gemini-arbitration-card ${isSelected ? 'active-focus' : ''}" data-issue-idx="${eIdx}" style="cursor: pointer;">
                <div class="gemini-card-header">
                  <span class="tag-gemini">✨ TẦNG 3: GEMINI VLM PHÂN XỬ 3 BÊN</span>
                  <span class="badge badge-purple">${err.error_type}</span>
                </div>
                <div class="gemini-three-party-grid">
                  <div class="three-party-col">
                    <span class="col-label">🤖 Model (YOLO/DINO):</span>
                    <span class="col-val">${err.model_status || 'Chỉ 10 class BDD100K hoặc không phát hiện'}</span>
                  </div>
                  <div class="three-party-col">
                    <span class="col-label">🏷️ Nhãn Annotator:</span>
                    <span class="col-val">${err.label_status || 'Chưa gán nhãn hoặc xung đột'}</span>
                  </div>
                  <div class="three-party-col">
                    <span class="col-label">📜 Sổ Quyết Định:</span>
                    <span class="col-val">${err.guideline_citation || 'Quy chuẩn gán nhãn dự án'}</span>
                  </div>
                </div>
                <div class="gemini-evidence">
                  <strong>💡 Phân tích & Khuyến nghị:</strong> ${err.explanation}
                </div>
              </div>
            `;
          } else {
            const errCardCls = err.severity ? err.severity.toLowerCase() : 'critical';
            const models = (err.models_flagged || []).map(m => `<span class="model-tag">${m}</span>`).join('');
            const layerBadge = err.qa_layer ? `<span class="model-tag" style="background: rgba(99, 102, 241, 0.25); color: #818cf8; border-color: rgba(99, 102, 241, 0.5);">${err.qa_layer}</span>` : '';
            const objBadge = err.object_id != null ? `<span class="model-tag" style="background: rgba(239, 68, 68, 0.2); color: #fca5a5;">ID #${err.object_id}</span>` : '';
            errHtml += `
              <div class="error-item-card ${errCardCls} ${isSelected ? 'active-focus' : ''}" data-issue-idx="${eIdx}" style="cursor: pointer;">
                <div style="display: flex; gap: 0.3rem; flex-wrap: wrap;">${layerBadge} ${objBadge} ${models}</div>
                <div style="margin-top: 0.3rem;"><strong>[Lỗi ${eIdx + 1}: ${err.error_type}]</strong> ${err.explanation}</div>
                <div style="margin-top: 0.25rem; color: var(--accent); font-weight: 500;">👉 Khuyến nghị: ${err.action || 'Sửa lại trên CVAT'}</div>
              </div>
            `;
          }
        });
        dom.frameErrorsList.innerHTML = errHtml;

        // Gắn listener click cho từng card lỗi để focus box trên canvas
        dom.frameErrorsList.querySelectorAll('[data-issue-idx]').forEach(card => {
          card.onclick = () => {
            const idx = parseInt(card.getAttribute('data-issue-idx'), 10);
            focusIssue(idx, false);
          };
        });
      }
    }

    // Render danh sách tags
    if (dom.frameObjectsList) {
      dom.frameObjectsList.innerHTML = labels.map(lbl => {
        const isErr = errorObjIds.has(String(lbl.id));
        const isSelected = state.selectedObjectId === String(lbl.id);
        return `<span class="badge ${isErr ? 'badge-danger' : 'badge-success'} ${isSelected ? 'active-focus' : ''}" data-obj-id="${lbl.id}" style="cursor: pointer;">${isErr ? '⚠️' : '✓'} #${lbl.id}: ${lbl.category}</span>`;
      }).join(' ');

      // Gắn listener click cho từng badge object
      dom.frameObjectsList.querySelectorAll('[data-obj-id]').forEach(tag => {
        tag.onclick = () => {
          const objId = tag.getAttribute('data-obj-id');
          focusObject(objId, true);
        };
      });
    }
  }

  const ctx = dom.canvas.getContext('2d');
  ctx.clearRect(0, 0, 1280, 720);

  // Kiểm tra nạp ảnh thật từ state.customImages hoặc backend qua /api/image/{filename}
  const customImg = state.customImages && state.customImages[frame.frame_name];
  const imgUrl = customImg || `${API_BASE}/api/image/${frame.frame_name}`;
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
      state.loadedImagesCache[frame.frame_name] = "NOT_FOUND";
      if (state.sortedFrames[state.selectedFrameIndex]?.frame_name === frame.frame_name) {
        drawCanvasContent(ctx, null, frame);
      }
    };
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
    ctx.fillStyle = 'rgba(15, 23, 42, 0.25)';
    ctx.fillRect(0, 0, 1280, 720);
  } else {
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, 1280, 720);

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

  // ==================================================================
  // HIỂN THỊ TINH GỌN DUY NHẤT 2 LỚP:
  // 1. NHÃN USER HỢP LỆ (MÀU XANH LÁ #10B981)
  // 2. TẤT CẢ VÙNG CÓ ISSUE (MÀU ĐỎ #EF4444 - KẾT QUẢ CUỐI CÙNG MODEL & GEMINI)
  // ==================================================================

  // LỚP 1 & 2a: VẼ BOX VÀ SEGMENT CỦA USER (PHÂN LOẠI XANH vs ĐỎ)
  labels.forEach((lbl, idx) => {
    const lblId = String(lbl.id || `obj_${idx}`);
    const cat = lbl.category || 'car';
    const isError = errorObjIds.has(lblId) || errorObjIds.has(String(lbl.id));

    // Lọc theo layer pill toggle:
    if (isError && state.showErrorBoxes === false) return;
    if (!isError && state.showCleanBoxes === false) return;

    // 1. Vẽ polygon segmentation nếu có
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
            // Segment có Issue -> MÀU ĐỎ
            ctx.fillStyle = 'rgba(239, 68, 68, 0.28)';
            ctx.strokeStyle = '#EF4444';
            ctx.lineWidth = 2.8;
          } else {
            // Segment Hợp Lệ -> MÀU XANH LÁ
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

    // 2. Vẽ Bounding Box 2D
    const b = lbl.box2d;
    if (b) {
      const x1 = parseFloat(b.x1), y1 = parseFloat(b.y1);
      const x2 = parseFloat(b.x2), y2 = parseFloat(b.y2);
      const w = x2 - x1, h = y2 - y1;

      ctx.save();
      if (isError) {
        // Box có Issue -> MÀU ĐỎ
        ctx.strokeStyle = '#EF4444';
        ctx.fillStyle = 'rgba(239, 68, 68, 0.22)';
        ctx.lineWidth = 3;
        ctx.fillRect(x1, y1, w, h);
        ctx.strokeRect(x1, y1, w, h);

        const labelText = `⚠️ #${lblId}: ${cat} (LỖI)`;
        ctx.font = 'bold 11px Inter';
        const tw = ctx.measureText(labelText).width;
        ctx.fillStyle = '#EF4444';
        ctx.fillRect(x1, y1 - 20, Math.max(80, tw + 10), 20);
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(labelText, x1 + 4, y1 - 6);
      } else {
        // Box Hợp Lệ -> MÀU XANH LÁ
        ctx.strokeStyle = '#10B981';
        ctx.fillStyle = 'rgba(16, 185, 129, 0.12)';
        ctx.lineWidth = 2;
        ctx.fillRect(x1, y1, w, h);
        ctx.strokeRect(x1, y1, w, h);

        const labelText = `✓ #${lblId}: ${cat}`;
        ctx.font = '600 11px Inter';
        const tw = ctx.measureText(labelText).width;
        ctx.fillStyle = '#10B981';
        ctx.fillRect(x1, y1 - 18, Math.max(60, tw + 10), 18);
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(labelText, x1 + 4, y1 - 5);
      }
      ctx.restore();
    }
  });

  // LỚP 2b: CÁC VÙNG CÓ ISSUE DO MODEL & GEMINI ĐỒNG THUẬN PHÁT HIỆN SÓT (CŨNG LÀ MÀU ĐỎ)
  if (state.showErrorBoxes !== false) {
    // 1. Box do Model phát hiện sót (Overlooked / Suggested)
    errors.forEach(err => {
      if (err.suggested_box && err.suggested_box.length === 4) {
        const [sx1, sy1, sx2, sy2] = err.suggested_box;
        ctx.save();
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = '#EF4444';
        ctx.lineWidth = 3;
        ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);
        ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
        ctx.fillRect(sx1, sy1, sx2 - sx1, sy2 - sy1);

        const text = `⚠️ SÓT: ${err.error_type} (Model/Gemini)`;
        ctx.font = 'bold 11px Inter';
        const tw = ctx.measureText(text).width;
        ctx.fillStyle = '#B91C1C';
        ctx.fillRect(sx1, sy1 - 20, tw + 10, 20);
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(text, sx1 + 5, sy1 - 6);
        ctx.restore();
      }
    });

    // 2. Box do Gemini VLM phân xử 3 bên phát hiện (nếu có)
    if (frame.gemini_proposals && frame.gemini_proposals.length > 0) {
      frame.gemini_proposals.forEach((prop, pIdx) => {
        if (prop.box_normalized && prop.box_normalized.length === 4) {
          const [px1, py1, px2, py2] = prop.box_normalized;
          const x1 = (px1 * 1280) / 1000;
          const y1 = (py1 * 720) / 1000;
          const x2 = (px2 * 1280) / 1000;
          const y2 = (py2 * 720) / 1000;
          const w = x2 - x1, h = y2 - y1;

          ctx.save();
          ctx.setLineDash([6, 4]);
          ctx.strokeStyle = '#EF4444';
          ctx.lineWidth = 3;
          ctx.strokeRect(x1, y1, w, h);
          ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
          ctx.fillRect(x1, y1, w, h);

          const tagText = `⚠️ GEMINI SÓT #${pIdx + 1}: ${prop.category}`;
          ctx.font = 'bold 11px Inter';
          const tw = ctx.measureText(tagText).width;
          ctx.fillStyle = '#B91C1C';
          ctx.fillRect(x1, y1 - 20, tw + 10, 20);
          ctx.fillStyle = '#FFFFFF';
          ctx.fillText(tagText, x1 + 5, y1 - 6);
          ctx.restore();
        }
      });
    }
  }

  // 3. AUTOFOCUS RETICLE & SPOTLIGHT (Khi chọn Issue hoặc Box)
  if (state.highlightedBox && state.highlightedBox.length === 4) {
    const [hx1, hy1, hx2, hy2] = state.highlightedBox;
    const hw = Math.max(1, hx2 - hx1);
    const hh = Math.max(1, hy2 - hy1);

    ctx.save();
    // Halo sáng vàng hổ phách
    ctx.strokeStyle = '#F59E0B';
    ctx.lineWidth = 3.5;
    ctx.shadowColor = 'rgba(245, 158, 11, 0.8)';
    ctx.shadowBlur = 14;
    ctx.strokeRect(hx1 - 4, hy1 - 4, hw + 8, hh + 8);

    // DSLR corner brackets [  ] trắng tương phản cao
    const bracketLen = Math.max(8, Math.min(22, hw / 3, hh / 3));
    ctx.lineWidth = 4;
    ctx.strokeStyle = '#FFFFFF';
    ctx.shadowBlur = 0;
    // Top-left
    ctx.beginPath(); ctx.moveTo(hx1 - 4, hy1 - 4 + bracketLen); ctx.lineTo(hx1 - 4, hy1 - 4); ctx.lineTo(hx1 - 4 + bracketLen, hy1 - 4); ctx.stroke();
    // Top-right
    ctx.beginPath(); ctx.moveTo(hx2 + 4 - bracketLen, hy1 - 4); ctx.lineTo(hx2 + 4, hy1 - 4); ctx.lineTo(hx2 + 4, hy1 - 4 + bracketLen); ctx.stroke();
    // Bottom-left
    ctx.beginPath(); ctx.moveTo(hx1 - 4, hy2 + 4 - bracketLen); ctx.lineTo(hx1 - 4, hy2 + 4); ctx.lineTo(hx1 - 4 + bracketLen, hy2 + 4); ctx.stroke();
    // Bottom-right
    ctx.beginPath(); ctx.moveTo(hx2 + 4 - bracketLen, hy2 + 4); ctx.lineTo(hx2 + 4, hy2 + 4); ctx.lineTo(hx2 + 4, hy2 + 4 - bracketLen); ctx.stroke();

    // Focus Tag
    const focusLabel = state.selectedIssueIndex != null
      ? `🎯 LỖI #${state.selectedIssueIndex + 1}: ${frame.errors[state.selectedIssueIndex]?.error_type || 'VI PHẠM'}`
      : `🎯 OBJECT #${state.selectedObjectId || ''}`;
    ctx.font = 'bold 12px Inter';
    const tagW = ctx.measureText(focusLabel).width + 16;
    ctx.fillStyle = '#F59E0B';
    ctx.fillRect(hx1 - 4, Math.max(0, hy1 - 28), tagW, 24);
    ctx.fillStyle = '#000000';
    ctx.fillText(focusLabel, hx1 + 4, Math.max(16, hy1 - 12));
    ctx.restore();
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
    const wrapRect = dom.canvasContainer ? dom.canvasContainer.getBoundingClientRect() : dom.canvas.getBoundingClientRect();
    const posX = Math.max(8, Math.min(wrapRect.width - 250, e.clientX - wrapRect.left + 15));
    const posY = Math.max(8, Math.min(wrapRect.height - 90, e.clientY - wrapRect.top + 15));
    dom.canvasTooltip.classList.remove('hidden');
    dom.canvasTooltip.style.left = `${posX}px`;
    dom.canvasTooltip.style.top = `${posY}px`;

    const matchedErr = (frame.errors || []).find(err => String(err.object_id) === String(found.id));
    const errText = matchedErr ? `<br><span style="color: #fca5a5;">⚠️ Vi phạm: ${matchedErr.error_type}</span>` : '<br><span style="color: #6ee7b7;">✓ Chuẩn (Đạt quy chuẩn)</span>';

    dom.canvasTooltip.innerHTML = `
      <strong>Object #${found.id}</strong> (${found.category}) ${errText}<br>
      <small style="color: #94a3b8;">Coords: [${Math.round(found.box2d.x1)}, ${Math.round(found.box2d.y1)}] ➔ [${Math.round(found.box2d.x2)}, ${Math.round(found.box2d.y2)}]</small><br>
      <small style="color: #38bdf8;">💡 Click để focus & xem chi tiết</small>
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
  if (!Array.isArray(selectedHoneypotReference) || selectedHoneypotReference.length === 0) {
    alert("Hãy chọn file JSON benchmark độc lập đã adjudicate trước khi đối chiếu. Hệ thống không tự dùng dữ liệu mẫu làm gold.");
    return;
  }
  if (!Array.isArray(state.currentRawData) || state.currentRawData.length === 0) {
    alert("Hãy nạp dữ liệu annotator cần đối chiếu trước.");
    return;
  }
  try {
    dom.btnRunHoneypotTest.disabled = true;
    dom.btnRunHoneypotTest.textContent = "Đang đối chiếu…";

    const res = await fetch(`${API_BASE}/api/qc/honeypot`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ground_truth_frames: selectedHoneypotReference,
        annotator_frames: state.currentRawData,
        annotator_name: "Annotator"
      })
    });

    if (!res.ok) throw new Error("Lỗi khi chấm điểm Honeypot");
    const json = await res.json();
    const evalData = json.honeypot_evaluation || json.evaluation;
    if (!evalData) throw new Error("Không nhận được kết quả chấm Honeypot từ server");

    const matchedCount = evalData.honeypot_frames_evaluated || 0;
    if (dom.hpFrames) dom.hpFrames.textContent = `${matchedCount} frames đối soát`;
    if (dom.hpPrecision) dom.hpPrecision.textContent = matchedCount ? `${evalData.precision}%` : "N/A";
    if (dom.hpRecall) dom.hpRecall.textContent = matchedCount ? `${evalData.recall}%` : "N/A";
    if (dom.hpIou) dom.hpIou.textContent = matchedCount ? `${evalData.mean_iou}%` : "N/A";
    if (dom.hpStatus) {
      const statusText = matchedCount ? (evalData.gold_standard_status || "Chưa có kết quả") : "Không có frame trùng";
      dom.hpStatus.textContent = statusText;
      dom.hpStatus.className = `badge ${statusText.includes("MEETS_CONFIGURED") ? 'badge-success' : 'badge-danger'}`;
    }
    if (dom.hpRec) {
      if (matchedCount === 0) {
        dom.hpRec.textContent = "Không có frame trùng tên; chưa tính metric. Kiểm tra file benchmark và dữ liệu annotator.";
      } else {
        dom.hpRec.textContent = `👉 Khuyến nghị: ${evalData.recommendation || ''}`;
      }
    }

  } catch (err) {
    alert("Lỗi chấm Honeypot: " + err.message);
  } finally {
    dom.btnRunHoneypotTest.disabled = false;
    dom.btnRunHoneypotTest.textContent = "Đối chiếu với tập đã chọn";
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
  // Thực sự đọc giá trị từ form và cập nhật state.config
  state.config.image_width = parseFloat(document.getElementById('cfg-img-w').value) || 1280;
  state.config.image_height = parseFloat(document.getElementById('cfg-img-h').value) || 720;
  state.config.min_box_width = parseFloat(document.getElementById('cfg-min-w').value) || 4.0;
  state.config.min_box_height = parseFloat(document.getElementById('cfg-min-h').value) || 4.0;
  state.config.min_box_area = parseFloat(document.getElementById('cfg-min-area').value) || 16.0;
  state.config.duplicate_box_iou_threshold = parseFloat(document.getElementById('cfg-dup-iou').value) || 0.85;
  state.config.mask_box_min_iou = parseFloat(document.getElementById('cfg-mask-iou').value) || 0.70;
  state.config.max_mask_box_drift_px = parseFloat(document.getElementById('cfg-mask-drift').value) || 15.0;
  const attrVal = document.getElementById('cfg-attributes').value;
  state.config.mandatory_attributes = attrVal.split(',').map(s => s.trim()).filter(Boolean);
  dom.configModal.classList.add('hidden');
  alert("✅ Đã lưu cấu hình thành công! Các ngưỡng mới sẽ áp dụng cho lần kiểm tra tiếp theo.");
}

function handleResetConfig() {
  const defaults = {
    image_width: 1280, image_height: 720,
    min_box_width: 4.0, min_box_height: 4.0, min_box_area: 16.0,
    duplicate_box_iou_threshold: 0.85,
    mandatory_attributes: ['occluded', 'truncated'],
    mask_box_min_iou: 0.70, max_mask_box_drift_px: 15.0
  };
  state.config = { ...defaults };
  document.getElementById('cfg-img-w').value = defaults.image_width;
  document.getElementById('cfg-img-h').value = defaults.image_height;
  document.getElementById('cfg-min-w').value = defaults.min_box_width;
  document.getElementById('cfg-min-h').value = defaults.min_box_height;
  document.getElementById('cfg-min-area').value = defaults.min_box_area;
  document.getElementById('cfg-dup-iou').value = defaults.duplicate_box_iou_threshold;
  document.getElementById('cfg-mask-iou').value = defaults.mask_box_min_iou;
  document.getElementById('cfg-mask-drift').value = defaults.max_mask_box_drift_px;
  document.getElementById('cfg-attributes').value = defaults.mandatory_attributes.join(', ');
  alert("🔄 Đã khôi phục tất cả cấu hình về giá trị mặc định.");
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
  URL.revokeObjectURL(url);
}

function setLoadingState(isLoading) {
  if (isLoading) {
    if (dom.btnLoadSample) { dom.btnLoadSample.disabled = true; dom.btnLoadSample.innerHTML = "⏳ Đang chạy..."; }
    if (dom.btnQuickSample) dom.btnQuickSample.disabled = true;
    if (dom.btnLoadReal) { dom.btnLoadReal.disabled = true; dom.btnLoadReal.innerHTML = "⏳ Đang nạp 20 frame..."; }
    if (dom.btnQuickReal) dom.btnQuickReal.disabled = true;
  } else {
    if (dom.btnLoadSample) {
      dom.btnLoadSample.disabled = false;
      dom.btnLoadSample.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="12" y1="18" x2="12" y2="12"/><line x1="9" y1="15" x2="15" y2="15"/></svg> ⚡ 4 Frame Mẫu';
    }
    if (dom.btnLoadReal) {
      dom.btnLoadReal.disabled = false;
      dom.btnLoadReal.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg> 🔥 20 Frame Thật (Archive)';
    }
    if (dom.btnQuickReal) dom.btnQuickReal.disabled = false;
  }
}

function handleGeminiKeyPrompt() {
  const currentKeyHint = state.modelsInfo?.gemini?.configured ? "(Đã có key máy chủ, nhập key mới nếu muốn đổi)" : "";
  const key = prompt(`Nhập Gemini API Key ${currentKeyHint}:\nKey sẽ được lưu bảo mật trong file .env máy chủ và tự động kích hoạt cho mọi người dùng mà không cần nhập lại.`, "");
  if (key && key.trim()) {
    fetch(`${API_BASE}/api/gemini/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: key.trim() })
    })
    .then(res => res.json())
    .then(data => {
      if (data.status === 'success') {
        alert("✅ Đã lưu và cấu hình Gemini API Key thành công trên máy chủ!");
        fetchInitialData();
      } else {
        alert("❌ Không thể lưu key: " + (data.detail || "Lỗi không xác định"));
      }
    })
    .catch(err => alert("Lỗi kết nối tới máy chủ: " + err.message));
  }
}

async function handleGeminiInspectCurrentFrame() {
  const frame = state.sortedFrames[state.selectedFrameIndex];
  if (!frame) {
    alert("Vui lòng chọn một frame trước khi gọi Gemini đối soát.");
    return;
  }

  const btn = dom.btnInspectGemini;
  if (!btn) return;
  const originalHtml = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `⏳ Đang đối soát 3 bên...`;

  try {
    const payload = {
      frame_name: frame.frame_name,
      labels: frame.labels || [],
      model_predictions: frame.model_predictions || [],
      guideline: state.guidelineText || null
    };

    if (state.customImages && state.customImages[frame.frame_name]) {
      payload.image_base64 = state.customImages[frame.frame_name];
    }

    const res = await fetch(`${API_BASE}/api/qc/gemini-inspect-frame`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      const detail = errData.detail || "Máy chủ không thể thực hiện đối soát Gemini.";
      if (res.status === 400 && detail.includes("chưa được cấu hình")) {
        const key = prompt("Máy chủ chưa có GEMINI_API_KEY. Nhập API key của bạn để lưu vào máy chủ:\n(Chỉ cần nhập 1 lần, các lượt sau máy chủ sẽ tự dùng)", "");
        if (key && key.trim()) {
          const cfgRes = await fetch(`${API_BASE}/api/gemini/config`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ api_key: key.trim() })
          });
          if (cfgRes.ok) {
            alert("Đã lưu API Key! Đang thực hiện lại đối soát...");
            await fetchInitialData();
            btn.disabled = false;
            btn.innerHTML = originalHtml;
            return handleGeminiInspectCurrentFrame();
          }
        }
      }
      throw new Error(detail);
    }

    const data = await res.json();
    const proposals = data.proposals || [];

    if (proposals.length === 0) {
      alert(`✅ Gemini VLM (${data.model}) đã đối soát chéo 3 bên:\nKhông phát hiện thêm bất thường nào giữa Model cục bộ, Nhãn Annotator và Sổ Quyết Định.`);
    } else {
      frame.gemini_proposals = proposals;
      if (!frame.errors) frame.errors = [];

      // Xóa các proposal cũ của Gemini nếu có để không bị nhân bản
      frame.errors = frame.errors.filter(e => !(e.qa_layer && e.qa_layer.includes("Gemini")));

      proposals.forEach((p, idx) => {
        const disc = p.discrepancy_type || "GEMINI_DISCREPANCY";
        let normBox = p.box_normalized || [0, 0, 1000, 1000];
        let pxBox = [
          Math.round((normBox[0] * 1280) / 1000),
          Math.round((normBox[1] * 720) / 1000),
          Math.round((normBox[2] * 1280) / 1000),
          Math.round((normBox[3] * 720) / 1000)
        ];

        frame.errors.push({
          error_type: disc,
          category: p.category,
          qa_layer: "Tầng 3 (Gemini VLM Arbitrator)",
          severity: disc.includes("UNLABELED") || disc.includes("CONFLICT") ? "HIGH" : "MEDIUM",
          models_flagged: [data.model],
          model_status: p.model_status,
          label_status: p.label_status,
          guideline_citation: p.guideline_citation,
          explanation: p.evidence,
          action: "Reviewer xem xét gán bổ sung hoặc hiệu chỉnh class trên CVAT",
          suggested_box: pxBox
        });
      });
      frame.error_count = frame.errors.length;

      renderCanvas();
      renderFrameSidebar();
      renderPriorityTable();
      alert(`✨ Gemini VLM (${data.model}) đã hoàn tất phân xử 3 bên:\nPhát hiện ${proposals.length} điểm bất đồng/bất thường giữa Model, Nhãn và Guideline!`);
    }
  } catch (err) {
    alert("❌ Lỗi khi gọi Gemini VLM: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = originalHtml;
  }
}

