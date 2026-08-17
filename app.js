/* ================================================================
   Amy — AI File Analysis Agent — Frontend Logic v3 (Multi-file)
   ================================================================ */

// --- DOM refs ---
const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
const uploadStatus = document.getElementById("uploadStatus");
const fileListCard = document.getElementById("fileListCard");
const fileList = document.getElementById("fileList");
const fileCount = document.getElementById("fileCount");
const fileInfoCard = document.getElementById("fileInfoCard");
const fileInfo = document.getElementById("fileInfo");
const centerPanel = document.getElementById("centerPanel");
const tabNav = document.getElementById("tabNav");
const suggestBtn = document.getElementById("suggestBtn");
const suggestionsContent = document.getElementById("suggestionsContent");
const previewContent = document.getElementById("previewContent");
const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatSendBtn = document.getElementById("chatSendBtn");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingText = document.getElementById("loadingText");

let currentSessionId = null;
let activeFileId = null;
let currentSummary = null;
let columnList = [];
let chatLoading = false;
let sessionFiles = {}; // {file_id: {filename, type, row_count, col_count}}
let cleaningLoaded = false;

// --- Loading ---
function showLoading(msg) { loadingText.textContent = msg; loadingOverlay.style.display = "flex"; }
function hideLoading() { loadingOverlay.style.display = "none"; }

// --- Tab switching ---
tabNav.addEventListener("click", (e) => {
  const btn = e.target.closest(".tab-btn");
  if (!btn) return;
  const tabName = btn.dataset.tab;
  tabNav.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  document.getElementById(`tab-${tabName}`).classList.add("active");
  if (tabName === "clean") loadCleaning();
});

// ====================================================================
// UPLOAD
// ====================================================================
uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("drag-over"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault(); uploadZone.classList.remove("drag-over");
  if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files.length > 0) handleFile(fileInput.files[0]); });

async function handleFile(file) {
  uploadStatus.className = "upload-status loading";
  uploadStatus.textContent = "Uploading & analyzing…";
  const formData = new FormData();
  formData.append("file", file);
  if (currentSessionId) formData.append("session_id", currentSessionId);

  try {
    showLoading("Parsing & running EDA…");
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    hideLoading();
    if (!res.ok) { const err = await res.json(); throw new Error(err.detail || "Upload failed"); }

    const data = await res.json();
    const sid = data.session_id;
    const fid = data.file_id;

    // First file: initialize session
    if (!currentSessionId) {
      currentSessionId = sid;
      enableChat();
      chatMessages.innerHTML = "";
      centerPanel.style.display = "flex";
    }

    // Store file info
    sessionFiles[fid] = {
      filename: data.filename,
      type: data.type,
      row_count: data.row_count,
      col_count: data.col_count,
    };

    // Switch to new file
    activeFileId = fid;
    currentSummary = data;
    columnList = data.columns || [];
    cleaningLoaded = false;

    uploadStatus.className = "upload-status success";
    uploadStatus.textContent = `✓ ${data.filename} added to session`;

    renderFileList();
    switchToActiveFile(data);
  } catch (err) {
    hideLoading();
    uploadStatus.className = "upload-status error";
    uploadStatus.textContent = `✗ ${err.message}`;
    console.error(err);
  }
}

// ====================================================================
// FILE LIST
// ====================================================================
function renderFileList() {
  fileListCard.style.display = "block";
  const count = Object.keys(sessionFiles).length;
  fileCount.textContent = `(${count})`;

  let html = "";
  for (const [fid, f] of Object.entries(sessionFiles)) {
    const isActive = fid === activeFileId;
    const icon = f.type === "tabular" ? "📊" : (f.type === "text" ? "📝" : "📦");
    html += `<div class="file-item ${isActive ? "active" : ""}" data-fid="${fid}">
      <span class="file-item-icon">${icon}</span>
      <div class="file-item-info">
        <div class="file-item-name">${escapeHtml(f.filename)}</div>
        <div class="file-item-meta">${f.type} · ${(f.row_count || 0).toLocaleString()} rows</div>
      </div>
      ${isActive ? '<span class="file-item-badge">active</span>' : ""}
    </div>`;
  }
  fileList.innerHTML = html;

  // Click handlers
  fileList.querySelectorAll(".file-item").forEach(item => {
    item.addEventListener("click", () => {
      const fid = item.dataset.fid;
      if (fid !== activeFileId) switchToFile(fid);
    });
  });
}

async function switchToFile(fid) {
  if (!currentSessionId || !sessionFiles[fid]) return;
  showLoading(`Switching to ${sessionFiles[fid].filename}…`);

  try {
    // Fetch full summary (preview_rows, columns, EDA, cleaning) for this file
    const res = await fetch(`/api/summary?session_id=${encodeURIComponent(currentSessionId)}&file_id=${encodeURIComponent(fid)}`, { method: "POST" });
    if (!res.ok) throw new Error("Failed to load file data");
    const data = await res.json();

    activeFileId = fid;
    currentSummary = data;
    columnList = data.columns || [];
    cleaningLoaded = false;

    // Clear stale cleaning & chart content
    document.getElementById("tab-clean").innerHTML = '<div class="card"><p class="muted-text">Click 🧹 Clean tab to load cleaning analysis.</p></div>';
    document.getElementById("chartResult").innerHTML = "";
    document.getElementById("chartSuggestionsGrid").innerHTML = '<p class="muted-text">Click "Get Suggestions" to see AI-recommended visualizations.</p>';
    suggestionsContent.style.display = "none";

    // Render everything
    renderFileInfo(data);
    renderPreview(data);
    if (data.eda) renderEDA(data.eda);
    populateChartBuilder();
    renderFileList();

    // Switch to EDA tab
    tabNav.querySelector('[data-tab="eda"]').click();

    hideLoading();
  } catch (err) {
    hideLoading();
    console.error("File switch error:", err);
  }
}

function switchToActiveFile(data) {
  // Called when a new file is uploaded and becomes active
  fileInfoCard.style.display = "block";
  renderFileInfo(data);
  renderFileList();

  if (data.type === "tabular" && data.eda) {
    renderEDA(data.eda);
    populateChartBuilder();
    if (data.cleaning) { renderCleaningIssues(data.cleaning); cleaningLoaded = true; }
    renderPreview(data);
    tabNav.querySelector('[data-tab="eda"]').click();
  } else {
    renderPreview(data);
    tabNav.querySelector('[data-tab="preview"]').click();
  }
}

// --- File Info ---
function renderFileInfo(data) {
  fileInfoCard.style.display = "block";
  let rows = `<div class="info-row"><span class="info-label">Filename</span><span class="info-value accent">${escapeHtml(data.filename)}</span></div>`;
  rows += `<div class="info-row"><span class="info-label">Type</span><span class="info-value">${escapeHtml(data.type)}</span></div>`;
  if (data.type === "tabular") {
    rows += `<div class="info-row"><span class="info-label">Rows</span><span class="info-value">${(data.row_count || 0).toLocaleString()}</span></div>`;
    rows += `<div class="info-row"><span class="info-label">Columns</span><span class="info-value">${data.col_count || 0}</span></div>`;
  }
  fileInfo.innerHTML = rows;
}

// ====================================================================
// EDA
// ====================================================================
function renderEDA(eda) {
  const container = document.getElementById("tab-eda");
  const score = eda.quality_score;
  const cls = score >= 80 ? "good" : (score >= 50 ? "ok" : "bad");

  let html = '<div class="card"><div class="eda-quality">';
  html += `<div class="quality-circle ${cls}">${score}%</div>`;
  html += `<div class="quality-details"><strong>Data Quality Score</strong><br/>${eda.total_missing} missing (${eda.missing_pct}%) · ${eda.duplicate_rows} duplicates · ${eda.constant_columns.length} const cols</div></div>`;

  html += '<div class="eda-grid">';
  html += `<div class="eda-metric"><div class="metric-value">${eda.row_count.toLocaleString()}</div><div class="metric-label">Rows</div></div>`;
  html += `<div class="eda-metric"><div class="metric-value">${eda.col_count}</div><div class="metric-label">Cols</div></div>`;
  html += `<div class="eda-metric"><div class="metric-value">${eda.numeric_col_count}</div><div class="metric-label">Numeric</div></div>`;
  html += `<div class="eda-metric"><div class="metric-value">${eda.categorical_col_count}</div><div class="metric-label">Cat</div></div>`;
  html += `<div class="eda-metric${eda.duplicate_rows > 0 ? ' warn' : ''}"><div class="metric-value">${eda.duplicate_rows}</div><div class="metric-label">Dups</div></div>`;
  html += `<div class="eda-metric${eda.missing_pct > 5 ? ' danger' : ''}"><div class="metric-value">${eda.missing_pct}%</div><div class="metric-label">Missing</div></div>`;
  html += '</div>';

  const missingCols = Object.entries(eda.missing_by_column || {}).filter(([,v]) => v.count > 0);
  if (missingCols.length) {
    html += '<div class="eda-section-title">Missing Values</div><div class="missing-bar-wrap">';
    for (const [col, v] of missingCols.slice(0, 10)) {
      const sev = v.pct > 20 ? "high" : (v.pct > 5 ? "medium" : "low");
      html += `<div class="missing-bar-row"><span class="missing-bar-label">${escapeHtml(col)}</span><div class="missing-bar-track"><div class="missing-bar-fill ${sev}" style="width:${Math.min(v.pct,100)}%"></div></div><span style="font-size:0.72rem;color:var(--text-dim);width:55px">${v.count} (${v.pct}%)</span></div>`;
    }
    html += '</div>';
  }

  if (eda.correlations && eda.correlations.length) {
    html += '<div class="eda-section-title">Top Correlations</div><div class="corr-list">';
    for (const c of eda.correlations.slice(0, 8)) {
      html += `<div class="corr-item"><span class="corr-pair">${escapeHtml(c.col1)} ↔ ${escapeHtml(c.col2)}</span><span class="corr-value ${c.correlation>0?'positive':'negative'}">${c.correlation.toFixed(3)}</span></div>`;
    }
    html += '</div>';
  }

  const oils = Object.entries(eda.outliers || {}).filter(([,v]) => v.count > 0);
  if (oils.length) {
    html += '<div class="eda-section-title">Outliers (IQR)</div><div class="corr-list">';
    for (const [col, v] of oils.slice(0, 8)) {
      html += `<div class="corr-item"><span class="corr-pair">${escapeHtml(col)}</span><span style="color:var(--orange);font-size:0.78rem">${v.count} (${v.pct}%)</span></div>`;
    }
    html += '</div>';
  }
  html += '</div>';
  container.innerHTML = html;
}

// ====================================================================
// CLEANING
// ====================================================================
async function loadCleaning() {
  if (cleaningLoaded || !currentSessionId || !activeFileId) return;
  const container = document.getElementById("tab-clean");
  container.innerHTML = '<div class="card"><p class="muted-text">Loading cleaning analysis…</p></div>';
  try {
    const res = await fetch(`/api/cleaning?session_id=${encodeURIComponent(currentSessionId)}&file_id=${encodeURIComponent(activeFileId)}`, { method: "POST" });
    if (!res.ok) throw new Error("Failed");
    renderFullCleaning(await res.json());
    cleaningLoaded = true;
  } catch (err) {
    container.innerHTML = `<div class="card"><p class="muted-text">Error: ${escapeHtml(err.message)}</p></div>`;
  }
}

function renderCleaningIssues(data) { renderFullCleaning(data); cleaningLoaded = true; }

function renderFullCleaning(data) {
  const container = document.getElementById("tab-clean");
  let html = '<div class="card"><div class="cleaning-summary">';
  html += `<div class="cleaning-stat"><div class="stat-num">${data.total_issues}</div><div class="stat-label">Total</div></div>`;
  html += `<div class="cleaning-stat"><div class="stat-num high">${data.issues_by_severity?.high||0}</div><div class="stat-label">High</div></div>`;
  html += `<div class="cleaning-stat"><div class="stat-num medium">${data.issues_by_severity?.medium||0}</div><div class="stat-label">Med</div></div>`;
  html += `<div class="cleaning-stat"><div class="stat-num low">${data.issues_by_severity?.low||0}</div><div class="stat-label">Low</div></div>`;
  html += '</div>';

  if (data.issues && data.issues.length) {
    html += '<div class="eda-section-title">Issues Found</div><div class="issue-list">';
    for (const i of data.issues) {
      html += `<div class="issue-item"><span class="issue-severity ${i.severity}">${i.severity}</span><div class="issue-detail"><span class="issue-column">[${i.type.replace(/_/g,' ')}]</span> ${escapeHtml(i.detail)}</div></div>`;
    }
    html += '</div>';
  } else {
    html += '<p style="color:var(--green);font-size:0.84rem;padding:8px 0;">✅ No significant issues found!</p>';
  }

  if (data.ai_suggestions) {
    html += '<div class="eda-section-title">🤖 AI Cleaning Recommendations</div>';
    html += `<div class="ai-suggestions">${escapeHtml(data.ai_suggestions)}</div>`;
  }
  html += '</div>';
  container.innerHTML = html;
}

// ====================================================================
// CHARTS
// ====================================================================
function populateChartBuilder() {
  const xSel = document.getElementById("chartXCol");
  const ySel = document.getElementById("chartYCol");
  xSel.innerHTML = columnList.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");
  ySel.innerHTML = '<option value="">(none)</option>' + columnList.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");
}

document.getElementById("loadChartSuggestionsBtn").addEventListener("click", loadChartSuggestions);
document.getElementById("drawChartBtn").addEventListener("click", drawCustomChart);

async function loadChartSuggestions() {
  if (!currentSessionId || !activeFileId) return;
  const grid = document.getElementById("chartSuggestionsGrid");
  grid.innerHTML = '<p class="muted-text">🤖 Asking AI for chart ideas…</p>';
  try {
    const res = await fetch(`/api/chart-suggestions?session_id=${encodeURIComponent(currentSessionId)}&file_id=${encodeURIComponent(activeFileId)}`, { method: "POST" });
    if (!res.ok) throw new Error("Failed");
    const data = await res.json();
    const suggestions = data.suggestions || [];
    if (!suggestions.length) { grid.innerHTML = '<p class="muted-text">No suggestions available.</p>'; return; }

    grid.innerHTML = "";
    for (const s of suggestions) {
      const card = document.createElement("div");
      card.className = "chart-card";
      card.innerHTML = `<div class="chart-info"><div class="chart-title-text">${escapeHtml(s.title||s.chart_type)}</div><div class="chart-reason">${escapeHtml(s.reason||'')}</div></div><div class="chart-btn-row"><button class="btn btn-xs render-chart-btn" data-ctype="${escapeHtml(s.chart_type)}" data-x="${escapeHtml(s.x_column||'')}" data-y="${escapeHtml(s.y_column||'')}" data-title="${escapeHtml(s.title||'')}">🎨 Render</button></div><div class="chart-result-inline"></div>`;
      grid.appendChild(card);

      card.querySelector(".render-chart-btn").addEventListener("click", async function () {
        const rd = card.querySelector(".chart-result-inline");
        rd.innerHTML = '<p style="padding:12px;color:var(--text-dim);">Generating…</p>';
        try {
          rd.innerHTML = `<img src="${await fetchChart(this.dataset.ctype, this.dataset.x, this.dataset.y||null, this.dataset.title)}" />`;
        } catch (err) { rd.innerHTML = `<p style="padding:12px;color:var(--red);">${escapeHtml(err.message)}</p>`; }
      });
    }
  } catch (err) { grid.innerHTML = `<p class="muted-text" style="color:var(--red)">Error: ${escapeHtml(err.message)}</p>`; }
}

async function drawCustomChart() {
  const ct = document.getElementById("chartType").value;
  const x = document.getElementById("chartXCol").value;
  const y = document.getElementById("chartYCol").value || null;
  const t = document.getElementById("chartTitle").value || null;
  const rd = document.getElementById("chartResult");
  if (!x) { rd.innerHTML = '<p style="color:var(--red)">Select X column.</p>'; return; }
  rd.innerHTML = '<p style="color:var(--text-dim);padding:12px;">Generating…</p>';
  try { rd.innerHTML = `<img src="${await fetchChart(ct, x, y, t)}" />`; }
  catch (err) { rd.innerHTML = `<p style="color:var(--red)">${escapeHtml(err.message)}</p>`; }
}

async function fetchChart(ctype, x, y, title) {
  const res = await fetch("/api/chart", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: currentSessionId, file_id: activeFileId, chart_type: ctype, x_column: x, y_column: y || null, title: title || null }),
  });
  if (!res.ok) { const e = await res.json(); throw new Error(e.detail); }
  return (await res.json()).image;
}

// ====================================================================
// PREVIEW
// ====================================================================
function renderPreview(data) {
  if (data.type === "tabular" && data.preview_rows && data.preview_rows.length) {
    const cols = data.columns || Object.keys(data.preview_rows[0] || {});
    let t = '<table class="preview-table"><thead><tr>';
    cols.forEach(c => { t += `<th>${escapeHtml(String(c))}</th>`; });
    t += "</tr></thead><tbody>";
    data.preview_rows.forEach(row => { t += "<tr>"; cols.forEach(c => { t += `<td>${escapeHtml(String(row[c]??''))}</td>`; }); t += "</tr>"; });
    t += "</tbody></table>";
    previewContent.innerHTML = t;
    suggestBtn.style.display = "inline-flex";
  } else if (data.type === "text" || data.type === "powerbi") {
    previewContent.innerHTML = `<div class="preview-text">${escapeHtml(data.preview||'(no preview)')}</div>`;
    suggestBtn.style.display = data.type === "text" ? "inline-flex" : "none";
  } else {
    previewContent.innerHTML = '<div class="preview-text">No preview.</div>';
    suggestBtn.style.display = "none";
  }
}

suggestBtn.addEventListener("click", async () => {
  if (!currentSessionId || !activeFileId) return;
  showLoading("Thinking…");
  try {
    const res = await fetch(`/api/suggest?session_id=${encodeURIComponent(currentSessionId)}&file_id=${encodeURIComponent(activeFileId)}`, { method: "POST" });
    if (!res.ok) throw new Error("Failed");
    const d = await res.json();
    suggestionsContent.textContent = d.suggestions;
    suggestionsContent.style.display = "block";
  } catch (err) { suggestionsContent.textContent = `Failed: ${err.message}`; }
  hideLoading();
});

// ====================================================================
// CHAT
// ====================================================================
function enableChat() {
  chatInput.disabled = false;
  chatSendBtn.disabled = false;
  chatInput.placeholder = "Ask about any file in your session…";
  const ph = chatMessages.querySelector(".chat-placeholder");
  if (ph) ph.remove();
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = chatInput.value.trim();
  if (!message || !currentSessionId || chatLoading) return;

  addChatMessage("user", message);
  chatInput.value = "";
  chatLoading = true;
  chatInput.disabled = true;
  chatSendBtn.disabled = true;

  try {
    const res = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: currentSessionId, file_id: activeFileId }),
    });
    if (!res.ok) { const err = await res.json(); throw new Error(err.detail || "Chat failed"); }
    addChatMessage("assistant", (await res.json()).reply);
  } catch (err) { addChatMessage("error", `Error: ${err.message}`); }
  finally {
    chatLoading = false;
    chatInput.disabled = false;
    chatSendBtn.disabled = false;
    chatInput.focus();
  }
});

function addChatMessage(role, text) {
  const div = document.createElement("div");
  div.className = `chat-msg ${role}`;
  if (role === "user") div.innerHTML = `<span class="msg-label">You</span><div>${escapeHtml(text)}</div>`;
  else if (role === "assistant") div.innerHTML = `<span class="msg-label">Amy</span><div>${formatMessage(text)}</div>`;
  else div.innerHTML = `<span class="msg-label">Error</span><div>${escapeHtml(text)}</div>`;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// --- Helpers ---
function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
function formatMessage(text) {
  let h = escapeHtml(text);
  h = h.replace(/\*\*(.+?)\*\*/g,"<strong>$1</strong>").replace(/`([^`]+)`/g,"<code>$1</code>").replace(/\n/g,"<br>");
  return h;
}
