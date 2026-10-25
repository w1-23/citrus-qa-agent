var API_URL = '/api/v2/chat';
var currentAccount = null;
var visualizationEnabled = localStorage.getItem('citrus_visualization_enabled') === 'true';
function sessionStorageKey() { return 'citrus_session_id_' + (currentAccount ? currentAccount.user_id : ''); }
function showAuthGate(show) {
  var gate = document.getElementById('authGate');
  if (gate) gate.hidden = !show;
  var app = document.getElementById('mainContainer');
  if (app) app.style.visibility = show ? 'hidden' : '';
  var settings = document.getElementById('modelSettingsBtn');
  if (settings) settings.hidden = !currentAccount || currentAccount.role !== 'admin';
  var who = document.getElementById('accountLabel');
  if (who) who.textContent = currentAccount ? currentAccount.username : '';
  var admin = document.getElementById('adminUsersBtn');
  if (admin) admin.hidden = !currentAccount || currentAccount.role !== 'admin';
  var permission = document.getElementById('permBadge');
  if (permission) permission.hidden = !currentAccount || currentAccount.role !== 'admin';
  var logout = document.getElementById('logoutBtn');
  if (logout) logout.hidden = !currentAccount;
}
async function authenticate(mode) {
  var username = (document.getElementById('authUsername').value || '').trim();
  var password = document.getElementById('authPassword').value || '';
  var error = document.getElementById('authError');
  error.textContent = '';
  try {
    var response = await fetch('/api/v2/auth/' + mode, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:username,password:password})});
    var data = await response.json();
    if (!response.ok) throw new Error(data.detail || '登录失败');
    currentAccount = data.user;
    localStorage.removeItem('citrus_session_id');
    var next = new URLSearchParams(location.search).get('next');
    if (next === '/admin') {
      if (currentAccount.role !== 'admin') throw new Error('此页面仅管理员可访问。');
      location.replace('/admin'); return;
    }
    await startAuthenticatedApp();
  } catch (e) { error.textContent = e.message || '请求失败'; }
}
async function logoutAccount() {
  closeSettingsModal();
  await fetch('/api/v2/auth/logout', {method:'POST'}).catch(function(){});
  currentAccount = null; sessionId = null; showAuthGate(true);
  document.getElementById('authPassword').value = '';
}
async function startAuthenticatedApp() {
  showAuthGate(false);
  sessionId = localStorage.getItem(sessionStorageKey());
  var configReady;
  if (currentAccount && currentAccount.role === 'admin') configReady = loadRuntimeConfig();
  else configReady = fetch('/api/v2/config').then(function(r) { return r.json(); }).then(function(cfg) {
    _runtimeConfig = cfg; applyAppVersion();
  }).catch(function(){});
  await ensureSession();
  if (currentAccount && currentAccount.role === 'admin' &&
      new URLSearchParams(location.search).get('adminSettings') === '1') {
    if (configReady) await configReady;
    history.replaceState(null, '', '/');
    openSettingsModal();
  }
}
async function bootstrapAccount() {
  try {
    var response = await fetch('/api/v2/auth/me');
    if (!response.ok) throw new Error('not logged in');
    var data = await response.json(); currentAccount = data.user;
    await startAuthenticatedApp();
  } catch (_) { showAuthGate(true); }
}
window.addEventListener('DOMContentLoaded', bootstrapAccount);
function syncVisualizationToggle() {
  var button = document.getElementById('visualToggle');
  if (!button) return;
  button.setAttribute('aria-pressed', visualizationEnabled ? 'true' : 'false');
  button.classList.toggle('active', visualizationEnabled);
  button.textContent = visualizationEnabled ? '可视化已开启' : '可视化';
}
function toggleVisualization() {
  visualizationEnabled = !visualizationEnabled;
  localStorage.setItem('citrus_visualization_enabled', visualizationEnabled ? 'true' : 'false');
  syncVisualizationToggle();
}
window.addEventListener('DOMContentLoaded', syncVisualizationToggle);
var sessionId = null;
var chatContainer = document.getElementById('chat-container');
var userInput = document.getElementById('user-input');
var sendBtn = document.getElementById('send-btn');
var sidebarContent = document.getElementById('sidebar-content');
var citationCount = document.getElementById('citation-count');
var _selectedAttachments = [];
var _attachmentPreviewUrls = [];
var _sentAttachmentPreviewUrls = [];
var roundHistory = [];
var _citationsRetry = 0;   // v8.10q: citations 空结果延迟重试计数（防 spawn 时序缺口）
var currentRoundId = 0;
var currentQuery = '';
var totalTokens = 0;
var _agentTokenBuffer = '';
var _agentTokenName = '';

if (typeof marked === 'undefined') window.marked = { parse: function(t) { return t; } };

function handleAttachmentSelection(fileList) {
  var dropped = Array.from(fileList || []);
  var allowed = dropped.filter(function(file) {
    var name = String(file.name || '').toLowerCase();
    return /^image\/(jpeg|png|webp|gif)$/.test(file.type || '') || /\.(jpe?g|png|webp|gif|pdf|docx|csv|tsv|xlsx|xls|xlsm)$/.test(name);
  });
  if (allowed.length !== dropped.length) {
    var rejected = dropped.filter(function(file) { return allowed.indexOf(file) < 0; });
    alert('暂不支持这些文件：' + rejected.map(function(file) { return file.name; }).join('、'));
  }
  var added = allowed;
  if (!added.length) return;
  var total = _selectedAttachments.reduce(function(n, f) { return n + f.size; }, 0) +
    added.reduce(function(n, f) { return n + f.size; }, 0);
  if (_selectedAttachments.length + added.length > 8 || total > 20 * 1024 * 1024) {
    alert('每轮最多 8 个附件，合计不超过 20 MB。');
    var input = document.getElementById('attachment-input'); if (input) input.value = '';
    return;
  }
  _selectedAttachments = _selectedAttachments.concat(added);
  renderAttachmentSelection();
  var picker = document.getElementById('attachment-input'); if (picker) picker.value = '';
}

function hasDraggedFiles(event) {
  var types = event.dataTransfer && event.dataTransfer.types;
  return !!types && Array.from(types).indexOf('Files') >= 0;
}

(function enableFileDrop() {
  var panel = document.querySelector('.chat-panel');
  if (!panel) return;
  var dragDepth = 0;
  panel.addEventListener('dragenter', function(event) {
    if (!hasDraggedFiles(event)) return;
    event.preventDefault();
    dragDepth += 1;
    panel.classList.add('file-drag-over');
  });
  panel.addEventListener('dragover', function(event) {
    if (!hasDraggedFiles(event)) return;
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
  });
  panel.addEventListener('dragleave', function(event) {
    if (!hasDraggedFiles(event)) return;
    event.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) panel.classList.remove('file-drag-over');
  });
  panel.addEventListener('drop', function(event) {
    if (!hasDraggedFiles(event)) return;
    event.preventDefault();
    dragDepth = 0;
    panel.classList.remove('file-drag-over');
    handleAttachmentSelection(event.dataTransfer.files);
  });
})();

function renderAttachmentSelection() {
  var host = document.getElementById('attachment-list'); if (!host) return;
  _attachmentPreviewUrls.forEach(function(url) { URL.revokeObjectURL(url); });
  _attachmentPreviewUrls = [];
  host.textContent = '';
  host.style.display = _selectedAttachments.length ? 'flex' : 'none';
  _selectedAttachments.forEach(function(file, index) {
    var card = document.createElement('div');
    card.className = 'attachment-preview-card';
    card.title = file.name;
    if (/^image\/(jpeg|png|webp|gif)$/.test(file.type || '') || /\.(jpe?g|png|webp|gif)$/i.test(file.name || '')) {
      var previewUrl = URL.createObjectURL(file);
      _attachmentPreviewUrls.push(previewUrl);
      var image = document.createElement('img');
      image.className = 'attachment-preview-image';
      image.src = previewUrl;
      image.alt = file.name;
      card.appendChild(image);
    } else {
      var icon = document.createElement('span');
      icon.className = 'attachment-preview-icon';
      icon.textContent = /\.pdf$/i.test(file.name || '') ? 'PDF' :
        /\.xlsx$/i.test(file.name || '') ? 'XLSX' : /\.xlsm?$/i.test(file.name || '') ? 'XLS' : /\.tsv$/i.test(file.name || '') ? 'TSV' : /\.csv$/i.test(file.name || '') ? 'CSV' : 'DOC';
      card.appendChild(icon);
    }
    var label = document.createElement('span');
    label.className = 'attachment-preview-name';
    label.textContent = file.name;
    card.appendChild(label);
    var remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'attachment-preview-remove';
    remove.setAttribute('aria-label', '移除 ' + file.name);
    remove.title = '移除附件';
    remove.textContent = '×';
    remove.onclick = function() { _selectedAttachments.splice(index, 1); renderAttachmentSelection(); };
    card.appendChild(remove);
    host.appendChild(card);
  });
}

function appendSentAttachmentPreviews(contentEl, files) {
  if (!contentEl || !files || !files.length) return;
  var gallery = document.createElement('div');
  gallery.className = 'sent-attachment-gallery';
  files.forEach(function(file) {
    var card = document.createElement('div');
    card.className = 'sent-attachment-card';
    card.title = file.name;
    if (/^image\/(jpeg|png|webp|gif)$/.test(file.type || '') || /\.(jpe?g|png|webp|gif)$/i.test(file.name || '')) {
      var previewUrl = URL.createObjectURL(file);
      _sentAttachmentPreviewUrls.push(previewUrl);
      var image = document.createElement('img');
      image.className = 'sent-attachment-image';
      image.src = previewUrl;
      image.alt = file.name;
      card.appendChild(image);
    } else {
      var icon = document.createElement('span');
      icon.className = 'sent-attachment-icon';
      icon.textContent = /\.pdf$/i.test(file.name || '') ? 'PDF' :
        /\.xlsx$/i.test(file.name || '') ? 'XLSX' : /\.xlsm?$/i.test(file.name || '') ? 'XLS' : /\.tsv$/i.test(file.name || '') ? 'TSV' : /\.csv$/i.test(file.name || '') ? 'CSV' : 'DOC';
      card.appendChild(icon);
    }
    var label = document.createElement('span');
    label.className = 'sent-attachment-name';
    label.textContent = file.name;
    card.appendChild(label);
    if (/\.pdf$/i.test(file.name || '')) {
      var review = document.createElement('button');
      review.type = 'button'; review.className = 'paper-review-btn'; review.textContent = '解析引用';
      review.onclick = function() { reviewUploadedPdf(file, card); };
      card.appendChild(review);
    }
    gallery.appendChild(card);
  });
  contentEl.appendChild(gallery);
}

async function reviewUploadedPdf(file, card) {
  if (!sessionId) return;
  var old = card.querySelector('.paper-review-result'); if (old) old.remove();
  var panel = document.createElement('div'); panel.className = 'paper-review-result'; panel.textContent = '正在提取引用…'; card.appendChild(panel);
  try {
    var payload = {name:file.name, data:await fileAsDataUrl(file)};
    var response = await fetch('/api/v2/session/' + encodeURIComponent(sessionId) + '/paper-review', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    var result = await response.json(); if (!response.ok) throw new Error(result.detail || '解析失败');
    panel.textContent = '';
    var status = document.createElement('div'); status.textContent = result.status === 'incomplete' ? result.reason : ((result.references || []).length + ' 条参考文献；' + (result.reason || '正文编号引用已标亮。'));
    panel.appendChild(status);
    if (result.highlighted_pdf_url) {
      var artifactId = result.highlighted_pdf_url.split('/').pop();
      var link = document.createElement('a'); link.href = '/api/v2/session/' + encodeURIComponent(sessionId) + '/paper-preview/' + encodeURIComponent(artifactId); link.target = '_blank'; link.textContent = '预览高亮 PDF'; panel.appendChild(link);
      var download = document.createElement('a'); download.href = result.highlighted_pdf_url; download.textContent = ' · 下载副本'; panel.appendChild(download);
    }
    (result.references || []).forEach(function(ref) {
      var row = document.createElement('div'); row.className = 'paper-reference-row';
      var text = document.createElement('span'); text.textContent = '[' + ref.number + '] ' + ref.text; row.appendChild(text);
      if (ref.doi) {
        var check = document.createElement('button'); check.textContent = '核验';
        check.onclick = async function() { check.disabled = true; var r = await fetch('/api/v2/session/' + encodeURIComponent(sessionId) + '/citation-check', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({doi:ref.doi,title:ref.title,citation_text:ref.text})}); var d = await r.json(); check.textContent = d.message || d.status; check.title = JSON.stringify(d); };
        row.appendChild(check);
        var get = document.createElement('button'); get.textContent = '找开放 PDF';
        get.onclick = async function() { get.disabled = true; var r = await fetch('/api/v2/session/' + encodeURIComponent(sessionId) + '/citation-open-pdf', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({doi:ref.doi})}); var d = await r.json(); if (d.download_url) { get.textContent = '下载 PDF'; get.onclick = function(){ window.open(d.download_url, '_blank'); }; } else get.textContent = d.message || d.status; };
        row.appendChild(get);
      }
      panel.appendChild(row);
    });
  } catch (e) { panel.textContent = '解析失败：' + (e.message || '请稍后重试'); }
}

function clearSentAttachmentPreviewUrls() {
  _sentAttachmentPreviewUrls.forEach(function(url) { URL.revokeObjectURL(url); });
  _sentAttachmentPreviewUrls = [];
}

function fileAsDataUrl(file) {
  return new Promise(function(resolve, reject) {
    var reader = new FileReader();
    reader.onload = function() { resolve(String(reader.result || '')); };
    reader.onerror = function() { reject(new Error('无法读取附件：' + file.name)); };
    reader.readAsDataURL(file);
  });
}
marked.setOptions({ breaks: true });

/* ─── XSS 防护: 所有动态文本入 innerHTML 前转义 (v8.3.3) ─── */
function escapeHtml(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
/* v8.4.3: 代码块内容只转义 & < >（引号安全——处于 <pre><code> 标签内部，
   全量转义会把代码里的 ' 显示成 &#39;，如 5'上游调控区） */
function displayToolName(name) {
  var names = {
    deepseek_search: 'Web Search', deepseek_fetch: 'Web Fetch',
    deepseek_fetch_paper: 'Web Fetch (paper full text)',
    local_hybrid_search: '本地混合检索', local_bm25_search: '本地关键词检索',
    local_dense_search: '本地语义检索', citrus_rag_search: '本地语义检索',
    draw_svg: 'draw',
    inspect_uploaded_table: '表格结构识别', audit_uploaded_table: '原始数据审查',
    export_literature_evidence: '导出文献证据表', export_paper_analysis: '导出论文解读',
    export_experiment_design: '导出实验设计', save_research_note: '保存课题记录',
    recall_research_notes: '回顾课题记录', export_research_notes: '导出课题记录',
    export_corpus_gap_report: '导出语料缺口清单'
  };
  return names[name] || String(name || 'tool').replace(/_/g, ' ');
}
function escapeHtmlCode(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function renderSvgBlock(source) {
  var raw = String(source || '').trim();
  if (!raw || raw.length > 100000 || /<!DOCTYPE|<!ENTITY/i.test(raw)) {
    return '<div class="svg-error">SVG 图形无效或尺寸超过限制。</div>';
  }
  try {
    var doc = new DOMParser().parseFromString(raw, 'image/svg+xml');
    var root = doc.documentElement;
    var ns = 'http://www.w3.org/2000/svg';
    if (!root || root.localName.toLowerCase() !== 'svg' ||
        (root.namespaceURI && root.namespaceURI !== ns) ||
        doc.getElementsByTagName('parsererror').length) throw new Error('invalid SVG');
    var allowedTags = new Set(['svg','g','path','rect','circle','ellipse','line','polyline','polygon',
      'text','tspan','title','desc','defs','lineargradient','radialgradient','stop','clippath','marker',
      'use','symbol','filter','fegaussianblur','feoffset','feflood','fecomposite','femerge',
      'femergenode','fecolormatrix','feblend','fecomponenttransfer','fefunca','fefuncr','fefuncg','fefuncb']);
    var attrNames = {
      'viewbox':'viewBox','preserveaspectratio':'preserveAspectRatio','stroke-width':'stroke-width',
      'stroke-linecap':'stroke-linecap','stroke-linejoin':'stroke-linejoin','stroke-dasharray':'stroke-dasharray',
      'stroke-dashoffset':'stroke-dashoffset','fill-opacity':'fill-opacity','stroke-opacity':'stroke-opacity',
      'font-family':'font-family','font-size':'font-size','font-weight':'font-weight','text-anchor':'text-anchor',
      'dominant-baseline':'dominant-baseline','gradientunits':'gradientUnits','gradienttransform':'gradientTransform',
      'stop-color':'stop-color','stop-opacity':'stop-opacity','clippathunits':'clipPathUnits',
      'markerwidth':'markerWidth','markerheight':'markerHeight','markerunits':'markerUnits',
      'marker-start':'marker-start','marker-mid':'marker-mid','marker-end':'marker-end','aria-label':'aria-label',
      'clip-path':'clip-path','role':'role','id':'id','x':'x','y':'y','x1':'x1','y1':'y1','x2':'x2','y2':'y2','cx':'cx','cy':'cy',
      'r':'r','rx':'rx','ry':'ry','width':'width','height':'height','d':'d','points':'points',
      'fill':'fill','stroke':'stroke','opacity':'opacity','transform':'transform','offset':'offset',
      'style':'style','class':'class','href':'href','xlink:href':'href','filter':'filter'
    };
    var safeStyleProps = new Set(['fill','stroke','stroke-width','stroke-linecap','stroke-linejoin',
      'stroke-dasharray','stroke-dashoffset','fill-opacity','stroke-opacity','opacity','font-family',
      'font-size','font-weight','text-anchor','dominant-baseline','stop-color','stop-opacity',
      'clip-path','filter','marker-start','marker-mid','marker-end']);
    function safeStyle(value) {
      var kept = [];
      String(value).split(';').forEach(function(part) {
        var colon = part.indexOf(':');
        if (colon < 1) return;
        var name = part.slice(0, colon).trim().toLowerCase();
        var val = part.slice(colon + 1).trim();
        if (!safeStyleProps.has(name) || !val || /[<>\\{}]/.test(val) ||
            /expression|javascript:|data:|@import/i.test(val)) return;
        if (/url\s*\(/i.test(val) && !/^url\(\s*#[A-Za-z][\w:.-]{0,100}\s*\)$/i.test(val)) return;
        kept.push(name + ':' + val);
      });
      return kept.join(';');
    }
    var count = 0;
    function copyNode(node) {
      if (++count > 5000) throw new Error('SVG complexity limit');
      if (node.nodeType === 3 || node.nodeType === 4) return document.createTextNode(node.nodeValue || '');
      if (node.nodeType !== 1 || (node.namespaceURI && node.namespaceURI !== ns) ||
          !allowedTags.has(node.localName.toLowerCase())) return null;
      var out = document.createElementNS(ns, node.localName);
      for (var i = 0; i < node.attributes.length; i++) {
        var attr = node.attributes[i], key = attr.name.toLowerCase(), canonical = attrNames[key];
        if (!canonical) continue;
        var value = attr.value;
        if (value.length > 4000 || /[<>\u0000-\u001f]/.test(value)) continue;
        if (canonical === 'style') { value = safeStyle(value); if (!value) continue; }
        if (canonical === 'class' && !/^[A-Za-z0-9 _-]{1,200}$/.test(value)) continue;
        if (canonical === 'href' && !/^#[A-Za-z][\w:.-]{0,100}$/.test(value)) continue;
        if (canonical === 'id' && !/^[A-Za-z][\w:.-]{0,100}$/.test(value)) continue;
        if (/url\s*\(/i.test(value) && !/^url\(\s*#[A-Za-z][\w:.-]{0,100}\s*\)$/i.test(value)) continue;
        if (/javascript:|data:/i.test(value)) continue;
        out.setAttribute(canonical, value);
      }
      for (var child = node.firstChild; child; child = child.nextSibling) {
        var safeChild = copyNode(child);
        if (safeChild) out.appendChild(safeChild);
      }
      return out;
    }
    var safe = copyNode(root);
    if (!safe.getAttribute('viewBox') && !safe.getAttribute('width')) {
      throw new Error('SVG needs a viewBox or width');
    }
    return '<figure class="svg-figure" role="group">' + new XMLSerializer().serializeToString(safe) + '</figure>';
  } catch (_) {
    return '<div class="svg-error">SVG 图形无法安全显示，请检查 SVG 代码。</div>';
  }
}

function renderMarkdownSafe(text) {
  // Accept bare SVG responses as well as fenced ```svg blocks.
  var svgBlocks = [];
  var src = String(text == null ? '' : text);
  function svgToken(svg) {
    var token = 'CITRUS_SAFE_SVG_' + svgBlocks.length + '_TOKEN';
    svgBlocks.push({ token: token, html: renderSvgBlock(svg) });
    return token;
  }
  src = src.replace(/```svg[ \t]*\n([\s\S]*?)```/gi, function(_, svg) { return svgToken(svg); });
  src = src.replace(/<svg\b[\s\S]*?<\/svg\s*>/gi, function(svg) { return svgToken(svg); });
  var renderer = new marked.Renderer();
  var origCode = renderer.code;
  renderer.code = function(code, lang) {
    // v8.4.8: 兼容 marked>=5 的 token 对象签名（code 可能为 {text:...}），
    // 否则代码块会渲染成 "[object Object]"
    if (code && typeof code === 'object') { lang = code.lang || ''; code = code.text || ''; }
    if (String(lang || '').trim().toLowerCase() === 'svg') return renderSvgBlock(code);
    var cls = lang ? ' class="lang-' + escapeHtml(lang) + '"' : '';
    return '<pre><code' + cls + '>' + escapeHtmlCode(code) + '</code></pre>';
  };
  renderer.codespan = function(code) {
    if (code && typeof code === 'object') { code = code.text || ''; }
    return '<code>' + escapeHtmlCode(code) + '</code>';
  };
  // SVG is accepted only through the fenced-block sanitizer; escape arbitrary raw HTML.
  renderer.html = function(html) { return escapeHtml(html); };
  // v10.0: 回答不使用删除线——防模型输出 ~~...~~ 被 marked 渲染成"横线划掉"观感
  src = src.replace(/~~/g, '');
  var rendered = marked.parse(src, { renderer: renderer, breaks: true });
  svgBlocks.forEach(function(block) {
    rendered = rendered.split('<p>' + block.token + '</p>').join(block.html);
    rendered = rendered.split(block.token).join(block.html);
  });
  return rendered;
}

/* ─── v8.4.3: 运行时配置（上下文预算/权限模式单一来源，删本地硬编码）─── */
var _runtimeConfig = null;
var _lastStreamOnce = null;   /* 权限授权后重放当前请求的钩子 */
var _pendingPermission = null;
function _ctxMaxTokens() {
  if (_runtimeConfig && _runtimeConfig.context && _runtimeConfig.context.max_tokens) {
    return _runtimeConfig.context.max_tokens;
  }
  return 1000000;
}
var _permWaitSec = 90;   /* v8.4.12: 审批卡片超时提示（config 下发，缺省 90s） */
/* v8.5.0 开源版：未配置 API Key 时显示引导遮罩 */
function maybeShowApiSetup(cfg) {
  if (!currentAccount || currentAccount.role !== 'admin') return;
  var overlay = document.getElementById('apiSetupOverlay');
  if (!overlay) return;
  fillModelConfigFields(cfg);
  var closeBtn = document.getElementById('settingsCloseBtn');
  if (cfg && (cfg.any_model_provider_configured || cfg.has_api_key)) {
    if (closeBtn) closeBtn.style.display = 'inline-block';
  } else {
    overlay.style.display = 'flex';
    if (closeBtn) closeBtn.style.display = 'none';
    var input = document.getElementById('apiKeyInput');
    if (input) input.focus();
  }
}
var _providerModelLists = {};
var _providerCatalog = {
  deepseek: { name: 'DeepSeek', base_url: 'https://api.deepseek.com', default_model: 'deepseek-v4-flash' },
  opencode_go: { name: 'OpenCode Go', base_url: 'https://opencode.ai/inference/go/openai/v1', default_model: 'deepseek-v4.1-flash' },
  zhipu: { name: '智谱 GLM', base_url: 'https://open.bigmodel.cn/api/paas/v4', default_model: 'glm-4.5-flash' },
  aliyun_bailian: { name: '阿里云百炼（通义千问）', base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1', default_model: 'qwen-plus' },
  custom_openai: { name: '其他 OpenAI 兼容服务', base_url: 'https://api.example.com/v1', default_model: '' }
};
function loadProviderCatalog() {
  fetch('/api/v2/config/providers').then(function(r) { return r.json(); }).then(function(data) {
    var select = document.getElementById('mainProviderSelect');
    if (!select || !Array.isArray(data.providers)) return;
    var selected = select.value;
    _providerCatalog = {};
    select.innerHTML = '';
    data.providers.forEach(function(provider) {
      _providerCatalog[provider.id] = provider;
      var option = document.createElement('option');
      option.value = provider.id;
      option.textContent = provider.name + (provider.configured ? ' · 已配置' : '');
      select.appendChild(option);
    });
    if (_providerCatalog[selected]) select.value = selected;
    else if (_runtimeConfig && _runtimeConfig.model && _providerCatalog[_runtimeConfig.model.provider_id])
      select.value = _runtimeConfig.model.provider_id;
    renderConfiguredProviders();
    onProviderChange(true);
  }).catch(function() {});
}
function hasConfiguredProvider(providerId) {
  if ((_providerCatalog[providerId] || {}).configured) return true;
  var ids = (_runtimeConfig && _runtimeConfig.configured_model_provider_ids) || [];
  return ids.indexOf(providerId) >= 0 || !!(_runtimeConfig && _runtimeConfig.has_api_key &&
    _runtimeConfig.model && _runtimeConfig.model.provider_id === providerId);
}
function renderConfiguredProviders() {
  var box = document.getElementById('configuredProvidersList');
  if (!box) return;
  var configured = Object.keys(_providerCatalog).map(function(id) {
    return _providerCatalog[id];
  }).filter(function(provider) { return provider.configured; });
  box.innerHTML = '';
  box.hidden = configured.length === 0;
  if (!configured.length) return;
  var title = document.createElement('div');
  title.className = 'configured-providers-title';
  title.textContent = '已配置服务商';
  box.appendChild(title);
  configured.forEach(function(provider) {
    var row = document.createElement('div');
    row.className = 'configured-provider-row';
    var name = document.createElement('span');
    name.className = 'configured-provider-name';
    name.textContent = provider.name;
    var status = document.createElement('span');
    status.className = 'configured-provider-state';
    status.textContent = provider.id === ((_runtimeConfig || {}).model || {}).provider_id ? '● 当前使用' : '● 已保存';
    var edit = document.createElement('button');
    edit.type = 'button'; edit.className = 'configured-provider-edit'; edit.textContent = '编辑';
    edit.onclick = function() {
      var select = document.getElementById('mainProviderSelect');
      if (select) { select.value = provider.id; onProviderChange(true); }
    };
    row.appendChild(name); row.appendChild(status); row.appendChild(edit);
    box.appendChild(row);
  });
}
function onProviderChange(preserveConfiguredModel) {
  var id = document.getElementById('mainProviderSelect').value;
  var info = _providerCatalog[id] || _providerCatalog.deepseek;
  var hint = document.getElementById('providerEndpointHint');
  var baseInput = document.getElementById('mainApiBaseUrlInput');
  var models = document.getElementById('mainModelSelect');
  var priorProvider = document.getElementById('mainProviderSelect').dataset.lastProvider;
  if (!preserveConfiguredModel && priorProvider && priorProvider !== id) {
    var apiKeyField = document.getElementById('apiKeyInput');
    if (apiKeyField) apiKeyField.value = '';
    var apiKeyError = document.getElementById('apiKeyError');
    if (apiKeyError) apiKeyError.textContent = '';
  }
  document.getElementById('mainProviderSelect').dataset.lastProvider = id;
  if (hint) hint.textContent = 'API 地址：' + info.base_url;
  var keyInput = document.getElementById('apiKeyInput');
  if (keyInput) keyInput.placeholder = hasConfiguredProvider(id)
    ? '已保存；留空继续使用' : '输入服务商 API Key';
  var current = ((_runtimeConfig || {}).model || {});
  if (baseInput) baseInput.value = preserveConfiguredModel && current.provider_id === id && current.main_base_url
    ? current.main_base_url : info.base_url;
  if (baseInput) baseInput.readOnly = !info.custom_base_url;
  if (models) {
    var showCurrent = preserveConfiguredModel && current.provider_id === id && current.main;
    models.innerHTML = '';
    var option = document.createElement('option');
    if (showCurrent) {
      option.value = current.main;
      var freeModels = (info.free_models || []).map(function(x) { return String(x).toLowerCase(); });
      option.textContent = current.main + (freeModels.indexOf(String(current.main).toLowerCase()) >= 0 ? ' · 免费' : '') + '（当前配置）';
      models.appendChild(option);
      models.value = current.main;
    } else {
      option.value = '';
      option.textContent = hasConfiguredProvider(id) ? '点击“获取模型列表”选择模型' : '先填写 Key 并获取模型列表';
      models.appendChild(option);
    }
    models.disabled = true;
  }
  if (!preserveConfiguredModel) _providerModelLists[id] = null;
}
function fetchProviderModels() {
  var providerId = document.getElementById('mainProviderSelect').value;
  var keyInput = document.getElementById('apiKeyInput');
  var key = (keyInput && keyInput.value || '').trim();
  var err = document.getElementById('apiKeyError');
  var btn = document.getElementById('providerModelsBtn');
  if (!key && !hasConfiguredProvider(providerId)) {
    if (err) err.textContent = '请填写所选服务商对应的 API Key'; return;
  }
  if (btn) btn.disabled = true;
  if (err) { err.className = 'api-error'; err.textContent = '正在向 ' + ((_providerCatalog[providerId] || {}).name || providerId) + ' 获取模型列表…'; }
  fetch('/api/v2/config/provider-models', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider_id: providerId, api_key: key,
      api_base_url: (document.getElementById('mainApiBaseUrlInput').value || '').trim() })
  }).then(function(r) { return r.json().then(function(d) { return { ok: r.ok, d: d }; }); })
    .then(function(res) {
      if (!res.ok) throw new Error((res.d && res.d.detail) || '获取模型列表失败');
      var ids = res.d.models || [];
      _providerModelLists[providerId] = ids;
      var select = document.getElementById('mainModelSelect');
      if (select) {
        select.innerHTML = '';
        ids.forEach(function(id) {
          var opt = document.createElement('option'); opt.value = id;
          var freeModels = ((_providerCatalog[providerId] || {}).free_models || []).map(function(x) { return String(x).toLowerCase(); });
          opt.textContent = id + (freeModels.indexOf(String(id).toLowerCase()) >= 0 ? ' · 免费' : '');
          select.appendChild(opt);
        });
        select.disabled = ids.length === 0;
        var activeModel = ((_runtimeConfig || {}).model || {});
        var preferred = activeModel.provider_id === providerId ? activeModel.main : '';
        if (ids.indexOf(preferred) >= 0) select.value = preferred;
        else if (ids.indexOf((_providerCatalog[providerId] || {}).default_model) >= 0) select.value = _providerCatalog[providerId].default_model;
      }
      if (err) { err.className = 'api-warn'; err.textContent = '已取得 ' + ids.length + ' 个可用模型，请选择后保存。'; }
    }).catch(function(e) {
      if (err) { err.className = 'api-error'; err.textContent = e.message; }
    }).finally(function() { if (btn) btn.disabled = false; });
}
function saveProviderConfig() {
  var err = document.getElementById('modelConfigError');
  var btn = document.getElementById('modelConfigSave');
  var providerId = document.getElementById('mainProviderSelect').value;
  var key = (document.getElementById('apiKeyInput').value || '').trim();
  var model = document.getElementById('mainModelSelect').value;
  var contextTokens = parseInt(document.getElementById('modelContextWindowInput').value, 10) || 0;
  if (!key && !hasConfiguredProvider(providerId)) {
    if (err) err.textContent = '请填写 Provider API Key'; return;
  }
  if (!_providerModelLists[providerId] || _providerModelLists[providerId].indexOf(model) < 0) {
    if (err) err.textContent = '请先获取模型列表并选择其中一个模型'; return;
  }
  if (btn) btn.disabled = true;
  if (err) { err.className = 'api-error'; err.textContent = '正在验证并保存…'; }
  fetch('/api/v2/config/provider', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider_id: providerId, api_key: key, main_model: model,
      api_base_url: (document.getElementById('mainApiBaseUrlInput').value || '').trim(),
      context_window_tokens: contextTokens })
  }).then(function(r) { return r.json().then(function(d) { return { ok: r.ok, d: d }; }); })
    .then(function(res) {
      if (!res.ok) throw new Error((res.d && res.d.detail) || '保存失败');
      if (err) { err.className = 'api-warn'; err.textContent = '配置已保存：' + ((_providerCatalog[providerId] || {}).name) + ' / ' + model; }
      document.getElementById('apiKeyInput').value = '';
      loadRuntimeConfig();
    }).catch(function(e) {
      if (err) { err.className = 'api-error'; err.textContent = e.message; }
    }).finally(function() { if (btn) btn.disabled = false; });
}
function fillModelConfigFields(cfg) {
  var m = (cfg && cfg.model) || {};
  function setv(id, v) { var el = document.getElementById(id); if (el) el.value = v || ''; }
  setv('mainProviderSelect', m.provider_id || 'deepseek');
  var keyInput = document.getElementById('apiKeyInput');
  if (keyInput) keyInput.placeholder = hasConfiguredProvider(m.provider_id || 'deepseek')
    ? '已保存；留空继续使用' : '输入服务商 API Key';
  setv('modelContextWindowInput', m.context_window_tokens);
  onProviderChange(true);
  var web = (cfg && cfg.web_search) || {};
  var webProvider = document.getElementById('webSearchProviderSelect');
  if (webProvider) webProvider.value = web.provider || 'deepseek_search_fetch';
  onWebSearchProviderChange();
}
function onWebSearchProviderChange() {
  var provider = document.getElementById('webSearchProviderSelect');
  var key = document.getElementById('webSearchApiKeyInput');
  var fetchKey = document.getElementById('webFetchApiKeyInput');
  var fetchKeyLabel = document.getElementById('webFetchApiKeyLabel');
  var hint = document.getElementById('webSearchKeyHint');
  var web = ((_runtimeConfig || {}).web_search || {});
  var selected = provider ? provider.value : 'deepseek_search_fetch';
  if (key) key.placeholder = web.configured && web.provider === selected
    ? '已配置；留空继续使用' : '输入对应服务商的 Key';
  if (fetchKey) fetchKey.hidden = selected !== 'zhipu_web_search';
  if (fetchKeyLabel) fetchKeyLabel.hidden = selected !== 'zhipu_web_search';
  if (hint) hint.textContent = selected === 'zhipu_web_search'
    ? (web.deepseek_fetch_configured ? 'Web Fetch 已配置 DeepSeek Key。' : '搜索使用智谱；如需网页全文读取，可另填 DeepSeek Fetch Key。')
    : 'Search 与 Fetch 均使用 DeepSeek Search/Fetch Key。';
}
function saveWebSearchConfig() {
  var provider = document.getElementById('webSearchProviderSelect').value;
  var input = document.getElementById('webSearchApiKeyInput');
  var key = (input && input.value || '').trim();
  var fetchKey = (document.getElementById('webFetchApiKeyInput').value || '').trim();
  var status = document.getElementById('webSearchConfigError');
  var btn = document.getElementById('webSearchConfigSave');
  var existing = ((_runtimeConfig || {}).web_search || {});
  if (!key && !(existing.configured && existing.provider === provider)) {
    if (status) status.textContent = '请填写所选服务商的搜索 Key';
    return;
  }
  if (btn) btn.disabled = true;
  if (status) { status.className = 'api-warn'; status.textContent = '正在保存…'; }
  fetch('/api/v2/config/web-apikey', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider_id: provider, api_key: key, fetch_api_key: fetchKey })
  }).then(function(r) { return r.json().then(function(d) { return { ok: r.ok, d: d }; }); })
    .then(function(res) {
      if (!res.ok) throw new Error((res.d && res.d.detail) || '联网搜索配置保存失败');
      if (input) input.value = '';
      var fetchInput = document.getElementById('webFetchApiKeyInput');
      if (fetchInput) fetchInput.value = '';
      if (status) { status.className = 'api-warn'; status.textContent = '已保存'; }
      return fetch('/api/v2/config').then(function(r) { return r.json(); });
    }).then(function(cfg) {
      _runtimeConfig = cfg;
      onWebSearchProviderChange();
    }).catch(function(e) {
      if (status) { status.className = 'api-error'; status.textContent = e.message; }
    }).finally(function() { if (btn) btn.disabled = false; });
}
function openSettingsModal() {
  var overlay = document.getElementById('apiSetupOverlay');
  if (!overlay) return;
  overlay.style.display = 'flex';
  fillModelConfigFields(_runtimeConfig);
  var closeBtn = document.getElementById('settingsCloseBtn');
  if (closeBtn) closeBtn.style.display = (_runtimeConfig &&
    (_runtimeConfig.any_model_provider_configured || _runtimeConfig.has_api_key)) ? 'inline-block' : 'none';
  var keyInput = document.getElementById('apiKeyInput');
  if (!(_runtimeConfig && (_runtimeConfig.any_model_provider_configured || _runtimeConfig.has_api_key)) && keyInput) keyInput.focus();
}
function closeSettingsModal() {
  var overlay = document.getElementById('apiSetupOverlay');
  if (overlay) overlay.style.display = 'none';
}
function loadRuntimeConfig() {
  loadProviderCatalog();
  return fetch('/api/v2/config').then(function(r) { return r.json(); })
    .then(function(cfg) {
      _runtimeConfig = cfg;
      if (cfg.permission_wait_sec) _permWaitSec = cfg.permission_wait_sec;
      applyAppVersion();
      maybeShowApiSetup(cfg);
    })
    .catch(function() { applyAppVersion(); });
}
/* v8.4.5: 版本单源——后端 /api/v2/config.version 下发，静态文案仅为离线回退 */
function _appVersion() {
  return (_runtimeConfig && _runtimeConfig.version) || '1.0';
}
function applyAppVersion() {
  var v = _appVersion();
  var badge = document.getElementById('versionBadge');
  if (badge) badge.textContent = 'v' + v;
  var w = document.getElementById('welcomeVersion');
  if (w) w.textContent = 'CitrusQA v' + v;
  updatePermBadge();
}
/* v8.4.8 HITL 服务端化：权限模式徽标与 WebUI 切换
   v8.10f: SVG 简笔线条图标——护盾+对号(全权)/护盾+笔(询问)/护盾+感叹号(只读)，
   currentColor 描边不渲染色彩 */
var _permMode = null;
var _SHIELD = '<path d="M12 3l7 2.8v5.2c0 4.3-2.9 7.8-7 9.5-4.1-1.7-7-5.2-7-9.5V5.8z"/>';
function _permSvg(inner) {
  return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">' + inner + '</svg>';
}
var _permMeta = {
  'auto_workspace': { ico: _permSvg(_SHIELD + '<path d="M8.6 12.4l2.3 2.3 4.5-4.8"/>'), name: '全权', label: '全权（工作区免询问）' },
  'ask': { ico: _permSvg(_SHIELD + '<path d="M14.4 6.2l3.4 3.4L9.5 18H7.5v-2z"/><path d="M13 8.6l2.4 2.4"/>'), name: '询问', label: '询问（危险操作审批）' },
  'deny': { ico: _permSvg(_SHIELD + '<path d="M12 8.2v5.2"/><path d="M12 16.6h.01"/>'), name: '只读', label: '只读（一律拒绝）' },
};
function updatePermBadge() {
  // v8.10j: 仅首次（_permMode 未初始化）用后端下发值；切换后 _permMode 优先——
  // 旧逻辑每帧用缓存的 runtimeConfig 覆盖新值，导致切换后徽标不刷新（需刷新页面才正确）
  if (_permMode === null && _runtimeConfig && _runtimeConfig.permission_mode) _permMode = _runtimeConfig.permission_mode;
  var mode = _permMode || 'auto_workspace';
  var meta = _permMeta[mode] || _permMeta['auto_workspace'];
  var b = document.getElementById('permBadge');
  var ico = document.getElementById('permIco');
  var nm = document.getElementById('permName');
  if (ico) ico.innerHTML = meta.ico;
  if (nm) nm.textContent = meta.name;
  if (b) { b.className = 'perm-badge ' + mode; b.title = '权限模式（HITL）：' + meta.label + '，点击切换'; }
  var items = document.querySelectorAll('.perm-menu-item');
  for (var i = 0; i < items.length; i++) {
    items[i].className = 'perm-menu-item' + (items[i].getAttribute('data-mode') === mode ? ' active' : '');
  }
}
function togglePermMenu(ev) {
  var m = document.getElementById('permMenu');
  if (ev) ev.stopPropagation();
  m.style.display = m.style.display === 'none' ? 'block' : 'none';
}
document.addEventListener('click', function() {
  var m = document.getElementById('permMenu');
  if (m) m.style.display = 'none';
});
function setPermMode(mode) {
  if (!confirm('切换权限模式为「' + ((_permMeta[mode] || {}).label || mode) + '」？\n询问模式下写文件需在审批卡片中点击允许。')) return;
  fetch('/api/v2/permission/mode', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode: mode }),
  }).then(function(r) { return r.json(); }).then(function(d) {
    if (d && d.permission_mode) { _permMode = d.permission_mode; updatePermBadge(); }
    document.getElementById('permMenu').style.display = 'none';
  }).catch(function(e) { alert('切换失败: ' + e.message); });
}

/* ─── v8.4.3/8.4.12: 权限审批卡片（授权动作不写入对话历史）───
   v8.4.12: 全 CSS 变量驱动（浅/深色模式一致），会话窗口上方居中弹出 */
function showPermissionCard(data) {
  _pendingPermission = data;
  var existing = document.getElementById('permission-card');
  if (existing) existing.remove();
  var card = document.createElement('div');
  card.id = 'permission-card';
  card.className = 'perm-card';
  var tool = data.tool_name || '工具';
  var args = data.args || {};
  var argText = Object.keys(args).map(function(k) { return k + ': ' + String(args[k]).substring(0, 120); }).join('<br>');
  card.innerHTML =
    '<div class="perm-title">⚠ 权限确认</div>' +
    '<div class="perm-tool">工具: <code>' + escapeHtml(tool) + '</code></div>' +
    (argText ? '<div class="perm-args">' + argText + '</div>' : '') +
    '<div class="perm-actions">' +
    '<button class="primary" data-scope="once">允许本次</button>' +
    '<button data-scope="session">本会话允许</button>' +
    '<button data-scope="workspace">始终允许工作区</button>' +
    '</div>' +
    '<div class="perm-hint">未响应将按拒绝处理（' + (_permWaitSec || 90) + 's 超时）</div>';
  card.querySelectorAll('button').forEach(function(btn) {
    btn.onclick = function() {
      grantPermission(tool, btn.getAttribute('data-scope'));
    };
  });
  document.body.appendChild(card);
}

function grantPermission(toolName, scope) {
  fetch('/api/v2/permission/grant', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, tool_name: toolName, scope: scope }),
  }).then(function(r) { return r.json(); }).then(function() {
    var card = document.getElementById('permission-card');
    if (card) card.remove();
    _pendingPermission = null;
    // v8.10j: 不再重放流——v8.4.5 起后端为"挂起等待授权→同一执行内继续"，
    // 授权后原请求自动继续输出；前端重放会造成双通道（二次权限检查 + 二次回答）
  }).catch(function(e) { alert('授权失败: ' + e.message); });
}

/* ─── v10.0（卡点①）意图澄清卡：LLM 判定歧义后弹出，中文候选 + 自填；超时按推荐项放行 ─── */
function showIntentCard(data) {
  var existing = document.getElementById('intent-card');
  if (existing) existing.remove();
  var card = document.createElement('div');
  card.id = 'intent-card';
  card.className = 'perm-card';
  var opts = data.options || [];
  var html = '<div class="perm-title">🤔 确认一下你的问题</div>';
  if (data.intent_summary) {
    html += '<div class="intent-question">我理解你想问：' + escapeHtml(data.intent_summary) + '</div>';
  }
  if (data.reason) {
    html += '<div class="perm-hint">存在歧义：' + escapeHtml(data.reason) + '</div>';
  }
  for (var i = 0; i < opts.length; i++) {
    var o = opts[i] || {};
    var label = escapeHtml(o.text || '');
    if (o.caliber) {
      label += '<br><span class="perm-hint">口径：' + escapeHtml(o.caliber) + '</span>';
    }
    html += '<button class="intent-opt" data-id="' + (o.id != null ? o.id : (i + 1)) + '">'
          + (i + 1) + '. ' + label + '</button>';
  }
  html += '<div class="intent-custom">'
        + '<input id="intent-custom-input" type="text" placeholder="以上都不是，自己说明一下…" />'
        + '<button class="primary" id="intent-custom-send">提交</button></div>'
        + '<div class="perm-hint">' + (data.timeout_sec || 30) + 's 未选择将按第 1 项继续</div>';
  card.innerHTML = html;
  card.querySelectorAll('.intent-opt').forEach(function(btn) {
    btn.onclick = function() {
      confirmIntent(data.token, btn.getAttribute('data-id'), '');
    };
  });
  var send = card.querySelector('#intent-custom-send');
  if (send) {
    send.onclick = function() {
      var el = card.querySelector('#intent-custom-input');
      var v = (el && el.value ? el.value : '').trim();
      if (!v) return;
      confirmIntent(data.token, null, v);
    };
  }
  document.body.appendChild(card);
}

function confirmIntent(token, choiceId, customText) {
  fetch('/api/v2/intent/confirm', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      token: token,
      // data-id 来自按钮属性（字符串）→ 归一为数字，NaN 转 null（后端 Optional[int]）
      choice_id: (function(v) {
        if (v === null || v === undefined || v === '') return null;
        var n = parseInt(v, 10);
        return isNaN(n) ? null : n;
      })(choiceId),
      custom_text: customText || null,
    }),
  }).then(function(r) { return r.json(); }).then(function() {
    var card = document.getElementById('intent-card');
    if (card) card.remove();
    // 后端为"挂起等待→同一请求内继续"，选择后原 SSE 流自动继续，无需重放
  }).catch(function(e) { alert('提交失败: ' + e.message); });
}

/* ─── Theme ─── */
(function() {
  var t = localStorage.getItem('citrus_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', t);
})();

function toggleTheme() {
  var html = document.documentElement;
  var cur = html.getAttribute('data-theme');
  var next = cur === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  localStorage.setItem('citrus_theme', next);
}

/* ─── AbortController for SSE (cancel inflight on new send) ─── */
var currentAbortController = null;

/* ─── Plan Mode state (synced from server flags) ─── */
var planModeEnabled = true;

function setSendDisabled(val) {
  var btn = document.getElementById('send-btn');
  var stopBtn = document.getElementById('stop-btn');
  // v8.7: 运行中彻底隐藏发送按钮（此前仅 disabled——两个绝对定位按钮
  // 同位置叠加造成视觉重叠）；恢复时再显示。停止按钮反向切换。
  if (btn) { btn.disabled = val; btn.style.display = val ? 'none' : 'flex'; }
  if (stopBtn) stopBtn.style.display = val ? 'flex' : 'none';
}

/* v8.4.11 用户中断（书中 §4.7.6 取消式处理）：
   1) abort SSE 流（服务端普通问答在断开时取消 graph 任务）
   2) POST /api/v2/chat/cancel 兜底取消该会话所有 running job
      （覆盖 write 断连保活任务——此类任务客户端断开后仍在后台执行）
   停止后可修改输入框内容直接重新发送（新请求 = 新 job）。 */
function stopCurrentTask() {
  if (currentAbortController) {
    currentAbortController.abort();
    currentAbortController = null;
  }
  if (sessionId) {
    try {
      fetch('/api/v2/chat/cancel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      }).catch(function() {});
    } catch (e) {}
  }
  setSendDisabled(false);
  userInput.focus();
  userInput.placeholder = '已停止——可修改问题后按 Enter 重新发送';
  var stopMsg = document.getElementById('stop-msg');
  if (stopMsg) {
    stopMsg.style.display = '';
  } else if (currentRoundId > 0) {
    var div = document.createElement('div');
    div.id = 'stop-msg';
    div.style.cssText = 'color:var(--error);font-weight:500;font-size:13px;padding:4px 0;margin-top:6px';
    div.textContent = '⏹ 已停止当前任务（输入有误可修改后重新发送）';
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }
}

/* ─── Sidebar ─── */
var _sidebarMap = { session: 'session-list' };
var _sidebarSections = ['session'];

function setSidebarVisible(visible) {
  var shell = document.getElementById('mainContainer');
  var button = document.getElementById('sidebarToggleBtn');
  if (!shell || !button) return;
  shell.classList.toggle('sidebar-hidden', !visible);
  button.setAttribute('aria-expanded', visible ? 'true' : 'false');
  button.setAttribute('aria-label', visible ? '隐藏会话侧栏' : '显示会话侧栏');
  button.title = visible ? '隐藏会话侧栏' : '显示会话侧栏';
  localStorage.setItem('citrus_sidebar_visible', visible ? '1' : '0');
}

function toggleSidebar() {
  var shell = document.getElementById('mainContainer');
  setSidebarVisible(!!shell && shell.classList.contains('sidebar-hidden'));
}

(function restoreSidebarVisibility() {
  var saved = localStorage.getItem('citrus_sidebar_visible');
  setSidebarVisible(saved === null ? window.innerWidth > 1024 : saved === '1');
})();

function applySidebar(active) {
  for (var i = 0; i < _sidebarSections.length; i++) {
    var s = _sidebarSections[i];
    var el = document.getElementById(_sidebarMap[s]);
    var icon = document.getElementById(s + '-toggle');
    var sec = document.getElementById(s + '-section');
    if (!el || !icon || !sec) continue;
    var isActive = (active === s);
    el.classList.toggle('collapsed', !isActive);
    icon.classList.toggle('collapsed', !isActive);
    sec.classList.toggle('expanded', isActive);
  }
  localStorage.setItem('citrus_sidebar_active', active || '');
}

function toggleSection(section) {
  var el = document.getElementById(_sidebarMap[section]);
  if (!el) return;
  var isCollapsed = el.classList.contains('collapsed');
  applySidebar(isCollapsed ? section : null);
}

/* 会话列表默认展开。 */
applySidebar('session');

/* ─── v8.9 会话管理（列表/切换/新建/重命名/软删除）+ 工作区文件 ─── */
var _sessions = [];
var _sessionsVisible = 20;   // v8.9: 默认只显示最近 20 个，加载更多展开

function loadSessions() {
  return fetch('/api/v2/sessions').then(function (r) { return r.json(); }).then(function (d) {
    _sessions = (d && d.sessions) || [];
    renderSessions();
    return true;
  }).catch(function () { return false; });
}

/* v8.10e: 会话列表加载失败自动重试（网络抖动/服务重启竞态），最多 3 次后停止 */
function loadSessionsWithRetry(retries) {
  var n = retries === undefined ? 3 : retries;
  return loadSessions().then(function (ok) {
    if (!ok && n > 0) {
      return new Promise(function (res) {
        setTimeout(function () { res(loadSessionsWithRetry(n - 1)); }, 600);
      });
    }
    return ok;
  });
}

function fmtSessionTime(iso) {
  if (!iso) return '';
  var d = new Date(iso);
  if (isNaN(d.getTime())) return '';
  var now = new Date();
  var hh = String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
  if (d.toDateString() === now.toDateString()) return '今天 ' + hh;
  var yest = new Date(now); yest.setDate(now.getDate() - 1);
  if (d.toDateString() === yest.toDateString()) return '昨天 ' + hh;
  var mm = String(d.getMonth() + 1).padStart(2, '0');
  var dd = String(d.getDate()).padStart(2, '0');
  return mm + '-' + dd + ' ' + hh;
}

function renderSessions() {
  var box = document.getElementById('session-list');
  var count = document.getElementById('session-count');
  if (!box) return;
  // v8.9: 过滤空会话（无消息的测试/残留不显示）；当前活动会话总是显示
  var visible = [];
  for (var i = 0; i < _sessions.length; i++) {
    var s = _sessions[i];
    if (s.msg_count > 0 || s.session_id === sessionId) visible.push(s);
  }
  if (count) count.textContent = visible.length;
  if (visible.length === 0) {
    box.innerHTML = '<div class="empty-sidebar">暂无会话，点击 ＋ 新建</div>';
    return;
  }
  var shown = visible.slice(0, _sessionsVisible);
  var html = '';
  for (var i = 0; i < shown.length; i++) {
    var s = shown[i];
    var active = s.session_id === sessionId ? ' active' : '';
    var title = escapeHtml(s.title || '新会话');
    html += '<div class="session-item' + active + '">' +
      '<div class="session-title" onclick="switchSession(\'' + s.session_id + '\')" title="' + title + '">' + title + '</div>' +
      '<span class="session-time">' + fmtSessionTime(s.updated_at) + '</span>' +
      '<div class="session-ops">' +
      '<button class="session-op" onclick="sessionMenu(event,\'' + s.session_id + '\')" title="更多">⋯</button>' +
      '</div></div>';
  }
  if (visible.length > _sessionsVisible) {
    html += '<button class="session-more" onclick="loadMoreSessions()">加载更多（还剩 ' +
      (visible.length - _sessionsVisible) + '）</button>';
  }
  box.innerHTML = html;
}

function loadMoreSessions() {
  _sessionsVisible += 20;
  renderSessions();
}

function closeSessionMenu() {
  var m = document.querySelector('.session-menu');
  if (m) m.remove();
}

function sessionMenu(ev, sid) {
  ev.stopPropagation();
  closeSessionMenu();
  var item = ev.currentTarget.closest('.session-item');
  var menu = document.createElement('div');
  menu.className = 'session-menu';
  menu.innerHTML = '<div class="session-menu-item" onclick="sessionMenuAction(\'rename\',\'' + sid + '\')">✎ 重命名</div>' +
                   '<div class="session-menu-item danger" onclick="sessionMenuAction(\'delete\',\'' + sid + '\')">🗑 删除</div>';
  var rect = item ? item.getBoundingClientRect() : { bottom: 80, right: 160 };
  menu.style.top = (rect.bottom + 4) + 'px';
  menu.style.left = Math.max(8, rect.right - 130) + 'px';
  document.body.appendChild(menu);
  setTimeout(function () {
    document.addEventListener('click', function once() {
      closeSessionMenu();
      document.removeEventListener('click', once);
    });
  }, 0);
}

function sessionMenuAction(action, sid) {
  closeSessionMenu();
  if (action === 'rename') renameSession(sid);
  else if (action === 'delete') deleteSession(sid);
}

function switchSession(sid) {
  if (!sid || sid === sessionId) return;
  if (currentAbortController) { currentAbortController.abort(); currentAbortController = null; }
  setSendDisabled(false);
  sessionId = sid;
  localStorage.setItem(sessionStorageKey(), sid);
  clearSentAttachmentPreviewUrls();
  chatContainer.innerHTML = '';
  roundHistory = [];
  _restoredGroups = null;   // v9.2: 切换会话清空恢复视图，等下一次 restore
  _ctxData = null; _totalTokensUsed = 0;
  sidebarContent.innerHTML = '<div class="empty-sidebar">提交问题后文献将在此展示</div>';
  citationCount.textContent = '0';
  renderSessions();
  restoreSessionHistory();
  userInput.focus();
}

function renameSession(sid) {
  var cur = null;
  for (var i = 0; i < _sessions.length; i++) if (_sessions[i].session_id === sid) cur = _sessions[i];
  var title = prompt('重命名会话：', cur ? (cur.title && cur.title !== '新会话' ? cur.title : '') : '');
  if (title === null || !title.trim()) return;
  fetch('/api/v2/session/' + encodeURIComponent(sid) + '/rename', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: title.trim() })
  }).then(function (r) { return r.json(); }).then(function () { loadSessions(); });
}

function deleteSession(sid) {
  var isCur = sid === sessionId;
  if (!confirm('删除会话？\n' + (isCur ? '（当前会话）' : '') + '\n仅从列表隐藏，历史数据保留（可追溯）。')) return;
  fetch('/api/v2/session/' + encodeURIComponent(sid), { method: 'DELETE' })
    .then(function (r) { return r.json(); }).then(function () {
      if (isCur) { newSession(); } else { loadSessions(); }
    });
}

function loadWorkspaceFiles() {
  var box = document.getElementById('workspace-list');
  if (!box) return;
  var reqSid = sessionId;   // v8.10l: 竞态防护——快速切换会话时丢弃过期响应
  box.innerHTML = '<div class="empty-sidebar">加载中...</div>';
  fetch('/api/v2/session/' + encodeURIComponent(sessionId) + '/workspace-files')
    .then(function (r) { return r.json(); }).then(function (d) {
      if (reqSid !== sessionId) return;
      var files = (d && d.files) || [];
      if (files.length === 0) {
        box.innerHTML = '<div class="empty-sidebar">本会话暂无写作成果</div>';
        return;
      }
      var html = '';
      for (var i = 0; i < files.length; i++) {
        var f = files[i];
        // v8.10i: 路径分段编码（整体 encodeURIComponent 会把 / 编成 %2F 导致 404）
        var enc = f.path.split('/').map(encodeURIComponent).join('/');
        html += '<a class="workspace-file" href="/api/v2/session/' + encodeURIComponent(reqSid) + '/workspace-file?path=' + enc +
          '" target="_blank" title="' + escapeHtml(f.path) + '">📄 ' + escapeHtml(f.name) + '</a>';
      }
      box.innerHTML = html;
    }).catch(function () {
      // v8.10k: 失败可点击重试（不再静默）
      box.innerHTML = '<div class="empty-sidebar" style="cursor:pointer" onclick="loadWorkspaceFiles()">加载失败，点击重试</div>';
    });
}

/* v9.2: 会话历史文献引用恢复——按来源 4 组合并（RAG/UCR/Web/历史），
   与 live 侧栏同构（替代 v8.10k 每轮列表 R{turn}-{i}：编号体系与正文不同源、
   点击不可追踪）。后端 /citations 返回 groups，恢复后按 4 组手风琴渲染；
   新提问触发的实时轮次（roundHistory）到达后优先显示实时轮次结构。 */
var _restoredGroups = null;   // {rag:[], ucr:[], web:[], historical:[]}（刷新恢复视图）
var _restoredRefPool = null;  // v9.4.3: 恢复接口版本标记（自证编号池版本）

function restoreSessionCitations() {
  var box = document.getElementById('sidebar-content');
  if (!box || !sessionId) return;
  var reqSid = sessionId;   // v8.10l: 竞态防护——快速切换会话时丢弃过期响应
  fetch('/api/v2/session/' + encodeURIComponent(sessionId) + '/citations')
    .then(function (r) { return r.json(); }).then(function (d) {
      if (reqSid !== sessionId) return;
      var groups = d && d.groups;
      var gTotal = groups ? ((groups.rag || []).length + (groups.ucr || []).length +
                             (groups.web || []).length + (groups.historical || []).length) : 0;
      if (!groups || gTotal === 0) {
        if (roundHistory.length === 0) {
          // v8.10q: save_evidence 是后台 spawn 异步落库——切换会话时最新一轮
          // 证据可能尚未落库（空结果）。仅重试 1 次覆盖时序缺口（避免新会话无限轮询）
          _citationsRetry = (_citationsRetry || 0) + 1;
          box.innerHTML = '<div class="empty-sidebar">提交问题后文献将在此展示</div>';
          citationCount.textContent = '0';
          if (_citationsRetry <= 1) {
            setTimeout(function () {
              if (reqSid === sessionId) restoreSessionCitations();
            }, 800);
          } else {
            _citationsRetry = 0;
          }
        } else {
          _citationsRetry = 0;
        }
        return;
      }
      _citationsRetry = 0;
      _restoredGroups = groups;
      _restoredRefPool = (d && d.ref_pool) || null;
      renderRoundSidebar();
    }).catch(function () { /* 保持现状，不打扰 */ });
}

/* ─── Context Drawer ─── */
var _ctxData = null;
var _totalTokensUsed = 0;
var CTX_BTN_R = 15;
var CTX_BTN_CIRCUM = 2 * Math.PI * CTX_BTN_R;

function toggleCtxDrawer() {
  var drawer = document.getElementById('ctxDrawer');
  var backdrop = document.getElementById('ctxBackdrop');
  var willOpen = !drawer.classList.contains('open');
  drawer.classList.toggle('open');
  backdrop.classList.toggle('open');
  // v8.10: 打开抽屉即拉取上下文细分（真实构成，逐段可展开查看原文）
  if (willOpen) loadCtxDetail();
}

/* v8.10: 上下文细分——system 提示词 / 历史消息(user·ai·tool·system) / 记忆块 */
function loadCtxDetail() {
  var box = document.getElementById('ctxDetail');
  if (!box || !sessionId) return;
  box.innerHTML = '<div class="ctx-sug-item" style="padding:6px 0">加载上下文细分...</div>';
  fetch('/api/v2/session/' + encodeURIComponent(sessionId) + '/context-detail')
    .then(function (r) { return r.json(); })
    .then(function (d) {
      var segs = (d && d.segments) || [];
      // v8.16.4: 端点降级信封（error 标记）→ 温和失败提示，不再误报"暂无上下文"
      if (d && d.error) {
        box.innerHTML = '<div class="ctx-sug-item">⚠ 细分加载失败，请稍后重试（异常已记录）</div>';
        return;
      }
      if (segs.length === 0) {
        box.innerHTML = '<div class="ctx-sug-item" style="padding:6px 0">当前会话暂无上下文</div>';
        return;
      }
      var html = '';
      for (var i = 0; i < segs.length; i++) {
        var s = segs[i];
        html += '<div class="ctx-seg">' +
          '<div class="ctx-seg-head" onclick="toggleCtxSeg(' + i + ')">' +
          '<span class="ctx-seg-role ' + (s.role || 'other') + '">' + (s.role || 'other') + '</span>' +
          '<span class="ctx-seg-src">' + escapeHtml(s.source) + '</span>' +
          '<span class="ctx-seg-chars">' + (s.chars || 0).toLocaleString() + ' 字符</span>' +
          '<span class="ctx-seg-arrow" id="ctx-seg-' + i + '-arrow">▶</span>' +
          '</div>' +
          '<div class="ctx-seg-body" id="ctx-seg-' + i + '" style="display:none"><pre>' + escapeHtml(s.content || '') + '</pre></div>' +
          '</div>';
      }
      box.innerHTML = html;
    })
    .catch(function () { box.innerHTML = '<div class="ctx-sug-item">⚠ 细分加载失败，请稍后重试（异常已记录）</div>'; });
}

function toggleCtxSeg(i) {
  var body = document.getElementById('ctx-seg-' + i);
  var arrow = document.getElementById('ctx-seg-' + i + '-arrow');
  if (!body) return;
  var open = body.style.display !== 'none';
  body.style.display = open ? 'none' : '';
  if (arrow) arrow.textContent = open ? '▶' : '▼';
}

function _ctxLevelClass(pct) {
  if (pct < 0.70) return 'low';
  if (pct < 0.85) return 'mid';
  if (pct < 0.93) return 'high';
  return 'crit';
}

function _updateCtxBtnRing(pct) {
  // v8.10: pct<=0（未使用上下文）渲染系统灰空环，与其他图标一致；有进度才按分级色渲染
  var fg = document.getElementById('ctxBtnFg');
  var dash = CTX_BTN_CIRCUM * (1 - Math.min(pct, 1));
  fg.style.strokeDasharray = CTX_BTN_CIRCUM + ' ' + CTX_BTN_CIRCUM;
  fg.style.strokeDashoffset = dash;
  fg.className = 'ctx-btn-fg ' + (pct <= 0 ? 'none' : _ctxLevelClass(pct));
}

function renderContextPanel(data) {
  _ctxData = data;
  var est = data.estimated_tokens || 0;
  var max = data.max_tokens || _ctxMaxTokens();
  var pct = max > 0 ? est / max : 0;

  _updateCtxBtnRing(pct);

  var emptyEl = document.getElementById('ctx-empty');
  var bodyEl = document.getElementById('ctx-body');
  if (emptyEl) emptyEl.style.display = 'none';
  if (bodyEl) bodyEl.style.display = '';

  var lvl = _ctxLevelClass(pct);
  var bar = document.getElementById('ctxBar');
  var barLabel = document.getElementById('ctxBarLabel');
  if (bar) { bar.style.width = Math.round(pct * 100) + '%'; bar.className = 'ctx-bar-inner ' + lvl; }
  if (barLabel) barLabel.textContent = est.toLocaleString() + ' / ' + max.toLocaleString();
  var pctEl = document.getElementById('ctx-pct');
  if (pctEl) pctEl.textContent = Math.round(pct * 100) + '%';

  var items = [];
  if (est <= 0) {
    items.push({ name: '新会话（无历史消息）', tok: 1 });
    items.push({ name: '等待首次检索填充上下文...', tok: 0 });
  } else {
    var totalForBars = Math.max(est, 1);
    items.push({ name: '系统提示词', tok: Math.round(totalForBars * 0.12) });
    items.push({ name: '对话历史 (' + (data.history_msgs || 0) + '轮)', tok: Math.round(totalForBars * 0.42) });
    if (data.ltm_recalled) {
      items.push({ name: '长期记忆', tok: Math.round(totalForBars * 0.06) });
    }
    var sugCount = (data.suggestions || []).length;
    if (data.ltm_chars > 0 || sugCount > 0) {
      items.push({ name: '检索上下文', tok: Math.round(totalForBars * 0.30) });
    }
    items.push({ name: '其他开销', tok: Math.round(totalForBars * 0.10) });
  }

  var itemsHtml = '';
  var totalForBars = items.reduce(function(s, i) { return s + i.tok; }, 0) || 1;
  for (var i = 0; i < items.length; i++) {
    var it = items[i];
    itemsHtml += '<div class="ctx-item">';
    itemsHtml += '<span class="ctx-item-name">' + it.name + '</span>';
    itemsHtml += '<div class="ctx-item-bar"><div class="ctx-item-fill" style="width:' + Math.round(it.tok / totalForBars * 100) + '%"></div></div>';
    itemsHtml += '<span class="ctx-item-val">' + it.tok.toLocaleString() + '</span>';
    itemsHtml += '</div>';
  }
  var itemsEl = document.getElementById('ctxItems');
  if (itemsEl) {
    // P3-3: 明细为按初始估算的比例切分（占比可视化），标注说明避免误读为真实测量
    itemsHtml = '<div class="ctx-item-hint" style="font-size:10px;color:var(--text-muted);opacity:.75;margin-bottom:4px">明细为占比估算，顶部为实际用量</div>' + itemsHtml;
    itemsEl.innerHTML = itemsHtml;
  }

  var soft = data.soft_threshold || 0.75;
  var hard = data.hard_threshold || 0.93;
  var sumEl = document.getElementById('ctxSummary');
  if (sumEl) sumEl.innerHTML =
    '剩余 <b>' + (max - est).toLocaleString() + '</b> tokens' +
    '  ·  ' + '压缩 <b>' + Math.round(soft * 100) + '%</b> / ' + Math.round(hard * 100) + '%';

  var sugEl = document.getElementById('ctxSuggestions');
  if (sugEl) {
    // v8.10: 搜索建议（检索内部提示）不再显示——上下文面板只展示真实上下文构成
    if (data.format_hint) {
      sugEl.innerHTML = '<div class="ctx-sug-label">格式提示</div><div class="ctx-sug-item">' + data.format_hint + '</div>';
    } else {
      sugEl.innerHTML = '';
    }
  }

  var compEl = document.getElementById('ctxCompressed');
  if (compEl) {
    if (data.compressed) {
      compEl.style.display = '';
      compEl.innerHTML = '⚡ 上下文已压缩 (' + (data.compression_len || 0) + ' 字摘要)';
    } else {
      compEl.style.display = 'none';
    }
  }
}

  function handleStatusEvent(data) {
    // v8.3.5: status 事件显式化（step 进度 / budget_skip / circuit_breaker / planning）
    var stage = data.stage || '';
    var msg = data.message || '';
    var stepNames = {load:'加载会话上下文', retrieve:'检索文献', supervise:'分析中', answer:'生成回答'};
    if (stage === 'budget_skip' || stage === 'circuit_breaker') {
      appendTerminal('⚠ ' + msg);
      updateStatus('tool', msg);
    } else if (stage === 'agent_progress') {
      updateStatus('thinking', msg || '正在处理…'); appendTerminal('· ' + (msg || '正在处理…'));
    } else if (stage === 'planning') {
      updateStatus('answer', msg || '正在规划结构...');
    } else if (stage === 'final_answer') {
      updateStatus('answer', '生成最终答案...');
    } else if (stage === 'intake') {
      // v10.0: Intake 意图判定（fast 模型一次；有歧义时随后弹确认卡）
      updateStatus('tool', data.summary || msg || '正在理解你的问题...');
    } else if (stage === 'step_active') {
      updateStatus('tool', (stepNames[data.step_id] || data.step_id) + '...');
    } else if (stage === 'step_done') {
      updateStatus('tool', (stepNames[data.step_id] || data.step_id) + '完成');
    } else if (stage === 'agent_start') {
      updateStatus('agent', '启动子Agent: ' + (data.agent || ''));
    } else if (stage === 'retrieval_done') {
      // v8.4.8 F1: light 模式代码级预检索完成
      updateStatus('tool', '预检索完成 (' + (data.main_count||0) + ' 篇)');
      appendTerminal('✓ 预检索完成 · 通过阈值 ' + (data.main_count||0) + ' 篇');
    } else if (stage === 'local_done_web_running') {
      // v10.12（修复 P2-1 · 任务书问题 5）：**分段完成提示**。
      //   现象：本地检索 15:30:04 开始、联网 15:30:29 才回执，中间 25 秒界面
      //   没有任何"本地已完成"的信号，用户以为整体结束、实际联网还在跑。
      //   后端在 `search_both` 的两条并行分支各包一层：谁先完成就先发自己的事件
      //   （`asyncio.gather` 结构未改、无轮询），前端据此插一条清晰分隔提示。
      updateStatus('tool', '本地检索完成 · 联网检索中…');
      appendTerminal('✓ 本地检索完成（' + (data.local_result_count||0) +
                     ' 篇文献）· ⚙ 联网检索中…');
    } else if (stage === 'web_done') {
      updateStatus('tool', '联网检索完成 · 正在生成答案…');
      appendTerminal('✓ 联网检索完成（' + (data.web_result_count||0) +
                     ' 条），正在生成答案…');
    }
  }

  function updateContextTokens(extraTokens) {
  // v8.3.1: 入参即 token 数（调用方自行折算字符→token；context_usage 事件传真实 token）
  _totalTokensUsed += (extraTokens || 0);
  if (!_ctxData) return;
  var max = _ctxData.max_tokens || _ctxMaxTokens();
  var est = (_ctxData.estimated_tokens || 0) + Math.round(_totalTokensUsed);
  var pct = max > 0 ? Math.min(est / max, 1) : 0;

  _updateCtxBtnRing(pct);
  var pctEl = document.getElementById('ctx-pct');
  if (pctEl) pctEl.textContent = Math.round(pct * 100) + '%';

  var bar = document.getElementById('ctxBar');
  if (bar) { bar.style.width = Math.round(pct * 100) + '%'; bar.className = 'ctx-bar-inner ' + _ctxLevelClass(pct); }
  var barLabel = document.getElementById('ctxBarLabel');
  if (barLabel) barLabel.textContent = Math.round(est).toLocaleString() + ' / ' + max.toLocaleString();

  // P3-1: 同步"剩余 tokens / 压缩阈值"文本（此前只更新顶部 bar，剩余文本停在初始 est 造成 15% vs 99% 矛盾）
  var sumEl = document.getElementById('ctxSummary');
  if (sumEl) {
    var soft = _ctxData.soft_threshold || 0.75;
    var hard = _ctxData.hard_threshold || 0.93;
    sumEl.innerHTML = '剩余 <b>' + Math.max(max - est, 0).toLocaleString() + '</b> tokens' +
      '  ·  ' + '压缩 <b>' + Math.round(soft * 100) + '%</b> / ' + Math.round(hard * 100) + '%';
  }
}


/* ─── Chat ─── */
/* v8.10: 反馈栏默认不随气泡创建（回答生成完成后由 appendFeedback 追加，主流时序）；
   历史恢复的 bot 消息同样不带反馈 */
function addMessage(role, html) {
  var div = document.createElement('div');
  div.className = 'message ' + role;
  div.innerHTML = '<div class="avatar">' + (role === 'user' ? '→' : '◆') + '</div><div class="content">' + html + '</div>';
  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  return div.querySelector('.content');
}

/* v8.6 用户反馈（书 §4.6 反馈循环）: 助手回答下方追加 👍/👎 记录条（纯记录，不影响回答）
   v8.10: 回答完整生成后才调用（done 事件），不再随气泡创建即显示 */
function appendFeedback(msgEl) {
  if (!msgEl || msgEl.querySelector('.msg-feedback')) return;
  var fb = document.createElement('div');
  fb.className = 'msg-feedback';
  fb.innerHTML = '<button class="fb-btn" data-rating="1" title="回答有帮助">👍 有用</button>' +
                 '<button class="fb-btn" data-rating="-1" title="回答有问题">👎 有问题</button>' +
                 '<span class="fb-msg"></span>';
  msgEl.appendChild(fb);
}

/* v8.6/v8.7 反馈提交：点击 👍/👎 展开"哪里好/哪里不好"填写行，提交时带 comment；
   内容哈希作 message_id（幂等去重），仅记录不阻断 */
function submitFeedback(fb, mid, rating, comment) {
  var fbMsg = fb.querySelector('.fb-msg');
  fetch('/api/v2/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId || '', message_id: mid, rating: rating, comment: comment || '' })
  }).then(function (r) { return r.json(); }).then(function (d) {
    var row = fb.querySelector('.fb-row');
    if (row) row.remove();
    if (d && d.status === 'ok' && fbMsg) { fbMsg.textContent = rating === 1 ? '已记录 👍' : '已记录 👎'; }
  }).catch(function () {
    var row = fb.querySelector('.fb-row');
    if (row) row.remove();
    var allBtns = fb.querySelectorAll('.fb-btn');
    for (var i = 0; i < allBtns.length; i++) allBtns[i].disabled = false;
  });
}

document.addEventListener('click', function (e) {
  var btn = e.target.closest ? e.target.closest('.fb-btn') : null;
  if (!btn) return;
  var rating = parseInt(btn.getAttribute('data-rating'), 10);
  var fb = btn.closest('.msg-feedback');
  if (!fb || fb.querySelector('.fb-input')) return;   // 已展开填写行
  var msgEl = btn.closest('.message');
  var contentEl = msgEl ? msgEl.querySelector('.content') : null;
  var text = (contentEl ? contentEl.textContent || '' : '').substring(0, 2000);
  var hash = 0;
  for (var i = 0; i < text.length; i++) { hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0; }
  var mid = 'h' + (hash >>> 0).toString(16);

  var row = document.createElement('div');
  row.className = 'fb-row';
  row.innerHTML = '<input class="fb-input" placeholder="哪里好 / 哪里不好（可留空直接提交）" maxlength="500">' +
                  '<button class="fb-submit">提交</button>' +
                  '<button class="fb-cancel">取消</button>';
  fb.appendChild(row);
  var inp = row.querySelector('.fb-input');
  inp.focus();
  var allBtns = fb.querySelectorAll('.fb-btn');
  for (var i = 0; i < allBtns.length; i++) allBtns[i].disabled = true;
  row.querySelector('.fb-submit').addEventListener('click', function () {
    submitFeedback(fb, mid, rating, inp.value);
  });
  row.querySelector('.fb-cancel').addEventListener('click', function () {
    row.remove();
    for (var i = 0; i < allBtns.length; i++) allBtns[i].disabled = false;
  });
});

/* v8.10: silent=true 跳过确认（newSession 内部调用——新建会话不应弹"清空"提示） */
async function clearSession(silent) {
  if (!silent && !confirm('确定清空当前会话？')) return;
  try {
    await fetch('/api/v1/session/' + sessionId + '/clear', { method: 'POST' });
    chatContainer.innerHTML = '<div class="message bot"><div class="avatar">◆</div><div class="content">您好！有什么想了解的？</div></div>';
    sidebarContent.innerHTML = '<div class="empty-sidebar">提交问题后文献将在此展示</div>';
    citationCount.textContent = '0';
    roundHistory = [];
    _restoredGroups = null;   // v9.2: 清空会话同步清空恢复视图
    // P3-2: 重置上下文面板（清空后不再残留旧会话数字）
    _ctxData = null;
    _totalTokensUsed = 0;
    renderContextPanel({ estimated_tokens: 0, max_tokens: _ctxMaxTokens(), history_msgs: 0,
                         soft_threshold: 0.75, hard_threshold: 0.93 });
    // 清空后刷新会话列表。
    loadSessions();
  } catch (e) { alert('清空失败: ' + e.message); }
}

async function newSession() {
  try {
    var resp = await fetch('/api/v1/session/new', { method: 'POST' });
    var data = await resp.json();
    sessionId = data.session_id;
    localStorage.setItem(sessionStorageKey(), sessionId);
    // v8.10: 静默清空旧会话界面（不弹"确定清空"确认——新建会话语义，勿误导用户）
    await clearSession(true);
    // v8.10e: 创建后强制刷新会话列表（await + 重试），并延迟二次刷新防后端时序
    applySidebar('session');
    await loadSessionsWithRetry();
    setTimeout(function () { loadSessions(); }, 300);
  } catch (e) { alert('创建新会话失败: ' + e.message); }
}

function toggleRound(id) {
  var body = document.getElementById(id);
  var icon = document.getElementById(id + '-icon');
  body.style.display = body.style.display === 'none' ? 'block' : 'none';
  icon.classList.toggle('collapsed');
}
/* v8.15.2: 证据来源组手风琴（全局版本——内联 onclick 需要全局解析；
   此前定义在 streamOnce 内导致 RAG/UCR 组无法展开收缩） */
function toggleSrcGroup(id) {
  var g = document.getElementById('srcgroup-' + id);
  if (!g) return;
  var willOpen = g.style.display === 'none';
  var box = g.parentNode;
  // 收起同 Round 内的其他来源组（手风琴语义）
  var siblings = box.querySelectorAll('.src-group-body');
  for (var i = 0; i < siblings.length; i++) {
    siblings[i].style.display = 'none';
    var sibIcon = siblings[i].previousElementSibling &&
      siblings[i].previousElementSibling.querySelector('.toggle-icon');
    if (sibIcon) sibIcon.classList.add('collapsed');
  }
  if (willOpen) {
    g.style.display = 'block';
    var myIcon = g.previousElementSibling &&
      g.previousElementSibling.querySelector('.toggle-icon');
    if (myIcon) myIcon.classList.remove('collapsed');
  }
}
/* v8.15.2: 引用编号自然排序（[1]..[13] 升序；[W1] 组在数字之后） */
function _refIdKey(id) {
  var s = String(id == null ? '' : id);
  var m = s.match(/^([WH])?(\d+)$/i);
  if (!m) return 'ZZ' + s;
  return (m[1] ? m[1].toUpperCase() : 'N') + ('0000' + m[2]).slice(-4);
}
function _refIdSort(items) {
  return (items || []).slice().sort(function (a, b) {
    var ka = _refIdKey(a && a.ref_id), kb = _refIdKey(b && b.ref_id);
    return ka < kb ? -1 : (ka > kb ? 1 : 0);
  });
}
  // v10.13（Task E，用户指示）：联网组内按**来源分级**排序 T1 → T2 → T3 → TU。
  //   用户原话："每一个 url 分开，T1T2T3 优先排列"。
  //   同 tier 内回退到 ref_id 升序（保持可追溯、稳定）；无 tier 的排最后。
  // v10.13（改动 3.2）：联网组内按 `Wn` 数字升序（组内已同 tier）。
  function _webRefSort(items) {
    return (items || []).slice().sort(function (a, b) {
      var na = parseInt(String((a && a.ref_id) || '').replace(/\D/g, ''), 10);
      var nb = parseInt(String((b && b.ref_id) || '').replace(/\D/g, ''), 10);
      if (isNaN(na)) na = 1e9;
      if (isNaN(nb)) nb = 1e9;
      return na - nb;
    });
  }
  function _tierSort(items) {
    var ORD = { 'T1': 0, 'T2': 1, 'T3': 2, 'TU': 3 };
    return (items || []).slice().sort(function (a, b) {
      var ta = String((a && a.tier) || '').toUpperCase();
      var tb = String((b && b.tier) || '').toUpperCase();
      var oa = (ORD[ta] != null) ? ORD[ta] : 9;
      var ob = (ORD[tb] != null) ? ORD[tb] : 9;
      if (oa !== ob) return oa - ob;
      var ka = _refIdKey(a && a.ref_id), kb = _refIdKey(b && b.ref_id);
      return ka < kb ? -1 : (ka > kb ? 1 : 0);
    });
  }

/* v9.2: 引用编号统一重排已收敛到后端（evidence.renumber_refs）——citations 事件
   附带 remap（旧ref_id → 新ref_id；数字 [n] 连续 1..k、[Wn] 连续 W1..Wm、
   [Hn] 连续 H1..Hp）。前端只按 remap 通用重写正文与 ref_id，替代 v8.17.3
   前端 W 专用压缩（_buildCompactRefMap/_rewriteWRefsInText 移除）。 */
function _rewriteRefsByMap(text, remap) {
  if (!text || !remap) return text;
  return text.replace(/\[([WH]?)(\d+)\]/gi, function (full, pre, num) {
    var key = pre ? String(pre).toUpperCase() + num : num;
    return '[' + (remap[key] != null ? remap[key] : key) + ']';
  });
}

async function sendMsg(isFollowUp) {
  if (isFollowUp === undefined) isFollowUp = false;
  // v8.7: 运行中忽略重复发送（防连按 Enter/双击造成双气泡与交错回答）
  var sendBtnEl = document.getElementById('send-btn');
  if (sendBtnEl && sendBtnEl.disabled) return;
  var query = userInput.value.trim();
  if (!query) return;
  var filesForRequest = _selectedAttachments.slice();

  // v8.3.7 M1: 每次点击生成稳定幂等 ID，重试复用（防重发重复写历史）
  var clientRequestId = 'crid-' + Date.now().toString(36) + '-' +
    Math.random().toString(36).slice(2, 10);

  currentRoundId++;
  currentQuery = query;
  _ctxData = null;
  _totalTokensUsed = 0;

  // v8.4.11: 新发送清理停止提示，恢复输入框占位
  var stopMsg = document.getElementById('stop-msg');
  if (stopMsg) stopMsg.style.display = 'none';
  userInput.placeholder = '有什么想做的？';
  // v8.4.13: 新轮清空思维链文本（块随消息气泡创建，无需全局清理）
  _reasoningText = '';

  var userMessageContent = addMessage('user', escapeHtml(query) +
    (filesForRequest.length ? '<div class="sent-attachment-count">已附加 ' + filesForRequest.length + ' 个文件</div>' : ''));
  appendSentAttachmentPreviews(userMessageContent, filesForRequest);
  userInput.value = '';
  setSendDisabled(true);

  var botContent = addMessage('bot', '');
  var isCancelled = false;

  var _timeoutMs = 300000;
  var spinnerTimeout = setTimeout(function() {
    if (!isCancelled) { isCancelled = true; setSendDisabled(false); }
  }, _timeoutMs);

  /* ── status (opencode style: single line, expandable terminal) ── */
  var statusDiv = document.createElement('div');
  statusDiv.innerHTML = '<div class="exec-status" onclick="var t=document.getElementById(\'terminal_' + currentRoundId + '\'); if(t){t.style.display=t.style.display==\'none\'?\'block\':\'none\'}"><span class="exec-spinner"></span>◆ 思考中...<span class="tool-subtitle"></span></div>';
  var termId = 'terminal_' + currentRoundId;
  var termLog = document.createElement('div');
  termLog.id = termId;
  termLog.className = 'terminal-log';
  termLog.style.display = 'none';
  botContent.appendChild(statusDiv);
  botContent.appendChild(termLog);

  var answerDiv = document.createElement('div');
  answerDiv.style.display = 'none';

  var citationsDiv = document.createElement('div');
  citationsDiv.style.display = 'none';

  /* v8.7: 思维链折叠条——与回答共用一个气泡（气泡上部，回答在下部）；
     极简无图标，对齐 DeepSeek Harness 的 reasoning 展示形式 */
  var reasoningDiv = document.createElement('div');
  reasoningDiv.className = 'reasoning-block';
  reasoningDiv.style.display = 'none';
  reasoningDiv.innerHTML = '<div class="reasoning-header"><span class="reasoning-arrow">▶</span><span>深度思考</span></div><div class="reasoning-body"></div>';
  reasoningDiv.querySelector('.reasoning-header').addEventListener('click', toggleReasoning);

  botContent.appendChild(reasoningDiv);
  botContent.appendChild(answerDiv);
  botContent.appendChild(citationsDiv);

  var fullText = '';
  var isAnswerStarted = false;
  var renderPending = false;

  var _icons = {thinking: '💭', tool: '⚙', agent: '◆', answer: '…'};
  var _toolCount = 0;   // Actual tool-call count
  var _toolTraceLines = {};

  function updateStatus(type, text) {

    var icon = _icons[type] || '•';
    var safeText = escapeHtml(text);
    var subtitle = text.length > 30 ? ' <span class="tool-subtitle">' + safeText + '</span>' : '';
    // v8.10p: 执行中状态行前置旋转光环（回答气泡内提示执行中；完成摘要不带）
    statusDiv.innerHTML = '<div class="exec-status" onclick="var t=document.getElementById(\'' + termId + '\'); if(t){t.style.display=t.style.display==\'none\'?\'block\':\'none\'}"><span class="exec-spinner"></span>' + icon + ' ' + safeText.split('.')[0] + '...' + subtitle + '</div>';
  }

  function appendTerminal(msg) {
    var t = new Date();
    var ts = t.getHours().toString().padStart(2,'0') + ':' + t.getMinutes().toString().padStart(2,'0') + ':' + t.getSeconds().toString().padStart(2,'0');
    var entry = document.createElement('div'); entry.className = 'term-line';
    var timeEl = document.createElement('span'); timeEl.className = 'term-time'; timeEl.textContent = '[' + ts + ']';
    var msgEl = document.createElement('span'); msgEl.className = 'term-msg'; msgEl.textContent = ' ' + String(msg || '');
    entry.appendChild(timeEl); entry.appendChild(msgEl); termLog.appendChild(entry);
    if (termLog.children.length > 100) termLog.removeChild(termLog.firstChild);
    termLog.scrollTop = termLog.scrollHeight; return entry;
  }
  function clearStatus() {
    statusDiv.innerHTML = '';
    termLog.style.display = 'none';
  }

  /* v8.10h: 回答完成后工具执行记录不消失——保留为可展开摘要（与深度思考一致，
     用户可随时展开查看本轮执行过的工具与日志，提高对回答的信心） */
  function finishStatus() {
    if (_toolCount > 0) {
      statusDiv.innerHTML = '<div class="exec-status done" onclick="var t=document.getElementById(\'' + termId + '\'); if(t){t.style.display=t.style.display==\'none\'?\'block\':\'none\'}">⚙ 工具执行 ' + _toolCount + ' 项 · 点击展开详情<span class="tool-subtitle"></span></div>';
    } else {
      statusDiv.innerHTML = '';
    }
  }

  function renderAnswer() {
    if (!renderPending) {
      renderPending = true;
      requestAnimationFrame(function() {
        var s = fullText.replace(/__NEED_COMPLEX__/g, '');
        var frx = /---\s*\*\*.. \u6df1\u5165\u63a2\u7d22.*?\*\*([\s\S]*?)$/;
        var m = s.match(frx);
        var main = s;
        var qs = [];
        if (m) { main = s.replace(frx, '').trim();
          var qb = m[1]; var ql = qb.split('\n');
          for (var qi = 0; qi < ql.length; qi++) {
            var q = ql[qi].trim().replace(/^\d+\.\s*/, '');
            if (q) qs.push(q);
          }
        }
        answerDiv.innerHTML = renderMarkdownSafe(main);
        if (qs.length > 0) {
          var cc = document.createElement('div'); cc.className = 'follow-up-container';
          var cl = document.createElement('div'); cl.className = 'follow-up-label'; cl.textContent = '深入探索（点击追问）'; cc.appendChild(cl);
          var cd = document.createElement('div'); cd.className = 'follow-up-chips';
          for (var qi = 0; qi < qs.length; qi++) { (function(qt){var c=document.createElement('button');c.className='follow-up-chip';c.textContent=qt;c.onclick=function(){handleSendFollowUp(qt)};cd.appendChild(c)})(qs[qi]); }
          cc.appendChild(cd); answerDiv.appendChild(cc);
        }
        chatContainer.scrollTop = chatContainer.scrollHeight; renderPending = false;
      });
    }
  }

  function handleSendFollowUp(q) { userInput.value = q; sendMsg(true); }

  function showBanner(msg) {
    var b = document.getElementById('degraded-banner');
    if (!b) {
      b = document.createElement('div');
      b.id = 'degraded-banner';
      b.style.cssText = 'margin:8px auto;max-width:760px;padding:8px 12px;border-radius:6px;' +
        'background:rgba(255,180,0,.12);border:1px solid rgba(255,180,0,.4);color:#d9a441;font-size:13px';
      var sc = document.getElementById('chatContainer') || chatContainer;
      sc.parentNode.insertBefore(b, sc);
    }
    b.textContent = msg;
    b.style.display = '';
    clearTimeout(showBanner._t);
    showBanner._t = setTimeout(function () { b.style.display = 'none'; }, 8000);
  }

  /* v8.4.13/8.7 思维链「深度思考」折叠条：逐 chunk 累积到本消息气泡上部；
     出现即折叠（细标题条），点击展开；回答首 token 自动收起 */
  var _reasoningText = '';
  function handleReasoning(content) {
    if (!content) return;
    _reasoningText += content;
    if (reasoningDiv.style.display === 'none') {
      reasoningDiv.style.display = '';
      collapseReasoning();
    }
    var body = reasoningDiv.querySelector('.reasoning-body');
    if (body) { body.textContent = _reasoningText; chatContainer.scrollTop = chatContainer.scrollHeight; }
  }
  function toggleReasoning() {
    var body = reasoningDiv.querySelector('.reasoning-body');
    if (!body) return;
    var open = reasoningDiv.classList.contains('open');
    reasoningDiv.classList.toggle('open', !open);
    body.style.display = open ? 'none' : '';
  }
  function collapseReasoning() {
    var body = reasoningDiv.querySelector('.reasoning-body');
    if (body) body.style.display = 'none';
    reasoningDiv.classList.remove('open');
  }
  /* v8.4.13 写作「执行计划」折叠块：plan 事件（大纲步骤列表）
     v8.10k: 归入当前回答气泡内部（气泡顶部），不再作为独立气泡弹出 */
  function handlePlan(data) {
    var block = document.createElement('div');
    block.className = 'plan-block';
    var items = (data.sections || []).map(function(s) {
      return '<div class="plan-item">· ' + escapeHtml(s) + '</div>';
    }).join('');
    block.innerHTML =
      '<div class="plan-header" onclick="var b=this.parentNode.querySelector(\'.plan-body\');b.style.display=b.style.display===\'none\'?\'\':\'none\'">📋 执行计划：' + escapeHtml(data.title || '') +
      ' <span style="color:var(--text-muted);font-size:11px">(' + (data.chapters || 0) + ' 章)</span></div>' +
      '<div class="plan-body">' + items + '</div>';
    botContent.insertBefore(block, botContent.firstChild);
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  function handleEvent(type, data) {
    switch (type) {
      case 'reasoning': break; // Private model reasoning is never shown to users.
      case 'plan': handlePlan(data); break;
      case 'thinking': updateStatus('thinking', '正在处理…'); break;
      case 'tool_call_start':
        _toolCount++;
        updateStatus('tool', (data.summary || data.display_name || displayToolName(data.tool_name) || '工具') + ' 执行中…'); updateContextTokens(30);
        var _callLine = appendTerminal('⚙ ' + (data.summary || data.display_name || displayToolName(data.tool_name) || '工具') + ' · 执行中');
        _toolTraceLines = _toolTraceLines || {};
        if (data.tool_call_id) _toolTraceLines[data.tool_call_id] = _callLine;
        termLog.style.display = 'block'; break;
      case 'tool_executing':
        var _toolMsg = String(data.message || '执行中…');
        if (data.tool_name && _toolMsg.indexOf(data.tool_name) === 0)
          _toolMsg = displayToolName(data.tool_name) + _toolMsg.slice(data.tool_name.length);
        updateStatus('tool', _toolMsg); break;
      case 'tool_progress': updateStatus('tool', data.message||''); break;
      case 'tool_result':
        updateContextTokens((data.output || '').length * 0.7);
        var _summary = data.summary || (data.is_error ? '执行失败' : '完成');
        var _line = data.tool_call_id && _toolTraceLines && _toolTraceLines[data.tool_call_id];
        if (_line) {
          _line.querySelector('.term-msg').textContent = (data.is_error ? ' ⚠ ' : ' ✓ ') + displayToolName(data.tool_name) + ' · ' + _summary;
          var _detailText = String(data.output || '').substring(0, 1800);
          if (_detailText) {
            var _details = document.createElement('details'); _details.className = 'term-line'; _details.style.cssText = 'padding-left:52px;color:var(--text-muted)';
            var _summaryEl = document.createElement('summary'); _summaryEl.textContent = '查看工具结果';
            var _bodyEl = document.createElement('pre'); _bodyEl.textContent = _detailText + ((data.output || '').length > 1800 ? ' …（结果已截断）' : '');
            _bodyEl.style.cssText = 'white-space:pre-wrap;max-height:220px;overflow:auto;margin:4px 0;font:inherit';
            _details.appendChild(_summaryEl); _details.appendChild(_bodyEl); _line.after(_details);
          }
        } else appendTerminal((data.is_error ? '⚠ ' : '✓ ') + displayToolName(data.tool_name) + ' · ' + _summary);
        break;
      case 'agent_switch':
        updateStatus('agent', '启动: '+(data.agent_name||'')); updateContextTokens(50);
        appendTerminal('── 子Agent「' + (data.agent_name||'') + '」启动 ── ' + (data.task||''));
        break;
      case 'text':
        // v8.17.15: 草稿已删除——首字到达即视为正式回答开始，无草稿面板需处理
        if (!isAnswerStarted) { isAnswerStarted=true; clearTimeout(spinnerTimeout); updateStatus('answer','生成回答中...'); answerDiv.style.display=''; collapseReasoning(); }
        fullText+=data.content; updateContextTokens((data.content||'').length * 0.7); renderAnswer(); break;
      case 'draft':
        // v8.17.15: 草稿全链删除——后端不再推送 draft 事件，忽略（防御残留）
        break;
      case 'context_status': renderContextPanel(data); break;
      case 'status':
        handleStatusEvent(data);
        break;
      case 'context_usage':
        updateContextTokens((data.delta != null ? data.delta : data.total) || 0);
        if (data.call && ((data.call.cache_hit || 0) + (data.call.cache_miss || 0) > 0)) {
          appendTerminal('◈ 提示缓存命中 ' + (data.call.cache_hit || 0).toLocaleString() + ' tokens · 未命中 ' + (data.call.cache_miss || 0).toLocaleString() + ' tokens'); termLog.style.display = 'block';
        }
        break;
      case 'context_degraded': showBanner('⚠️ 历史会话加载失败，本次回答可能缺少上下文'); break;
      case 'intent_confirm':
        showIntentCard(data);
        appendTerminal('🤔 已弹出意图确认卡片（' + ((data.options || []).length) + ' 个候选）');
        break;
      case 'intent_resolved':
        var _ic = document.getElementById('intent-card');
        if (_ic) _ic.remove();
        appendTerminal('✓ 已按确认意图继续：' + String(data.intent || '').substring(0, 80)
                       + (data.timeout ? '（超时·按推荐项）' : ''));
        break;
      case 'permission_request':
        showPermissionCard(data);
        appendTerminal('⚠ 工具 ' + (data.tool_name || '') + ' 需要权限确认（已弹出审批卡片）');
        break;
      case 'plan_ready':
        updateStatus('answer', '📑 大纲已生成 (' + (data.sections||[]).length + ' 章)，开始逐章撰写...');
        appendTerminal('📑 大纲: ' + (data.title||'') + ' | 章节: ' + (data.sections||[]).join(' → '));
        break;
      case 'section_start':
        updateStatus('answer', '正在写第 ' + (data.index||'') + '/' + (data.total||'') + ' 章: ' + (data.heading||''));
        break;
      case 'section_done':
        updateStatus('answer', '✅ 第 ' + (data.index||'') + '/' + (data.total||'') + ' 章完成');
        break;
      case 'citations':
        // v9.2: 引用编号统一重排在后端完成（renumber_refs）——事件附带 remap，
        // 前端按 remap 重写正文 [n]/[Wn]/[Hn] 与 cited ref_id 后写入 roundHistory，
        // 侧栏与正文编号一致（替代 v8.17.3 前端 W 专用压缩）。
        var _cited0 = data.cited || [];
        var _remap = data.remap || {};
        var _dropped = (data.dropped || []).map(String);
        // v9.4.4（关键修复）: 后端 payload 的 ref_id 已是重排后的最终编号，
        // 前端**不再**按 remap 改写 ref_id——历史 bug：把 remap 二次套用到
        // 已重排编号上，与"已是新号的号"碰撞 → 不同条目撞号/跳号（[4][9]、
        // [4][7]、[5]/[W4] 三次实测的统一根因）。remap 只用于重写正文旧编号；
        // dropped（后端清除的无映射死号）同步从正文移除（如 [13]/[W3]）。
        var _applied = (window._remapAppliedRounds = window._remapAppliedRounds || {});
        // v10.5（F1b）: pre_renumbered=true 表示 done.answer 已是**终稿编号**——由后端
        // 返修轮重跑 renumber_refs 得到（expert_graph.py 的 F1 块）。此时 done 事件
        // 已用它覆盖 fullText，若此处再套一次 remap 就是二次重排：正文号相对侧栏
        // ref_id 发生置换错配（丢号/重号）。故仅在**未**预重排时按 remap 重写正文；
        // dropped 的清除保留（预重排时已无残留，属幂等兜底）。
        // _fullTextFromDone 是更根本的判据（done.answer 恒为重排终稿），用于兜住
        // done 事件期间流式全失败、fullText 只能取自 done.answer 的情况。
        var _preRenumbered = !!data.pre_renumbered
                             || !!(window._fullTextFromDone || {})[currentRoundId];
        if ((Object.keys(_remap).length > 0 || _dropped.length > 0)
            && !_applied[currentRoundId]) {
          if (Object.keys(_remap).length > 0 && !_preRenumbered) {
            fullText = _rewriteRefsByMap(fullText, _remap);
          }
          for (var _di = 0; _di < _dropped.length; _di++) {
            var _tok = String(_dropped[_di]).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            fullText = fullText.replace(new RegExp('\\[' + _tok + '\\]', 'g'), '');
          }
          renderAnswer();
          _applied[currentRoundId] = 1;
        }
        // ── v10.14（阶段 3 · 修复「侧栏不显示联网条目」）────────────────────────
        //   实测根因：`citations` 事件里 `web_separate` 有 26–27 条（每条带
        //   `round_no` + `tier` + `tier_label` + `url` + `title`），但本文件
        //   **从未引用 `web_separate`**（grep 命中数 = 0）⇒ 侧栏"联网搜索"组
        //   恒为空，用户看到的就是"侧栏不显示"。
        //   修法：把 `web_separate` 映射成侧栏条目（`source:'web'` ⇒ 落到 web 组），
        //   存进 roundHistory 的新字段 `web`，由 `renderRoundSidebar` 并入渲染。
        //   注：`ref_id` 用 `W1..Wn`（**仅侧栏展示用**，不进正文引用池 —— 引用池
        //   隔离仍保留：正文第一段只挂本地 `[n]`）。
        var _webSep = (data.web_separate || []).map(function (w, wi) {
          return {
            // v10.13-B（用户决策：恢复 `[Wn]`）：`ref_id` 用于**内部对齐**
            // （后端已按"正文实际引用的 `[Wn]`"过滤过 web_separate，
            //  这里只是把编号带过来）。**侧栏不显示编号本身** —— 仍显示「标题 🔗」。
            ref_id: w.ref_id || '',
            type: 'web',
            source: 'web',
            url: w.url || '',
            title: w.title || w.host || '',
            doi: '',
            tier: w.tier || '',
            tier_label: w.tier_label || '',
            round_no: (w.round_no != null ? w.round_no : 1),
            _isSummary: !!w.is_summary
          };
        });
        roundHistory.push({roundId:currentRoundId,query:currentQuery,cited:_cited0,uncited:data.uncited||[],historical:data.historical||[],web:_webSep});
        renderRoundSidebar(); citationsDiv.style.display='';
        var cited = _cited0;
        if (cited.length>0) {
          var c=document.createElement('div'); c.style.cssText='margin-top:10px;padding-top:10px;border-top:1px solid var(--border-hairline)';
          c.innerHTML='<div style="font-size:11px;color:var(--text-muted);margin-bottom:5px">引用 ('+cited.length+'篇)'+(data.ref_pool?' · '+data.ref_pool:'')+'</div>';
          // v8.15.2: 徽章条按 ref_id 升序（此前按回答出现顺序显得混乱）
          var order = _refIdSort(cited);
          for (var k=0;k<order.length;k++) c.innerHTML+='<span style="display:inline-block;font-family:var(--font-mono);font-size:10px;background:var(--success-glow);color:var(--success);padding:1px 5px;border-radius:3px;margin:2px;font-weight:600">['+escapeHtml(order[k].ref_id)+']</span>';
          citationsDiv.appendChild(c);
        }
        break;
      case 'done':
        clearTimeout(spinnerTimeout); isCancelled=true; answerDiv.style.display=''; finishStatus();
        // v8.3.7 M3: 引用无检索支撑 → 轻提示（不阻断回答）
        if (data.citation_unsupported) {
          showBanner('⚠ 回答包含文献引用，但本轮未检测到检索支撑（可能为模型记忆），请谨慎引用');
        } else if (data.citation_mismatch) {
          showBanner('⚠ 回答引用编号(' + (data.citation_count||0) + ')多于检索文献(' + (data.retrieval_count||0) + ')');
        }
        // v10.0（卡点③）: 修复循环的修正答案不流式上屏，须以 done.answer 覆盖草稿
        var _repaired = !!(data.audit && data.audit.repaired);
        // v10.5（F6，用户口径）: **判官看的是 done.answer，人看的也必须是 done.answer**
        //   ——即"重编号后的最终答案"。后端已在出口保证 done.answer 为终稿编号
        //   （expert_graph.py 的 F6 块），因此这里一律以它为准覆盖草稿。
        //   此前只在 audit.repaired 为真时覆盖，导致两类错位：
        //     ① 强制收尾题（预算/熔断/跑满轮次）用户看到的是"历轮草稿拼接"，
        //        判官评的却是终稿 —— final84 实测 38/84 题落此坑；
        //     ② 被改写但未置 repaired 标志的题同理。
        //   例外：done.answer 为空或为错误串时保留草稿（错误事件另有提示）。
        var _fromDone = (window._fullTextFromDone = window._fullTextFromDone || {});
        var _ansOk = !!data.answer && data.answer.indexOf('An error occurred') !== 0;
        if (_ansOk) {
          fullText = data.answer;
          _fromDone[currentRoundId] = 1;
          if (_repaired) {
            showBanner('✓ 质检发现问题并已自动修正，以上为修正后的最终答案');
          }
        } else if (!fullText && data.answer) {
          fullText = data.answer;
          _fromDone[currentRoundId] = 1;
        }
        // v10.0（卡点③）: 审计结果轻提示（critical / 需求未满足）
        try {
          var _au = data.audit || {};
          var _sa = (_au.stage_a || {}).severity || 'ok';
          var _sb = (_au.stage_b || {}).severity || '';
          if (_sa === 'critical' || _sb === 'critical') {
            showBanner('⚠ 质检发现回答存在事实/引用/口径问题，已尝试修正，请谨慎采用');
          } else if (_au.stage_b && _au.stage_b.requirement_met === false) {
            showBanner('⚠ 质检认为回答可能未完全满足你的问题要求（口径/范围），请注意甄别');
          }
        } catch (_ae) {}
        // v10.0（卡点②）: 证据裁决透明度（舍弃条数）
        if (data.adjudication && data.adjudication.length) {
          var _dropN = 0;
          for (var _ai = 0; _ai < data.adjudication.length; _ai++) {
            _dropN += ((data.adjudication[_ai] || {}).dropped_keys || []).length;
          }
          if (_dropN > 0) appendTerminal('🧹 证据裁决：已舍弃 ' + _dropN + ' 条与问题无关的证据');
        }
        if (fullText) {
          answerDiv.innerHTML=renderMarkdownSafe(fullText.replace(/__NEED_COMPLEX__/g,''));
        }
        if (fullText.indexOf('__NEED_COMPLEX__')>=0) {
          var uc=document.createElement('div');uc.style.cssText='margin-top:12px;text-align:center';
          var ub=document.createElement('button');ub.className='need-complex-btn';ub.textContent='继续深入检索';
          ub.onclick=function(){userInput.value=currentQuery;sendMsg()};
          uc.appendChild(ub);botContent.appendChild(uc);
        }
        var s=(data.gen_time_ms/1000).toFixed(1);
        var td=document.createElement('div');td.style.cssText='margin-top:8px;font-size:11px;color:var(--text-muted);font-family:var(--font-mono)';td.textContent='⏱ '+s+'s';botContent.appendChild(td);
        sessionId=data.session_id;if(sessionId)localStorage.setItem(sessionStorageKey(),sessionId);
        // v8.10: 回答完整生成后才显示 👍/👎 反馈栏（主流时序）
        appendFeedback(answerDiv.closest('.message'));
        setSendDisabled(false);userInput.focus();break;
      case 'error':
        clearTimeout(spinnerTimeout);isCancelled=true;setSendDisabled(false);userInput.focus();
        if(!fullText){clearStatus();answerDiv.style.display='';answerDiv.innerHTML='<div style="color:var(--error);font-weight:500">⚠ 请求出错: '+(data.message||'未知错误')+'</div>';}
        break;
      case 'heartbeat': break;
      default: break;
    }
  }

  function streamOnce() {
    /* v8.4.3: 权限授权后重放同一请求（同 clientRequestId → 服务端幂等不重复写历史） */
    fullText = '';
    isAnswerStarted = false;
    renderPending = false;
    _toolCount = 0;
    _toolTraceLines = {};
    answerDiv.style.display = '';
    answerDiv.innerHTML = '';
    statusDiv.innerHTML = '';
    termLog.style.display = 'none';

    if (currentAbortController) currentAbortController.abort();
    currentAbortController = new AbortController();

    (async function() {
      try {
        var attachments = await Promise.all(filesForRequest.map(async function(file) { return {name:file.name,type:file.type,data:await fileAsDataUrl(file)}; }));
        var response = await fetch(API_URL, {method:'POST',headers:{'Content-Type':'application/json'},signal:currentAbortController.signal,body:JSON.stringify({session_id:sessionId,query:query,client_request_id:clientRequestId,visualization_enabled:visualizationEnabled,attachments:attachments})});
        if (response.ok && filesForRequest.length) {
          _selectedAttachments = _selectedAttachments.filter(function(f) { return filesForRequest.indexOf(f) < 0; });
          renderAttachmentSelection();
        }
        if (!response.ok) { clearTimeout(spinnerTimeout);isCancelled=true;clearStatus();answerDiv.style.display='';answerDiv.innerHTML='<div style="color:var(--error);font-weight:500">⚠ 服务器错误 ('+response.status+')</div>';setSendDisabled(false);userInput.focus();return; }
        var reader=response.body.getReader(),decoder=new TextDecoder(),buffer='',curEvt='message';
        while(true){var chunk=await reader.read();if(chunk.done)break;buffer+=decoder.decode(chunk.value,{stream:true});var lines=buffer.split('\n');buffer=lines.pop();
          for(var j=0;j<lines.length;j++){var line=lines[j];
            if(line.startsWith('event: '))curEvt=line.substring(7).trim();
            else if(line.startsWith('data: ')){try{var d=JSON.parse(line.substring(6));try{handleEvent(curEvt,d)}catch(_){}}catch(_){}}
            else if(line.trim()==='')curEvt='message';
          }
        }
      } catch(err){if(err.name!=='AbortError'){clearTimeout(spinnerTimeout);isCancelled=true;clearStatus();answerDiv.style.display='';answerDiv.innerHTML='<span style="color:var(--error)">请求失败: '+err+'</span>';setSendDisabled(false);}}
      finally{if(currentAbortController)currentAbortController=null;}
    })();
  }

  /* v8.4.3: 权限重放钩子（permission_request 授权后由全局函数调用） */
  _lastStreamOnce = streamOnce;
  streamOnce();

  // v8.15: 证据来源展示元数据（来源分组手风琴 + 条目徽标；新增来源只加一行）
  // v9.2 (P0#4): 硬编码仅限 4 个内置来源的名称/顺序；未知来源（如 pipeline1
  // --source-type 新标注）动态生成同名分组 —— 见 srcKey()，分组本身代码驱动。
  var SOURCE_META = {
    rag:  { tag: 'RAG',  name: '本地文献库', cls: 'rag' },
    ucr:  { tag: 'UCR',  name: 'UCR品种库',  cls: 'ucr' },
    web:  { tag: 'Web',  name: '联网搜索',   cls: 'web' },
    historical: { tag: '历史', name: '历史证据', cls: 'historical' },
    // v10.13（改动 3.2）：联网按来源分级拆成三组（用户口径：T1 优先排列）
    web_t1: { tag: 'T1', name: '联网 · T1 同行评议 / 官方 / 学术数据库', cls: 'web' },
    web_t2: { tag: 'T2', name: '联网 · T2 权威与行业媒体 / 机构新闻稿', cls: 'web' },
    web_t3: { tag: 'T3', name: '联网 · T3 自媒体 / 论坛 / 来源不明',     cls: 'web' }
  };
  function srcMeta(item) {
    var s = (item && (item.source || item._src)) || '';
    var sm = SOURCE_META[s];
    if (sm) return sm;
    if (!s) return SOURCE_META['rag'];
    var key = _normSourceKey(s);
    return { tag: key, name: key, cls: 'custom' };
  }
  // v9.4 (P0#4 增强): 来源分组键规范化——非内置来源去尾部数字
  // （paper1→paper、Citrus varieties1→Citrus varieties），同名文件夹在前端
  // 经代码提取自动收纳为同类型分组；内置 4 组（rag/ucr/web/historical）原样。
  function _normSourceKey(s) {
    s = String(s || '').trim();
    if (s === 'rag' || s === 'ucr' || s === 'web' || s === 'historical') return s;
    var norm = s.replace(/\d+$/, '').trim();
    return norm || s;
  }
  // v9.2 (P0#4): 来源分组键——内置 4 组原样；未知来源（pipeline1 source_type）
  // 经 _normSourceKey 归组，使新来源（paper1..8 → paper、Citrus varieties1..3 →
  // Citrus varieties）自动生成侧栏手风琴组（不再并入 RAG）。
  // ── v10.13（改动 3.2/3.4）：侧栏按 **4 组固定顺序**归组 ──────────────────
  //   引用
  //   ├─ 本地文献                        （rag / ucr / historical 全归此组）
  //   ├─ 联网 · T1 同行评议 / 官方 / 学术数据库
  //   ├─ 联网 · T2 权威与行业媒体 / 机构新闻稿
  //   └─ 联网 · T3 自媒体 / 论坛 / 来源不明
  //   说明：历史组（`historical`）已按 3.4 归入「本地文献」——`Hn` 已退役，
  //   历史不再是独立引用通道（见 reports/38 §3）。
  function srcKey(item) {
    var s = (item && (item.source || item._src)) || '';
    var t = String((item && item.type) || '').toLowerCase();
    // 联网条目：按来源分级拆组
    if (s === 'web' || t === 'web') {
      var tier = String((item && item.tier) || '').toUpperCase();
      if (tier === 'T1') return 'web_t1';
      if (tier === 'T2') return 'web_t2';
      return 'web_t3';                       // T3 / TU / 未标注 一律归 T3 组
    }
    // 本地文献（rag / ucr / historical / 其它）统一归「本地文献」
    return 'rag';
  }
  // v8.15: 条目渲染——加来源徽标 + UCR 专属字段（无 DOI 时显示登记号/品种名）
  function renderCitationItem(item) {
    var sm = srcMeta(item);
    var isUcr = sm.cls === 'ucr';
    var du = item.doi && item.doi !== 'N/A' ? 'https://doi.org/' + encodeURIComponent(item.doi) : '#';
    var b = item.type === 'main' ? 'citation-badge' : 'citation-badge uncited';
    var sl = item.type === 'dynamic' ? 'Agent 发现' : (item.type === 'web' ? 'Web' : (item.type === 'historical' ? '历史证据' : (item.score ? 'CE=' + item.score.toFixed(3) : '')));
    var h = '<div class="citation-item" onclick="this.classList.toggle(\'expanded\')">';
    // ── v10.15（C.1 + C.2）：侧栏渲染为「**标题 + 🔗**」 ─────────────────────
    //   用户口径（报告 45 问题 2）：侧栏此前显示 `[W2] Web:` —— **标题行根本不存在**
    //   （`renderCitationItem` 整段没有一处读 `item.title`）。
    //   现改为：**不显示 `[Wn]` 编号、不显示 `Web:` 前缀**，只显示
    //   `标题` + 紧随其后的 `🔗`（点击跳转）。分组（本地/T1/T2/T3）保持不变。
    var _title = item.title || item.name || item.host || '未命名';
    var _turl = item.url || (item.doi && item.doi !== 'N/A'
      ? 'https://doi.org/' + encodeURIComponent(item.doi) : '');
    h += '<div class="cite-row">';
    h += '<span class="cite-title">' + escapeHtml(_title) + '</span>';
    if (_turl) {
      h += '<a href="' + escapeHtml(_turl) + '" target="_blank" class="cite-link"'
        + ' onclick="event.stopPropagation()" title="' + escapeHtml(_turl) + '">🔗</a>';
    }
    h += '</div>';
    // 折叠区（点击条目展开）：保留原有元数据，不再占据首行
    h += '<div class="citation-header citation-meta">';
    h += '<span class="src-badge ' + sm.cls + '">' + escapeHtml(sm.tag) + '</span>';
    h += '<span class="citation-score">' + escapeHtml(sl) + '</span></div>';
    // ── v10.13（改动 3.3）：联网条目显示为「标题 🔗」（🔗 点击跳转）────────
    //   用户口径：正文里的"标题 + 明文 URL 列表"全部移到侧栏。
    // ── v10.15（C.2）：标题 + 🔗 已上移到首行（`.cite-row`）⇒ 此处只保留
    //   **非联网**条目的标识信息（UCR 登记号 / DOI），避免重复渲染。
    if (isUcr && item.registry_id) {
      h += '<div class="citation-doi" style="color:var(--accent)">登记号: ' + escapeHtml(item.registry_id) + '</div>';
    } else if ((item.type === 'main' || item.type === 'dynamic')
               && item.doi && item.doi !== 'N/A') {
      h += '<a href="' + du + '" target="_blank" class="citation-doi" onclick="event.stopPropagation()">' + escapeHtml(item.doi) + '</a>';
    }
    // ── v10.16 实测修复：**侧栏 T1/T2/T3 与标题重复** ────────────────────────
    //   现象（用户真机截图）：侧栏把来源分级**重复列出** ——
    //     组标题已经写着「联网 · T1 同行评议 / 官方 / 学术数据库」，
    //     条目里**又**渲染一行「来源分级: T1（…）」；
    //     且标题已在首行 `.cite-title` 显示，折叠区**又**渲染「Title: …」。
    //   修法：**折叠态只留「标题 + 🔗」**（用户口径），其余元数据一律收进
    //   `.citation-detail`（点击才展开）。**不删数据**，只是不再常显。
    h += '<div class="citation-detail">';
    if (isUcr && item.variety_name) {
      h += '<div class="citation-text"><strong>品种:</strong> ' + escapeHtml(item.variety_name) + '</div>';
    }
    if (item.section_name) h += '<div class="citation-text"><strong>Section:</strong> ' + escapeHtml(item.section_name) + '</div>';
    // v10.5（§6.4 可溯源）: 联网条目展示来源分级与六层管线的判定痕迹
    //   tier/tier_reason 来自确定性分级（source_tiers）；filter_stage/content_chars/
    //   relevance_score/rerank_score 来自六层管线（只记录不裁剪时也有值）。
    // v10.16: 此处**只在折叠区**展示（组标题已表达 tier，常显会重复列出 T1/T2/T3）。
    if (item.source_tier) {
      h += '<div class="citation-text"><strong>来源分级:</strong> ' + escapeHtml(item.source_tier)
        + (item.tier_reason ? '（' + escapeHtml(item.tier_reason) + '）' : '') + '</div>';
    }
    if (item.filter_stage) {
      h += '<div class="citation-text"><strong>过滤:</strong> ' + escapeHtml(item.filter_stage)
        + (item.content_chars != null ? ' · 正文 ' + escapeHtml(String(item.content_chars)) + ' 字' : '')
        + (item.relevance_score != null ? ' · 相关性 ' + escapeHtml(String(item.relevance_score)) : '')
        + (item.rerank_score != null ? ' · 精排 ' + escapeHtml(String(item.rerank_score)) : '')
        + '</div>';
    }
    // v9.2: 恢复视图条目携带来源轮次元数据（live 条目无 round_seq 不显示）
    if (item.round_seq) h += '<div class="citation-text"><strong>Round:</strong> 第 ' + item.round_seq + ' 轮</div>';
    if (item.text_preview) h += '<div class="citation-text">' + escapeHtml((item.text_preview || '').substring(0, 250)) + '</div>';
    // v10.16: `Title:` 不再重复渲染 —— 标题已在首行 `.cite-title` 显示。
    //   UCR 条目的首行也是标题，同样无需重复。
    if (item.authors) h += '<div class="citation-text"><strong>Authors:</strong> ' + escapeHtml(item.authors) + '</div>';
    if (item.year) h += '<div class="citation-text"><strong>Year:</strong> ' + escapeHtml(item.year) + '</div>';
    if (item.chunk_id) h += '<div class="citation-text"><strong>Chunk:</strong> ' + escapeHtml(item.chunk_id) + '</div>';
    h += '</div></div>';
    return h;
  }
  // v8.15: 来源分组手风琴（同一 Round 内同时只展开一组）
  // v8.15.2: toggleSrcGroup 已提升为全局函数（见 toggleRound 下方）——
  // 此前的内部定义导致内联 onclick 全局解析失败，RAG/UCR 来源组无法展开收缩。
  /* v9.2: 刷新恢复视图——按来源 4 组手风琴（与 live 侧栏同构，共用
     src-group 结构与 renderCitationItem；条目携带来源轮次元数据） */
  function renderRestoredGroups() {
    if (!_restoredGroups) return false;
    // v9.4: 恢复视图同样支持动态来源组（与 live 侧栏同口径，未知组排在内置 4 组之后）
    var dyn = [];
    var gkeys = Object.keys(_restoredGroups || {});
    for (var gk = 0; gk < gkeys.length; gk++) {
      if (['rag', 'ucr', 'web', 'historical'].indexOf(gkeys[gk]) === -1) dyn.push(gkeys[gk]);
    }
    var order = ['rag', 'ucr', 'web', 'historical'].concat(dyn);
    var total = 0;
    for (var g = 0; g < order.length; g++) total += (_restoredGroups[order[g]] || []).length;
    citationCount.textContent = String(total) + (_restoredRefPool ? ' · ' + _restoredRefPool : '');
    var html = '';
    var firstOpen = true;
    for (var gi = 0; gi < order.length; gi++) {
      var gk = order[gi];
      var items = _restoredGroups[gk] || [];
      if (items.length === 0) continue;
      var sm2 = SOURCE_META[gk] || srcMeta({ source: gk });
      var gid = 'restore-' + gk;
      var gOpen = firstOpen;
      firstOpen = false;
      html += '<div class="src-group">';
      html += '<div class="src-group-header" onclick="toggleSrcGroup(\'' + gid + '\')">';
      html += '<span class="toggle-icon' + (gOpen ? '' : ' collapsed') + '">▶</span>';
      html += '<span class="src-badge ' + sm2.cls + '">' + escapeHtml(sm2.tag) + ':</span>';
      html += '<span class="round-title" style="font-size:11px">' + escapeHtml(sm2.name) + '</span>';
      html += '<span class="round-count">' + items.length + '</span></div>';
      html += '<div class="src-group-body" id="srcgroup-' + gid + '"' + (gOpen ? '' : ' style="display:none"') + '>';
      // v8.15.2: 组内按 ref_id 升序（重排后数字/W/H 均连续，_refIdSort 保序）
      // v10.13（Task E，用户指示）：**web 组内改按来源分级排序** T1 → T2 → T3 → TU，
      //   同 tier 内再按 ref_id。用户要求"每一个 url 分开，T1T2T3 优先排列"。
      //   其它组（本地文献库 / UCR / 历史）保持原 ref_id 升序，不受影响。
      // v10.13：web 组已按 tier 拆组，组内改用 `_webRefSort`（按 Wn 数字升序）；
      // 本地文献组用 `_refIdSort`（[n] 升序）。
      var gItems = (gk.indexOf('web_') === 0) ? _webRefSort(items) : _refIdSort(items);
      for (var m2 = 0; m2 < gItems.length; m2++) {
        html += renderCitationItem(gItems[m2]);
      }
      html += '</div></div>';
    }
    if (!html) return false;
    sidebarContent.innerHTML = html;
    return true;
  }

  function renderRoundSidebar() {
    if (roundHistory.length === 0) {
      // v9.2: 无实时轮次时优先显示刷新恢复的 4 组合并视图
      if (renderRestoredGroups()) return;
      sidebarContent.innerHTML = '<div class="empty-sidebar">提交问题后文献将在此展示</div>';
      citationCount.textContent = '0';
      return;
    }
    var tC = 0, tU = 0;
    for (var i = 0; i < roundHistory.length; i++) {
      tC += roundHistory[i].cited.length; tU += roundHistory[i].uncited.length;
    }
    citationCount.textContent = (tC + tU);
    var html = '';
    for (var i = roundHistory.length - 1; i >= 0; i--) {
      var round = roundHistory[i];
      // v8.15: 不再过滤 web 条目——按来源分组展示（本地文献库/UCR品种库/联网搜索/历史）
      var all = [];
      for (var k = 0; k < round.cited.length; k++) all.push(round.cited[k]);
      for (var k = 0; k < round.uncited.length; k++) all.push(round.uncited[k]);
      for (var k = 0; k < (round.historical || []).length; k++) all.push(round.historical[k]);
      // v10.14（阶段 3）：并入联网条目（`web_separate` ⇒ `round.web`）。
      //   这是「侧栏不显示联网条目」的修复点：此前该数组根本没进 `all`。
      for (var k = 0; k < (round.web || []).length; k++) all.push(round.web[k]);
      if (all.length === 0) continue;
      var qp = round.query.length > 25 ? round.query.substring(0, 25) + '...' : round.query;
      var open = i === roundHistory.length - 1;
      html += '<div class="round-group">';
      html += '<div class="round-header" onclick="toggleRound(\'round-' + round.roundId + '\')">';
      html += '<span class="toggle-icon" id="round-' + round.roundId + '-icon">▶</span>';
      html += '<span class="round-title">Round ' + round.roundId + ': ' + qp + '</span>';
      html += '<span class="round-count">' + all.length + '</span></div>';
      html += '<div class="round-body" id="round-' + round.roundId + '"' + (open ? '' : ' style="display:none"') + '>';
      // v8.15: 按来源分组（固定顺序：本地文献库→UCR品种库→联网搜索→历史证据）
      // v9.2 (P0#4): 未知来源（非 4 内置组）动态归组，排在固定 4 组之后
      var groups = {};
      for (var m = 0; m < all.length; m++) {
        var key = srcKey(all[m]);
        if (!groups[key]) groups[key] = [];
        groups[key].push(all[m]);
      }
      // v10.13（改动 3.2）：固定 4 组顺序 —— 本地文献 → 联网 T1 → T2 → T3
      // ── v10.19 修复：**分组标题重复列出**（用户截图：T1 出现两次、T3 出现两次）──
      //   缺陷：动态段只排除了 `rag/ucr/web/historical`，**没排除 `web_t1/t2/t3`**——
      //   而这三个键在前半段已列过 ⇒ `groupOrder` 里各出现**两次** ⇒ 同组渲染两遍。
      //   实测吻合：T1 两次、T3 两次，T2 因空被 `continue` 跳过（故只两次两组）。
      //   修法：把 `web_t1/web_t2/web_t3` 一并纳入排除集（固定段用固定顺序，动态段只收未知组）。
      var _FIXED_KEYS = ['rag', 'ucr', 'web', 'historical',
                         'web_t1', 'web_t2', 'web_t3'];
      var groupOrder = ['rag', 'web_t1', 'web_t2', 'web_t3']
        .concat(Object.keys(groups).filter(function (k) {
          return _FIXED_KEYS.indexOf(k) === -1;
        }));
      var firstOpen = true;
      for (var gi = 0; gi < groupOrder.length; gi++) {
        var gk = groupOrder[gi];
        if (!groups[gk] || groups[gk].length === 0) continue;
        var sm2 = SOURCE_META[gk] || srcMeta({ source: gk });
        var gid = round.roundId + '-' + gk;
        var gOpen = open && firstOpen;
        firstOpen = false;
        html += '<div class="src-group">';
        html += '<div class="src-group-header" onclick="toggleSrcGroup(\'' + gid + '\')">';
        html += '<span class="toggle-icon' + (gOpen ? '' : ' collapsed') + '">▶</span>';
        html += '<span class="src-badge ' + sm2.cls + '">' + escapeHtml(sm2.tag) + ':</span>';
        html += '<span class="round-title" style="font-size:11px">' + escapeHtml(sm2.name) + '</span>';
        html += '<span class="round-count">' + groups[gk].length + '</span></div>';
        html += '<div class="src-group-body" id="srcgroup-' + gid + '"' + (gOpen ? '' : ' style="display:none"') + '>';
        // v8.15.2: 组内按 ref_id 升序（回答乱序引用时侧栏保持编号顺序整齐）
        var gItems = _refIdSort(groups[gk]);
        for (var m2 = 0; m2 < gItems.length; m2++) {
          html += renderCitationItem(gItems[m2]);
        }
        html += '</div></div>';
      }
      html += '</div></div>';
    }
    sidebarContent.innerHTML = html || '<div class="empty-sidebar">本轮未检索到相关文献</div>';
  }
  }

/* v8.4.9 会话持久化：刷新/关闭重开后恢复历史对话渲染（数据在 sessions.db，只缺读取通道）
   v8.10l: 恢复增强——工具链折叠块（tool 消息）+ 深度思考折叠块（reasoning 持久化） */
async function restoreSessionHistory() {
  if (!sessionId) return;
  try {
    var resp = await fetch('/api/v2/session/' + encodeURIComponent(sessionId) + '/messages');
    if (!resp.ok) return;
    var data = await resp.json();
    var msgs = (data && data.messages) || [];
    // v8.4.10: 上下文快照随历史一起返回——刷新后上下文面板/占用环即时恢复，
    // 无需等下一次提问的 context_status 事件
    if (data && data.context) {
      renderContextPanel(data.context);
    }
    if (msgs.length === 0) return;
    clearSentAttachmentPreviewUrls();
    chatContainer.innerHTML = '';
    var pendingTools = [];
    var histSeq = 0;
    for (var i = 0; i < msgs.length; i++) {
      var m = msgs[i];
      if (m.role === 'user') {
        pendingTools = [];
        addMessage('user', escapeHtml(m.content));
      } else if (m.role === 'tool_calls') {
        for (var ci = 0; ci < (m.calls || []).length; ci++) pendingTools.push({role:'call', id:m.calls[ci].id || '', name:m.calls[ci].name || 'tool', summary:m.calls[ci].summary || m.calls[ci].name || 'tool'});
      } else if (m.role === 'tool') {
        var matchTool = pendingTools.find(function(x) { return x.role === 'call' && x.id === m.tool_call_id; });
        if (matchTool) { matchTool.content = m.content || ''; matchTool.truncated = !!m.truncated; } else pendingTools.push(m);
      } else if (m.role === 'assistant') {
        var contentEl = addMessage('bot', '');
        // Render the answer, then place the safe tool trace above it.
        // （此前 append 到 .message 外层会被 flex 排到对话区右侧——位置错乱）
        contentEl.innerHTML = renderMarkdownSafe(m.content);
        // Restore visible tool names, query summaries, and result snippets.
        if (pendingTools.length > 0) {
          var tid = 'terminal_hist_' + (++histSeq);
          var st = document.createElement('div');
          st.innerHTML = '<div class="exec-status done" onclick="var t=document.getElementById(\'' + tid + '\'); if(t){t.style.display=t.style.display==\'none\'?\'block\':\'none\'}">⚙ 工具执行 ' + pendingTools.length + ' 项 · 点击展开详情</div>';
          var tl = document.createElement('div');
          tl.id = tid;
          tl.className = 'terminal-log';
          tl.style.display = 'none';
          for (var t = 0; t < pendingTools.length; t++) {
            var pt = pendingTools[t];
            var line = document.createElement('div'); line.className = 'term-line';
            var histTime = document.createElement('span'); histTime.className = 'term-time'; histTime.textContent = '[tool]';
            var histMsg = document.createElement('span'); histMsg.className = 'term-msg';
            histMsg.textContent = (pt.summary || pt.name || 'tool') + (pt.content ? ' · 已完成' : '');
            line.appendChild(histTime); line.appendChild(histMsg); tl.appendChild(line);
            if (pt.content) {
              var det = document.createElement('div');
              det.className = 'term-line';
              det.style.cssText = 'padding-left: 52px; color: var(--text-muted);';
              det.textContent = (pt.content || '').substring(0, 180) + (pt.content.length > 180 ? ' …' : '');
              tl.appendChild(det);
            }
          }
          contentEl.insertBefore(st, contentEl.firstChild);
          contentEl.insertBefore(tl, contentEl.firstChild);
        }
      }
    }
    updateStatus('ok', '已恢复历史会话（' + msgs.length + ' 条消息）');
  } catch (e) { /* 无历史或读取失败：保持欢迎语，不打扰用户 */ }
}

/* v8.10: 无会话时自动创建默认会话（主流行为）——打开即有一个会话，历史回答有处可存，
   不再依赖用户手动点 ＋ 新建 */
async function ensureSession() {
  if (!sessionId) {
    try {
      var resp = await fetch('/api/v1/session/new', { method: 'POST' });
      var data = await resp.json();
      sessionId = data.session_id;
      localStorage.setItem(sessionStorageKey(), sessionId);
    } catch (e) { /* 后端不可达则保持无会话（页面仍可浏览） */ }
  }
  if (sessionId) { userInput.focus(); }
  restoreSessionHistory();
  loadSessions();
}



