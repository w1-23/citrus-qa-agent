(function () {
  'use strict';
  var pageSize = 25;
  var offset = 0;
  var toastTimer = null;
  function el(id) { return document.getElementById(id); }
  function esc(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, function (c) {
      return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]);
    });
  }
  function notify(message) {
    var box = el('toast'); if (!box) return;
    box.textContent = message; box.classList.add('visible');
    clearTimeout(toastTimer); toastTimer = setTimeout(function () { box.classList.remove('visible'); }, 3000);
  }
  async function jsonRequest(url, options) {
    var response = await fetch(url, options || {credentials:'same-origin'});
    var data = await response.json().catch(function () { return {}; });
    if (response.status === 401) { location.replace('/?next=%2Fadmin'); throw new Error('登录已失效'); }
    if (response.status === 403) throw new Error('仅管理员可访问');
    if (!response.ok) throw new Error(data.detail || '请求失败');
    return data;
  }
  function renderUsers(users) {
    var body = el('usersTable');
    if (!users.length) { body.innerHTML = '<tr><td colspan="4" class="muted">暂无账号</td></tr>'; return; }
    body.innerHTML = users.map(function (user) {
      var state = user.active ? '<span class="state active">启用</span>' : '<span class="state inactive">停用</span>';
      var actions = '<button class="small-button" data-action="reset" data-id="' + user.user_id + '">重置密码</button>';
      if (user.role !== 'admin') {
        actions += '<button class="small-button" data-action="limit" data-id="' + user.user_id + '" data-limit="' + user.daily_limit + '">设置限额</button>';
        actions += '<button class="small-button danger" data-action="active" data-id="' + user.user_id + '" data-active="' + (!user.active) + '">' + (user.active ? '停用' : '启用') + '</button>';
      }
      return '<tr><td><strong>' + esc(user.username) + '</strong><small>' + esc(user.role) + '</small></td><td>' + state + '</td>' +
        '<td>' + esc(user.requests_today) + ' / ' + esc(user.daily_limit) + '</td><td class="actions">' + actions + '</td></tr>';
    }).join('');
  }
  function renderAudit(items) {
    var host = el('auditList');
    if (!items.length) { host.innerHTML = '<p class="muted">没有匹配的请求记录</p>'; return; }
    host.innerHTML = items.map(function (item) {
      var tools = (item.tools || []).map(function (tool) { return tool.name || tool.tool || ''; }).filter(Boolean).join('、') || '无';
      var attachments = (item.attachments || []).map(function (file) {
        var kb = Math.max(1, Math.round((file.approx_bytes || 0) / 1024));
        return esc(file.name) + '（' + kb + ' KB）';
      }).join('、') || '无';
      var when = item.created_at ? new Date(item.created_at).toLocaleString() : '时间未知';
      return '<details class="audit-item"><summary><span class="audit-user">' + esc(item.username) + '</span><span class="audit-status">' + esc(item.status) + '</span><time>' + esc(when) + '</time><span class="audit-question">' + esc(item.query) + '</span></summary>' +
        '<div class="audit-content"><p><b>问题</b><span>' + esc(item.query) + '</span></p>' +
        '<p><b>回答</b><span>' + esc(item.answer || '尚未完成') + '</span></p>' +
        '<p><b>工具</b><span>' + esc(tools) + '</span></p>' +
        '<p><b>附件</b><span>' + attachments + '</span></p>' +
        '<p><b>会话与耗时</b><span>' + esc(item.session_id) + ' · ' + esc(item.elapsed_ms == null ? '—' : item.elapsed_ms + ' ms') + '</span></p></div></details>';
    }).join('');
  }
  async function loadOverview() {
    var results = await Promise.all([
      jsonRequest('/api/v2/admin/users'),
      jsonRequest('/api/v2/admin/queue'),
      jsonRequest('/api/v2/config'),
      jsonRequest('/api/v2/config/providers')
    ]);
    var users = results[0].users || [], queue = results[1], config = results[2], providers = results[3].providers || [];
    el('userCount').textContent = users.length;
    el('requestCount').textContent = users.reduce(function (sum, user) { return sum + (Number(user.requests_today) || 0); }, 0);
    el('activeCount').textContent = queue.active + ' / ' + queue.max_active;
    el('queuedCount').textContent = queue.queued + ' / ' + queue.max_pending;
    renderUsers(users);
    var activeProvider = providers.find(function (p) { return p.id === config.model.provider_id; });
    el('modelStatus').textContent = activeProvider
      ? '主模型：' + activeProvider.name + ' · ' + config.model.main + ' · ' + (config.model.image_input ? '支持图像输入' : '不支持图像输入')
      : '主模型：' + config.model.main;
    var configured = providers.filter(function (p) { return p.configured; });
    var webName = config.web_search && config.web_search.provider === 'zhipu_web_search' ? '智谱 Web Search' : 'DeepSeek Search/Fetch';
    el('serviceSummary').innerHTML = '<span class="service-chip ' + (configured.length ? 'ok' : 'warn') + '">主模型服务 ' + (configured.length ? '已配置' : '未配置') + '</span>' +
      '<span class="service-chip ' + (config.web_search && config.web_search.configured ? 'ok' : 'warn') + '">' + esc(webName) + (config.web_search && config.web_search.configured ? ' 已配置' : ' 未配置') + '</span>' +
      '<span class="service-chip neutral">处理 ' + queue.active + ' · 排队 ' + queue.queued + '</span>';
  }
  async function loadAudit() {
    var query = (el('auditQuery').value || '').trim();
    var data = await jsonRequest('/api/v2/admin/requests?limit=' + pageSize + '&offset=' + offset + '&q=' + encodeURIComponent(query));
    renderAudit(data.items || []);
    var total = Number(data.total) || 0;
    el('auditPageLabel').textContent = total ? (offset + 1) + '–' + Math.min(offset + pageSize, total) + ' / ' + total : '0 条';
    el('olderButton').disabled = offset + pageSize >= total;
    el('newerButton').disabled = offset <= 0;
  }
  async function refresh() {
    el('refreshButton').disabled = true;
    try { await Promise.all([loadOverview(), loadAudit()]); }
    catch (error) { notify(error.message); }
    finally { el('refreshButton').disabled = false; }
  }
  async function handleUserAction(event) {
    var button = event.target.closest('button[data-action]'); if (!button) return;
    var id = button.dataset.id, action = button.dataset.action;
    try {
      if (action === 'reset') {
        if (!confirm('重置此账号密码？临时密码只显示一次。')) return;
        var reset = await jsonRequest('/api/v2/admin/users/' + id + '/reset-password', {method:'POST'});
        window.prompt('请通过安全渠道交付此临时密码：', reset.temporary_password);
      } else if (action === 'active') {
        await jsonRequest('/api/v2/admin/users/' + id + '/active', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({active:button.dataset.active === 'true'})});
      } else if (action === 'limit') {
        var value = Number(window.prompt('每日请求上限（1–10000）', button.dataset.limit));
        if (!Number.isInteger(value) || value < 1 || value > 10000) return;
        await jsonRequest('/api/v2/admin/users/' + id + '/daily-limit', {method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({daily_limit:value})});
      }
      await loadOverview(); notify('已更新');
    } catch (error) { notify(error.message); }
  }
  async function init() {
    var me = await jsonRequest('/api/v2/auth/me');
    if (!me.user || me.user.role !== 'admin') { document.body.innerHTML = '<main class="denied"><h1>仅管理员可访问</h1><a href="/">返回 CitrusQA</a></main>'; return; }
    el('adminName').textContent = me.user.username;
    el('refreshButton').addEventListener('click', refresh);
    el('usersTable').addEventListener('click', handleUserAction);
    el('auditSearch').addEventListener('submit', function (event) { event.preventDefault(); offset = 0; loadAudit().catch(function (e) { notify(e.message); }); });
    el('olderButton').addEventListener('click', function () { offset += pageSize; loadAudit().catch(function (e) { notify(e.message); }); });
    el('newerButton').addEventListener('click', function () { offset = Math.max(0, offset - pageSize); loadAudit().catch(function (e) { notify(e.message); }); });
    el('logoutButton').addEventListener('click', async function () { await fetch('/api/v2/auth/logout', {method:'POST'}); location.replace('/'); });
    await refresh();
  }
  document.addEventListener('DOMContentLoaded', function () { init().catch(function (e) { notify(e.message); }); });
}());
