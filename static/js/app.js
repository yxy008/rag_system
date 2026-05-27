/**
 * app.js - Spring AI RAG 系统前端交互
 * 负责：消息发送、SSE 流式接收、对话历史、混合检索开关、来源展示、仪表盘、用户认证
 */

// ============================================================
// 状态管理
// ============================================================

let isStreaming = false;
let currentAbortController = null;
let currentRecognition = null;
let currentRecognitionAborted = false;
let sessionId = localStorage.getItem("rag_session_id") || generateSessionId();
let currentSources = [];
let conversationHistory = [];
let currentTab = 'chat';
let currentUser = null;
let currentAnswerStyle = 'detailed';
let currentConfidenceData = null;
let lastQuestion = '';
let lastAnswer = '';

function generateSessionId() {
  const id = "sess_" + Date.now() + "_" + Math.random().toString(36).slice(2, 8);
  localStorage.setItem("rag_session_id", id);
  return id;
}

// ============================================================
// 用户认证模块
// ============================================================

const AUTH_TOKEN_KEY = 'rag_auth_token';
const AUTH_USER_KEY = 'rag_auth_user';

function getAuthToken() {
  return localStorage.getItem(AUTH_TOKEN_KEY);
}

function setAuthToken(token) {
  localStorage.setItem(AUTH_TOKEN_KEY, token);
}

function removeAuth() {
  localStorage.removeItem(AUTH_TOKEN_KEY);
  localStorage.removeItem(AUTH_USER_KEY);
  currentUser = null;
}

function saveUserInfo(user) {
  currentUser = user;
  localStorage.setItem(AUTH_USER_KEY, JSON.stringify(user));
}

function loadUserInfo() {
  const saved = localStorage.getItem(AUTH_USER_KEY);
  if (saved) {
    try { currentUser = JSON.parse(saved); } catch(e) { currentUser = null; }
  }
}

function isAuthenticated() {
  return !!getAuthToken() && !!currentUser;
}

async function apiFetch(url, options) {
  const token = getAuthToken();
  const headers = options && options.headers ? { ...options.headers } : {};
  if (token) {
    headers['X-Auth-Token'] = token;
  }
  if (!(options && options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }

  const res = await fetch(url, { ...options, headers });
  if (res.status === 401) {
    removeAuth();
    enterAuthPage();
    throw new Error('登录已过期，请重新登录');
  }
  return res;
}

function showLoginForm() {
  document.getElementById('login-form').style.display = '';
  document.getElementById('register-form').style.display = 'none';
  var loginErr = document.getElementById('login-error');
  loginErr.textContent = '';
  loginErr.style.color = '';
  document.getElementById('register-error').textContent = '';
}

function showRegisterForm() {
  document.getElementById('login-form').style.display = 'none';
  document.getElementById('register-form').style.display = '';
  document.getElementById('login-error').textContent = '';
  document.getElementById('register-error').textContent = '';
}

function initAuthState() {
  loadUserInfo();
  if (isAuthenticated()) {
    enterApp();
  } else {
    enterAuthPage();
  }
}

function enterAuthPage() {
  var appContainer = document.getElementById('app-container');
  var authWrapper = document.getElementById('auth-page-wrapper');
  if (appContainer) appContainer.classList.add('app-hidden');
  if (authWrapper) authWrapper.style.display = '';
  updateSidebarUser();
  currentTab = 'auth';
  showLoginForm();
}

function enterApp() {
  var appContainer = document.getElementById('app-container');
  var authWrapper = document.getElementById('auth-page-wrapper');
  if (authWrapper) authWrapper.style.display = 'none';
  if (appContainer) appContainer.classList.remove('app-hidden');
  updateSidebarUser();
  switchTab('chat');
  loadHistoryList();
}

function updateSidebarUser() {
  var userNameEl = document.getElementById('sidebar-user-name');
  var userAvatarEl = document.getElementById('sidebar-user-avatar');
  if (!userNameEl) return;

  if (isAuthenticated() && currentUser) {
    userNameEl.textContent = currentUser.nickname || currentUser.username || '用户';
    if (userAvatarEl) {
      var initial = (currentUser.nickname || currentUser.username || 'U').charAt(0).toUpperCase();
      userAvatarEl.innerHTML = '<span style="font-weight:600;font-size:14px;">' + initial + '</span>';
    }
  } else {
    userNameEl.textContent = '未登录';
    if (userAvatarEl) {
      userAvatarEl.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>';
    }
  }
}

async function handleLogin() {
  var username = document.getElementById('login-username').value.trim();
  var password = document.getElementById('login-password').value;
  var errorEl = document.getElementById('login-error');

  if (!username || !password) {
    errorEl.textContent = '请输入用户名和密码';
    return;
  }

  errorEl.textContent = '';

  try {
    var res = await apiFetch('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username: username, password: password })
    });
    var data = await res.json();

    if (data.success) {
      setAuthToken(data.token);
      var userInfo = {
        id: data.user_id,
        username: data.username,
        nickname: data.nickname,
        createdAt: data.created_at || '',
        lastLoginAt: data.last_login_at || ''
      };
      saveUserInfo(userInfo);
      enterApp();
      showToast('登录成功，欢迎回来！', 'success');
    } else {
      errorEl.textContent = data.error || '登录失败，请检查用户名和密码';
    }
  } catch(e) {
    errorEl.textContent = e.message === '登录已过期，请重新登录' ? e.message : '网络错误，请稍后重试';
  }
}

async function handleRegister() {
  var username = document.getElementById('reg-username').value.trim();
  var nickname = document.getElementById('reg-nickname').value.trim();
  var password = document.getElementById('reg-password').value;
  var confirm = document.getElementById('reg-confirm').value;
  var errorEl = document.getElementById('register-error');

  if (!username || username.length < 3 || username.length > 20) {
    errorEl.textContent = '用户名需要 3-20 个字符';
    return;
  }
  if (!password || password.length < 6) {
    errorEl.textContent = '密码至少需要 6 个字符';
    return;
  }
  if (password !== confirm) {
    errorEl.textContent = '两次输入的密码不一致';
    return;
  }

  errorEl.textContent = '';

  try {
    var res = await apiFetch('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({
        username: username,
        password: password,
        nickname: nickname || null
      })
    });
    var data = await res.json();

    if (data.success) {
      errorEl.textContent = '';
      document.getElementById('register-form').style.display = 'none';
      var loginForm = document.getElementById('login-form');
      loginForm.style.display = '';
      document.getElementById('login-username').value = username;
      document.getElementById('login-password').value = '';
      document.getElementById('login-error').textContent = '注册成功！请使用账号密码登录';
      document.getElementById('login-error').style.color = '#22c55e';
    } else {
      errorEl.textContent = data.error || '注册失败，请稍后重试';
    }
  } catch(e) {
    errorEl.textContent = '网络错误，请稍后重试';
  }
}

async function handleLogout() {
  try {
    await apiFetch('/api/auth/logout', { method: 'POST' });
  } catch(e) {}
  removeAuth();
  enterAuthPage();
  showToast('已退出登录', 'info');
}

// ============================================================
// 个人中心
// ============================================================

function loadProfile() {
  if (!currentUser) {
    showToast('请先登录后再查看个人中心', 'warning');
    return;
  }
  document.getElementById('profile-username').textContent = currentUser.username || '—';
  document.getElementById('profile-nickname').textContent = currentUser.nickname || '—';
  document.getElementById('profile-created-at').textContent = currentUser.createdAt || '—';
  document.getElementById('profile-last-login').textContent = currentUser.lastLoginAt || '—';
  document.getElementById('profile-new-nickname').value = '';
  document.getElementById('profile-old-pwd').value = '';
  document.getElementById('profile-new-pwd').value = '';
  document.getElementById('profile-confirm-pwd').value = '';
  document.getElementById('profile-nickname-error').textContent = '';
  document.getElementById('profile-pwd-error').textContent = '';
}

async function updateNickname() {
  var nickname = document.getElementById('profile-new-nickname').value.trim();
  var errorEl = document.getElementById('profile-nickname-error');
  if (!nickname) {
    errorEl.textContent = '请输入昵称';
    return;
  }
  errorEl.textContent = '';
  try {
    var res = await apiFetch('/api/auth/update-profile', {
      method: 'PUT',
      body: JSON.stringify({ nickname: nickname })
    });
    var data = await res.json();
    if (data.success) {
      currentUser.nickname = nickname;
      saveUserInfo(currentUser);
      loadProfile();
      showToast('昵称修改成功', 'success');
    } else {
      errorEl.textContent = data.error || '修改失败';
    }
  } catch(e) {
    errorEl.textContent = e.message || '网络错误';
  }
}

async function changePassword() {
  var oldPwd = document.getElementById('profile-old-pwd').value;
  var newPwd = document.getElementById('profile-new-pwd').value;
  var confirmPwd = document.getElementById('profile-confirm-pwd').value;
  var errorEl = document.getElementById('profile-pwd-error');
  if (!oldPwd || !newPwd || !confirmPwd) {
    errorEl.textContent = '请填写所有密码字段';
    return;
  }
  if (newPwd.length < 6) {
    errorEl.textContent = '新密码至少需要6个字符';
    return;
  }
  if (newPwd !== confirmPwd) {
    errorEl.textContent = '两次输入的新密码不一致';
    return;
  }
  errorEl.textContent = '';
  try {
    var res = await apiFetch('/api/auth/change-password', {
      method: 'PUT',
      body: JSON.stringify({ oldPassword: oldPwd, newPassword: newPwd })
    });
    var data = await res.json();
    if (data.success) {
      document.getElementById('profile-old-pwd').value = '';
      document.getElementById('profile-new-pwd').value = '';
      document.getElementById('profile-confirm-pwd').value = '';
      showToast('密码修改成功', 'success');
    } else {
      errorEl.textContent = data.error || '修改失败';
    }
  } catch(e) {
    errorEl.textContent = e.message || '网络错误';
  }
}

// ============================================================
// Tab 切换
// ============================================================

function switchTab(tab) {
  currentTab = tab;

  document.querySelectorAll('.tab-panel').forEach(function(panel) {
    panel.classList.remove('active');
    panel.style.display = '';
  });
  var panel = document.getElementById('panel-' + tab);
  if (panel) {
    panel.classList.add('active');
    panel.style.display = '';
  }

  if (tab === 'dashboard') {
    setTimeout(function() { loadDashboard(); }, 100);
  }
  if (tab === 'knowledge') {
    loadStatus();
  }
  if (tab === 'settings') {
    loadSettingsStatus();
  }
  if (tab === 'profile') {
    loadProfile();
  }
  if (tab === 'explore') {
    // 知识探索面板无需预加载
  }
  if (tab === 'collaboration') {
    loadFeedbackStats();
    loadFeedbackList();
    loadExpertRouting();
  }
  if (tab === 'bookmarks') {
    loadBookmarks();
  }
  if (tab === 'openapi') {
    loadApiKeys();
  }

  var menu = document.getElementById('sidebar-user-menu');
  if (menu && menu.style.display === 'block') {
    menu.style.display = 'none';
    var chevron = document.getElementById('sidebar-user-chevron');
    if (chevron) chevron.classList.remove('open');
  }
}

// 页面初始化
document.addEventListener('DOMContentLoaded', function() {
  initAuthState();

  // 仪表盘子Tab切换
  document.querySelectorAll('.dash-subtab').forEach(function(btn) {
    btn.addEventListener('click', function() {
      var dashtab = this.getAttribute('data-dashtab');
      if (dashtab) switchDashTab(dashtab);
    });
  });

  // 登录表单回车提交
  var loginPwd = document.getElementById('login-password');
  if (loginPwd) {
    loginPwd.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') handleLogin();
    });
  }
  var regConfirm = document.getElementById('reg-confirm');
  if (regConfirm) {
    regConfirm.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') handleRegister();
    });
  }

  // 管理弹窗点击遮罩关闭
  var modalOverlay = document.getElementById('manage-modal-overlay');
  if (modalOverlay) {
    modalOverlay.addEventListener('click', function(e) {
      if (e.target === modalOverlay) closeManageDialog();
    });
  }

  // 点击其他地方关闭用户菜单
  document.addEventListener('click', function(e) {
    var menu = document.getElementById('sidebar-user-menu');
    var btn = document.getElementById('sidebar-user-area');
    if (menu && btn && !btn.contains(e.target) && menu.style.display !== 'none') {
      menu.style.display = 'none';
      var chevron = document.getElementById('sidebar-user-chevron');
      if (chevron) chevron.classList.remove('open');
    }
  });
});

// 仪表盘子Tab切换
function switchDashTab(dashtab) {
  document.querySelectorAll('.dash-subtab').forEach(function(btn) {
    btn.classList.toggle('active', btn.getAttribute('data-dashtab') === dashtab);
  });
  document.querySelectorAll('.dash-subpanel').forEach(function(panel) {
    panel.classList.remove('active');
  });
  var subpanel = document.getElementById('dash-subpanel-' + dashtab);
  if (subpanel) {
    subpanel.classList.add('active');
  }
  // 切换到图表子Tab时重新渲染图表
  if (dashtab === 'charts') {
    setTimeout(function() { loadDashboard(); }, 50);
  }
}

// ============================================================
// 系统状态
// ============================================================

async function loadStatus() {
  try {
    const res = await apiFetch('/api/status');
    const data = await res.json();

    var docCountEl = document.getElementById('kb-doc-count');
    if (docCountEl) {
      docCountEl.textContent = data.knowledge_base_ready
        ? data.doc_count + ' 个向量'
        : '知识库为空';
      docCountEl.style.color = data.knowledge_base_ready ? '#22c55e' : '#ef4444';
    }

    var embedEl = document.getElementById('kb-embed-model');
    if (embedEl) embedEl.textContent = data.embedding_model || '—';

    var llmEl = document.getElementById('kb-llm-model');
    if (llmEl) llmEl.textContent = data.llm_model || '—';

    var sessionEl = document.getElementById('kb-session-count');
    if (sessionEl) sessionEl.textContent = data.active_sessions || 0;

    const hybridToggle = document.getElementById('hybrid-toggle');
    if (hybridToggle) {
      hybridToggle.checked = data.hybrid_search !== false;
    }

    try {
      const configRes = await apiFetch('/api/config');
      const config = await configRes.json();
      const rerankerToggle = document.getElementById('reranker-toggle');
      if (rerankerToggle) {
        rerankerToggle.checked = config.reranker !== false;
      }
    } catch (_) {}
  } catch (e) {
    var docCountEl = document.getElementById('kb-doc-count');
    if (docCountEl) docCountEl.textContent = '连接失败';
  }
}

// ============================================================
// 知识库健康检查
// ============================================================

async function loadHealthCheck(quick) {
  var resultEl = document.getElementById('health-result');
  var loadingEl = document.getElementById('health-loading');

  if (loadingEl) loadingEl.style.display = '';
  if (resultEl) resultEl.style.display = 'none';

  try {
    var url = '/api/health';
    if (quick) url += '?quick=true';
    var res = await apiFetch(url);
    var data = await res.json();

    renderHealthReport(data);
  } catch (e) {
    if (resultEl) {
      resultEl.style.display = '';
      resultEl.innerHTML = '<div class="health-error">健康检查失败：' + escapeHtml(e.message) + '</div>';
    }
  } finally {
    if (loadingEl) loadingEl.style.display = 'none';
  }
}

function renderHealthReport(data) {
  var resultEl = document.getElementById('health-result');
  if (!resultEl) return;
  resultEl.style.display = '';

  // 综合评分
  var overallEl = document.getElementById('health-overall');
  if (overallEl) {
    var score = data.overall_score || 0;
    var status = data.overall_status || 'unknown';
    var statusText = status === 'healthy' ? '健康' : status === 'warning' ? '警告' : status === 'error' ? '异常' : status === 'empty' ? '空库' : status;
    var statusColor = score >= 80 ? '#22c55e' : score >= 60 ? '#f59e0b' : '#ef4444';
    var duration = data.check_duration_seconds != null ? ' · 耗时 ' + data.check_duration_seconds + 's' : '';

    overallEl.innerHTML = '<div class="health-overall-card">'
      + '<div class="health-overall-score" style="color:' + statusColor + '">' + score + '</div>'
      + '<div class="health-overall-info">'
      + '<div class="health-overall-status" style="color:' + statusColor + '">' + statusText + '</div>'
      + '<div class="health-overall-time">' + escapeHtml(data.timestamp || '') + duration + '</div>'
      + '</div></div>';
  }

  // 各维度得分
  var dimsEl = document.getElementById('health-dimensions');
  if (dimsEl) {
    var dimensions = [
      { key: 'documents', label: '文档层', icon: 'D' },
      { key: 'chunks', label: '切片层', icon: 'C' },
      { key: 'vectors', label: '向量层', icon: 'V' },
      { key: 'retrieval', label: '检索层', icon: 'R' },
      { key: 'index', label: '索引层', icon: 'I' },
    ];

    var dimsHtml = '<div class="health-dims-grid">';
    dimensions.forEach(function(d) {
      var dim = data[d.key];
      if (!dim) return;
      var dimScore = dim.score != null ? dim.score : '—';
      var dimStatus = dim.status || 'unknown';
      var dimColor = dimScore >= 80 ? '#22c55e' : dimScore >= 60 ? '#f59e0b' : dimScore === '—' ? '#6b7280' : '#ef4444';
      var detail = '';
      if (d.key === 'documents') {
        detail = '文档 ' + (dim.total_documents != null ? dim.total_documents : '—') + ' · 重复 ' + (dim.duplicate_count != null ? dim.duplicate_count : '—');
      } else if (d.key === 'chunks') {
        detail = '切片 ' + (dim.total_chunks != null ? dim.total_chunks : '—') + ' · 空 ' + (dim.empty_count != null ? dim.empty_count : '—');
      } else if (d.key === 'vectors') {
        detail = '零向量 ' + (dim.zero_vector_count != null ? dim.zero_vector_count : '—');
      } else if (d.key === 'index') {
        detail = '向量库 ' + (dim.vector_store_ok ? '正常' : '异常') + ' · BM25 ' + (dim.bm25_ok ? '正常' : '异常');
      } else {
        detail = dimStatus;
      }

      dimsHtml += '<div class="health-dim-card">'
        + '<div class="health-dim-icon" style="background:' + dimColor + '20;color:' + dimColor + '">' + d.icon + '</div>'
        + '<div class="health-dim-info">'
        + '<div class="health-dim-label">' + d.label + '</div>'
        + '<div class="health-dim-detail">' + detail + '</div>'
        + '</div>'
        + '<div class="health-dim-score" style="color:' + dimColor + '">' + dimScore + '</div>'
        + '</div>';
    });
    dimsHtml += '</div>';
    dimsEl.innerHTML = dimsHtml;
  }

  // 警告
  var warningsEl = document.getElementById('health-warnings');
  if (warningsEl) {
    var warnings = data.warnings || [];
    if (warnings.length > 0) {
      var wHtml = '<div class="health-section-title">警告 (' + warnings.length + ')</div>';
      warnings.forEach(function(w) {
        var levelColor = w.level === 'error' ? '#ef4444' : w.level === 'warning' ? '#f59e0b' : '#3b82f6';
        wHtml += '<div class="health-warning-item">'
          + '<span class="health-warning-level" style="background:' + levelColor + '">' + escapeHtml(w.level) + '</span>'
          + '<span class="health-warning-msg">' + escapeHtml(w.message) + '</span>'
          + '</div>';
      });
      warningsEl.innerHTML = wHtml;
    } else {
      warningsEl.innerHTML = '';
    }
  }

  // 建议
  var suggestionsEl = document.getElementById('health-suggestions');
  if (suggestionsEl) {
    var suggestions = data.suggestions || [];
    if (suggestions.length > 0) {
      var sHtml = '<div class="health-section-title">修复建议 (' + suggestions.length + ')</div>';
      suggestions.forEach(function(s) {
        sHtml += '<div class="health-suggestion-item">'
          + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
          + '<span>' + escapeHtml(s.message) + '</span>'
          + '</div>';
      });
      suggestionsEl.innerHTML = sHtml;
    } else {
      suggestionsEl.innerHTML = '';
    }
  }
}

async function loadSettingsStatus() {
  try {
    const res = await apiFetch('/api/dashboard');
    const data = await res.json();
    const cache = data.semantic_cache || {};
    var entriesEl = document.getElementById('set-cache-entries');
    if (entriesEl) entriesEl.textContent = cache.entry_count || 0;

    const warmup = data.cache_warmup || {};
    var warmupEl = document.getElementById('set-warmup-ready');
    if (warmupEl) {
      warmupEl.innerHTML = warmup.ready
        ? '<span style="color:#22c55e">已预热</span>'
        : '<span style="color:#f59e0b">未预热</span>';
    }
  } catch (_) {}
}

// ============================================================
// 混合检索/Reranker 开关
// ============================================================

async function toggleHybridSearch(enabled) {
  try {
    await apiFetch('/api/config', {
      method: 'POST',
      body: JSON.stringify({ hybrid_search: enabled }),
    });
    showToast('混合检索已' + (enabled ? '启用' : '关闭'), 'info');
  } catch (e) {
    showToast('配置更新失败', 'error');
  }
}

async function toggleReranker(enabled) {
  try {
    await apiFetch('/api/config', {
      method: 'POST',
      body: JSON.stringify({ reranker: enabled }),
    });
    showToast('Reranker 重排已' + (enabled ? '启用' : '关闭'), 'info');
  } catch (e) {
    showToast('配置更新失败', 'error');
  }
}

// ============================================================
// 文档入库
// ============================================================

async function triggerIngest() {
  if (!confirm('确认重新入库文档？（将保留现有数据，追加新文档）')) return;

  showToast('正在入库文档，请稍候...', 'info');

  try {
    const res = await apiFetch('/api/ingest', {
      method: 'POST',
      body: JSON.stringify({ clear: false }),
    });
    const data = await res.json();

    if (data.success) {
      showToast('入库成功！新增 ' + data.chunks_added + ' 个 Chunk', 'success');
      loadStatus();
    } else {
      showToast(data.error || '入库失败', 'error');
    }
  } catch (e) {
    showToast('入库请求失败：' + e.message, 'error');
  }
}

// ============================================================
// 文件上传
// ============================================================

function triggerUpload() {
  const fileInput = document.getElementById('file-input');
  if (fileInput) {
    fileInput.click();
  }
}

async function handleFileUpload(file) {
  if (!file) return;

  const supportedExts = ['.txt', '.pdf', '.docx', '.doc', '.md', '.xlsx', '.xls', '.csv', '.html', '.htm'];
  const fileName = file.name.toLowerCase();
  const ext = '.' + fileName.split('.').pop();
  if (!supportedExts.includes(ext)) {
    showToast('不支持的文件格式：' + ext, 'error');
    return;
  }

  const maxSize = 50 * 1024 * 1024;
  if (file.size > maxSize) {
    showToast('文件大小不能超过 50MB', 'error');
    return;
  }

  showToast('正在上传并处理：' + file.name + '...', 'info');

  try {
    const formData = new FormData();
    formData.append('file', file);

    const res = await apiFetch('/api/upload', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();

    if (data.success) {
      showToast('上传成功！' + file.name + ' 已入库，新增 ' + data.chunks_added + ' 个 Chunk', 'success');
      loadStatus();
    } else {
      showToast(data.error || '上传失败', 'error');
    }
  } catch (e) {
    showToast('上传请求失败：' + e.message, 'error');
  }
}

// ============================================================
// 消息发送
// ============================================================

function setQuestion(question) {
  const input = document.getElementById('question-input');
  if (input) {
    input.value = question;
    input.focus();
    autoResizeTextarea(input);
  }
}

async function sendMessage() {
  if (isStreaming) {
    stopGeneration();
    return;
  }

  const input = document.getElementById('question-input');
  const question = input.value.trim();
  if (!question) return;

  input.value = '';
  autoResizeTextarea(input);

  var welcome = document.querySelector('.chat-welcome');
  if (welcome) welcome.style.display = 'none';

  appendMessage('user', question);
  conversationHistory.push({ role: 'user', content: question });

  const aiMsgId = 'msg-' + Date.now();
  appendTypingMessage(aiMsgId);

  setBusy(true);

  currentAbortController = new AbortController();

  try {
    await streamAnswer(question, aiMsgId, currentAbortController.signal);
  } catch (e) {
    if (e.name === 'AbortError') {
      updateMessage(aiMsgId, (document.getElementById(aiMsgId)?.querySelector('.message-text')?.textContent || '') + '\n\n[已停止生成]');
    } else {
      updateMessage(aiMsgId, '请求失败：' + e.message);
    }
  } finally {
    setBusy(false);
    currentAbortController = null;
    loadDashboard();
    setTimeout(function() { loadHistoryList(); }, 500);
  }
}

function stopGeneration() {
  if (currentAbortController) {
    currentAbortController.abort();
  }
}

// ============================================================
// SSE 流式接收
// ============================================================

async function streamAnswer(question, aiMsgId, signal) {
  const hybridToggle = document.getElementById('hybrid-toggle');
  const rerankerToggle = document.getElementById('reranker-toggle');
  const hybrid = hybridToggle ? hybridToggle.checked : true;
  const reranker = rerankerToggle ? rerankerToggle.checked : true;

  lastQuestion = question;

  const res = await apiFetch('/api/chat/stream', {
    method: 'POST',
    body: JSON.stringify({
      question,
      session_id: sessionId,
      hybrid,
      reranker,
      style: currentAnswerStyle,
      username: currentUser ? currentUser.username : null
    }),
    signal: signal,
  });

  if (!res.ok) {
    const err = await res.json().catch(function() { return { error: '未知错误' }; });
    updateMessage(aiMsgId, '错误：' + (err.error || '未知错误'));
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let fullAnswer = '';

  const msgEl = document.getElementById(aiMsgId);
  if (msgEl) {
    var textEl = msgEl.querySelector('.message-text');
    if (textEl) {
      textEl.innerHTML = '';
      textEl.classList.add('streaming');
    }
  }

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();

    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      if (!line.startsWith('data:')) continue;
      const jsonStr = line.slice(5).trim();
      if (!jsonStr) continue;

      try {
        const event = JSON.parse(jsonStr);

        if (event.type === 'sources') {
          if (event.sources && event.sources.length > 0) {
            currentSources = event.sources;
          }
        } else if (event.type === 'token') {
          fullAnswer += event.content;
          const textEl = document.getElementById(aiMsgId)?.querySelector('.message-text');
          if (textEl) {
            textEl.textContent = fullAnswer;
            scrollToBottom();
          }
        } else if (event.type === 'done') {
          const textEl = document.getElementById(aiMsgId)?.querySelector('.message-text');
          if (textEl) {
            textEl.classList.remove('streaming');
            textEl.innerHTML = formatAnswerText(fullAnswer);
          }

          lastAnswer = fullAnswer;
          conversationHistory.push({ role: 'assistant', content: fullAnswer });
          updateSessionIndicator();

          if (event.confidence) {
            currentConfidenceData = event.confidence;
            renderConfidenceInline(event.confidence, aiMsgId);
          }

          renderMessageActions(aiMsgId);

          setTimeout(function() { loadHistoryList(); }, 300);
          break;
        } else if (event.type === 'error') {
          updateMessage(aiMsgId, '错误：' + event.message);
          break;
        }
      } catch (_) {}
    }
  }
}

// ============================================================
// 来源展示
// ============================================================

function renderSourcesInline(sources, aiMsgId) {
  const msgEl = document.getElementById(aiMsgId);
  if (!msgEl) return;

  const contentEl = msgEl.querySelector('.message-content');
  if (!contentEl) return;

  const oldSources = contentEl.querySelector('.message-sources');
  if (oldSources) oldSources.remove();

  var sortedSources = sources.slice().sort(function(a, b) {
    return (b.similarity_value || 0) - (a.similarity_value || 0);
  });

  let chipsHtml = '';
  sortedSources.forEach(function(s) {
    let scoreColor = '#22c55e';
    if (s.similarity) {
      const sim = parseFloat(s.similarity);
      if (sim < 60) scoreColor = '#ef4444';
      else if (sim < 80) scoreColor = '#f59e0b';
      else scoreColor = '#22c55e';
    }

    const typeLabel = s.retrieval_type
      ? '<span class="source-chip-type">' + escapeHtml(s.retrieval_type) + '</span>'
      : '';

    let pageLabel = '';
    if (s.page != null && s.page !== undefined) {
      if (s.total_pages != null && s.total_pages !== undefined) {
        pageLabel = '<span class="source-chip-page">第 ' + s.page + '/' + s.total_pages + ' 页</span>';
      } else {
        pageLabel = '<span class="source-chip-page">第 ' + s.page + ' 页</span>';
      }
    }

    chipsHtml += '<div class="source-chip" onclick="event.stopPropagation()">';
    chipsHtml += '<span class="source-chip-title">' + escapeHtml(s.source) + '</span>';
    chipsHtml += pageLabel;
    if (s.similarity) {
      chipsHtml += '<span class="source-chip-score" style="color:' + scoreColor + '">' + s.similarity + '</span>';
    }
    chipsHtml += typeLabel;
    chipsHtml += '<span class="source-chip-toggle">▶</span>';
    chipsHtml += '<div class="source-chip-preview">' + escapeHtml(s.full_content || s.preview) + '</div>';
    chipsHtml += '</div>';
  });

  const sourcesSection = document.createElement('div');
  sourcesSection.className = 'message-sources';
  sourcesSection.innerHTML = '<div class="message-sources-header" onclick="toggleInlineSources(this)">'
    + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">'
    + '<path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>'
    + '<polyline points="14 2 14 8 20 8"/></svg>'
    + '<span>已参考 ' + sources.length + ' 个数据源</span>'
    + '<span class="message-sources-chevron">▸</span></div>'
    + '<div class="message-sources-body" style="display:none">' + chipsHtml + '</div>';

  const textEl = contentEl.querySelector('.message-text');
  contentEl.insertBefore(sourcesSection, textEl);

  sourcesSection.querySelectorAll('.source-chip').forEach(function(chip) {
    chip.addEventListener('click', function() {
      const wasExpanded = chip.classList.contains('expanded');
      // 先收起同组所有已展开的 chip
      sourcesSection.querySelectorAll('.source-chip.expanded').forEach(function(c) {
        c.classList.remove('expanded');
        c.querySelector('.source-chip-toggle').textContent = '▶';
      });
      // 如果之前未展开，则展开当前 chip
      if (!wasExpanded) {
        chip.classList.add('expanded');
        chip.querySelector('.source-chip-toggle').textContent = '▼';
      }
    });
  });
}

function renderSourcesToElement(msgEl, sources) {
  const contentEl = msgEl.querySelector('.message-content');
  if (!contentEl) return;

  var sortedSources = sources.slice().sort(function(a, b) {
    return (b.similarity_value || 0) - (a.similarity_value || 0);
  });

  let chipsHtml = '';
  sortedSources.forEach(function(s) {
    let scoreColor = '#22c55e';
    if (s.similarity) {
      const sim = parseFloat(s.similarity);
      if (sim < 60) scoreColor = '#ef4444';
      else if (sim < 80) scoreColor = '#f59e0b';
      else scoreColor = '#22c55e';
    }

    const typeLabel = s.retrieval_type
      ? '<span class="source-chip-type">' + escapeHtml(s.retrieval_type) + '</span>'
      : '';

    let pageLabel = '';
    if (s.page != null && s.page !== undefined) {
      if (s.total_pages != null && s.total_pages !== undefined) {
        pageLabel = '<span class="source-chip-page">第 ' + s.page + '/' + s.total_pages + ' 页</span>';
      } else {
        pageLabel = '<span class="source-chip-page">第 ' + s.page + ' 页</span>';
      }
    }

    chipsHtml += '<div class="source-chip" onclick="event.stopPropagation()">';
    chipsHtml += '<span class="source-chip-title">' + escapeHtml(s.source) + '</span>';
    chipsHtml += pageLabel;
    if (s.similarity) {
      chipsHtml += '<span class="source-chip-score" style="color:' + scoreColor + '">' + s.similarity + '</span>';
    }
    chipsHtml += typeLabel;
    chipsHtml += '<span class="source-chip-toggle">▶</span>';
    chipsHtml += '<div class="source-chip-preview">' + escapeHtml(s.full_content || s.preview) + '</div>';
    chipsHtml += '</div>';
  });

  const sourcesSection = document.createElement('div');
  sourcesSection.className = 'message-sources';
  sourcesSection.innerHTML = '<div class="message-sources-header" onclick="toggleInlineSources(this)">'
    + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">'
    + '<path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>'
    + '<polyline points="14 2 14 8 20 8"/></svg>'
    + '<span>已参考 ' + sources.length + ' 个数据源</span>'
    + '<span class="message-sources-chevron">▸</span></div>'
    + '<div class="message-sources-body" style="display:none">' + chipsHtml + '</div>';

  const textEl = contentEl.querySelector('.message-text');
  contentEl.insertBefore(sourcesSection, textEl);

  sourcesSection.querySelectorAll('.source-chip').forEach(function(chip) {
    chip.addEventListener('click', function() {
      const wasExpanded = chip.classList.contains('expanded');
      sourcesSection.querySelectorAll('.source-chip.expanded').forEach(function(c) {
        c.classList.remove('expanded');
        c.querySelector('.source-chip-toggle').textContent = '▶';
      });
      if (!wasExpanded) {
        chip.classList.add('expanded');
        chip.querySelector('.source-chip-toggle').textContent = '▼';
      }
    });
  });
}

function toggleInlineSources(headerEl) {
  const section = headerEl.parentElement;
  const body = section.querySelector('.message-sources-body');
  const chevron = section.querySelector('.message-sources-chevron');
  const isOpen = body.style.display !== 'none';

  if (isOpen) {
    body.style.display = 'none';
    chevron.textContent = '▸';
    section.classList.remove('expanded');
  } else {
    body.style.display = '';
    chevron.textContent = '▾';
    section.classList.add('expanded');
  }
}

// ============================================================
// 对话历史管理
// ============================================================

function updateSessionIndicator() {
  const indicator = document.getElementById('session-indicator');
  if (indicator) {
    const turn = Math.floor(conversationHistory.length / 2);
    indicator.textContent = turn > 0 ? '第 ' + turn + ' 轮对话' : '新对话';
  }
}

async function clearCurrentChat() {
  const messages = document.getElementById('messages');
  messages.innerHTML = '<div class="chat-welcome">'
    + '<div class="welcome-icon">'
    + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="48" height="48">'
    + '<path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>'
    + '</div><h2>RAG 智能问答助手</h2>'
    + '<p>' + (window._techStackDesc || '基于您的知识库文档，精准回答问题。支持多轮对话、混合检索与 Reranker 重排。') + '</p>'
    + '<div class="welcome-hints">'
    + '<button class="hint-chip" onclick="setQuestion(\'请总结知识库中的核心内容\')">总结核心内容</button>'
    + '<button class="hint-chip" onclick="setQuestion(\'知识库中有哪些关键概念\')">关键概念</button>'
    + '<button class="hint-chip" onclick="setQuestion(\'请帮我分析知识库中的主要观点\')">分析主要观点</button>'
    + '</div></div>';
  conversationHistory = [];

  try {
    await apiFetch('/api/history/' + sessionId, { method: 'DELETE' });
  } catch (_) {}

  updateSessionIndicator();
}

async function startNewSession() {
  sessionId = generateSessionId();
  localStorage.setItem("rag_session_id", sessionId);
  if (currentTab !== 'chat') {
    switchTab('chat');
  }
  clearCurrentChat();
  loadHistoryList();
  showToast('已开启新对话', 'info');
}

// ============================================================
// 侧边栏交互
// ============================================================

function toggleSidebar() {
  var sidebar = document.getElementById('sidebar');
  if (sidebar) {
    sidebar.classList.toggle('collapsed');
  }
}

function toggleUserMenu() {
  var menu = document.getElementById('sidebar-user-menu');
  var chevron = document.getElementById('sidebar-user-chevron');
  if (!menu) return;

  if (menu.style.display === 'none' || !menu.style.display) {
    menu.style.display = 'block';
    if (chevron) chevron.classList.add('open');
  } else {
    menu.style.display = 'none';
    if (chevron) chevron.classList.remove('open');
  }
}

function openManageDialog() {
  var overlay = document.getElementById('manage-modal-overlay');
  if (overlay) {
    overlay.style.display = 'flex';
    renderManageList(allSessionsCache);
  }
}

function closeManageDialog() {
  var overlay = document.getElementById('manage-modal-overlay');
  if (overlay) {
    overlay.style.display = 'none';
  }
}

// ============================================================
// 历史对话列表（侧边栏 + 管理弹窗共用）
// ============================================================

var allSessionsCache = [];

async function loadHistoryList() {
  try {
    const res = await apiFetch('/api/sessions');
    if (!res.ok) throw new Error('服务器返回 ' + res.status);
    const data = await res.json();
    const sessions = data.sessions || [];
    allSessionsCache = sessions;

    renderSidebarHistory(sessions);
    renderManageList(sessions);
  } catch (e) {
    console.error('加载历史列表失败：', e);
  }
}

function renderSidebarHistory(sessions) {
  var listEl = document.getElementById('sidebar-history-list');
  if (!listEl) return;

  if (!sessions || sessions.length === 0) {
    listEl.innerHTML = '<div class="sidebar-history-empty">暂无历史对话</div>';
    return;
  }

  var searchTerm = (document.getElementById('sidebar-search-input')?.value || '').toLowerCase();
  var filtered = sessions;
  if (searchTerm) {
    filtered = sessions.filter(function(s) {
      var name = (s.session_name || s.session_id || '').toLowerCase();
      return name.indexOf(searchTerm) !== -1;
    });
  }

  if (filtered.length === 0) {
    listEl.innerHTML = '<div class="sidebar-history-empty">未找到匹配的对话</div>';
    return;
  }

  var html = '';
  filtered.forEach(function(s) {
    var isActive = s.session_id === sessionId;
    var name = s.session_name || s.session_id.substring(0, 12) + '...';
    var time = formatHistoryTime(s.last_active_at || s.created_at);
    html += '<div class="sidebar-history-item' + (isActive ? ' active' : '') + '" data-session-id="' + escapeHtml(s.session_id) + '" data-session-name="' + escapeHtml(name) + '">';
    html += '<span class="sidebar-history-item-title" data-sid="' + escapeHtml(s.session_id) + '">' + escapeHtml(name) + '</span>';
    html += '<button class="sidebar-history-item-rename-btn" title="重命名" data-sid="' + escapeHtml(s.session_id) + '">';
    html += '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>';
    html += '</button>';
    html += '<span class="sidebar-history-item-time">' + escapeHtml(time) + '</span>';
    html += '</div>';
  });

  listEl.innerHTML = html;

  // 绑定会话项单击切换 + 编辑按钮 + 双击重命名
  listEl.querySelectorAll('.sidebar-history-item').forEach(function(itemEl) {
    var sid = itemEl.getAttribute('data-session-id');
    var clickTimer = null;

    itemEl.addEventListener('click', function(e) {
      // 编辑按钮的点击不触发切换
      if (e.target.closest('.sidebar-history-item-rename-btn')) return;
      // 如果是标题区双击，延迟切换让双击优先
      if (e.target.closest('.sidebar-history-item-title')) {
        if (clickTimer) {
          clearTimeout(clickTimer);
          clickTimer = null;
          return; // 双击已处理，不切换
        }
        clickTimer = setTimeout(function() {
          clickTimer = null;
          switchToSession(sid);
        }, 300);
        return;
      }
      switchToSession(sid);
    });

    var titleEl = itemEl.querySelector('.sidebar-history-item-title');
    if (titleEl) {
      titleEl.addEventListener('dblclick', function(e) {
        e.stopPropagation();
        if (clickTimer) {
          clearTimeout(clickTimer);
          clickTimer = null;
        }
        startRename(titleEl);
      });
    }
  });

  // 绑定编辑按钮事件
  listEl.querySelectorAll('.sidebar-history-item-rename-btn').forEach(function(btn) {
    btn.addEventListener('click', function(e) {
      e.stopPropagation();
      e.preventDefault();
      var sid = btn.getAttribute('data-sid');
      var titleEl = btn.parentElement.querySelector('.sidebar-history-item-title');
      if (titleEl) startRename(titleEl);
    });
  });
}

function filterSidebarHistory() {
  renderSidebarHistory(allSessionsCache);
}

function startRename(titleEl) {
  const sid = titleEl.getAttribute('data-sid');
  if (!sid) return;

  const oldName = titleEl.textContent;
  const input = document.createElement('input');
  input.type = 'text';
  input.value = oldName;
  input.className = 'sidebar-rename-input';
  input.style.cssText = 'width:100%;padding:2px 4px;border:1px solid var(--accent);border-radius:4px;font-size:12px;background:var(--bg-surface);color:var(--text-primary);outline:none;';

  titleEl.textContent = '';
  titleEl.appendChild(input);
  input.focus();
  input.select();

  var committed = false;

  function commit() {
    if (committed) return;
    committed = true;
    var newName = input.value.trim();
    if (!newName || newName === oldName) {
      titleEl.textContent = oldName;
      return;
    }
    finishRename(sid, newName, titleEl);
  }

  input.addEventListener('blur', commit);
  input.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      input.blur();
    } else if (e.key === 'Escape') {
      committed = true;
      titleEl.textContent = oldName;
    }
  });
}

async function finishRename(sid, newName, titleEl) {
  try {
    var res = await apiFetch('/api/sessions/' + sid + '/rename', {
      method: 'POST',
      body: JSON.stringify({ name: newName }),
    });
    if (res.ok) {
      titleEl.textContent = newName;
      showToast('重命名成功', 'success');
      loadHistoryList();
    } else {
      var data = await res.json();
      titleEl.textContent = titleEl.getAttribute('data-sid') ? (allSessionsCache.find(function(s) { return s.session_id === sid; })?.session_name || sid.substring(0, 12) + '...') : sid.substring(0, 12) + '...';
      showToast(data.error || '重命名失败', 'error');
    }
  } catch (e) {
    showToast('请求失败：' + e.message, 'error');
  }
}

function renderManageList(sessions) {
  var listEl = document.getElementById('manage-modal-list');
  if (!listEl) return;

  if (!sessions || sessions.length === 0) {
    listEl.innerHTML = '<div class="manage-modal-empty">暂无对话记录</div>';
    return;
  }

  var searchTerm = (document.getElementById('manage-search-input')?.value || '').toLowerCase();
  var filtered = sessions;
  if (searchTerm) {
    filtered = sessions.filter(function(s) {
      var name = (s.session_name || s.session_id || '').toLowerCase();
      return name.indexOf(searchTerm) !== -1;
    });
  }

  if (filtered.length === 0) {
    listEl.innerHTML = '<div class="manage-modal-empty">未找到匹配的对话</div>';
    return;
  }

  var html = '';
  filtered.forEach(function(s) {
    var name = s.session_name || s.session_id.substring(0, 12) + '...';
    var time = formatHistoryTime(s.last_active_at || s.created_at);
    var msgCount = s.message_count || 0;
    html += '<div class="manage-modal-item">';
    html += '<label class="manage-modal-item-check">';
    html += '<input type="checkbox" class="manage-checkbox" value="' + escapeHtml(s.session_id) + '" onchange="updateBatchDeleteButton()" />';
    html += '</label>';
    html += '<div class="manage-modal-item-info">';
    html += '<div class="manage-modal-item-title-wrap">';
    html += '<span class="manage-modal-item-title" data-sid="' + escapeHtml(s.session_id) + '" title="双击重命名">' + escapeHtml(name) + '</span>';
    html += '<button class="manage-modal-item-rename-btn" title="重命名" data-sid="' + escapeHtml(s.session_id) + '" onclick="event.stopPropagation(); startManageRename(this)">';
    html += '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>';
    html += '</button>';
    html += '</div>';
    html += '<div class="manage-modal-item-meta">' + escapeHtml(time) + ' · ' + msgCount + ' 条消息</div>';
    html += '</div>';
    html += '<button class="manage-modal-item-delete" title="删除" onclick="deleteSessionFromManage(\'' + escapeHtml(s.session_id) + '\')">';
    html += '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>';
    html += '</button>';
    html += '</div>';
  });

  listEl.innerHTML = html;

  // 绑定管理列表标题的双击重命名
  listEl.querySelectorAll('.manage-modal-item-title').forEach(function(titleEl) {
    var clickTimer = null;
    titleEl.addEventListener('click', function(e) {
      if (clickTimer) { clearTimeout(clickTimer); clickTimer = null; return; }
      clickTimer = setTimeout(function() { clickTimer = null; }, 300);
    });
    titleEl.addEventListener('dblclick', function(e) {
      e.stopPropagation();
      if (clickTimer) { clearTimeout(clickTimer); clickTimer = null; }
      startManageRenameFromTitle(titleEl);
    });
  });

  // 重置全选状态
  var selectAll = document.getElementById('manage-select-all');
  if (selectAll) selectAll.checked = false;
  updateBatchDeleteButton();
}

function startManageRename(btn) {
  var sid = btn.getAttribute('data-sid');
  var titleWrap = btn.parentElement;
  var titleEl = titleWrap.querySelector('.manage-modal-item-title');
  if (!titleEl) return;
  startManageRenameFromTitle(titleEl);
}

function startManageRenameFromTitle(titleEl) {
  var sid = titleEl.getAttribute('data-sid');
  if (!sid) return;

  var oldName = titleEl.textContent;
  var input = document.createElement('input');
  input.type = 'text';
  input.value = oldName;
  input.className = 'manage-rename-input';
  input.style.cssText = 'width:100%;padding:2px 4px;border:1px solid var(--accent);border-radius:4px;font-size:13px;background:var(--bg-surface);color:var(--text-primary);outline:none;';

  titleEl.textContent = '';
  titleEl.appendChild(input);
  input.focus();
  input.select();

  var committed = false;

  function commit() {
    if (committed) return;
    committed = true;
    var newName = input.value.trim();
    if (!newName || newName === oldName) {
      titleEl.textContent = oldName;
      return;
    }
    finishManageRename(sid, newName, titleEl);
  }

  input.addEventListener('blur', commit);
  input.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
    else if (e.key === 'Escape') { committed = true; titleEl.textContent = oldName; }
  });
}

async function finishManageRename(sid, newName, titleEl) {
  try {
    var res = await apiFetch('/api/sessions/' + sid + '/rename', {
      method: 'POST',
      body: JSON.stringify({ name: newName }),
    });
    if (res.ok) {
      titleEl.textContent = newName;
      showToast('重命名成功', 'success');
      loadHistoryList();
    } else {
      var data = await res.json();
      var oldName = (allSessionsCache.find(function(s) { return s.session_id === sid; }) || {}).session_name || sid.substring(0, 12) + '...';
      titleEl.textContent = oldName;
      showToast(data.error || '重命名失败', 'error');
    }
  } catch (e) {
    showToast('请求失败：' + e.message, 'error');
  }
}

function filterManageList() {
  renderManageList(allSessionsCache);
}

function toggleSelectAllSessions() {
  var selectAll = document.getElementById('manage-select-all');
  var checked = selectAll ? selectAll.checked : false;
  document.querySelectorAll('.manage-checkbox').forEach(function(cb) {
    cb.checked = checked;
  });
  updateBatchDeleteButton();
}

function updateBatchDeleteButton() {
  var checkedCount = document.querySelectorAll('.manage-checkbox:checked').length;
  var btn = document.getElementById('manage-batch-delete-btn');
  if (btn) {
    btn.style.display = checkedCount > 0 ? '' : 'none';
    btn.textContent = '批量删除(' + checkedCount + ')';
  }
}

async function batchDeleteSessions() {
  var checked = document.querySelectorAll('.manage-checkbox:checked');
  if (checked.length === 0) return;

  if (!confirm('确定要删除选中的 ' + checked.length + ' 个对话记录吗？此操作不可恢复。')) return;

  var deletedCount = 0;
  var currentSessionDeleted = false;

  for (var i = 0; i < checked.length; i++) {
    var sid = checked[i].value;
    try {
      var res = await apiFetch('/api/history/' + sid, { method: 'DELETE' });
      if (res.ok) {
        deletedCount++;
        if (sid === sessionId) {
          currentSessionDeleted = true;
        }
      }
    } catch (_) {}
  }

  if (currentSessionDeleted) {
    startNewSession();
  }

  showToast('已删除 ' + deletedCount + ' 个对话', 'success');
  loadHistoryList();
}

async function deleteSessionFromManage(sid) {
  if (!confirm('确定要删除该对话记录吗？此操作不可恢复。')) return;

  try {
    var res = await apiFetch('/api/history/' + sid, { method: 'DELETE' });
    var data = await res.json();
    if (res.ok) {
      showToast('对话已删除', 'success');
      if (sid === sessionId) {
        startNewSession();
      }
      loadHistoryList();
    } else {
      showToast(data.error || '删除失败', 'error');
    }
  } catch (e) {
    showToast('请求失败：' + e.message, 'error');
  }
}

function formatHistoryTime(isoStr) {
  if (!isoStr) return '';
  try {
    var d = new Date(isoStr);
    var now = new Date();
    var diff = now - d;
    if (diff < 60000) return '刚刚';
    if (diff < 3600000) return Math.floor(diff / 60000) + '分钟前';
    if (diff < 86400000) return Math.floor(diff / 3600000) + '小时前';
    return (d.getMonth() + 1) + '/' + d.getDate() + ' ' +
           String(d.getHours()).padStart(2, '0') + ':' +
           String(d.getMinutes()).padStart(2, '0');
  } catch (_) {
    return '';
  }
}

async function switchToSession(targetSessionId) {
  if (!targetSessionId) return;

  var chatPanel = document.getElementById('panel-chat');
  if (!chatPanel) {
    showToast('切换失败：找不到对话面板', 'error');
    return;
  }

  // 切换到同一个会话：只切回对话面板，不重新加载历史
  if (targetSessionId === sessionId) {
    document.querySelectorAll('.tab-panel').forEach(function(p) {
      p.classList.remove('active');
    });
    chatPanel.classList.add('active');
    currentTab = 'chat';
    loadHistoryList();
    scrollToBottom();
    return;
  }

  var previousSessionId = sessionId;

  document.querySelectorAll('.tab-panel').forEach(function(p) {
    p.classList.remove('active');
  });
  chatPanel.classList.add('active');

  currentTab = 'chat';

  var messagesEl = document.getElementById('messages');
  if (messagesEl) messagesEl.innerHTML = '';

  var welcome = document.querySelector('.chat-welcome');
  if (welcome) welcome.style.display = 'none';

  sessionId = targetSessionId;
  localStorage.setItem('rag_session_id', sessionId);

  conversationHistory = [];
  updateSessionIndicator();
  loadHistoryList();

  try {
    var res = await apiFetch('/api/history/' + encodeURIComponent(targetSessionId));
    if (!res.ok) throw new Error('服务器返回 ' + res.status);
    var data = await res.json();
    var history = data.history || [];

    if (history.length > 0) {
      var lastUserQuestion = '';
      var qaPairs = [];
      var msgIds = [];
      history.forEach(function(msg) {
        conversationHistory.push({ role: msg.role, content: msg.content });
        var msgId = appendMessage(msg.role, msg.content, msg.sources);
        if (msg.role === 'user') {
          lastUserQuestion = msg.content;
        } else if (msg.role === 'assistant') {
          qaPairs.push({ question: lastUserQuestion, answer: msg.content });
          msgIds.push({ msgId: msgId, question: lastUserQuestion, answer: msg.content });
          if (msg.confidence) {
            currentConfidenceData = msg.confidence;
            renderConfidenceInline(msg.confidence, msgId);
          }
        }
      });

      // 批量查询反馈状态
      var feedbackMap = {};
      var bookmarkMap = {};
      if (qaPairs.length > 0 && currentUser) {
        try {
          var fbRes = await apiFetch('/api/feedback/batch-check', {
            method: 'POST',
            body: JSON.stringify({
              qa_pairs: qaPairs,
              username: currentUser.username
            })
          });
          var fbData = await fbRes.json();
          feedbackMap = fbData.feedback_map || {};
        } catch (_) {}

        try {
          var bmRes = await apiFetch('/api/bookmarks/' + encodeURIComponent(currentUser.username) + '/batch-check', {
            method: 'POST',
            body: JSON.stringify({
              qa_pairs: qaPairs
            })
          });
          var bmData = await bmRes.json();
          bookmarkMap = bmData.bookmark_map || {};
        } catch (_) {}
      }

      // 渲染消息操作按钮（带反馈状态和收藏状态）
      msgIds.forEach(function(item) {
        var key = item.question + '|||' + item.answer;
        var fb = feedbackMap[key];
        var isBookmarked = bookmarkMap[key] || false;
        renderMessageActions(item.msgId, item.question, item.answer, fb ? fb.rating : null, isBookmarked);
      });
    } else {
      if (welcome) welcome.style.display = '';
    }

    updateSessionIndicator();
    scrollToBottom();
    showToast('已切换到历史对话', 'info');
  } catch (e) {
    sessionId = previousSessionId;
    localStorage.setItem('rag_session_id', previousSessionId);
    updateSessionIndicator();
    showToast('切换对话失败：' + e.message, 'error');
  }
}

async function renameSession(sid) {
  const newName = prompt('请输入新的会话名称：');
  if (!newName || !newName.trim()) return;

  try {
    const res = await apiFetch('/api/sessions/' + sid + '/rename', {
      method: 'POST',
      body: JSON.stringify({ name: newName.trim() }),
    });
    const data = await res.json();
    if (data.success) {
      showToast('重命名成功', 'success');
      loadHistoryList();
    } else {
      showToast(data.error || '重命名失败', 'error');
    }
  } catch (e) {
    showToast('请求失败：' + e.message, 'error');
  }
}

async function deleteSession(sid) {
  if (!confirm('确定要删除该会话吗？此操作不可恢复。')) return;

  try {
    const res = await apiFetch('/api/history/' + sid, { method: 'DELETE' });
    const data = await res.json();
    if (res.ok) {
      showToast('会话已删除', 'success');
      if (sid === sessionId) {
        startNewSession();
      } else {
        loadHistoryList();
      }
    } else {
      showToast(data.error || '删除失败', 'error');
    }
  } catch (e) {
    showToast('请求失败：' + e.message, 'error');
  }
}

// ============================================================
// 工具函数
// ============================================================

function appendMessage(role, text, sources) {
  const messages = document.getElementById('messages');
  const div = document.createElement('div');
  var msgId = 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 6);
  div.id = msgId;
  div.className = 'message ' + role;
  var displayText = role === 'assistant' ? formatAnswerText(text) : escapeHtml(text);
  div.innerHTML = '<div class="message-avatar">' + (role === 'user' ? '👤' : '🤖') + '</div>'
    + '<div class="message-content"><div class="message-text">' + displayText + '</div></div>';
  messages.appendChild(div);

  if (sources && sources.length > 0) {
    renderSourcesToElement(div, sources);
  }

  scrollToBottom();
  return msgId;
}

function appendTypingMessage(id) {
  const messages = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'message assistant';
  div.id = id;
  div.innerHTML = '<div class="message-avatar">🤖</div>'
    + '<div class="message-content"><div class="message-text">'
    + '<div class="typing-indicator"><span></span><span></span><span></span></div>'
    + '</div></div>';
  messages.appendChild(div);
  scrollToBottom();
}

function updateMessage(id, text) {
  const el = document.getElementById(id);
  if (el) {
    const textEl = el.querySelector('.message-text');
    textEl.innerHTML = formatAnswerText(text);
    textEl.classList.remove('streaming');
  }
}

function clearChat() {
  clearCurrentChat();
}

function scrollToBottom() {
  const messages = document.getElementById('messages');
  messages.scrollTop = messages.scrollHeight;
}

function setBusy(busy) {
  isStreaming = busy;
  const btn = document.getElementById('send-btn');
  const sendIcon = document.getElementById('send-icon');
  const stopIcon = document.getElementById('stop-icon');
  const input = document.getElementById('question-input');

  if (busy) {
    btn.title = '停止生成';
    btn.onclick = stopGeneration;
    if (sendIcon) sendIcon.style.display = 'none';
    if (stopIcon) stopIcon.style.display = '';
    btn.classList.add('stop-active');
  } else {
    btn.title = '发送';
    btn.onclick = sendMessage;
    if (sendIcon) sendIcon.style.display = '';
    if (stopIcon) stopIcon.style.display = 'none';
    btn.classList.remove('stop-active');
  }

  input.disabled = busy;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function formatAnswerText(text) {
  var raw = String(text).replace(/\\n/g, '\n');
  raw = escapeHtml(raw);

  /* 代码块：``` ... ``` */
  var codeBlocks = [];
  raw = raw.replace(/```([\s\S]*?)```/g, function(match, code) {
    var idx = codeBlocks.length;
    codeBlocks.push('<pre><code>' + code.trim() + '</code></pre>');
    return '\x00CODE' + idx + '\x00';
  });

  /* 行内代码：`...` */
  raw = raw.replace(/`([^`\n]+)`/g, '<code>$1</code>');

  /* 标题：# / ## / ### */
  raw = raw.replace(/^### (.+)$/gm, '<h4>$1</h4>');
  raw = raw.replace(/^## (.+)$/gm, '<h3>$1</h3>');
  raw = raw.replace(/^# (.+)$/gm, '<h3 class="msg-h3">$1</h3>');

  /* 粗体：**text** 或 __text__ */
  raw = raw.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  raw = raw.replace(/__(.+?)__/g, '<strong>$1</strong>');

  /* 斜体：*text* 或 _text_（排除已处理的粗体） */
  raw = raw.replace(/(?<![*])\*([^*\n]+)\*(?!\*)/g, '<em>$1</em>');
  raw = raw.replace(/(?<![_])_([^_\n]+)_(?!_)/g, '<em>$1</em>');

  /* 无序列表：- item 或 * item（行首） */
  raw = raw.replace(/^[\s]*[-*] (.+)$/gm, '<li>$1</li>');
  raw = raw.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

  /* 有序列表：1. item（行首） */
  raw = raw.replace(/^[\s]*\d+[.)] (.+)$/gm, '<oli>$1</oli>');
  raw = raw.replace(/((?:<oli>.*<\/oli>\n?)+)/g, function(m) { return m.replace(/<oli>/g, '<li>').replace(/<\/oli>/g, '</li>').replace(/^(<li>.+<\/li>\n?)+$/m, '<ol>$&</ol>'); });

  /* 引用块：> text */
  raw = raw.replace(/^&gt; (.+)$/gm, '<blockquote-line>$1</blockquote-line>');
  raw = raw.replace(/((?:<blockquote-line>.*<\/blockquote-line>\n?)+)/g, function(m) { return m.replace(/<blockquote-line>/g, '').replace(/<\/blockquote-line>/g, '').replace(/^/gm, '> '); return '<blockquote><p>' + m + '</p></blockquote>'; });

  /* 分隔线：--- 或 *** */
  raw = raw.replace(/^[-*]{3,}$/gm, '<hr>');

  /* 段落：连续空行分隔为 <p> */
  var parts = raw.split(/\n{2,}/);
  for (var i = 0; i < parts.length; i++) {
    var p = parts[i].trim();
    if (!p || /^(<[hou][tl]|<pre|<ul|<ol|<blockquote|<hr)/.test(p)) continue;
    parts[i] = '<p>' + p + '</p>';
  }
  raw = parts.join('\n');

  /* 单行换行：<br>（在段落和列表内部） */
  raw = raw.replace(/\n/g, '<br>');

  /* 恢复代码块占位符 */
  for (var j = 0; j < codeBlocks.length; j++) {
    raw = raw.replace('\x00CODE' + j + '\x00', codeBlocks[j]);
  }

  return raw;
}

function showToast(msg, type) {
  type = type || 'info';
  var container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.className = 'toast-container';
    container.id = 'toast-container';
    document.body.appendChild(container);
  }
  const toast = document.createElement('div');
  toast.className = 'toast ' + type;
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(function() { toast.remove(); }, 3500);
}

function autoResizeTextarea(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

// ============================================================
// 仪表盘
// ============================================================

async function loadDashboard() {
  try {
    const res = await apiFetch('/api/dashboard');
    const data = await res.json();

    // 更新首页技术栈描述
    var techStackEl = document.getElementById('welcome-tech-stack');
    if (techStackEl && data.tech_stack) {
      window._techStackDesc = data.tech_stack + '。支持多轮对话、混合检索与 Reranker 重排。';
      techStackEl.textContent = window._techStackDesc;
    }

    // 概览卡片
    const kb = data.knowledge_base || {};
    var docCountEl = document.getElementById('dash-doc-count');
    if (docCountEl) {
      docCountEl.textContent = kb.document_count || 0;
      docCountEl.style.color = kb.ready ? '#22c55e' : '#ef4444';
    }

    const sessions = data.sessions || {};
    var sessionsEl = document.getElementById('dash-sessions');
    if (sessionsEl) sessionsEl.textContent = sessions.active_count || 0;

    const evalData = data.evaluation || {};
    var totalReqEl = document.getElementById('dash-total-requests');
    if (totalReqEl) totalReqEl.textContent = evalData.total_requests || 0;
    var llmReqEl = document.getElementById('dash-llm-requests');
    if (llmReqEl) llmReqEl.textContent = evalData.llm_requests || 0;
    var hitRateEl = document.getElementById('dash-cache-hit-rate');
    if (hitRateEl) hitRateEl.textContent = evalData.overall_hit_rate || '0.0%';

    var avgLatencyEl = document.getElementById('dash-avg-latency');
    if (avgLatencyEl) {
      var avgLatency = evalData.avg_llm_latency_ms || 0;
      avgLatencyEl.textContent = avgLatency + ' ms';
    }

    // Token 统计卡片
    var totalTokensEl = document.getElementById('dash-total-tokens');
    if (totalTokensEl) totalTokensEl.textContent = (evalData.total_tokens || 0).toLocaleString();
    var inputTokensEl = document.getElementById('dash-input-tokens');
    if (inputTokensEl) inputTokensEl.textContent = (evalData.total_input_tokens || 0).toLocaleString();
    var outputTokensEl = document.getElementById('dash-output-tokens');
    if (outputTokensEl) outputTokensEl.textContent = (evalData.total_output_tokens || 0).toLocaleString();
    var avgTokensEl = document.getElementById('dash-avg-tokens');
    if (avgTokensEl) {
      var avgTokens = evalData.total_requests > 0
        ? Math.round((evalData.total_tokens || 0) / evalData.total_requests)
        : 0;
      avgTokensEl.textContent = avgTokens.toLocaleString();
    }

    // 最近请求详情表格
    var recentRequests = evalData.recent_requests || [];
    var tbody = document.getElementById('dash-recent-requests-body');
    if (tbody) {
      if (recentRequests.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" class="td-empty">暂无请求记录</td></tr>';
      } else {
        var rowsHtml = '';
        for (var i = recentRequests.length - 1; i >= 0; i--) {
          var r = recentRequests[i];
          var question = r.question || '—';
          var source = r.from_cache ? '<span class="badge badge-cache">缓存</span>' : '<span class="badge badge-llm">LLM</span>';
          var matchType = r.cache_match_type || '—';
          var retrievalLatency = r.retrieval_latency_ms != null ? r.retrieval_latency_ms + ' ms' : '—';
          var llmLatency = r.from_cache ? '—' : (r.latency_ms != null ? r.latency_ms + ' ms' : '—');
          var sourceCount = r.source_count != null ? r.source_count : '—';
          var inputTokens = r.input_tokens != null ? r.input_tokens.toLocaleString() : '0';
          var outputTokens = r.output_tokens != null ? r.output_tokens.toLocaleString() : '0';
          var totalTokens = r.total_tokens != null ? r.total_tokens.toLocaleString() : '0';
          var time = r.timestamp ? r.timestamp.substring(11, 19) : '—';
          rowsHtml += '<tr>'
            + '<td title="' + escapeHtml(r.question || '') + '">' + escapeHtml(question) + '</td>'
            + '<td>' + source + '</td>'
            + '<td>' + matchType + '</td>'
            + '<td>' + retrievalLatency + '</td>'
            + '<td>' + llmLatency + '</td>'
            + '<td>' + sourceCount + '</td>'
            + '<td>' + inputTokens + '</td>'
            + '<td>' + outputTokens + '</td>'
            + '<td>' + totalTokens + '</td>'
            + '<td>' + time + '</td>'
            + '</tr>';
        }
        tbody.innerHTML = rowsHtml;
      }
    }

    // 系统信息
    const sys = data.system || {};
    var sysStatusEl = document.getElementById('dash-sys-status');
    if (sysStatusEl) sysStatusEl.innerHTML = '<span style="color:#22c55e">运行中</span>';
    var pyVerEl = document.getElementById('dash-py-ver');
    if (pyVerEl) pyVerEl.textContent = sys.python_version || '—';
    var osEl = document.getElementById('dash-os');
    if (osEl) osEl.textContent = sys.os || '—';
    var cpuEl = document.getElementById('dash-cpu');
    if (cpuEl) cpuEl.textContent = sys.cpu_count || '—';
    var memUsedEl = document.getElementById('dash-mem-used');
    if (memUsedEl) memUsedEl.textContent = (sys.memory_used_mb || 0) + ' MB';
    var memFreeEl = document.getElementById('dash-mem-free');
    if (memFreeEl) memFreeEl.textContent = (sys.memory_free_mb || 0) + ' MB';

    // 模型信息
    const models = data.models || {};
    var llmModelEl = document.getElementById('dash-llm');
    if (llmModelEl) llmModelEl.textContent = models.llm || '—';
    var embedModelEl = document.getElementById('dash-embed');
    if (embedModelEl) embedModelEl.textContent = models.embedding || '—';
    var dimEl = document.getElementById('dash-dim');
    if (dimEl) dimEl.textContent = models.embedding_dimension || '—';
    var rerankerEl = document.getElementById('dash-reranker');
    if (rerankerEl) rerankerEl.textContent = models.reranker || '—';
    var hybridEl = document.getElementById('dash-hybrid');
    if (hybridEl) {
      hybridEl.innerHTML = models.hybrid_search
        ? '<span style="color:#22c55e">已启用</span>'
        : '<span style="color:#999">已禁用</span>';
    }

    // 缓存统计
    const cache = data.semantic_cache || {};
    var cacheStatusEl = document.getElementById('dash-cache-status');
    if (cacheStatusEl) {
      cacheStatusEl.innerHTML = cache.enabled
        ? '<span style="color:#22c55e">已启用</span>'
        : '<span style="color:#999">已禁用</span>';
    }
    var cacheEntriesEl = document.getElementById('dash-cache-entries');
    if (cacheEntriesEl) cacheEntriesEl.textContent = cache.entry_count || 0;
    var cacheExactEl = document.getElementById('dash-cache-exact');
    if (cacheExactEl) cacheExactEl.textContent = cache.exact_hit_count || 0;
    var cacheSemanticEl = document.getElementById('dash-cache-semantic');
    if (cacheSemanticEl) cacheSemanticEl.textContent = cache.semantic_hit_count || 0;
    var cacheMissesEl = document.getElementById('dash-cache-misses');
    if (cacheMissesEl) cacheMissesEl.textContent = cache.miss_count || 0;
    var cacheThresholdEl = document.getElementById('dash-cache-threshold');
    if (cacheThresholdEl) {
      cacheThresholdEl.textContent = (cache.similarity_threshold || '—') + ' / ' + (cache.ttl_hours || '—') + 'h';
    }

    // 评估统计
    var evalTotalEl = document.getElementById('dash-eval-total');
    if (evalTotalEl) evalTotalEl.textContent = evalData.total_requests || 0;
    var evalLlmEl = document.getElementById('dash-eval-llm');
    if (evalLlmEl) evalLlmEl.textContent = evalData.llm_requests || 0;
    var evalHitRateEl = document.getElementById('dash-eval-hit-rate');
    if (evalHitRateEl) evalHitRateEl.textContent = evalData.overall_hit_rate || '0.0%';
    var evalExactRateEl = document.getElementById('dash-eval-exact-rate');
    if (evalExactRateEl) evalExactRateEl.textContent = evalData.exact_hit_rate || '0.0%';
    var evalSemanticRateEl = document.getElementById('dash-eval-semantic-rate');
    if (evalSemanticRateEl) evalSemanticRateEl.textContent = evalData.semantic_hit_rate || '0.0%';
    var evalLlmLatencyEl = document.getElementById('dash-eval-llm-latency');
    if (evalLlmLatencyEl) evalLlmLatencyEl.textContent = (evalData.avg_llm_latency_ms || '0') + ' ms';
    var evalRetrievalLatencyEl = document.getElementById('dash-eval-retrieval-latency');
    if (evalRetrievalLatencyEl) evalRetrievalLatencyEl.textContent = (evalData.avg_retrieval_latency_ms || '0') + ' ms';

    // 渲染图表
    renderDashCharts(evalData, cache);

  } catch (e) {
    console.error('加载仪表盘失败：', e);
  }
}

let dashCharts = {};

function renderDashCharts(evalData, cacheData) {
  Object.values(dashCharts).forEach(function(c) { c.destroy(); });
  dashCharts = {};

  var total = evalData.total_requests || 0;
  var cacheHits = evalData.cache_hits || 0;
  var cacheMisses = evalData.cache_misses || 0;
  var llmRequests = evalData.llm_requests || 0;
  var exactHits = evalData.exact_cache_hits || 0;
  var semanticHits = evalData.semantic_cache_hits || 0;

  // 缓存命中率饼图
  var hitCtx = document.getElementById('chart-cache-hit');
  if (hitCtx && total > 0) {
    dashCharts.cacheHit = new Chart(hitCtx, {
      type: 'doughnut',
      data: {
        labels: ['缓存命中', 'LLM 调用'],
        datasets: [{
          data: [cacheHits, cacheMisses],
          backgroundColor: ['#22c55e', '#f59e0b'],
          borderColor: ['#16a34a', '#d97706'],
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom', labels: { padding: 16, font: { size: 11 } } } }
      }
    });
  }

  // 请求统计柱状图
  var reqCtx = document.getElementById('chart-requests');
  if (reqCtx) {
    dashCharts.requests = new Chart(reqCtx, {
      type: 'bar',
      data: {
        labels: ['总请求', 'LLM调用', '缓存命中', '精确命中', '语义命中'],
        datasets: [{
          label: '次数',
          data: [total, llmRequests, cacheHits, exactHits, semanticHits],
          backgroundColor: ['#5b5fe3', '#f59e0b', '#22c55e', '#06b6d4', '#8b5cf6'],
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } }
      }
    });
  }

  // 匹配类型分布饼图
  var matchCtx = document.getElementById('chart-match-type');
  if (matchCtx && cacheHits > 0) {
    dashCharts.matchType = new Chart(matchCtx, {
      type: 'doughnut',
      data: {
        labels: ['精确匹配', '语义匹配'],
        datasets: [{
          data: [exactHits, semanticHits],
          backgroundColor: ['#06b6d4', '#8b5cf6'],
          borderColor: ['#0891b2', '#7c3aed'],
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom', labels: { padding: 16, font: { size: 11 } } } }
      }
    });
  }

  // Token 消耗趋势折线图
  var tokenCtx = document.getElementById('chart-token-trend');
  if (tokenCtx) {
    var recentRequests = evalData.recent_requests || [];
    var labels = [];
    var inputData = [];
    var outputData = [];
    var totalData = [];

    recentRequests.forEach(function(r) {
      var time = r.timestamp ? r.timestamp.substring(11, 19) : '';
      labels.push(time);
      inputData.push(r.input_tokens || 0);
      outputData.push(r.output_tokens || 0);
      totalData.push(r.total_tokens || 0);
    });

    dashCharts.tokenTrend = new Chart(tokenCtx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: '输入 Token',
            data: inputData,
            borderColor: '#5b5fe3',
            backgroundColor: 'rgba(91,95,227,0.08)',
            fill: true,
            tension: 0.3,
            pointRadius: 2
          },
          {
            label: '输出 Token',
            data: outputData,
            borderColor: '#22c55e',
            backgroundColor: 'rgba(34,197,94,0.08)',
            fill: true,
            tension: 0.3,
            pointRadius: 2
          },
          {
            label: '总 Token',
            data: totalData,
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245,158,11,0.08)',
            fill: true,
            tension: 0.3,
            pointRadius: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom', labels: { padding: 16, font: { size: 11 } } } },
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              callback: function(value) {
                return value >= 1000 ? (value / 1000) + 'k' : value;
              }
            }
          }
        }
      }
    });
  }
}

async function resetEvaluation() {
  if (!confirm('确定要重置评估统计吗？')) return;
  try {
    const res = await apiFetch('/api/evaluation/reset', { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      showToast('评估统计已重置', 'success');
      loadDashboard();
    } else {
      showToast('重置失败', 'error');
    }
  } catch (e) {
    showToast('请求失败：' + e.message, 'error');
  }
}

async function triggerWarmup() {
  if (!confirm('将使用默认 FAQ 列表进行缓存预热，确定继续？')) return;
  try {
    const res = await apiFetch('/api/cache/warmup', {
      method: 'POST',
      body: JSON.stringify({
        entries: [
          { question: '什么是RAG？', answer: 'RAG（检索增强生成）是一种将信息检索与大语言模型生成相结合的技术架构。' },
          { question: '什么是语义缓存？', answer: '语义缓存通过向量相似度匹配历史问答，直接返回缓存结果。' },
          { question: '混合检索是什么？', answer: '混合检索结合语义搜索和关键词搜索，通过RRF融合结果。' }
        ]
      })
    });
    const data = await res.json();
    if (data.success) {
      showToast(data.message, 'success');
      loadDashboard();
    } else {
      showToast(data.message || '预热失败', 'error');
    }
  } catch (e) {
    showToast('请求失败：' + e.message, 'error');
  }
}

async function clearCacheFromDash() {
  if (!confirm('确认清空语义缓存？')) return;
  try {
    const res = await apiFetch('/api/cache/clear', { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      showToast('缓存已清空', 'success');
      loadDashboard();
    } else {
      showToast('清空失败', 'error');
    }
  } catch (e) {
    showToast('请求失败：' + e.message, 'error');
  }
}

// ============================================================
// 方向1：千人千面 -- 个性化适配引擎
// ============================================================

function changeAnswerStyle() {
  var select = document.getElementById('answer-style-select');
  if (select) {
    currentAnswerStyle = select.value;
    showToast('回答风格已切换为：' + select.options[select.selectedIndex].text, 'info');
  }
}

async function loadUserProfile() {
  if (!currentUser) return;
  try {
    var res = await apiFetch('/api/profile/' + encodeURIComponent(currentUser.username));
    var data = await res.json();
    if (data.error) return;

    var styleSelect = document.getElementById('answer-style-select');
    if (styleSelect && data.style_preference) {
      styleSelect.value = data.style_preference;
      currentAnswerStyle = data.style_preference;
    }
  } catch (_) {}
}

async function addBookmark(question, answer, sources, btnEl) {
  if (!currentUser) {
    showToast('请先登录后再收藏', 'warning');
    return;
  }
  try {
    var res = await apiFetch('/api/bookmarks/' + encodeURIComponent(currentUser.username), {
      method: 'POST',
      body: JSON.stringify({
        question: question,
        answer: answer,
        sources: sources || null,
        note: ''
      })
    });
    var data = await res.json();
    if (data.success) {
      if (btnEl) {
        if (data.action === 'deleted') {
          btnEl.classList.remove('active');
          btnEl.style.opacity = '';
          showToast('已取消收藏', 'info');
        } else {
          btnEl.classList.add('active');
          btnEl.style.opacity = '1';
          showToast('已添加到个人知识空间', 'success');
        }
      }
    } else {
      showToast(data.error || '收藏失败', 'error');
    }
  } catch (e) {
    showToast('收藏失败：' + e.message, 'error');
  }
}

async function loadBookmarks() {
  if (!currentUser) return;
  try {
    var res = await apiFetch('/api/bookmarks/' + encodeURIComponent(currentUser.username));
    var data = await res.json();
    var bookmarks = data.bookmarks || [];
    renderBookmarks(bookmarks);
  } catch (_) {}
}

function renderBookmarks(bookmarks) {
  var container = document.getElementById('bookmarks-list');
  if (!container) return;

  if (!bookmarks || bookmarks.length === 0) {
    container.innerHTML = '<div class="bookmarks-empty">暂无收藏内容</div>';
    return;
  }

  var html = '';
  bookmarks.forEach(function(b) {
    html += '<div class="bookmark-item" data-question="' + escapeHtml(b.question || '') + '" data-answer="' + escapeHtml(b.answer || '') + '">';
    html += '<div class="bookmark-question">' + escapeHtml(b.question || '') + '</div>';
    html += '<div class="bookmark-answer">' + escapeHtml((b.answer || '').substring(0, 200)) + '...</div>';
    html += '<div class="bookmark-time">' + escapeHtml(b.created_at || '') + '</div>';
    html += '<button class="btn-sm btn-sm-danger" onclick="removeBookmark(' + b.id + ', this)">删除</button>';
    html += '</div>';
  });
  container.innerHTML = html;
}

async function removeBookmark(bookmarkId, btnEl) {
  if (!currentUser) return;

  // 获取收藏项的问答内容，用于同步取消对话中的高亮
  var question = '';
  var answer = '';
  if (btnEl && btnEl.closest) {
    var itemEl = btnEl.closest('.bookmark-item');
    if (itemEl) {
      question = itemEl.getAttribute('data-question') || '';
      answer = itemEl.getAttribute('data-answer') || '';
    }
  }

  try {
    var res = await apiFetch('/api/bookmarks/' + encodeURIComponent(currentUser.username) + '/' + bookmarkId, {
      method: 'DELETE'
    });
    var data = await res.json();
    if (data.success) {
      showToast('已删除收藏', 'success');

      // 同步取消对话中对应消息的收藏高亮
      if (question && answer) {
        var allBookmarkBtns = document.querySelectorAll('.msg-bookmark-btn');
        allBookmarkBtns.forEach(function(btn) {
          var contentEl = btn.closest('.message-content');
          if (contentEl) {
            var btnQ = contentEl.getAttribute('data-question') || '';
            var btnA = contentEl.getAttribute('data-answer') || '';
            if (btnQ === question && btnA === answer) {
              btn.classList.remove('active');
              btn.style.opacity = '';
            }
          }
        });
      }

      loadBookmarks();
    } else {
      showToast(data.error || '删除失败', 'error');
    }
  } catch (e) {
    showToast('删除失败：' + e.message, 'error');
  }
}

// ============================================================
// 方向2：知其所以然 -- 思维链与溯源可视化
// ============================================================

function renderConfidenceInline(confidence, aiMsgId) {
  var msgEl = document.getElementById(aiMsgId);
  if (!msgEl) return;

  var contentEl = msgEl.querySelector('.message-content');
  if (!contentEl) return;

  var oldConfidence = contentEl.querySelector('.message-confidence');
  if (oldConfidence) oldConfidence.remove();

  var score = confidence.overall_score || 0;
  var level = confidence.level || 'medium';
  var levelText = level === 'high' ? '高可信' : level === 'medium' ? '中等可信' : '低可信';
  var levelColor = level === 'high' ? '#22c55e' : level === 'medium' ? '#f59e0b' : '#ef4444';

  var dims = confidence.dimensions || {};
  var dimsHtml = '';
  var dimLabels = {
    'source_match': '来源匹配',
    'authority': '权威性',
    'consistency': '一致性',
    'freshness': '时效性',
    'completeness': '完整性'
  };
  Object.keys(dimLabels).forEach(function(key) {
    var dim = dims[key];
    if (!dim) return;
    var dimScore = dim.score || 0;
    var dimColor = dimScore >= 80 ? '#22c55e' : dimScore >= 60 ? '#f59e0b' : '#ef4444';
    dimsHtml += '<div class="confidence-dim">'
      + '<span class="confidence-dim-label">' + dimLabels[key] + '</span>'
      + '<div class="confidence-dim-bar"><div class="confidence-dim-fill" style="width:' + dimScore + '%;background:' + dimColor + '"></div></div>'
      + '<span class="confidence-dim-score" style="color:' + dimColor + '">' + dimScore + '</span>'
      + '</div>';
  });

  var provenanceHtml = '';
  if (confidence.provenance_tree) {
    provenanceHtml = '<div class="provenance-toggle" onclick="toggleProvenance(this)">'
      + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">'
      + '<path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>'
      + '<polyline points="14 2 14 8 20 8"/></svg>'
      + '<span>查看溯源树</span>'
      + '<span class="provenance-chevron">▸</span>'
      + '</div>'
      + '<div class="provenance-tree" style="display:none">'
      + renderProvenanceTree(confidence.provenance_tree)
      + '</div>';
  }

  var section = document.createElement('div');
  section.className = 'message-confidence';
  section.innerHTML = '<div class="confidence-header">'
    + '<div class="confidence-score-circle" style="border-color:' + levelColor + ';color:' + levelColor + '">'
    + score + '</div>'
    + '<div class="confidence-info">'
    + '<div class="confidence-level" style="color:' + levelColor + '">' + levelText + '</div>'
    + '<div class="confidence-label">答案可信度评估</div>'
    + '</div>'
    + '</div>'
    + '<div class="confidence-dims">' + dimsHtml + '</div>'
    + provenanceHtml;

  var textEl = contentEl.querySelector('.message-text');
  if (textEl && textEl.nextSibling) {
    contentEl.insertBefore(section, textEl.nextSibling);
  } else {
    contentEl.appendChild(section);
  }
}

function renderProvenanceTree(tree) {
  if (!tree || !tree.stages || tree.stages.length === 0) return '';

  var html = '<div class="provenance-tree-root">';

  tree.stages.forEach(function(stage) {
    var stageIcon = '';
    if (stage.type === 'retrieval') stageIcon = '检索';
    else if (stage.type === 'rerank') stageIcon = '精排';
    else if (stage.type === 'generation') stageIcon = '生成';
    else stageIcon = stage.type || '';

    html += '<div class="provenance-stage">';
    html += '<div class="provenance-stage-header">';
    html += '<span class="provenance-stage-icon">' + escapeHtml(stageIcon) + '</span>';
    html += '<span class="provenance-stage-name">' + escapeHtml(stage.name || '') + '</span>';
    if (stage.details) {
      html += '<span class="provenance-stage-detail">' + escapeHtml(stage.details.method || '') + '，候选 ' + (stage.details.candidate_count || 0) + ' 条</span>';
    }
    html += '</div>';

    if (stage.items && stage.items.length > 0) {
      html += '<div class="provenance-stage-items">';
      stage.items.forEach(function(item) {
        if (stage.type === 'generation') {
          html += '<div class="provenance-item generation">' + escapeHtml(item.description || '') + '</div>';
        } else {
          html += '<div class="provenance-item">';
          html += '<span class="provenance-item-index">#' + (item.index || '') + '</span>';
          html += '<span class="provenance-item-source">' + escapeHtml(item.source || '未知') + '</span>';
          if (item.similarity) {
            html += '<span class="provenance-item-sim">' + escapeHtml(String(item.similarity)) + '</span>';
          }
          if (item.retrieval_type) {
            html += '<span class="provenance-item-type">' + escapeHtml(item.retrieval_type) + '</span>';
          }
          if (item.preview) {
            html += '<div class="provenance-item-preview">' + escapeHtml(item.preview.substring(0, 100)) + '</div>';
          }
          html += '</div>';
        }
      });
      html += '</div>';
    }

    html += '</div>';
  });

  html += '</div>';
  return html;
}

function toggleProvenance(toggleEl) {
  var section = toggleEl.parentElement;
  var tree = section.querySelector('.provenance-tree');
  var chevron = section.querySelector('.provenance-chevron');
  if (!tree || !chevron) return;

  if (tree.style.display === 'none') {
    tree.style.display = '';
    chevron.textContent = '▾';
  } else {
    tree.style.display = 'none';
    chevron.textContent = '▸';
  }
}

// ============================================================
// 方向3：不只是问答 -- 知识探索与创作工具
// ============================================================

async function buildKnowledgeGraph() {
  var loadingEl = document.getElementById('kg-loading');
  var containerEl = document.getElementById('kg-container');

  if (loadingEl) loadingEl.style.display = '';
  if (containerEl) containerEl.style.display = 'none';

  try {
    var res = await apiFetch('/api/knowledge-graph/build', { method: 'POST' });
    var data = await res.json();

    if (data.success) {
      renderKnowledgeGraph(data.graph);
    } else {
      showToast(data.error || '构建知识图谱失败', 'error');
    }
  } catch (e) {
    showToast('构建知识图谱失败：' + e.message, 'error');
  } finally {
    if (loadingEl) loadingEl.style.display = 'none';
  }
}

function renderKnowledgeGraph(graph) {
  if (typeof d3 === 'undefined') {
    showToast('D3.js 库加载失败，请检查网络连接后刷新页面重试', 'error');
    return;
  }

  var containerEl = document.getElementById('kg-container');
  var statsEl = document.getElementById('kg-stats');
  var canvasEl = document.getElementById('kg-canvas');

  if (!containerEl) return;
  containerEl.style.display = '';

  if (statsEl && graph) {
    statsEl.innerHTML = '节点数：<strong>' + (graph.node_count || 0) + '</strong> · 关系数：<strong>' + (graph.edge_count || 0) + '</strong>';
  }

  if (!canvasEl || !graph || !graph.nodes || graph.nodes.length === 0) {
    if (canvasEl) canvasEl.innerHTML = '<div class="kg-empty">暂无知识图谱数据，请先入库文档</div>';
    return;
  }

  canvasEl.innerHTML = '';

  var nodes = graph.nodes || [];
  var edges = graph.edges || [];

  var width = canvasEl.clientWidth || 800;
  var height = canvasEl.clientHeight || 500;

  // 节点颜色映射
  var categoryColors = {
    'policy': '#f59e0b',
    'process': '#22c55e',
    'action': '#ef4444',
    'concept': '#06b6d4'
  };
  var defaultColor = '#5b5fe3';

  // 转换数据格式
  var nodeData = nodes.map(function(n) {
    return {
      id: String(n.id),
      name: n.name || n.label || String(n.id),
      category: n.category || n.type || '',
      color: categoryColors[n.category || n.type] || defaultColor,
      originalId: n.id
    };
  });

  var edgeData = edges.map(function(e) {
    return {
      source: String(e.source),
      target: String(e.target),
      relation: e.relation || e.label || ''
    };
  });

  // 创建 SVG
  var svg = d3.select('#kg-canvas')
    .append('svg')
    .attr('width', width)
    .attr('height', height)
    .attr('viewBox', [0, 0, width, height])
    .style('background', 'var(--bg-primary)');

  // 缩放行为
  var g = svg.append('g');

  var zoom = d3.zoom()
    .scaleExtent([0.1, 4])
    .on('zoom', function(event) {
      g.attr('transform', event.transform);
    });

  svg.call(zoom);

  // 力导向模拟
  var simulation = d3.forceSimulation(nodeData)
    .force('link', d3.forceLink(edgeData).id(function(d) { return d.id; }).distance(120))
    .force('charge', d3.forceManyBody().strength(-400))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(40));

  // 绘制边
  var link = g.append('g')
    .attr('class', 'kg-links')
    .selectAll('line')
    .data(edgeData)
    .join('line')
    .attr('stroke', 'var(--border)')
    .attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '4,2');

  // 边标签
  var linkLabel = g.append('g')
    .attr('class', 'kg-link-labels')
    .selectAll('text')
    .data(edgeData)
    .join('text')
    .text(function(d) { return d.relation; })
    .attr('font-size', '10px')
    .attr('fill', 'var(--text-tertiary)')
    .attr('text-anchor', 'middle')
    .attr('dy', '-4');

  // 绘制节点组
  var node = g.append('g')
    .attr('class', 'kg-nodes')
    .selectAll('g')
    .data(nodeData)
    .join('g')
    .call(d3.drag()
      .on('start', function(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', function(event, d) {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on('end', function(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      })
    );

  // 节点圆形
  node.append('circle')
    .attr('r', 22)
    .attr('fill', function(d) { return d.color; })
    .attr('opacity', 0.9)
    .attr('stroke', 'var(--bg-primary)')
    .attr('stroke-width', 2);

  // 节点文字
  node.append('text')
    .text(function(d) { return d.name.substring(0, 4); })
    .attr('text-anchor', 'middle')
    .attr('dy', '4')
    .attr('font-size', '10px')
    .attr('fill', 'white')
    .attr('font-weight', '600')
    .style('pointer-events', 'none');

  // 节点标签
  node.append('text')
    .text(function(d) { return d.name; })
    .attr('text-anchor', 'middle')
    .attr('dy', '36')
    .attr('font-size', '11px')
    .attr('fill', 'var(--text-primary)')
    .style('pointer-events', 'none');

  // 节点悬停效果
  node.on('mouseenter', function(event, d) {
    d3.select(this).select('circle')
      .transition().duration(200)
      .attr('r', 28)
      .attr('opacity', 1);
  }).on('mouseleave', function(event, d) {
    d3.select(this).select('circle')
      .transition().duration(200)
      .attr('r', 22)
      .attr('opacity', 0.9);
  });

  // 模拟更新
  simulation.on('tick', function() {
    link
      .attr('x1', function(d) { return d.source.x; })
      .attr('y1', function(d) { return d.source.y; })
      .attr('x2', function(d) { return d.target.x; })
      .attr('y2', function(d) { return d.target.y; });

    linkLabel
      .attr('x', function(d) { return (d.source.x + d.target.x) / 2; })
      .attr('y', function(d) { return (d.source.y + d.target.y) / 2; });

    node.attr('transform', function(d) { return 'translate(' + d.x + ',' + d.y + ')'; });
  });

  // 初始缩放适配
  var initialScale = 0.8;
  svg.call(zoom.transform, d3.zoomIdentity.translate(width * 0.1, height * 0.1).scale(initialScale));
}

async function doCompareQA() {
  var topicA = document.getElementById('compare-topic-a').value.trim();
  var topicB = document.getElementById('compare-topic-b').value.trim();

  if (!topicA || !topicB) {
    showToast('请输入两个对比主题', 'warning');
    return;
  }

  var loadingEl = document.getElementById('compare-loading');
  var resultEl = document.getElementById('compare-result');

  if (loadingEl) loadingEl.style.display = '';
  if (resultEl) resultEl.style.display = 'none';

  try {
    var res = await apiFetch('/api/compare', {
      method: 'POST',
      body: JSON.stringify({ topic_a: topicA, topic_b: topicB })
    });
    var data = await res.json();

    if (resultEl) {
      resultEl.style.display = '';
      if (data.error) {
        resultEl.innerHTML = '<div class="result-error">' + escapeHtml(data.error) + '</div>';
      } else {
        resultEl.innerHTML = '<div class="compare-summary"><h4>对比分析：' + escapeHtml(topicA) + ' vs ' + escapeHtml(topicB) + '</h4><div class="compare-text">' + formatAnswerText(data.answer || '无结果') + '</div></div>';
      }
    }
  } catch (e) {
    if (resultEl) {
      resultEl.style.display = '';
      resultEl.innerHTML = '<div class="result-error">对比分析失败：' + escapeHtml(e.message) + '</div>';
    }
  } finally {
    if (loadingEl) loadingEl.style.display = 'none';
  }
}

async function doSimulateQA() {
  var scenario = document.getElementById('simulate-scenario').value.trim();

  if (!scenario) {
    showToast('请输入模拟场景', 'warning');
    return;
  }

  var loadingEl = document.getElementById('simulate-loading');
  var resultEl = document.getElementById('simulate-result');

  if (loadingEl) loadingEl.style.display = '';
  if (resultEl) resultEl.style.display = 'none';

  try {
    var res = await apiFetch('/api/simulate', {
      method: 'POST',
      body: JSON.stringify({ scenario: scenario })
    });
    var data = await res.json();

    if (resultEl) {
      resultEl.style.display = '';
      if (data.error) {
        resultEl.innerHTML = '<div class="result-error">' + escapeHtml(data.error) + '</div>';
      } else {
        resultEl.innerHTML = '<div class="simulate-result-content">'
          + '<div class="simulate-scenario-label">场景：' + escapeHtml(scenario) + '</div>'
          + '<div class="simulate-text">' + formatAnswerText(data.answer || '无结果') + '</div>'
          + '</div>';
      }
    }
  } catch (e) {
    if (resultEl) {
      resultEl.style.display = '';
      resultEl.innerHTML = '<div class="result-error">模拟分析失败：' + escapeHtml(e.message) + '</div>';
    }
  } finally {
    if (loadingEl) loadingEl.style.display = 'none';
  }
}

async function doSummarize() {
  var content = document.getElementById('summarize-content').value.trim();
  var level = document.getElementById('summarize-level').value;

  if (!content) {
    showToast('请输入需要摘要的文档内容', 'warning');
    return;
  }

  var loadingEl = document.getElementById('summarize-loading');
  var resultEl = document.getElementById('summarize-result');

  if (loadingEl) loadingEl.style.display = '';
  if (resultEl) resultEl.style.display = 'none';

  try {
    var res = await apiFetch('/api/summarize', {
      method: 'POST',
      body: JSON.stringify({ content: content, level: level })
    });
    var data = await res.json();

    if (resultEl) {
      resultEl.style.display = '';
      if (data.error) {
        resultEl.innerHTML = '<div class="result-error">' + escapeHtml(data.error) + '</div>';
      } else {
        var levelLabels = {
          'one_line': '一句话总结',
          'paragraph': '段落摘要',
          'structured': '结构化摘要',
          'bullets': '关键要点列表',
          'actions': '行动项提取'
        };
        resultEl.innerHTML = '<div class="summarize-result-content">'
          + '<div class="summarize-level-label">摘要类型：' + (levelLabels[level] || level) + '</div>'
          + '<div class="summarize-text">' + formatAnswerText(data.summary || '无结果') + '</div>'
          + '</div>';
      }
    }
  } catch (e) {
    if (resultEl) {
      resultEl.style.display = '';
      resultEl.innerHTML = '<div class="result-error">摘要生成失败：' + escapeHtml(e.message) + '</div>';
    }
  } finally {
    if (loadingEl) loadingEl.style.display = 'none';
  }
}

// ============================================================
// 方向4：群体智慧 -- 协作式知识生态
// ============================================================

async function loadFeedbackStats() {
  try {
    var res = await apiFetch('/api/feedback/stats');
    var data = await res.json();

    var totalEl = document.getElementById('fb-total');
    if (totalEl) totalEl.textContent = data.total_feedback || 0;

    var positiveEl = document.getElementById('fb-positive');
    if (positiveEl) positiveEl.textContent = data.positive || 0;

    var negativeEl = document.getElementById('fb-negative');
    if (negativeEl) negativeEl.textContent = data.negative || 0;

    var satisfactionEl = document.getElementById('fb-satisfaction');
    if (satisfactionEl) {
      satisfactionEl.textContent = (data.satisfaction_rate || 0) + '%';
    }
  } catch (_) {}
}

async function loadFeedbackList() {
  try {
    var res = await apiFetch('/api/feedback?limit=20');
    var data = await res.json();
    var feedbacks = data.feedback || [];

    var listEl = document.getElementById('feedback-list');
    if (!listEl) return;

    if (feedbacks.length === 0) {
      listEl.innerHTML = '<div class="feedback-empty">暂无反馈记录</div>';
      return;
    }

    var html = '';
    feedbacks.forEach(function(fb) {
      var ratingIcon = fb.rating === 'positive' ? '👍' : '👎';
      var ratingClass = fb.rating === 'positive' ? 'positive' : 'negative';
      html += '<div class="feedback-item ' + ratingClass + '">';
      html += '<div class="feedback-rating">' + ratingIcon + '</div>';
      html += '<div class="feedback-content">';
      html += '<div class="feedback-question">' + escapeHtml((fb.question || '').substring(0, 100)) + '</div>';
      if (fb.comment) {
        html += '<div class="feedback-comment">' + escapeHtml(fb.comment) + '</div>';
      }
      html += '<div class="feedback-time">' + escapeHtml(fb.created_at || '') + '</div>';
      html += '</div></div>';
    });
    listEl.innerHTML = html;
  } catch (_) {}
}

async function submitFeedback(rating, comment, btnEl) {
  var q = lastQuestion;
  var a = lastAnswer;
  if (btnEl && btnEl.closest) {
    var contentEl = btnEl.closest('.message-content');
    if (contentEl) {
      q = contentEl.getAttribute('data-question') || lastQuestion;
      a = contentEl.getAttribute('data-answer') || lastAnswer;
    }
  }
  if (!q || !a) {
    showToast('暂无问答内容可反馈', 'warning');
    return;
  }
  try {
    var res = await apiFetch('/api/feedback', {
      method: 'POST',
      body: JSON.stringify({
        question: q,
        answer: a,
        rating: rating,
        comment: comment || '',
        username: currentUser ? currentUser.username : 'anonymous'
      })
    });
    var data = await res.json();
    if (data.success) {
      if (btnEl) {
        var actionsDiv = btnEl.parentElement;
        if (actionsDiv) {
          var allBtns = actionsDiv.querySelectorAll('.msg-action-btn');
          if (data.action === 'deleted') {
            // 取消反馈：移除所有高亮，恢复按钮
            allBtns.forEach(function(b) {
              b.classList.remove('active');
              b.disabled = false;
              b.style.opacity = '';
              b.style.cursor = '';
            });
            showToast('已取消反馈', 'info');
          } else {
            // 创建或更新反馈：高亮当前按钮，禁用相反按钮
            var oppositeRating = rating === 'positive' ? 'negative' : 'positive';
            allBtns.forEach(function(b) {
              b.classList.remove('active');
              b.disabled = false;
              b.style.opacity = '';
              b.style.cursor = '';
            });
            btnEl.classList.add('active');
            // 禁用相反评分的按钮（点赞和点踩互斥）
            var oppositeBtn = actionsDiv.querySelector('[data-rating="' + oppositeRating + '"]');
            if (oppositeBtn) {
              oppositeBtn.disabled = true;
              oppositeBtn.style.opacity = '0.4';
              oppositeBtn.style.cursor = 'default';
            }
            if (data.action === 'updated') {
              showToast('反馈已更新', 'success');
            } else {
              showToast('感谢您的反馈！', 'success');
            }
          }
        }
      }
    } else {
      showToast(data.error || '反馈提交失败', 'error');
    }
  } catch (e) {
    showToast('反馈提交失败：' + e.message, 'error');
  }
}

async function loadExpertRouting() {
  try {
    var statsRes = await apiFetch('/api/expert-routing/stats');
    var statsData = await statsRes.json();

    var pendingEl = document.getElementById('er-pending');
    if (pendingEl) pendingEl.textContent = statsData.pending || 0;

    var resolvedEl = document.getElementById('er-resolved');
    if (resolvedEl) resolvedEl.textContent = statsData.resolved || 0;

    var rateEl = document.getElementById('er-rate');
    if (rateEl) {
      rateEl.textContent = (statsData.resolution_rate || 0) + '%';
    }

    var questionsRes = await apiFetch('/api/expert-routing/pending');
    var questionsData = await questionsRes.json();

    var questions = questionsData.questions || [];
    var questionsEl = document.getElementById('expert-questions');
    if (!questionsEl) return;

    if (questions.length === 0) {
      questionsEl.innerHTML = '<div class="expert-empty">暂无待处理问题</div>';
      return;
    }

    var html = '';
    questions.forEach(function(q) {
      var statusClass = q.status === 'resolved' ? 'resolved' : 'pending';
      var statusText = q.status === 'resolved' ? '已解决' : '待处理';
      html += '<div class="expert-question-item ' + statusClass + '">';
      html += '<div class="expert-question-text">' + escapeHtml((q.question || '').substring(0, 120)) + '</div>';
      html += '<div class="expert-question-meta">';
      html += '<span class="expert-status">' + statusText + '</span>';
      if (q.assigned_expert) {
        html += '<span class="expert-name">分配给：' + escapeHtml(q.assigned_expert) + '</span>';
      }
      html += '<span class="expert-confidence">置信度：' + (q.confidence_score || 0) + '%</span>';
      html += '</div></div>';
    });
    questionsEl.innerHTML = html;
  } catch (_) {}
}

// ============================================================
// 方向5：AI 原生体验 -- 重新定义交互
// ============================================================

function renderMessageActions(aiMsgId, question, answer, initialRating, initialBookmarked) {
  var msgEl = document.getElementById(aiMsgId);
  if (!msgEl) return;

  var contentEl = msgEl.querySelector('.message-content');
  if (!contentEl) return;

  var oldActions = contentEl.querySelector('.message-actions');
  if (oldActions) oldActions.remove();

  var q = question || lastQuestion || '';
  var a = answer || lastAnswer || '';
  contentEl.setAttribute('data-question', q);
  contentEl.setAttribute('data-answer', a);

  var actionsHtml = '<div class="message-actions">';

  actionsHtml += '<button class="msg-action-btn" data-rating="positive" onclick="submitFeedback(\'positive\', \'\', this)" title="点赞">'
    + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">'
    + '<path d="M14 9V5a3 3 0 00-3-3l-4 9v11h11.28a2 2 0 002-1.7l1.38-9a2 2 0 00-2-2.3H14z"/>'
    + '</svg></button>';

  actionsHtml += '<button class="msg-action-btn" data-rating="negative" onclick="submitFeedback(\'negative\', \'\', this)" title="点踩">'
    + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">'
    + '<path d="M10 15v4a3 3 0 003 3l4-9V2H5.72a2 2 0 00-2 1.7l-1.38 9a2 2 0 002 2.3H10z"/>'
    + '</svg></button>';

  actionsHtml += '<button class="msg-action-btn msg-bookmark-btn" onclick="handleBookmark(this)" title="收藏">'
    + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">'
    + '<path d="M19 21l-7-5-7 5V5a2 2 0 012-2h10a2 2 0 012 2z"/>'
    + '</svg></button>';

  actionsHtml += '<button class="msg-action-btn" onclick="handleExportAnswer(this)" title="导出答案">'
    + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">'
    + '<path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>'
    + '<polyline points="7 10 12 15 17 10"/>'
    + '<line x1="12" y1="15" x2="12" y2="3"/>'
    + '</svg></button>';

  actionsHtml += '</div>';

  var actionsDiv = document.createElement('div');
  actionsDiv.innerHTML = actionsHtml;
  var actionsEl = actionsDiv.firstElementChild;
  contentEl.appendChild(actionsEl);

  // 恢复初始反馈状态
  if (initialRating) {
    var targetBtn = actionsEl.querySelector('[data-rating="' + initialRating + '"]');
    if (targetBtn) {
      targetBtn.classList.add('active');
      var oppositeRating = initialRating === 'positive' ? 'negative' : 'positive';
      var oppositeBtn = actionsEl.querySelector('[data-rating="' + oppositeRating + '"]');
      if (oppositeBtn) {
        oppositeBtn.disabled = true;
        oppositeBtn.style.opacity = '0.4';
        oppositeBtn.style.cursor = 'default';
      }
    }
  }

  // 恢复初始收藏状态
  if (initialBookmarked) {
    var bookmarkBtn = actionsEl.querySelector('.msg-bookmark-btn');
    if (bookmarkBtn) {
      bookmarkBtn.classList.add('active');
      bookmarkBtn.style.opacity = '1';
    }
  }
}

function getMsgContentData(btnEl) {
  var contentEl = btnEl.closest('.message-content');
  if (!contentEl) return { question: lastQuestion || '', answer: lastAnswer || '' };
  return {
    question: contentEl.getAttribute('data-question') || lastQuestion || '',
    answer: contentEl.getAttribute('data-answer') || lastAnswer || ''
  };
}

function handleBookmark(btnEl) {
  var data = getMsgContentData(btnEl);
  addBookmark(data.question, data.answer, currentSources, btnEl);
}

function handleExportAnswer(btnEl) {
  var data = getMsgContentData(btnEl);
  exportAnswerData(data.question, data.answer);
}

function exportAnswerData(question, answer) {
  if (!question || !answer) {
    showToast('暂无问答内容可导出', 'warning');
    return;
  }

  var content = '问题：' + question + '\n\n';
  content += '回答：\n' + answer + '\n\n';

  if (currentConfidenceData) {
    content += '---\n';
    content += '可信度评估：' + (currentConfidenceData.overall_score || 0) + '分\n';
    var dims = currentConfidenceData.dimensions || {};
    Object.keys(dims).forEach(function(key) {
      content += '  - ' + key + '：' + (dims[key].score || 0) + '分\n';
    });
  }

  if (currentSources && currentSources.length > 0) {
    content += '\n---\n参考来源：\n';
    currentSources.forEach(function(s, i) {
      content += (i + 1) + '. ' + (s.source || '未知') + ' (相似度：' + (s.similarity || '—') + ')\n';
    });
  }

  var blob = new Blob(['\uFEFF' + content], { type: 'text/plain;charset=utf-8' });
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url;
  a.download = 'rag_answer_' + new Date().toISOString().slice(0, 10) + '.txt';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast('答案已导出', 'success');
}

function toggleVoiceInput() {
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    showToast('您的浏览器不支持语音输入功能', 'warning');
    return;
  }

  var voiceBtn = document.getElementById('voice-btn');

  // 如果已有正在进行的识别，先中止它
  if (currentRecognition) {
    currentRecognitionAborted = true;
    currentRecognition.abort();
    currentRecognition = null;
    if (voiceBtn) voiceBtn.classList.remove('recording');
    return;
  }

  var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  var recognition = new SpeechRecognition();
  recognition.lang = 'zh-CN';
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  currentRecognition = recognition;
  currentRecognitionAborted = false;

  if (voiceBtn) voiceBtn.classList.add('recording');

  try {
    recognition.start();
  } catch (e) {
    currentRecognition = null;
    if (voiceBtn) voiceBtn.classList.remove('recording');
    showToast('语音识别启动失败：' + e.message, 'error');
    return;
  }

  recognition.onresult = function(event) {
    var transcript = event.results[0][0].transcript;
    var input = document.getElementById('question-input');
    if (input) {
      input.value = transcript;
      autoResizeTextarea(input);
    }
    showToast('语音识别完成', 'success');
  };

  recognition.onerror = function(event) {
    // aborted 是用户主动点击取消，不显示错误提示
    if (event.error === 'aborted' && currentRecognitionAborted) {
      return;
    }
    showToast('语音识别失败：' + event.error, 'error');
  };

  recognition.onend = function() {
    currentRecognition = null;
    currentRecognitionAborted = false;
    if (voiceBtn) voiceBtn.classList.remove('recording');
  };
}

async function doBatchQA() {
  var questionsText = prompt('请输入批量问题（每行一个问题）：');
  if (!questionsText || !questionsText.trim()) return;

  var questions = questionsText.split('\n').filter(function(q) { return q.trim(); });
  if (questions.length === 0) {
    showToast('未检测到有效问题', 'warning');
    return;
  }

  if (questions.length > 10) {
    showToast('批量问答最多支持10个问题', 'warning');
    return;
  }

  showToast('正在处理 ' + questions.length + ' 个问题...', 'info');

  try {
    var res = await apiFetch('/api/batch-qa', {
      method: 'POST',
      body: JSON.stringify({ questions: questions })
    });
    var data = await res.json();

    if (data.error) {
      showToast(data.error, 'error');
      return;
    }

    var results = data.results || [];
    var messagesEl = document.getElementById('messages');
    var welcome = document.querySelector('.chat-welcome');
    if (welcome) welcome.style.display = 'none';

    results.forEach(function(r) {
      appendMessage('user', r.question);
      appendMessage('assistant', r.answer, r.sources);
      conversationHistory.push({ role: 'user', content: r.question });
      conversationHistory.push({ role: 'assistant', content: r.answer });
    });

    updateSessionIndicator();
    showToast('批量问答完成，共处理 ' + results.length + ' 个问题', 'success');
  } catch (e) {
    showToast('批量问答失败：' + e.message, 'error');
  }
}

// ============================================================
// 事件绑定
// ============================================================

document.addEventListener('DOMContentLoaded', function() {
  loadStatus();
  updateSessionIndicator();
  loadHistoryList();
  loadDashboard();

  const input = document.getElementById('question-input');

  input.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  input.addEventListener('input', function() { autoResizeTextarea(input); });

  const hybridToggle = document.getElementById('hybrid-toggle');
  if (hybridToggle) {
    hybridToggle.addEventListener('change', function(e) {
      toggleHybridSearch(e.target.checked);
    });
  }

  const rerankerToggle = document.getElementById('reranker-toggle');
  if (rerankerToggle) {
    rerankerToggle.addEventListener('change', function(e) {
      toggleReranker(e.target.checked);
    });
  }

  const fileInput = document.getElementById('file-input');
  if (fileInput) {
    fileInput.addEventListener('change', function(e) {
      if (e.target.files && e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
        e.target.value = '';
      }
    });
  }
});


// ======================================================================
//  RAGAS 评估相关函数
// ======================================================================

let ragasChart = null;

async function ragasRunPhase1() {
  var btn = document.getElementById('ragas-btn-phase1');
  var status = document.getElementById('ragas-status');
  var limit = parseInt(document.getElementById('ragas-sample-limit').value) || 10;

  btn.disabled = true;
  status.textContent = 'Phase 1 评估中...';
  status.style.color = 'var(--primary)';

  try {
    var res = await apiFetch('/api/evaluation/ragas/phase1', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_limit: limit })
    });
    var data = await res.json();

    if (data.error) {
      status.textContent = data.error;
      status.style.color = 'var(--danger)';
      btn.disabled = false;
      return;
    }

    status.textContent = 'Phase 1 完成 (' + data.sample_count + ' 条样本)';
    status.style.color = '#27ae60';
    renderRagasResults(data, 'Phase 1');

    // 加载趋势
    ragasLoadTrend('phase1');

  } catch (e) {
    status.textContent = '网络错误: ' + e.message;
    status.style.color = 'var(--danger)';
    btn.disabled = false;
  }
}

async function ragasRunPhase2() {
  var btn = document.getElementById('ragas-btn-phase2');
  var status = document.getElementById('ragas-status');
  var limit = parseInt(document.getElementById('ragas-sample-limit').value) || 10;

  btn.disabled = true;
  status.textContent = 'Phase 2 评估中...';
  status.style.color = '#6c5ce7';

  try {
    var res = await apiFetch('/api/evaluation/ragas/phase2', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_limit: limit })
    });
    var data = await res.json();

    if (data.error) {
      status.textContent = data.error;
      status.style.color = 'var(--danger)';
      btn.disabled = false;
      return;
    }

    status.textContent = 'Phase 2 完成 (' + data.sample_count + ' 条样本)';
    status.style.color = '#27ae60';
    renderRagasResults(data, 'Phase 2');

    // 加载趋势
    ragasLoadTrend('phase2');

  } catch (e) {
    status.textContent = '网络错误: ' + e.message;
    status.style.color = 'var(--danger)';
    btn.disabled = false;
  }
}

function renderRagasResults(data, phase) {
  var container = document.getElementById('ragas-results');
  var cardsContainer = document.getElementById('ragas-metrics-cards');
  var phaseLabel = document.getElementById('ragas-phase-label');

  container.style.display = 'block';
  phaseLabel.textContent = ' (' + phase + ', ' + data.sample_count + ' 条样本)';

  var metrics = data.metrics || {};
  var metricDefs = [
    { key: 'faithfulness',          label: '忠实度',          desc: '答案声明是否源自检索上下文' },
    { key: 'answer_relevancy',      label: '答案相关性',      desc: '答案是否与问题相关' },
    { key: 'context_precision',     label: '上下文精确度',    desc: '检索文档信噪比（位置加权）' },
    { key: 'context_relevancy',     label: '上下文相关度',    desc: '检索文档逐句相关句子占比' },
    { key: 'context_recall',        label: '上下文召回率',    desc: 'ground truth 中信息覆盖率' },
    { key: 'context_entity_recall', label: '实体召回率',      desc: 'ground truth 关键实体覆盖度' },
    { key: 'answer_correctness',    label: '答案正确性',      desc: '与 ground truth 的事实准确性' },
    { key: 'answer_similarity',     label: '语义相似度',      desc: '与 ground truth 的语义相似度' },
  ];

  var html = '';
  metricDefs.forEach(function(m) {
    if (metrics[m.key] !== undefined) {
      var pct = (metrics[m.key] * 100).toFixed(1);
      var color = metrics[m.key] >= 0.8 ? '#27ae60' : metrics[m.key] >= 0.6 ? '#f39c12' : '#e74c3c';
      html += '<div class="dash-card">' +
        '<div class="dash-card-value" style="color:' + color + ';">' + pct + '%</div>' +
        '<div class="dash-card-label">' + m.label + '</div>' +
        '<div class="dash-card-desc" style="font-size:11px;color:var(--muted);margin-top:4px;">' + m.desc + '</div>' +
        '</div>';
    }
  });
  cardsContainer.innerHTML = html;

  // 显示趋势面板
  document.getElementById('ragas-trend-panel').style.display = 'block';
}

async function ragasLoadTrend(phase) {
  try {
    var res = await apiFetch('/api/evaluation/ragas/trend?phase=' + phase + '&limit=30');
    var data = await res.json();
    var trend = data.trend || [];

    if (trend.length === 0) return;

    var ctx = document.getElementById('chart-ragas-trend').getContext('2d');

    if (ragasChart) ragasChart.destroy();

    var labels = trend.map(function(r) {
      return r.created_at ? r.created_at.substr(5, 11) : '';
    });
    var metrics = trend[0].metrics || {};
    var metricKeys = Object.keys(metrics);

    var colors = ['#3498db', '#27ae60', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c', '#e67e22'];
    var datasets = metricKeys.map(function(key, i) {
      return {
        label: key.replace('_', ' '),
        data: trend.map(function(r) {
          var m = r.metrics || {};
          return (m[key] || 0) * 100;
        }),
        borderColor: colors[i % colors.length],
        backgroundColor: 'transparent',
        tension: 0.3,
        borderWidth: 2,
      };
    });

    ragasChart = new Chart(ctx, {
      type: 'line',
      data: { labels: labels, datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 10 } } }
        },
        scales: {
          y: { min: 0, max: 100, ticks: { callback: function(v) { return v + '%'; } } }
        }
      }
    });
  } catch (e) {
    console.error('加载 RAGAS 趋势失败:', e);
  }
}

// 加载评估样本列表（用于标注 ground truth）
async function ragasLoadSamples() {
  var container = document.getElementById('ragas-samples-list');
  container.innerHTML = '<p style="color:var(--muted);">加载中...</p>';

  try {
    var res = await apiFetch('/api/evaluation/ragas/samples?limit=10');
    var data = await res.json();
    var samples = data.samples || [];

    if (samples.length === 0) {
      container.innerHTML = '<p style="color:var(--muted);">暂无评估样本，请先进行几次问答</p>';
      return;
    }

    var html = '<table class="dash-table"><thead><tr><th>问题</th><th>答案预览</th><th>上下文数</th><th>操作</th></tr></thead><tbody>';
    samples.forEach(function(s, i) {
      html += '<tr>' +
        '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;">' + escapeHtml(s.question) + '</td>' +
        '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;">' + escapeHtml(s.answer_preview || '') + '</td>' +
        '<td>' + (s.context_count || 0) + '</td>' +
        '<td><button class="btn-sm" onclick="ragasUseSample(' + i + ')">标注</button></td>' +
        '</tr>';
    });
    html += '</tbody></table>';
    container.innerHTML = html;

    // 缓存样本数据
    window._ragasSamples = samples;

  } catch (e) {
    container.innerHTML = '<p style="color:var(--danger);">加载失败: ' + e.message + '</p>';
  }
}

function ragasClearSamples() {
  document.getElementById('ragas-samples-list').innerHTML = '';
  window._ragasSamples = [];
}

function ragasUseSample(idx) {
  var samples = window._ragasSamples || [];
  if (idx >= 0 && idx < samples.length) {
    document.getElementById('ragas-gt-question').value = samples[idx].question || '';
    document.getElementById('ragas-gt-answer').value = samples[idx].answer || '';
  }
}

async function ragasAddGroundTruth() {
  var question = document.getElementById('ragas-gt-question').value.trim();
  var groundTruth = document.getElementById('ragas-gt-answer').value.trim();
  var status = document.getElementById('ragas-status');

  if (!question || !groundTruth) {
    status.textContent = '问题和标准答案均不能为空';
    status.style.color = 'var(--danger)';
    return;
  }

  try {
    var res = await apiFetch('/api/evaluation/ragas/ground-truth', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: question, ground_truth: groundTruth })
    });
    var data = await res.json();

    if (data.success) {
      status.textContent = '添加成功';
      status.style.color = '#27ae60';
      document.getElementById('ragas-gt-question').value = '';
      document.getElementById('ragas-gt-answer').value = '';
      ragasLoadGroundTruths();
    } else {
      status.textContent = '添加失败';
      status.style.color = 'var(--danger)';
    }

  } catch (e) {
    status.textContent = '网络错误: ' + e.message;
    status.style.color = 'var(--danger)';
  }
}

async function ragasLoadGroundTruths() {
  var container = document.getElementById('ragas-gt-list');
  var countEl = document.getElementById('ragas-gt-count');

  try {
    var res = await apiFetch('/api/evaluation/ragas/ground-truth');
    var data = await res.json();
    var entries = data.entries || [];

    countEl.textContent = entries.length;

    if (entries.length === 0) {
      container.innerHTML = '<p style="color:var(--muted);">暂无 ground truth</p>';
      return;
    }

    var html = '<table class="dash-table"><thead><tr><th>问题</th><th>标准答案</th><th>时间</th><th>操作</th></tr></thead><tbody>';
    entries.forEach(function(entry) {
      html += '<tr>' +
        '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;">' + escapeHtml(entry.question) + '</td>' +
        '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;">' + escapeHtml(entry.ground_truth) + '</td>' +
        '<td>' + (entry.created_at || '').substr(0, 16) + '</td>' +
        '<td><button class="btn-sm" style="background:#e74c3c;" onclick="ragasDeleteGroundTruth(' + entry.id + ')">删除</button></td>' +
        '</tr>';
    });
    html += '</tbody></table>';
    container.innerHTML = html;
  } catch (e) {
    container.innerHTML = '<p style="color:var(--danger);">加载失败: ' + e.message + '</p>';
  }
}


// ============================================================
// OpenAPI 管理
// ============================================================

var _newlyCreatedKey = '';

function loadApiKeys() {
  var container = document.getElementById('openapi-keys-list');
  if (!container) return;

  apiFetch('/api/open/keys?include_inactive=1')
  .then(function(r) { return r.json(); })
  .then(function(data) {
    var keys = data.keys || [];
    if (keys.length === 0) {
      container.innerHTML = '<div class="openapi-empty">暂无 API Key，点击上方按钮创建<br><span style="font-size:12px;color:var(--text-muted);margin-top:6px;display:inline-block;">Key 仅在创建时显示一次，如遗忘请吊销后重新创建</span></div>';
      return;
    }

    var html = '';
    keys.forEach(function(key) {
      var statusClass = key.is_active ? 'openapi-status-active' : 'openapi-status-revoked';
      var statusText = key.is_active ? '活跃' : '已吊销';
      var lastUsed = key.last_used_at ? key.last_used_at.substr(0, 16) : '—';

      html += '<div class="openapi-key-card">';
      html += '  <div class="openapi-key-info">';
      html += '    <div class="openapi-key-name">' + escapeHtml(key.name) + ' <span class="openapi-status-badge ' + statusClass + '">' + statusText + '</span></div>';
      if (key.description) {
        html += '    <div class="openapi-key-desc">' + escapeHtml(key.description) + '</div>';
      }
      html += '    <div class="openapi-key-meta">';
      html += '      <span>' + key.rate_limit + ' 次/分</span>';
      html += '      <span>调用 ' + key.usage_count + ' 次</span>';
      html += '      <span>创建于 ' + (key.created_at || '').substr(0, 10) + '</span>';
      if (lastUsed !== '—') {
        html += '      <span>最后使用 ' + lastUsed + '</span>';
      }
      html += '    </div>';
      html += '  </div>';
      html += '  <div class="openapi-key-actions">';
      if (key.is_active) {
        html += '    <button class="btn-sm" onclick="revokeApiKey(' + key.id + ')">吊销</button>';
      } else {
        html += '    <button class="btn-sm" onclick="activateApiKey(' + key.id + ')">激活</button>';
      }
      html += '    <button class="btn-sm btn-sm-danger" onclick="deleteApiKey(' + key.id + ')">删除</button>';
      html += '  </div>';
      html += '</div>';
    });

    container.innerHTML = html;
  })
  .catch(function(err) {
    container.innerHTML = '<div class="openapi-empty" style="color:var(--danger);">加载失败：' + err.message + '</div>';
  });
}

function showCreateKeyDialog() {
  document.getElementById('create-key-name').value = '';
  document.getElementById('create-key-desc').value = '';
  document.getElementById('create-key-rate').value = '60';
  document.getElementById('create-key-error').textContent = '';
  document.getElementById('create-key-modal').style.display = 'flex';
}

function closeCreateKeyDialog() {
  document.getElementById('create-key-modal').style.display = 'none';
}

function doCreateKey() {
  var name = document.getElementById('create-key-name').value.trim();
  var desc = document.getElementById('create-key-desc').value.trim();
  var rate = parseInt(document.getElementById('create-key-rate').value) || 60;
  var errorEl = document.getElementById('create-key-error');

  if (!name) {
    errorEl.textContent = '请输入名称';
    return;
  }

  apiFetch('/api/open/keys', {
    method: 'POST',
    body: JSON.stringify({ name: name, description: desc, rate_limit: rate })
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.error) {
      errorEl.textContent = data.error;
      return;
    }
    closeCreateKeyDialog();
    _newlyCreatedKey = data.api_key || '';
    document.getElementById('show-key-value').textContent = _newlyCreatedKey;
    document.getElementById('show-key-modal').style.display = 'flex';
    loadApiKeys();
  })
  .catch(function(err) {
    errorEl.textContent = '创建失败：' + err.message;
  });
}

function closeShowKeyDialog() {
  document.getElementById('show-key-modal').style.display = 'none';
  _newlyCreatedKey = '';
}

function copyNewKey() {
  if (!_newlyCreatedKey) return;
  navigator.clipboard.writeText(_newlyCreatedKey).then(function() {
    alert('已复制到剪贴板');
  }).catch(function() {
    var ta = document.createElement('textarea');
    ta.value = _newlyCreatedKey;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    alert('已复制到剪贴板');
  });
}

function revokeApiKey(keyId) {
  if (!confirm('确定要吊销此 API Key 吗？')) return;
  apiFetch('/api/open/keys/' + keyId + '/revoke', { method: 'POST' })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.success) loadApiKeys();
    else alert(data.error || '操作失败');
  })
  .catch(function(err) { alert('操作失败：' + err.message); });
}

function activateApiKey(keyId) {
  apiFetch('/api/open/keys/' + keyId + '/activate', { method: 'POST' })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.success) loadApiKeys();
    else alert(data.error || '操作失败');
  })
  .catch(function(err) { alert('操作失败：' + err.message); });
}

function deleteApiKey(keyId) {
  if (!confirm('确定要永久删除此 API Key 吗？此操作不可恢复。')) return;
  apiFetch('/api/open/keys/' + keyId, { method: 'DELETE' })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.success) loadApiKeys();
    else alert(data.error || '操作失败');
  })
  .catch(function(err) { alert('操作失败：' + err.message); });
}

async function ragasDeleteGroundTruth(id) {
  if (!confirm('确定删除此 ground truth？')) return;
  try {
    var res = await apiFetch('/api/evaluation/ragas/ground-truth/' + id, { method: 'DELETE' });
    var data = await res.json();
    if (data.success) {
      ragasLoadGroundTruths();
    }
  } catch (e) {
    console.error('删除失败:', e);
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// 进入仪表盘时自动加载 ground truth 数量
document.addEventListener('DOMContentLoaded', function() {
  // 劫持仪表盘切换事件，进入 ragas tab 时加载数据
  var origTabSwitch = window.switchDashSubtab;
  window.switchDashSubtab = function(tabName) {
    if (origTabSwitch) origTabSwitch(tabName);
    if (tabName === 'ragas') {
      ragasLoadGroundTruths();
    }
  };
});