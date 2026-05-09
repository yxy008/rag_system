/**
 * app.js - 前端交互逻辑
 * 负责：消息发送、SSE 流式接收、对话历史、混合检索开关、来源展示
 */

// ============================================================
// 状态管理
// ============================================================

let isStreaming = false;
let sessionId = localStorage.getItem("rag_session_id") || generateSessionId();
let currentSources = [];  // 当前回答的来源
let conversationHistory = [];  // 本地对话历史（用于显示）

// 生成会话 ID
function generateSessionId() {
  const id = "sess_" + Date.now() + "_" + Math.random().toString(36).slice(2, 8);
  localStorage.setItem("rag_session_id", id);
  return id;
}

// ============================================================
// 系统状态
// ============================================================

async function loadStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();

    document.getElementById('doc-count').textContent =
      data.knowledge_base_ready
        ? `${data.doc_count} 个向量 ✓`
        : '⚠ 知识库为空';

    document.getElementById('embed-model').textContent =
      data.embedding_model || '—';

    document.getElementById('llm-model').textContent =
      data.llm_model || '—';

    // 混合检索开关
    const hybridToggle = document.getElementById('hybrid-toggle');
    if (hybridToggle) {
      hybridToggle.checked = data.hybrid_search !== false;
    }

    // 会话数
    const sessionCount = document.getElementById('session-count');
    if (sessionCount) {
      sessionCount.textContent = data.active_sessions || 0;
    }

    // 从 /api/config 获取 Reranker 状态
    try {
      const configRes = await fetch('/api/config');
      const config = await configRes.json();
      const rerankerToggle = document.getElementById('reranker-toggle');
      if (rerankerToggle) {
        rerankerToggle.checked = config.reranker !== false;
      }
    } catch (_) {}

    if (!data.knowledge_base_ready) {
      document.getElementById('doc-count').style.color = '#e05555';
    } else {
      document.getElementById('doc-count').style.color = '#4caf82';
    }
  } catch (e) {
    document.getElementById('doc-count').textContent = '连接失败';
  }
}

// ============================================================
// 混合检索/Reranker 开关
// ============================================================

async function toggleHybridSearch(enabled) {
  try {
    await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hybrid_search: enabled }),
    });
    showToast(`混合检索已${enabled ? '启用' : '关闭'}`, 'info');
  } catch (e) {
    showToast('配置更新失败', 'error');
  }
}

async function toggleReranker(enabled) {
  try {
    await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reranker: enabled }),
    });
    showToast(`Reranker 重排已${enabled ? '启用' : '关闭'}`, 'info');
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
    const res = await fetch('/api/ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ clear: false }),
    });
    const data = await res.json();

    if (data.success) {
      showToast(`入库成功！新增 ${data.chunks_added} 个 Chunk`, 'success');
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
    showToast(`不支持的文件格式：${ext}`, 'error');
    return;
  }

  const maxSize = 50 * 1024 * 1024;
  if (file.size > maxSize) {
    showToast('文件大小不能超过 50MB', 'error');
    return;
  }

  showToast(`正在上传并处理：${file.name}...`, 'info');

  try {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();

    if (data.success) {
      showToast(`上传成功！${file.name} 已入库，新增 ${data.chunks_added} 个 Chunk`, 'success');
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

async function sendMessage() {
  if (isStreaming) return;

  const input = document.getElementById('question-input');
  const question = input.value.trim();
  if (!question) return;

  input.value = '';
  autoResizeTextarea(input);

  // 添加用户消息
  appendMessage('user', question);
  conversationHistory.push({ role: 'user', content: question });

  // 添加 AI 消息占位
  const aiMsgId = 'msg-' + Date.now();
  appendTypingMessage(aiMsgId);

  setBusy(true);

  try {
    await streamAnswer(question, aiMsgId);
  } catch (e) {
    updateMessage(aiMsgId, '请求失败：' + e.message);
  } finally {
    setBusy(false);
  }
}

// ============================================================
// SSE 流式接收
// ============================================================

async function streamAnswer(question, aiMsgId) {
  const hybridToggle = document.getElementById('hybrid-toggle');
  const rerankerToggle = document.getElementById('reranker-toggle');
  const hybrid = hybridToggle ? hybridToggle.checked : true;
  const reranker = rerankerToggle ? rerankerToggle.checked : true;

  const res = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: sessionId, hybrid, reranker }),
  });

  if (!res.ok) {
    const err = await res.json();
    updateMessage(aiMsgId, '错误：' + (err.error || '未知错误'));
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let fullAnswer = '';

  const msgEl = document.getElementById(aiMsgId);
  if (msgEl) {
    msgEl.querySelector('.message-text').innerHTML = '';
    msgEl.querySelector('.message-text').classList.add('streaming');
  }

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const jsonStr = line.slice(6).trim();
      if (!jsonStr) continue;

      try {
        const event = JSON.parse(jsonStr);

        if (event.type === 'sources') {
          // 先渲染数据源（折叠状态，在答案上方）
          if (event.sources && event.sources.length > 0) {
            currentSources = event.sources;
            renderSourcesInline(event.sources, aiMsgId);
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
          }

          // 保存助手回复
          conversationHistory.push({ role: 'assistant', content: fullAnswer });

          // 更新 session_id
          if (event.session_id && event.session_id !== sessionId) {
            sessionId = event.session_id;
            localStorage.setItem('rag_session_id', sessionId);
          }

          updateSessionIndicator();
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
// 来源展示（思索过程样式：在答案上方折叠显示）
// ============================================================

function renderSourcesInline(sources, aiMsgId) {
  const msgEl = document.getElementById(aiMsgId);
  if (!msgEl) return;

  const contentEl = msgEl.querySelector('.message-content');
  if (!contentEl) return;

  // 移除旧的来源区域（如果存在）
  const oldSources = contentEl.querySelector('.message-sources');
  if (oldSources) oldSources.remove();

  // 构建来源芯片列表
  let chipsHtml = '';
  sources.forEach((s) => {
    let scoreColor = '#4caf82';
    if (s.similarity) {
      const sim = parseFloat(s.similarity);
      if (sim < 60) scoreColor = '#e05555';
      else if (sim < 80) scoreColor = '#ff9800';
      else scoreColor = '#4caf82';
    }

    const typeLabel = s.retrieval_type
      ? `<span class="source-chip-type">${escapeHtml(s.retrieval_type)}</span>`
      : '';

    let pageLabel = '';
    if (s.page != null && s.page !== undefined) {
      if (s.total_pages != null && s.total_pages !== undefined) {
        pageLabel = `<span class="source-chip-page">第 ${s.page}/${s.total_pages} 页</span>`;
      } else {
        pageLabel = `<span class="source-chip-page">第 ${s.page} 页</span>`;
      }
    }

    chipsHtml += `
      <div class="source-chip" onclick="event.stopPropagation()">
        <div class="source-chip-header">
          <span class="source-chip-title">${escapeHtml(s.source)}</span>
          ${pageLabel}
          ${s.similarity ? `<span class="source-chip-score" style="color:${scoreColor}">${s.similarity}</span>` : ''}
          ${typeLabel}
          <span class="source-chip-toggle">▶</span>
        </div>
        <div class="source-chip-preview">${escapeHtml(s.preview)}</div>
        <div class="source-chip-full" style="display:none">${escapeHtml(s.full_content)}</div>
      </div>
    `;
  });

  // 创建来源折叠区域
  const sourcesSection = document.createElement('div');
  sourcesSection.className = 'message-sources';
  sourcesSection.innerHTML = `
    <div class="message-sources-header" onclick="toggleInlineSources(this)">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
      </svg>
      <span>已参考 ${sources.length} 个数据源</span>
      <span class="message-sources-chevron">▸</span>
    </div>
    <div class="message-sources-body" style="display:none">
      ${chipsHtml}
    </div>
  `;

  // 插入到消息文本之前
  const textEl = contentEl.querySelector('.message-text');
  contentEl.insertBefore(sourcesSection, textEl);

  // 绑定每个 source-chip 的点击展开事件
  sourcesSection.querySelectorAll('.source-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const isExpanded = chip.classList.toggle('expanded');
      chip.querySelector('.source-chip-preview').style.display = isExpanded ? 'none' : 'block';
      chip.querySelector('.source-chip-full').style.display = isExpanded ? 'block' : 'none';
      chip.querySelector('.source-chip-toggle').textContent = isExpanded ? '▼' : '▶';
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
    body.style.display = 'block';
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
    indicator.textContent = turn > 0 ? `第 ${turn} 轮对话` : '新对话';
  }
}

async function clearCurrentChat() {
  const messages = document.getElementById('messages');
  messages.innerHTML = `
    <div class="message assistant">
      <div class="message-avatar">🤖</div>
      <div class="message-content">
        <div class="message-text">对话已清空，您可以继续提问。</div>
      </div>
    </div>
  `;
  conversationHistory = [];

  try {
    await fetch(`/api/history/${sessionId}`, { method: 'DELETE' });
  } catch (_) {}

  updateSessionIndicator();
}

async function startNewSession() {
  sessionId = generateSessionId();
  clearCurrentChat();
  showToast('已开启新对话', 'info');
}

// ============================================================
// 工具函数
// ============================================================

function appendMessage(role, text) {
  const messages = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = `message ${role}`;
  div.innerHTML = `
    <div class="message-avatar">${role === 'user' ? '👤' : '🤖'}</div>
    <div class="message-content">
      <div class="message-text">${escapeHtml(text)}</div>
    </div>
  `;
  messages.appendChild(div);
  scrollToBottom();
}

function appendTypingMessage(id) {
  const messages = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'message assistant';
  div.id = id;
  div.innerHTML = `
    <div class="message-avatar">🤖</div>
    <div class="message-content">
      <div class="message-text">
        <div class="typing-indicator">
          <span></span><span></span><span></span>
        </div>
      </div>
    </div>
  `;
  messages.appendChild(div);
  scrollToBottom();
}

function updateMessage(id, text) {
  const el = document.getElementById(id);
  if (el) {
    const textEl = el.querySelector('.message-text');
    textEl.textContent = text;
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
  btn.disabled = busy;
  const input = document.getElementById('question-input');
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

function showToast(msg, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}

function autoResizeTextarea(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

// ============================================================
// 事件绑定
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
  loadStatus();
  updateSessionIndicator();

  const input = document.getElementById('question-input');

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  input.addEventListener('input', () => autoResizeTextarea(input));

  const hybridToggle = document.getElementById('hybrid-toggle');
  if (hybridToggle) {
    hybridToggle.addEventListener('change', (e) => {
      toggleHybridSearch(e.target.checked);
    });
  }

  const rerankerToggle = document.getElementById('reranker-toggle');
  if (rerankerToggle) {
    rerankerToggle.addEventListener('change', (e) => {
      toggleReranker(e.target.checked);
    });
  }

  const fileInput = document.getElementById('file-input');
  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
        e.target.value = '';
      }
    });
  }
});
