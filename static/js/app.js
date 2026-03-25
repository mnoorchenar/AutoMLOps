/* AutoMLOps — shared utilities */

// ── Toast notifications ────────────────────────────────────────────────────
function showToast(message, type = 'info', duration = 3500) {
  const area = document.getElementById('toast-area');
  if (!area) return;

  const icons = { success: '✅', error: '❌', info: 'ℹ️' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${message}</span>`;
  area.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(40px)';
    toast.style.transition = 'opacity .3s, transform .3s';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ── Responsive sidebar toggle ──────────────────────────────────────────────
(function () {
  function checkWidth() {
    const toggle = document.getElementById('sidebar-toggle');
    if (toggle) toggle.style.display = window.innerWidth <= 768 ? 'inline-flex' : 'none';
  }
  window.addEventListener('resize', checkWidth);
  document.addEventListener('DOMContentLoaded', checkWidth);

  // Close sidebar when clicking outside on mobile
  document.addEventListener('click', (e) => {
    const sidebar = document.getElementById('sidebar');
    const toggle  = document.getElementById('sidebar-toggle');
    if (window.innerWidth <= 768 && sidebar && sidebar.classList.contains('open')) {
      if (!sidebar.contains(e.target) && e.target !== toggle) {
        sidebar.classList.remove('open');
      }
    }
  });
})();

// ── Generic tab switcher ───────────────────────────────────────────────────
function activateTab(panelId, btn, groupClass) {
  document.querySelectorAll(`.${groupClass}`).forEach(p => p.classList.remove('active'));
  document.querySelectorAll(`[data-tab-group="${groupClass}"]`).forEach(b => b.classList.remove('active'));
  document.getElementById(panelId)?.classList.add('active');
  if (btn) btn.classList.add('active');
}

// ── Light / dark theme ─────────────────────────────────────────────────────
function toggleTheme() {
  document.body.classList.add('theme-transition');
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const next    = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
  const icon = document.getElementById('theme-icon');
  if (icon) icon.className = next === 'dark' ? 'fa-solid fa-moon' : 'fa-solid fa-sun';
  document.dispatchEvent(new CustomEvent('themechange', { detail: next }));
  setTimeout(() => document.body.classList.remove('theme-transition'), 300);
}

document.addEventListener('DOMContentLoaded', function () {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const icon = document.getElementById('theme-icon');
  if (icon) icon.className = current === 'dark' ? 'fa-solid fa-moon' : 'fa-solid fa-sun';
});
