/* ytedit web editor — vanilla JS, no build step.
 *
 * Layout of this file:
 *   1. helpers (DOM, time, fetch, toasts, markdown)
 *   2. app state
 *   3. routing + boot
 *   4. project list / top bar / stage buttons
 *   5. clip list
 *   6. player, waveform (wavesurfer + regions), transcript
 *   7. the mute/duck tool
 *   8. timeline editor (the "Advanced" tab: raw lists)
 *  8b. the Program view — the final cut: preview player, segment strip, inspector
 *   9. plan / footage log / preview / qc / publish tabs
 *  10. jobs panel
 *  11. keyboard shortcuts
 *
 * The server owns all judgement calls it can (the "next step" hint, stage
 * status, validation); this file only renders them.
 */
'use strict';

/* ------------------------------------------------------------------ 1 */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

/** Create an element with attributes/children in one call. */
function el(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k === 'text') node.textContent = v;
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else if (v === true) node.setAttribute(k, '');
    else node.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return node;
}

/** ``replaceChildren`` that drops null/undefined/false children.
 *
 * The native DOM method stringifies them ("null" showing up in the UI), while
 * every render function here uses `cond ? el(...) : null` for optional bits. */
function fill(node, ...kids) {
  node.replaceChildren(
    ...kids.flat().filter((k) => k !== null && k !== undefined && k !== false));
  return node;
}

/** Seconds -> ``m:ss.d`` (or ``h:mm:ss`` past an hour). */
function tc(s, decimals = 1) {
  if (!isFinite(s) || s === null) return '–';
  const sign = s < 0 ? '-' : '';
  s = Math.abs(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  const ss = sec.toFixed(decimals).padStart(decimals ? 3 + decimals : 2, '0');
  return h ? `${sign}${h}:${String(m).padStart(2, '0')}:${ss}` : `${sign}${m}:${ss}`;
}

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const num = (v, d = 0) => (v === '' || v === null || v === undefined || isNaN(+v) ? d : +v);
const round3 = (v) => Math.round(v * 1000) / 1000;

/** fetch() wrapper: JSON in, JSON out, HTTP errors become thrown Errors. */
async function api(path, { method = 'GET', body, quiet = false } = {}) {
  let res;
  try {
    res = await fetch(path, {
      method,
      headers: body !== undefined ? { 'Content-Type': 'application/json' } : {},
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    if (!quiet) toast(`network error: ${e.message}`, 'err');
    throw e;
  }
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  if (!res.ok) {
    const err = new Error(detailText(data) || `${res.status} ${res.statusText}`);
    err.status = res.status;
    err.data = data;
    if (!quiet) toast(err.message, 'err');
    throw err;
  }
  return data;
}

/** Pull a readable message out of a FastAPI error body. */
function detailText(data) {
  const d = data && data.detail !== undefined ? data.detail : data;
  if (typeof d === 'string') return d;
  if (d && typeof d === 'object') {
    if (d.issues) return `${d.detail || 'invalid'}: ${d.issues.slice(0, 3).join('; ')}`;
    if (d.errors) return `${d.detail || 'invalid'}: ${d.errors.slice(0, 3)
      .map((e) => `${(e.loc || []).join('.')} ${e.msg}`).join('; ')}`;
    if (d.detail) return String(d.detail);
  }
  return '';
}

function toast(message, kind = '', ms = 4200) {
  const node = el('div', { class: `toast ${kind}`, text: message });
  $('#toasts').append(node);
  setTimeout(() => { node.style.opacity = '0'; setTimeout(() => node.remove(), 250); }, ms);
}

/** A deliberately small markdown subset: headings, lists, tables, code, emphasis. */
function md2html(src) {
  if (!src) return '<p class="faint">empty</p>';
  const lines = String(src).replace(/\r\n?/g, '\n').split('\n');
  const out = [];
  let inCode = false, listType = null, para = [];
  const inline = (t) => esc(t)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
    .replace(/(^|\W)\*([^*\n]+)\*/g, '$1<i>$2</i>')
    .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  const flushPara = () => { if (para.length) { out.push(`<p>${inline(para.join(' '))}</p>`); para = []; } };
  const closeList = () => { if (listType) { out.push(`</${listType}>`); listType = null; } };

  for (const raw of lines) {
    const line = raw.replace(/\s+$/, '');
    if (/^```/.test(line)) {
      flushPara(); closeList();
      out.push(inCode ? '</code></pre>' : '<pre><code>');
      inCode = !inCode; continue;
    }
    if (inCode) { out.push(esc(raw) + '\n'); continue; }
    if (!line.trim()) { flushPara(); closeList(); continue; }
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) { flushPara(); closeList(); out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); continue; }
    if (/^(---|\*\*\*|___)\s*$/.test(line)) { flushPara(); closeList(); out.push('<hr>'); continue; }
    if (/^>\s?/.test(line)) { flushPara(); closeList(); out.push(`<blockquote>${inline(line.replace(/^>\s?/, ''))}</blockquote>`); continue; }
    if (/^\|.*\|$/.test(line)) {
      flushPara(); closeList();
      if (/^\|[\s:|-]+\|$/.test(line)) continue;
      const cells = line.slice(1, -1).split('|').map((c) => `<td>${inline(c.trim())}</td>`).join('');
      if (!out.length || !out[out.length - 1].startsWith('<table')) out.push('<table>');
      out.push(`<tr>${cells}</tr>`);
      continue;
    }
    const li = line.match(/^\s*([-*+]|\d+\.)\s+(.*)$/);
    if (li) {
      flushPara();
      const want = /^\d/.test(li[1]) ? 'ol' : 'ul';
      if (listType !== want) { closeList(); out.push(`<${want}>`); listType = want; }
      out.push(`<li>${inline(li[2])}</li>`);
      continue;
    }
    para.push(line.trim());
  }
  flushPara(); closeList();
  if (inCode) out.push('</code></pre>');
  return out.join('\n').replace(/(<table>(?:(?!<\/table>)[\s\S])*?)(?=<(?:h\d|p|ul|ol|pre|hr)\b|$)/g,
    (m) => (m.includes('</table>') ? m : m + '</table>'));
}

/* ------------------------------------------------------------------ 2 */
const S = {
  slug: null,
  state: null,          // GET /api/p/<slug>/state
  projects: [],
  clip: null,           // current clip record
  transcript: null,
  analysis: null,
  timeline: null,       // parsed plan/timeline.json
  timelineMeta: null,   // { has_draft, issues, mtime, ... }
  dirty: false,
  ws: null,             // wavesurfer instance
  regions: null,        // regions plugin
  addingStatic: false,  // suppresses region-created while seeding regions
  sel: null,            // { start, end, region }
  tab: 'program',
  jobTimer: null,
  stateTimer: null,
  openJob: null,        // job id whose log the panel shows
  hideExcluded: false,

  // --- program view ---
  positions: null,      // GET /timeline/positions (server-computed geometry)
  progSel: null,        // index of the selected segment in tracks.video
  progZoom: null,       // px per second; null = fit the strip to the panel
  progWave: null,       // wavesurfer instance of the inspector's source player
  progRegions: null,
  progLoop: null,       // { in, out } the mini source player loops over
  progRaf: 0,           // requestAnimationFrame handle of the playhead loop
  progPreviewV: null,   // cache-busting stamp of the <video> currently mounted
  stripTimer: 0,        // debounce handle for strip rebuilds while trimming
  trCache: {},          // clip id -> transcript document (inspector words)
};

const STAGES = [
  ['ingest', 'Ingest'], ['transcribe', 'Transcribe'], ['analyze', 'Analyze'],
  ['plan', 'Plan'], ['music', 'Music'], ['render_preview', 'Preview'],
  ['render_master', 'Master'], ['qc', 'QC'], ['publish', 'Publish'],
];

/* ------------------------------------------------------------------ 3 */
async function boot() {
  wireChrome();
  const match = location.pathname.match(/^\/p\/([^/]+)\/?$/);
  S.slug = match ? decodeURIComponent(match[1]) : null;
  await loadProjects();
  if (S.slug) {
    $('#editor').classList.remove('hidden');
    $('#jobbar').classList.remove('hidden');
    $('#projects').classList.add('hidden');
    await refreshState();
    await loadTimeline();
    // The Program tab is the point of the editor, but it is useless before a
    // plan exists — a fresh project opens on the raw lists instead.
    setTab(S.timeline ? 'program' : 'advanced');
    pollJobs();
    S.stateTimer = setInterval(() => refreshState({ soft: true }), 15000);
  } else {
    $('#projects').classList.remove('hidden');
    $('#onboard').classList.add('hidden');      // it describes the project page
    renderProjectList();
    $('#next-step').textContent = 'Pick a project, or create one and drop clips into its input/.';
  }
}

/* ------------------------------------------------------------------ 4 */
async function loadProjects() {
  try {
    const data = await api('/api/projects');
    S.projects = data.projects || [];
  } catch { S.projects = []; }
  const sel = $('#project-select');
  fill(sel, 
    el('option', { value: '' }, '— project —'),
    ...S.projects.map((p) => el('option', { value: p.slug, selected: p.slug === S.slug },
      `${p.title} (${p.slug})`)));
  sel.onchange = () => { if (sel.value) location.href = `/p/${encodeURIComponent(sel.value)}`; };
}

function renderProjectList() {
  const grid = $('#project-grid');
  if (!S.projects.length) {
    fill(grid, el('p', { class: 'empty' }, 'No projects yet — create one to start.'));
    return;
  }
  fill(grid, ...S.projects.map((p) => el('a', { class: 'pcard', href: `/p/${p.slug}` },
    el('h3', {}, p.title || p.slug),
    el('div', { class: 'slug' }, p.slug),
    el('div', { class: 'dots' }, ...STAGES.map(([key]) =>
      el('i', { dataset: { s: (p.stages?.[key]?.status) || 'pending' }, title: key }))),
    el('div', { class: 'muted' },
      `${p.clips} clip${p.clips === 1 ? '' : 's'} · ${p.language} · $${(p.costs?.spent ?? 0).toFixed(2)} / $${(p.costs?.budget ?? 0).toFixed(0)}`))));
}

/** Reload state.json + derived info. `soft` keeps the current selection quiet. */
async function refreshState({ soft = false } = {}) {
  let data;
  try {
    data = await api(`/api/p/${S.slug}/state`, { quiet: soft });
  } catch (e) {
    if (!soft) $('#next-step').textContent = `Cannot load ${S.slug}: ${e.message}`;
    return;
  }
  S.state = data;
  document.title = `${data.title} — ytedit`;
  renderTopbar();
  renderStages();
  renderClips();
  if (!soft || !S.clip) {
    const wanted = S.clip && data.clips.find((c) => c.id === S.clip.id);
    if (wanted) S.clip = wanted;
    else if (data.clips.length && !S.clip) selectClip(data.clips[0].id);
  } else if (S.clip) {
    const fresh = data.clips.find((c) => c.id === S.clip.id);
    if (fresh) S.clip = fresh;
  }
  // A background poll must not yank the tab the user is reading back to
  // "loading…" every 15 s; only an explicit refresh re-fetches the tab.
  if (!soft) renderTabHeader();
  else if (S.tab === 'preview') renderPreview();
  else if (S.tab === 'program' && data.files?.preview_url !== S.progPreviewV) {
    // A finished render publishes a new ?v=<mtime>; swap the player without
    // rebuilding the strip (and without losing the selection).
    renderProgViewer();
  }
}

function renderTopbar() {
  const d = S.state;
  $('#next-step').textContent = d.next_step || '';
  $('#next-step').title = d.next_step || '';
  const c = d.costs || {};
  const over = (c.spent ?? 0) > (c.budget ?? 0);
  $('#spend').className = `spend${over ? ' over' : ''}`;
  $('#spend').innerHTML = `<b>$${(c.spent ?? 0).toFixed(3)}</b> / $${(c.budget ?? 0).toFixed(2)}`;
  $('#spend').title = Object.entries(c.by_service || {})
    .map(([k, v]) => `${k}: $${v.toFixed(4)}`).join('\n') || 'nothing spent yet';
}

function renderStages() {
  const bar = $('#stagebar');
  const stages = S.state.stages || {};
  const running = S.state.running_job;
  fill(bar, ...STAGES.flatMap(([key, label], i) => {
    const st = stages[key] || { status: 'pending' };
    const live = running && running.stage === key;
    const btn = el('button', {
      class: 'stage',
      dataset: { status: live ? 'running' : st.status, stale: String(!!st.stale) },
      title: `${label}: ${st.status}${st.stale ? ' (stale — inputs changed since it ran)' : ''}`
        + (st.error ? `\n${st.error}` : '')
        + (st.finished ? `\nfinished ${st.finished}` : ''),
      disabled: !!running,
      onclick: (ev) => startJob(key, { force: ev.shiftKey }),
    }, el('span', { class: 'dot' }), label);
    return i && (key === 'render_preview' || key === 'qc')
      ? [el('span', { class: 'stage-sep' }, '·'), btn] : [btn];
  }));
  bar.append(el('span', { class: 'grow' }),
    el('span', { class: 'faint', title: 'Hold shift when clicking a stage to pass --force' },
      'shift-click = --force'));
}

/* ------------------------------------------------------------------ 5 */
function renderClips() {
  const list = $('#clip-list');
  const clips = (S.state.clips || []).filter((c) => !(S.hideExcluded && c.exclude));
  $('#clip-count').textContent = `${clips.length}`;
  if (!clips.length) {
    fill(list, el('div', { class: 'empty' },
      'No clips yet. Drop files into the project\'s input/ folder and run Ingest.'));
    return;
  }
  fill(list, ...clips.map((c) => {
    const f = c.flags || {};
    const marks = [
      f.instructions ? '📌' : '', f.takes ? '🔁' : '', f.background_music ? '⚠' : '',
      f.language_mismatch ? '🌐' : '', f.thumbnail_candidate ? '★' : '',
    ].filter(Boolean).join('');
    return el('div', {
      class: `clip${S.clip && S.clip.id === c.id ? ' active' : ''}${c.exclude ? ' excluded' : ''}`,
      dataset: { id: c.id },
      onclick: () => selectClip(c.id),
      title: f.summary || '',
    },
      c.poster_url
        ? el('img', { class: 'thumb', src: c.poster_url, loading: 'lazy', alt: c.id })
        : el('div', { class: 'thumb ph' }, 'no img'),
      el('div', {},
        el('div', { class: 'row spread' },
          el('span', { class: 'cid' }, c.id),
          el('span', { class: 'flags' }, marks)),
        el('div', { class: 'src nowrap' }, (c.source_file || '').split('/').pop() || '—'),
        el('div', { class: 'meta' },
          el('span', { class: 'mono' }, tc(c.duration || 0, 0)),
          el('span', { class: `badge ${c.orientation === 'vertical' ? 'v' : 'h'}` },
            c.orientation === 'vertical' ? '▯ vert' : '▭ horiz'),
          c.has_audio === false ? el('span', { class: 'badge na' }, 'no audio') : null,
          (c.kind_override || f.kind) ? el('span', { class: 'badge kind' }, c.kind_override || f.kind) : null,
          c.exclude ? el('span', { class: 'badge na' }, 'excluded') : null)));
  }));
}

/* ------------------------------------------------------------------ 6 */
async function selectClip(id) {
  const clip = (S.state?.clips || []).find((c) => c.id === id);
  if (!clip) return;
  S.clip = clip;
  S.transcript = null;
  S.analysis = null;
  clearSelection();
  renderClips();
  renderClipEditor();

  const video = $('#video');
  video.src = clip.proxy_url || '';
  video.load();
  renderPlayerMeta();

  const [tr, an] = await Promise.all([
    clip.has_transcript ? api(`/api/p/${S.slug}/transcript/${id}`, { quiet: true }).catch(() => null) : null,
    clip.has_analysis ? api(`/api/p/${S.slug}/analysis/${id}`, { quiet: true }).catch(() => null) : null,
  ]);
  if (S.clip?.id !== id) return;           // the user moved on while we fetched
  S.transcript = tr;
  S.analysis = an;
  renderTranscript();
  await buildWaveform(clip);
}

function renderPlayerMeta() {
  const c = S.clip;
  const box = $('#player-meta');
  if (!c) { fill(box); return; }
  const f = c.flags || {};
  fill(box, 
    el('span', { class: 'mono' }, c.id),
    el('span', { class: 'tc', id: 'cur-time' }, '0:00.0'),
    el('span', {}, `/ ${tc(c.duration || 0)}`),
    el('span', {}, `${c.width || '?'}×${c.height || '?'} · ${(c.fps || 0).toFixed
      ? (c.fps || 0).toFixed(2) : c.fps} fps${c.vfr ? ' · VFR' : ''}${c.hdr ? ' · ' + c.hdr : ''}`),
    f.location ? el('span', {}, `📍 ${f.location}`) : null,
    c.recorded_at ? el('span', { class: 'faint' }, c.recorded_at.replace('T', ' ').slice(0, 16)) : null,
    !c.proxy_url ? el('span', { class: 'badge na' }, 'no proxy — run Ingest') : null);
}

function renderClipEditor() {
  const c = S.clip;
  $('#clip-notes').value = c?.notes || '';
  $('#clip-exclude').checked = !!c?.exclude;
  $('#clip-kind').value = c?.kind_override || '';
  renderDenoise();
}

/** Per-clip noise removal: paid (ElevenLabs) or free (local), with an A/B. */
function renderDenoise() {
  const box = $('#clip-denoise');
  if (!box) return;
  const c = S.clip;
  if (!c) { fill(box); return; }
  const d = c.denoise || {};
  const busy = !!S.state?.running_job;
  fill(box,
    el('span', { class: 'faint' }, 'Denoise:'),
    el('button', {
      class: 'small', disabled: busy,
      title: 'ElevenLabs Audio Isolation — paid API, roughly $0.12 per minute of audio',
      onclick: () => denoiseClip('elevenlabs'),
    }, 'ElevenLabs (paid, ~$0.12/min)'),
    el('button', {
      class: 'small', disabled: busy, title: 'Local filter chain — free, weaker',
      onclick: () => denoiseClip('local'),
    }, 'Local (free)'),
    d.use ? el('span', { class: 'badge kind' }, `denoised (${d.engine || 'unknown'})`) : null,
    d.use ? el('button', {
      class: 'small ghost', disabled: busy, title: 'Go back to the raw recording',
      onclick: () => denoiseClip(null, { off: true }),
    }, 'turn off') : null,
    // The A/B only compares files an earlier run already wrote, so it is free
    // — and pointless before one has run.
    d.use ? el('button', {
      class: 'small ghost', disabled: busy,
      title: 'Write a 6 s original → denoised comparison (free)',
      onclick: () => denoiseClip(null, { preview: true }),
    }, 'build A/B') : null,
    d.ab_url ? el('span', { class: 'grow ab' },
      el('audio', { src: d.ab_url, controls: true, preload: 'none' }),
      el('span', { class: 'faint' }, '6 s original → 6 s denoised')) : null);
}

function denoiseClip(engine, { off = false, preview = false } = {}) {
  if (!S.clip) return;
  // ElevenLabs Audio Isolation is metered per minute of audio.
  if (engine === 'elevenlabs' && !confirm(
    `Run ElevenLabs denoise on ${S.clip.id} (${tc(S.clip.duration || 0, 0)})?\n`
    + 'This is a paid API call — roughly $0.12 per minute of audio.')) return;
  const args = off ? { clip: S.clip.id, off: true }
    : preview ? { clip: S.clip.id, preview: true }
      : { clip: S.clip.id, engine };
  startJob('denoise', args);
}

async function saveClip(patch) {
  if (!S.clip) return;
  try {
    const res = await api(`/api/p/${S.slug}/clips/${S.clip.id}`, { method: 'POST', body: patch });
    Object.assign(S.clip, res.clip);
    renderClips();
    toast('clip saved', 'ok', 1600);
  } catch { /* toasted already */ }
}

/** Build (or rebuild) the waveform for a clip from its precomputed peaks. */
async function buildWaveform(clip) {
  const box = $('#waveform');
  if (S.ws) { try { S.ws.destroy(); } catch { /* ignore */ } S.ws = null; S.regions = null; }
  fill(box);
  if (!clip.peaks_url) {
    box.classList.remove('loading');
    box.append(el('div', { class: 'empty' }, 'No waveform — run Ingest to build media/peaks/.'));
    return;
  }
  box.classList.add('loading');
  const data = await loadPeaks(clip.peaks_url);
  box.classList.remove('loading');
  if (!data) {
    box.append(el('div', { class: 'empty' }, 'peaks failed to load'));
    return;
  }
  const { tops, bottoms } = data;
  const duration = data.duration || clip.duration || 0;

  S.ws = WaveSurfer.create({
    container: box,
    media: $('#video'),
    peaks: [tops, bottoms],
    duration,
    height: 78,
    waveColor: '#3d4f66',
    progressColor: '#4ea3ff',
    cursorColor: '#ffffff',
    cursorWidth: 1,
    // Phone audio is often recorded 20 dB below full scale; without
    // normalize the wave is a hairline and useless for spotting speech.
    normalize: true,
    interact: true,
  });
  S.regions = S.ws.registerPlugin(WaveSurfer.Regions.create());
  S.regions.enableDragSelection({ color: 'rgba(78,163,255,0.22)' });
  S.regions.on('region-created', onRegionCreated);
  S.regions.on('region-updated', (r) => { if (isSelection(r)) setSelection(r); });
  S.regions.on('region-clicked', (r, ev) => {
    if (r.id.startsWith('mute:')) { ev.stopPropagation(); openMutePopover(r, ev); }
  });
  S.ws.on('ready', () => seedRegions());
  S.ws.on('timeupdate', onTimeUpdate);
  seedRegions();
}

const isSelection = (r) => !/^(mute|music|take|instr):/.test(r.id || '');

function onRegionCreated(region) {
  if (S.addingStatic || !isSelection(region)) return;
  if (S.sel && S.sel.region && S.sel.region !== region) {
    try { S.sel.region.remove(); } catch { /* already gone */ }
  }
  setSelection(region);
}

function setSelection(region) {
  S.sel = { start: region.start, end: region.end, region };
  $('#sel-info').textContent = `${tc(region.start)} → ${tc(region.end)}  (${(region.end - region.start).toFixed(2)}s)`;
  for (const id of ['#btn-sel-mute', '#btn-sel-duck', '#btn-sel-add', '#btn-sel-clear']) $(id).disabled = false;
}

function clearSelection() {
  if (S.sel && S.sel.region) { try { S.sel.region.remove(); } catch { /* gone */ } }
  S.sel = null;
  $('#sel-info').textContent = 'no selection';
  for (const id of ['#btn-sel-mute', '#btn-sel-duck', '#btn-sel-add', '#btn-sel-clear']) $(id).disabled = true;
  hidePopover();
}

/** Draw mute ranges, detected music and repeated takes onto the waveform. */
function seedRegions() {
  if (!S.regions || !S.clip) return;
  S.addingStatic = true;
  try {
    // getRegions() hands back the plugin's live array — iterate a copy, or
    // remove() splices under us and every second region survives.
    for (const r of [...S.regions.getRegions()]) if (!isSelection(r)) r.remove();
    const mutes = (S.timeline?.mute_ranges || []).filter((m) => m.clip === S.clip.id);
    mutes.forEach((m, i) => S.regions.addRegion({
      id: `mute:${i}`, start: m.s, end: m.e, drag: false, resize: false,
      color: 'rgba(255,92,92,0.26)',
      content: `${m.gain_db <= -40 ? 'mute' : m.gain_db + ' dB'}${m.reason ? ' · ' + m.reason : ''}`,
    }));
    const music = (S.analysis?.background_music || []);
    music.forEach((m, i) => S.regions.addRegion({
      id: `music:${i}`, start: +m.s || 0, end: +m.e || 0, drag: false, resize: false,
      color: 'rgba(240,180,41,0.16)', content: `♪ ${m.suggest || 'music'}`,
    }));
    (S.analysis?.instructions || []).forEach((x, i) => S.regions.addRegion({
      id: `instr:${i}`, start: +x.s || 0, end: +x.e || 0, drag: false, resize: false,
      color: 'rgba(240,180,41,0.30)', content: '📌 instruction',
    }));
    (S.analysis?.takes || []).forEach((t, ti) => (t.attempts || []).forEach((a, ai) => {
      S.regions.addRegion({
        id: `take:${ti}.${ai}`, start: +a.s || 0, end: +a.e || 0, drag: false, resize: false,
        color: ai === (t.keep ?? (t.attempts.length - 1))
          ? 'rgba(167,139,250,0.22)' : 'rgba(167,139,250,0.08)',
        content: `take ${ai + 1}${ai === t.keep ? ' ✓' : ''}`,
      });
    }));
  } catch (e) {
    console.warn('region seeding failed', e);
  } finally {
    S.addingStatic = false;
  }
}

function onTimeUpdate(t) {
  const cur = $('#cur-time');
  if (cur) cur.textContent = tc(t);
  const words = $$('#transcript .w');
  if (!words.length) return;
  let hit = null;
  for (const w of words) {
    const s = +w.dataset.s, e = +w.dataset.e;
    if (t >= s && t < e) { hit = w; break; }
  }
  if (hit !== S._word) {
    if (S._word) S._word.classList.remove('now');
    if (hit) hit.classList.add('now');
    S._word = hit;
  }
}

function renderTranscript() {
  const box = $('#transcript');
  const tr = S.transcript;
  if (!tr) {
    fill(box, el('div', { class: 'empty' },
      S.clip?.has_transcript ? 'transcript unreadable' : 'No transcript yet — run Transcribe.'));
    return;
  }
  const takeSpans = (S.analysis?.takes || []).flatMap((t) => (t.attempts || [])
    .map((a) => [+a.s, +a.e]));
  const instrSpans = (S.analysis?.instructions || []).map((i) => [+i.s, +i.e]);
  const mutes = (S.timeline?.mute_ranges || []).filter((m) => m.clip === S.clip?.id)
    .map((m) => [m.s, m.e]);
  const inAny = (spans, s, e) => spans.some(([a, b]) => e > a && s < b);

  const head = el('div', { class: 'row wrap faint', style: 'margin-bottom:6px' },
    el('span', { class: 'badge' }, tr.language || '?'),
    tr.language_mismatch ? el('span', { class: 'badge na' }, '🌐 language mismatch') : null,
    tr.engine ? el('span', {}, tr.engine) : null,
    el('span', {}, `${(tr.words || []).length} words`));

  const words = (tr.words || []).map((w) => {
    const s = +(w.s ?? w.start ?? 0), e = +(w.e ?? w.end ?? s);
    const classes = ['w'];
    if (inAny(takeSpans, s, e)) classes.push('take');
    if (inAny(instrSpans, s, e)) classes.push('instr');
    if (inAny(mutes, s, e)) classes.push('muted-range');
    return el('span', {
      class: classes.join(' '), dataset: { s, e }, title: `${tc(s)}`,
      onclick: () => { const v = $('#video'); v.currentTime = s; v.play().catch(() => {}); },
    }, (w.t ?? w.text ?? w.word ?? '') + ' ');
  });
  fill(box, head, el('div', {}, ...(words.length ? words
    : [el('span', { class: 'faint' }, tr.text || 'no words')])));
  S._word = null;
}

/* ------------------------------------------------------------------ 7 */
async function postMute(body) {
  if (!S.clip) return;
  try {
    const res = await api(`/api/p/${S.slug}/mute_ranges`, { method: 'POST', body });
    if (!S.timeline) await loadTimeline();
    else { S.timeline.mute_ranges = res.mute_ranges; renderTimelinePanel(); }
    seedRegions();
    renderTranscript();
    toast(body.op === 'delete' ? 'mute range removed' : 'mute range saved', 'ok', 2000);
  } catch { /* toasted */ }
}

function applySelectionMute(gainDb, reason = '') {
  if (!S.sel) { toast('select a range on the waveform first', 'warn'); return; }
  postMute({
    op: 'add', clip: S.clip.id, s: round3(S.sel.start), e: round3(S.sel.end),
    gain_db: gainDb, reason,
  }).then(() => clearSelection());
}

function showPopover(anchorEvent, ...children) {
  const pop = $('#popover');
  // The same click keeps bubbling to the document listener below, which would
  // close the popover we are opening right now; remember it and let it pass.
  S._popEvent = anchorEvent || null;
  fill(pop, ...children);
  pop.classList.remove('hidden');
  const x = Math.min((anchorEvent?.clientX ?? 200) + 8, window.innerWidth - pop.offsetWidth - 12);
  const y = Math.min((anchorEvent?.clientY ?? 200) + 8, window.innerHeight - pop.offsetHeight - 12);
  pop.style.left = `${Math.max(8, x)}px`;
  pop.style.top = `${Math.max(8, y)}px`;
}
const hidePopover = () => $('#popover').classList.add('hidden');

/** Popover for the drag selection: mute / duck / custom dB + reason. */
function openSelectionPopover(ev) {
  if (!S.sel) return;
  const gain = el('input', { type: 'number', value: '-60', step: '1', style: 'width:70px' });
  const reason = el('input', { type: 'text', placeholder: 'reason (e.g. bar music)', class: 'grow' });
  showPopover(ev,
    el('h5', {}, `mute ${tc(S.sel.start)} → ${tc(S.sel.end)}`),
    el('div', { class: 'row' },
      el('button', { class: 'small', onclick: () => { gain.value = '-60'; } }, 'Mute'),
      el('button', { class: 'small', onclick: () => { gain.value = '-12'; } }, 'Duck −12'),
      el('button', { class: 'small', onclick: () => { gain.value = '-6'; } }, '−6'),
      gain, el('span', { class: 'faint' }, 'dB')),
    el('div', { class: 'row' }, reason),
    el('div', { class: 'row spread' },
      el('button', { class: 'small ghost', onclick: hidePopover }, 'Cancel'),
      el('button', {
        class: 'small primary',
        onclick: () => { hidePopover(); applySelectionMute(num(gain.value, -60), reason.value); },
      }, 'Save')));
}

/** Popover for an existing mute region: edit gain/reason or delete it. */
function openMutePopover(region, ev) {
  const idx = +region.id.split(':')[1];
  const mutesForClip = (S.timeline?.mute_ranges || [])
    .map((m, i) => ({ m, i })).filter((x) => x.m.clip === S.clip.id);
  const entry = mutesForClip[idx];
  if (!entry) return;
  const gain = el('input', { type: 'number', value: String(entry.m.gain_db), step: '1', style: 'width:70px' });
  const reason = el('input', { type: 'text', value: entry.m.reason || '', class: 'grow' });
  showPopover(ev,
    el('h5', {}, `mute range ${tc(entry.m.s)} → ${tc(entry.m.e)}`),
    el('div', { class: 'row' }, gain, el('span', { class: 'faint' }, 'dB'), reason),
    el('div', { class: 'row spread' },
      el('button', {
        class: 'small danger',
        onclick: () => { hidePopover(); postMute({ op: 'delete', index: entry.i, clip: S.clip.id }); },
      }, 'Delete'),
      el('button', {
        class: 'small primary',
        onclick: () => {
          hidePopover();
          postMute({
            op: 'update', index: entry.i, clip: S.clip.id, s: entry.m.s, e: entry.m.e,
            gain_db: num(gain.value, -60), reason: reason.value,
          });
        },
      }, 'Save')));
}

/* ------------------------------------------------------------------ 8 */
async function loadTimeline() {
  try {
    const data = await api(`/api/p/${S.slug}/timeline`, { quiet: true });
    S.timeline = data.timeline;
    S.timelineMeta = data;
  } catch (e) {
    // 404 means "no timeline yet"; the body carries has_draft when a plan draft
    // is waiting, and state.files knows it too after a soft refresh.
    S.timeline = null;
    const body = e.data && typeof e.data.detail === 'object' ? e.data.detail : {};
    const files = S.state?.files || {};
    S.timelineMeta = {
      has_draft: !!(body.has_draft || files.draft),
      draft_newer: !!files.draft && !files.timeline,
      issues: [],
    };
  }
  S.dirty = false;
  await loadPositions();
  renderTimelinePanel();
  if (S.tab === 'program') renderProgram();
  seedRegions();
}

/** Fetch the server-computed segment geometry (see `progPositions`). */
async function loadPositions() {
  if (!S.timeline) { S.positions = null; return; }
  try {
    S.positions = await api(`/api/p/${S.slug}/timeline/positions`, { quiet: true });
  } catch {
    S.positions = null;
  }
}

/** Absolute start/end of every video segment (mirrors Timeline.segment_positions).
 *  Everything is counted in whole frames at the timeline fps — that is how the
 *  render cuts, so a fractional in/out never drifts the segments after it. */
function segmentPositions(tl) {
  const out = [];
  const fps = +tl?.fps > 0 ? +tl.fps : 30;
  const toFrames = (s) => Math.max(0, Math.floor(s * fps + 0.5));
  const toSeconds = (f) => Math.round((f / fps) * 1e6) / 1e6;
  let cursor = 0;
  let previous = 0;
  (tl?.tracks?.video || []).forEach((seg, i) => {
    const speed = +seg.speed > 0 ? +seg.speed : 1;
    const frames = Math.max(1, toFrames(Math.max(0, (+seg.out || 0) - (+seg.in || 0)) / speed));
    let overlap = 0;
    if (i > 0 && seg.transition_in?.type === 'xfade') {
      overlap = toFrames(Math.max(0, +seg.transition_in.duration || 0));
      overlap = Math.min(overlap, previous, frames);
    }
    const start = Math.max(0, cursor - overlap);
    const end = start + frames;
    out.push({ seg, start: toSeconds(start), end: toSeconds(end), dur: toSeconds(frames) });
    cursor = end;
    previous = frames;
  });
  return out;
}

const totalRuntime = (tl) => { const p = segmentPositions(tl); return p.length ? p[p.length - 1].end : 0; };

function markDirty() {
  S.dirty = true;
  const badge = $('#tl-dirty');
  if (badge) badge.textContent = '● unsaved';
}

function nextId(list, prefix) {
  let n = 1;
  const used = new Set((list || []).map((x) => x.id));
  while (used.has(prefix + String(n).padStart(3, '0'))) n++;
  return prefix + String(n).padStart(3, '0');
}

function renderTimelinePanel() {
  const panel = $('#panel-timeline');
  const meta = S.timelineMeta || {};
  const tl = S.timeline;

  const toolbar = el('div', { class: 'toolbar' },
    el('button', { class: 'primary small', onclick: saveTimeline, disabled: !tl }, 'Save'),
    el('button', { class: 'small', onclick: validateTimeline, disabled: !tl }, 'Validate'),
    meta.has_draft ? el('button', {
      class: 'small', title: 'Replace timeline.json with the plan draft (a backup is kept)',
      onclick: acceptDraft,
    }, meta.draft_newer ? 'Accept draft (newer)' : 'Accept draft') : null,
    el('span', { class: 'grow' }),
    el('span', { class: 'dirty', id: 'tl-dirty' }, S.dirty ? '● unsaved' : ''),
    el('span', { class: 'mono faint' }, tl ? `⏱ ${tc(totalRuntime(tl))}` : ''));

  if (!tl) {
    fill(panel, toolbar, el('div', { class: 'empty' },
      meta.has_draft
        ? 'No timeline yet, but a plan draft exists — click "Accept draft".'
        : 'No timeline yet. Run Plan, or mark mute ranges on the waveform to start one.'));
    return;
  }

  const issues = meta.issues || [];
  const issueBox = el('div', { class: `issues${issues.length ? '' : ' ok'}` },
    issues.length ? el('b', {}, `${issues.length} issue${issues.length === 1 ? '' : 's'}`)
      : el('b', {}, 'timeline validates'),
    issues.length ? el('ul', {}, ...issues.slice(0, 12).map((i) => el('li', {}, i))) : null);

  fill(panel, 
    toolbar, issueBox,
    markersStrip(tl),
    el('h4', {}, `Segments (${(tl.tracks?.video || []).length})`),
    segmentsList(tl),
    el('div', { class: 'row', style: 'margin:6px 0 2px' },
      el('button', { class: 'small', onclick: addSegmentFromSelection },
        '+ add current selection'),
      el('button', { class: 'small ghost', onclick: () => addSegment() }, '+ empty segment')),
    el('h4', {}, `Captions (${(tl.tracks?.captions || []).length})`), captionsList(tl),
    el('h4', {}, `Music cues (${(tl.tracks?.music || []).length})`), musicList(tl),
    el('h4', {}, `Mute ranges (${(tl.mute_ranges || []).length})`), muteList(tl),
    el('h4', {}, `Markers (${(tl.markers || []).length}) & chapters (${(tl.chapters || []).length})`),
    markerList(tl),
    el('h4', {}, 'Canvas'),
    el('div', { class: 'list-row faint' },
      `${tl.width}×${tl.height} @ ${tl.fps} fps · ${tl.language}`,
      el('span', { class: 'grow' }),
      tl.meta?.edited_by_human ? el('span', { class: 'badge kind' }, 'edited by human') : null,
      tl.meta?.generated_by ? el('span', { class: 'badge' }, tl.meta.generated_by) : null));
}

function markersStrip(tl) {
  const total = totalRuntime(tl) || 1;
  const strip = el('div', { class: 'markers', title: 'structural beats' });
  for (const m of tl.markers || []) {
    strip.append(el('span', {
      class: 'm', style: `left:${Math.min(99, (m.at / total) * 100)}%`, title: `${m.label} @ ${tc(m.at)}`,
    }, el('span', {}, m.label)));
  }
  return strip;
}

function segmentsList(tl) {
  const positions = segmentPositions(tl);
  const clipIds = (S.state?.clips || []).map((c) => c.id);
  const box = el('div', { id: 'seg-list' });
  (tl.tracks?.video || []).forEach((seg, i) => {
    const pos = positions[i] || { start: 0, end: 0, dur: 0 };
    const row = el('div', {
      class: 'seg', draggable: 'true', dataset: { i: String(i) },
      ondragstart: (e) => { e.dataTransfer.setData('text/plain', String(i)); row.classList.add('dragging'); },
      ondragend: () => row.classList.remove('dragging'),
      ondragover: (e) => { e.preventDefault(); row.classList.add('dragover'); },
      ondragleave: () => row.classList.remove('dragover'),
      ondrop: (e) => {
        e.preventDefault(); row.classList.remove('dragover');
        const from = +e.dataTransfer.getData('text/plain');
        if (isNaN(from) || from === i) return;
        const arr = tl.tracks.video;
        arr.splice(i, 0, arr.splice(from, 1)[0]);
        markDirty(); renderTimelinePanel();
      },
    });
    const durOut = el('span', { class: 'pos' }, `${tc(pos.start)}–${tc(pos.end)} · ${pos.dur.toFixed(2)}s`);
    const upd = () => { markDirty(); refreshSegDerived(); };

    row.append(
      el('div', { class: 'row' },
        el('span', { class: 'handle', title: 'drag to reorder' }, '⋮⋮'),
        el('span', { class: 'idx' }, seg.id || `s${i + 1}`),
        select(clipIds, seg.clip, (v) => { seg.clip = v; upd(); }, 'clip'),
        el('span', { class: 'grow' }), durOut,
        el('button', {
          class: 'small ghost', title: 'preview this cut',
          onclick: () => { selectClip(seg.clip).then(() => { const v = $('#video'); v.currentTime = +seg.in || 0; v.play().catch(() => {}); }); },
        }, '▶'),
        el('button', {
          class: 'small danger', title: 'delete segment',
          onclick: () => { tl.tracks.video.splice(i, 1); markDirty(); renderTimelinePanel(); },
        }, '✕')),
      el('div', { class: 'seg-grid' },
        field('in', numberInput(seg.in, (v) => { seg.in = v; upd(); })),
        field('out', numberInput(seg.out, (v) => { seg.out = v; upd(); })),
        field('role', textInput(seg.role, (v) => { seg.role = v; markDirty(); })),
        field('fit', select(['cover', 'contain', 'blur-fill', 'crop-pan'],
          seg.transform?.fit || 'cover', (v) => {
            seg.transform = Object.assign({ fit: 'cover', zoom: 1 }, seg.transform, { fit: v });
            markDirty();
          })),
        field('trans', select(['cut', 'fade', 'xfade'], seg.transition_in?.type || 'cut', (v) => {
          seg.transition_in = Object.assign({ type: 'cut', duration: 0, name: 'fade' },
            seg.transition_in, { type: v, duration: v === 'cut' ? 0 : (seg.transition_in?.duration || 0.5) });
          upd(); renderTimelinePanel();
        })),
        field('t-dur', numberInput(seg.transition_in?.duration ?? 0, (v) => {
          seg.transition_in = Object.assign({ type: 'cut', duration: 0 }, seg.transition_in, { duration: v });
          upd();
        }, 0.05)),
        field('speed', numberInput(seg.speed ?? 1, (v) => { seg.speed = v || 1; upd(); }, 0.05)),
        field('gain dB', numberInput(seg.source_audio_gain_db ?? 0, (v) => {
          seg.source_audio_gain_db = v; markDirty();
        }, 1)),
        el('label', { class: 'field' },
          el('input', {
            type: 'checkbox', checked: !!seg.mute_source,
            onchange: (e) => { seg.mute_source = e.target.checked; markDirty(); },
          }), 'mute source'),
        el('div', { class: 'field', style: 'grid-column: span 2' },
          'notes', textInput(seg.notes, (v) => { seg.notes = v; markDirty(); }))));
    box.append(row);
  });
  if (!(tl.tracks?.video || []).length) {
    box.append(el('div', { class: 'empty' }, 'No segments. Select a range in the player and click "+ add current selection".'));
  }
  return box;
}

/** Update only the derived read-outs so typing never rebuilds the DOM. */
function refreshSegDerived() {
  const tl = S.timeline;
  if (!tl) return;
  const positions = segmentPositions(tl);
  $$('#seg-list .seg').forEach((row, i) => {
    const out = $('.pos', row);
    const p = positions[i];
    if (out && p) out.textContent = `${tc(p.start)}–${tc(p.end)} · ${p.dur.toFixed(2)}s`;
  });
  const runtime = $('.toolbar .mono', $('#panel-timeline'));
  if (runtime) runtime.textContent = `⏱ ${tc(totalRuntime(tl))}`;
}

const field = (label, input) => el('label', { class: 'field' }, label, input);
function numberInput(value, onchange, step = 0.05) {
  return el('input', {
    type: 'number', step: String(step), value: value ?? 0,
    oninput: (e) => onchange(num(e.target.value, 0)),
  });
}
function textInput(value, onchange, placeholder = '') {
  return el('input', {
    type: 'text', value: value ?? '', placeholder,
    oninput: (e) => onchange(e.target.value),
  });
}
function select(options, value, onchange, placeholder) {
  const sel = el('select', { onchange: (e) => onchange(e.target.value) });
  if (placeholder && !options.includes(value)) sel.append(el('option', { value: value ?? '' }, value || placeholder));
  for (const o of options) sel.append(el('option', { value: o, selected: o === value }, o));
  return sel;
}

function addSegment(spec) {
  const tl = S.timeline || newLocalTimeline();
  tl.tracks = tl.tracks || {};
  tl.tracks.video = tl.tracks.video || [];
  tl.tracks.video.push(Object.assign({
    id: nextId(tl.tracks.video, 's'), clip: S.clip?.id || (S.state?.clips?.[0]?.id ?? ''),
    in: 0, out: Math.min(4, S.clip?.duration || 4), role: '',
    transform: { fit: 'cover', zoom: 1 }, grade: 'default',
    transition_in: { type: 'cut', duration: 0 }, speed: 1,
    mute_source: false, source_audio_gain_db: 0, notes: '',
  }, spec || {}));
  S.timeline = tl;
  markDirty();
  renderTimelinePanel();
  setTab('timeline');
}

function addSegmentFromSelection() {
  if (!S.sel || !S.clip) { toast('select a range on the waveform first', 'warn'); return; }
  addSegment({ clip: S.clip.id, in: round3(S.sel.start), out: round3(S.sel.end) });
  toast(`added ${S.clip.id} ${tc(S.sel.start)}–${tc(S.sel.end)}`, 'ok', 2200);
}

function newLocalTimeline() {
  return {
    version: 1, fps: 30, width: 1920, height: 1080, language: S.state?.language || 'pl',
    tracks: { video: [], voice: [], music: [], captions: [], sfx: [] },
    mute_ranges: [], markers: [], chapters: [],
    meta: { title_candidates: [], generated_by: 'web-editor', edited_by_human: true, notes: '' },
  };
}

function captionsList(tl) {
  const caps = tl.tracks?.captions || (tl.tracks.captions = []);
  const box = el('div', {});
  caps.forEach((c, i) => box.append(el('div', { class: 'list-row' },
    el('span', { class: 'mono faint' }, c.id || `t${i + 1}`),
    numberInput(c.at, (v) => { c.at = v; markDirty(); }),
    numberInput(c.end, (v) => { c.end = v; markDirty(); }),
    el('span', { class: 'grow' }, textInput(c.text, (v) => { c.text = v; markDirty(); }, 'caption text')),
    select(['location', 'hook', 'subtitle', 'note'], c.style || 'location', (v) => { c.style = v; markDirty(); }),
    select(['lower-left', 'lower-right', 'lower-center', 'center', 'upper-left', 'upper-right', 'upper-center'],
      c.position || 'lower-left', (v) => { c.position = v; markDirty(); }),
    el('button', {
      class: 'small danger', onclick: () => { caps.splice(i, 1); markDirty(); renderTimelinePanel(); },
    }, '✕'))));
  box.append(el('button', {
    class: 'small', onclick: () => {
      const at = $('#video').currentTime || 0;
      caps.push({ id: nextId(caps, 't'), at: round3(at), end: round3(at + 3), text: '', style: 'location', position: 'lower-left' });
      markDirty(); renderTimelinePanel();
    },
  }, '+ caption'));
  return box;
}

function musicList(tl) {
  const cues = tl.tracks?.music || (tl.tracks.music = []);
  const files = (S.state?.music || []).map((m) => m.file);
  const box = el('div', {});
  cues.forEach((c, i) => box.append(el('div', { class: 'list-row' },
    el('span', { class: 'mono faint' }, c.id || `m${i + 1}`),
    el('span', { class: 'grow' }, select(files, c.file, (v) => { c.file = v; markDirty(); }, 'file')),
    field('at', numberInput(c.at, (v) => { c.at = v; markDirty(); })),
    field('end', numberInput(c.end, (v) => { c.end = v; markDirty(); })),
    field('dB', numberInput(c.gain_db ?? -18, (v) => { c.gain_db = v; markDirty(); }, 1)),
    field('duck', numberInput(c.duck?.amount_db ?? -12, (v) => {
      c.duck = Object.assign({ mode: 'auto', amount_db: -12, attack: 0.15, release: 0.6 }, c.duck, { amount_db: v });
      markDirty();
    }, 1)),
    files.includes(c.file) ? null : el('span', { class: 'badge na', title: 'file not on disk' }, 'missing'),
    el('button', {
      class: 'small danger', onclick: () => { cues.splice(i, 1); markDirty(); renderTimelinePanel(); },
    }, '✕'))));
  box.append(el('button', {
    class: 'small', disabled: !files.length,
    title: files.length ? '' : 'no music generated yet — run Music',
    onclick: () => {
      cues.push({
        id: nextId(cues, 'm'), file: files[0], at: 0, end: round3(totalRuntime(tl) || 60),
        gain_db: -18, fade_in: 2, fade_out: 3,
        duck: { mode: 'auto', amount_db: -12, attack: 0.15, release: 0.6 },
      });
      markDirty(); renderTimelinePanel();
    },
  }, '+ music cue'));
  return box;
}

function muteList(tl) {
  const box = el('div', {});
  (tl.mute_ranges || []).forEach((m, i) => box.append(el('div', { class: 'list-row' },
    el('a', {
      class: 'mono', href: '#',
      onclick: (e) => { e.preventDefault(); selectClip(m.clip).then(() => { $('#video').currentTime = m.s; }); },
    }, m.clip),
    el('span', { class: 'mono faint' }, `${tc(m.s)} → ${tc(m.e)}`),
    field('dB', numberInput(m.gain_db, (v) => { m.gain_db = v; markDirty(); }, 1)),
    el('span', { class: 'grow' }, textInput(m.reason, (v) => { m.reason = v; markDirty(); }, 'reason')),
    el('button', {
      class: 'small danger',
      onclick: () => postMute({ op: 'delete', index: i, clip: m.clip }),
    }, '✕'))));
  if (!(tl.mute_ranges || []).length) box.append(el('div', { class: 'faint' }, 'none — drag on the waveform to add one'));
  return box;
}

function markerList(tl) {
  const box = el('div', {});
  (tl.markers || []).forEach((m, i) => box.append(el('div', { class: 'list-row' },
    numberInput(m.at, (v) => { m.at = v; markDirty(); }),
    el('span', { class: 'grow' }, textInput(m.label, (v) => { m.label = v; markDirty(); }, 'label')),
    el('button', { class: 'small danger', onclick: () => { tl.markers.splice(i, 1); markDirty(); renderTimelinePanel(); } }, '✕'))));
  (tl.chapters || []).forEach((c, i) => box.append(el('div', { class: 'list-row' },
    el('span', { class: 'badge' }, 'chapter'),
    numberInput(c.at, (v) => { c.at = v; markDirty(); }),
    el('span', { class: 'grow' }, textInput(c.title, (v) => { c.title = v; markDirty(); }, 'title')),
    el('button', { class: 'small danger', onclick: () => { tl.chapters.splice(i, 1); markDirty(); renderTimelinePanel(); } }, '✕'))));
  box.append(
    el('button', {
      class: 'small', onclick: () => {
        (tl.markers = tl.markers || []).push({ at: round3($('#video').currentTime || 0), label: 'beat' });
        markDirty(); renderTimelinePanel();
      },
    }, '+ marker'),
    el('button', {
      class: 'small', onclick: () => {
        (tl.chapters = tl.chapters || []).push({ at: 0, title: '' });
        markDirty(); renderTimelinePanel();
      },
    }, '+ chapter'));
  return box;
}

/** Save the timeline; returns true when it actually landed on disk. */
async function saveTimeline({ force = false } = {}) {
  if (!S.timeline) return false;
  try {
    const res = await api(`/api/p/${S.slug}/timeline${force ? '?force=true' : ''}`,
      { method: 'PUT', body: S.timeline, quiet: true });
    S.dirty = false;
    toast(`timeline saved (${tc(res.duration)})${res.backup ? ' · backup kept' : ''}`, 'ok');
    await loadTimeline();
    refreshState({ soft: true });
    return true;
  } catch (e) {
    const issues = e.data?.detail?.issues;
    if (e.status === 400 && issues) {
      S.timelineMeta = Object.assign({}, S.timelineMeta, { issues });
      if (S.tab === 'program') {
        // No issue list on screen here, so ask outright rather than sending
        // the user hunting through the Advanced tab.
        return confirm(`The timeline has ${issues.length} issue(s):\n\n`
          + `${issues.slice(0, 8).join('\n')}\n\nSave anyway?`)
          ? saveTimeline({ force: true })
          : false;
      }
      toast(`${issues.length} issue(s) — see the Advanced tab`, 'warn', 6000);
      renderTimelinePanel();
      const box = $('#panel-timeline .issues');
      if (box) box.append(el('div', { class: 'row', style: 'margin-top:6px' },
        el('button', { class: 'small danger', onclick: () => saveTimeline({ force: true }) },
          'Save anyway')));
    } else {
      toast(e.message, 'err', 6000);
    }
    return false;
  }
}

async function validateTimeline() {
  try {
    const res = await api(`/api/p/${S.slug}/timeline/validate`, { method: 'POST', body: S.timeline });
    S.timelineMeta = Object.assign({}, S.timelineMeta, { issues: res.issues });
    renderTimelinePanel();
    toast(res.ok ? `valid · runtime ${tc(res.duration)}` : `${res.issues.length} issue(s)`,
      res.ok ? 'ok' : 'warn');
  } catch { /* toasted */ }
}

async function acceptDraft() {
  if (!confirm('Replace timeline.json with the plan draft? The current file is backed up.')) return;
  try {
    await api(`/api/p/${S.slug}/timeline/accept_draft`, { method: 'POST' });
    toast('draft accepted', 'ok');
    await loadTimeline();
    refreshState({ soft: true });
  } catch { /* toasted */ }
}

/* ----------------------------------------------------------------- 8b */
/* The Program view. Everything here answers one question: what does the
 * finished video look like, and how do I fix this one cut?
 *
 * Geometry comes from `GET /timeline/positions`, which runs the real
 * `Timeline.segment_positions()`; the JS mirror below is only used while there
 * are unsaved edits (the server cannot know about those yet), and every save
 * re-fetches the authoritative numbers.
 */
const ROLES = ['', 'cold-open', 'hook', 'a-roll', 'b-roll', 'cutaway', 'transition',
  'payoff', 'cta', 'outro'];

/** Read-only label for an overlay cutaway: which clip range it is heard as. */
function audioFromLabel(seg) {
  const a = seg?.audio_from;
  if (!a) return '';
  return `audio: ${a.clip} ${(+a.in || 0).toFixed(2)}–${(+a.out || 0).toFixed(2)}`;
}

/** Segment geometry: server-computed when saved, locally mirrored while dirty. */
function progPositions() {
  const tl = S.timeline;
  if (!tl) return [];
  const segs = tl.tracks?.video || [];
  const fromServer = S.positions?.positions || [];
  if (!S.dirty && fromServer.length === segs.length) return fromServer;
  return segmentPositions(tl).map(({ seg, start, end, dur }) => ({
    id: seg.id, clip: seg.clip, start, end, duration: dur,
    in: +seg.in || 0, out: +seg.out || 0, role: seg.role || '',
    mute_source: !!seg.mute_source, fit: seg.transform?.fit || 'cover',
    transition: seg.transition_in?.type || 'cut',
  }));
}

/** Mute ranges (stored in clip time) mapped onto programme time. */
function progMutes(positions) {
  if (!S.dirty && S.positions?.mutes) return S.positions.mutes;
  const tl = S.timeline;
  const out = [];
  for (const m of tl?.mute_ranges || []) {
    positions.forEach((p, i) => {
      const seg = (tl.tracks?.video || [])[i];
      if (!seg || seg.clip !== m.clip || seg.mute_source) return;
      const lo = Math.max(m.s, p.in), hi = Math.min(m.e, p.out);
      if (hi <= lo) return;
      const speed = +seg.speed > 0 ? +seg.speed : 1;
      out.push({
        clip: m.clip, segment: p.id, gain_db: m.gain_db, reason: m.reason,
        start: p.start + (lo - p.in) / speed, end: p.start + (hi - p.in) / speed,
      });
    });
  }
  return out;
}

const progTotal = (positions) => (positions.length ? positions[positions.length - 1].end : 0);
const progVideo = () => $('#prog-video');
const srcVideo = () => $('#src-video');
/** Thumbnail of `clip` at `t`. Quantised to 0.25 s: nudging in/out must not
 *  spawn an ffmpeg run (and a cache file) per keystroke. */
const frameUrl = (clip, t, w = 320) =>
  `/api/p/${S.slug}/frame?clip=${encodeURIComponent(clip)}`
  + `&t=${(Math.round((+t || 0) * 4) / 4).toFixed(2)}&w=${w}`;

/** Pixels per second: the explicit zoom, or whatever fits the panel. */
function progPxPerSec(total) {
  if (S.progZoom) return S.progZoom;
  // The scroll box only exists from the second render on; the wrap around it is
  // already in the DOM on the first one, so "fit" is stable either way.
  const box = $('#prog-strip-scroll') || $('#prog-strip-wrap') || $('#panel-program');
  const width = (box ? box.clientWidth : 0) || 760;
  return total > 0 ? Math.max(4, (width - 16) / total) : 40;
}

function setProgZoom(factor) {
  const positions = progPositions();
  S.progZoom = Math.max(2, Math.min(400, progPxPerSec(progTotal(positions)) * factor));
  renderProgStrip();
}

/* ---------------------------------------------------------------- render */
function renderProgram() {
  const panel = $('#panel-program');
  const files = S.state?.files || {};
  const tl = S.timeline;

  if (!S.timelineMeta) {                      // boot: the fetch is still in flight
    fill(panel, el('div', { class: 'empty' }, 'loading the cut…'));
    return;
  }
  if (!tl || !(tl.tracks?.video || []).length) {
    const hasDraft = !!(S.timelineMeta?.has_draft || files.draft);
    fill(panel, el('div', { class: 'empty' },
      el('p', {}, hasDraft
        ? 'No cut yet, but a plan draft is waiting — accept it and it shows up here.'
        : 'No cut yet. Run Plan to turn the footage log into a timeline, '
          + 'then come back here to watch and trim it.'),
      hasDraft
        ? el('button', { class: 'primary', onclick: acceptDraft }, 'Accept plan draft')
        : el('button', { class: 'primary', onclick: () => startJob('plan') }, 'Run Plan')));
    return;
  }
  // A fresh plan draft is waiting on top of the cut that is on screen.
  const draftNotice = (S.timelineMeta?.draft_newer)
    ? el('div', { class: 'issues', style: 'margin:6px 10px' },
      el('b', {}, 'A newer plan draft is waiting. '),
      el('button', { class: 'small', onclick: acceptDraft }, 'Accept draft'),
      ' — it replaces the cut below (the current one is backed up).')
    : null;

  fill(panel, el('div', { class: 'prog' },
    el('div', { class: 'prog-main' },
      progToolbar(files),
      draftNotice,
      el('div', { class: 'prog-viewer', id: 'prog-viewer' }),
      el('div', { class: 'prog-strip-wrap', id: 'prog-strip-wrap' })),
    el('aside', { class: 'prog-inspector', id: 'prog-inspector' })));

  renderProgViewer();
  renderProgStrip();
  renderProgInspector();
}

function progToolbar(files) {
  const stale = files.preview && files.preview_mtime < files.timeline_mtime;
  return el('div', { class: 'toolbar' },
    el('button', { class: 'primary small', onclick: () => saveTimeline() }, 'Save'),
    el('button', {
      class: 'small', title: 'Save the timeline, then re-render the 720p preview',
      onclick: saveAndPreview,
    }, 'Save & render preview'),
    el('button', {
      class: 'small',
      title: 'Snap every cut to speech boundaries (~0.3 s before / ~0.5 s after)',
      onclick: () => startJob('tidy'),
    }, 'Tidy cuts'),
    el('span', { class: 'dirty', id: 'tl-dirty' }, S.dirty ? '● unsaved' : ''),
    stale ? el('span', { class: 'badge na', title: 'the timeline changed since this preview' },
      'preview stale') : null,
    el('span', { class: 'grow' }),
    el('span', { class: 'mono faint', id: 'prog-runtime' }, `⏱ ${tc(progTotal(progPositions()))}`),
    el('button', { class: 'small ghost', title: 'zoom out (ctrl+wheel)', onclick: () => setProgZoom(1 / 1.4) }, '−'),
    el('button', { class: 'small ghost', title: 'zoom in (ctrl+wheel)', onclick: () => setProgZoom(1.4) }, '+'),
    el('button', { class: 'small ghost', title: 'fit the whole programme', onclick: () => { S.progZoom = null; renderProgStrip(); } }, 'fit'));
}

/** The programme player: the rendered preview, or the button that makes one. */
function renderProgViewer() {
  const box = $('#prog-viewer');
  if (!box) return;
  const files = S.state?.files || {};
  if (!files.preview_url) {
    fill(box, el('div', { class: 'empty' },
      el('p', {}, 'No preview rendered yet — this is where the finished video plays.'),
      el('button', { class: 'primary', onclick: () => startJob('render_preview') },
        'Render preview')));
    S.progPreviewV = null;
    return;
  }
  S.progPreviewV = files.preview_url;
  const video = el('video', {
    id: 'prog-video', src: files.preview_url, controls: true, playsinline: true,
    preload: 'metadata',
  });
  video.addEventListener('play', startPlayheadLoop);
  video.addEventListener('pause', stopPlayheadLoop);
  video.addEventListener('seeked', movePlayhead);
  video.addEventListener('timeupdate', movePlayhead);
  fill(box, video);
}

/** Rebuild the strip and its lanes without touching the players. */
function renderProgStrip() {
  const wrap = $('#prog-strip-wrap');
  if (!wrap || !S.timeline) return;
  const positions = progPositions();
  const total = progTotal(positions) || 1;
  const pps = progPxPerSec(total);
  const width = Math.max(120, total * pps);
  const segs = S.timeline.tracks.video;

  const strip = el('div', { class: 'prog-strip', style: `width:${width}px` });
  positions.forEach((p, i) => {
    const seg = segs[i];
    const w = Math.max(3, (p.end - p.start) * pps);
    const block = el('div', {
      class: `prog-seg${i === S.progSel ? ' sel' : ''}`,
      style: `left:${p.start * pps}px;width:${w}px`,
      draggable: 'true', dataset: { i: String(i) },
      title: `${p.id} · ${p.clip} ${tc(p.in)}–${tc(p.out)} (${(p.end - p.start).toFixed(2)}s)`
        + (seg?.notes ? `\n${seg.notes}` : ''),
      onclick: () => selectProgSegment(i, { seek: true }),
      ondragstart: (e) => { e.dataTransfer.setData('text/plain', String(i)); block.classList.add('dragging'); },
      ondragend: () => block.classList.remove('dragging'),
      ondragover: (e) => { e.preventDefault(); block.classList.add('dragover'); },
      ondragleave: () => block.classList.remove('dragover'),
      ondrop: (e) => {
        e.preventDefault(); e.stopPropagation();
        block.classList.remove('dragover');
        moveSegment(+e.dataTransfer.getData('text/plain'), i);
      },
    },
      w > 26 ? el('img', {
        class: 'sh', src: frameUrl(p.clip, p.in, 320), loading: 'lazy', alt: '',
        onerror: (e) => { e.target.remove(); },
      }) : null,
      el('div', { class: 'lbl' },
        el('span', { class: 'cid' }, p.clip),
        seg?.mute_source ? el('span', { class: 'mute-ic', title: 'source audio muted' }, '🔇') : null,
        seg?.audio_from ? el('span', { class: 'mute-ic', title: audioFromLabel(seg) }, '🎞') : null,
        p.role ? el('span', { class: 'role' }, p.role) : null),
      w > 54 ? el('div', { class: 'tcs' }, `${(p.end - p.start).toFixed(1)}s`) : null);
    strip.append(block);
  });
  strip.addEventListener('click', (e) => {
    if (e.target !== strip) return;                 // a block handles its own click
    seekProgram(e.offsetX / pps);
  });

  const scroll = el('div', {
    class: 'prog-strip-scroll', id: 'prog-strip-scroll',
    onwheel: (e) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      setProgZoom(e.deltaY < 0 ? 1.15 : 1 / 1.15);
    },
  },
    // The playhead is a child of the scroll box, not the strip, so the line
    // runs down through every lane and scrolls with the content.
    el('div', { class: 'prog-playhead', id: 'prog-playhead' }),
    strip,
    lane('captions', width, (S.timeline.tracks?.captions || []).map((c) => {
      // A location card (or an anchored hook line) is pinned to a video
      // segment rather than absolute time (see Timeline.resolve_anchors /
      // ytedit/ai/locations.py) so it follows its picture through later
      // padding/overlay/dedupe passes — shown read-only, same as the voice
      // lane's anchor tag.
      const anchorTag = c.anchor ? ` ⚓${c.anchor.segment}` : '';
      const anchorNote = c.anchor
        ? `\nanchored to ${c.anchor.segment} +${(+c.anchor.offset || 0).toFixed(2)}s`
        : '';
      return {
        start: +c.at || 0, end: +c.end || 0, label: `${c.text || c.id}${anchorTag}`, cls: 'cap',
        title: `${c.id} · ${c.style} · ${tc(c.at)}–${tc(c.end)}\n${c.text || ''}${anchorNote}`,
      };
    }), pps),
    lane('music', width, (S.timeline.tracks?.music || []).map((m) => ({
      start: +m.at || 0, end: +m.end || 0,
      label: `♪ ${m.id} ${m.gain_db ?? -18} dB`, cls: 'mus',
      title: `${m.file || ''}\n${tc(m.at)}–${tc(m.end)} · ${m.gain_db} dB · duck ${m.duck?.amount_db ?? -12} dB`,
    })), pps),
    lane('voice', width, (S.timeline.tracks?.voice || []).map((v) => {
      const end = v.end != null ? +v.end : +v.at || 0;
      const name = (v.file || '').split('/').pop() || v.id;
      // Anchored items are pinned to a video segment (see
      // Timeline.resolve_voice_anchors) rather than absolute time — shown
      // read-only so the user can see why a pickup follows its picture.
      const anchorTag = v.anchor ? ` ⚓${v.anchor.segment}` : '';
      const anchorNote = v.anchor
        ? `\nanchored to ${v.anchor.segment} +${(+v.anchor.offset || 0).toFixed(2)}s`
        : '';
      return {
        start: +v.at || 0, end, label: `🎙 ${name}${anchorTag}`, cls: 'voi',
        title: `${v.file || ''}\n${tc(v.at)}–${tc(end)} · ${v.gain_db ?? 0} dB${anchorNote}`,
      };
    }), pps),
    lane('muted src', width, progMutes(positions).map((m) => ({
      start: m.start, end: m.end,
      label: m.gain_db <= -40 ? 'mute' : `${m.gain_db} dB`, cls: 'mut',
      title: `${m.clip} in ${m.segment}: ${m.reason || 'muted'}`,
    })), pps),
    markerLane(S.timeline, width, pps));

  fill(wrap, scroll);
  movePlayhead();
}

/** One horizontal lane of labelled blocks under the strip. */
function lane(name, width, items, pps) {
  const row = el('div', { class: 'prog-lane', style: `width:${width}px` },
    el('span', { class: 'lane-name' }, name));
  for (const it of items) {
    const w = Math.max(2, (it.end - it.start) * pps);
    row.append(el('div', {
      class: `lane-block ${it.cls}`, style: `left:${it.start * pps}px;width:${w}px`,
      title: it.title, onclick: () => seekProgram(it.start),
    }, w > 24 ? el('span', {}, it.label) : null));
  }
  if (!items.length) row.append(el('span', { class: 'lane-empty' }, '—'));
  return row;
}

function markerLane(tl, width, pps) {
  const row = el('div', { class: 'prog-lane markers-lane', style: `width:${width}px` },
    el('span', { class: 'lane-name' }, 'beats'));
  for (const m of tl.markers || []) {
    row.append(el('div', {
      class: 'tick', style: `left:${(+m.at || 0) * pps}px`, title: `${m.label} @ ${tc(m.at)}`,
      onclick: () => seekProgram(+m.at || 0),
    }, el('span', {}, m.label)));
  }
  for (const c of tl.chapters || []) {
    row.append(el('div', {
      class: 'tick chap', style: `left:${(+c.at || 0) * pps}px`, title: `chapter: ${c.title}`,
      onclick: () => seekProgram(+c.at || 0),
    }, el('span', {}, `§ ${c.title}`)));
  }
  return row;
}

/* -------------------------------------------------------------- playhead */
function movePlayhead() {
  const head = $('#prog-playhead');
  const video = progVideo();
  if (!head) return;
  const positions = progPositions();
  const pps = progPxPerSec(progTotal(positions) || 1);
  const t = video ? video.currentTime || 0 : 0;
  head.style.left = `${t * pps}px`;
  const scroll = $('#prog-strip-scroll');
  if (scroll && video && !video.paused) {
    const x = t * pps;
    if (x < scroll.scrollLeft + 40 || x > scroll.scrollLeft + scroll.clientWidth - 60) {
      scroll.scrollLeft = Math.max(0, x - scroll.clientWidth / 3);
    }
  }
  const read = $('#prog-time');
  if (read) read.textContent = tc(t);
}

function startPlayheadLoop() {
  stopPlayheadLoop();
  const step = () => { movePlayhead(); S.progRaf = requestAnimationFrame(step); };
  S.progRaf = requestAnimationFrame(step);
}
function stopPlayheadLoop() {
  if (S.progRaf) cancelAnimationFrame(S.progRaf);
  S.progRaf = 0;
  movePlayhead();
}

function seekProgram(t) {
  const video = progVideo();
  if (!video) return;
  video.currentTime = Math.max(0, t);
  movePlayhead();
}

/* ------------------------------------------------------------- selection */
function selectProgSegment(i, { seek = false } = {}) {
  const segs = S.timeline?.tracks?.video || [];
  if (i === null || i < 0 || i >= segs.length) { S.progSel = null; }
  else S.progSel = i;
  $$('#prog-strip-wrap .prog-seg').forEach((b, bi) => b.classList.toggle('sel', bi === S.progSel));
  if (seek && S.progSel !== null) {
    const p = progPositions()[S.progSel];
    if (p) seekProgram(p.start + 0.01);
  }
  renderProgInspector();
}

const selectedSegment = () => (S.progSel === null ? null
  : (S.timeline?.tracks?.video || [])[S.progSel] || null);

/** Rebuild the strip after a short quiet period: holding down a nudge button
 *  or typing in the in/out fields must not rebuild 13 blocks per keystroke. */
function scheduleStripRender() {
  clearTimeout(S.stripTimer);
  S.stripTimer = setTimeout(() => { if (S.tab === 'program') renderProgStrip(); }, 180);
}

/** Redraw only the numbers a nudge changes — the players keep playing. */
function progTouch() {
  markDirty();
  scheduleStripRender();
  const seg = selectedSegment();
  if (seg) {
    S.progLoop = { in: +seg.in || 0, out: +seg.out || 0 };
    const dur = $('#insp-dur');
    if (dur) dur.textContent = `${(Math.max(0, seg.out - seg.in)).toFixed(2)}s`;
    const inp = $('#insp-in'), outp = $('#insp-out');
    if (inp && document.activeElement !== inp) inp.value = round3(+seg.in || 0);
    if (outp && document.activeElement !== outp) outp.value = round3(+seg.out || 0);
    if (S.progRegions) {
      const region = S.progRegions.getRegions().find((r) => r.id === 'cut');
      if (region) region.setOptions({ start: +seg.in || 0, end: +seg.out || 0 });
    }
    highlightSourceWords();
  }
  const runtime = $('#prog-runtime');
  if (runtime) runtime.textContent = `⏱ ${tc(progTotal(progPositions()))}`;
}

function moveSegment(from, to) {
  const arr = S.timeline?.tracks?.video;
  if (!arr || isNaN(from) || from === to || from < 0 || from >= arr.length) return;
  arr.splice(to, 0, arr.splice(from, 1)[0]);
  S.progSel = to;
  markDirty();
  renderProgStrip();
  renderProgInspector();
}

function deleteSegment() {
  const arr = S.timeline?.tracks?.video;
  if (!arr || S.progSel === null) return;
  const [gone] = arr.splice(S.progSel, 1);
  toast(`removed ${gone.id} (${gone.clip})`, 'warn', 2600);
  S.progSel = arr.length ? Math.min(S.progSel, arr.length - 1) : null;
  markDirty();
  renderProgStrip();
  renderProgInspector();
}

/** Split the selected segment at the playhead (programme or source). */
function splitSegment() {
  const seg = selectedSegment();
  if (!seg) { toast('select a segment first', 'warn'); return; }
  const p = progPositions()[S.progSel];
  const speed = +seg.speed > 0 ? +seg.speed : 1;
  const video = progVideo();
  let at = null;
  if (video && p && video.currentTime > p.start && video.currentTime < p.end) {
    at = (+seg.in || 0) + (video.currentTime - p.start) * speed;
  } else if (srcVideo()) {
    at = srcVideo().currentTime;
  }
  if (at === null || at <= +seg.in + 0.05 || at >= +seg.out - 0.05) {
    toast('put the playhead inside the segment first', 'warn');
    return;
  }
  const copy = JSON.parse(JSON.stringify(seg));
  copy.id = nextId(S.timeline.tracks.video, 's');
  copy.in = round3(at);
  copy.transition_in = { type: 'cut', duration: 0, name: 'fade' };
  seg.out = round3(at);
  S.timeline.tracks.video.splice(S.progSel + 1, 0, copy);
  markDirty();
  renderProgStrip();
  renderProgInspector();
  toast(`split at ${tc(at)}`, 'ok', 2000);
}

/* ------------------------------------------------------------- inspector */
function renderProgInspector() {
  const box = $('#prog-inspector');
  if (!box) return;
  destroyProgWave();
  const seg = selectedSegment();
  if (!seg) {
    fill(box, el('div', { class: 'empty' },
      'Click a block in the strip to trim it, change its role or delete it.'));
    return;
  }
  const clip = (S.state?.clips || []).find((c) => c.id === seg.clip);
  const clipIds = (S.state?.clips || []).map((c) => c.id);
  S.progLoop = { in: +seg.in || 0, out: +seg.out || 0 };

  fill(box,
    el('div', { class: 'insp-head' },
      el('b', { class: 'mono' }, seg.id || `s${S.progSel + 1}`),
      el('span', { class: 'grow' }),
      el('span', { class: 'mono faint', id: 'insp-dur' },
        `${Math.max(0, seg.out - seg.in).toFixed(2)}s`)),

    el('div', { class: 'row wrap' },
      select(clipIds, seg.clip, (v) => { seg.clip = v; progTouch(); renderProgInspector(); }, 'clip'),
      el('a', {
        href: '#', class: 'faint',
        title: 'Open this source in the clip view (waveform, transcript, mute tool)',
        onclick: (e) => { e.preventDefault(); openSourceInClipView(seg.clip); },
      }, 'open source in clip view →')),

    el('div', { class: 'insp-grid' },
      el('div', { class: 'field wide' }, 'in',
        nudge(-0.5, () => bumpEdge('in', -0.5)), nudge(-0.1, () => bumpEdge('in', -0.1)),
        el('input', {
          type: 'number', step: '0.05', id: 'insp-in', value: round3(+seg.in || 0),
          oninput: (e) => { seg.in = num(e.target.value, 0); progTouch(); },
        }),
        nudge(0.1, () => bumpEdge('in', 0.1)), nudge(0.5, () => bumpEdge('in', 0.5))),
      el('div', { class: 'field wide' }, 'out',
        nudge(-0.5, () => bumpEdge('out', -0.5)), nudge(-0.1, () => bumpEdge('out', -0.1)),
        el('input', {
          type: 'number', step: '0.05', id: 'insp-out', value: round3(+seg.out || 0),
          oninput: (e) => { seg.out = num(e.target.value, 0); progTouch(); },
        }),
        nudge(0.1, () => bumpEdge('out', 0.1)), nudge(0.5, () => bumpEdge('out', 0.5))),
      field('role', select(ROLES, seg.role || '', (v) => { seg.role = v; progTouch(); }, 'role')),
      field('fit', select(['cover', 'contain', 'blur-fill', 'crop-pan'],
        seg.transform?.fit || 'cover', (v) => {
          seg.transform = Object.assign({ fit: 'cover', zoom: 1 }, seg.transform, { fit: v });
          markDirty();
        })),
      field('transition', select(['cut', 'fade', 'xfade'], seg.transition_in?.type || 'cut', (v) => {
        seg.transition_in = Object.assign({ type: 'cut', duration: 0, name: 'fade' },
          seg.transition_in,
          { type: v, duration: v === 'cut' ? 0 : (seg.transition_in?.duration || 0.5) });
        progTouch(); renderProgInspector();
      })),
      field('t-dur', numberInput(seg.transition_in?.duration ?? 0, (v) => {
        seg.transition_in = Object.assign({ type: 'cut', duration: 0, name: 'fade' },
          seg.transition_in, { duration: v });
        progTouch();
      }, 0.05)),
      field('gain dB', numberInput(seg.source_audio_gain_db ?? 0, (v) => {
        seg.source_audio_gain_db = v; markDirty();
      }, 1)),
      el('label', { class: 'field' },
        el('input', {
          type: 'checkbox', checked: !!seg.mute_source,
          onchange: (e) => { seg.mute_source = e.target.checked; progTouch(); },
        }), 'mute source'),
      seg.audio_from
        ? el('div', { class: 'field', title: 'overlay cutaway — audio comes from another clip' },
          '🎞 ', audioFromLabel(seg))
        : null,
      el('div', { class: 'field wide' }, 'notes',
        textInput(seg.notes, (v) => { seg.notes = v; markDirty(); }, 'why this cut is here'))),

    el('div', { class: 'row wrap insp-actions' },
      el('button', { class: 'small', onclick: splitSegment, title: 'Split at the playhead' }, '✂ Split'),
      el('button', { class: 'small', onclick: () => moveSegment(S.progSel, S.progSel - 1), disabled: S.progSel === 0 }, '↑ up'),
      el('button', {
        class: 'small', onclick: () => moveSegment(S.progSel, S.progSel + 1),
        disabled: S.progSel >= (S.timeline.tracks.video.length - 1),
      }, '↓ down'),
      el('span', { class: 'grow' }),
      el('button', { class: 'small danger', onclick: deleteSegment }, '✕ Delete')),

    el('h4', {}, 'Source — the cut loops'),
    clip?.proxy_url
      ? el('video', {
        id: 'src-video', src: clip.proxy_url, controls: true, playsinline: true,
        preload: 'metadata', class: 'src-video',
      })
      : el('div', { class: 'empty' }, 'no proxy for this clip — run Ingest'),
    el('div', { class: 'row wrap' },
      el('button', { class: 'small', onclick: () => setEdgeFromSource('in'), title: '[' }, 'Set in = playhead'),
      el('button', { class: 'small', onclick: () => setEdgeFromSource('out'), title: ']' }, 'Set out = playhead'),
      el('span', { class: 'grow' }),
      el('button', {
        class: 'small ghost', title: 'replay the cut from its start',
        onclick: () => { const v = srcVideo(); if (v) { v.currentTime = +seg.in || 0; v.play().catch(() => {}); } },
      }, '↻ replay')),
    el('div', { class: 'insp-wave', id: 'insp-wave' }),
    el('div', { class: 'faint hint-line' },
      'Cuts snap ~0.3 s before / ~0.5 s after speech — run “Tidy cuts” to apply that to every cut.'),
    el('div', { class: 'insp-words', id: 'insp-words' },
      el('span', { class: 'faint' }, 'loading transcript…')));

  mountSourcePlayer(seg, clip);
  loadInspectorTranscript(seg.clip);
}

const nudge = (delta, onclick) => el('button', {
  class: 'small ghost nudge', onclick, title: `${delta > 0 ? '+' : ''}${delta}s`,
}, `${delta > 0 ? '+' : ''}${delta}`);

function bumpEdge(edge, delta) {
  const seg = selectedSegment();
  if (!seg) return;
  const value = round3(Math.max(0, (+seg[edge] || 0) + delta));
  if (edge === 'in' && value >= +seg.out - 0.05) return;
  if (edge === 'out' && value <= +seg.in + 0.05) return;
  seg[edge] = value;
  progTouch();
  const v = srcVideo();
  if (v) v.currentTime = value;
}

function setEdgeFromSource(edge) {
  const seg = selectedSegment();
  const v = srcVideo();
  if (!seg || !v) { toast('select a segment first', 'warn'); return; }
  const t = round3(v.currentTime);
  if (edge === 'in' && t >= +seg.out - 0.05) { toast('in must stay before out', 'warn'); return; }
  if (edge === 'out' && t <= +seg.in + 0.05) { toast('out must stay after in', 'warn'); return; }
  seg[edge] = t;
  progTouch();
  toast(`${edge} = ${tc(t)}`, 'ok', 1400);
}

/** Loop the mini source player over [in, out] so the cut can be heard. */
function mountSourcePlayer(seg, clip) {
  const v = srcVideo();
  if (!v) return;
  v.addEventListener('loadedmetadata', () => { v.currentTime = S.progLoop?.in ?? 0; });
  v.addEventListener('timeupdate', () => {
    const loop = S.progLoop;
    if (!loop || v.paused) return;
    if (v.currentTime >= loop.out || v.currentTime < loop.in - 0.4) v.currentTime = loop.in;
  });
  if (clip?.peaks_url) buildInspectorWave(clip, v);
}

/** Waveform of the source clip with the cut as a draggable region. */
async function buildInspectorWave(clip, media) {
  const box = $('#insp-wave');
  if (!box || typeof WaveSurfer === 'undefined') return;
  const data = await loadPeaks(clip.peaks_url);
  if (!data || $('#insp-wave') !== box) return;
  const { tops, bottoms, duration } = data;
  try {
    S.progWave = WaveSurfer.create({
      container: box, media, peaks: [tops, bottoms],
      duration: duration || clip.duration || 0,
      height: 56, waveColor: '#3d4f66', progressColor: '#4ea3ff',
      cursorColor: '#fff', cursorWidth: 1, normalize: true, interact: true,
    });
    S.progRegions = S.progWave.registerPlugin(WaveSurfer.Regions.create());
    const seg = selectedSegment();
    if (seg) {
      S.progRegions.addRegion({
        id: 'cut', start: +seg.in || 0, end: +seg.out || 0, drag: true, resize: true,
        color: 'rgba(78,163,255,0.22)', content: 'cut',
      });
    }
    for (const m of (S.timeline?.mute_ranges || []).filter((x) => x.clip === clip.id)) {
      S.progRegions.addRegion({
        id: `imute:${m.s}`, start: m.s, end: m.e, drag: false, resize: false,
        color: 'rgba(255,92,92,0.24)',
        content: m.gain_db <= -40 ? 'mute' : `${m.gain_db} dB`,
      });
    }
    S.progRegions.on('region-updated', (r) => {
      const current = selectedSegment();
      if (r.id !== 'cut' || !current) return;
      current.in = round3(r.start);
      current.out = round3(r.end);
      progTouch();
    });
  } catch (e) {
    console.warn('inspector waveform failed', e);
  }
}

function destroyProgWave() {
  if (S.progWave) { try { S.progWave.destroy(); } catch { /* ignore */ } }
  S.progWave = null;
  S.progRegions = null;
}

/** Peaks JSON -> the channel pair wavesurfer wants. Shared by both waveforms. */
async function loadPeaks(url) {
  let data;
  try { data = await (await fetch(url)).json(); } catch { return null; }
  const flat = data.peaks || [];
  const n = Math.floor(flat.length / 2);
  const tops = new Float32Array(n), bottoms = new Float32Array(n);
  for (let i = 0; i < n; i++) { bottoms[i] = flat[i * 2]; tops[i] = flat[i * 2 + 1]; }
  return { tops, bottoms, duration: data.duration || 0 };
}

async function loadInspectorTranscript(clipId) {
  const box = $('#insp-words');
  if (!box) return;
  if (S.trCache[clipId] === undefined) {
    S.trCache[clipId] = await api(`/api/p/${S.slug}/transcript/${clipId}`, { quiet: true })
      .catch(() => null);
  }
  if ($('#insp-words') !== box) return;              // selection moved on
  const tr = S.trCache[clipId];
  const words = tr?.words || [];
  if (!words.length) {
    fill(box, el('span', { class: 'faint' }, 'no transcript for this clip'));
    return;
  }
  fill(box, ...words.map((w) => {
    const s = +(w.s ?? w.start ?? 0), e = +(w.e ?? w.end ?? s);
    return el('span', {
      class: 'w', dataset: { s, e },
      title: `${tc(s)} — click: in · shift-click: out`,
      onclick: (ev) => {
        const seg = selectedSegment();
        if (!seg) return;
        if (ev.shiftKey) seg.out = round3(e); else seg.in = round3(s);
        progTouch();
        const v = srcVideo();
        if (v) v.currentTime = ev.shiftKey ? Math.max(0, e - 1) : s;
      },
    }, (w.t ?? w.text ?? w.word ?? '') + ' ');
  }));
  highlightSourceWords();
}

/** Grey out the words the current in/out excludes. */
function highlightSourceWords() {
  const seg = selectedSegment();
  if (!seg) return;
  for (const node of $$('#insp-words .w')) {
    const s = +node.dataset.s, e = +node.dataset.e;
    node.classList.toggle('in-range', e > +seg.in && s < +seg.out);
  }
}

function openSourceInClipView(clipId) {
  setTab('advanced');
  selectClip(clipId);
}

async function saveAndPreview() {
  if (await saveTimeline()) startJob('render_preview');
}

/* ------------------------------------------------------------------ 9 */
function setTab(name) {
  S.tab = name;
  $$('#tabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  $$('.panel').forEach((p) => p.classList.toggle('active', p.dataset.panel === name));
  // The program view needs the whole window: the clip column and the source
  // player belong to the "Clips" workflow and only get in the way here.
  $('#editor').classList.toggle('program-mode', name === 'program');
  if (name !== 'program') { stopPlayheadLoop(); destroyProgWave(); }
  renderTabHeader();
}

function renderTabHeader() {
  if (!S.state) return;
  if (S.tab === 'program') renderProgram();
  else if (S.tab === 'plan') renderPlan();
  else if (S.tab === 'footage') renderFootage();
  else if (S.tab === 'preview') renderPreview();
  else if (S.tab === 'qc') renderQC();
  else if (S.tab === 'publish') renderPublish();
}

async function renderPlan() {
  const panel = $('#panel-plan');
  fill(panel, el('div', { class: 'empty' }, 'loading…'));
  try {
    const d = await api(`/api/p/${S.slug}/plan`, { quiet: true });
    fill(panel, 
      el('div', { class: 'toolbar' },
        el('b', {}, 'Edit plan'),
        el('span', { class: 'grow' }),
        el('button', { class: 'small ghost', onclick: () => renderPlan() }, '⟳')),
      el('div', { class: 'md', html: md2html(d.edit_plan_md || '') }),
      el('h4', {}, 'Narration requests'),
      el('div', { class: 'md', html: md2html(d.narration_requests_md || '_none_') }),
      d.edit_plan ? el('details', {}, el('summary', { class: 'faint' }, 'raw edit_plan.json'),
        el('pre', { class: 'mono' }, JSON.stringify(d.edit_plan, null, 2))) : null);
  } catch (e) {
    fill(panel, el('div', { class: 'empty' }, e.message));
  }
}

async function renderFootage() {
  const panel = $('#panel-footage');
  fill(panel, el('div', { class: 'empty' }, 'loading…'));
  try {
    const d = await api(`/api/p/${S.slug}/footage_log`, { quiet: true });
    const rows = Array.isArray(d) ? d : (d.clips || d.entries || []);
    fill(panel, el('table', { class: 'data' },
      el('thead', {}, el('tr', {}, ...['clip', 'kind', 'location', 'summary', 'topics']
        .map((h) => el('th', {}, h)))),
      el('tbody', {}, ...rows.map((r) => el('tr', {
        onclick: () => r.clip && selectClip(r.clip),
      },
        el('td', { class: 'mono' }, r.clip || r.id || ''),
        el('td', {}, r.kind || ''),
        el('td', {}, [r.location?.name, r.location?.city].filter(Boolean).join(', ')),
        el('td', {}, r.summary || ''),
        el('td', { class: 'faint' }, (r.topics || []).join(', ')))))));
  } catch (e) {
    fill(panel, el('div', { class: 'empty' }, e.message));
  }
}

function renderPreview() {
  const panel = $('#panel-preview');
  const files = S.state?.files || {};
  const kids = [el('div', { class: 'toolbar' },
    el('b', {}, 'Renders'),
    el('span', { class: 'grow' }),
    el('button', { class: 'small', onclick: () => startJob('render_preview') }, 'Render preview'),
    el('button', { class: 'small', onclick: () => startJob('render_master') }, 'Render master'))];
  if (files.preview_url) {
    kids.push(el('h4', {}, `Preview${files.preview_mtime < files.timeline_mtime ? ' — stale' : ''}`),
      el('video', { src: files.preview_url, controls: true, style: 'width:100%;background:#000' }));
  }
  if (files.master_url) {
    kids.push(el('h4', {}, `Master — ${files.master}`),
      el('video', { src: files.master_url, controls: true, style: 'width:100%;background:#000' }));
  }
  if (!files.preview_url && !files.master_url) {
    kids.push(el('div', { class: 'empty' }, 'Nothing rendered yet.'));
  }
  fill(panel, ...kids);
}

async function renderQC() {
  const panel = $('#panel-qc');
  fill(panel, el('div', { class: 'empty' }, 'loading…'));
  try {
    const d = await api(`/api/p/${S.slug}/qc`, { quiet: true });
    fill(panel, 
      el('div', { class: 'toolbar' }, el('b', {}, 'QC report'), el('span', { class: 'grow' }),
        el('button', { class: 'small', onclick: () => startJob('qc') }, 'Run QC')),
      el('div', { class: 'md', html: md2html(d.markdown || '') }),
      d.report ? el('details', {}, el('summary', { class: 'faint' }, 'raw qc_report.json'),
        el('pre', { class: 'mono' }, JSON.stringify(d.report, null, 2))) : null);
  } catch (e) {
    fill(panel, 
      el('div', { class: 'toolbar' }, el('b', {}, 'QC report'), el('span', { class: 'grow' }),
        el('button', { class: 'small', onclick: () => startJob('qc') }, 'Run QC')),
      el('div', { class: 'empty' }, e.message));
  }
}

async function renderPublish() {
  const panel = $('#panel-publish');
  fill(panel, el('div', { class: 'empty' }, 'loading…'));
  const bar = el('div', { class: 'toolbar' }, el('b', {}, 'Publish pack'), el('span', { class: 'grow' }),
    el('button', { class: 'small', onclick: () => startJob('publish') }, 'Run Publish'));
  try {
    const d = await api(`/api/p/${S.slug}/publish`, { quiet: true });
    const p = d.publish || {};
    const titles = p.titles || p.title_candidates || [];
    const description = p.description || '';
    const chapters = p.chapters || [];
    const descBox = el('textarea', { rows: '10' });
    descBox.value = description;
    fill(panel, bar,
      el('h4', {}, 'Titles'),
      titles.length ? el('div', {}, ...titles.map((t) => {
        const text = typeof t === 'string' ? t : (t.title || '');
        return el('div', { class: 'list-row' },
          el('span', { class: 'grow' }, text),
          el('span', { class: 'faint mono' }, `${text.length}c`),
          el('button', { class: 'small', onclick: () => copy(text) }, 'copy'));
      })) : el('div', { class: 'faint' }, 'none'),
      el('h4', {}, 'Description'),
      descBox,
      el('div', { class: 'row' }, el('button', { class: 'small', onclick: () => copy(descBox.value) }, 'copy description')),
      el('h4', {}, 'Chapters'),
      chapters.length ? el('table', { class: 'data' }, el('tbody', {},
        ...chapters.map((c) => el('tr', {}, el('td', { class: 'mono' }, tc(c.at, 0)), el('td', {}, c.title)))))
        : el('div', { class: 'faint' }, 'none'),
      el('h4', {}, 'Thumbnails'),
      d.thumbnails?.length ? el('div', { class: 'thumbs' }, ...d.thumbnails.map((t) =>
        el('figure', {}, el('img', { src: t.url, loading: 'lazy', alt: t.name }),
          el('figcaption', {}, t.name + (t.preview ? ' (120px check)' : ''))))) : el('div', { class: 'faint' }, 'none'),
      d.markdown ? el('details', {}, el('summary', { class: 'faint' }, 'publish.md'),
        el('div', { class: 'md', html: md2html(d.markdown) })) : null);
  } catch (e) {
    fill(panel, bar, el('div', { class: 'empty' }, e.message));
  }
}

function copy(text) {
  navigator.clipboard.writeText(text).then(() => toast('copied', 'ok', 1500),
    () => toast('clipboard blocked', 'err'));
}

/* ----------------------------------------------------------------- 10 */
async function startJob(stage, args = {}) {
  try {
    const job = await api(`/api/p/${S.slug}/jobs`, { method: 'POST', body: { stage, args } });
    S.openJob = job.id;
    $('#jobbar').classList.remove('collapsed');
    toast(`${stage} started${args.force ? ' (--force)' : ''}`, 'ok', 2500);
    pollJobs();
    refreshState({ soft: true });
  } catch { /* toasted */ }
}

async function pollJobs() {
  clearTimeout(S.jobTimer);
  if (!S.slug) return;
  let data;
  try {
    data = await api(`/api/p/${S.slug}/jobs`, { quiet: true });
  } catch { S.jobTimer = setTimeout(pollJobs, 5000); return; }

  const jobs = data.jobs || [];
  const running = data.running;
  const last = jobs[jobs.length - 1];
  const shown = jobs.find((j) => j.id === S.openJob) || running || last;

  $('#job-title').textContent = shown ? `${shown.stage}` : 'Jobs';
  $('#job-status').textContent = shown
    ? `${shown.status}${shown.returncode !== null && shown.returncode !== undefined ? ' (rc ' + shown.returncode + ')' : ''}`
    : 'idle';
  $('#job-status').className = `mono status-${shown?.status || 'idle'}`;
  $('#btn-cancel-job').hidden = !running;
  $('#btn-cancel-job').onclick = () => { if (running) cancelJob(running.id); };

  const prog = shown?.progress;
  const bar = $('#job-progress');
  if (prog && typeof prog.percent === 'number') {
    bar.hidden = false;
    $('i', bar).style.width = `${Math.max(0, Math.min(100, prog.percent))}%`;
    bar.title = `${prog.step || ''} ${prog.percent.toFixed(0)}%${prog.eta ? ' eta ' + prog.eta : ''}`;
  } else bar.hidden = true;

  fill($('#jobs-list'), ...jobs.slice(-12).reverse().map((j) => el('div', {
    class: 'job-row', onclick: () => { S.openJob = j.id; pollJobs(); },
  },
    el('span', { class: `mono status-${j.status}` }, j.status.padEnd(9)),
    el('span', { class: 'mono' }, j.stage),
    el('span', { class: 'faint' }, (j.started || '').replace('T', ' ').slice(5, 16)),
    j.id === shown?.id ? el('span', { class: 'badge' }, 'shown') : null)));

  if (shown && shown.log_tail && shown.log_tail.length) {
    $('#job-log').textContent = shown.log_tail.join('\n');
  } else if (shown) {
    try {
      const full = await api(`/api/p/${S.slug}/jobs/${shown.id}`, { quiet: true });
      $('#job-log').textContent = full.log || '(no output)';
    } catch { $('#job-log').textContent = ''; }
  }
  const log = $('#job-log');
  if (running) log.scrollTop = log.scrollHeight;

  if (running) {
    S.jobTimer = setTimeout(pollJobs, 2000);
  } else if (S._wasRunning) {
    S._wasRunning = false;
    toast(`${last?.stage || 'job'} ${last?.status || 'finished'}`,
      last?.status === 'done' ? 'ok' : 'err', 6000);
    await refreshState();
    await loadTimeline();
    renderTabHeader();
    S.jobTimer = setTimeout(pollJobs, 6000);
  } else {
    S.jobTimer = setTimeout(pollJobs, 6000);
  }
  if (running) S._wasRunning = true;
}

async function cancelJob(id) {
  try {
    await api(`/api/p/${S.slug}/jobs/${id}/cancel`, { method: 'POST' });
    toast('cancel requested', 'warn');
    pollJobs();
  } catch { /* toasted */ }
}

/* ----------------------------------------------------------------- 11 */
function wireChrome() {
  $('#btn-refresh').onclick = () => { refreshState(); loadTimeline(); };
  $('#btn-help').onclick = () => $('#dlg-help').showModal();
  $('#help-close').onclick = () => $('#dlg-help').close();
  $('#btn-new-project').onclick = () => $('#dlg-new').showModal();
  $('#np-cancel').onclick = () => $('#dlg-new').close();
  $('#np-create').onclick = async () => {
    const slug = $('#np-slug').value.trim();
    try {
      await api('/api/projects', {
        method: 'POST',
        body: { slug, title: $('#np-title').value.trim() || null, language: $('#np-language').value.trim() || 'pl' },
      });
      location.href = `/p/${encodeURIComponent(slug)}`;
    } catch { /* toasted */ }
  };
  $('#hide-excluded').onchange = (e) => { S.hideExcluded = e.target.checked; renderClips(); };
  $('#clip-notes').onblur = (e) => {
    if (S.clip && (S.clip.notes || '') !== e.target.value) saveClip({ notes: e.target.value });
  };
  $('#clip-exclude').onchange = (e) => saveClip({ exclude: e.target.checked });
  $('#clip-kind').onchange = (e) => saveClip({ kind_override: e.target.value || null });

  $('#btn-sel-mute').onclick = (ev) => openSelectionPopover(ev);
  $('#btn-sel-duck').onclick = () => applySelectionMute(-12, 'duck');
  $('#btn-sel-add').onclick = addSegmentFromSelection;
  $('#btn-sel-clear').onclick = clearSelection;

  $$('#tabs button').forEach((b) => { b.onclick = () => setTab(b.dataset.tab); });

  const banner = $('#onboard');
  if (banner) {
    let dismissed = false;
    try { dismissed = localStorage.getItem('ytedit.onboard') === 'off'; } catch { /* private mode */ }
    banner.classList.toggle('hidden', dismissed);
    $('#onboard-close').onclick = () => {
      banner.classList.add('hidden');
      try { localStorage.setItem('ytedit.onboard', 'off'); } catch { /* ignore */ }
    };
  }

  $('#jobbar-header').onclick = (e) => {
    if (e.target.tagName === 'BUTTON') return;
    const bar = $('#jobbar');
    bar.classList.toggle('collapsed');
    $('#jobbar-toggle').textContent = bar.classList.contains('collapsed') ? '▲' : '▼';
  };
  document.addEventListener('click', (e) => {
    if (e === S._popEvent) return;              // the click that opened it
    if (!$('#popover').contains(e.target)) hidePopover();
  });
  window.addEventListener('beforeunload', (e) => {
    if (S.dirty) { e.preventDefault(); e.returnValue = ''; }
  });
  document.addEventListener('keydown', onKey);
}

function onKey(e) {
  const t = e.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) {
    if (e.key === 'Escape') t.blur();
    return;
  }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (S.tab === 'program' && onProgramKey(e)) return;
  const v = $('#video');
  const clips = S.state?.clips || [];
  const idx = clips.findIndex((c) => c.id === S.clip?.id);
  switch (e.key) {
    case ' ':
      e.preventDefault();
      if (v.paused) v.play().catch(() => {}); else v.pause();
      break;
    case 'i': case 'I': e.preventDefault(); setSelectionEdge('start', v.currentTime); break;
    case 'o': case 'O': e.preventDefault(); setSelectionEdge('end', v.currentTime); break;
    case 'm': case 'M': e.preventDefault(); applySelectionMute(-60, 'marked in the editor'); break;
    case 's': case 'S': e.preventDefault(); saveTimeline(); break;
    case 'r': case 'R': e.preventDefault(); refreshState(); loadTimeline(); break;
    case 'Escape': clearSelection(); break;
    case 'ArrowLeft': e.preventDefault(); v.currentTime = Math.max(0, v.currentTime - (e.shiftKey ? 5 : 1)); break;
    case 'ArrowRight': e.preventDefault(); v.currentTime += (e.shiftKey ? 5 : 1); break;
    case 'ArrowUp': e.preventDefault(); if (idx > 0) selectClip(clips[idx - 1].id); break;
    case 'ArrowDown': e.preventDefault(); if (idx >= 0 && idx < clips.length - 1) selectClip(clips[idx + 1].id); break;
    default: break;
  }
}

/** Program-view keys. Returns true when the event was consumed. */
function onProgramKey(e) {
  const v = progVideo();
  switch (e.key) {
    case ' ':
      if (!v) return false;
      e.preventDefault();
      if (v.paused) v.play().catch(() => {}); else v.pause();
      return true;
    case 'ArrowLeft':
      if (!v) return false;
      e.preventDefault();
      v.currentTime = Math.max(0, v.currentTime - (e.shiftKey ? 1 : 0.1));
      movePlayhead();
      return true;
    case 'ArrowRight':
      if (!v) return false;
      e.preventDefault();
      v.currentTime += (e.shiftKey ? 1 : 0.1);
      movePlayhead();
      return true;
    case '[': e.preventDefault(); setEdgeFromSource('in'); return true;
    case ']': e.preventDefault(); setEdgeFromSource('out'); return true;
    case 'Delete': case 'Backspace': e.preventDefault(); deleteSegment(); return true;
    case 's': case 'S': e.preventDefault(); saveTimeline(); return true;
    default: return false;
  }
}

/** I/O keys: extend or create the waveform selection at the playhead. */
function setSelectionEdge(edge, t) {
  if (!S.regions) return;
  let start = S.sel ? S.sel.start : t;
  let end = S.sel ? S.sel.end : t + 1;
  if (edge === 'start') start = t; else end = t;
  if (end <= start) { if (edge === 'start') end = start + 0.5; else start = Math.max(0, end - 0.5); }
  if (S.sel && S.sel.region) { S.sel.region.setOptions({ start, end }); setSelection(S.sel.region); }
  else {
    const region = S.regions.addRegion({ start, end, color: 'rgba(78,163,255,0.22)', drag: true, resize: true });
    setSelection(region);
  }
  toast(`selection ${tc(start)} → ${tc(end)}`, '', 1200);
}

document.addEventListener('DOMContentLoaded', boot);
