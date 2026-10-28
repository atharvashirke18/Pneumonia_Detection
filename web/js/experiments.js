/* experiments.js — Experiment Results page.
   One shared selection (a validation case) drives every chart and a single
   inspector panel, so the page shows 3–4 images instead of 100+. */
'use strict';

const $ = id => document.getElementById(id);
const num = v => { const n = parseFloat(v); return Number.isFinite(n) ? n : NaN; };
const jget = async u => { try { const r = await fetch(u); return r.ok ? await r.json() : null; } catch (_) { return null; } };
const fmt = (v, d = 3) => XC.fmt(v, d);

const S = { cases: [], byIdx: {}, sel: null, tab: '01', mode: { '2a': 'cam', '2b': 'pert' }, sc: 'ups', ratio: 'ups', order: 'case', cond: 'border_black', train: 'loss' };
const D = {};
const PANES = {};

/* ---------- small UI helpers ---------- */
function countUp(el, to, { d = 0, suffix = '', ms = 900 } = {}) {
  if (!el) return;
  const t0 = performance.now(), f = t => { const k = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - k, 3); el.textContent = (to * e).toFixed(d) + suffix; if (k < 1) requestAnimationFrame(f); };
  requestAnimationFrame(f);
}
function runCounters(root) {
  root.querySelectorAll('[data-count]').forEach(el => countUp(el, +el.dataset.count, { d: +(el.dataset.d || 0), suffix: el.dataset.suffix || '' }));
}
function empty(el, msg) { el.classList.remove('loading'); el.innerHTML = `<div class="empty"><b>Not available</b><span>${msg}</span></div>`; }
function seg(id, onPick) {
  const root = $(id);
  root.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    root.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
    onPick(b.dataset.v);
  });
}
const onPoint = e => { const i = e && e.data && e.data.idx; if (i != null) select(i); };
const kpi = (label, val, sub, color, extra = '') => `<div class="kpi" style="--kc:${color}"><div class="l">${label}</div><div class="v">${val}</div><div class="s">${sub}</div>${extra}</div>`;

/* ---------- data ---------- */
async function loadAll() {
  const [hist, m1, meta, audit, auditMeta, sSum, sRows] = await Promise.all([
    jget('/api/experiments/exp01/training-history'), jget('/api/experiments/exp01/metrics'),
    jget('/api/experiments/exp02a/metadata'), jget('/api/experiments/exp02a/numerical-audit'), jget('/api/experiments/exp02a/audit-meta'),
    jget('/api/experiments/exp02b/summary'), jget('/api/experiments/exp02b/results'),
  ]);
  Object.assign(D, { hist, m1, meta, audit, auditMeta, sSum, sRows });
  const numRow = r => { const o = {}; for (const k in r) o[k] = num(r[k]); return o; };
  const A = {}, R = {};
  (audit || []).forEach(r => { A[parseInt(r.ds_idx)] = numRow(r); });
  (sRows || []).forEach(r => { R[parseInt(r.ds_idx)] = numRow(r); });
  S.cases = ((meta && meta.examples) || []).map(ex => {
    const a = Object.assign({}, A[ex.ds_idx] || {});
    if (!Number.isFinite(a.stored_border_mean)) a.stored_border_mean = ex.border_mean_activation;
    if (!Number.isFinite(a.stored_inner_mean)) a.stored_inner_mean = ex.inner_mean_activation;
    if (!Number.isFinite(a.stored_ratio)) a.stored_ratio = ex.border_to_inner_ratio;
    if (a.nat1_inner_mean === 0) a.nat1_ratio = NaN;
    return { idx: ex.ds_idx, stem: ex.stem, outcome: ex.outcome, gt: ex.gt_class, pred: ex.pred_class, p: ex.pneumonia_prob, files: ex.files, hm: ex.heatmap_mean, a, hasNat: Number.isFinite(a.nat1_border_mean), s: R[ex.ds_idx] || null };
  }).sort((x, y) => x.idx - y.idx);
  S.cases.forEach(c => { S.byIdx[c.idx] = c; });
}

/* ---------- selection + inspector ---------- */
const camUrl = (c, k) => `/api/experiments/exp02a/image/${k}s/${c.files[k].split('/').pop()}`;
const pertUrl = (c, k) => `/api/experiments/exp02b/image/${c.idx}/${k}`;
const natUrl = c => `/api/experiments/exp02a/comparison/${c.stem}_comparison.png`;
const signed = v => (v >= 0 ? '+' : '') + fmt(v);
const insp = { el: null, mode: null, tilesMode: null, mini: null };

function select(idx) {
  if (!S.byIdx[idx]) return;
  S.sel = +idx;
  if (PANES[S.tab] && PANES[S.tab].insp) history.replaceState(null, '', `#${S.tab}/${S.sel}`);
  renderInspector();
  const P = PANES[S.tab]; if (P && P.built && P.onSel) P.onSel();
}
function step(d) {
  const i = S.cases.findIndex(c => c.idx === S.sel);
  select(S.cases[(i + d + S.cases.length) % S.cases.length].idx);
}

function buildTiles(mode) {
  if (insp.mini) { insp.mini.dispose(); insp.mini = null; }
  const tile = (k, label) => `<div class="tile loading" data-k="${k}"><img alt="${label}"/><div class="cap"><span>${label}</span><span class="p"></span></div><div class="na">Image unavailable</div></div>`;
  $('i-tiles').innerHTML = mode === 'cam'
    ? tile('original', 'Original') + tile('heatmap', 'Grad-CAM heatmap') + tile('overlay', 'Overlay') + `<div class="tile chartTile"><span class="ttl">Activation</span><div class="chart" id="i-mini"></div></div>`
    : mode === 'pert'
      ? tile('original', 'Original') + XC.CONDS.map(k => tile(k.key, k.name)).join('')
      : tile('nat', 'Native 7×7 vs upsampled').replace('class="tile loading"', 'class="tile wide loading"');
  insp.tilesMode = mode;
}
function setImg(t, src) {
  const im = t.querySelector('img'); if (im.dataset.src === src) return;
  im.dataset.src = src; im.classList.remove('ok'); t.classList.add('loading'); t.classList.remove('err');
  im.onload = () => { im.classList.add('ok'); t.classList.remove('loading'); };
  im.onerror = () => { t.classList.remove('loading'); t.classList.add('err'); };
  im.src = src;
}
function renderInspector() {
  if (!insp.el || insp.el.hidden) return;
  const c = S.byIdx[S.sel]; if (!c) return;
  const mode = insp.mode;
  if (insp.tilesMode !== mode) buildTiles(mode);
  $('i-title').textContent = `Case #${c.idx}`;
  const pill = $('i-pill'); pill.textContent = c.outcome; pill.dataset.o = c.outcome;
  $('i-p').textContent = `P = ${fmt(c.p)} · ${c.gt.toLowerCase()}`;
  $('i-rail').querySelectorAll('.chip').forEach(b => b.classList.toggle('on', +b.dataset.i === c.idx));
  $('i-mode').querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.v === mode));
  const T = k => $('i-tiles').querySelector(`[data-k="${k}"]`);
  let facts = [];
  if (mode === 'cam' || mode === 'nat') {
    if (mode === 'cam') {
      setImg(T('original'), camUrl(c, 'original')); setImg(T('heatmap'), camUrl(c, 'heatmap')); setImg(T('overlay'), camUrl(c, 'overlay'));
      if (!insp.mini) insp.mini = XC.mk($('i-mini'));
      insp.mini.setOption(XC.miniBI(c));
    } else {
      setImg(T('nat'), natUrl(c)); T('nat').querySelector('.na').textContent = 'No native-vs-upsampled comparison was saved for this case.';
    }
    facts = [['Ground truth', c.gt], ['Predicted', c.pred], ['P(pneumonia)', fmt(c.p)], ['Heatmap mean', fmt(c.hm)],
      ['Border mean', fmt(c.a.stored_border_mean, 4)], ['Inner mean', fmt(c.a.stored_inner_mean, 4)],
      ['Ratio · 22 px', XC.fmtRatio(c.a.stored_ratio)], ['Ratio · native', XC.fmtRatio(c.a.nat1_ratio)],
      ['Border energy', c.hasNat ? (c.a.nat1_energy_share * 100).toFixed(1) + '%' : '—'], ['CAM peak (r, c)', c.hasNat ? `(${c.a.nat1_peak_row}, ${c.a.nat1_peak_col})` : '—']];
  } else {
    setImg(T('original'), camUrl(c, 'original')); T('original').querySelector('.p').textContent = c.s ? fmt(c.s.original_prob) : '';
    XC.CONDS.forEach(k => {
      const t = T(k.key), cap = t.querySelector('.cap'); t.querySelector('.flipmark') && t.querySelector('.flipmark').remove();
      t.querySelector('.na').textContent = 'Perturbation results not available for this case.';
      if (!c.s) { t.classList.add('err'); return; }
      setImg(t, pertUrl(c, k.key));
      cap.querySelector('.p').innerHTML = `${fmt(c.s[k.key + '_prob'])} <span class="d">${signed(c.s[k.key + '_delta'])}</span>`;
      if (c.s[k.key + '_flip']) t.insertAdjacentHTML('beforeend', '<span class="flipmark">★ FLIP</span>');
    });
    const fl = c.s ? XC.CONDS.filter(k => c.s[k.key + '_flip']).length : null;
    facts = [['Ground truth', c.gt], ['Baseline P', c.s ? fmt(c.s.original_prob) : '—'],
      ...XC.CONDS.map(k => [k.short + ' ΔP', c.s ? signed(c.s[k.key + '_delta']) : '—']), ['Flips', fl == null ? '—' : `${fl} / 3`]];
  }
  $('i-facts').innerHTML = facts.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');
}
function setMode(m) { insp.mode = m; if (PANES[S.tab]) S.mode[S.tab] = m; renderInspector(); }
function initInspector() {
  insp.el = $('insp');
  $('i-rail').innerHTML = S.cases.map(c => `<button class="chip" data-o="${c.outcome}" data-i="${c.idx}" title="${c.outcome} · P=${fmt(c.p)}">#${c.idx}</button>`).join('');
  $('i-rail').addEventListener('click', e => { const b = e.target.closest('.chip'); if (b) select(+b.dataset.i); });
  $('i-mode').innerHTML = [['cam', 'Grad-CAM'], ['pert', 'Perturbations'], ['nat', 'Native 7×7']].map(([v, l]) => `<button data-v="${v}">${l}</button>`).join('');
  $('i-mode').addEventListener('click', e => { const b = e.target.closest('button'); if (b) setMode(b.dataset.v); });
  $('i-prev').onclick = () => step(-1); $('i-next').onclick = () => step(1);
  const lb = $('lb');
  $('i-tiles').addEventListener('click', e => {
    const t = e.target.closest('.tile'); if (!t || t.classList.contains('chartTile') || t.classList.contains('err')) return;
    const im = t.querySelector('img'); if (!im || !im.classList.contains('ok')) return;
    $('lb-i').src = im.src; $('lb-t').textContent = `Case #${S.sel} · ${t.querySelector('.cap span').textContent}`; lb.showModal();
  });
  $('lb-x').onclick = () => lb.close(); lb.addEventListener('click', e => { if (e.target === lb) lb.close(); });
  document.addEventListener('keydown', e => {
    if (!(PANES[S.tab] && PANES[S.tab].insp) || $('lb').open || /input|select|textarea/i.test(e.target.tagName)) return;
    if (e.key === 'ArrowRight') { step(1); e.preventDefault(); } else if (e.key === 'ArrowLeft') { step(-1); e.preventDefault(); }
  });
}
function mountInspector(tab) {
  const slot = $('slot-' + tab);
  if (!slot) { insp.el.hidden = true; return; }
  slot.appendChild(insp.el); insp.el.hidden = false;
  insp.mode = S.mode[tab]; insp.tilesMode = null; renderInspector();
}

/* ---------- panes ---------- */
PANES['01'] = {
  build() {
    const h = D.hist, m = D.m1;
    if (!h || !m) { ['g-auc', 'g-sens', 'g-spec', 'g-f1', 'c-train', 'c-cm', 'c-traj', 'c-lr'].forEach(i => empty($(i), 'Experiment 1 training history could not be loaded. Is the server running?')); return; }
    [['g-auc', m.val_auc, XC.T.c[0], v => v.toFixed(4)], ['g-sens', m.val_sensitivity, XC.T.c[1], v => (v * 100).toFixed(1) + '%'],
      ['g-spec', m.val_specificity, XC.T.c[2], v => (v * 100).toFixed(1) + '%'], ['g-f1', m.val_f1, XC.T.c[3], v => v.toFixed(4)]]
      .forEach(([id, v, col, f]) => XC.mk($(id)).setOption(XC.gauge(v, col, f)));
    countUp($('k-loss'), m.val_loss, { d: 4 }); countUp($('k-epoch'), m.best_epoch, { d: 0 });
    const tr = XC.mk($('c-train')); tr.setOption(XC.training(h, S.train, m.best_epoch));
    seg('seg-train', v => { S.train = v; tr.setOption(XC.training(h, v, m.best_epoch), true); });
    const cm = D.auditMeta && D.auditMeta.confusion;
    if (cm) XC.mk($('c-cm')).setOption(XC.confusion(cm)); else empty($('c-cm'), 'Confusion counts come from the Experiment 2A audit metadata, which was not found.');
    XC.mk($('c-traj')).setOption(XC.trajectory(h, m.best_epoch));
    XC.mk($('c-lr')).setOption(XC.lrTime(h));
  },
};

PANES['2a'] = {
  insp: true,
  build() {
    const cs = S.cases, n = cs.length;
    if (!n) { ['c-scatter', 'c-ratio', 'c-energy', 'c-peak'].forEach(i => empty($(i), 'Experiment 2A metadata could not be loaded.')); return; }
    const bd = D.auditMeta && D.auditMeta.border_dominance_summary, K = XC.T.c;
    const gt1 = cs.filter(c => c.a.stored_ratio > 1).length, n1 = cs.filter(c => c.a.nat1_ratio > 1).length, u1 = cs.filter(c => c.hasNat && !Number.isFinite(c.a.nat1_ratio)).length;
    const n2 = bd ? bd.native_7x7_2cell_ratio_gt1 : cs.filter(c => c.a.nat2_ratio > 1).length;
    const med = cs.map(c => c.a.stored_ratio).sort((a, b) => a - b), median = (med[(n - 1) >> 1] + med[n >> 1]) / 2;
    const cnt = (v, sfx, d = 0) => `<span data-count="${v}" data-d="${d}" data-suffix="${sfx}">0</span>`;
    $('kpis-2a').innerHTML =
      kpi('Border > inner · 22 px', cnt(bd ? bd.upsampled_22px_ratio_gt1_finite_only : gt1, `/${n}`), 'Every outcome class: TP, TN, FP and FN', K[0]) +
      kpi('Native 1-cell', cnt(bd ? bd.native_7x7_1cell_ratio_gt1 : n1, `/${n}`), `${bd ? bd.native_7x7_1cell_undefined_NaN : u1} more undefined: inner activation is exactly 0`, K[1]) +
      kpi('Native 2-cell', cnt(n2, `/${n}`), 'Remaining cases have zero activation in the 3×3 centre', K[2]) +
      kpi('Median ratio', cnt(median, '×', 2), `Range ${fmt(med[0], 2)}× – ${fmt(med[n - 1], 2)}×`, K[3]) +
      kpi('Zero-CAM cases', cnt(bd ? bd.zero_cam_cases : 0, ''), 'Numerical audit: ε-patch never triggered', K[4]);
    runCounters($('kpis-2a'));
    const hasNat = cs.every(c => c.hasNat);
    if (!hasNat) $('seg-sc').style.display = $('seg-ratio').style.display = 'none';
    const sc = XC.mk($('c-scatter'), onPoint), rt = XC.mk($('c-ratio'), onPoint), en = XC.mk($('c-energy'), onPoint), pk = XC.mk($('c-peak'), onPoint);
    this.ch = { sc, rt, en, pk };
    sc.setOption(XC.scatter(cs, S.sc, S.sel)); rt.setOption(XC.ratioBars(cs, S.ratio, S.sel));
    if (hasNat) { en.setOption(XC.energy(cs, S.sel)); pk.setOption(XC.peakGrid(cs, S.sel)); } else { empty($('c-energy'), 'Native 7×7 audit columns not found.'); empty($('c-peak'), 'Native 7×7 audit columns not found.'); }
    seg('seg-sc', v => { S.sc = v; sc.setOption(XC.scatter(cs, v, S.sel), true); });
    seg('seg-ratio', v => { S.ratio = v; rt.setOption(XC.ratioBars(cs, v, S.sel), true); });
    this.hasNat = hasNat;
  },
  onSel() {
    const { sc, rt, en, pk } = this.ch, cs = S.cases;
    sc.setOption(XC.scatter(cs, S.sc, S.sel)); rt.setOption(XC.ratioBars(cs, S.ratio, S.sel));
    if (this.hasNat) { en.setOption(XC.energy(cs, S.sel)); pk.setOption(XC.peakGrid(cs, S.sel)); }
  },
};

PANES['2b'] = {
  insp: true,
  build() {
    const cs = S.cases, sum = D.sSum;
    if (!sum || !cs.some(c => c.s)) { ['c-heat', 'c-slope', 'c-radar', 'c-strip'].forEach(i => empty($(i), 'Experiment 2B results could not be loaded.')); return; }
    const rows = cs.filter(c => c.s), by = sum.by_condition;
    $('kpis-2b').innerHTML = XC.CONDS.map(k => {
      const b = by[k.key];
      return kpi(k.name, `<span data-count="${b.mean_abs_delta}" data-d="3">0</span>`, `mean |ΔP| · median ${fmt(b.median_abs_delta)} · max ${fmt(b.max_abs_delta)}`, XC.T.k[k.key],
        `<div class="dots">${Array.from({ length: b.n }, (_, i) => `<i class="${i < b.n_flip ? 'on' : ''}"></i>`).join('')}</div><div class="s" style="margin-top:.35rem"><b style="color:var(--kc)">${b.n_flip}</b> of ${b.n} predictions flipped</div>`);
    }).join('') + kpi('Weights unchanged', sum.params_unchanged ? '✓' : '✗', `Σθ before = after = ${fmt(sum.param_sum_before_experiment, 3)}`, XC.T.c[0]);
    runCounters($('kpis-2b'));
    const hm = XC.mk($('c-heat'), onPoint), sl = XC.mk($('c-slope'), onPoint), rd = XC.mk($('c-radar')), st = XC.mk($('c-strip'), onPoint);
    this.ch = { hm, sl, st };
    hm.setOption(XC.sensHeat(cs, S.order, S.sel)); sl.setOption(XC.slope(cs, S.cond, S.sel)); rd.setOption(XC.radar(sum)); st.setOption(XC.strip(cs, sum, S.sel));
    seg('seg-order', v => { S.order = v; hm.setOption(XC.sensHeat(cs, v, S.sel), true); });
    seg('seg-cond', v => { S.cond = v; sl.setOption(XC.slope(cs, v, S.sel), true); });
  },
  onSel() {
    const { hm, sl, st } = this.ch, cs = S.cases;
    hm.setOption(XC.sensHeat(cs, S.order, S.sel)); sl.setOption(XC.slope(cs, S.cond, S.sel)); st.setOption(XC.strip(cs, D.sSum, S.sel));
  },
};

/* ---------- tabs ---------- */
function showTab(t) {
  const P = PANES[t]; if (!P) return;
  S.tab = t;
  document.querySelectorAll('.tab-btn[data-tab]').forEach(b => b.classList.toggle('active', b.dataset.tab === t));
  document.querySelectorAll('.pane').forEach(p => p.classList.toggle('active', p.id === 'pane-' + t));
  mountInspector(t);
  if (!P.built) { P.built = true; P.build(); } else if (P.onSel) P.onSel();
  requestAnimationFrame(() => XC.resizeAll());
  history.replaceState(null, '', `#${t}${P.insp ? '/' + S.sel : ''}`);
}

async function init() {
  XC.initTheme();
  await loadAll();
  initInspector();
  const [t0, i0] = (location.hash.slice(1) || '').split('/');
  S.sel = S.byIdx[i0] ? +i0 : (S.byIdx[241] ? 241 : (S.cases[0] && S.cases[0].idx));
  document.querySelectorAll('.tab-btn[data-tab]').forEach(b => b.addEventListener('click', () => showTab(b.dataset.tab)));
  showTab(PANES[t0] ? t0 : '01');
}
window.addEventListener('DOMContentLoaded', init);
