/* charts.js — ECharts theme + chart builders for the experiments page.
   Every colour is read from the CSS tokens in theme.css (--c1 … --c5), so the
   palette lives in exactly one place. Builders are pure: data in, option out. */
'use strict';

const XC = (() => {
  const T = {};
  const SYM = { TP: 'circle', TN: 'rect', FP: 'triangle', FN: 'diamond' };
  const OUTCOMES = ['TP', 'TN', 'FP', 'FN'];
  const CONDS = [
    { key: 'border_black', name: 'Border black', short: 'Border', cv: '--k-border' },
    { key: 'central_black', name: 'Central black', short: 'Central', cv: '--k-central' },
    { key: 'gamma_dark', name: 'Gamma dark', short: 'Gamma', cv: '--k-gamma' },
  ];

  /* ---------- helpers ---------- */
  const rgba = (hex, a) => {
    const h = hex.replace('#', ''), n = parseInt(h.length === 3 ? h.split('').map(x => x + x).join('') : h, 16);
    return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`;
  };
  const lerpHex = (a, b, t) => {
    const p = h => { const n = parseInt(h.replace('#', ''), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; };
    const A = p(a), B = p(b);
    return '#' + A.map((v, i) => Math.round(v + (B[i] - v) * t).toString(16).padStart(2, '0')).join('');
  };
  const fmt = (v, d = 3) => (v == null || !Number.isFinite(v)) ? '—' : v.toFixed(d);
  const fmtRatio = r => Number.isFinite(r) ? r.toFixed(2) + '×' : 'undef.';
  const tip = (title, rows) =>
    `<div style="font:600 12px ${T.font};margin-bottom:6px">${title}</div>` +
    rows.map(r => `<div style="display:flex;justify-content:space-between;gap:18px;font:12px ${T.mono};line-height:1.75">` +
      `<span style="color:${T.muted}">${r[2] ? `<i style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${r[2]};margin-right:6px"></i>` : ''}${r[0]}</span><b>${r[1]}</b></div>`).join('');
  const richOut = () => Object.fromEntries(OUTCOMES.map(o => [o, { color: T.o[o], fontFamily: T.mono, fontSize: 11 }]));
  const catFmt = v => `{${v.split(' ')[0]}|${v}}`;
  const lbl = c => `${c.outcome} #${c.idx}`;

  function initTheme() {
    const s = getComputedStyle(document.documentElement), g = n => s.getPropertyValue(n).trim();
    Object.assign(T, {
      c: [g('--c1'), g('--c2'), g('--c3'), g('--c4'), g('--c5')],
      text: g('--text'), muted: g('--text-muted'), dim: g('--text-dim'), border: g('--border'), borderL: g('--border-light'),
      bg: g('--bg'), card: g('--card'), font: g('--font'), mono: g('--mono'),
      o: { TP: g('--o-tp'), TN: g('--o-tn'), FP: g('--o-fp'), FN: g('--o-fn') },
      k: Object.fromEntries(CONDS.map(c => [c.key, g(c.cv)])),
    });
    const axis = {
      axisLine: { lineStyle: { color: T.border } }, axisTick: { show: false },
      axisLabel: { color: T.muted, fontFamily: T.mono, fontSize: 11 },
      splitLine: { lineStyle: { color: rgba(T.muted, .11) } },
      nameTextStyle: { color: T.muted, fontFamily: T.font, fontSize: 11 },
    };
    echarts.registerTheme('pneumo', {
      color: T.c, backgroundColor: 'transparent',
      textStyle: { fontFamily: T.font, color: T.muted },
      legend: { textStyle: { color: T.muted, fontFamily: T.font, fontSize: 11 }, icon: 'roundRect', itemWidth: 12, itemHeight: 7, inactiveColor: T.dim },
      tooltip: {
        backgroundColor: '#0D1524', borderColor: T.borderL, borderWidth: 1, padding: [10, 12],
        textStyle: { color: T.text, fontFamily: T.font, fontSize: 12 },
        extraCssText: 'border-radius:10px;box-shadow:0 10px 30px rgba(0,0,0,.55);', confine: true,
      },
      categoryAxis: axis, valueAxis: axis, logAxis: axis,
      toolbox: { iconStyle: { borderColor: T.muted }, emphasis: { iconStyle: { borderColor: T.c[0] } } },
    });
  }

  const reg = new Map();
  function mk(el, onClick) {
    if (reg.has(el)) { reg.get(el).dispose(); }
    el.classList.remove('loading');
    const ch = echarts.init(el, 'pneumo');
    new ResizeObserver(() => ch.resize()).observe(el);
    if (onClick) ch.on('click', onClick);
    reg.set(el, ch);
    return ch;
  }
  const resizeAll = () => reg.forEach(c => { if (!c.isDisposed()) c.resize(); });
  const toolbox = { right: 4, top: 0, itemSize: 13, feature: { saveAsImage: { title: 'PNG', pixelRatio: 2, backgroundColor: '#0F1726' } } };

  /* ======================================================================
     01 · Baseline
     ====================================================================== */
  function gauge(value, color, formatter) {
    return {
      animationDuration: 1600, animationEasing: 'cubicOut',
      series: [{
        type: 'gauge', min: .9, max: 1, startAngle: 205, endAngle: -25, radius: '98%', center: ['50%', '62%'], splitNumber: 5,
        progress: { show: true, width: 12, roundCap: true, itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [{ offset: 0, color: T.c[1] }, { offset: 1, color }]) } },
        axisLine: { roundCap: true, lineStyle: { width: 12, color: [[1, '#1C2740']] } },
        pointer: { show: false }, axisTick: { show: false }, splitLine: { show: false },
        axisLabel: { show: false },
        anchor: { show: false },
        title: { show: false },
        detail: { valueAnimation: true, offsetCenter: [0, '-4%'], fontSize: 24, fontWeight: 600, fontFamily: T.mono, color: T.text, formatter },
        data: [{ value }],
      }],
    };
  }

  const TRAIN_METRICS = {
    loss: { title: 'Loss', series: [['Train', 'train_loss'], ['Validation', 'val_loss']], y: { min: 0 } },
    'loss-log': { title: 'Loss (log)', series: [['Train', 'train_loss'], ['Validation', 'val_loss']], y: { type: 'log' } },
    auc: { title: 'ROC-AUC', series: [['Train', 'train_auc'], ['Validation', 'val_auc']], y: { scale: true } },
    f1: { title: 'F1', series: [['Train', 'train_f1'], ['Validation', 'val_f1']], y: { scale: true } },
    ss: { title: 'Sensitivity / specificity', series: [['Val sensitivity', 'val_sensitivity'], ['Val specificity', 'val_specificity']], y: { scale: true } },
  };
  function training(hist, metric, bestEpoch) {
    const M = TRAIN_METRICS[metric], ep = hist.map(r => String(r.epoch)), bi = hist.findIndex(r => r.epoch === bestEpoch);
    const cols = metric === 'ss' ? [T.c[3], T.c[0]] : [T.c[0], T.c[1]];
    const d = 4;
    return {
      animationDuration: 900, animationEasing: 'cubicOut',
      grid: { left: 58, right: 26, top: 46, bottom: 74 }, toolbox,
      legend: { top: 4, left: 4 },
      tooltip: {
        trigger: 'axis', axisPointer: { type: 'cross', crossStyle: { color: T.borderL }, label: { backgroundColor: '#0D1524', fontFamily: T.mono } },
        formatter: ps => tip(`Epoch ${ps[0].axisValue}${+ps[0].axisValue === bestEpoch ? ' · best (min val loss)' : ''}`,
          ps.map(p => [p.seriesName, fmt(p.value, d), p.color])),
      },
      xAxis: { type: 'category', data: ep, boundaryGap: false, name: 'epoch (0-indexed)', nameLocation: 'middle', nameGap: 30 },
      yAxis: { ...M.y, axisLabel: { color: T.muted, fontFamily: T.mono, fontSize: 11, formatter: v => v >= 1 || v === 0 ? v : +v.toPrecision(2) } },
      dataZoom: [
        { type: 'inside' },
        { type: 'slider', height: 18, bottom: 10, borderColor: T.border, backgroundColor: T.bg, fillerColor: rgba(T.c[0], .12), textStyle: { color: T.muted, fontFamily: T.mono, fontSize: 10 }, handleStyle: { color: T.c[0], borderColor: T.c[0] }, dataBackground: { lineStyle: { color: T.borderL }, areaStyle: { color: rgba(T.muted, .08) } } },
      ],
      series: M.series.map(([name, key], i) => ({
        id: key, name, type: 'line', smooth: .25, symbol: 'circle', symbolSize: 6, showSymbol: true,
        data: hist.map(r => r[key]), lineStyle: { width: 2.6, color: cols[i] }, itemStyle: { color: cols[i] },
        areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: rgba(cols[i], .28) }, { offset: 1, color: rgba(cols[i], 0) }]) },
        emphasis: { focus: 'series', lineStyle: { width: 3.4 } },
        markLine: i === 1 ? { symbol: 'none', silent: true, lineStyle: { type: 'dashed', color: T.c[3], width: 1.4 }, label: { formatter: `best · epoch ${bestEpoch}`, color: T.c[3], fontFamily: T.mono, fontSize: 10, position: 'insideEndTop' }, data: [{ xAxis: bi }] } : undefined,
        markPoint: i === 1 ? { symbolSize: 54, itemStyle: { color: T.c[3] }, label: { color: '#1a1304', fontFamily: T.mono, fontSize: 9, fontWeight: 700, formatter: p => fmt(p.value, metric.startsWith('loss') ? 3 : 4) }, data: [{ coord: [bi, hist[bi][key]], value: hist[bi][key] }] } : undefined,
      })),
    };
  }

  function confusion(cm) {
    const n = cm.TP + cm.TN + cm.FP + cm.FN, rowN = { 0: cm.TN + cm.FP, 1: cm.FN + cm.TP };
    const cell = (x, y, name, v, good) => {
      const rate = v / rowN[y], col = T.o[name];
      return { name, rate, value: [x, y, v], itemStyle: { color: rgba(col, good ? .14 + .6 * rate : .6), borderColor: T.bg, borderWidth: 4, borderRadius: 10 } };
    };
    return {
      animationDuration: 1000,
      grid: { left: 92, right: 14, top: 14, bottom: 64 },
      tooltip: { formatter: p => tip(`${p.data.name} · ${['Actual NORMAL', 'Actual PNEUMONIA'][p.value[1]]} → ${['predicted NORMAL', 'predicted PNEUMONIA'][p.value[0]]}`, [['Count', p.value[2], T.o[p.data.name]], ['Row rate', (p.data.rate * 100).toFixed(2) + '%'], ['Share of val', (p.value[2] / n * 100).toFixed(2) + '%']]) },
      xAxis: { type: 'category', data: ['NORMAL', 'PNEUMONIA'], name: 'Predicted', nameLocation: 'middle', nameGap: 34, splitLine: { show: false }, axisLine: { show: false }, axisLabel: { color: T.text, fontFamily: T.font, fontWeight: 600, fontSize: 12 } },
      yAxis: { type: 'category', data: ['NORMAL', 'PNEUMONIA'], inverse: true, name: 'Actual', nameLocation: 'middle', nameGap: 74, splitLine: { show: false }, axisLine: { show: false }, axisLabel: { color: T.text, fontFamily: T.font, fontWeight: 600, fontSize: 12 } },
      series: [{
        type: 'heatmap', animationDelay: i => i * 120,
        data: [cell(0, 0, 'TN', cm.TN, true), cell(1, 0, 'FP', cm.FP, false), cell(0, 1, 'FN', cm.FN, false), cell(1, 1, 'TP', cm.TP, true)],
        label: { show: true, formatter: p => `{n|${p.value[2]}}\n{l|${p.data.name} · ${(p.data.rate * 100).toFixed(1)}%}`, rich: { n: { fontSize: 32, fontFamily: T.mono, fontWeight: 600, color: T.text, lineHeight: 42, align: 'center' }, l: { fontSize: 11, color: rgba(T.text, .85), lineHeight: 18, align: 'center' } } },
        emphasis: { itemStyle: { borderColor: T.text, borderWidth: 2 } },
      }],
    };
  }

  function trajectory(hist, bestEpoch) {
    const xs = hist.map(r => r.val_specificity), ys = hist.map(r => r.val_sensitivity);
    const lo = (v, pad) => Math.floor((Math.min(...v) - pad) * 20) / 20, f1 = v => v > 1.0001 ? '' : v.toFixed(2);
    const last = hist.length - 1, col = r => lerpHex(T.c[2], T.c[0], r.epoch / last);
    const best = hist.find(r => r.epoch === bestEpoch);
    return {
      animationDuration: 1200, grid: { left: 62, right: 26, top: 24, bottom: 50 }, toolbox,
      tooltip: { formatter: p => p.data ? tip(`Epoch ${p.data[2]}`, [['Val sensitivity', fmt(p.data[1], 4)], ['Val specificity', fmt(p.data[0], 4)]]) : '' },
      xAxis: { type: 'value', name: 'validation specificity →', nameLocation: 'middle', nameGap: 30, min: lo(xs, .004), max: 1.005, axisLabel: { color: T.muted, fontFamily: T.mono, fontSize: 11, formatter: f1 } },
      yAxis: { type: 'value', name: 'validation sensitivity →', nameLocation: 'middle', nameGap: 44, min: lo(ys, .005), max: 1.005, axisLabel: { color: T.muted, fontFamily: T.mono, fontSize: 11, formatter: f1 } },
      series: [
        { type: 'line', id: 'path', z: 2, data: hist.map(r => ({ value: [r.val_specificity, r.val_sensitivity, r.epoch], itemStyle: { color: col(r) } })), symbol: 'circle', symbolSize: r => 9, lineStyle: { width: 1.6, color: rgba(T.text, .28) }, animationDuration: 2200,
          label: { show: true, formatter: p => p.data.value[2], position: 'top', distance: 6, fontFamily: T.mono, fontSize: 10, color: T.muted } },
        { type: 'effectScatter', id: 'best', z: 5, silent: true, data: [[best.val_specificity, best.val_sensitivity, best.epoch]], symbolSize: 16, rippleEffect: { scale: 3, brushType: 'stroke' }, itemStyle: { color: T.c[3] } },
        { type: 'scatter', id: 'ideal', silent: true, data: [[1, 1]], symbol: 'path://M0,-1 L0.3,-0.3 L1,0 L0.3,0.3 L0,1 L-0.3,0.3 L-1,0 L-0.3,-0.3 Z', symbolSize: 14, itemStyle: { color: T.text, opacity: .55 }, label: { show: true, formatter: 'ideal', position: 'left', color: T.muted, fontSize: 10, fontFamily: T.mono } },
      ],
    };
  }

  function lrTime(hist) {
    return {
      animationDuration: 900, grid: { left: 58, right: 62, top: 40, bottom: 44 }, toolbox, legend: { top: 4, left: 4 },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow', shadowStyle: { color: rgba(T.muted, .06) } }, formatter: ps => tip(`Epoch ${ps[0].axisValue}`, ps.map(p => [p.seriesName, p.seriesName.startsWith('Learning') ? p.value.toExponential(2) : p.value.toFixed(1) + ' s', p.color])) },
      xAxis: { type: 'category', data: hist.map(r => String(r.epoch)), name: 'epoch', nameLocation: 'middle', nameGap: 28 },
      yAxis: [{ type: 'value', min: 0, axisLabel: { color: T.muted, fontFamily: T.mono, fontSize: 11, formatter: v => v + 's' } }, { type: 'log', position: 'right', splitLine: { show: false }, axisLabel: { color: T.muted, fontFamily: T.mono, fontSize: 10, formatter: v => v.toExponential(0) } }],
      series: [
        { name: 'Epoch time (s)', type: 'bar', data: hist.map(r => r.epoch_time_s), barWidth: '56%', itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: rgba(T.c[2], .85) }, { offset: 1, color: rgba(T.c[2], .25) }]), borderRadius: [5, 5, 0, 0] } },
        { name: 'Learning rate', type: 'line', yAxisIndex: 1, data: hist.map(r => r.lr), smooth: .2, symbol: 'circle', symbolSize: 6, lineStyle: { width: 2.4, color: T.c[3] }, itemStyle: { color: T.c[3] } },
      ],
    };
  }

  /* ======================================================================
     2A · Grad-CAM audit
     ====================================================================== */
  const bordInn = (c, mode) => mode === 'nat' ? [c.a.nat1_border_mean, c.a.nat1_inner_mean] : [c.a.stored_border_mean, c.a.stored_inner_mean];
  const ratioOf = (c, mode) => mode === 'nat' ? c.a.nat1_ratio : c.a.stored_ratio;

  function scatter(cases, mode, selIdx) {
    const pts = cases.map(c => ({ c, xy: bordInn(c, mode) }));
    const mx = Math.ceil(Math.max(...pts.map(p => Math.max(...p.xy))) * 1.08 * 20) / 20;
    const sel = pts.find(p => p.c.idx === selIdx);
    const guide = (id, k, label, dash) => ({ id, type: 'line', silent: true, symbol: 'none', z: 1, data: [[0, 0], [mx, mx / k]], lineStyle: { type: dash, color: rgba(T.text, k === 1 ? .5 : .25), width: 1.2 }, endLabel: { show: true, formatter: label, color: T.muted, fontFamily: T.mono, fontSize: 10, distance: 4 } });
    const ax = { axisLabel: { color: T.muted, fontFamily: T.mono, fontSize: 11, formatter: v => +v.toFixed(2) } };
    return {
      animationDuration: 800, grid: { left: 62, right: 56, top: 40, bottom: 52 }, toolbox,
      legend: { top: 4, left: 4, data: OUTCOMES.map(o => ({ name: o, icon: SYM[o] })), itemWidth: 11, itemHeight: 11 },
      tooltip: { trigger: 'item', formatter: p => { const c = p.data && p.data.idx != null && cases.find(x => x.idx === p.data.idx); return c ? tip(`Case #${c.idx} · ${c.outcome}`, [['P(pneumonia)', fmt(c.p)], ['Border mean', fmt(p.value[0], 4)], ['Inner mean', fmt(p.value[1], 4)], ['Ratio', fmtRatio(ratioOf(c, mode))]]) : ''; } },
      xAxis: { type: 'value', min: 0, max: mx, name: 'border mean activation →', nameLocation: 'middle', nameGap: 32, ...ax },
      yAxis: { type: 'value', min: 0, max: mx, name: 'inner mean activation →', nameLocation: 'middle', nameGap: 46, ...ax },
      series: [
        guide('g1', 1, '1×', 'dashed'), guide('g25', 2.5, '2.5×', 'dotted'),
        ...OUTCOMES.map(o => ({
          id: 's' + o, name: o, type: 'scatter', symbol: SYM[o], z: 3,
          data: pts.filter(p => p.c.outcome === o).map(p => ({ value: p.xy, idx: p.c.idx, symbolSize: 11 + 17 * p.c.p })),
          itemStyle: { color: T.o[o], opacity: .92, borderColor: '#0B1220', borderWidth: 1.5 }, emphasis: { scale: 1.35 },
        })),
        { id: 'sel', type: 'effectScatter', silent: true, z: 6, symbolSize: 7, rippleEffect: { scale: 4.2, brushType: 'stroke', period: 3 }, itemStyle: { color: T.text }, data: sel ? [sel.xy] : [] },
      ],
    };
  }

  const catAxisY = (cases, extra = {}) => ({
    type: 'category', inverse: true, data: cases.map(lbl), axisLine: { show: false }, splitLine: { show: false },
    axisLabel: { formatter: catFmt, rich: richOut(), margin: 10 }, ...extra,
  });

  function ratioBars(cases, mode, selIdx) {
    const rows = cases.map(c => ({ c, r: ratioOf(c, mode) })).sort((a, b) => (Number.isFinite(b.r) ? b.r : Infinity) - (Number.isFinite(a.r) ? a.r : Infinity) || a.c.idx - b.c.idx);
    const fin = rows.filter(x => Number.isFinite(x.r)).map(x => x.r), cap = Math.max(...fin) * (mode === 'nat' ? 2 : 1.12);
    const sc = mode === 'nat';
    return {
      animationDuration: 800, grid: { left: 74, right: 76, top: 14, bottom: 36 }, toolbox: { ...toolbox, top: -4 },
      tooltip: { trigger: 'item', formatter: p => { const c = p.data.c; return tip(`Case #${c.idx} · ${c.outcome}`, [['Ratio (border/inner)', fmtRatio(ratioOf(c, mode))], ['Border mean', fmt(bordInn(c, mode)[0], 4)], ['Inner mean', fmt(bordInn(c, mode)[1], 4)]]); } },
      xAxis: sc ? { type: 'log', min: 1, max: cap, name: 'border-to-inner ratio (log)', nameLocation: 'middle', nameGap: 26 } : { type: 'value', min: 0, max: cap, name: 'border-to-inner ratio', nameLocation: 'middle', nameGap: 26 },
      yAxis: catAxisY(rows.map(x => x.c)),
      series: [{
        id: 'ratio', type: 'bar', barWidth: '62%',
        data: rows.map(x => {
          const und = !Number.isFinite(x.r), s = x.c.idx === selIdx, col = T.o[x.c.outcome];
          return { c: x.c, idx: x.c.idx, value: und ? cap : x.r, und, itemStyle: { color: rgba(col, und ? .22 : s ? 1 : .72), borderColor: s ? T.text : col, borderWidth: s ? 2 : und ? 1 : 0, borderType: und ? 'dashed' : 'solid', borderRadius: [0, 5, 5, 0], decal: und ? { symbol: 'rect', dashArrayX: [1, 0], dashArrayY: [2, 4], rotation: -Math.PI / 4, color: rgba(col, .35) } : undefined } };
        }),
        label: { show: true, position: 'right', fontFamily: T.mono, fontSize: 10, color: T.muted, formatter: p => p.data.und ? 'undef.' : p.value >= 100 ? p.value.toFixed(0) + '×' : p.value.toFixed(2) + '×' },
        markLine: { symbol: 'none', silent: true, lineStyle: { color: rgba(T.text, .5), type: 'dashed' }, label: { formatter: '1×', color: T.muted, fontFamily: T.mono, fontSize: 10 }, data: [{ xAxis: 1 }] },
      }],
    };
  }

  function energy(cases, selIdx) {
    const rows = cases.slice().sort((a, b) => b.a.nat1_energy_share - a.a.nat1_energy_share || a.idx - b.idx);
    const mkS = (id, name, color, get) => ({
      id, name, type: 'bar', stack: 'e', barWidth: '62%', itemStyle: { color }, data: rows.map(c => ({ idx: c.idx, c, value: get(c), itemStyle: { color: rgba(color, c.idx === selIdx ? 1 : .72), borderColor: c.idx === selIdx ? T.text : 'transparent', borderWidth: c.idx === selIdx ? 1.5 : 0 } })),
      label: id === 'eb' ? { show: true, position: 'insideLeft', fontFamily: T.mono, fontSize: 10, color: '#1a1304', formatter: p => (p.value * 100).toFixed(0) + '%' } : { show: false }, emphasis: { focus: 'series' },
    });
    return {
      animationDuration: 800, grid: { left: 74, right: 18, top: 34, bottom: 34 }, toolbox: { ...toolbox, top: -4 },
      legend: { top: 0, left: 0 },
      tooltip: { trigger: 'item', formatter: p => { const c = p.data.c; return tip(`Case #${c.idx} · ${c.outcome}`, [['Border (24 cells)', (c.a.nat1_energy_share * 100).toFixed(1) + '%', T.c[3]], ['Inner (25 cells)', ((1 - c.a.nat1_energy_share) * 100).toFixed(1) + '%', T.c[2]]]); } },
      xAxis: { type: 'value', min: 0, max: 1, axisLabel: { color: T.muted, fontFamily: T.mono, fontSize: 11, formatter: v => v * 100 + '%' }, name: 'share of native 7×7 CAM energy', nameLocation: 'middle', nameGap: 24 },
      yAxis: catAxisY(rows),
      series: [mkS('eb', 'Border cells', T.c[3], c => c.a.nat1_energy_share), mkS('ei', 'Inner cells', T.c[2], c => 1 - c.a.nat1_energy_share)],
    };
  }

  function peakGrid(cases, selIdx) {
    const cellMap = {};
    cases.forEach(c => { const k = c.a.nat1_peak_row + ',' + c.a.nat1_peak_col; (cellMap[k] = cellMap[k] || []).push(c); });
    const selC = cases.find(c => c.idx === selIdx), selK = selC ? selC.a.nat1_peak_row + ',' + selC.a.nat1_peak_col : '';
    const data = [];
    for (let r = 0; r < 7; r++) for (let c = 0; c < 7; c++) { const cs = cellMap[r + ',' + c] || []; data.push({ value: [c, r, cs.length, 0], cases: cs, idx: cs[0] && cs[0].idx, sel: (r + ',' + c) === selK }); }
    for (let i = 0; i < 7; i++) { data.push({ value: [i, -.85, 0, 1], text: String(i) }); data.push({ value: [-.85, i, 0, 1], text: String(i) }); }
    data.push({ value: [3, -1.45, 0, 2], text: 'column →' }); data.push({ value: [-1.45, 3, 0, 2], text: 'row ↓' });
    const ax = { type: 'value', min: -1.9, max: 6.55, show: false };
    return {
      animationDuration: 700, grid: { left: 10, right: 10, top: 14, bottom: 6 }, toolbox: { ...toolbox, top: -4 },
      tooltip: { formatter: p => { const d = p.data, r = d.value[1], c = d.value[0]; if (!d.cases) return ''; const ring = r === 0 || r === 6 || c === 0 || c === 6; return tip(`Cell (row ${r}, col ${c}) · ${ring ? 'border ring' : 'inner 5×5'}`, d.cases.length ? d.cases.map(x => [`#${x.idx}`, x.outcome, T.o[x.outcome]]) : [['No CAM peaks here', '']]); } },
      xAxis: ax, yAxis: { ...ax, inverse: true },
      series: [{
        type: 'custom', id: 'peak', data,
        renderItem: (params, api) => {
          const c = api.value(0), r = api.value(1), n = api.value(2), kind = api.value(3), d = data[params.dataIndex];
          if (kind) { const q = api.coord([c, r]); return { type: 'text', style: { text: d.text, x: q[0], y: q[1], textAlign: 'center', textVerticalAlign: 'middle', fill: kind === 2 ? T.dim : T.muted, font: `${kind === 2 ? 500 : 600} ${kind === 2 ? 10 : 11}px ${T.mono}` } }; }
          const a = api.coord([c - .5, r - .5]), b = api.coord([c + .5, r + .5]);
          const x = Math.min(a[0], b[0]), y = Math.min(a[1], b[1]), w = Math.abs(b[0] - a[0]), h = Math.abs(b[1] - a[1]), ring = r === 0 || r === 6 || c === 0 || c === 6;
          const kids = [{ type: 'rect', shape: { x: x + 2, y: y + 2, width: w - 4, height: h - 4, r: 6 }, style: { fill: n ? rgba(T.c[0], Math.min(.95, .22 + .26 * n)) : ring ? rgba(T.c[3], .06) : rgba(T.muted, .05), stroke: d.sel ? T.text : ring ? rgba(T.c[3], .38) : rgba(T.muted, .14), lineWidth: d.sel ? 2.5 : 1 } }];
          if (n) kids.push({ type: 'text', style: { text: String(n), x: x + w / 2, y: y + h / 2, textAlign: 'center', textVerticalAlign: 'middle', fill: T.text, font: `600 ${Math.min(22, h * .4)}px ${T.mono}` } });
          return { type: 'group', children: kids };
        },
      }],
    };
  }

  /* ======================================================================
     2B · Sensitivity
     ====================================================================== */
  const dOf = (c, k) => c.s ? c.s[k + '_abs_delta'] : NaN;

  function sensHeat(cases, order, selIdx) {
    const rows = cases.filter(c => c.s).slice();
    if (order === 'sens') rows.sort((a, b) => CONDS.reduce((s, k) => s + dOf(b, k.key), 0) - CONDS.reduce((s, k) => s + dOf(a, k.key), 0) || a.idx - b.idx);
    const hd = [], flips = [];
    rows.forEach((c, yi) => CONDS.forEach((k, xi) => { hd.push({ value: [xi, yi, dOf(c, k.key)], idx: c.idx, c, k }); if (c.s[k.key + '_flip']) flips.push({ value: [xi, yi], idx: c.idx, c, k }); }));
    const si = rows.findIndex(c => c.idx === selIdx);
    return {
      animationDuration: 800, grid: { left: 74, right: 16, top: 38, bottom: 62 }, toolbox: { ...toolbox, top: -4 },
      tooltip: { formatter: p => { const c = p.data.c, k = p.data.k; if (!c) return ''; return tip(`Case #${c.idx} · ${c.outcome} · ${k.name}`, [['P original', fmt(c.s.original_prob)], ['P perturbed', fmt(c.s[k.key + '_prob'])], ['ΔP', (c.s[k.key + '_delta'] >= 0 ? '+' : '') + fmt(c.s[k.key + '_delta'])], ['Prediction flip', c.s[k.key + '_flip'] ? '★ yes' : 'no', c.s[k.key + '_flip'] ? T.c[4] : null]]); } },
      xAxis: { type: 'category', data: CONDS.map(k => k.name), position: 'top', splitLine: { show: false }, axisLine: { show: false }, axisLabel: { color: T.text, fontFamily: T.font, fontWeight: 600, fontSize: 11 } },
      yAxis: catAxisY(rows),
      visualMap: { type: 'continuous', seriesIndex: 0, dimension: 2, min: 0, max: 1, orient: 'horizontal', left: 'center', bottom: 2, itemWidth: 12, itemHeight: 150, calculable: false, text: ['|ΔP| 1', '0'], textStyle: { color: T.muted, fontFamily: T.mono, fontSize: 10 }, inRange: { color: ['#16233A', '#1F5D73', T.c[0], T.c[3], T.c[4]] } },
      series: [
        { id: 'hm', type: 'heatmap', data: hd, itemStyle: { borderColor: T.bg, borderWidth: 2, borderRadius: 5 }, label: { show: true, fontFamily: T.mono, fontSize: 10, color: '#fff', textShadowColor: 'rgba(0,0,0,.75)', textShadowBlur: 4, formatter: p => p.value[2] < .005 ? '·' : p.value[2].toFixed(2) }, emphasis: { itemStyle: { borderColor: T.text, borderWidth: 2 } } },
        { id: 'flip', type: 'scatter', z: 4, silent: true, data: flips, symbol: 'path://M12 2l3 7h7l-5.5 4.5 2 7.5-6.5-4.5-6.5 4.5 2-7.5L2 9h7z', symbolSize: 12, symbolOffset: [-26, 0], itemStyle: { color: '#fff', borderColor: '#000', borderWidth: .6 } },
        { id: 'rowsel', type: 'custom', z: 8, silent: true, data: si < 0 ? [] : [0, 1, 2].map(xi => [xi, si]),
          renderItem: (p, api) => { const c = api.coord([api.value(0), api.value(1)]), z = api.size([1, 1]); return { type: 'rect', shape: { x: c[0] - z[0] / 2 + 1, y: c[1] - z[1] / 2 + 1, width: z[0] - 2, height: z[1] - 2, r: 5 }, style: { fill: 'none', stroke: T.text, lineWidth: 2.2 } }; } },
      ],
    };
  }

  function slope(cases, condKey, selIdx) {
    const cond = CONDS.find(k => k.key === condKey), rows = cases.filter(c => c.s);
    const band = (y0, y1, col, text, pos) => [{ yAxis: y0, itemStyle: { color: rgba(col, .07) }, label: { show: true, formatter: text, color: rgba(col, .9), fontSize: 10, fontFamily: T.mono, position: pos } }, { yAxis: y1 }];
    return {
      animationDuration: 900, grid: { left: 52, right: 60, top: 36, bottom: 40 }, toolbox: { ...toolbox, top: -4 },
      tooltip: { trigger: 'item', formatter: p => { const c = p.data && p.data.c; return c ? tip(`Case #${c.idx} · ${c.outcome}`, [['P original', fmt(c.s.original_prob)], [`P ${cond.short.toLowerCase()}`, fmt(c.s[condKey + '_prob'])], ['Flip', c.s[condKey + '_flip'] ? '★ yes' : 'no', c.s[condKey + '_flip'] ? T.c[4] : null]]) : ''; } },
      xAxis: { type: 'category', data: ['Original', cond.name], boundaryGap: ['18%', '18%'], axisLine: { show: false }, splitLine: { show: false }, axisLabel: { color: T.text, fontFamily: T.font, fontWeight: 600, fontSize: 11 } },
      yAxis: { type: 'value', min: 0, max: 1, name: 'P(pneumonia)', nameGap: 10, interval: .25 },
      series: [
        { id: 'bands', type: 'line', silent: true, data: [], markArea: { silent: true, data: [band(.5, 1, T.c[4], 'predicted PNEUMONIA', 'insideTopLeft'), band(0, .5, T.c[1], 'predicted NORMAL', 'insideBottomRight')] }, markLine: { silent: true, symbol: 'none', lineStyle: { type: 'dashed', color: rgba(T.text, .5) }, label: { formatter: 'threshold 0.5', color: T.muted, fontFamily: T.mono, fontSize: 10, position: 'insideStartTop' }, data: [{ yAxis: .5 }] } },
        ...rows.map(c => {
          const fl = c.s[condKey + '_flip'], s = c.idx === selIdx, col = T.o[c.outcome];
          return {
            id: 'c' + c.idx, type: 'line', data: [{ value: c.s.original_prob, c, idx: c.idx }, { value: c.s[condKey + '_prob'], c, idx: c.idx }], symbol: SYM[c.outcome], symbolSize: s || fl ? 13 : 8, z: s ? 9 : fl ? 5 : 2,
            lineStyle: { color: col, width: s ? 4.5 : fl ? 3 : 1.5, opacity: s ? 1 : fl ? .95 : .45 }, itemStyle: { color: col, borderColor: s ? T.text : '#0B1220', borderWidth: s ? 2 : 1 }, emphasis: { focus: 'self', lineStyle: { width: 4 } },
            endLabel: { show: s || fl, formatter: `#${c.idx}${fl ? ' ★' : ''}`, color: s ? T.text : col, fontFamily: T.mono, fontSize: 10, distance: 6 },
          };
        }),
      ],
    };
  }

  function radar(sum) {
    const by = sum.by_condition, ind = [['Mean |ΔP|', 1], ['Median |ΔP|', 1], ['Max |ΔP|', 1], ['Flip rate', 1], ['P-increase rate', 1]];
    const vals = k => { const b = by[k]; return [b.mean_abs_delta, b.median_abs_delta, b.max_abs_delta, b.n_flip / b.n, b.n_prob_increase / b.n]; };
    return {
      animationDuration: 1200, legend: { bottom: 0, left: 'center' }, toolbox: { ...toolbox, top: -4 },
      tooltip: { formatter: p => { const b = by[CONDS.find(k => k.name === p.name).key]; return tip(p.name, [['Mean |ΔP|', fmt(b.mean_abs_delta)], ['Median |ΔP|', fmt(b.median_abs_delta)], ['Max |ΔP|', fmt(b.max_abs_delta)], ['Flips', `${b.n_flip} / ${b.n}`], ['P increased', `${b.n_prob_increase} / ${b.n}`], ['P decreased', `${b.n_prob_decrease} / ${b.n}`]]); } },
      radar: { indicator: ind.map(([name, max]) => ({ name, max })), radius: '64%', center: ['50%', '50%'], splitNumber: 4, axisName: { color: T.muted, fontSize: 11, fontFamily: T.font }, splitLine: { lineStyle: { color: rgba(T.muted, .2) } }, splitArea: { show: false }, axisLine: { lineStyle: { color: rgba(T.muted, .2) } } },
      series: [{ type: 'radar', symbolSize: 6, data: CONDS.map(k => ({ name: k.name, value: vals(k.key), lineStyle: { color: T.k[k.key], width: 2.4 }, itemStyle: { color: T.k[k.key] }, areaStyle: { color: rgba(T.k[k.key], .17) } })), emphasis: { areaStyle: { opacity: .4 } } }],
    };
  }

  function jitter(idx, ci) { return ci + 1 + (((idx * 37 + ci * 11) % 100) / 100 - .5) * .56; }
  function strip(cases, sum, selIdx) {
    const rows = cases.filter(c => c.s), names = ['', ...CONDS.map(k => k.name), ''];
    const sel = rows.find(c => c.idx === selIdx);
    return {
      animationDuration: 800, grid: { left: 48, right: 18, top: 40, bottom: 40 }, toolbox: { ...toolbox, top: -4 },
      legend: { top: 0, left: 0, data: OUTCOMES.map(o => ({ name: o, icon: SYM[o] })), itemWidth: 11, itemHeight: 11 },
      tooltip: { trigger: 'item', formatter: p => { const c = p.data && p.data.c; return c ? tip(`Case #${c.idx} · ${c.outcome}`, [[p.data.k.name, fmt(p.value[1]), T.k[p.data.k.key]], ['Flip', c.s[p.data.k.key + '_flip'] ? '★ yes' : 'no']]) : ''; } },
      xAxis: { type: 'value', min: 0, max: 4, interval: 1, splitLine: { show: false }, axisLine: { show: false }, axisLabel: { color: T.text, fontFamily: T.font, fontWeight: 600, fontSize: 11, formatter: v => names[v] || '' } },
      yAxis: { type: 'value', min: 0, max: 1, interval: .25, boundaryGap: false },
      series: [
        { id: 'med', type: 'line', silent: true, symbol: 'none', connectNulls: false, lineStyle: { color: T.text, width: 2, type: 'dashed', opacity: .8 }, data: CONDS.flatMap((k, i) => [[i + .62, sum.by_condition[k.key].median_abs_delta], [i + 1.38, sum.by_condition[k.key].median_abs_delta], null]), endLabel: { show: false } },
        ...OUTCOMES.map(o => ({
          id: 'p' + o, name: o, type: 'scatter', symbol: SYM[o], z: 3, clip: false, itemStyle: { color: T.o[o] },
          data: rows.filter(c => c.outcome === o).flatMap(c => CONDS.map((k, ci) => ({ value: [jitter(c.idx, ci), dOf(c, k.key)], c, k, idx: c.idx, symbolSize: c.s[k.key + '_flip'] ? 15 : 10, itemStyle: { color: T.o[o], borderColor: c.s[k.key + '_flip'] ? T.text : '#0B1220', borderWidth: c.s[k.key + '_flip'] ? 2 : 1, opacity: .9 } }))),
        })),
        { id: 'sel', type: 'effectScatter', silent: true, z: 6, symbolSize: 6, rippleEffect: { scale: 4, brushType: 'stroke', period: 3 }, itemStyle: { color: T.text }, data: sel ? CONDS.map((k, ci) => [jitter(sel.idx, ci), dOf(sel, k.key)]) : [] },
      ],
    };
  }

  /* mini bar used inside the inspector (Grad-CAM mode) */
  function miniBI(c) {
    const b = c.a.stored_border_mean, i = c.a.stored_inner_mean, mx = Math.max(b, i) * 1.35;
    return {
      animationDuration: 600, grid: { left: 46, right: 14, top: 34, bottom: 8 },
      xAxis: { type: 'value', min: 0, max: mx, show: false }, yAxis: { type: 'category', inverse: true, data: ['Border', 'Inner'], axisLine: { show: false }, axisLabel: { color: T.muted, fontSize: 10, fontFamily: T.font } },
      series: [{ type: 'bar', barWidth: 14, data: [{ value: b, itemStyle: { color: T.c[3], borderRadius: 4 } }, { value: i, itemStyle: { color: T.c[2], borderRadius: 4 } }], label: { show: true, position: 'right', fontFamily: T.mono, fontSize: 10, color: T.text, formatter: p => p.value.toFixed(3) }, showBackground: true, backgroundStyle: { color: rgba(T.muted, .08), borderRadius: 4 } }],
      graphic: [{ type: 'text', right: 10, top: 6, style: { text: fmtRatio(c.a.stored_ratio), fill: T.text, font: `700 14px ${T.mono}` } }],
    };
  }

  return { T, OUTCOMES, CONDS, SYM, initTheme, mk, resizeAll, rgba, fmt, fmtRatio, tip, lbl,
    gauge, training, confusion, trajectory, lrTime, scatter, ratioBars, energy, peakGrid, sensHeat, slope, radar, strip, miniBI };
})();
