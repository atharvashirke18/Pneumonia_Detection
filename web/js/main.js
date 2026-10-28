/* main.js — detection page */
'use strict';
const $ = id => document.getElementById(id);
const MAX_BYTES = 15 * 1024 * 1024, OK_NAME = /\.(jpe?g|png|bmp|tiff?)$/i;
const b64 = s => 'data:image/png;base64,' + s;

function setImg(id, src) { const el = $(id); el.src = src; el.hidden = false; }
function showError(msg) {
  $('resContent').classList.add('hidden'); $('resEmpty').classList.remove('hidden');
  $('resBox').className = 'res';
  $('resEmpty').innerHTML = '<div class="card-title" style="color:var(--danger)">Analysis failed</div><p style="margin-top:.5rem"></p>';
  $('resEmpty').querySelector('p').textContent = msg;
}

/* Heatmap repair: if the PNG is a raw 0–1 (or grayscale) map it renders black.
   Detect that, min-max normalise and apply a Jet colormap client-side. */
// function ensureJet(src) {
//   return new Promise(resolve => {
//     const im = new Image();
//     im.onload = () => {
//       const c = document.createElement('canvas'); c.width = im.naturalWidth; c.height = im.naturalHeight;
//       const x = c.getContext('2d'); x.drawImage(im, 0, 0);
//       const d = x.getImageData(0, 0, c.width, c.height), p = d.data;
//       let lo = 255, hi = 0, colored = false;
//       for (let i = 0; i < p.length; i += 4) {
//         if (Math.abs(p[i] - p[i + 2]) > 8 || Math.abs(p[i] - p[i + 1]) > 8) { colored = true; break; }
//         lo = Math.min(lo, p[i]); hi = Math.max(hi, p[i]);
//       }
//       if (colored && hi > 30) return resolve(src);           // already a proper colormap
//       const span = Math.max(hi - lo, 1), cl = v => Math.max(0, Math.min(1, v));
//       for (let i = 0; i < p.length; i += 4) {
//         const v = (p[i] - lo) / span;
//         p[i] = 255 * cl(1.5 - Math.abs(4 * v - 3));
//         p[i + 1] = 255 * cl(1.5 - Math.abs(4 * v - 2));
//         p[i + 2] = 255 * cl(1.5 - Math.abs(4 * v - 1));
//         p[i + 3] = 255;
//       }
//       x.putImageData(d, 0, 0); resolve(c.toDataURL('image/png'));
//     };
//     im.onerror = () => resolve(src);
//     im.src = src;
//   });
// }

function countUp(el, to, ms = 700) {
  const t0 = performance.now();
  const f = t => { const k = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - k, 3); el.textContent = (to * e).toFixed(3); if (k < 1) requestAnimationFrame(f); };
  requestAnimationFrame(f);
}

function handleFile(file) {
  if (!OK_NAME.test(file.name)) return showError('Unsupported file type. Use PNG, JPG, BMP or TIFF.');
  if (file.size > MAX_BYTES) return showError('File is larger than 15 MB.');
  const r = new FileReader();
  r.onload = e => { setImg('prevImg', e.target.result); $('prevPh').classList.add('hidden'); $('prevMark').classList.remove('hidden'); };
  r.readAsDataURL(file);
  runPrediction(file);
}

async function runPrediction(file) {
  $('predictLoader').classList.remove('hidden');
  $('resEmpty').classList.add('hidden'); $('resContent').classList.add('hidden');
  const fd = new FormData(); fd.append('file', file);
  try {
    const res = await fetch('/api/predict', { method: 'POST', body: fd });
    if (!res.ok) { let m = `HTTP ${res.status}`; try { m = (await res.json()).detail || m; } catch (_) {} throw new Error(m); }
    await displayResult(await res.json());
  } catch (e) { showError(e.message || 'Request failed'); }
  finally { $('predictLoader').classList.add('hidden'); }
}

async function displayResult(d) {
  const pos = d.class_name === 'PNEUMONIA', p = d.pneumonia_probability;
  $('resBox').className = 'res ' + (pos ? 'res-pos' : 'res-neg');
  $('resContent').classList.remove('hidden');
  $('clsName').textContent = d.class_name;
  $('predBadge').textContent = pos ? 'Pneumonia pattern' : 'Normal pattern';
  $('classFormula').textContent = pos ? 'P(PNEUMONIA | X)' : `P(PNEUMONIA | X) · NORMAL = ${(1 - p).toFixed(3)}`;
  countUp($('probValue'), p); $('probPct').textContent = (p * 100).toFixed(1) + '%';
  const bar = $('probBar'); bar.style.width = '0%';
  requestAnimationFrame(() => requestAnimationFrame(() => { bar.style.width = (p * 100).toFixed(1) + '%'; }));

  if (d.original_b64) { setImg('imgOrig', b64(d.original_b64)); setImg('prevImg', b64(d.original_b64)); }
  if (d.heatmap_b64) setImg('imgHeat', b64(d.heatmap_b64));
  if (d.overlay_b64) setImg('imgOver', b64(d.overlay_b64));

  const g = d.gradcam_details, rows = $('gradcamTable').querySelectorAll('tr');
  if (g && rows.length >= 5)
    [g.target_layer, (g.native_resolution || '').replace('x', ' × '), g.upsampling, g.normalization, g.relu]
      .forEach((v, i) => { if (v) rows[i].cells[1].textContent = v; });
}

const pick = () => $('fileInput').click();
$('newBtn').addEventListener('click', pick);
$('chooseBtn').addEventListener('click', e => { e.stopPropagation(); pick(); });
$('uploadZone').addEventListener('click', pick);
$('uploadZone').addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
$('fileInput').addEventListener('change', e => { const f = e.target.files[0]; if (f) handleFile(f); e.target.value = ''; });
const card = $('uploadCard');
['dragover', 'dragenter'].forEach(ev => card.addEventListener(ev, e => { e.preventDefault(); $('uploadZone').classList.add('drag-over'); }));
card.addEventListener('dragleave', e => { if (e.target === card) $('uploadZone').classList.remove('drag-over'); });
card.addEventListener('drop', e => { e.preventDefault(); $('uploadZone').classList.remove('drag-over'); const f = e.dataTransfer.files[0]; if (f) handleFile(f); });