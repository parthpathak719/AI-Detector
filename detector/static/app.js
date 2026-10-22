'use strict';
const $ = id => document.getElementById(id);
const MAX_FILE = 10 * 1024 * 1024;
let mode = 'paste', working = false, ready = false, selectedFile = null, lastReport = null, visibleSegments = 0;
const sample = 'I went back to the little bookshop near our old house yesterday. The owner still remembered the mystery novels I used to borrow after school.';
function clearResult() {
  lastReport = null;
  $('result').hidden = true;
  $('empty-result').hidden = false;
  $('segments-section').hidden = true;
}
function setMode(next) {
  if (working) return;
  if (mode !== next) clearResult();
  mode = next;
  for (const key of ['paste', 'upload']) {
    $(`${key}-tab`).setAttribute('aria-selected', String(key === mode));
    $(`${key}-tab`).tabIndex = key === mode ? 0 : -1;
    $(`${key}-panel`).hidden = key !== mode;
  }
  $('error').hidden = true;
}
for (const key of ['paste', 'upload']) {
  $(`${key}-tab`).addEventListener('click', () => setMode(key));
  $(`${key}-tab`).addEventListener('keydown', e => {
    if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) {
      e.preventDefault();
      setMode(e.key === 'Home' ? 'paste' : e.key === 'End' ? 'upload' : mode === 'paste' ? 'upload' : 'paste');
      $(`${mode}-tab`).focus();
    }
  });
}
function count() {
  const text = $('text').value;
  const n = text.trim().split(/\s+/u).filter(Boolean).length;
  $('word-count').textContent = `${n.toLocaleString()} words · ${[...text].length.toLocaleString()} / 100,000 chars`;
}
$('text').addEventListener('input', () => { count(); clearResult(); });
$('language').addEventListener('change', clearResult);
$('example').addEventListener('click', () => { $('text').value = sample; $('language').value = 'en'; count(); clearResult(); $('error').hidden = true; });
$('clear').addEventListener('click', () => { $('text').value = ''; count(); $('error').hidden = true; clearResult(); $('text').focus(); });
function chooseFile(file) {
  if (working) return;
  $('error').hidden = true; clearResult();
  if (file && !/\.(txt|md|pdf|docx|doc)$/iu.test(file.name)) {
    showError('Choose a TXT, Markdown, PDF, DOCX or DOC file.'); file = null;
  }
  if (file && file.size > MAX_FILE) { showError('Please choose a file under 10 MB.'); file = null; }
  selectedFile = file;
  $('file-name').textContent = file ? `${file.name} · ${(file.size / 1024).toFixed(1)} KB` : 'No file selected';
  $('remove-file').hidden = !file;
}
$('file').addEventListener('change', () => chooseFile($('file').files[0] || null));
$('remove-file').addEventListener('click', () => { $('file').value = ''; chooseFile(null); });
for (const type of ['dragenter', 'dragover']) $('drop-zone').addEventListener(type, e => {
  e.preventDefault(); if (!working) $('drop-zone').classList.add('dragging');
});
for (const type of ['dragleave', 'drop']) $('drop-zone').addEventListener(type, e => {
  e.preventDefault(); $('drop-zone').classList.remove('dragging');
});
$('drop-zone').addEventListener('drop', e => {
  if (working) return;
  if (e.dataTransfer.files.length !== 1) { chooseFile(null); return showError('Choose one document at a time.'); }
  $('file').value = ''; chooseFile(e.dataTransfer.files[0]);
});
function bars(target, scores) {
  $(target).replaceChildren();
  for (const [name, score] of Object.entries(scores)) {
    const row = document.createElement('div'); row.className = `score-row ${name}`;
    const label = document.createElement('div'); label.className = 'score-label';
    const title = document.createElement('span'); title.textContent = name === 'ai' ? 'AI' : name;
    const value = document.createElement('span'); value.textContent = `${(score * 100).toFixed(1)}%`;
    label.append(title, value);
    const bar = document.createElement('progress'); bar.max = 1; bar.value = score; bar.setAttribute('aria-label', `${name} model score`);
    row.append(label, bar); $(target).append(row);
  }
}
function showSegments() {
  const end = Math.min(visibleSegments + 12, lastReport.segments.length);
  for (const segment of lastReport.segments.slice(visibleSegments, end)) {
    const card = document.createElement('article'); card.className = `segment ${segment.label}`;
    const head = document.createElement('div'); head.className = 'segment-head';
    const number = document.createElement('span'); number.textContent = `SEGMENT ${String(segment.index).padStart(2, '0')}`;
    const verdict = document.createElement('strong'); verdict.textContent = `${segment.label === 'ai' ? 'AI-leaning' : 'Human-leaning'} · ${(segment.ai_score * 100).toFixed(1)}% AI score`;
    head.append(number, verdict);
    const text = document.createElement('p'); text.textContent = segment.text;
    const meta = document.createElement('small'); meta.textContent = `${segment.word_count} words · ${segment.token_count} model tokens`;
    card.append(head, text, meta); $('segment-list').append(card);
  }
  visibleSegments = end;
  $('more-segments').hidden = end === lastReport.segments.length;
  $('more-segments').textContent = `Show more segments (${end} / ${lastReport.segments.length})`;
}
$('more-segments').addEventListener('click', showSegments);
function showResult(data, elapsed) {
  lastReport = data;
  $('empty-result').hidden = true; $('result').hidden = false;
  $('prediction').textContent = data.label === 'ai' ? 'Likely AI-generated' : 'Likely human-written';
  $('prediction-note').textContent = data.uncertain ? 'Mixed signals. The overall scores are close; interpret this result cautiously.' : data.chunk_count > 1 ? 'Overall signal across the document. Inspect individual segments below for differences.' : 'The patterns in this passage lean toward this class. This is an estimate, not verification.';
  $('confidence-tag').textContent = data.uncertain ? 'Uncertain signal' : 'Analysis complete';
  $('confidence').textContent = `${(data.score * 100).toFixed(1)}% top score`;
  $('result-words').textContent = data.word_count.toLocaleString();
  $('result-segments').textContent = data.chunk_count.toLocaleString();
  $('result-time').textContent = elapsed < 60 ? `${elapsed.toFixed(1)}s` : `${(elapsed / 60).toFixed(1)}m`;
  bars('detection-bars', data.scores);
  $('family-section').hidden = !data.family;
  if (data.family) {
    $('family-name').textContent = data.family.label;
    $('family-note').textContent = data.family.uncertain ? 'Attribution is inconclusive. This is the closest of the three known families.' : `Closest known family across ${data.family.analyzed_segments} AI-leaning segment(s). Unknown generators may resemble these families.`;
    bars('family-bars', data.family.scores);
  }
  $('warnings').replaceChildren(...data.warnings.map(w => { const li = document.createElement('li'); li.textContent = w; return li; }));
  $('result-meta').textContent = `${data.chunk_count > 1 ? 'Word-weighted segment scores' : 'Single-segment result'} · ${data.model_version}`;
  $('segments-section').hidden = false;
  $('segment-summary').textContent = `${data.ai_segments} of ${data.chunk_count} segments lean AI`;
  $('segment-list').replaceChildren(); visibleSegments = 0; showSegments();
  if (matchMedia('(max-width:760px)').matches) $('result').scrollIntoView({behavior: matchMedia('(prefers-reduced-motion:reduce)').matches ? 'instant' : 'smooth', block: 'start'});
}
$('download-report').addEventListener('click', () => {
  if (!lastReport) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastReport, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = 'origin-analysis.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
function lockForm(value) {
  for (const id of ['paste-tab', 'upload-tab', 'text', 'file', 'language', 'clear', 'example', 'remove-file']) $(id).disabled = value;
  $('submit').disabled = value || !ready;
  $('analyze-form').setAttribute('aria-busy', String(value));
  $('loading-note').hidden = !value;
}
$('analyze-form').addEventListener('submit', async event => {
  event.preventDefault(); if (working || !ready) return;
  $('error').hidden = true;
  let path, options;
  if (mode === 'upload') {
    if (!selectedFile) return showError('Choose a document first.');
    const body = new FormData(); body.append('file', selectedFile); body.append('language', $('language').value);
    path = '/api/analyze-file'; options = {method: 'POST', body};
  } else {
    if (!$('text').value.trim()) return showError('Paste some text before analyzing.');
    path = '/api/analyze'; options = {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text: $('text').value, language: $('language').value})};
  }
  working = true; lockForm(true); clearResult();
  const started = performance.now();
  $('submit-label').textContent = 'Analyzing content…';
  const ticker = setInterval(() => { $('submit-label').textContent = `Analyzing content… ${Math.floor((performance.now() - started) / 1000)}s`; }, 1000);
  const controller = new AbortController(); const timeout = setTimeout(() => controller.abort(), 15 * 60 * 1000);
  try {
    const response = await fetch(path, {...options, signal: controller.signal});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Check the language and document size (100,000 characters maximum).');
    showResult(data, (performance.now() - started) / 1000);
  } catch (e) {
    showError(e.name === 'AbortError' ? 'The request timed out. The server may still be working; wait before retrying with a smaller document.' : e instanceof TypeError ? 'Could not reach the server. Check that it is running and try again.' : e.message);
  } finally {
    clearTimeout(timeout); clearInterval(ticker); working = false; lockForm(false); $('submit-label').textContent = 'Analyze content';
  }
});
function showError(message) { $('error').textContent = message; $('error').hidden = false; }
async function health() {
  try { const response = await fetch('/api/health'); ready = response.ok; } catch { ready = false; }
  $('model-status').textContent = ready ? 'Model ready' : 'Model unavailable';
  $('model-status').className = `status ${ready ? 'ready' : 'unavailable'}`;
  $('submit').disabled = working || !ready;
  if (!ready) setTimeout(health, 10000);
}
health();
