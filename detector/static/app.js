'use strict';
const $ = id => document.getElementById(id);
let mode = 'paste', working = false, ready = false;
const samples = {en: 'I went back to the little bookshop near our old house yesterday. The owner still remembered the mystery novels I used to borrow after school.'};
function setMode(next) {
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
function count() { const n = $('text').value.trim().split(/\s+/u).filter(Boolean).length; $('word-count').textContent = `${n} word${n === 1 ? '' : 's'}`; }
$('text').addEventListener('input', count);
$('example').addEventListener('click', () => { $('text').value = samples.en; $('language').value = 'en'; count(); });
$('clear').addEventListener('click', () => { $('text').value = ''; count(); $('error').hidden = true; $('result').hidden = true; $('empty-result').hidden = false; $('text').focus(); });
$('file').addEventListener('change', () => { $('file-name').textContent = $('file').files[0]?.name || 'No file selected'; $('error').hidden = true; });
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
function showResult(data) {
  $('empty-result').hidden = true; $('result').hidden = false;
  $('prediction').textContent = data.label === 'ai' ? 'Likely AI-generated' : 'Likely human-written';
  $('prediction-note').textContent = data.uncertain ? 'Low-confidence result. The detector found mixed signals.' : 'The patterns in this passage lean toward this class. This is an estimate, not a verification.';
  $('confidence').textContent = `${(data.score * 100).toFixed(1)}% top score`;
  bars('detection-bars', data.scores);
  $('family-section').hidden = !data.family;
  if (data.family) {
    $('family-name').textContent = data.family.label;
    $('family-note').textContent = data.family.uncertain ? 'Attribution is inconclusive; this is only the closest of the three trained families.' : 'Closest match among the three trained families, conditional on the text being AI-generated.';
    bars('family-bars', data.family.scores);
  }
  $('warnings').replaceChildren(...data.warnings.map(w => { const li = document.createElement('li'); li.textContent = w; return li; }));
  $('result-meta').textContent = `${data.word_count} cleaned words · ${data.token_count} tokens · ${data.model_version}`;
  if (matchMedia('(max-width:640px)').matches) $('result').scrollIntoView({behavior: 'smooth', block: 'start'});
}
$('analyze-form').addEventListener('submit', async event => {
  event.preventDefault(); if (working || !ready) return;
  $('error').hidden = true;
  let path, options;
  if (mode === 'upload') {
    const file = $('file').files[0];
    if (!file) return showError('Choose a .txt or .md file first.');
    if (file.size > 48000) return showError('Please choose a file under 48 KB.');
    const body = new FormData(); body.append('file', file); body.append('language', $('language').value);
    path = '/api/analyze-file'; options = {method: 'POST', body};
  } else {
    if (!$('text').value.trim()) return showError('Paste a passage before analyzing.');
    path = '/api/analyze'; options = {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({text: $('text').value, language: $('language').value})};
  }
  working = true; $('submit').disabled = true; $('submit-label').textContent = 'Reading the patterns…'; $('analyze-form').setAttribute('aria-busy', 'true');
  // Clear the old result so a failed request cannot leave a stale prediction visible.
  $('result').hidden = true; $('empty-result').hidden = false;
  const controller = new AbortController(); const timer = setTimeout(() => controller.abort(), 60000);
  try {
    const response = await fetch(path, {...options, signal: controller.signal});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Please check your passage and language selection.');
    showResult(data);
  } catch (e) { showError(e.name === 'AbortError' ? 'Analysis took too long. Please try a shorter passage.' : e.message || 'Could not reach the server. Please try again.'); }
  finally { clearTimeout(timer); working = false; $('submit').disabled = !ready; $('submit-label').textContent = 'Analyze passage'; $('analyze-form').setAttribute('aria-busy', 'false'); }
});
function showError(message) { $('error').textContent = message; $('error').hidden = false; }
async function health() {
  try { const response = await fetch('/api/health'); ready = response.ok; }
  catch { ready = false; }
  $('model-status').textContent = ready ? 'Model ready' : 'Model unavailable';
  $('model-status').className = `status ${ready ? 'ready' : 'unavailable'}`;
  $('submit').disabled = working || !ready;
  if (!ready) setTimeout(health, 10000);
}
health();
