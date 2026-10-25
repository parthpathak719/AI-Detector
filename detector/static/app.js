'use strict';
const $ = id => document.getElementById(id);
const MAX_FILE = 10 * 1024 * 1024;
const languages = {en:'English',hi:'Hindi',ta:'Tamil',kn:'Kannada',hi_en:'Hinglish',ta_en:'Tanglish',kn_en:'Kanglish'};
let mode='paste', working=false, ready=false, exporting=false, preparing=false, selectedFiles=[], items=[], lastReport=null, revision=0, stopRequested=false, suggestionRevision=0;
const sample = 'I went back to the little bookshop near our old house yesterday. The owner still remembered the mystery novels I used to borrow after school.';
async function request(path, options) {
  const controller=new AbortController();
  const timer=setTimeout(()=>controller.abort(),15*60*1000);
  try {
    const response=await fetch(path,{...options,signal:controller.signal});
    const data=await response.json();
    if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'Check the input format and size.');
    return data;
  }catch(e){
    if(e.name==='AbortError')throw new Error('Request timed out. The server may still be processing this item; wait before retrying.');
    throw e;
  }finally{clearTimeout(timer);}
}
function jsonRequest(path, body) { return request(path, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); }
function clearResult() {
  lastReport=null; $('result').hidden=true; $('empty-result').hidden=false; $('export-error').hidden=true;
  $('batch-results').hidden=true; $('batch-list').replaceChildren();
}
function setMode(next) {
  if (working || preparing) return;
  if(mode!==next) clearResult(); mode=next;
  for(const key of ['paste','upload']) {
    $(`${key}-tab`).setAttribute('aria-selected',String(key===mode));
    $(`${key}-tab`).tabIndex=key===mode?0:-1; $(`${key}-panel`).hidden=key!==mode;
  }
  $('error').hidden=true; updateLanguageHint();
}
for(const key of ['paste','upload']) {
  $(`${key}-tab`).addEventListener('click',()=>setMode(key));
  $(`${key}-tab`).addEventListener('keydown',e=>{
    if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)) {
      e.preventDefault();setMode(e.key==='Home'?'paste':e.key==='End'?'upload':mode==='paste'?'upload':'paste');$(`${mode}-tab`).focus();
    }
  });
}
function count() { const text=$('text').value; $('word-count').textContent=`${text.trim().split(/\s+/u).filter(Boolean).length.toLocaleString()} words · ${[...text].length.toLocaleString()} / 100,000 chars`; }
let suggestTimer;
function scheduleSuggestion() { clearTimeout(suggestTimer); const id=++suggestionRevision; suggestTimer=setTimeout(async()=>{
  if(mode!=='paste'||$('language').value!=='auto') return;
  try {const s=await jsonRequest('/api/suggest-language',{text:$('text').value});if(id===suggestionRevision&&$('language').value==='auto'&&mode==='paste') $('language-hint').textContent=`Suggested: ${s.name}. ${s.reason}`;}catch{if(id===suggestionRevision) $('language-hint').textContent='Suggestion unavailable. Choose a language manually.';}
},400); }
function updateLanguageHint() {
  ++suggestionRevision;
  $('language-hint').textContent=$('language').value==='auto'?(mode==='upload'?'Each preview has its own suggested language. You can change it before analysis.':'Suggestions can be changed. Romanized text may need manual selection.'):'Your selected language will be used for every item.';
  if(mode==='paste')scheduleSuggestion();else renderPreview();
}
$('text').addEventListener('input',()=>{count();clearResult();scheduleSuggestion();});
$('language').addEventListener('change',()=>{clearResult();updateLanguageHint();});
$('example').addEventListener('click',()=>{$('text').value=sample;count();clearResult();$('error').hidden=true;updateLanguageHint();});
$('clear').addEventListener('click',()=>{$('text').value='';count();clearResult();$('error').hidden=true;updateLanguageHint();$('text').focus();});
for(const [value,label] of Object.entries(languages)) $('preview-language').add(new Option(label,value));
function renderPreview() {
  $('preview').hidden=!items.length;
  if(!items.length)return;
  const item=items[Number($('preview-item').value)||0];
  $('preview-text').value=item.text;
  $('preview-notes').textContent=`${items.length} item(s) ready. ${item.notes.join(' ')}`;
  const automatic=$('language').value==='auto';
  $('preview-language').value=automatic?item.language:$('language').value;
  $('preview-language').disabled=!automatic||working||preparing;
  $('preview-hint').textContent=automatic?item.suggestion.reason:'Using the language selected below for all items.';
}
$('preview-item').addEventListener('change',renderPreview);
$('preview-language').addEventListener('change',()=>{items[Number($('preview-item').value)].language=$('preview-language').value;clearResult();});
async function chooseFiles(files) {
  if(working||preparing)return;
  const list=Array.from(files);clearResult();$('error').hidden=true;
  if(list.length>10){await chooseFiles([]);return showError('Choose up to 10 files at once.');}
  selectedFiles=list;items=[];revision++;const current=revision;
  $('drop-zone').classList.toggle('has-files',list.length>0);
  $('jsonl-options').hidden=!list.some(f=>/\.jsonl$/i.test(f.name));
  $('file-name').textContent=list.length===1?`${list[0].name} · ${(list[0].size/1024).toFixed(1)} KB`:`${list.length} files selected`;
  $('remove-file').hidden=!list.length;$('preview').hidden=true;$('preview-item').replaceChildren();
  if(!list.length)return;
  preparing=true;lockForm(true);$('preparing').hidden=false;
  const errors=[];
  try {
    for(const file of list) {
      try {
        if(file.size>MAX_FILE)throw new Error('File exceeds 10 MB.');
        const body=new FormData();body.append('file',file);body.append('jsonl_mode',$('jsonl-mode').value);
        const data=await request('/api/preview-file',{method:'POST',body});
        if(items.length+data.documents.length>100)throw new Error('Batch limit is 100 items; split this file into a smaller batch.');
        for(const doc of data.documents)items.push({...doc,name:file.name+(doc.name?' / '+doc.name:''),language:doc.suggestion.language});
      }catch(e){errors.push(`${file.name}: ${e.message}`);}
    }
    if(current!==revision)return;
    items.forEach((item,i)=>$('preview-item').add(new Option(item.name,String(i))));
    if(errors.length)showError(errors.join(' '));
  }finally{preparing=false;lockForm(false);$('preparing').hidden=true;renderPreview();updateLanguageHint();}
}
$('file').addEventListener('change',()=>chooseFiles($('file').files));
$('remove-file').addEventListener('click',()=>{$('file').value='';chooseFiles([]);});
$('jsonl-mode').addEventListener('change',()=>chooseFiles(selectedFiles));
for(const type of ['dragenter','dragover'])$('drop-zone').addEventListener(type,e=>{e.preventDefault();if(!working&&!preparing)$('drop-zone').classList.add('dragging');});
for(const type of ['dragleave','drop'])$('drop-zone').addEventListener(type,e=>{e.preventDefault();$('drop-zone').classList.remove('dragging');});
$('drop-zone').addEventListener('drop',e=>{if(!working&&!preparing){$('file').value='';chooseFiles(e.dataTransfer.files);}});
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
function showResult(data, elapsed, sourceName = 'Pasted text') {
  lastReport = {...data, source_name: sourceName.slice(0, 240)};
  $('result-source').textContent = sourceName;
  $('empty-result').hidden = true; $('result').hidden = false;
  const documentResult = data.assessment === 'document_model_signal';
  $('prediction').textContent = documentResult ? 'Authorship inconclusive' : data.label === 'ai' ? 'AI-leaning model signal' : 'Human-leaning model signal';
  $('prediction-note').textContent = documentResult
    ? `The model leans ${data.label === 'ai' ? 'AI' : 'human'}, based on the combined segment scores. Even a high score can be wrong.`
    : data.model_uncertain ? 'The model scores are close. Neither authorship nor generator identity is established.'
    : 'This is a learned pattern match, not a verified origin or probability of authorship.';
  $('confidence-tag').textContent = documentResult ? 'Document model signal' : 'Model signal only';
  $('confidence').textContent = 'Not authorship confidence';

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
  $('result-meta').textContent = `${data.chunk_count > 1 ? 'Word-weighted segment scores' : 'Single-segment result'} · ${data.model_version} · ${data.encoder_variant}`;
  if ((!working || $('batch-results').hidden) && matchMedia('(max-width:760px)').matches) $('result').scrollIntoView({behavior: matchMedia('(prefers-reduced-motion:reduce)').matches ? 'instant' : 'smooth', block: 'start'});
}
$('download-report').addEventListener('click', async () => {
  if (!lastReport || exporting || working) return;
  const report = lastReport;
  exporting = true;
  $('download-report').disabled = true;
  $('download-report').textContent = 'Creating PDF…';
  $('export-error').hidden = true;
  $('export-status').hidden = false;
  $('export-status').textContent = 'Preparing the complete report, including all segments.';
  try {
    const response = await fetch('/api/report.pdf', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(report)
    });
    if (!response.ok) throw new Error('The PDF could not be created. Check the server and try again.');
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url; link.download = 'lexora-analysis.pdf'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  } catch (error) {
    $('export-error').textContent = error.message;
    $('export-error').hidden = false;
  } finally {
    exporting = false;
    $('download-report').disabled = false;
    $('download-report').textContent = 'Download PDF report';
    $('export-status').hidden = true;
  }
});
function lockForm(value) {
  for(const id of ['paste-tab','upload-tab','text','file','language','clear','example','remove-file','jsonl-mode','preview-item','preview-language'])$(id).disabled=value;
  $('submit').disabled=value||!ready;
  $('analyze-form').setAttribute('aria-busy',String(value));$('loading-note').hidden=!working;
  if(!value)renderPreview();
}
$('stop-batch').addEventListener('click',()=>{stopRequested=true;$('stop-batch').disabled=true;$('stop-batch').textContent='Stopping after current item…';});
$('analyze-form').addEventListener('submit',async event=>{
  event.preventDefault();if(working||preparing||!ready)return;
  $('error').hidden=true;
  if(mode==='upload'&&!items.length)return showError('Choose files and wait for their text previews.');
  if(mode==='paste'&&!$('text').value.trim())return showError('Paste some text before analyzing.');
  working=true;stopRequested=false;lockForm(true);clearResult();
  let jobs=mode==='upload'?items.map(i=>({...i})):[{name:'Pasted text',text:$('text').value,notes:[]}];
  const globalLanguage=$('language').value;
  $('stop-batch').hidden=jobs.length<2;$('stop-batch').disabled=false;$('stop-batch').textContent='Stop after current item';
  $('batch-results').hidden=jobs.length<2;
  const rows=jobs.map(job=>{
    const row=document.createElement('div');row.className='batch-row';
    const name=document.createElement('span');name.textContent=job.name;
    const status=document.createElement('span');status.className='batch-outcome';status.textContent='Queued';
    const view=document.createElement('button');view.type='button';view.textContent='View result';view.hidden=true;
    row.append(name,status,view);$('batch-list').append(row);return {status,view};
  });
  let completed=0,failed=0;
  try {
    for(let i=0;i<jobs.length;i++) {
      if(stopRequested){for(let j=i;j<jobs.length;j++)rows[j].status.textContent='Not analyzed';break;}
      const job=jobs[i],row=rows[i];row.status.textContent='Analyzing…';
      $('submit-label').textContent=`Analyzing ${i+1} / ${jobs.length}…`;
      $('batch-status').textContent=`${completed} complete · ${failed} failed · ${jobs.length-i} remaining`;
      const started=performance.now();
      try {
        let language=globalLanguage;
        if(language==='auto')language=job.language||(await jsonRequest('/api/suggest-language',{text:job.text})).language;
        const data=await jsonRequest('/api/analyze',{text:job.text,language});
        data.warnings.push(...job.notes);const elapsed=(performance.now()-started)/1000;
        row.status.textContent=`${data.label==='ai'?'AI':'Human'}-leaning · ${(data.scores.ai*100).toFixed(1)}% AI${data.family?' · '+data.family.label:''}`;
        row.view.hidden=false;row.view.addEventListener('click',()=>showResult(data,elapsed,job.name));
        showResult(data,elapsed,job.name);completed++;
      }catch(e){failed++;row.status.textContent=e.message;if(jobs.length===1)showError(e.message);}
    }
  }finally{
    $('batch-status').textContent=`${completed} complete · ${failed} failed${stopRequested?' · stopped':''}. Select an item to view its result and download its PDF.`;
    working=false;lockForm(false);$('submit-label').textContent='Analyze content';$('stop-batch').hidden=true;
  }
});
function showError(message){$('error').textContent=message;$('error').hidden=false;}
async function health(){
  try{ready=(await fetch('/api/health')).ok;}catch{ready=false;}
  $('model-status').textContent=ready?'Model ready':'Model unavailable';$('model-status').className=`status ${ready?'ready':'unavailable'}`;
  $('submit').disabled=working||preparing||!ready;if(!ready)setTimeout(health,10000);
}
health();
