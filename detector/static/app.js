'use strict';
const $ = id => document.getElementById(id);
const MAX_FILE = 10 * 1024 * 1024;
let mode='paste', working=false, ready=false, exporting=false, preparing=false, selectedFiles=[], items=[], lastReport=null, revision=0, stopRequested=false, cancelRequested=false, activeJobCount=0;
const examplesByLanguage = {
  en: [
    'I went back to the little bookshop near our old house yesterday. The owner still remembered the mystery novels I used to borrow after school.',
    'The rain started just as we reached the station. We waited beneath the old clock, sharing a packet of warm roasted peanuts.',
    'My grandmother keeps a small garden behind her kitchen. Every morning she checks the tomato plants and waters the pots before making tea.',
    'The new science museum has a room where visitors can build simple machines. I spent an hour testing bridges made from paper and tape.',
    'We tried making vegetable soup from whatever was left in the fridge. A little lemon and fresh coriander made the whole kitchen smell bright.',
    'Our school band practiced in the auditorium after classes. The drummer kept losing the beat, but by sunset everyone was playing together.',
    'The trail followed a quiet stream through the hills. We stopped at a wooden bridge to watch leaves drift slowly beneath our feet.',
    'A tiny repair shop opened beside the market last month. The owner fixed my grandfather’s radio and showed me how its tuning dial works.'
  ],
  hi: [
    'सुबह की हल्की धूप खिड़की से कमरे में आई। मैंने चाय बनाई और बालकनी में रखे तुलसी के पौधों को पानी दिया। दिन की शुरुआत बहुत शांत थी।',
    'रविवार को हम सब पास के पुराने किले देखने गए। रास्ते में बच्चों ने पेड़ों की तस्वीरें बनाईं और दादाजी ने हमें उस जगह की कहानी सुनाई।',
    'हमारे मोहल्ले के पुस्तकालय में नई किताबें आई हैं। शाम को मैं वहाँ बैठकर कहानी पढ़ता हूँ और घर लौटते समय अपने छोटे भाई के लिए भी किताब लेता हूँ।',
    'बारिश रुकने के बाद सड़क के किनारे छोटे छोटे पोखर चमक रहे थे। हम धीरे धीरे चलते हुए बाजार पहुँचे और गरम समोसे लेकर घर लौट आए।',
    'विद्यालय के बगीचे में इस महीने कई रंगों के फूल खिले हैं। हर कक्षा के बच्चों ने एक पौधा चुना और उसकी देखभाल की जिम्मेदारी बाँट ली।'
  ],
  ta: [
    'காலை நேரத்தில் எங்கள் வீட்டின் அருகிலுள்ள பூங்காவிற்கு நடந்துசென்றேன். அங்கே குழந்தைகள் விளையாடினார்கள், பறவைகள் மரக்கிளைகளில் அமர்ந்து இனிமையாகக் குரல் கொடுத்தன.',
    'ஞாயிற்றுக்கிழமை நாங்கள் குடும்பத்துடன் கடற்கரைக்குச் சென்றோம். அலைகளைப் பார்த்துக்கொண்டே நடந்தோம், பிறகு அருகிலிருந்த கடையில் சூடான தேநீர் குடித்தோம்.',
    'எங்கள் ஊர் நூலகத்தில் புதிய புத்தகங்கள் வந்துள்ளன. பள்ளி முடிந்ததும் அங்கே சென்று ஒரு கதையைப் படிப்பேன், பின்னர் நண்பர்களுடன் அதைப் பற்றி பேசுவேன்.',
    'மழை நின்ற பிறகு தெருவெங்கும் குளிர்ந்த காற்று வீசியது. அம்மா செய்த காய்கறி சூப்பை அனைவரும் சேர்ந்து குடித்தோம், வீட்டில் நல்ல மணம் பரவியது.',
    'பள்ளியின் அறிவியல் கண்காட்சிக்காக நாங்கள் ஒரு சிறிய நீர்வடிகட்டியை உருவாக்கினோம். மணல், கற்கள், பருத்தி ஆகியவற்றைப் பயன்படுத்தி தண்ணீரை எப்படி சுத்தம் செய்யலாம் என்று பார்த்தோம்.'
  ],
  kn: [
    'ಬೆಳಿಗ್ಗೆ ನಮ್ಮ ಮನೆಯ ಹತ್ತಿರದ ಉದ್ಯಾನವನಕ್ಕೆ ನಡೆದುಕೊಂಡು ಹೋದೆ. ಅಲ್ಲಿ ಮಕ್ಕಳು ಆಟವಾಡುತ್ತಿದ್ದರು, ಮರಗಳ ಮೇಲೆ ಕುಳಿತಿದ್ದ ಹಕ್ಕಿಗಳು ಮಧುರವಾಗಿ ಕೂಗುತ್ತಿದ್ದವು.',
    'ಭಾನುವಾರ ನಾವು ಕುಟುಂಬದೊಂದಿಗೆ ಹಳೆಯ ಕೋಟೆಯನ್ನು ನೋಡಲು ಹೋದೆವು. ದಾರಿಯಲ್ಲಿ ತಾತ ಅದರ ಇತಿಹಾಸದ ಕಥೆ ಹೇಳಿದರು, ಮಕ್ಕಳು ಗೋಡೆಗಳ ಚಿತ್ರಗಳನ್ನು ತೆಗೆದರು.',
    'ನಮ್ಮ ಊರಿನ ಗ್ರಂಥಾಲಯಕ್ಕೆ ಹೊಸ ಪುಸ್ತಕಗಳು ಬಂದಿವೆ. ಶಾಲೆಯ ನಂತರ ಅಲ್ಲಿ ಕುಳಿತು ಕಥೆ ಓದುತ್ತೇನೆ, ಮನೆಗೆ ಹಿಂತಿರುಗಿದಾಗ ಅದನ್ನು ತಂಗಿಯೊಂದಿಗೆ ಹಂಚಿಕೊಳ್ಳುತ್ತೇನೆ.',
    'ಮಳೆ ನಿಂತ ಮೇಲೆ ರಸ್ತೆಯ ಪಕ್ಕದ ಗಿಡಗಳ ಮೇಲೆ ನೀರಿನ ಹನಿಗಳು ಹೊಳೆಯುತ್ತಿದ್ದವು. ನಾವು ನಿಧಾನವಾಗಿ ಮಾರುಕಟ್ಟೆಗೆ ನಡೆದು ತಾಜಾ ಹಣ್ಣುಗಳನ್ನು ತೆಗೆದುಕೊಂಡು ಮನೆಗೆ ಬಂದೆವು.',
    'ಶಾಲೆಯ ವಿಜ್ಞಾನ ಪ್ರದರ್ಶನಕ್ಕಾಗಿ ನಾವು ಕಾಗದ ಮತ್ತು ಮರದ ಕಡ್ಡಿಗಳಿಂದ ಒಂದು ಸೇತುವೆ ಮಾಡಿದೆವು. ಹೆಚ್ಚು ತೂಕವನ್ನು ತಾಳುವಂತೆ ಅದರ ಆಕಾರವನ್ನು ಹಲವು ಬಾರಿ ಬದಲಾಯಿಸಿ ನೋಡಿದೆವು.'
  ],
  hi_en: [
    'Kal subah main apne dost ke saath station gaya. Train late thi, isliye humne platform par chai pi aur weekend ke plans ke baare mein baat ki.',
    'Meri nani ke ghar ke peeche ek chhota garden hai. Har Sunday hum plants ko paani dete hain aur phir fresh nimbu paani banate hain.',
    'School ke baad hum local library gaye. Maine ek adventure story choose ki, aur mere bhai ne animals par ek interesting book li.',
    'Aaj market mein bahut bheed thi, phir bhi humne sabzi aur fruit aaram se kharide. Ghar aakar sabne milkar dinner prepare kiya.',
    'Hamari class ne science fair ke liye ek simple water filter banaya. Sabne apna idea share kiya, aur project dekhne ke liye parents bhi aaye.'
  ],
  ta_en: [
    'Nethu evening naan friends kooda beach ku ponen. Waves romba calm ah irundhuchu, so naanga sunset paathutu nearby shop la tea kudichom.',
    'Innaiku school mudinja apram naan library ku ponen. Oru new story book eduthen, appuram homework finish panna quiet ah konjam time irundhuchu.',
    'Sunday morning amma kooda market ponen. Fresh vegetables vaanginom, veetukku vandhu ellarum serndhu lunch prepare pannom.',
    'Enga class science fair ku small rainwater model create pannuchu. Each student oru idea share pannanga, teacher project pathi explain panna sonnaanga.',
    'Bus late aana nala naanga stop la wait pannom. Friend oru funny story sonna, time fast ah pochu and finally bus vandhuduchu.'
  ],
  kn_en: [
    'Ninne evening naanu friends jothe park ge hode. Aakasha tumba chennagittu, naavu bench mele kootu music keli swalpa hotthu maatadidevu.',
    'Ivattu school mugida mele library ge hode. Ondu interesting story book tagonde, manege bandu adara bagge tangige helide.',
    'Sunday morning naanu amma jothe market ge hode. Fresh vegetables tandvi, manege bandu ellaru seri tasty lunch prepare madidevu.',
    'Namma class science fair ge small water filter model maditu. Prati student ondond idea helidaru, teacher ella projects nodi appreciate madidaru.',
    'Bus barodu late aayitu, aadaroo stop alli wait madta friends jothe maatadide. Swalpa time aadmele bus bantu, naavu safe aagi manege bandevu.'
  ]
};
const exampleIndexes = Object.fromEntries(Object.keys(examplesByLanguage).map(language=>[language,-1]));
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
  $('error').hidden=true;
}
for(const key of ['paste','upload']) {
  $(`${key}-tab`).addEventListener('click',()=>setMode(key));
  $(`${key}-tab`).addEventListener('keydown',e=>{
    if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)) {
      e.preventDefault();setMode(e.key==='Home'?'paste':e.key==='End'?'upload':mode==='paste'?'upload':'paste');$(`${mode}-tab`).focus();
    }
  });
}
function count() { const text=$('text').value; $('word-count').textContent=`${text.trim().split(/\s+/u).filter(Boolean).length.toLocaleString()} words · ${[...text].length.toLocaleString()} / 500,000 chars`; }
$('text').addEventListener('input',()=>{count();clearResult();});
$('language').addEventListener('change',clearResult);
$('example').addEventListener('click',()=>{
  const selectedLanguage=$('language').value;
  const language=examplesByLanguage[selectedLanguage]?selectedLanguage:'en';
  const examples=examplesByLanguage[language];
  const current=$('text').value;
  for(let offset=1;offset<=examples.length;offset++){
    const next=(exampleIndexes[language]+offset)%examples.length;
    if(examples[next]!==current){exampleIndexes[language]=next;break;}
  }
  $('text').value=examples[exampleIndexes[language]];count();clearResult();$('error').hidden=true;
});
$('clear').addEventListener('click',()=>{$('text').value='';count();clearResult();$('error').hidden=true;$('text').focus();});
function renderPreview() {
  $('preview').hidden=!items.length;
  if(!items.length)return;
  const item=items[Number($('preview-item').value)||0];
  $('preview-text').value=item.text;
  $('preview-notes').textContent=`${items.length} item(s) ready. ${item.notes.join(' ')}`;

}
$('preview-item').addEventListener('change',renderPreview);
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
        if(items.length+data.documents.length>2000)throw new Error('Batch limit is 2,000 items; split this file into a smaller batch.');
        for(const doc of data.documents)items.push({...doc,name:file.name+(doc.name?' / '+doc.name:''),filename:file.name,recordName:doc.name});
      }catch(e){errors.push(`${file.name}: ${e.message}`);}
    }
    if(current!==revision)return;
    items.forEach((item,i)=>$('preview-item').add(new Option(item.name,String(i))));
    if(errors.length)showError(errors.join(' '));
  }finally{preparing=false;lockForm(false);$('preparing').hidden=true;renderPreview();}
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
  for(const id of ['paste-tab','upload-tab','text','file','language','clear','example','remove-file','jsonl-mode','preview-item'])$(id).disabled=value;
  $('submit').disabled=value||!ready;
  $('analyze-form').setAttribute('aria-busy',String(value));$('loading-note').hidden=!working;
  if(!value)renderPreview();
}
$('stop-batch').addEventListener('click',async()=>{
  if(activeJobCount===1){
    cancelRequested=true;$('stop-batch').disabled=true;$('stop-batch').textContent='Stopping analysis…';
    try{await fetch('/api/cancel',{method:'POST'});}catch{}
    return;
  }
  stopRequested=true;$('stop-batch').disabled=true;$('stop-batch').textContent='Stopping after current item…';
});
$('analyze-form').addEventListener('submit',async event=>{
  event.preventDefault();if(working||preparing||!ready)return;
  $('error').hidden=true;
  if(mode==='upload'&&!items.length)return showError('Choose files and wait for their text previews.');
  if(mode==='paste'&&!$('text').value.trim())return showError('Paste some text before analyzing.');
  working=true;stopRequested=false;cancelRequested=false;lockForm(true);clearResult();
  let jobs=mode==='upload'?items.map(i=>({...i})):[{name:'Pasted text',text:$('text').value,notes:[]}];
  activeJobCount=jobs.length;
  const globalLanguage=$('language').value;
  $('stop-batch').hidden=false;$('stop-batch').disabled=false;$('stop-batch').textContent=jobs.length===1?'Stop analysis':'Stop after current item';
  $('batch-results').hidden=jobs.length<2;
  let selectedIndex=-1;
  const rows=jobs.map((job,index)=>{
    const view=document.createElement('button');view.type='button';view.className='batch-card';view.disabled=true;
    view.setAttribute('aria-pressed','false');
    const number=document.createElement('span');number.className='batch-number';number.textContent=String(index+1).padStart(2,'0');
    const body=document.createElement('span');body.className='batch-card-body';
    const name=document.createElement('strong');name.className='batch-name';name.textContent=job.recordName||job.name;name.title=name.textContent;
    const file=document.createElement('span');file.className='batch-file';file.textContent=job.recordName?job.filename:'';file.title=file.textContent;file.hidden=!job.recordName;
    const status=document.createElement('span');status.className='batch-outcome';status.textContent='Queued';
    const action=document.createElement('span');action.className='batch-action';action.textContent='Waiting';
    body.append(name,file,status,action);view.append(number,body);$('batch-list').append(view);
    return {status,view,action};
  });
  function selectResult(index,data,elapsed) {
    selectedIndex=index;
    rows.forEach((row,i)=>{row.view.setAttribute('aria-pressed',String(i===index));if(!row.view.disabled)row.action.textContent=i===index?'Selected · analysis below':'View analysis →';});
    if(mode==='upload'){$('preview-item').value=String(index);renderPreview();}
    showResult(data,elapsed,jobs[index].name);
    $('result-source').textContent=(jobs.length>1?`Viewing item ${index+1} of ${jobs.length} · `:'')+jobs[index].name;
  }
  let completed=0,failed=0,cancelled=false;
  try {
    for(let i=0;i<jobs.length;i++) {
      if(stopRequested){for(let j=i;j<jobs.length;j++){rows[j].status.textContent='Not analyzed';rows[j].action.textContent='Stopped';}break;}
      const job=jobs[i],row=rows[i];row.status.textContent='Analyzing…';row.action.textContent='Processing';
      $('submit-label').textContent=`Analyzing ${i+1} / ${jobs.length}…`;
      $('batch-status').textContent=`${completed} complete · ${failed} failed · ${jobs.length-i} remaining`;
      const started=performance.now();
      try {
        const language=globalLanguage;
        const data=await jsonRequest('/api/analyze',{text:job.text,language});
        data.warnings.push(...job.notes);const elapsed=(performance.now()-started)/1000;
        row.status.textContent=`${data.label==='ai'?'AI':'Human'}-leaning · ${(data.scores.ai*100).toFixed(1)}% AI${data.family?' · '+data.family.label:''}`;
        row.view.disabled=false;row.action.textContent='View analysis →';row.view.addEventListener('click',()=>selectResult(i,data,elapsed));
        if(selectedIndex===-1)selectResult(i,data,elapsed);completed++;
      }catch(e){
        if(cancelRequested){cancelled=true;row.action.textContent='Stopped';row.status.textContent='Analysis stopped';if(jobs.length===1)showError('Analysis stopped.');break;}
        failed++;row.view.classList.add('failed');row.action.textContent='Could not analyze';row.status.textContent=e.message;if(jobs.length===1)showError(e.message);
      }
    }
  }finally{
    $('batch-status').textContent=`${completed} complete · ${failed} failed${stopRequested||cancelled?' · stopped':''}`;
    working=false;activeJobCount=0;lockForm(false);$('submit-label').textContent='Analyze content';$('stop-batch').hidden=true;
  }
});
function showError(message){$('error').textContent=message;$('error').hidden=false;}
async function health(){
  try{ready=(await fetch('/api/health')).ok;}catch{ready=false;}
  $('model-status').textContent=ready?'Model ready':'Model unavailable';$('model-status').className=`status ${ready?'ready':'unavailable'}`;
  $('submit').disabled=working||preparing||!ready;if(!ready)setTimeout(health,10000);
}
health();
