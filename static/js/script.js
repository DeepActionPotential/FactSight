class IntroController {
  constructor() {
    this.state = { uploadedImages: [], isAnalyzing: false };
    this.cacheEls();
    if (!this.el.newsForm) return;
    this.bind();
  }
  cacheEls() {
    this.el = {
      newsForm: document.getElementById('newsForm'),
      newsText: document.getElementById('newsText'),
      fileInput: document.getElementById('fileInput'),
      uploadArea: document.getElementById('uploadArea'),
      browseBtn: document.getElementById('browseBtn'),
      imagePreviews: document.getElementById('imagePreviews'),
      analyzeBtn: document.getElementById('analyzeBtn'),
      clearBtn: document.getElementById('clearBtn'),
      btnLoader: document.getElementById('btnLoader'),
      charCounter: document.getElementById('charCounter'),
      messageContainer: document.getElementById('messageContainer')
    };
  }
  bind() {
    this.el.newsText.addEventListener('input', () => this.updateCharCount());
    this.el.browseBtn.addEventListener('click', () => this.el.fileInput.click());
    this.el.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
    this.el.uploadArea.addEventListener('click', () => this.el.fileInput.click());
    this.el.uploadArea.addEventListener('dragover', (e) => this.dragOver(e));
    this.el.uploadArea.addEventListener('dragleave', (e) => this.dragLeave(e));
    this.el.uploadArea.addEventListener('drop', (e) => this.drop(e));
    this.el.newsForm.addEventListener('submit', (e) => this.submit(e));
    this.el.clearBtn.addEventListener('click', () => this.clearAll());
  }
  updateCharCount() {
    const len = this.el.newsText.value.length;
    this.el.charCounter.textContent = `${len}/10000`;
    if (len > 9000) this.el.charCounter.style.color = 'var(--accent-error)';
    else if (len > 7000) this.el.charCounter.style.color = 'var(--accent-warning)';
    else this.el.charCounter.style.color = 'var(--text-muted)';
  }
  dragOver(e){ e.preventDefault(); this.el.uploadArea.classList.add('drag-over'); }
  dragLeave(e){ e.preventDefault(); this.el.uploadArea.classList.remove('drag-over'); }
  drop(e){ e.preventDefault(); this.el.uploadArea.classList.remove('drag-over'); this.processFiles(Array.from(e.dataTransfer.files)); }
  handleFileSelect(e){ this.processFiles(Array.from(e.target.files)); this.el.fileInput.value = ''; }
  processFiles(files){
    files.forEach(file=>{
      if(!file.type.startsWith('image/')){ this.toast('Invalid File','Please upload only image files','error'); return; }
      if(file.size > 10*1024*1024){ this.toast('File Too Large','Image must be smaller than 10MB','error'); return; }
      const reader=new FileReader();
      reader.onload=(ev)=>{
        const imageData={ id:Date.now()+Math.random(), name:file.name, size:this.hSize(file.size), data:ev.target.result };
        this.state.uploadedImages.push(imageData);
        this.renderPreview(imageData);
      };
      reader.readAsDataURL(file);
    });
  }
  hSize(bytes){
    if(bytes===0) return '0 B';
    const k=1024, sizes=['B','KB','MB']; const i=Math.floor(Math.log(bytes)/Math.log(k));
    return `${parseFloat((bytes/Math.pow(k,i)).toFixed(1))} ${sizes[i]}`;
  }
  renderPreview(imageData){
    const el=document.createElement('div');
    el.className='preview-item';
    el.innerHTML=`
      <img src="${imageData.data}" alt="${imageData.name}" class="preview-image">
      <button type="button" class="remove-btn" data-id="${imageData.id}">×</button>
    `;
    this.el.imagePreviews.appendChild(el);
    el.querySelector('.remove-btn').addEventListener('click',()=>this.removeImage(imageData.id));
  }
  removeImage(id){
    this.state.uploadedImages=this.state.uploadedImages.filter(x=>x.id!==id);
    const btn=this.el.imagePreviews.querySelector(`[data-id="${id}"]`);
    btn?.closest('.preview-item')?.remove();
  }
  async submit(e){
    e.preventDefault();
    if(!this.validate()){ this.toast('Missing Information','Please add both text and at least one image.','error'); return; }
    if(this.state.isAnalyzing) return;
    this.state.isAnalyzing=true; this.el.analyzeBtn.disabled=true; this.el.btnLoader.classList.add('active');
    this.el.analyzeBtn.querySelector('.btn-text').textContent='Analyzing...';
    try{
      const payload={ text:this.el.newsText.value.trim(), images:this.state.uploadedImages.map(img=>({name:img.name,data:img.data})) };
      const res=await fetch('/analyze',{ method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
      const json=await res.json();
      if(json.success){ window.location.href=`/analysis/${json.analysis_id}`; }
      else{ throw new Error(json.error || 'Analysis failed'); }
    }catch(err){
      console.error(err); this.toast('Analysis Failed', err.message,'error');
    }finally{
      this.state.isAnalyzing=false; this.el.analyzeBtn.disabled=false; this.el.btnLoader.classList.remove('active');
      this.el.analyzeBtn.querySelector('.btn-text').textContent='Analyze News';
    }
  }
  validate(){ return this.el.newsText.value.trim().length>0 && this.state.uploadedImages.length>0; }
  clearAll(){
    this.el.newsText.value=''; this.state.uploadedImages=[]; this.el.imagePreviews.innerHTML=''; this.updateCharCount();
    this.toast('Cleared','All inputs have been reset.','success');
  }
  toast(title,text,type='info'){
    const msg=document.createElement('div'); msg.className=`message ${type}`;
    msg.innerHTML=`<div class="message-content"><div class="message-title">${title}</div><div class="message-text">${text}</div></div><button class="message-close">×</button>`;
    const root=document.getElementById('messageContainer') || document.body; root.appendChild(msg);
    setTimeout(()=>msg.classList.add('show'),50);
    msg.querySelector('.message-close').addEventListener('click',()=>this.hideToast(msg));
    setTimeout(()=>this.hideToast(msg),5000);
  }
  hideToast(msg){ msg.classList.remove('show'); setTimeout(()=>msg.remove(),250); }
}

class OverlayController {
  constructor() {
    const dataEl = document.getElementById('analysis-data');
    if (!dataEl) return;
    try { this.analysis = JSON.parse(dataEl.textContent || '{}'); }
    catch { this.analysis = null; }
    if (!this.analysis || !Array.isArray(this.analysis.images)) return;
    this.initOverlays();
    window.addEventListener('resize', () => this.redrawAll());
  }
  initOverlays() {
    const imgs = document.querySelectorAll('.image-with-overlay');
    imgs.forEach(img => {
      if (img.complete) this.setupCanvasFor(img);
      else img.onload = () => this.setupCanvasFor(img);
    });
  }
  setupCanvasFor(imgEl) {
    const idx = parseInt(imgEl.dataset.index, 10);
    const canvas = document.querySelector(`.overlay-canvas[data-index="${idx}"]`);
    if (!canvas) return;
    const rect = imgEl.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width  = Math.floor(rect.width  * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const natW = imgEl.naturalWidth || rect.width;
    const natH = imgEl.naturalHeight || rect.height;
    const sx = rect.width / natW;
    const sy = rect.height / natH;
    this.drawFaces(ctx, this.analysis.images[idx], sx, sy);
  }
  redrawAll() {
    document.querySelectorAll('.image-with-overlay').forEach(img => this.setupCanvasFor(img));
  }
  drawFaces(ctx, imageAnalysis, sx, sy) {
    const faces = imageAnalysis.faces || [];
    const deepfakes = imageAnalysis.deepfake_faces || [];
    const BOX_COLOR = '#ef4444';
    const BOX_COLOR_DF = '#f59e0b';
    const POINT_COLOR = '#10b981';
    const POINT_COLOR_DF = '#f59e0b';
    const LINE_W = 2;
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    faces.forEach((f, i) => {
      const isDF = !!deepfakes[i];
      const stroke = isDF ? BOX_COLOR_DF : BOX_COLOR;
      const dot    = isDF ? POINT_COLOR_DF : POINT_COLOR;
      ctx.lineWidth = LINE_W;
      ctx.strokeStyle = stroke;
      const x1 = f.box_start_point[0] * sx + 0.5;
      const y1 = f.box_start_point[1] * sy + 0.5;
      const x2 = f.box_end_point[0]   * sx + 0.5;
      const y2 = f.box_end_point[1]   * sy + 0.5;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const drawPoint = (pt) => {
        const px = pt[0] * sx, py = pt[1] * sy, size = 4;
        ctx.fillStyle = dot; ctx.fillRect(px - size/2, py - size/2, size, size);
      };
      drawPoint(f.left_eye);
      drawPoint(f.right_eye);
      drawPoint(f.nose);
      drawPoint(f.left_mouth);
      drawPoint(f.right_mouth);
    });
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new IntroController();
  new OverlayController();
});
