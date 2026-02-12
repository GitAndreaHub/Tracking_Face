const uploadBtn = document.getElementById('uploadBtn');
const processBtn = document.getElementById('processBtn');
const applyBtn = document.getElementById('applyBtn');
const uploadStatus = document.getElementById('uploadStatus');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
const intensity = document.getElementById('intensity');
const intensityVal = document.getElementById('intensityVal');
const resultVideo = document.getElementById('resultVideo');
const downloadLink = document.getElementById('downloadLink');

let uploadedVideoName = null;
let trackingJobId = null;

intensity.addEventListener('input', () => {
  intensityVal.textContent = intensity.value;
});

uploadBtn.addEventListener('click', async () => {
  const file = document.getElementById('videoFile').files[0];
  if (!file) return alert('Choose a video first');

  const form = new FormData();
  form.append('video', file);

  uploadStatus.textContent = 'Uploading...';
  const res = await fetch('/api/upload', { method: 'POST', body: form });
  const data = await res.json();

  if (!res.ok) {
    uploadStatus.textContent = data.detail || 'Upload failed';
    return;
  }

  uploadedVideoName = data.video_path;
  uploadStatus.textContent = `Uploaded: ${uploadedVideoName}`;
});

processBtn.addEventListener('click', async () => {
  if (!uploadedVideoName) return alert('Upload a video first');

  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return alert('Enter a tracking prompt');

  const form = new FormData();
  form.append('video_path', uploadedVideoName);
  form.append('prompt', prompt);

  const res = await fetch('/api/process', { method: 'POST', body: form });
  const data = await res.json();
  if (!res.ok) return alert(data.detail || 'Failed to start processing');

  trackingJobId = data.job_id;
  pollStatus();
});

async function pollStatus() {
  if (!trackingJobId) return;
  const res = await fetch(`/api/status/${trackingJobId}`);
  const data = await res.json();

  if (!res.ok) {
    progressText.textContent = 'Status check failed';
    return;
  }

  progressBar.value = Math.round((data.progress || 0) * 100);
  progressText.textContent = data.message || data.state;

  if (data.state === 'ready') {
    if (data.preview_url) {
      resultVideo.src = `${data.preview_url}?t=${Date.now()}`;
      resultVideo.load();
    }
    return;
  }

  if (data.state === 'error') {
    alert(`Processing error: ${data.message}`);
    return;
  }

  setTimeout(pollStatus, 1000);
}

applyBtn.addEventListener('click', async () => {
  if (!trackingJobId) return alert('Run processing first');

  const payload = {
    job_id: trackingJobId,
    effect: document.getElementById('effect').value,
    intensity: parseInt(intensity.value, 10),
  };

  const res = await fetch('/api/effect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) return alert(data.detail || 'Failed to apply effect');

  resultVideo.src = `${data.video_url}?t=${Date.now()}`;
  resultVideo.load();
  downloadLink.href = data.download_url;
});
