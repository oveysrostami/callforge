const state = { offset: 0, limit: 30, total: 0, selectedId: null, selectedItem: null, debounce: null };

const $ = (id) => document.getElementById(id);
const list = $("file-list");

const statusLabels = {
  completed: "متن آماده",
  pending: "در انتظار",
  running: "در حال پردازش",
  failed: "ناموفق",
};

const directionLabels = {
  inbound: "تماس ورودی",
  outbound: "تماس خروجی",
  internal: "داخلی به داخلی",
};

function faNumber(value) {
  return new Intl.NumberFormat("fa-IR").format(value ?? 0);
}

function duration(value) {
  if (value === null || value === undefined) return "—";
  const seconds = Math.round(value);
  const mins = Math.floor(seconds / 60);
  const rest = seconds % 60;
  const restFa = String(rest).padStart(2, "0").replace(/\d/g, (digit) => "۰۱۲۳۴۵۶۷۸۹"[Number(digit)]);
  return `${faNumber(mins)}:${restFa}`;
}

function fileSize(value) {
  if (value === null || value === undefined) return "—";
  const units = ["بایت", "KB", "MB", "GB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) { size /= 1024; unit += 1; }
  return `${new Intl.NumberFormat("fa-IR", { maximumFractionDigits: 1 }).format(size)} ${units[unit]}`;
}

function dateTime(value) {
  if (!value) return "نامشخص";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return value;
  return new Intl.DateTimeFormat("fa-IR", { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function badge(status) {
  const span = document.createElement("span");
  span.className = `status-badge status-${status || "unknown"}`;
  span.textContent = statusLabels[status] || "نامشخص";
  return span;
}

async function api(path) {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || `خطای ${response.status}`);
  }
  return response.json();
}

async function post(path, payload = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      "X-CallForge-UI": "1",
    },
    body: JSON.stringify(payload),
  });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(result.error || `خطای ${response.status}`);
  return result;
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.remove("hidden");
  window.setTimeout(() => toast.classList.add("hidden"), 3500);
}

async function loadStats() {
  const stats = await api("/api/stats");
  $("stat-files").textContent = faNumber(stats.audio_files || 0);
  $("stat-transcripts").textContent = faNumber(stats.current_transcripts || 0);
  $("stat-pending").textContent = faNumber(stats.pending || 0);
  $("stat-failed").textContent = faNumber(stats.failed || 0);
}

function queryString() {
  const params = new URLSearchParams({ limit: state.limit, offset: state.offset });
  const values = {
    q: $("search").value.trim(),
    direction: $("direction").value,
    status: $("status").value,
    transcript: $("transcript-filter").value,
  };
  Object.entries(values).forEach(([key, value]) => { if (value) params.set(key, value); });
  return params.toString();
}

function renderRows(items) {
  list.replaceChildren();
  for (const item of items) {
    const row = document.createElement("tr");
    row.tabIndex = 0;
    row.dataset.id = item.id;
    if (item.id === state.selectedId) row.classList.add("selected");

    const call = document.createElement("td");
    const primary = document.createElement("span");
    primary.className = "file-primary";
    primary.textContent = item.filename;
    const secondary = document.createElement("span");
    secondary.className = "file-secondary";
    secondary.textContent = [directionLabels[item.direction] || "جهت نامشخص", item.remote_number].filter(Boolean).join(" · ");
    call.append(primary, secondary);

    const time = document.createElement("td");
    time.textContent = dateTime(item.recorded_at);
    const length = document.createElement("td");
    length.textContent = duration(item.duration_seconds);
    const status = document.createElement("td");
    status.append(badge(item.job_status));
    row.append(call, time, length, status);
    row.addEventListener("click", () => selectFile(item.id));
    row.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") selectFile(item.id); });
    list.append(row);
  }
}

async function loadFiles() {
  try {
    const payload = await api(`/api/files?${queryString()}`);
    state.total = payload.total;
    renderRows(payload.items);
    $("empty-state").classList.toggle("hidden", payload.items.length > 0);
    $("result-count").textContent = `${faNumber(payload.total)} نتیجه`;
    const page = Math.floor(state.offset / state.limit) + 1;
    const pages = Math.max(1, Math.ceil(payload.total / state.limit));
    $("page-label").textContent = `صفحه ${faNumber(page)} از ${faNumber(pages)}`;
    $("prev-page").disabled = state.offset === 0;
    $("next-page").disabled = state.offset + state.limit >= payload.total;
  } catch (error) {
    $("connection").classList.add("offline");
    showToast(error.message);
  }
}

function metadataItem(label, value) {
  const wrapper = document.createElement("div");
  const term = document.createElement("dt");
  const description = document.createElement("dd");
  term.textContent = label;
  description.textContent = value ?? "—";
  wrapper.append(term, description);
  return wrapper;
}

function isInteractivelyQueued(item) {
  return item.job_status === "pending" && Number(item.job_priority || 0) >= 1000;
}

function updateTranscribeButton(item) {
  const button = $("transcribe-button");
  const label = button.querySelector("span");
  const processing = item.job_status === "running" || isInteractivelyQueued(item);
  button.disabled = processing;
  if (item.job_status === "running") label.textContent = "در حال تبدیل…";
  else if (isInteractivelyQueued(item)) label.textContent = "در صف پردازش";
  else if (item.transcript_id) label.textContent = "تبدیل مجدد";
  else label.textContent = "تبدیل به متن";
}

async function selectFile(id, reloadAudio = true) {
  state.selectedId = id;
  document.querySelectorAll("tbody tr").forEach((row) => row.classList.toggle("selected", Number(row.dataset.id) === id));
  try {
    const item = await api(`/api/files/${id}`);
    if (state.selectedId !== id) return;
    state.selectedItem = item;
    $("detail-empty").classList.add("hidden");
    $("detail-content").classList.remove("hidden");
    $("detail-direction").textContent = directionLabels[item.direction] || "جهت نامشخص";
    $("detail-name").textContent = item.filename;
    $("detail-path").textContent = item.relative_path;
    const status = $("detail-status");
    status.className = `status-badge status-${item.job_status || "unknown"}`;
    status.textContent = statusLabels[item.job_status] || "نامشخص";
    if (reloadAudio) {
      const player = $("audio-player");
      player.pause();
      player.src = item.audio_url;
      player.load();
    }
    updateTranscribeButton(item);

    const metadata = $("metadata");
    metadata.replaceChildren(
      metadataItem("زمان تماس", dateTime(item.recorded_at)),
      metadataItem("مدت", duration(item.duration_seconds)),
      metadataItem("حجم", fileSize(item.size_bytes)),
      metadataItem("داخلی", item.agent_extension),
      metadataItem(item.direction === "internal" ? "داخلی مقابل" : "شماره مقابل", item.remote_number),
      metadataItem("نرخ نمونه", item.sample_rate ? `${faNumber(item.sample_rate)} Hz` : "—"),
      metadataItem("کانال", item.channels ? faNumber(item.channels) : "—"),
      metadataItem("Bitrate", item.bitrate ? `${faNumber(Math.round(item.bitrate / 1000))} kbps` : "—"),
      metadataItem("تعداد نامفهوم", faNumber(item.unclear_count || 0)),
    );

    const hasTranscript = Boolean(item.transcript_id && item.transcript_content);
    $("transcript").classList.toggle("hidden", !hasTranscript);
    $("no-transcript").classList.toggle("hidden", hasTranscript);
    $("transcript").textContent = item.transcript_content || "";
    $("transcript-version").textContent = hasTranscript ? `نسخه ${faNumber(item.transcript_version)} · ${dateTime(item.transcript_created_at)}` : "";
  } catch (error) {
    showToast(error.message);
  }
}

async function transcribeSelected() {
  const item = state.selectedItem;
  if (!item) return;
  if (item.transcript_id) {
    const confirmed = window.confirm("متن فعلی حفظ می‌شود و پس از موفقیت، نسخهٔ جدید جایگزین نسخهٔ جاری خواهد شد. ادامه می‌دهید؟");
    if (!confirmed) return;
  }
  const button = $("transcribe-button");
  button.disabled = true;
  button.querySelector("span").textContent = "در حال ارسال…";
  try {
    const result = await post(`/api/files/${item.id}/transcribe`);
    showToast(result.status === "running" ? "این فایل در حال پردازش است." : "فایل به صف پردازش اضافه شد.");
    await Promise.all([selectFile(item.id, false), loadFiles(), loadStats()]);
  } catch (error) {
    showToast(error.message);
    updateTranscribeButton(item);
  }
}

function resetAndLoad() {
  state.offset = 0;
  loadFiles();
}

$("search").addEventListener("input", () => {
  window.clearTimeout(state.debounce);
  state.debounce = window.setTimeout(resetAndLoad, 250);
});
["direction", "status", "transcript-filter"].forEach((id) => $(id).addEventListener("change", resetAndLoad));
$("next-page").addEventListener("click", () => { state.offset += state.limit; loadFiles(); });
$("prev-page").addEventListener("click", () => { state.offset = Math.max(0, state.offset - state.limit); loadFiles(); });
$("transcribe-button").addEventListener("click", transcribeSelected);

Promise.all([loadStats(), loadFiles()]).catch((error) => {
  $("connection").classList.add("offline");
  showToast(error.message);
});

window.setInterval(() => {
  loadStats().catch(() => {});
  loadFiles().catch(() => {});
  if (state.selectedId && (state.selectedItem?.job_status === "running" || isInteractivelyQueued(state.selectedItem || {}))) {
    selectFile(state.selectedId, false).catch(() => {});
  }
}, 15000);
