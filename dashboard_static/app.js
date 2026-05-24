const ROUTES = ["overview", "process", "calibration", "settings"];
const STORAGE_KEY = "juice_dashboard_event_v2";
const PROCESS_STAGE_NAMES = [
  "Loading Match Footage",
  "Calibrating Trackers",
  "Tracking Match",
  "Cleaning Up",
];

const state = {
  healthOk: false,
  isLoadingEvent: false,
  hardware: null,
  usage: null,
  examples: null,
  downloadedVideos: [],
  settings: null,
  currentEvent: null,
  jobs: [],
  route: "select",
  processMatchKey: null,
  selectedOverviewTeam: null,
  processJobPollInFlight: false,
  calibration: {
    videoUrl: null,
    sourceMode: "none",
    selectedSource: "",
    corners: [],
    draggingIndex: null,
    frameReady: false,
    sliderDragActive: false,
    seekRequestId: 0,
  },
};

const dom = {};

document.addEventListener("DOMContentLoaded", () => {
  bindDom();
  bindEvents();
  initializeState();
  boot();
});

function bindDom() {
  dom.brandSubtitle = document.getElementById("brandSubtitle");
  dom.healthLabel = document.getElementById("healthLabel");
  dom.mainNav = document.getElementById("mainNav");
  dom.progressNudge = document.getElementById("progressNudge");
  dom.nudgeProcessed = document.getElementById("nudgeProcessed");
  dom.nudgeProcessing = document.getElementById("nudgeProcessing");
  dom.nudgeQueued = document.getElementById("nudgeQueued");
  dom.nudgeUpcoming = document.getElementById("nudgeUpcoming");
  dom.nudgeLabel = document.getElementById("nudgeLabel");
  dom.selectView = document.getElementById("selectView");
  dom.overviewView = document.getElementById("overviewView");
  dom.processView = document.getElementById("processView");
  dom.calibrationView = document.getElementById("calibrationView");
  dom.settingsView = document.getElementById("settingsView");
  dom.eventForm = document.getElementById("eventForm");
  dom.seasonInput = document.getElementById("seasonInput");
  dom.eventCodeInput = document.getElementById("eventCodeInput");
  dom.resolvedUrlInput = document.getElementById("resolvedUrlInput");
  dom.loadEventButton = document.getElementById("loadEventButton");
  dom.loadEventButtonLabel = document.getElementById("loadEventButtonLabel");
  dom.eventLoadingShell = document.getElementById("eventLoadingShell");
  dom.eventLoadingTitle = document.getElementById("eventLoadingTitle");
  dom.eventLoadingDetail = document.getElementById("eventLoadingDetail");
  dom.selectStatus = document.getElementById("selectStatus");
  dom.eventTitle = document.getElementById("eventTitle");
  dom.reloadEventButton = document.getElementById("reloadEventButton");
  dom.matchList = document.getElementById("matchList");
  dom.statusDonut = document.getElementById("statusDonut");
  dom.statusLegend = document.getElementById("statusLegend");
  dom.refreshUsageButton = document.getElementById("refreshUsageButton");
  dom.usageGrid = document.getElementById("usageGrid");
  dom.agentGrid = document.getElementById("agentGrid");
  dom.processTitle = document.getElementById("processTitle");
  dom.processTitleMeta = document.getElementById("processTitleMeta");
  dom.processStartButton = document.getElementById("processStartButton");
  dom.processBackButton = document.getElementById("processBackButton");
  dom.processPreviewShell = document.getElementById("processPreviewShell");
  dom.processAuthNotice = document.getElementById("processAuthNotice");
  dom.processStats = document.getElementById("processStats");
  dom.processChecklist = document.getElementById("processChecklist");
  dom.processLogs = document.getElementById("processLogs");
  dom.copyProcessLogsButton = document.getElementById("copyProcessLogsButton");
  dom.processVisualizerShell = document.getElementById("processVisualizerShell");
  dom.calibrationVideoSelect = document.getElementById("calibrationVideoSelect");
  dom.calibrationVideoInput = document.getElementById("calibrationVideoInput");
  dom.frameSliderInput = document.getElementById("frameSliderInput");
  dom.cornersFilenameInput = document.getElementById("cornersFilenameInput");
  dom.downloadCornersButton = document.getElementById("downloadCornersButton");
  dom.saveCornersButton = document.getElementById("saveCornersButton");
  dom.calibrationStatus = document.getElementById("calibrationStatus");
  dom.calibrationCanvas = document.getElementById("calibrationCanvas");
  dom.calibrationVideo = document.getElementById("calibrationVideo");
  dom.cornersPreview = document.getElementById("cornersPreview");
  dom.settingsForm = document.getElementById("settingsForm");
  dom.settingScrapeWorkers = document.getElementById("settingScrapeWorkers");
  dom.settingParallelJobs = document.getElementById("settingParallelJobs");
  dom.settingOutputRoot = document.getElementById("settingOutputRoot");
  dom.settingDebugEvery = document.getElementById("settingDebugEvery");
  dom.settingYtdlpCookiesPath = document.getElementById("settingYtdlpCookiesPath");
  dom.settingYtdlpExtractorArgs = document.getElementById("settingYtdlpExtractorArgs");
  dom.settingDebugFrames = document.getElementById("settingDebugFrames");
  dom.settingDebugVideo = document.getElementById("settingDebugVideo");
  dom.settingAutoOpen = document.getElementById("settingAutoOpen");
  dom.settingsStatus = document.getElementById("settingsStatus");
  dom.settingsHardware = document.getElementById("settingsHardware");
}

function bindEvents() {
  dom.seasonInput.addEventListener("input", updateResolvedEventUrl);
  dom.eventCodeInput.addEventListener("input", updateResolvedEventUrl);
  dom.eventForm.addEventListener("submit", handleEventLoad);
  dom.reloadEventButton.addEventListener("click", reloadCurrentEvent);
  dom.refreshUsageButton.addEventListener("click", loadUsage);
  dom.processBackButton.addEventListener("click", () => setRoute("overview"));
  dom.processStartButton.addEventListener("click", startCurrentProcessMatch);
  dom.copyProcessLogsButton.addEventListener("click", copyProcessLogs);
  dom.calibrationVideoSelect.addEventListener("change", handleCalibrationSourceSelect);
  dom.calibrationVideoInput.addEventListener("change", handleCalibrationVideoUpload);
  dom.frameSliderInput.addEventListener("input", handleFrameScrubPreview);
  dom.downloadCornersButton.addEventListener("click", downloadCornersJson);
  dom.saveCornersButton.addEventListener("click", saveCornersInRepo);
  dom.calibrationVideo.addEventListener("loadedmetadata", handleCalibrationMetadata);
  dom.calibrationVideo.addEventListener("loadeddata", drawCalibrationFrame);
  dom.calibrationVideo.addEventListener("canplay", handleCalibrationCanPlay);
  dom.calibrationVideo.addEventListener("seeked", handleCalibrationSeeked);
  dom.frameSliderInput.addEventListener("change", handleFrameScrubCommit);
  dom.frameSliderInput.addEventListener("pointerdown", beginCalibrationSliderDrag);
  dom.frameSliderInput.addEventListener("pointerup", endCalibrationSliderDrag);
  dom.frameSliderInput.addEventListener("pointercancel", endCalibrationSliderDrag);
  dom.calibrationCanvas.addEventListener("mousedown", beginCornerDrag);
  window.addEventListener("mousemove", continueCornerDrag);
  window.addEventListener("mouseup", endCornerDrag);
  dom.settingsForm.addEventListener("submit", saveSettings);
  window.addEventListener("hashchange", syncRouteFromHash);
  document.querySelectorAll("[data-route]").forEach((button) => {
    button.addEventListener("click", () => setRoute(button.dataset.route));
  });
}

function initializeState() {
  dom.seasonInput.value = String(new Date().getFullYear());
  updateResolvedEventUrl();
  const saved = safeParse(localStorage.getItem(STORAGE_KEY));
  if (saved && saved.currentEvent) {
    state.currentEvent = saved.currentEvent;
  }
}

async function boot() {
  syncRouteFromHash();
  await Promise.all([
    refreshHealth(),
    loadHardware(),
    loadUsage(),
    loadExamples(),
    loadSettings(),
    refreshJobs(),
  ]);
  if (state.currentEvent) {
    showEventWorkspace(state.currentEvent, false);
  } else {
    setRoute("select");
  }
  setInterval(refreshJobs, 2500);
  setInterval(() => {
    if (state.route === "process" && state.processMatchKey) {
      refreshCurrentProcessJob();
    }
  }, 1000);
  setInterval(() => {
    if (state.currentEvent) {
      loadUsage();
    }
    if (state.route === "process" && state.processMatchKey) {
      renderProcessView();
    }
  }, 4000);
}

async function refreshHealth() {
  try {
    await apiGet("/api/health");
    state.healthOk = true;
    dom.healthLabel.textContent = "Ready";
  } catch (_error) {
    state.healthOk = false;
    dom.healthLabel.textContent = "Offline";
  }
}

async function loadHardware() {
  try {
    state.hardware = await apiGet("/api/hardware");
    renderSettingsHardware();
  } catch (_error) {
    state.hardware = null;
  }
}

async function loadUsage() {
  try {
    state.usage = await apiGet("/api/usage");
    renderUsage();
  } catch (_error) {
    state.usage = null;
    renderUsage();
  }
}

async function loadExamples() {
  try {
    state.examples = await apiGet("/api/examples");
    renderCalibrationSources();
  } catch (_error) {
    state.examples = { examples: [] };
  }
}

async function loadSettings() {
  try {
    state.settings = await apiGet("/api/settings");
    renderSettings();
  } catch (_error) {
    state.settings = {};
  }
}

async function refreshJobs() {
  try {
    const payload = await apiGet("/api/jobs");
    state.jobs = payload.jobs || [];
    if (state.currentEvent) {
      loadDownloadedVideos();
    }
    renderOverview();
    renderProgressNudge();
    renderCalibrationSources();
    if (state.route === "process") {
      renderProcessView();
    }
  } catch (_error) {
    state.jobs = [];
  }
}

function updateResolvedEventUrl() {
  const season = dom.seasonInput.value.trim();
  const code = dom.eventCodeInput.value.trim().toUpperCase();
  dom.resolvedUrlInput.value = season && code
    ? `https://ftc-events.firstinspires.org/${season}/${code}/qualifications`
    : "";
}

async function handleEventLoad(event) {
  event.preventDefault();
  const season = dom.seasonInput.value.trim();
  const eventCode = dom.eventCodeInput.value.trim().toUpperCase();
  if (!season || !eventCode) {
    dom.selectStatus.textContent = "Enter both a season and an event code.";
    return;
  }
  setEventLoadingState(true, "Loading event", "Scraping qualification matches and looking for clip links...");
  dom.selectStatus.textContent = "Loading event and scraping qualification matches...";
  try {
    const payload = await apiPost("/api/event/discover", {
      season,
      event_code: eventCode,
      io_workers: normalizedScrapeWorkers(),
    });
    showEventWorkspace(payload, true);
    dom.selectStatus.textContent = "Event loaded.";
  } catch (error) {
    dom.selectStatus.textContent = error.message;
  } finally {
    setEventLoadingState(false);
  }
}

function showEventWorkspace(eventPayload, persist = true) {
  const matches = (eventPayload.matches || []).map((match, index) => ({
    ...match,
    match_key: match.match_key || slugify(`${match.phase}-${match.label}-${index}`),
    match_number: extractMatchNumber(match.label),
  }));
  state.currentEvent = {
    ...eventPayload,
    matches,
  };
  if (persist) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ currentEvent: state.currentEvent }));
  }
  dom.mainNav.classList.remove("hidden");
  dom.brandSubtitle.textContent = `${state.currentEvent.title} • ${state.currentEvent.event_code} • ${state.currentEvent.season}`;
  loadDownloadedVideos();
  renderOverview();
  renderCalibrationSources();
  renderProgressNudge();
  const targetRoute = state.route === "process" ? "process" : "overview";
  setRoute(targetRoute);
}

async function loadDownloadedVideos() {
  if (!state.currentEvent?.event_code) {
    state.downloadedVideos = [];
    renderCalibrationSources();
    return;
  }
  state.downloadedVideos = [];
  renderCalibrationSources();
  try {
    const payload = await apiGet(`/api/event/downloads?event_code=${encodeURIComponent(state.currentEvent.event_code)}`);
    if (payload.event_code && payload.event_code !== state.currentEvent.event_code) {
      return;
    }
    state.downloadedVideos = payload.videos || [];
    renderCalibrationSources();
  } catch (_error) {
    state.downloadedVideos = [];
    renderCalibrationSources();
  }
}

async function reloadCurrentEvent() {
  if (!state.currentEvent) {
    return;
  }
  setEventLoadingState(true, "Reloading event", "Refreshing qualification listings and clip availability...");
  dom.eventTitle.textContent = "Reloading event...";
  try {
    const payload = await apiPost("/api/event/discover", {
      season: state.currentEvent.season,
      event_code: state.currentEvent.event_code,
      io_workers: normalizedScrapeWorkers(),
    });
    showEventWorkspace(payload, true);
  } catch (_error) {
    dom.eventTitle.textContent = state.currentEvent.title;
  } finally {
    setEventLoadingState(false);
  }
}

function setRoute(route) {
  if (route === "select" || !state.currentEvent) {
    state.route = "select";
    window.location.hash = "#select";
  } else if (ROUTES.includes(route)) {
    state.route = route;
    window.location.hash = "#" + route;
  }
  renderRoute();
}

function syncRouteFromHash() {
  const hash = window.location.hash.replace(/^#/, "");
  if (!hash) {
    renderRoute();
    return;
  }
  if (hash === "select") {
    state.route = "select";
  } else if (ROUTES.includes(hash) && state.currentEvent) {
    state.route = hash;
  }
  renderRoute();
}

function renderRoute() {
  const views = {
    select: dom.selectView,
    overview: dom.overviewView,
    process: dom.processView,
    calibration: dom.calibrationView,
    settings: dom.settingsView,
  };
  Object.values(views).forEach((view) => view.classList.remove("active"));
  if (!state.currentEvent || state.route === "select") {
    dom.selectView.classList.add("active");
    dom.mainNav.classList.add("hidden");
    dom.progressNudge.classList.add("hidden");
  } else {
    views[state.route].classList.add("active");
    dom.mainNav.classList.remove("hidden");
    if (state.route !== "overview") {
      dom.progressNudge.classList.remove("hidden");
    } else {
      dom.progressNudge.classList.add("hidden");
    }
    document.querySelectorAll(".nav-btn").forEach((button) => {
      button.classList.toggle("active", button.dataset.route === state.route);
    });
  }
  if (state.route === "overview") {
    renderOverview();
  } else if (state.route === "process") {
    renderProcessView();
  } else if (state.route === "calibration") {
    renderCalibrationSources();
  } else if (state.route === "settings") {
    renderSettings();
    renderSettingsHardware();
  }
}

function renderOverview() {
  if (!state.currentEvent) {
    return;
  }
  dom.eventTitle.textContent = state.currentEvent.title;
  const qualificationMatches = currentQualificationMatches();
  const counts = computeStatusCounts(qualificationMatches);
  dom.matchList.innerHTML = qualificationMatches.map((match) => renderMatchCard(match)).join("") || `<div class="empty-state">No qualification matches found.</div>`;
  dom.matchList.querySelectorAll("[data-open-match]").forEach((button) => {
    button.addEventListener("click", () => {
      state.processMatchKey = button.dataset.openMatch;
      setRoute("process");
    });
  });
  dom.matchList.querySelectorAll("[data-team-filter]").forEach((button) => {
    button.addEventListener("click", () => {
      const team = String(button.dataset.teamFilter || "").trim();
      if (!team) {
        return;
      }
      state.selectedOverviewTeam = state.selectedOverviewTeam === team ? null : team;
      renderOverview();
    });
  });
  renderDonut(counts);
  renderAgentGrid(counts);
}

function renderMatchCard(match) {
  const meta = matchStatusMeta(match);
  const summary = renderOverviewAllianceSummary(match, meta.detail);
  const isHighlighted = state.selectedOverviewTeam && matchHasTeam(match, state.selectedOverviewTeam);
  return `
    <article class="match-item${isHighlighted ? " team-highlighted" : ""}">
      <div class="match-item-head compact">
        <div class="match-mainline">
          <div class="match-title">${escapeHtml(match.label)}</div>
          <div class="match-subtle">${summary}</div>
        </div>
        <div class="pill-row compact">
          <span class="pill ${meta.className}">${meta.label}</span>
          <button type="button" class="tiny-btn" data-open-match="${escapeHtmlAttr(match.match_key)}">Open</button>
        </div>
      </div>
    </article>
  `;
}

function renderDonut(counts) {
  const entries = [
    { key: "processed", label: "Processed", color: "#50dd75", value: counts.processed },
    { key: "processing", label: "Processing", color: "#7eb8ff", value: counts.processing },
    { key: "queued", label: "Queued", color: "#fe8f00", value: counts.queued },
    { key: "upcoming", label: "Upcoming", color: "#555555", value: counts.upcoming },
  ];
  const total = Math.max(1, entries.reduce((sum, entry) => sum + entry.value, 0));
  let offset = 0;
  const circles = entries.map((entry) => {
    const portion = entry.value / total;
    const dash = portion * 314;
    const circle = `
      <circle cx="60" cy="60" r="50" fill="none" stroke="${entry.color}" stroke-width="16"
        stroke-dasharray="${dash} 314" stroke-dashoffset="${-offset}" transform="rotate(-90 60 60)"></circle>`;
    offset += dash;
    return circle;
  }).join("");
  dom.statusDonut.innerHTML = `
    <svg viewBox="0 0 120 120" width="200" height="200">
      <circle cx="60" cy="60" r="50" fill="none" stroke="#211910" stroke-width="16"></circle>
      ${circles}
      <text x="60" y="56" text-anchor="middle" fill="#e2e2e2" font-size="16" font-weight="700">${counts.total}</text>
      <text x="60" y="74" text-anchor="middle" fill="#ababab" font-size="10">qualification matches</text>
    </svg>
  `;
  dom.statusLegend.innerHTML = entries.map((entry) => `
    <div class="legend-row">
      <div><span class="swatch" style="background:${entry.color}"></span>${entry.label}</div>
      <strong>${entry.value}</strong>
    </div>
  `).join("");
}

function renderUsage() {
  if (!state.usage) {
    dom.usageGrid.innerHTML = `<div class="empty-state">Usage data unavailable on this machine.</div>`;
    return;
  }
  const memoryPercent = state.usage.memory && state.usage.memory.used_percent != null
    ? `${state.usage.memory.used_percent}%`
    : "Unknown";
  const netIn = state.usage.network && state.usage.network.bytes_in_total != null
    ? formatBytes(state.usage.network.bytes_in_total)
    : "Unknown";
  const netOut = state.usage.network && state.usage.network.bytes_out_total != null
    ? formatBytes(state.usage.network.bytes_out_total)
    : "Unknown";
  const load = state.usage.load_average
    ? `${state.usage.load_average.one_min} / ${state.usage.load_average.five_min} / ${state.usage.load_average.fifteen_min}`
    : "Unavailable";
  dom.usageGrid.innerHTML = `
    <div class="stat-card">
      <div class="section-title">CPU estimate</div>
      <strong>${state.usage.cpu_percent_estimate != null ? state.usage.cpu_percent_estimate + "%" : "Unknown"}</strong>
    </div>
    <div class="stat-card">
      <div class="section-title">Memory used</div>
      <strong>${memoryPercent}</strong>
    </div>
    <div class="stat-card">
      <div class="section-title">Load average</div>
      <strong>${load}</strong>
    </div>
    <div class="stat-card">
      <div class="section-title">Network totals</div>
      <strong>In ${netIn} / Out ${netOut}</strong>
    </div>
  `;
}

function renderAgentGrid(counts) {
  const runningJobs = jobsForEvent().filter((job) => job.status === "running");
  const queueMatches = currentQualificationMatches().filter((match) => matchStatusMeta(match).label === "Queued");
  const parallelAgents = Math.max(1, Number((state.settings || {}).parallel_track_jobs_override || (state.hardware?.recommendations?.recommended_parallel_track_jobs ?? 1)));
  const cards = [];
  cards.push(`
    <article class="agent-card">
      <div class="section-title">Event Agent</div>
      <h3>${escapeHtml(state.currentEvent.title)}</h3>
      <div class="job-meta">Tracking ${counts.total} qualification matches and refreshing FTC Events scrape data on demand.</div>
    </article>
  `);
  cards.push(`
    <article class="agent-card">
      <div class="section-title">Hardware Agent</div>
      <h3>${escapeHtml(state.hardware?.cpu_name || "Machine monitor")}</h3>
      <div class="job-meta">Suggesting ${state.hardware?.recommendations?.recommended_parallel_track_jobs ?? 1} parallel track jobs and ${state.hardware?.recommendations?.recommended_scrape_workers ?? 4} scrape workers.</div>
    </article>
  `);
  for (let index = 0; index < parallelAgents; index += 1) {
    const running = runningJobs[index];
    const queued = queueMatches[index];
    const task = running
      ? `Processing ${running.match_label || running.job_id}`
      : queued
        ? `Ready for ${queued.label}`
        : "Idle";
    cards.push(`
      <article class="agent-card">
        <div class="section-title">Tracker Agent ${index + 1}</div>
        <h3>${escapeHtml(task)}</h3>
        <div class="job-meta">${running ? "Live job active." : queued ? "Next queued match available." : "No assigned work right now."}</div>
      </article>
    `);
  }
  dom.agentGrid.innerHTML = cards.join("");
}

function renderProgressNudge() {
  if (!state.currentEvent) {
    dom.progressNudge.classList.add("hidden");
    return;
  }
  const counts = computeStatusCounts(currentQualificationMatches());
  const total = Math.max(1, counts.total);
  dom.nudgeProcessed.style.flexBasis = `${(counts.processed / total) * 100}%`;
  dom.nudgeProcessing.style.flexBasis = `${(counts.processing / total) * 100}%`;
  dom.nudgeQueued.style.flexBasis = `${(counts.queued / total) * 100}%`;
  dom.nudgeUpcoming.style.flexBasis = `${(counts.upcoming / total) * 100}%`;
  dom.nudgeLabel.textContent = `${counts.processed} processed / ${counts.processing} processing / ${counts.queued} queued / ${counts.upcoming} upcoming`;
}

function renderProcessView() {
  const match = currentProcessMatch();
  if (!match) {
    dom.processTitle.textContent = "Select a match";
    dom.processTitleMeta.innerHTML = "";
    dom.processTitleMeta.classList.add("hidden");
    dom.processStats.innerHTML = `<div class="empty-state">No match selected.</div>`;
    dom.processChecklist.innerHTML = renderChecklistRows(null);
    dom.processLogs.textContent = "No process output yet.";
    dom.processPreviewShell.innerHTML = `<div class="empty-state">No preview available yet.</div>`;
    dom.processVisualizerShell.innerHTML = `<div class="empty-state">The data visualizer will appear here after a match finishes and exports robot position data.</div>`;
    renderProcessAuthNotice(null);
    return;
  }
  const job = latestJobForMatch(match);
  const meta = matchStatusMeta(match);
  dom.processTitle.textContent = match.label;
  const processSummary = renderProcessAllianceSummary(match);
  if (processSummary) {
    dom.processTitleMeta.innerHTML = processSummary;
    dom.processTitleMeta.classList.remove("hidden");
  } else {
    dom.processTitleMeta.innerHTML = "";
    dom.processTitleMeta.classList.add("hidden");
  }
  const activeJob = isJobActive(job);
  dom.processStartButton.classList.toggle("danger", activeJob);
  dom.processStartButton.classList.toggle("primary", !activeJob);
  if (job && job.status === "stopping") {
    dom.processStartButton.disabled = true;
    dom.processStartButton.textContent = "Stopping...";
  } else if (activeJob) {
    dom.processStartButton.disabled = false;
    dom.processStartButton.textContent = "Stop Process";
  } else {
    dom.processStartButton.disabled = meta.label === "Upcoming";
    dom.processStartButton.textContent = meta.label === "Processed" ? "Reprocess Match" : "Start Processing";
  }

  if (job && job.latest_preview_url) {
    if ((job.latest_preview_path || "").toLowerCase().endsWith(".mp4")) {
      dom.processPreviewShell.innerHTML = `<video controls src="${escapeHtmlAttr(job.latest_preview_url)}"></video>`;
    } else {
      dom.processPreviewShell.innerHTML = `<img src="${escapeHtmlAttr(job.latest_preview_url)}" alt="Latest process preview">`;
    }
  } else {
    dom.processPreviewShell.innerHTML = `<div class="empty-state">No preview available yet. Dashboard jobs produce previews when debug frames are enabled.</div>`;
  }

  const stats = [
    statCard("Status", meta.label),
    statCard("Phase", match.phase),
    statCard("Video", match.video_url ? "Available" : "Not found"),
    statCard("Latest job", job ? (job.job_id + (job.return_code != null ? ` • rc ${job.return_code}` : "")) : "None"),
    statCard("Debug frames", job ? String(job.debug_frames_count ?? 0) : "0"),
    statCard("Output", job && job.output_dir ? escapeHtml(job.output_dir) : "Not started"),
  ];
  dom.processStats.innerHTML = stats.join("");
  dom.processChecklist.innerHTML = renderChecklistRows(job);
  renderProcessVisualizer(job);
  renderProcessAuthNotice(job);
  dom.processLogs.textContent = job && job.log_tail && job.log_tail.length
    ? job.log_tail.join("\n")
    : "No process output yet.";
}

function statCard(label, value) {
  return `<div class="stat-card"><div class="section-title">${label}</div><strong>${value}</strong></div>`;
}

function renderChecklistRows(job) {
  const stages = processStagesForJob(job);
  return stages.map((stage) => `
    <div class="check-row">
      <span class="check-icon ${escapeHtmlAttr(checkIconClass(stage.status))}"></span>
      <div class="check-copy">
        <strong>${escapeHtml(stage.name)}</strong>
        <span>${escapeHtml(stage.detail || checklistStatusLabel(stage.status))}</span>
        ${renderStageProgress(stage)}
      </div>
    </div>
  `).join("");
}

function renderStageProgress(stage) {
  if (stage.status !== "in_progress") {
    return "";
  }
  const fraction = Number(stage.progress_fraction);
  if (Number.isFinite(fraction)) {
    const pct = clamp(fraction, 0, 1) * 100;
    return `
      <div class="check-progress" aria-hidden="true">
        <div class="check-progress-fill" style="width:${pct.toFixed(1)}%"></div>
      </div>
    `;
  }
  return `
    <div class="check-progress" aria-hidden="true">
      <div class="check-progress-fill indeterminate"></div>
    </div>
  `;
}

function processStagesForJob(job) {
  const stageMap = new Map((job?.process_stages || []).map((stage) => [stage.name, stage]));
  return PROCESS_STAGE_NAMES.map((name) => {
    const stage = stageMap.get(name);
    return stage || { name, status: "pending", detail: "" };
  });
}

function checkIconClass(status) {
  if (status === "in_progress") {
    return "running";
  }
  if (status === "completed") {
    return "completed";
  }
  if (status === "failed" || status === "cancelled") {
    return "failed";
  }
  return "pending";
}

function checklistStatusLabel(status) {
  if (status === "in_progress") {
    return "In progress";
  }
  if (status === "completed") {
    return "Completed";
  }
  if (status === "failed" || status === "cancelled") {
    return "Failed";
  }
  return "Waiting to start";
}

function renderProcessVisualizer(job) {
  const dataUrl = job?.jlog_workspace_url || job?.csv_workspace_url;
  if (job?.status === "completed" && dataUrl) {
    const iframeUrl = `/tools/data_visualizer.html?embed=1&data=${encodeURIComponent(dataUrl)}&image=${encodeURIComponent("/assets/decode-field.png")}`;
    const existingFrame = dom.processVisualizerShell.querySelector("iframe");
    if (existingFrame && existingFrame.dataset.src === iframeUrl) {
      return;
    }
    dom.processVisualizerShell.innerHTML = `<iframe data-src="${escapeHtmlAttr(iframeUrl)}" src="${escapeHtmlAttr(iframeUrl)}" title="Integrated data visualizer"></iframe>`;
    return;
  }
  if (dataUrl) {
    dom.processVisualizerShell.innerHTML = `<div class="empty-state">Data exports are being prepared. The visualizer will appear automatically when the job completes.</div>`;
    return;
  }
  dom.processVisualizerShell.innerHTML = `<div class="empty-state">The data visualizer will appear here after a match finishes and exports robot position data.</div>`;
}

function renderProcessAuthNotice(job) {
  const authIssue = getProcessAuthIssue(job);
  if (!authIssue) {
    dom.processAuthNotice.classList.add("hidden");
    dom.processAuthNotice.innerHTML = "";
    return;
  }
  dom.processAuthNotice.classList.remove("hidden");
  dom.processAuthNotice.innerHTML = authIssue;
}

function getProcessAuthIssue(job) {
  if (!job || !Array.isArray(job.log_tail) || !job.log_tail.length) {
    return "";
  }
  const joinedOutput = job.log_tail.join("\n").toLowerCase();
  const tlsFailed = (
    joinedOutput.includes("certificate_verify_failed")
    || joinedOutput.includes("unable to get local issuer certificate")
    || joinedOutput.includes("python tls certificate verification failed")
  );
  if (tlsFailed) {
    return (
      `<strong>Python TLS certificates are not configured for this interpreter.</strong>` +
      `Run <code>/Applications/Python 3.14/Install Certificates.command</code>, then start the job again. ` +
      `If that still fails, reinstall <code>yt-dlp</code> with <code>python3 -m pip install -U yt-dlp</code>.`
    );
  }
  const rateLimited = (
    joinedOutput.includes("http error 429")
    || joinedOutput.includes("too many requests")
    || joinedOutput.includes("rate-limiting this machine")
  );
  if (rateLimited) {
    return (
      `<strong>YouTube is rate-limiting this machine.</strong>` +
      `Wait a bit before retrying, avoid starting several downloads back-to-back, and if this keeps happening ` +
      `add authenticated yt-dlp access in Settings.`
    );
  }
  const authBlocked = (
    joinedOutput.includes("youtube only exposed storyboard/blocked formats")
    || joinedOutput.includes("sign in to confirm you're not a bot")
    || joinedOutput.includes("sign in to confirm you’re not a bot")
    || joinedOutput.includes("only images are available for download")
    || joinedOutput.includes("requested format is not available")
    || joinedOutput.includes("the anonymous retry paths were exhausted")
    || joinedOutput.includes("po token")
  );
  if (!authBlocked) {
    return "";
  }

  const hasConfiguredAuth = Boolean(
    (state.settings?.ytdlp_cookies_path || "").trim()
    || (state.settings?.ytdlp_extractor_args || "").trim()
  );
  if (hasConfiguredAuth) {
    return (
      `<strong>Authenticated YouTube access is needed for this match.</strong>` +
      `The current job still failed after the anonymous retries. Check the values in Settings for ` +
      `<code>yt-dlp cookies.txt path</code> or <code>yt-dlp extractor args override</code>, then re-run the match.`
    );
  }
  return (
    `<strong>This match needs authenticated YouTube access.</strong>` +
    `Open Settings and add either a <code>yt-dlp cookies.txt path</code> or a ` +
    `<code>yt-dlp extractor args override</code>, then start the job again.`
  );
}

async function startCurrentProcessMatch() {
  const match = currentProcessMatch();
  const job = match ? latestJobForMatch(match) : null;
  if (job && isJobActive(job)) {
    await stopCurrentProcessJob(job);
    return;
  }
  if (!match || !match.video_url) {
    return;
  }
  const outputRoot = (state.settings && state.settings.output_root) || "./output_dashboard";
  const outputDir = `${outputRoot.replace(/\/$/, "")}/${state.currentEvent.event_code}/${slugify(match.label)}`;
  try {
    const payload = await apiPost("/api/jobs/start-track", {
      source_type: "youtube",
      url: match.video_url,
      output_dir: outputDir,
      corners: state.examples?.default_corners_path || "",
      debug: state.settings?.default_debug_frames,
      debug_video: state.settings?.default_debug_video,
      debug_every: state.settings?.default_debug_every,
      event_code: state.currentEvent.event_code,
      match_label: match.label,
      match_phase: match.phase,
    });
    state.processMatchKey = match.match_key;
    await refreshJobs();
    if (state.settings?.auto_open_process_view) {
      setRoute("process");
    }
    await refreshCurrentProcessJob();
    dom.processLogs.textContent = (payload.log_tail || []).join("\n") || "Job started.";
  } catch (error) {
    dom.processLogs.textContent = error.message;
  }
}

async function stopCurrentProcessJob(job) {
  try {
    const payload = await apiPost(`/api/jobs/${job.job_id}/stop`, {});
    mergeJobIntoState(payload);
    renderProcessView();
  } catch (error) {
    dom.processLogs.textContent = error.message;
  }
}

async function copyProcessLogs() {
  const content = dom.processLogs.textContent || "";
  if (!content.trim()) {
    setCopyLogsButtonState("Nothing To Copy");
    return;
  }
  try {
    await navigator.clipboard.writeText(content);
    setCopyLogsButtonState("Copied");
  } catch (_error) {
    setCopyLogsButtonState("Copy Failed");
  }
}

function renderCalibrationSources() {
  const selectedValue = state.calibration.selectedSource || dom.calibrationVideoSelect.value || "";
  const options = [];
  const availableValues = new Set();
  const seenWorkspaceUrls = new Set();
  options.push(`<option value="">Select a loaded video</option>`);
  for (const example of state.examples?.examples || []) {
    options.push(`<option value="workspace:${escapeHtmlAttr(example.workspace_url)}">Example • ${escapeHtml(example.name)}</option>`);
    availableValues.add(`workspace:${example.workspace_url}`);
  }
  for (const job of jobsForEvent()) {
    if (job.source_workspace_url) {
      options.push(`<option value="workspace:${escapeHtmlAttr(job.source_workspace_url)}">Job Source • ${escapeHtml(job.match_label || job.job_id)}</option>`);
      availableValues.add(`workspace:${job.source_workspace_url}`);
      seenWorkspaceUrls.add(job.source_workspace_url);
    }
    if (job.downloaded_video_workspace_url && !seenWorkspaceUrls.has(job.downloaded_video_workspace_url)) {
      options.push(`<option value="workspace:${escapeHtmlAttr(job.downloaded_video_workspace_url)}">Downloaded Match • ${escapeHtml(job.match_label || job.job_id)}</option>`);
      availableValues.add(`workspace:${job.downloaded_video_workspace_url}`);
      seenWorkspaceUrls.add(job.downloaded_video_workspace_url);
    }
  }
  for (const video of state.downloadedVideos || []) {
    if (!video.workspace_url || seenWorkspaceUrls.has(video.workspace_url)) {
      continue;
    }
    const knownMatch = currentQualificationMatches().find((match) => slugify(match.label) === video.match_slug);
    const displayLabel = knownMatch ? knownMatch.label : humanizeSlug(video.match_slug || "downloaded-match");
    options.push(`<option value="workspace:${escapeHtmlAttr(video.workspace_url)}">Downloaded Match • ${escapeHtml(displayLabel)}</option>`);
    availableValues.add(`workspace:${video.workspace_url}`);
    seenWorkspaceUrls.add(video.workspace_url);
  }
  for (const match of currentQualificationMatches()) {
    const meta = matchStatusMeta(match);
    if (match.video_url && (meta.label === "Queued" || meta.label === "Upcoming")) {
      options.push(`<option value="youtube:${escapeHtmlAttr(match.video_url)}">Queued Match Clip • ${escapeHtml(match.label)}</option>`);
      availableValues.add(`youtube:${match.video_url}`);
    }
  }
  dom.calibrationVideoSelect.innerHTML = options.join("");
  if (selectedValue && availableValues.has(selectedValue)) {
    dom.calibrationVideoSelect.value = selectedValue;
  }
}

async function handleCalibrationSourceSelect() {
  const value = dom.calibrationVideoSelect.value;
  if (!value) {
    state.calibration.selectedSource = "";
    return;
  }
  const separatorIndex = value.indexOf(":");
  const sourceKind = separatorIndex >= 0 ? value.slice(0, separatorIndex) : "workspace";
  const sourceValue = separatorIndex >= 0 ? value.slice(separatorIndex + 1) : value;
  resetCalibrationState();
  state.calibration.selectedSource = value;
  dom.calibrationStatus.textContent = "Resolving selected video source...";
  try {
    const payload = sourceKind === "youtube"
      ? await apiPost("/api/resolve-video-source", { youtube_url: sourceValue })
      : await apiPost("/api/resolve-video-source", { workspace_url: sourceValue });
    state.calibration.sourceMode = payload.kind;
    dom.calibrationVideo.crossOrigin = "anonymous";
    dom.calibrationVideo.preload = "auto";
    dom.calibrationVideo.src = payload.playable_url;
    dom.calibrationVideo.load();
    dom.calibrationStatus.textContent = payload.kind === "youtube"
      ? "Loading queued YouTube match clip..."
      : "Loading selected system video...";
  } catch (error) {
    dom.calibrationStatus.textContent = error.message;
  }
}

function handleCalibrationVideoUpload(event) {
  const file = event.target.files[0];
  if (!file) {
    return;
  }
  resetCalibrationState();
  state.calibration.sourceMode = "upload";
  state.calibration.selectedSource = "";
  dom.calibrationVideoSelect.value = "";
  state.calibration.videoUrl = URL.createObjectURL(file);
  dom.calibrationVideo.crossOrigin = "anonymous";
  dom.calibrationVideo.preload = "auto";
  dom.calibrationVideo.src = state.calibration.videoUrl;
  dom.calibrationVideo.load();
  dom.calibrationStatus.textContent = "Loading uploaded video...";
}

function resetCalibrationState() {
  if (state.calibration.videoUrl) {
    URL.revokeObjectURL(state.calibration.videoUrl);
  }
  state.calibration.videoUrl = null;
  state.calibration.corners = [];
  state.calibration.draggingIndex = null;
  state.calibration.frameReady = false;
  state.calibration.seekRequestId += 1;
  dom.frameSliderInput.disabled = true;
  dom.frameSliderInput.value = "0";
}

function handleCalibrationMetadata() {
  dom.frameSliderInput.disabled = false;
  dom.frameSliderInput.value = "500";
  dom.calibrationVideo.pause();
  state.calibration.frameReady = false;
  const midpoint = (dom.calibrationVideo.duration || 0) / 2;
  requestCalibrationSeek(midpoint);
  dom.calibrationStatus.textContent = `Loaded ${dom.calibrationVideo.videoWidth}x${dom.calibrationVideo.videoHeight} video. Starting at the midpoint for calibration.`;
}

function handleFrameScrubPreview() {
  if (!dom.calibrationVideo.duration || Number.isNaN(dom.calibrationVideo.duration)) {
    return;
  }
  const fraction = Number(dom.frameSliderInput.value) / 1000;
  const targetTime = dom.calibrationVideo.duration * fraction;
  dom.calibrationStatus.textContent = `Selected frame at ${targetTime.toFixed(2)}s. Release to seek.`;
}

function handleFrameScrubCommit() {
  if (!dom.calibrationVideo.duration || Number.isNaN(dom.calibrationVideo.duration)) {
    return;
  }
  const fraction = Number(dom.frameSliderInput.value) / 1000;
  const targetTime = dom.calibrationVideo.duration * fraction;
  dom.calibrationStatus.textContent = `Seeking to ${targetTime.toFixed(2)}s`;
  requestCalibrationSeek(targetTime);
}

function beginCalibrationSliderDrag() {
  state.calibration.sliderDragActive = true;
}

function endCalibrationSliderDrag() {
  state.calibration.sliderDragActive = false;
}

function handleCalibrationCanPlay() {
  if (!state.calibration.frameReady && dom.calibrationVideo.readyState >= 2) {
    drawCalibrationFrame();
  }
}

function handleCalibrationSeeked() {
  if (!state.calibration.sliderDragActive) {
    syncCalibrationSliderToVideo();
  }
  drawCalibrationFrame();
  dom.calibrationStatus.textContent = `Showing frame at ${dom.calibrationVideo.currentTime.toFixed(2)}s`;
}

function drawCalibrationFrame() {
  const canvas = dom.calibrationCanvas;
  const ctx = canvas.getContext("2d");
  const width = dom.calibrationVideo.videoWidth;
  const height = dom.calibrationVideo.videoHeight;
  if (!width || !height) {
    return;
  }
  canvas.width = width;
  canvas.height = height;
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(dom.calibrationVideo, 0, 0, width, height);
  if (!state.calibration.corners.length) {
    state.calibration.corners = defaultCalibrationCorners(width, height);
  }
  drawCornerOverlay(ctx, state.calibration.corners);
  state.calibration.frameReady = true;
  updateCornerPreview();
}

function defaultCalibrationCorners(width, height) {
  const mx = width * 0.2;
  const my = height * 0.2;
  const px = width * 0.8;
  const py = height * 0.8;
  return [
    { x: mx, y: my, label: "TL", color: "#e6b422" },
    { x: px, y: my, label: "TR", color: "#37b5e5" },
    { x: px, y: py, label: "BR", color: "#d95b43" },
    { x: mx, y: py, label: "BL", color: "#4f80ff" },
  ];
}

function drawCornerOverlay(ctx, corners) {
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(255,255,255,0.92)";
  ctx.fillStyle = "rgba(20, 20, 20, 0.18)";
  ctx.beginPath();
  corners.forEach((corner, index) => {
    if (index === 0) {
      ctx.moveTo(corner.x, corner.y);
    } else {
      ctx.lineTo(corner.x, corner.y);
    }
  });
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  corners.forEach((corner) => {
    ctx.fillStyle = corner.color;
    ctx.beginPath();
    ctx.arc(corner.x, corner.y, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 15px Poppins, sans-serif";
    ctx.fillText(corner.label, corner.x + 12, corner.y - 12);
  });
  ctx.restore();
}

function beginCornerDrag(event) {
  if (!state.calibration.frameReady) {
    return;
  }
  const point = canvasPointFromEvent(event);
  let bestIndex = -1;
  let bestDistance = Number.POSITIVE_INFINITY;
  state.calibration.corners.forEach((corner, index) => {
    const distance = Math.hypot(point.x - corner.x, point.y - corner.y);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });
  if (bestIndex >= 0 && bestDistance <= 26) {
    state.calibration.draggingIndex = bestIndex;
  }
}

function continueCornerDrag(event) {
  if (state.calibration.draggingIndex == null) {
    return;
  }
  const point = canvasPointFromEvent(event);
  const corner = state.calibration.corners[state.calibration.draggingIndex];
  corner.x = clamp(point.x, 0, dom.calibrationCanvas.width - 1);
  corner.y = clamp(point.y, 0, dom.calibrationCanvas.height - 1);
  drawCalibrationFrame();
}

function endCornerDrag() {
  state.calibration.draggingIndex = null;
}

function canvasPointFromEvent(event) {
  const rect = dom.calibrationCanvas.getBoundingClientRect();
  const scaleX = dom.calibrationCanvas.width / rect.width;
  const scaleY = dom.calibrationCanvas.height / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

function currentCornersPayload() {
  if (state.calibration.corners.length !== 4) {
    return { corners_px: [] };
  }
  return {
    corners_px: [
      [round2(state.calibration.corners[3].x), round2(state.calibration.corners[3].y)],
      [round2(state.calibration.corners[2].x), round2(state.calibration.corners[2].y)],
      [round2(state.calibration.corners[1].x), round2(state.calibration.corners[1].y)],
      [round2(state.calibration.corners[0].x), round2(state.calibration.corners[0].y)],
    ],
  };
}

function updateCornerPreview() {
  dom.cornersPreview.textContent = JSON.stringify(currentCornersPayload(), null, 2);
}

function requestCalibrationSeek(targetTime) {
  const video = dom.calibrationVideo;
  video.pause();
  state.calibration.frameReady = false;
  state.calibration.seekRequestId += 1;
  const requestId = state.calibration.seekRequestId;
  const duration = Number.isFinite(video.duration) ? video.duration : 0;
  const safeTargetTime = duration
    ? clamp(targetTime, 0, Math.max(duration - 0.05, 0))
    : Math.max(targetTime, 0);
  const needsSeek = Math.abs(video.currentTime - safeTargetTime) > 0.01;
  if (needsSeek) {
    video.currentTime = safeTargetTime;
  } else {
    syncCalibrationSliderToVideo();
    drawCalibrationFrame();
  }
  const drawIfReady = () => {
    if (state.calibration.seekRequestId !== requestId) {
      return;
    }
    if (video.readyState >= 2) {
      drawCalibrationFrame();
    }
  };
  window.requestAnimationFrame(() => drawIfReady());
  window.clearTimeout(requestCalibrationSeek._fallbackTimer);
  window.clearTimeout(requestCalibrationSeek._lateFallbackTimer);
  requestCalibrationSeek._fallbackTimer = window.setTimeout(() => {
    drawIfReady();
  }, 220);
  requestCalibrationSeek._lateFallbackTimer = window.setTimeout(() => {
    drawIfReady();
  }, 700);
}

function syncCalibrationSliderToVideo() {
  const duration = dom.calibrationVideo.duration;
  if (!duration || Number.isNaN(duration)) {
    return;
  }
  const fraction = clamp(dom.calibrationVideo.currentTime / duration, 0, 1);
  dom.frameSliderInput.value = String(Math.round(fraction * 1000));
}

function setCopyLogsButtonState(label) {
  if (!dom.copyProcessLogsButton) {
    return;
  }
  dom.copyProcessLogsButton.textContent = label;
  dom.copyProcessLogsButton.disabled = true;
  window.clearTimeout(setCopyLogsButtonState._timer);
  setCopyLogsButtonState._timer = window.setTimeout(() => {
    dom.copyProcessLogsButton.textContent = "Copy Output";
    dom.copyProcessLogsButton.disabled = false;
  }, 1300);
}

function downloadCornersJson() {
  const payload = currentCornersPayload();
  const blob = new Blob([JSON.stringify(payload, null, 2) + "\n"], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = dom.cornersFilenameInput.value.trim() || "field_corners.json";
  anchor.click();
  URL.revokeObjectURL(url);
}

async function saveCornersInRepo() {
  try {
    const payload = currentCornersPayload();
    const result = await apiPost("/api/save-corners", {
      filename: dom.cornersFilenameInput.value.trim() || "field_corners.json",
      corners_px: payload.corners_px,
    });
    dom.calibrationStatus.textContent = `Saved corners to ${result.path}`;
  } catch (error) {
    dom.calibrationStatus.textContent = error.message;
  }
}

function renderSettings() {
  const settings = state.settings || {};
  dom.settingScrapeWorkers.value = settings.scrape_workers_override ?? "";
  dom.settingParallelJobs.value = settings.parallel_track_jobs_override ?? "";
  dom.settingOutputRoot.value = settings.output_root || "./output_dashboard";
  dom.settingDebugEvery.value = settings.default_debug_every ?? 1;
  dom.settingYtdlpCookiesPath.value = settings.ytdlp_cookies_path || "";
  dom.settingYtdlpExtractorArgs.value = settings.ytdlp_extractor_args || "";
  dom.settingDebugFrames.checked = Boolean(settings.default_debug_frames);
  dom.settingDebugVideo.checked = Boolean(settings.default_debug_video);
  dom.settingAutoOpen.checked = Boolean(settings.auto_open_process_view);
}

function renderSettingsHardware() {
  if (!state.hardware) {
    dom.settingsHardware.innerHTML = `<div class="empty-state">Hardware profile unavailable.</div>`;
    return;
  }
  const rec = state.hardware.recommendations || {};
  dom.settingsHardware.innerHTML = `
    ${statCard("CPU", escapeHtml(state.hardware.cpu_name || "Unknown"))}
    ${statCard("Scrape workers", rec.recommended_scrape_workers ?? "-")}
    ${statCard("Parallel track jobs", rec.recommended_parallel_track_jobs ?? "-")}
    ${statCard("Math threads / job", rec.recommended_native_math_threads_per_job ?? "-")}
  `;
}

async function saveSettings(event) {
  event.preventDefault();
  try {
    state.settings = await apiPost("/api/settings", {
      scrape_workers_override: blankToNull(dom.settingScrapeWorkers.value),
      parallel_track_jobs_override: blankToNull(dom.settingParallelJobs.value),
      output_root: dom.settingOutputRoot.value.trim() || "./output_dashboard",
      default_debug_every: Number(dom.settingDebugEvery.value || 1),
      ytdlp_cookies_path: dom.settingYtdlpCookiesPath.value.trim(),
      ytdlp_extractor_args: dom.settingYtdlpExtractorArgs.value.trim(),
      default_debug_frames: dom.settingDebugFrames.checked,
      default_debug_video: dom.settingDebugVideo.checked,
      auto_open_process_view: dom.settingAutoOpen.checked,
    });
    dom.settingsStatus.textContent = "Settings saved.";
  } catch (error) {
    dom.settingsStatus.textContent = error.message;
  }
}

function currentQualificationMatches() {
  return (state.currentEvent?.matches || [])
    .filter((match) => match.phase === "qualifications")
    .sort((a, b) => (a.match_number - b.match_number) || a.label.localeCompare(b.label));
}

function jobsForEvent() {
  if (!state.currentEvent) {
    return [];
  }
  return state.jobs.filter((job) => job.event_code === state.currentEvent.event_code);
}

function jobsForMatch(match) {
  return jobsForEvent().filter((job) => job.match_label === match.label);
}

function latestJobForMatch(match) {
  const jobs = jobsForMatch(match);
  return jobs.length ? jobs[0] : null;
}

function matchStatusMeta(match) {
  const job = latestJobForMatch(match);
  if (!match.video_url) {
    return {
      label: "Upcoming",
      className: "upcoming",
      detail: "No discoverable video clip yet. This likely means the match is in the future or the public page has no clip.",
    };
  }
  if (job && (job.status === "running" || job.status === "stopping")) {
    return {
      label: "Processing",
      className: "processing",
      detail: job.status === "stopping"
        ? `Stop requested for job ${job.job_id}.`
        : `Job ${job.job_id} is actively processing this match.`,
    };
  }
  if (job && job.status === "completed") {
    return {
      label: "Processed",
      className: "processed",
      detail: `Completed job ${job.job_id}${job.output_dir ? ` • ${job.output_dir}` : ""}`,
    };
  }
  return {
    label: "Queued",
    className: "queued",
    detail: job && (job.status === "failed" || job.status === "cancelled")
      ? `A previous attempt ${job.status === "cancelled" ? "was stopped" : "failed"}. Open the process view to inspect logs and retry.`
      : "Video is available and ready to process.",
  };
}

function renderOverviewAllianceSummary(match, fallbackText = "") {
  const data = normalizedAllianceData(match);
  if (!data.hasAny) {
    return escapeHtml(fallbackText);
  }
  return `
    <span class="alliance-inline">
      ${renderAllianceInlineChip("red", "Red", data.redTeams, data.redScore, data.winningAlliance === "red")}
      <span class="alliance-inline-divider">vs</span>
      ${renderAllianceInlineChip("blue", "Blue", data.blueTeams, data.blueScore, data.winningAlliance === "blue")}
    </span>
  `;
}

function renderProcessAllianceSummary(match) {
  const data = normalizedAllianceData(match);
  if (!data.hasAny) {
    return "";
  }
  return `
    <div class="alliance-stack">
      ${renderAllianceBlock("red", "Red Alliance", data.redTeams, data.redScore, data.winningAlliance === "red")}
      ${renderAllianceBlock("blue", "Blue Alliance", data.blueTeams, data.blueScore, data.winningAlliance === "blue")}
    </div>
  `;
}

function renderAllianceInlineChip(color, label, teams, score, isWinner) {
  const className = `alliance-chip inline ${color}${isWinner ? " winner" : ""}`;
  const teamText = teams.length
    ? teams.map((team) => renderOverviewTeamButton(team)).join(`<span class="alliance-team-separator">•</span>`)
    : "TBD";
  const scoreText = score != null ? String(score) : "—";
  return `
    <span class="${className}">
      <span class="alliance-team-list">${teamText}</span>
      <span class="alliance-score">${escapeHtml(scoreText)}</span>
    </span>
  `;
}

function renderAllianceBlock(color, label, teams, score, isWinner) {
  const className = `alliance-chip block ${color}${isWinner ? " winner" : ""}`;
  const teamText = teams.length ? teams.join(" • ") : "Teams TBD";
  return `
    <div class="${className}">
      <div class="alliance-chip-head">
        <div class="alliance-head-right">
          <span class="alliance-score">${score != null ? escapeHtml(String(score)) : "—"}</span>
        </div>
      </div>
      <div class="alliance-team-list">${escapeHtml(teamText)}</div>
    </div>
  `;
}

function normalizedAllianceData(match) {
  const redTeams = normalizeTeamList(match?.red_teams);
  const blueTeams = normalizeTeamList(match?.blue_teams);
  const redScore = normalizeNullableNumber(match?.red_score);
  const blueScore = normalizeNullableNumber(match?.blue_score);
  const winningAlliance = normalizeWinningAlliance(match?.winning_alliance, redScore, blueScore);
  return {
    redTeams,
    blueTeams,
    redScore,
    blueScore,
    winningAlliance,
    hasAny: Boolean(redTeams.length || blueTeams.length || redScore != null || blueScore != null),
  };
}

function normalizeTeamList(value) {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => String(item || "").trim())
    .filter(Boolean);
}

function normalizeNullableNumber(value) {
  if (value == null || value === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeWinningAlliance(value, redScore, blueScore) {
  if (value === "red" || value === "blue" || value === "tie") {
    return value;
  }
  if (redScore == null || blueScore == null) {
    return null;
  }
  if (redScore > blueScore) {
    return "red";
  }
  if (blueScore > redScore) {
    return "blue";
  }
  return "tie";
}

function renderOverviewTeamButton(team) {
  const normalizedTeam = String(team || "").trim();
  const isActive = normalizedTeam && state.selectedOverviewTeam === normalizedTeam;
  return `
    <button
      type="button"
      class="team-filter-btn${isActive ? " active" : ""}"
      data-team-filter="${escapeHtmlAttr(normalizedTeam)}"
    >${escapeHtml(normalizedTeam)}</button>
  `;
}

function matchHasTeam(match, team) {
  const normalizedTeam = String(team || "").trim();
  if (!normalizedTeam) {
    return false;
  }
  return normalizeTeamList(match?.red_teams).includes(normalizedTeam)
    || normalizeTeamList(match?.blue_teams).includes(normalizedTeam);
}

function computeStatusCounts(matches) {
  const counts = { processed: 0, processing: 0, queued: 0, upcoming: 0, total: matches.length };
  matches.forEach((match) => {
    const status = matchStatusMeta(match).label.toLowerCase();
    counts[status] += 1;
  });
  return counts;
}

function currentProcessMatch() {
  return currentQualificationMatches().find((match) => match.match_key === state.processMatchKey) || null;
}

async function refreshCurrentProcessJob() {
  if (state.processJobPollInFlight) {
    return;
  }
  const match = currentProcessMatch();
  const job = match ? latestJobForMatch(match) : null;
  if (!job) {
    return;
  }
  state.processJobPollInFlight = true;
  try {
    const payload = await apiGet(`/api/jobs/${job.job_id}`);
    mergeJobIntoState(payload);
    if (state.route === "process") {
      renderProcessView();
    }
  } catch (_error) {
    // Keep the last visible job snapshot if a polling request fails.
  } finally {
    state.processJobPollInFlight = false;
  }
}

function mergeJobIntoState(job) {
  const current = state.jobs.filter((item) => item.job_id !== job.job_id);
  current.push(job);
  current.sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
  state.jobs = current;
}

function isJobActive(job) {
  return Boolean(job && (job.status === "running" || job.status === "stopping"));
}

function normalizedScrapeWorkers() {
  const override = state.settings?.scrape_workers_override;
  if (override !== null && override !== undefined && override !== "") {
    return Number(override);
  }
  return state.hardware?.recommendations?.recommended_scrape_workers;
}

function extractMatchNumber(label) {
  const match = String(label || "").match(/(\d+)/);
  return match ? Number(match[1]) : Number.MAX_SAFE_INTEGER;
}

function slugify(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function formatBytes(value) {
  if (value == null || Number.isNaN(value)) {
    return "Unknown";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = Number(value);
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(size >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function blankToNull(value) {
  const trimmed = String(value).trim();
  return trimmed === "" ? null : Number(trimmed);
}

function setEventLoadingState(isLoading, title = "", detail = "") {
  state.isLoadingEvent = isLoading;
  dom.loadEventButton.disabled = isLoading;
  dom.loadEventButtonLabel.textContent = isLoading ? "Loading..." : "Load Event";
  dom.eventLoadingShell.classList.toggle("hidden", !isLoading);
  if (title) {
    dom.eventLoadingTitle.textContent = title;
  }
  if (detail) {
    dom.eventLoadingDetail.textContent = detail;
  }
}

function round2(value) {
  return Math.round(value * 100) / 100;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function humanizeSlug(value) {
  return String(value || "")
    .replace(/[-_]+/g, " ")
    .replace(/\b\w/g, (match) => match.toUpperCase())
    .trim();
}

async function apiGet(path) {
  const response = await fetch(path, { cache: "no-store" });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with ${response.status}`);
  }
  return payload;
}

async function apiPost(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with ${response.status}`);
  }
  return payload;
}

function safeParse(value) {
  if (!value) {
    return null;
  }
  try {
    return JSON.parse(value);
  } catch (_error) {
    return null;
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function escapeHtmlAttr(value) {
  return escapeHtml(value);
}
