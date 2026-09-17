const MANIFEST_URL = "demo_data/manifest.json";
const DISPLAY_ORDER_URL = "demo_data/display_order.json";
const MERT_SIMILARITY_URL = "demo_data/mert_similarity.json";
const RESULTS_URL = "results_data/main_results.json";
const DATA_ROOT = "demo_data/";
const PEAK_POINTS = 180;

let activeAudioController = null;
let waveformDecodeContext = null;
let showSuppressOthers = false;
const instantiatedSamplePlayers = new Set();

function formatTime(seconds) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const minutes = Math.floor(safe / 60);
  const remainder = Math.floor(safe % 60);
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function formatRange(item) {
  if (!Number.isFinite(item.start_ms) || !Number.isFinite(item.end_ms)) return "";
  return `${formatTime(item.start_ms / 1000)}–${formatTime(item.end_ms / 1000)}`;
}

function inferGroup(item) {
  const direct = String(item.group || "").trim().toLowerCase();
  if (direct === "single" || direct === "multiple") return direct;
  const title = String(item.title || "").trim().toLowerCase();
  if (title.includes("multiple")) return "multiple";
  if (title.includes("single")) return "single";
  throw new Error(`Cannot infer single/multiple group for ${item.sample_id || "an item"}`);
}

function rawVariant(item, group, id, label, sourceKey) {
  const src = item.audio?.[sourceKey];
  if (!src) throw new Error(`${item.sample_id || "Item"} is missing audio.${sourceKey}`);
  const meta = item.audio_meta?.[sourceKey] || {};
  return {
    id,
    label,
    source_key: sourceKey,
    src,
    duration_seconds: Number(meta.duration_seconds || item.audio_duration_seconds?.[sourceKey] || 0),
    peaks: item.waveform_peaks?.[sourceKey] || [],
  };
}

function normalizeItem(item, index, _config = {}, similarityItem = null) {
  const group = inferGroup(item);
  const groupLabel = group === "single" ? "Single-target suppression" : "Multiple-target suppression";
  let variants;

  if (Array.isArray(item.variants)) {
    variants = item.variants.map((variant) => ({
      ...variant,
      duration_seconds: Number(variant.duration_seconds || 0),
      peaks: Array.isArray(variant.peaks) ? variant.peaks : [],
    }));
  } else {
    const targetKey = group === "single" ? "suppress_single_target" : "suppress_multiple_target";
    const othersKey = group === "single" ? "suppress_single_others" : "suppress_multiple_others";
    variants = [
      rawVariant(item, group, "prompt", "Prompt", "prompt"),
      rawVariant(item, group, "reference", "Reference (GT)", "reference"),
      rawVariant(item, group, "no_control", "No Control", "no_control"),
      rawVariant(item, group, "enhance", "Enhance", "enhance_target"),
      rawVariant(item, group, "suppress", "Suppress", targetKey),
    ];
    if (item.audio?.[othersKey]) {
      variants.push(rawVariant(item, group, "suppress_others", "Suppress Others", othersKey));
    }
  }

  variants = variants
    .map((variant) => variant.id === "suppress"
      ? { ...variant, label: "Suppress" }
      : variant)
    .map((variant) => {
      const similarity = similarityItem?.values?.[variant.id];
      return {
        ...variant,
        mert_similarity_to_reference: typeof similarity === "number" && Number.isFinite(similarity)
          ? similarity
          : null,
      };
    });

  return {
    ordinal: item.ordinal || index + 1,
    sample_id: item.sample_id || `sample-${index + 1}`,
    sample_index: item.sample_index,
    group,
    group_label: item.group_label || groupLabel,
    artist: item.artist || item.source_artist_name || "Unknown artist",
    source_title: item.source_title || "Untitled source",
    caption: item.caption || "",
    clip_id: item.clip_id,
    start_ms: Number.isFinite(item.start_ms) ? item.start_ms : null,
    end_ms: Number.isFinite(item.end_ms) ? item.end_ms : null,
    variants,
  };
}

function normalizeManifest(manifest, config = {}, mertSimilarity = {}) {
  const excluded = new Set(manifest.excluded_sample_ids || []);
  const items = (manifest.items || [])
    .filter((item) => item.visible !== false && !excluded.has(item.sample_id))
    .map((item, index) => normalizeItem(
      item,
      index,
      config,
      mertSimilarity.items?.[item.sample_id],
    ));
  return { ...manifest, items };
}

function formatMertSimilarity(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? `MERT–GT ${value.toFixed(3)}`
    : "";
}

function makeMertScore(variant) {
  const text = formatMertSimilarity(variant.mert_similarity_to_reference);
  if (!text) return null;
  const score = document.createElement("span");
  score.className = "mert-score";
  score.textContent = text;
  score.title = `MERT cosine similarity to Reference (GT): ${variant.mert_similarity_to_reference.toFixed(6)}`;
  return score;
}

function sortItemsByDisplayOrder(items, order = [], includeUnlisted = true) {
  const rank = new Map();
  const hidden = new Set();
  if (Array.isArray(order)) {
    order.forEach((entry, index) => {
      const identifier = entry && typeof entry === "object"
        ? entry.sample_id ?? entry.sample_index
        : entry;
      if (identifier === null || identifier === undefined) return;
      const key = String(identifier);
      if (entry && typeof entry === "object" && entry.show === false) {
        hidden.add(key);
      } else if (!rank.has(key)) {
        rank.set(key, index);
      }
    });
  }

  return items
    .map((item, originalIndex) => {
      const identifiers = [item.sample_id, item.sample_index]
        .filter((value) => value !== null && value !== undefined)
        .map(String);
      const matchedIdentifier = identifiers.find((identifier) => rank.has(identifier));
      return {
        item,
        hidden: identifiers.some((identifier) => hidden.has(identifier)),
        originalIndex,
        rank: matchedIdentifier === undefined ? Number.POSITIVE_INFINITY : rank.get(matchedIdentifier),
      };
    })
    .filter((entry) => !entry.hidden && (includeUnlisted || Number.isFinite(entry.rank)))
    .sort((left, right) => left.rank - right.rank || left.originalIndex - right.originalIndex)
    .map(({ item }) => item);
}

function normalizeResults(results) {
  const metricGroups = Array.isArray(results.metric_groups) ? results.metric_groups : [];
  const rowColumns = Array.isArray(results.row_columns) ? results.row_columns : [];
  const metricIds = new Set();
  const rowColumnIds = new Set();
  rowColumns.forEach((column) => {
    if (!column.id || !column.label || rowColumnIds.has(column.id)) {
      throw new Error(`Invalid or duplicate result row column: ${column.id || "unnamed"}`);
    }
    rowColumnIds.add(column.id);
  });
  metricGroups.forEach((group) => {
    if (!group.label || !Array.isArray(group.metrics) || group.metrics.length === 0) {
      throw new Error("Each result metric group needs a label and at least one metric.");
    }
    group.metrics.forEach((metric) => {
      if (!metric.id || !metric.label || metricIds.has(metric.id)) {
        throw new Error(`Invalid or duplicate result metric: ${metric.id || "unnamed"}`);
      }
      metricIds.add(metric.id);
    });
  });
  if (metricIds.size === 0) throw new Error("No result metrics were provided.");
  if (!Array.isArray(results.sections) || results.sections.length === 0) {
    throw new Error("No result sections were provided.");
  }
  return {
    ...results,
    row_columns: rowColumns,
    metric_groups: metricGroups,
    metric_ids: [...metricIds],
  };
}

function formatResultValue(value, decimals = 3) {
  if (value === null || value === undefined || value === "" || value === "-" || value === "--") return "—";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "—";
    return value.toFixed(Number.isInteger(decimals) ? decimals : 3);
  }
  return String(value);
}

function renderResults(inputResults) {
  const results = normalizeResults(inputResults);
  const root = document.querySelector("[data-results-root]");
  const table = document.createElement("table");
  table.className = "results-table";

  const head = document.createElement("thead");
  const groupRow = document.createElement("tr");
  const setupHeading = document.createElement("th");
  setupHeading.className = "setup-column";
  setupHeading.scope = "col";
  setupHeading.rowSpan = 2;
  setupHeading.textContent = results.setup_label || "Inference Setup";
  groupRow.append(setupHeading);

  results.row_columns.forEach((column) => {
    const heading = document.createElement("th");
    heading.className = "results-meta-column";
    heading.scope = "col";
    heading.rowSpan = 2;
    heading.textContent = column.label;
    groupRow.append(heading);
  });

  results.metric_groups.forEach((group) => {
    const heading = document.createElement("th");
    heading.scope = "colgroup";
    heading.colSpan = group.metrics.length;
    heading.textContent = group.label;
    groupRow.append(heading);
  });

  const metricRow = document.createElement("tr");
  results.metric_groups.forEach((group) => {
    group.metrics.forEach((metric) => {
      const heading = document.createElement("th");
      heading.scope = "col";
      heading.textContent = metric.label;
      metricRow.append(heading);
    });
  });
  head.append(groupRow, metricRow);

  const body = document.createElement("tbody");
  const metrics = results.metric_groups.flatMap((group) => group.metrics);
  let stripeIndex = 0;
  let shadeCurrentSetup = false;
  results.sections.forEach((section) => {
    if (section.title) {
      const sectionRow = document.createElement("tr");
      sectionRow.className = "results-section-row";
      const sectionHeading = document.createElement("th");
      sectionHeading.scope = "rowgroup";
      sectionHeading.colSpan = metrics.length + results.row_columns.length + 1;
      sectionHeading.textContent = section.title;
      sectionRow.append(sectionHeading);
      body.append(sectionRow);
    }

    (section.rows || []).forEach((row) => {
      const tableRow = document.createElement("tr");
      tableRow.className = "results-data-row";
      if (!row.continuation) {
        shadeCurrentSetup = stripeIndex % 2 === 1;
        stripeIndex += 1;
      }
      if (shadeCurrentSetup) tableRow.classList.add("is-shaded");
      if (row.continuation) tableRow.classList.add("is-continuation");
      if (row.divider_before) tableRow.classList.add("has-divider");
      if (!row.continuation) {
        const label = document.createElement("th");
        label.scope = "row";
        label.textContent = row.label || row.id;
        if (Number.isInteger(row.row_span) && row.row_span > 1) label.rowSpan = row.row_span;
        tableRow.append(label);
      }

      results.row_columns.forEach((column) => {
        const cell = document.createElement("td");
        const rawValue = row.metadata?.[column.id];
        cell.className = "results-meta-cell";
        if (rawValue === "✓") cell.classList.add("is-enabled");
        if (rawValue === "✗") cell.classList.add("is-disabled");
        cell.textContent = formatResultValue(rawValue, column.decimals);
        tableRow.append(cell);
      });

      metrics.forEach((metric) => {
        const cell = document.createElement("td");
        const value = document.createElement("span");
        const highlight = row.highlights?.[metric.id];
        value.textContent = formatResultValue(row.values?.[metric.id], metric.decimals);
        if (highlight === "best") value.className = "result-best";
        if (highlight === "second") value.className = "result-second";
        cell.append(value);
        tableRow.append(cell);
      });
      body.append(tableRow);
    });
  });

  if (results.caption) {
    const caption = document.createElement("caption");
    const captionLabel = document.createElement("strong");
    captionLabel.textContent = results.caption_label || "Table 1.";
    caption.append(captionLabel, ` ${results.caption}`);
    table.append(caption);
  }
  table.append(head, body);
  root.replaceChildren(table);
}

function makeActive(controller) {
  if (activeAudioController && activeAudioController !== controller) activeAudioController.pause();
  activeAudioController = controller;
}

async function ensureWaveformData(variant) {
  if (variant.peaks?.length && variant.duration_seconds > 0) return variant;
  if (variant._waveformPromise) return variant._waveformPromise;

  variant._waveformPromise = (async () => {
    const response = await fetch(`${DATA_ROOT}${variant.src}`);
    if (!response.ok) throw new Error(`Audio request failed with ${response.status}: ${variant.src}`);
    const bytes = await response.arrayBuffer();
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) throw new Error("Web Audio is not supported in this browser");
    waveformDecodeContext ||= new AudioContextClass();
    const decoded = await waveformDecodeContext.decodeAudioData(bytes.slice(0));
    const frameCount = decoded.length;
    const bucketSize = Math.max(1, Math.floor(frameCount / PEAK_POINTS));
    const peaks = [];

    for (let bucket = 0; bucket < PEAK_POINTS; bucket += 1) {
      const start = bucket * bucketSize;
      const end = bucket === PEAK_POINTS - 1 ? frameCount : Math.min(frameCount, start + bucketSize);
      let peak = 0;
      for (let channelIndex = 0; channelIndex < decoded.numberOfChannels; channelIndex += 1) {
        const channel = decoded.getChannelData(channelIndex);
        for (let sampleIndex = start; sampleIndex < end; sampleIndex += 1) {
          peak = Math.max(peak, Math.abs(channel[sampleIndex]));
        }
      }
      peaks.push(Math.min(1, peak));
    }

    variant.peaks = peaks;
    variant.duration_seconds = decoded.duration;
    return variant;
  })();
  return variant._waveformPromise;
}

function drawWaveform(canvas, variant, position, selected = true) {
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const pixelRatio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(bounds.width * pixelRatio));
  const height = Math.max(1, Math.floor(bounds.height * pixelRatio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, width, height);
  const peaks = variant.peaks || [];

  if (!peaks.length) {
    context.strokeStyle = "#d7dfdc";
    context.lineWidth = pixelRatio;
    context.beginPath();
    context.moveTo(0, height / 2);
    context.lineTo(width, height / 2);
    context.stroke();
    return;
  }

  const slot = width / peaks.length;
  const barWidth = Math.max(1, slot * 0.56);
  const middle = height / 2;
  const drawBars = (color) => {
    context.fillStyle = color;
    peaks.forEach((peak, index) => {
      const barHeight = Math.max(2 * pixelRatio, peak * height * 0.82);
      const x = index * slot + (slot - barWidth) / 2;
      context.fillRect(x, middle - barHeight / 2, barWidth, barHeight);
    });
  };

  drawBars(selected ? "#b7c8c4" : "#cbd4d1");
  const progress = variant.duration_seconds ? Math.min(1, position / variant.duration_seconds) : 0;
  if (progress > 0) {
    context.save();
    context.beginPath();
    context.rect(0, 0, width * progress, height);
    context.clip();
    drawBars(selected ? "#287d72" : "#78a9a1");
    context.restore();
  }
}

class StandaloneWavePlayer {
  constructor(root, variant, itemTitle) {
    this.root = root;
    this.variant = variant;
    this.itemTitle = itemTitle;
    this.position = 0;
    this.isPlaying = false;
    this.loaded = false;
    this.frame = null;
    this.audio = new Audio();
    this.audio.preload = "none";
    this.render();
    this.bindAudio();
    this.resizeObserver = new ResizeObserver(() => this.draw());
    this.resizeObserver.observe(this.canvas);
    ensureWaveformData(this.variant)
      .then(() => { this.updateTime(); this.draw(); })
      .catch((error) => this.showError(error));
  }

  render() {
    this.root.className = "standalone-player";
    const top = document.createElement("div");
    top.className = "standalone-top";
    this.playButton = document.createElement("button");
    this.playButton.className = "mini-play";
    this.playButton.type = "button";
    this.playButton.textContent = "▶";
    this.playButton.setAttribute("aria-label", `Play ${this.variant.label} for ${this.itemTitle}`);
    this.playButton.addEventListener("click", () => this.toggle());
    const label = document.createElement("span");
    label.className = "standalone-label";
    label.textContent = this.variant.label;
    const labelGroup = document.createElement("span");
    labelGroup.className = "standalone-label-group";
    labelGroup.append(label);
    const score = makeMertScore(this.variant);
    if (score) labelGroup.append(score);
    this.time = document.createElement("span");
    this.time.className = "standalone-time";
    top.append(this.playButton, labelGroup, this.time);

    this.waveformButton = document.createElement("button");
    this.waveformButton.className = "standalone-waveform";
    this.waveformButton.type = "button";
    this.waveformButton.setAttribute("aria-label", `Seek ${this.variant.label}`);
    this.canvas = document.createElement("canvas");
    this.canvas.setAttribute("aria-hidden", "true");
    this.waveformButton.append(this.canvas);
    this.waveformButton.addEventListener("click", (event) => {
      const bounds = this.waveformButton.getBoundingClientRect();
      const ratio = bounds.width ? (event.clientX - bounds.left) / bounds.width : 0;
      this.position = Math.max(0, Math.min(1, ratio)) * this.variant.duration_seconds;
      if (this.loaded) this.audio.currentTime = this.position;
      this.updateTime();
      this.draw();
    });
    this.root.append(top, this.waveformButton);
    this.updateTime();
  }

  bindAudio() {
    this.audio.addEventListener("play", () => {
      makeActive(this);
      this.isPlaying = true;
      this.playButton.textContent = "Ⅱ";
      this.tick();
    });
    this.audio.addEventListener("pause", () => {
      this.position = Number.isFinite(this.audio.currentTime) ? this.audio.currentTime : this.position;
      this.isPlaying = false;
      this.playButton.textContent = "▶";
      if (this.frame) cancelAnimationFrame(this.frame);
      this.frame = null;
      this.updateTime();
      this.draw();
    });
    this.audio.addEventListener("ended", () => {
      this.position = 0;
      this.isPlaying = false;
      this.playButton.textContent = "▶";
      this.updateTime();
      this.draw();
    });
    this.audio.addEventListener("error", () => this.showError(new Error("Audio could not be loaded")));
  }

  toggle() {
    if (this.isPlaying) this.pause();
    else this.play();
  }

  play() {
    makeActive(this);
    if (!this.loaded) {
      this.audio.src = `${DATA_ROOT}${this.variant.src}`;
      this.loaded = true;
      const setPosition = () => { this.audio.currentTime = Math.min(this.position, Math.max(0, this.audio.duration - 0.02)); };
      if (this.audio.readyState >= 1) setPosition();
      else this.audio.addEventListener("loadedmetadata", setPosition, { once: true });
    }
    this.audio.play().catch(() => { this.isPlaying = false; this.playButton.textContent = "▶"; });
  }

  pause() { if (!this.audio.paused) this.audio.pause(); }

  tick() {
    if (!this.isPlaying) return;
    this.position = this.audio.currentTime;
    this.updateTime();
    this.draw();
    this.frame = requestAnimationFrame(() => this.tick());
  }

  updateTime() { this.time.textContent = `${formatTime(this.position)} / ${formatTime(this.variant.duration_seconds)}`; }
  draw() { drawWaveform(this.canvas, this.variant, this.position, true); }

  showError(error) {
    if (this.root.querySelector(".player-error")) return;
    const message = document.createElement("p");
    message.className = "player-error";
    message.textContent = "Audio unavailable";
    this.root.append(message);
    console.error(error);
  }
}

class GeneratedComparison {
  constructor(root, variants, itemTitle, includeSuppressOthers = false) {
    this.root = root;
    this.variants = variants;
    this.itemTitle = itemTitle;
    this.selectedIndex = 0;
    this.position = 0;
    this.isPlaying = false;
    this.loadedVariantId = null;
    this.rows = [];
    this.frame = null;
    this.audio = new Audio();
    this.audio.preload = "none";
    this.render();
    this.bindAudio();
    this.setShowSuppressOthers(includeSuppressOthers);
    this.resizeObserver = new ResizeObserver(() => this.drawAll());
    this.rows.forEach(({ canvas }) => this.resizeObserver.observe(canvas));
    this.variants.forEach((variant) => {
      ensureWaveformData(variant)
        .then(() => { this.updateTime(); this.drawAll(); })
        .catch((error) => this.showError(error));
    });
  }

  render() {
    const transport = document.createElement("div");
    transport.className = "player-transport";
    this.playButton = document.createElement("button");
    this.playButton.className = "play-button";
    this.playButton.type = "button";
    this.playButton.textContent = "▶";
    this.playButton.setAttribute("aria-label", `Play generated result for ${this.itemTitle}`);
    this.playButton.addEventListener("click", () => this.toggle());
    this.transportLabel = document.createElement("span");
    this.transportLabel.className = "transport-label";
    this.transportLabel.textContent = this.variants[0].label;
    const hint = document.createElement("span");
    hint.className = "transport-hint";
    hint.textContent = "Switch variants without losing position";
    this.transportTime = document.createElement("span");
    this.transportTime.className = "transport-time";
    transport.append(this.playButton, this.transportLabel, hint, this.transportTime);

    const list = document.createElement("div");
    list.className = "variant-list";
    this.variants.forEach((variant, index) => {
      const row = document.createElement("div");
      row.className = `variant-row${index === 0 ? " is-selected" : ""}`;
      const label = document.createElement("button");
      label.className = "variant-label";
      label.type = "button";
      const labelText = document.createElement("span");
      labelText.textContent = variant.label;
      label.append(labelText);
      const score = makeMertScore(variant);
      if (score) label.append(score);
      label.addEventListener("click", () => this.selectVariant(index));
      const waveformButton = document.createElement("button");
      waveformButton.className = "waveform-button";
      waveformButton.type = "button";
      waveformButton.setAttribute("aria-label", `Select and seek ${variant.label}`);
      const canvas = document.createElement("canvas");
      canvas.setAttribute("aria-hidden", "true");
      waveformButton.append(canvas);
      waveformButton.addEventListener("click", (event) => {
        const bounds = waveformButton.getBoundingClientRect();
        const ratio = bounds.width ? (event.clientX - bounds.left) / bounds.width : 0;
        this.position = Math.max(0, Math.min(1, ratio)) * variant.duration_seconds;
        this.selectVariant(index, { seek: true });
      });
      const duration = document.createElement("span");
      duration.className = "variant-duration";
      duration.textContent = formatTime(variant.duration_seconds);
      row.append(label, waveformButton, duration);
      list.append(row);
      this.rows.push({ row, canvas, duration, variant, labelText, waveformButton });
    });
    this.root.append(transport, list);
    this.updateTime();
  }

  bindAudio() {
    this.audio.addEventListener("play", () => {
      makeActive(this);
      this.isPlaying = true;
      this.playButton.textContent = "Ⅱ";
      this.tick();
    });
    this.audio.addEventListener("pause", () => {
      this.position = Number.isFinite(this.audio.currentTime) ? this.audio.currentTime : this.position;
      this.isPlaying = false;
      this.playButton.textContent = "▶";
      if (this.frame) cancelAnimationFrame(this.frame);
      this.frame = null;
      this.updateTime();
      this.drawAll();
    });
    this.audio.addEventListener("ended", () => {
      this.position = 0;
      this.isPlaying = false;
      this.playButton.textContent = "▶";
      this.updateTime();
      this.drawAll();
    });
    this.audio.addEventListener("error", () => this.showError(new Error("Audio could not be loaded")));
  }

  currentVariant() { return this.variants[this.selectedIndex]; }

  setShowSuppressOthers(show) {
    const suppressIndex = this.variants.findIndex((variant) => variant.id === "suppress");
    const othersIndex = this.variants.findIndex((variant) => variant.id === "suppress_others");
    if (suppressIndex >= 0) {
      const targetLabel = show ? "Suppress Target" : "Suppress";
      this.variants[suppressIndex].label = targetLabel;
      const suppressRow = this.rows[suppressIndex];
      suppressRow.labelText.textContent = targetLabel;
      suppressRow.waveformButton.setAttribute("aria-label", `Select and seek ${targetLabel}`);
    }
    if (othersIndex >= 0) this.rows[othersIndex].row.hidden = !show;
    if (!show && this.selectedIndex === othersIndex && suppressIndex >= 0) {
      this.pause();
      this.selectVariant(suppressIndex);
    } else {
      this.transportLabel.textContent = this.currentVariant().label;
      this.drawAll();
    }
  }

  selectVariant(index, options = {}) {
    const wasPlaying = this.isPlaying;
    if (this.loadedVariantId && Number.isFinite(this.audio.currentTime)) this.position = this.audio.currentTime;
    this.selectedIndex = index;
    const variant = this.currentVariant();
    this.position = Math.min(this.position, Math.max(0, variant.duration_seconds - 0.02));
    this.rows.forEach(({ row }, rowIndex) => row.classList.toggle("is-selected", rowIndex === index));
    this.transportLabel.textContent = variant.label;
    this.updateTime();
    this.drawAll();
    if (wasPlaying) this.loadAndPlay();
    else if (options.seek && this.loadedVariantId === variant.id) this.audio.currentTime = this.position;
  }

  toggle() { if (this.isPlaying) this.pause(); else this.loadAndPlay(); }

  loadAndPlay() {
    const variant = this.currentVariant();
    makeActive(this);
    if (this.loadedVariantId !== variant.id) {
      this.audio.pause();
      this.audio.src = `${DATA_ROOT}${variant.src}`;
      this.loadedVariantId = variant.id;
      const setPosition = () => { this.audio.currentTime = Math.min(this.position, Math.max(0, this.audio.duration - 0.02)); };
      if (this.audio.readyState >= 1) setPosition();
      else this.audio.addEventListener("loadedmetadata", setPosition, { once: true });
    } else {
      this.audio.currentTime = Math.min(this.position, Math.max(0, variant.duration_seconds - 0.02));
    }
    this.audio.play().catch(() => { this.isPlaying = false; this.playButton.textContent = "▶"; });
  }

  pause() { if (!this.audio.paused) this.audio.pause(); }

  tick() {
    if (!this.isPlaying) return;
    this.position = this.audio.currentTime;
    this.updateTime();
    this.drawAll();
    this.frame = requestAnimationFrame(() => this.tick());
  }

  updateTime() {
    const variant = this.currentVariant();
    this.transportTime.textContent = `${formatTime(this.position)} / ${formatTime(variant.duration_seconds)}`;
    this.rows.forEach(({ duration, variant: rowVariant }) => { duration.textContent = formatTime(rowVariant.duration_seconds); });
  }

  drawAll() {
    this.rows.forEach(({ canvas, variant }, index) => drawWaveform(canvas, variant, this.position, index === this.selectedIndex));
  }

  showError(error) {
    if (this.root.querySelector(".player-error")) return;
    const message = document.createElement("p");
    message.className = "player-error";
    message.textContent = "One or more generated audio files could not be loaded.";
    this.root.append(message);
    console.error(error);
  }
}

class SamplePlayer {
  constructor(root, item, includeSuppressOthers = false) {
    this.root = root;
    this.item = item;
    const prompt = item.variants.find((variant) => variant.id === "prompt");
    const reference = item.variants.find((variant) => variant.id === "reference");
    const generated = item.variants.filter((variant) => !["prompt", "reference"].includes(variant.id));

    const contextSection = document.createElement("section");
    contextSection.className = "player-section";
    const contextHeading = document.createElement("div");
    contextHeading.className = "player-section-heading";
    contextHeading.innerHTML = "<h4>Listening context</h4><p>Prompt and ground-truth reference are played independently.</p>";
    const contextGrid = document.createElement("div");
    contextGrid.className = "context-player-grid";
    const promptRoot = document.createElement("div");
    const referenceRoot = document.createElement("div");
    contextGrid.append(promptRoot, referenceRoot);
    contextSection.append(contextHeading, contextGrid);

    const generatedSection = document.createElement("section");
    generatedSection.className = "player-section";
    const generatedHeading = document.createElement("div");
    generatedHeading.className = "player-section-heading";
    generatedHeading.innerHTML = "<h4>Generated outputs</h4><p>Select a row to compare results at a shared playback position.</p>";
    const generatedRoot = document.createElement("div");
    generatedSection.append(generatedHeading, generatedRoot);
    this.root.append(contextSection, generatedSection);

    this.promptPlayer = new StandaloneWavePlayer(promptRoot, prompt, item.source_title);
    this.referencePlayer = new StandaloneWavePlayer(referenceRoot, reference, item.source_title);
    this.generatedPlayer = new GeneratedComparison(
      generatedRoot,
      generated,
      item.source_title,
      includeSuppressOthers,
    );
  }

  setShowSuppressOthers(show) { this.generatedPlayer.setShowSuppressOthers(show); }

  pause() {
    this.promptPlayer.pause();
    this.referencePlayer.pause();
    this.generatedPlayer.pause();
  }
}

function buildSampleCard(item, indexWithinGroup, autoOpen = false) {
  const template = document.querySelector("#sample-template");
  const card = template.content.firstElementChild.cloneNode(true);
  card.dataset.sampleId = item.sample_id;
  card.querySelector(".sample-number").textContent = String(indexWithinGroup + 1).padStart(2, "0");
  card.querySelector(".sample-title").textContent = item.source_title;
  card.querySelector(".sample-artist").textContent = item.artist;
  card.querySelector(".sample-duration").textContent = formatRange(item);
  card.querySelector(".sample-caption").textContent = item.caption;
  let player = null;
  const ensurePlayer = () => {
    if (!player) {
      player = new SamplePlayer(
        card.querySelector(".comparison-player"),
        item,
        showSuppressOthers,
      );
      instantiatedSamplePlayers.add(player);
    }
  };
  card.addEventListener("toggle", () => { if (card.open) ensurePlayer(); else player?.pause(); });
  if (autoOpen) { card.open = true; ensurePlayer(); }
  card._ensurePlayer = ensurePlayer;
  return card;
}

function configureGroupToggle(groupId) {
  const button = document.querySelector(`[data-expand-group="${groupId}"]`);
  const section = document.querySelector(`[data-group="${groupId}"]`);
  button.addEventListener("click", () => {
    const cards = [...section.querySelectorAll(".sample-card")];
    const shouldExpand = cards.some((card) => !card.open);
    cards.forEach((card) => { card.open = shouldExpand; if (shouldExpand) card._ensurePlayer?.(); });
    button.textContent = shouldExpand ? "Collapse all" : "Expand all";
  });
}

function configureSuppressOthersToggle() {
  const toggle = document.querySelector("[data-toggle-suppress-others]");
  if (!toggle) return;
  toggle.checked = false;
  showSuppressOthers = false;
  toggle.addEventListener("change", () => {
    showSuppressOthers = toggle.checked;
    instantiatedSamplePlayers.forEach((player) => player.setShowSuppressOthers(showSuppressOthers));
  });
}

function renderManifest(inputManifest, displayOrder = {}, mertSimilarity = {}) {
  const manifest = normalizeManifest(inputManifest, displayOrder, mertSimilarity);
  const groups = { single: [], multiple: [] };
  manifest.items.forEach((item) => groups[item.group]?.push(item));
  Object.entries(groups).forEach(([groupId, items]) => {
    const section = document.querySelector(`[data-group="${groupId}"]`);
    const list = document.querySelector(`[data-sample-list="${groupId}"]`);
    const orderedItems = sortItemsByDisplayOrder(
      items,
      displayOrder[groupId],
      displayOrder.include_unlisted_samples !== false,
    );
    orderedItems.forEach((item, index) => list.append(buildSampleCard(item, index, groupId === "single" && index === 0)));
    section.hidden = orderedItems.length === 0;
    if (orderedItems.length > 0) configureGroupToggle(groupId);
  });
}

function renderLoadError(error) {
  const state = document.querySelector("[data-load-state]");
  state.className = "error-state";
  state.innerHTML = `
    <h3>The audio manifest could not be loaded.</h3>
    <p>Check <code>manifest.json</code>, <code>display_order.json</code>, and the audio folder in <code>project_page/demo_data/</code>.</p>
  `;
  console.error(error);
}

function renderResultsError(error) {
  const root = document.querySelector("[data-results-root]");
  const message = document.createElement("p");
  message.className = "results-state is-error";
  message.textContent = "The quantitative results file could not be loaded. Check results_data/main_results.json.";
  root.replaceChildren(message);
  console.error(error);
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${url} request failed with ${response.status}`);
  return response.json();
}

async function fetchOptionalJson(url) {
  try {
    return await fetchJson(url);
  } catch (error) {
    console.warn(`Optional data could not be loaded from ${url}`, error);
    return {};
  }
}

async function initializeAudio() {
  try {
    const [manifest, displayOrder, mertSimilarity] = await Promise.all([
      fetchJson(MANIFEST_URL),
      fetchJson(DISPLAY_ORDER_URL),
      fetchOptionalJson(MERT_SIMILARITY_URL),
    ]);
    renderManifest(manifest, displayOrder, mertSimilarity);
    document.querySelector("[data-load-state]").remove();
  } catch (error) {
    renderLoadError(error);
  }
}

async function initializeResults() {
  try {
    renderResults(await fetchJson(RESULTS_URL));
  } catch (error) {
    renderResultsError(error);
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    formatMertSimilarity,
    formatResultValue,
    inferGroup,
    normalizeItem,
    normalizeManifest,
    normalizeResults,
    sortItemsByDisplayOrder,
  };
}

if (typeof document !== "undefined") {
  configureSuppressOthersToggle();
  initializeAudio();
  initializeResults();
}
