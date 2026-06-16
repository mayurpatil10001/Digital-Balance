/**
 * app.js — Digital Balance frontend logic
 * Calls GET /metadata on load, builds the form dynamically,
 * POSTs to /predict on submit, renders animated results.
 */

"use strict";

/* ── Config ──────────────────────────────────────────────────────── */
const API_BASE = "http://127.0.0.1:8000";

/* ── DOM refs ────────────────────────────────────────────────────── */
const form              = document.getElementById("wellnessForm");
const submitBtn         = document.getElementById("submitBtn");
const retryBtn          = document.getElementById("retryBtn");

const resultsPlaceholder= document.getElementById("resultsPlaceholder");
const resultsLoading    = document.getElementById("resultsLoading");
const resultsError      = document.getElementById("resultsError");
const resultsContent    = document.getElementById("resultsContent");
const errorMessage      = document.getElementById("errorMessage");

const categoryBadge     = document.getElementById("categoryBadge");
const gaugeFill         = document.getElementById("gaugeFill");
const gaugeScore        = document.getElementById("gaugeScore");
const factorsList       = document.getElementById("factorsList");
const tipsList          = document.getElementById("tipsList");
const disclaimerText    = document.getElementById("disclaimerText");
const statModel         = document.getElementById("statModel");
const screentimeWarning = document.getElementById("screentimeWarning");
const sleepQualityHint  = document.getElementById("sleepQualityHint");

/* ── Sleep quality star labels ───────────────────────────────────── */
const SLEEP_LABELS = ["", "Very poor", "Poor", "Average", "Good", "Excellent"];

/* ── Gauge maths ─────────────────────────────────────────────────── */
// The arc is drawn along a semicircle with circumference ≈ 283px (π × r = π × 90)
const GAUGE_CIRC = Math.PI * 90; // ≈ 282.7

function scoreToOffset(score) {
  // dashoffset 283 = empty, 0 = full
  return GAUGE_CIRC - (score / 100) * GAUGE_CIRC;
}

/* ── Utility: show / hide panels ─────────────────────────────────── */
function showPanel(name) {
  [resultsPlaceholder, resultsLoading, resultsError, resultsContent]
    .forEach(el => el.classList.add("hidden"));

  const map = {
    placeholder: resultsPlaceholder,
    loading:     resultsLoading,
    error:       resultsError,
    content:     resultsContent,
  };
  map[name]?.classList.remove("hidden");
}

/* ── Utility: populate <select> ──────────────────────────────────── */
function populateSelect(id, options, defaultValue = null) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = "";
  options.forEach(opt => {
    const o = document.createElement("option");
    o.value = opt;
    o.textContent = opt;
    if (defaultValue && opt === defaultValue) o.selected = true;
    el.appendChild(o);
  });
}

/* ── Utility: bind slider → output ──────────────────────────────── */
function bindSlider(id, decimals = 1) {
  const slider = document.getElementById(id);
  const output = document.getElementById(`${id}-val`);
  if (!slider || !output) return;

  const update = () => {
    const val = parseFloat(slider.value);
    output.textContent = decimals === 0 ? val.toFixed(0) : val.toFixed(decimals);

    // Visual fill track
    const pct = ((val - slider.min) / (slider.max - slider.min)) * 100;
    const isStress = id === "stress_level_0_10";
    const fillColor = isStress
      ? `hsl(${40 - pct * 0.4}, 90%, 55%)`
      : `hsl(${160 - pct * 0.6}, 60%, 40%)`;
    slider.style.background =
      `linear-gradient(to right, ${fillColor} ${pct}%, var(--clr-border) ${pct}%)`;
  };

  slider.addEventListener("input", update);
  update(); // initialise
}

/* ── Screen-time consistency check ──────────────────────────────── */
function checkScreenTimeConsistency() {
  const total   = parseFloat(document.getElementById("screen_time_hours")?.value || 0);
  const work    = parseFloat(document.getElementById("work_screen_hours")?.value || 0);
  const leisure = parseFloat(document.getElementById("leisure_screen_hours")?.value || 0);
  const warn = (work + leisure) > (total + 2); // allow 2-hr buffer before warning
  screentimeWarning?.classList.toggle("hidden", !warn);
}

/* ── Sleep quality hint ──────────────────────────────────────────── */
function updateSleepHint() {
  const checked = form.querySelector("input[name='sleep_quality_1_5']:checked");
  if (checked && sleepQualityHint) {
    const val = parseInt(checked.value);
    sleepQualityHint.textContent = `${SLEEP_LABELS[val]} (${val}/5)`;
  }
}

/* ── GET /metadata ───────────────────────────────────────────────── */
async function loadMetadata() {
  try {
    const res = await fetch(`${API_BASE}/metadata`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const meta = await res.json();

    // Populate dropdowns
    populateSelect("gender",     meta.gender_options,     "Female");
    populateSelect("occupation", meta.occupation_options, "Employed");
    populateSelect("work_mode",  meta.work_mode_options,  "Hybrid");

    // Update hero chip with model info
    if (meta.model_info?.model_name && statModel) {
      const r2 = meta.model_info.test_r2 != null
        ? ` · R²=${meta.model_info.test_r2}`
        : "";
      statModel.querySelector(".stat-text").textContent =
        `${meta.model_info.model_name}${r2}`;
    }

    // Store thresholds globally for result rendering
    window._categoryThresholds = meta.category_thresholds || {};

    console.log("✔ Metadata loaded", meta);
  } catch (err) {
    console.warn("Could not load metadata:", err.message);
    // Fallback: populate with known values
    populateSelect("gender",     ["Female", "Male", "Non-binary/Other"]);
    populateSelect("occupation", ["Employed", "Student", "Self-employed", "Retired", "Unemployed"]);
    populateSelect("work_mode",  ["Remote", "In-person", "Hybrid"]);
    if (statModel) statModel.querySelector(".stat-text").textContent = "Backend offline — start API";
  }
}

/* ── POST /predict ───────────────────────────────────────────────── */
async function submitPrediction(payload) {
  const res = await fetch(`${API_BASE}/predict`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(payload),
  });

  if (!res.ok) {
    const errBody = await res.json().catch(() => ({}));
    const detail  = errBody?.detail;
    if (Array.isArray(detail)) {
      throw new Error(detail.map(d => d.msg).join("; "));
    }
    throw new Error(detail || `Server error ${res.status}`);
  }

  return res.json();
}

/* ── Render results ──────────────────────────────────────────────── */
function renderResults(data) {
  const { predicted_score, category, top_factors, tips, disclaimer } = data;
  const score = Math.round(predicted_score * 10) / 10;

  // — Category badge —
  categoryBadge.textContent = category;
  categoryBadge.className   = `category-badge ${category.toLowerCase()}`;

  // — Gauge animation —
  gaugeScore.textContent    = score.toFixed(0);
  gaugeFill.style.strokeDashoffset = scoreToOffset(score);

  // — Gauge colour based on category —
  const GAUGE_COLORS = {
    Poor:      "#f59e0b",
    Fair:      "#84cc16",
    Good:      "#22c55e",
    Excellent: "#0d9488",
  };
  // Override gradient with solid category colour for cleaner look
  gaugeFill.setAttribute("stroke", GAUGE_COLORS[category] || "#0d9488");
  gaugeFill.removeAttribute("stroke"); // re-use gradient instead

  // — Top factors —
  factorsList.innerHTML = "";
  (top_factors || []).forEach((factor, i) => {
    const li = document.createElement("li");
    li.textContent = factor;
    li.style.animationDelay = `${i * 80}ms`;
    factorsList.appendChild(li);
  });

  // — Tips —
  tipsList.innerHTML = "";
  const TIP_ICONS = ["🌱", "💤", "🏃", "🧘", "📵", "☀️"];
  (tips || []).forEach((tip, i) => {
    const li   = document.createElement("li");
    const icon = document.createElement("span");
    icon.className   = "tip-icon";
    icon.textContent = TIP_ICONS[i % TIP_ICONS.length];
    icon.setAttribute("aria-hidden", "true");
    const text = document.createElement("span");
    text.textContent = tip;
    li.appendChild(icon);
    li.appendChild(text);
    li.style.animationDelay = `${i * 100}ms`;
    tipsList.appendChild(li);
  });

  // — Disclaimer —
  if (disclaimer && disclaimerText) disclaimerText.textContent = disclaimer;

  // — Show content —
  showPanel("content");
}

/* ── Collect form values ─────────────────────────────────────────── */
function collectPayload() {
  const get    = id => document.getElementById(id)?.value;
  const getNum = (id, int = false) => {
    const v = parseFloat(get(id));
    return int ? Math.round(v) : v;
  };

  const sqChecked = form.querySelector("input[name='sleep_quality_1_5']:checked");
  const sleepQ    = sqChecked ? parseInt(sqChecked.value) : 3;

  return {
    age:                       getNum("age", true),
    gender:                    get("gender"),
    occupation:                get("occupation"),
    work_mode:                 get("work_mode"),
    screen_time_hours:         getNum("screen_time_hours"),
    work_screen_hours:         getNum("work_screen_hours"),
    leisure_screen_hours:      getNum("leisure_screen_hours"),
    sleep_hours:               getNum("sleep_hours"),
    sleep_quality_1_5:         sleepQ,
    stress_level_0_10:         getNum("stress_level_0_10"),
    productivity_0_100:        getNum("productivity_0_100"),
    exercise_minutes_per_week: getNum("exercise_minutes_per_week", true),
    social_hours_per_week:     getNum("social_hours_per_week"),
  };
}

/* ── Form submit handler ─────────────────────────────────────────── */
async function handleSubmit(e) {
  e.preventDefault();
  submitBtn.disabled = true;
  submitBtn.querySelector(".btn-text").textContent = "Calculating…";

  showPanel("loading");

  try {
    const payload = collectPayload();
    const result  = await submitPrediction(payload);
    renderResults(result);

    // Scroll results into view on mobile
    document.getElementById("resultsPanel")?.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    errorMessage.textContent = err.message || "An unexpected error occurred. Please try again.";
    showPanel("error");
    console.error("Prediction error:", err);
  } finally {
    submitBtn.disabled = false;
    submitBtn.querySelector(".btn-text").textContent = "Calculate My Score";
  }
}

/* ── Retry button ────────────────────────────────────────────────── */
retryBtn?.addEventListener("click", () => {
  showPanel("placeholder");
  form.dispatchEvent(new Event("submit"));
});

/* ── Init ────────────────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", async () => {
  // 1. Load metadata (dropdowns + model info)
  await loadMetadata();

  // 2. Bind all sliders
  [
    ["age",                       0],
    ["screen_time_hours",         1],
    ["work_screen_hours",         1],
    ["leisure_screen_hours",      1],
    ["sleep_hours",               1],
    ["stress_level_0_10",         1],
    ["productivity_0_100",        0],
    ["exercise_minutes_per_week", 0],
    ["social_hours_per_week",     1],
  ].forEach(([id, dec]) => bindSlider(id, dec));

  // 3. Screen-time consistency listeners
  ["screen_time_hours", "work_screen_hours", "leisure_screen_hours"].forEach(id =>
    document.getElementById(id)?.addEventListener("input", checkScreenTimeConsistency)
  );

  // 4. Sleep quality hint listeners
  form.querySelectorAll("input[name='sleep_quality_1_5']").forEach(radio =>
    radio.addEventListener("change", updateSleepHint)
  );
  updateSleepHint(); // initialise

  // 5. Form submit
  form.addEventListener("submit", handleSubmit);

  // 6. Pre-select sq3 star so rating appears filled on load
  const sq3 = document.getElementById("sq3");
  if (sq3) sq3.checked = true;

  console.log("✔ Digital Balance frontend initialised");
});
