(function () {
  "use strict";

  const els = {
    navTabs: document.querySelectorAll(".nav-tab"),
    views: {
      dashboard: document.getElementById("view-dashboard"),
      assistant: document.getElementById("view-assistant"),
      log: document.getElementById("view-log")
    },
    connDot: document.getElementById("connStatusDot"),
    connLabel: document.getElementById("connStatusLabel"),
    apiTargetLabel: document.getElementById("apiTargetLabel"),

    form: document.getElementById("sensorForm"),
    runBtn: document.getElementById("runBtn"),
    loadSampleBtn: document.getElementById("loadSampleBtn"),

    gaugeSvg: document.getElementById("gaugeSvg"),
    gaugeValue: document.getElementById("gaugeValue"),
    gaugeTag: document.getElementById("gaugeTag"),
    gaugeTimestamp: document.getElementById("gaugeTimestamp"),

    diagMode: document.getElementById("diagMode"),
    diagConfidence: document.getElementById("diagConfidence"),
    recoList: document.getElementById("recoList"),

    terminalLog: document.getElementById("terminalLog"),
    chatForm: document.getElementById("chatForm"),
    chatInput: document.getElementById("chatInput"),
    quickPrompts: document.getElementById("quickPrompts"),

    historyBody: document.getElementById("historyBody"),
    historyCount: document.getElementById("historyCount")
  };

  const RISK_LABEL = { safe: "NOMINAL", watch: "ELEVATED", crit: "CRITICAL" };
  let sessionHistory = [];

  // ---------------- View switching ----------------
  els.navTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      els.navTabs.forEach(t => t.classList.remove("is-active"));
      tab.classList.add("is-active");
      Object.values(els.views).forEach(v => v.classList.remove("is-active"));
      els.views[tab.dataset.view].classList.add("is-active");
    });
  });

  // ---------------- Init ----------------
  function init() {
    MGGauge.build(els.gaugeSvg);
    updateConnectionBadge();

    els.form.addEventListener("submit", onRunDiagnostic);
    els.loadSampleBtn.addEventListener("click", loadRandomSample);
    els.chatForm.addEventListener("submit", onSendChat);
    els.quickPrompts.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-prompt]");
      if (!btn) return;
      els.chatInput.value = btn.dataset.prompt;
      els.chatForm.requestSubmit();
    });
  }

  function updateConnectionBadge() {
    if (MGApi.isDemoMode()) {
      els.connDot.dataset.state = "offline";
      els.connLabel.textContent = "demo mode";
      els.apiTargetLabel.textContent = "API: not configured (demo mode) — edit js/config.js";
    } else {
      els.connDot.dataset.state = "online";
      els.connLabel.textContent = "connected";
      els.apiTargetLabel.textContent = `API: ${window.MG_CONFIG.API_BASE_URL}`;
    }
  }

  // ---------------- Diagnostic form ----------------
  function readForm() {
    const type = els.form.querySelector('input[name="type"]:checked').value;
    return {
      type,
      airTemp: parseFloat(document.getElementById("airTemp").value),
      processTemp: parseFloat(document.getElementById("processTemp").value),
      rotSpeed: parseFloat(document.getElementById("rotSpeed").value),
      torque: parseFloat(document.getElementById("torque").value),
      toolWear: parseFloat(document.getElementById("toolWear").value)
    };
  }

  function loadRandomSample() {
    const rand = (min, max, decimals = 0) => {
      const v = Math.random() * (max - min) + min;
      return Number(v.toFixed(decimals));
    };
    const types = ["L", "M", "H"];
    document.getElementById(`type-${types[Math.floor(Math.random() * 3)]}`).checked = true;
    document.getElementById("airTemp").value = rand(295, 304, 1);
    document.getElementById("processTemp").value = rand(305, 313, 1);
    document.getElementById("rotSpeed").value = rand(1180, 2860, 0);
    document.getElementById("torque").value = rand(3, 76, 1);
    document.getElementById("toolWear").value = rand(0, 250, 0);
  }

  async function onRunDiagnostic(evt) {
    evt.preventDefault();
    const input = readForm();

    if (Object.values(input).some(v => typeof v === "number" && Number.isNaN(v))) {
      showToast("Fill in every sensor field before running a diagnostic.");
      return;
    }

    setRunning(true);
    try {
      const result = await MGApi.predict(input);
      applyResult(input, result);
    } catch (err) {
      console.error(err);
      showToast("Diagnostic request failed: " + err.message);
    } finally {
      setRunning(false);
    }
  }

  function setRunning(isRunning) {
    els.runBtn.classList.toggle("is-loading", isRunning);
    els.runBtn.disabled = isRunning;
    els.runBtn.querySelector(".btn-run__label").textContent = isRunning ? "Running…" : "Run Diagnostic";
  }

  function applyResult(input, result) {
    const pct = Math.round(result.failureProbability * 100);
    MGGauge.setValue(els.gaugeSvg, pct);
    els.gaugeValue.firstChild.textContent = pct;
    els.gaugeTag.textContent = RISK_LABEL[result.riskLevel] || "—";
    els.gaugeTag.dataset.level = result.riskLevel;
    els.gaugeTimestamp.textContent = new Date().toLocaleTimeString();

    els.diagMode.textContent = result.predictedMode || "—";
    els.diagConfidence.textContent = result.confidence ? `${Math.round(result.confidence * 100)}%` : "—";

    els.recoList.innerHTML = "";
    (result.recommendations && result.recommendations.length ? result.recommendations : ["No specific recommendation returned."])
      .forEach(text => {
        const li = document.createElement("li");
        li.textContent = text;
        els.recoList.appendChild(li);
      });

    recordHistory(input, result, pct);
  }

  // ---------------- History ----------------
  function recordHistory(input, result, pct) {
    sessionHistory.unshift({ time: new Date(), input, result, pct });
    if (sessionHistory.length > 50) sessionHistory.pop();
    renderHistory();
  }

  function renderHistory() {
    els.historyCount.textContent = `${sessionHistory.length} reading${sessionHistory.length === 1 ? "" : "s"}`;
    if (!sessionHistory.length) {
      els.historyBody.innerHTML = '<tr class="history-empty"><td colspan="8">No diagnostics run this session yet.</td></tr>';
      return;
    }
    els.historyBody.innerHTML = sessionHistory.map(row => `
      <tr>
        <td>${row.time.toLocaleTimeString()}</td>
        <td>${row.input.type}</td>
        <td>${row.input.airTemp.toFixed(1)} K</td>
        <td>${row.input.processTemp.toFixed(1)} K</td>
        <td>${row.input.rotSpeed.toFixed(0)} rpm</td>
        <td>${row.input.torque.toFixed(1)} Nm</td>
        <td>${row.input.toolWear.toFixed(0)} min</td>
        <td><span class="risk-chip" data-level="${row.result.riskLevel}">${row.pct}%</span></td>
      </tr>
    `).join("");
  }

  // ---------------- Assistant chat ----------------
  function appendTermLine(role, text) {
    const line = document.createElement("div");
    line.className = `term-line term-line--${role}`;
    const promptLabel = role === "user" ? "you" : role === "assistant" ? "ai" : "sys";
    line.innerHTML = `<span class="term-prompt">${promptLabel}</span><span></span>`;
    line.querySelector("span:last-child").textContent = text;
    els.terminalLog.appendChild(line);
    els.terminalLog.scrollTop = els.terminalLog.scrollHeight;
    return line;
  }

  async function onSendChat(evt) {
    evt.preventDefault();
    const message = els.chatInput.value.trim();
    if (!message) return;

    appendTermLine("user", message);
    els.chatInput.value = "";
    els.chatInput.disabled = true;
    const pending = appendTermLine("assistant", "Retrieving from knowledge base…");

    try {
      const resp = await MGApi.chat(message);
      pending.querySelector("span:last-child").textContent = resp.answer;
      if (resp.sources && resp.sources.length) {
        appendTermLine("system", `sources: ${resp.sources.join(", ")}`);
      }
    } catch (err) {
      pending.querySelector("span:last-child").textContent = "Request failed: " + err.message;
    } finally {
      els.chatInput.disabled = false;
      els.chatInput.focus();
    }
  }

  // ---------------- Toast ----------------
  let toastTimer;
  function showToast(message) {
    let toast = document.querySelector(".toast");
    if (!toast) {
      toast = document.createElement("div");
      toast.className = "toast";
      document.body.appendChild(toast);
    }
    toast.textContent = message;
    requestAnimationFrame(() => toast.classList.add("is-visible"));
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("is-visible"), 4200);
  }

  document.addEventListener("DOMContentLoaded", init);
})();