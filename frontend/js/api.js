/**
 * Thin API client for the MachineGuard AI backend.
 *
 * Expected backend contract (see README.md):
 *
 *   POST {API_BASE_URL}/api/predict
 *     body: { type, airTemp, processTemp, rotSpeed, torque, toolWear }
 *     resp: { failureProbability: 0-1, riskLevel: "safe"|"watch"|"crit",
 *             predictedMode: string, confidence: 0-1, recommendations: string[] }
 *
 *   POST {API_BASE_URL}/api/assistant/chat
 *     body: { message, context? }
 *     resp: { answer: string, sources?: string[] }
 *
 * When API_BASE_URL is empty, both calls resolve using a local
 * heuristic so the interface is fully explorable without a server.
 */
(function () {
  const cfg = window.MG_CONFIG;

  function isDemoMode() {
    return !cfg.API_BASE_URL || cfg.API_BASE_URL.trim() === "";
  }

  async function withTimeout(promise, ms) {
    let timer;
    const timeout = new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error("Request timed out")), ms);
    });
    try {
      return await Promise.race([promise, timeout]);
    } finally {
      clearTimeout(timer);
    }
  }

  async function postJSON(path, payload) {
    const url = cfg.API_BASE_URL.replace(/\/$/, "") + path;
    const res = await withTimeout(
      fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      }),
      cfg.REQUEST_TIMEOUT
    );
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`API responded ${res.status}: ${text || res.statusText}`);
    }
    return res.json();
  }

  // ---- Local demo heuristics (mirrors the general shape of the AI4I 2020
  // failure logic so the UI behaves sensibly offline; NOT the real model) ----
  function demoPredict(input) {
    const tempDelta = input.processTemp - input.airTemp;
    const power = input.torque * (input.rotSpeed * (2 * Math.PI / 60)); // ~ watts
    const wearFactor = input.toolWear / 250;
    const typeFactor = { L: 1.1, M: 1.0, H: 0.85 }[input.type] ?? 1;

    let score =
      (wearFactor * 0.45) +
      (Math.max(0, (tempDelta - 8) / 12) * 0.25) +
      (Math.max(0, (power - 3500) / 6000) * 0.2) +
      (Math.max(0, (input.torque - 55) / 40) * 0.15);

    score = Math.min(1, Math.max(0, score * typeFactor));
    // add mild deterministic jitter based on inputs so repeated identical
    // inputs give identical (not random) results
    const seed = (input.airTemp + input.processTemp + input.rotSpeed + input.torque + input.toolWear) % 7;
    score = Math.min(1, Math.max(0, score + (seed - 3) * 0.01));

    let mode = "Nominal operation";
    if (score > 0.65) {
      mode = wearFactor > 0.6 ? "Tool wear failure" : tempDelta < 8.6 ? "Heat dissipation failure" : "Power failure";
    } else if (score > 0.3) {
      mode = "Overstrain risk (elevated torque × wear)";
    }

    const riskLevel = score >= 0.65 ? "crit" : score >= 0.30 ? "watch" : "safe";

    const recommendations = buildDemoRecommendations(riskLevel, { tempDelta, wearFactor, torque: input.torque });

    return {
      failureProbability: score,
      riskLevel,
      predictedMode: mode,
      confidence: 0.78 + Math.random() * 0.15,
      recommendations,
      _demo: true
    };
  }

  function buildDemoRecommendations(riskLevel, ctx) {
    if (riskLevel === "crit") {
      return [
        "Schedule the unit for inspection before the next production run — do not defer past this shift.",
        ctx.wearFactor > 0.6
          ? "Tool wear is the dominant contributor; replace or recondition the cutting tool per the standard wear-limit procedure."
          : "Verify coolant flow and ambient ventilation; the air-to-process temperature margin is below the safe operating band.",
        "Log this reading in the maintenance record and flag the asset for the next preventive-maintenance cycle."
      ];
    }
    if (riskLevel === "watch") {
      return [
        "No immediate action required, but trend this reading against the last 5 cycles for drift.",
        "Confirm torque and rotational speed are within the product-type's rated envelope.",
        "Re-run the diagnostic after the next operating cycle to confirm the trend direction."
      ];
    }
    return [
      "Operating within nominal parameters. Continue standard monitoring cadence.",
      "No maintenance action indicated at this time."
    ];
  }

  function demoAssistantReply(message) {
    const m = message.toLowerCase();
    let answer;
    if (m.includes("heat") || m.includes("dissipation")) {
      answer = "Heat dissipation failure typically emerges when the gap between process and air temperature narrows below the safe margin while rotational speed stays low, limiting convective cooling. Watch for a shrinking temperature differential alongside sub-rated RPM, and confirm coolant flow and airflow paths are unobstructed before returning the unit to service.";
    } else if (m.includes("tool wear") || m.includes("wear")) {
      answer = "Tool wear failure risk climbs sharply as accumulated wear minutes approach the tool's rated life, especially when combined with above-average torque. A practical inspection cadence is every 20–25 operating hours for high-duty tooling, with immediate inspection triggered if predicted failure probability crosses the elevated band.";
    } else if (m.includes("torque") || m.includes("overstrain")) {
      answer = "Overstrain risk increases when torque and tool wear compound — high torque alone is often tolerable, but paired with wear beyond roughly 60% of rated life it becomes a leading indicator. As a rule of thumb, treat sustained torque above the product type's rated ceiling combined with elevated wear as a trigger for inspection.";
    } else {
      answer = "That's outside what the local demo knowledge base covers. Connect this console to the FAISS-backed assistant service (see maintenance_assistant.py) to answer questions grounded in the full maintenance_guidelines.txt corpus.";
    }
    return { answer, sources: ["maintenance_guidelines.txt (demo excerpt)"], _demo: true };
  }

  // ---- Public API ----
  async function predict(input) {
    if (isDemoMode()) {
      await new Promise(r => setTimeout(r, 550)); // simulate latency
      return demoPredict(input);
    }
    return postJSON(cfg.ENDPOINTS.predict, input);
  }

  async function chat(message, context) {
    if (isDemoMode()) {
      await new Promise(r => setTimeout(r, 500));
      return demoAssistantReply(message);
    }
    return postJSON(cfg.ENDPOINTS.assistant, { message, context });
  }

  async function checkHealth() {
    if (isDemoMode()) return { status: "demo" };
    const url = cfg.API_BASE_URL.replace(/\/$/, "") + cfg.ENDPOINTS.health;
    const res = await withTimeout(fetch(url), 5000);
    if (!res.ok) throw new Error("Health check failed");
    return res.json();
  }

  window.MGApi = { predict, chat, checkHealth, isDemoMode };
})();