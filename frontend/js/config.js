/**
 * MachineGuard AI — frontend configuration
 *
 * This frontend is backend-agnostic. Point it at whatever service wraps
 * src/models/predict.py and src/llm/maintenance_assistant.py — a FastAPI
 * or Flask app is the natural choice (see frontend/README.md for a
 * suggested contract).
 *
 * If API_BASE_URL is left empty, the console runs in DEMO MODE: it
 * generates a plausible prediction locally in the browser so the UI can
 * be reviewed and demoed without any backend running.
 */
window.MG_CONFIG = {
  // Example: "http://localhost:8000" or "https://your-api.example.com"
  API_BASE_URL: "http://localhost:8000",

  ENDPOINTS: {
    predict: "/api/predict",
    assistant: "/api/assistant/chat",
    health: "/api/health"
  },

  // Request timeout, ms
  REQUEST_TIMEOUT: 15000
};