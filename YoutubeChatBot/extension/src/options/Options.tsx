// Settings page (SRS FR-UI-07): backend URL, BYOK provider/key, auto-index,
// and a connection test that explains what is wrong in plain words.
import { useEffect, useState } from "preact/hooks";
import { ApiError, BackendClient, type ReadyStatus } from "../shared/api";
import { DEFAULT_SETTINGS, type Settings, loadSettings, saveSettings } from "../shared/storage";

const PROVIDERS = [
  { value: "", label: "Backend default (from backend .env)" },
  { value: "groq", label: "Groq" },
  { value: "google_genai", label: "Google Gemini" },
  { value: "openai", label: "OpenAI" },
  { value: "anthropic", label: "Anthropic" },
  { value: "ollama", label: "Ollama (local)" },
];

const CHECK_LABELS: Record<string, string> = {
  models_loaded: "Embedding & reranker models loaded",
  vector_store: "Vector store reachable",
  llm_key_present: "LLM API key configured on the backend",
};

type TestResult = { ok: true; ready: ReadyStatus } | { ok: false; message: string } | null;

export function Options() {
  const [form, setForm] = useState<Settings>(DEFAULT_SETTINGS);
  const [saved, setSaved] = useState(false);
  const [testing, setTesting] = useState(false);
  const [test, setTest] = useState<TestResult>(null);

  useEffect(() => {
    void loadSettings().then(setForm);
  }, []);

  const update = <K extends keyof Settings>(key: K, value: Settings[K]) => {
    setForm((f) => ({ ...f, [key]: value }));
    setSaved(false);
  };

  const save = async (e: Event) => {
    e.preventDefault();
    await saveSettings({ ...form, backendUrl: form.backendUrl.trim() || DEFAULT_SETTINGS.backendUrl });
    setSaved(true);
  };

  const runTest = async () => {
    setTesting(true);
    setTest(null);
    try {
      setTest({ ok: true, ready: await new BackendClient(form).ready() });
    } catch (err) {
      setTest({ ok: false, message: err instanceof ApiError ? err.message : "Connection test failed." });
    } finally {
      setTesting(false);
    }
  };

  return (
    <main class="options">
      <h1>YouTube Video Chat · Settings</h1>

      <form onSubmit={save}>
        <section>
          <h2>Backend</h2>
          <label for="backendUrl">Backend URL</label>
          <div class="row">
            <input
              id="backendUrl"
              type="url"
              value={form.backendUrl}
              onInput={(e) => update("backendUrl", e.currentTarget.value)}
              placeholder={DEFAULT_SETTINGS.backendUrl}
            />
            <button type="button" class="btn" onClick={runTest} disabled={testing}>
              {testing ? "Testing…" : "Test connection"}
            </button>
          </div>
          {test && <TestPanel result={test} usingOwnKey={Boolean(form.apiKey)} />}
        </section>

        <section>
          <h2>Language model</h2>
          <p class="hint">
            Leave these empty to use the backend's own configuration. Your key is stored only in this browser
            and is sent only to your backend, which passes it to the provider.
          </p>
          <label for="provider">Provider</label>
          <select id="provider" value={form.provider} onChange={(e) => update("provider", e.currentTarget.value)}>
            {PROVIDERS.map((p) => (
              <option key={p.value} value={p.value}>
                {p.label}
              </option>
            ))}
          </select>

          <label for="model">Model (optional)</label>
          <input
            id="model"
            value={form.model}
            onInput={(e) => update("model", e.currentTarget.value)}
            placeholder="Provider default"
            disabled={!form.provider}
          />

          <label for="apiKey">API key (optional)</label>
          <input
            id="apiKey"
            type="password"
            autocomplete="off"
            value={form.apiKey}
            onInput={(e) => update("apiKey", e.currentTarget.value)}
            placeholder="Uses the key from the backend .env"
            disabled={!form.provider}
          />
        </section>

        <section>
          <h2>Behaviour</h2>
          <label class="check">
            <input
              type="checkbox"
              checked={form.autoIndex}
              onChange={(e) => update("autoIndex", e.currentTarget.checked)}
            />
            Read each video as soon as it opens (faster first answer, more requests to YouTube)
          </label>
        </section>

        <div class="actions">
          <button type="submit" class="btn btn-primary">
            Save
          </button>
          {saved && (
            <span class="saved" role="status">
              Saved ✓
            </span>
          )}
        </div>
      </form>
    </main>
  );
}

function TestPanel({ result, usingOwnKey }: { result: NonNullable<TestResult>; usingOwnKey: boolean }) {
  if (!result.ok) {
    return (
      <div class="test test-bad" role="alert">
        <strong>Not connected.</strong> {result.message}
        <p class="hint">
          Start it from the <code>backend</code> folder:{" "}
          <code>..\venv\Scripts\python.exe -m uvicorn app.main:app --port 8000</code>
        </p>
      </div>
    );
  }
  const { ready } = result;
  return (
    <div class={`test ${ready.ready ? "test-ok" : "test-warn"}`} role="status">
      <strong>{ready.ready ? "Connected and ready." : "Connected, but not ready yet."}</strong>
      <ul>
        {Object.entries(ready.checks).map(([key, ok]) => (
          <li key={key}>
            {ok ? "✓" : "✗"} {CHECK_LABELS[key] ?? key}
            {key === "llm_key_present" && !ok && usingOwnKey && " (fine: you set your own key)"}
          </li>
        ))}
      </ul>
      <p class="hint">
        Backend LLM: {ready.llm.provider} / {ready.llm.model}
      </p>
    </div>
  );
}
