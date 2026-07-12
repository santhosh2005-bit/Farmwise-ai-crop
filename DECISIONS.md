# Architecture Decision Records (ADRs)

This file documents the key design decisions made for **FarmWise AI Copilot** and the reasoning behind each.

---

## ADR-001: Groq API with Tool Calling

**Decision:** Use Groq's APIs with native function/tool calling.
* **Default Model**: `llama-3.3-70b-versatile` is used as the default base model to guarantee logical correctness, accurate tool selection, and arithmetic/unit precision (preventing hallucinations).
* **Alternative Model**: `llama-3.1-8b-instant` (and others) are supported as fast, high-limit alternatives in the settings panel.

**Rationale:**
- The LLM dynamically selects which data operation to run — no hard-coded question→chart mappings.
- Tool calling produces structured JSON arguments, eliminating fragile prompt-based output parsing.
- Groq offers extremely low latency for inference.

---

## ADR-002: FastAPI + Streamlit Split

**Decision:** Separate the backend (FastAPI) from the frontend (Streamlit) as two independently-runnable processes.

**Rationale:**
- Clean separation of concerns — data logic vs. UI rendering.
- The API can be consumed by other clients (mobile, CLI) in the future.
- Each layer can be scaled, tested, and deployed independently.

---

## ADR-003: Pre-Merged Dataset as Default

**Decision:** Load `yield_df.csv` (the pre-merged table) by default instead of joining raw CSVs at runtime.

**Rationale:**
- Faster startup — no join logic on every reload.
- The merged file is the single source of truth for the analytics layer.
- Raw files remain in `data/` for reference and reproducibility.

---

## ADR-004: Module-Level Dataset Cache

**Decision:** Cache the loaded DataFrame at module level in `dataset.py` (a lazy singleton pattern).

**Rationale:**
- Pandas DataFrames are expensive to load repeatedly.
- A single cached copy keeps memory predictable and avoids re-reading the CSV on every request.
- The cache can be invalidated by passing an explicit `path` argument.

---

## ADR-005: In-Memory Conversation History & Memory Management

**Decision:** Store chat history in a per-session Python dataclass (`memory.py`) rather than an external store.

**Rationale:**
- Simplest thing that works for a process deployment.
- Swappable: The `ConversationMemory` class has a clean interface that can be easily replaced by Redis or a database later.

**Memory Management & Truncation (Addressing Token Limits):**
- **Turn Limits**: The conversation history is trimmed to retain the system prompt and the most recent user-assistant turns to keep context usage within bounds.
- **Nested Truncation**: To prevent large query outputs (e.g., comparing multiple countries) from overflowing context tokens, raw tool output is intercepted and nested lists under the `"data"` key are truncated to 5 rows when represented as JSON.

---

## ADR-006: Dynamic Chart Generation & Deciding What to Visualize

**Decision:** The chart type and parameters are determined dynamically by the LLM's response or auto-generated fallbacks based on tool data.

**Rationale:**
- Keeps the system flexible — new question types don't require code changes.
- Plotly figures serialise to JSON cleanly, making API transport trivial.
- The frontend is a pure renderer with no chart-building logic of its own.

**Deciding What to Visualize:**
* **Primary (LLM Spec)**: The LLM can output a JSON chart specification inside a ```chart ... ``` code block containing chart parameters (`chart_type`, `x`, `y`, `color`, `title`).
* **Fallback (Auto-Generation)**: If no spec is generated, the backend auto-generates charts depending on which tool was executed:
  - `compare_countries` → Multi-country Line Chart over time.
  - `get_top_countries` → Ranked Bar Chart of countries.
* **Zero Hardcoded Keyword Mappings**: No hardcoded keyword-to-chart mappings (e.g., `if "yield" in query: ...`) are used. The chart specification is driven entirely by the LLM's interpretation of user intent. Any fallback charting is determined solely by the structural columns returned by the executed database tool, ensuring flexibility for unseen questions.
  - `get_correlation` → Scatter Plot.
  - `filter_by_item` → Bar Chart of the crop's yield across countries.
* **Polish & Readability Rules**:
  - **Sorting**: Bar charts are automatically sorted in descending order by their Y-value to establish visual hierarchy.
  - **Auto-Legend Hiding**: Legends are hidden (`showlegend=False`) when `color` matches `x` (redundant) or when there are more than 10 categories, avoiding visual clutter.
  - **Tick Rotation**: Categorical x-axis labels are rotated by 45 degrees to prevent vertical overlap.

---

## ADR-007: Secrets via `.env`

**Decision:** All secrets (Groq API key) are read from a `.env` file via `python-dotenv`. The `.env` file is git-ignored; a `.env.example` template is committed.

**Rationale:**
- Prevents accidental key exposure in version control.
- Works identically in local dev and in CI/CD (inject env vars).

---

## ADR-008: Handling Unresolvable or Edge-Case Questions

**Decision:** Maintain strict anti-fabrication rules, provide warning payloads from tool dispatchers, and require step-by-step math verification.

**Rationale:**
- **No Data / Missing Entries**: When a user queries countries or crops not in the dataset, tools return warning details (e.g. `No data found for country X`). The LLM reads these warnings and transparently informs the user instead of hallucinating.
- **Math Verification**: To prevent arithmetic hallucinations when calculating differences, percentage growth, or ratios, the system prompt explicitly commands the model to show the formula step-by-step (e.g. `69,533 - 33,649 = 35,884 hg/ha`) and double-check calculations.
- **Unit Precision**: The prompt enforces strict alignment to standard dataset columns and units (`hg/ha_yield`, `average_rain_fall_mm_per_year` in mm/year, `avg_temp` in °C, and `pesticides_tonnes` in tonnes) to avoid unit conversion hallucinations.
