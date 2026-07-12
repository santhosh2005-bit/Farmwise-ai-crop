# FarmWise AI Copilot 🌾

An AI-powered agricultural analytics platform. Ask natural-language questions about global crop yields, rainfall, temperature, and pesticides, and receive data-backed summaries, automated statistical insights, and dynamic, interactive Plotly visualizations.

---

## 🛠️ Tech Stack & AI Integration

### AI Models & Orchestration
* **Default Model**: Groq API using **`llama-3.3-70b-versatile`** (default base model configured for high reasoning precision and zero hallucinations) with native tool/function calling for structured action selection.
* **Alternative Models Supported**: Configurable in the sidebar to `llama-3.1-8b-instant` (fast / high rate limits), `openai/gpt-oss-20b`, and `qwen/qwen3.6-27b`.
* **AI Tool Integration**: Built using Groq's native **Function Calling (Tool Use)** registry to execute structured Pandas queries on the backend without prompt injection vulnerability.

### 🔌 Registered AI Database Tools (10 Function Schemas)
The Copilot is registered with exactly 10 database querying functions. The model dynamically calls these functions using native JSON schemas based on user intent:
1. **`filter_by_country`**: Retrieves all records for a country (crops, yields, rainfall, temp, pesticides), optionally filtered by year.
2. **`filter_by_item`**: Retrieves data for a specific crop item (e.g. Wheat, Rice, paddy, Maize) across all countries.
3. **`get_yield_trend`**: Retrieves the year-over-year yield data for a specific country and crop.
4. **`compare_countries`**: Compares a specific metric (yields, rainfall, etc.) across multiple countries over time.
5. **`get_top_countries`**: Deduplicates and ranks countries by a given metric (e.g. top 5 rice producers in 2013).
6. **`get_correlation`**: Calculates the Pearson correlation coefficient between any two columns (e.g., rainfall vs. yield).
7. **`get_summary_statistics`**: Provides statistics (mean, median, standard deviation, min, max) for any column.
8. **`get_pesticide_usage`**: Extracts pesticide usage records for a country over the years.
9. **`forecast_yield_trend`**: Performs linear regression to forecast future crop yields for a country and crop.
10. **`get_yield_regression_factors`**: Computes the regression slopes and intercepts to show the direct impact of rainfall, temperature, and pesticides on crop yield.

### Coding & Development Assistant Tools
* Developed with the assistance of **Gemini** (for pair-programming, module validation, Plotly integration, layout refinement, and troubleshooting).

### Software Stack
* **Backend**: FastAPI (Python) & Pandas
* **Frontend**: Streamlit
* **Visualizations**: Plotly (interactive charts with custom glassmorphism styling)

---

## 🚀 Quick Start Guide

You can run the application entirely by following the instructions below.

### 1. Prerequisites
Ensure you have **Python 3.9 to 3.12** installed on your system.

### 2. Installation & Setup

1. **Clone or navigate to the project directory**:
   ```bash
   cd Farmwise
   ```

2. **Create a Python virtual environment and activate it**:
   * **On Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   * **On macOS/Linux**:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Dependencies**:
   Install all dependencies for both the backend and frontend:
   ```bash
   pip install -r backend/requirements.txt -r frontend/requirements.txt
   ```

### 3. Configuration
1. Copy the `.env.example` template to `.env`:
   ```bash
   cp backend/.env.example backend/.env
   ```
2. Open `backend/.env` in your text editor and enter your Groq API Key:
   ```env
   GROQ_API_KEY=your_actual_groq_api_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   ```
   *(Note: You can also override the API key directly in the Streamlit web interface sidebar.)*

### 4. Running the Application

Open two separate terminals (with the virtual environment activated in both):

* **Terminal 1: Start the Backend API Server**
  ```bash
  uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
  ```

* **Terminal 2: Start the Frontend UI Dashboard**
  ```bash
  streamlit run frontend/app.py
  ```

The browser will automatically open to `http://localhost:8501`.

---

## 📊 Dataset Structure
The system includes a pre-merged table at `data/yield_df.csv` containing global crop records:
* **`Area`**: Country name
* **`Item`**: Crop category (e.g. Wheat, Rice, Maize, etc.)
* **`Year`**: 1990 - 2013
* **`hg/ha_yield`**: Crop yield in hectograms per hectare
* **`average_rain_fall_mm_per_year`**: Annual rainfall (mm)
* **`pesticides_tonnes`**: Pesticide utilization (tonnes)
* **`avg_temp`**: Country-wide average temperature (°C)
