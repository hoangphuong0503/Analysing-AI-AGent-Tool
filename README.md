# 🤖 Amy — Autonomous AI Data Analysis Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-black?logo=ollama&logoColor=white" alt="Ollama" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/Matplotlib%20%7C%20Seaborn-Visualization-11557c" alt="Visualization" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
</p>

**Amy** is an intelligent, full-stack autonomous AI Data Analyst Agent built with **FastAPI**, **Pandas**, and local **LLMs (via Ollama)**. Amy empowers users to upload complex data files, perform automated exploratory data analysis (EDA), clean messy datasets, generate rich visualizations, query datasets using plain natural language, and export production-ready reports.

---

## 🌟 Key Features

### 1. 🧠 Autonomous ReAct AI Agent
- **Multi-Turn Reasoning**: Powered by a ReAct (Reasoning + Acting) loop with iterative tool calling.
- **Autonomous Tool Execution**: Amy dynamically decides when to query data, compute statistics, generate charts, or trigger cleaning routines based on your prompts.
- **Context-Aware Sessions**: Retains session conversation history and dataset context for natural follow-ups.

### 2. 📂 Multi-Format File Ingestion
- **Tabular Data**: Native high-speed parsing for `.csv` and Excel (`.xlsx`).
- **PowerBI Files**: Ingestion and schema exploration for `.pbix` archive files.
- **Text Logs & Documents**: File overview and content preview for `.txt` files.

### 3. 📊 Automated Exploratory Data Analysis (EDA)
- **Data Quality Scoring**: Real-time evaluation of dataset health and completeness.
- **Statistical Profiling**: Automated calculation of mean, median, standard deviation, quartiles, and skewness.
- **Anomaly & Outlier Detection**: Statistical boundary computation (IQR & Z-score methods).
- **Automated Quality Flags**: Highlights missing values, duplicate records, high-cardinality features, and constant columns.

### 4. 🧹 Intelligent Data Cleaning & Transformation
- **Automated Issue Detection**: Pinpoints data quality bottlenecks across all columns.
- **Configurable Imputation Strategies**: Median, mean, mode, linear interpolation, forward-fill, backward-fill, or drop missing.
- **Outlier Treatments**: Winsorization/capping, removal, median substitution, or retention.
- **Deduplication**: Intelligent duplicate identification and removal.
- **Snapshot & Rollback**: Safe state management with undo/redo capabilities to revert cleaning actions at any time.

### 5. 📈 Interactive & Programmatic Visualizations
- **Supported Chart Types**: Bar, Line, Pie, Scatter, Histogram, Box Plot, Heatmap, and Area charts.
- **Statistical Aggregations**: Group-by aggregations (`sum`, `mean`, `median`, `count`).
- **Dual Rendering**:
  - Server-side high-resolution rendering with **Matplotlib** and **Seaborn** (base64 PNG).
  - Client-side interactive rendering via **Chart.js**.
- **AI-Powered Suggestions**: Automatic chart recommendations tailored to column distributions and data types.

### 6. 💬 Conversational Data Analytics & Filtering
- Ask questions in plain English (e.g., *"Show me the distribution of revenue across regions and plot a bar chart"*).
- Executes safe pandas filtering, groupby operations, and slice sampling behind the scenes.
- Formats tabular results directly inside the chat interface.

### 7. 📄 Exporting & Reporting
- **Cleaned Data Export**: Download processed datasets as clean `.csv` files.
- **Comprehensive Markdown Reports**: Generate complete, structured EDA and Data Quality audit reports.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    User([User / Browser]) <--> UI[Modern Dark-Mode Web UI]
    UI <--> API[FastAPI Backend / REST API]

    subgraph Backend Services
        API --> UploadRouter[Files Router / Parsers]
        API --> EDARouter[EDA & Quality Engine]
        API --> CleanRouter[Data Cleaning Service]
        API --> ChartRouter[Visualization Engine]
        API --> AgentRouter[AI Agent & Chat Router]
        API --> ExportRouter[Export Engine]
    end

    subgraph Agent Loop
        AgentRouter <--> Ollama[Ollama Local LLM\n(e.g., Qwen2.5 / Llama 3)]
        AgentRouter --> ToolRegistry{Agent Tool Registry}
        ToolRegistry -->|draw_chart| ChartRouter
        ToolRegistry -->|apply_cleaning| CleanRouter
        ToolRegistry -->|query_data / compute_metric| PandasEngine[Pandas Data Engine]
        ToolRegistry -->|describe_column / sample_rows| EDARouter
    end

    subgraph Storage & Session
        UploadRouter --> SessionManager[(Session State & Snapshots)]
        PandasEngine --> SessionManager
    end
```

---

## 🛠️ Built-in Agent Tools

Amy uses structured tool calling to interact with datasets:

| Tool Name | Description | Parameters |
| :--- | :--- | :--- |
| `draw_chart` | Generates a visualization and returns the rendered image | `chart_type`, `x_column`, `y_column`, `aggregation`, `title` |
| `apply_cleaning` | Executes cleaning strategies on the active dataset | `missing_strategy`, `outlier_strategy`, `duplicate_strategy` |
| `query_data` | Evaluates a Pandas query expression and returns matching records | `query_string` (e.g. `Age > 30 and Salary > 50000`) |
| `describe_column` | Computes in-depth summary statistics for a specific column | `column` |
| `get_sample_rows` | Retrieves representative sample rows from the dataset | `n` (number of rows) |
| `compute_metric` | Computes custom group-by aggregations | `group_by`, `value_column`, `aggregation` |
| `get_data_summary` | Returns a refreshed overview of dataset shape and schema | *None* |

---

## 🚀 Getting Started

### Prerequisites
1. **Python 3.10 or higher** installed.
2. **[Ollama](https://ollama.ai/)** installed and running locally.
3. Pull the default LLM model:
   ```bash
   ollama pull qwen2.5:7b
   ```
   *(You can also use other models like `llama3.1:8b`, `mistral`, or `deepseek-r1` via environment configuration).*

---

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/amy.git
   cd amy
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # On macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

### Running the Application

1. **Ensure Ollama is running:**
   ```bash
   ollama serve
   ```

2. **Start the FastAPI application:**
   ```bash
   uvicorn main:app --reload --port 8000
   ```

3. **Open the Web Interface:**
   Navigate to [http://localhost:8000](http://localhost:8000) in your web browser.

4. **API Documentation:**
   Explore the interactive Swagger UI at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## ⚙️ Configuration

You can customize Amy's behavior using environment variables or by modifying [`services/config.py`](services/config.py):

| Environment Variable | Default Value | Description |
| :--- | :--- | :--- |
| `AMY_LLM_MODEL` | `qwen2.5:7b` | Name of the Ollama model to use for chat and agent reasoning. |
| `AMY_OLLAMA_HOST` | `http://localhost:11434` | Host address of the local Ollama instance. |
| `AMY_MAX_TOOL_ITERATIONS` | `5` | Maximum number of continuous tool execution cycles per request. |
| `AMY_MAX_HISTORY_MESSAGES` | `20` | Max conversation history messages retained for LLM context. |
| `AMY_LLM_MAX_RETRIES` | `2` | Number of retry attempts if LLM JSON parsing encounters errors. |

---

## 📂 Project Structure

```text
amy/
├── main.py                  # FastAPI application entry point & static mount
├── models.py                # Pydantic data schemas for requests and responses
├── requirements.txt         # Project dependencies
├── routers/                 # Modular API Route Handlers
│   ├── files.py             # File upload, session creation, and file switching
│   ├── analysis.py          # EDA computation and data cleaning endpoints
│   ├── charts.py            # Chart generation and AI chart recommendations
│   ├── chat.py              # Conversational ReAct AI agent loop
│   └── export.py            # CSV dataset and Markdown report export
├── services/                # Core Business Logic & AI Engines
│   ├── agent.py             # Agent tool registry and tool execution handlers
│   ├── charts.py            # Matplotlib/Seaborn visualization engines
│   ├── cleaning.py          # Tabular data cleaning & transformation logic
│   ├── config.py            # Central configuration & hyperparameters
│   ├── eda.py               # Exploratory data analysis & statistical metrics
│   ├── llm.py               # Ollama client, structured prompt templates & retries
│   ├── parsers.py           # Ingestion parsers (.csv, .xlsx, .txt, .pbix)
│   └── session.py           # In-memory session tracking, caching & snapshot undo
├── static/                  # Modern Web Frontend
│   ├── index.html           # Main application dashboard layout
│   ├── style.css            # Custom CSS styling (dark theme, responsive)
│   └── app.js               # Frontend JavaScript client & Chart.js integration
└── uploads/                 # Temporary storage for uploaded session files
```

---

## 📡 API Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/upload` | Upload a data file (`.csv`, `.xlsx`, `.txt`, `.pbix`) and initiate a session. |
| `POST` | `/api/eda` | Retrieve comprehensive EDA profiling and quality scores. |
| `POST` | `/api/cleaning` | Fetch detected data anomalies and AI cleaning recommendations. |
| `POST` | `/api/cleaning/apply` | Apply selected cleaning strategies (with snapshot rollback support). |
| `POST` | `/api/cleaning/undo` | Revert to the previous dataset snapshot. |
| `POST` | `/api/chart` | Generate a chart (returns base64-encoded PNG). |
| `POST` | `/api/chart/data` | Retrieve structured chart data formatted for Chart.js. |
| `POST` | `/api/chart-suggestions` | Get AI and heuristic-recommended charts for the active dataset. |
| `POST` | `/api/chat` | Send a prompt to the AI Agent (triggers ReAct tool execution loop). |
| `GET` | `/api/export/csv` | Download the active (cleaned) dataset as a CSV file. |
| `GET` | `/api/export/report` | Download the full EDA summary report in Markdown format. |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
