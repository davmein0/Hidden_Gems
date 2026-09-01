# Hidden Gems

Hidden Gems is an AI-powered research tool that looks for undervalued mid-cap stocks listed on the NASDAQ exchange (roughly $2B–$10B market cap). It combines a pretrained XGBoost classifier over fundamental metrics, FinBERT sentiment scoring of recent news, and a multi-agent LLM system that produces an explainable written analysis for a ticker.

The system ships as a FastAPI backend that serves predictions, features, sentiment, and search over a mid-cap universe, plus a React dashboard where users browse mid-cap stocks, run a valuation prediction, and read the generated analysis.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Backend API](#backend-api)
  - [Multi-agent analysis](#multi-agent-analysis)
  - [Models](#models)
  - [Data pipeline](#data-pipeline)
  - [Frontend](#frontend)
  - [Active vs. legacy components](#active-vs-legacy-components)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend setup](#backend-setup)
  - [Frontend setup](#frontend-setup)
  - [Environment variables](#environment-variables)
  - [Scraping and training](#scraping-and-training)
- [Data Layout](#data-layout)
- [Model Details](#model-details)

## Overview

For a given ticker, Hidden Gems:

1. Loads fundamental features (P/E, P/B, P/S, EV/EBITDA, ROE, FCF yield, quick ratio, market cap) from the scraped mid-cap dataset.
2. Scores the ticker with the pretrained XGBoost classifier to estimate how likely it is to be undervalued, along with a confidence category.
3. Computes FinBERT sentiment over recent news headlines for the ticker.
4. Optionally runs a multi-agent LLM workflow (industry, management, product, and financials agents coordinated by a supervisor) to produce a structured written report.

## Architecture

### Backend API

The FastAPI application lives in `backend/main.py` and mounts these routers from `backend/routers/`:

| Router      | Prefix       | Purpose                                                     |
| ----------- | ------------ | ----------------------------------------------------------- |
| `predict`   | `/predict`   | `POST` a ticker (plus optional feature overrides) and get an undervaluation probability, confidence category, and sentiment. |
| `midcaps`   | `/midcaps`   | List the mid-cap universe used by the dashboard.             |
| `search`    | `/search`    | Search the mid-cap universe by ticker or company name.       |
| `watchlist` | `/watchlist` | Read the saved watchlist of tickers.                         |
| `features`  | `/features`  | Return the raw fundamental features for a ticker.            |
| `sentiment` | `/sentiment` | Return FinBERT news sentiment for a ticker.                  |

Supporting code sits in `backend/services/` (model, feature, and sentiment loaders), `backend/models/` (Pydantic response schemas), and `backend/core/config.py` (paths to the model artifacts and mid-cap data).

Interactive API docs are available at `http://localhost:8000/docs` once the server is running.

### Multi-agent analysis

`backend/agents.py` defines a LlamaIndex ReAct multi-agent system on top of OpenAI models, with a SerpAPI web-search tool. Four specialist agents (industry, management, product, financials) are exposed as tools to a supervisor agent, which coordinates them into a single structured report for a ticker.

### Models

- **XGBoost classifier** — serialized at `xgboost_model.pkl` with its feature list and version in `model_config.json`. See [Model Details](#model-details).
- **FinBERT** (`ProsusAI/finbert`) — loaded through `transformers` in `backend/services/sentiment_loader.py` and applied to recent news headlines.

### Data pipeline

`src/hidden_gems/scrape/pipeline.py` builds the dataset: it fetches the NASDAQ ticker list, filters to mid-caps by historical market cap, pulls fundamentals from Yahoo Finance, pulls 10-K filings via the SEC API, and merges everything into a single dataset. Training code lives in `src/hidden_gems/ml/train.py`. Both are driven by the CLI wrappers in `scripts/` (`run_scrape.py`, `run_train.py`, `generate_labels.py`).

### Frontend

`hidden-gems-frontend/` is a React 19 + Vite + Tailwind dashboard. `src/pages/Dashboard.jsx` composes the `MidcapGrid`, `Watchlist`, and `AnalysisPanel` components, which call the backend with `axios` at `http://localhost:8000`.

### Active vs. legacy components

| Path                        | Status                                                            |
| --------------------------- | ----------------------------------------------------------------- |
| `hidden-gems-frontend/`     | **Active** frontend.                                               |
| `backend/`                  | **Active** FastAPI backend.                                        |
| `frontend/`                 | Legacy React prototype, kept for reference only. Do not build on it. |
| `train_and_save_xgboost.py` | Legacy Flask/training prototype, superseded by `src/hidden_gems/ml/train.py` and `scripts/run_train.py`. |
| `midcap_scrape.py`, `sentiment_scrape.py`, `finBert_news.py` | Legacy top-level scripts, superseded by `src/hidden_gems/scrape/pipeline.py`. |

## Getting Started

### Prerequisites

- Python 3.11 or newer
- Node.js `^20.19.0` or `>=22.12.0` (required by the pinned Vite 7 toolchain)

### Backend setup

From the repository root:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.\.venv\Scripts\activate
```

Install dependencies and the local package:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Run the API server from the repository root (not from `backend/`, so that the `backend.*` imports resolve):

```bash
uvicorn backend.main:app --reload --port 8000
```

The API is then available at `http://localhost:8000`.

### Frontend setup

In a second terminal:

```bash
cd hidden-gems-frontend
npm install
npm run dev
```

Vite serves the dashboard at `http://localhost:5173` and expects the backend at `http://localhost:8000`.

### Environment variables

| Variable                            | Read by                                                            | Source file the code reads |
| ----------------------------------- | ------------------------------------------------------------------ | -------------------------- |
| `SEC_API_KEY`                       | 10-K filing fetches in `src/hidden_gems/scrape/pipeline.py`          | process environment        |
| `MIN_MARKET_CAP` / `MAX_MARKET_CAP` | Mid-cap universe bounds (defaults: $2B–$10B)                        | process environment        |
| `YF_SLEEP`                          | Throttle between Yahoo Finance requests (default `0.22`)            | process environment        |
| `NEWSAPI_KEY`                       | News fetch for FinBERT sentiment (`backend/services/sentiment_loader.py`) | process environment  |
| `OPENAI_KEY`                        | Multi-agent analysis in `backend/agents.py`                          | `config.env`               |
| `SERP_API_KEY`                      | Web search tool used by the agents                                   | `config.env`               |

Nothing in the codebase auto-loads a root `.env`, so copying `.env.sample` on its own has no effect. Export the pipeline/backend variables into your shell before running anything:

```bash
cp .env.sample .env
set -a && source .env && set +a
```

The agents in `backend/agents.py` call `load_dotenv('config.env')`, so `OPENAI_KEY` and `SERP_API_KEY` must live in a `config.env` file in the directory you launch the process from (or be exported the same way).

Never commit your `.env` or `config.env` file.

### Scraping and training

```bash
# Build a fresh dataset for a given as-of date (writes under data/raw and data/interim)
python -m scripts.run_scrape --date 2025-11-28 --limit 100

# Label a merged dataset and train on it (writes models/xgb_undervalued.pkl)
python -m scripts.run_train --dataset data/processed/labeled_from_merged.csv
```

With no `--dataset`, `scripts/run_train.py` falls back to the tiny `data/example.csv` sample, which is only useful as a smoke test.

**Retraining does not update the served model.** The API loads `xgboost_model.pkl` and `model_config.json` from the repo root (see `backend/core/config.py`), while training writes `models/xgb_undervalued.pkl`. To serve a newly trained model, copy it to the repo root as `xgboost_model.pkl` and update `model_config.json` so `feature_columns` and `model_version` match the model you trained.

For other developer tasks, see `pyproject.toml` and `scripts/`.

## Data Layout

The project stores datasets and artifacts under a unified `data/` folder at the repo root:

- `data/raw/` — raw CSVs, intermediate outputs from scrapers (e.g., `midcaps_YYYY-MM-DD.csv`, `financials_YYYY-MM-DD.csv`, `filings_YYYY-MM-DD.csv`).
- `data/interim/` — merged/intermediate datasets generated by the pipeline (e.g., `merged_dataset_YYYY-MM-DD.csv`).
- `data/processed/` — final processed datasets for ML or the UI (e.g., `merged_combined.csv`, `labeled_from_merged.csv`).
- `models/` — serialized models saved by training, e.g. `xgb_undervalued.pkl`.

Files are written by the pipeline using relative `data/` paths (via `hidden_gems.io` helpers). Use the `scripts/` wrappers or the package entrypoints to run operations and avoid writing to the repo root inadvertently.

Examples:

```bash
# Scrape using data/raw as destination
python scripts/run_scrape.py --date 2025-11-28 --limit 100

# Use the pre-existing CSV under data/processed for training
python scripts/run_train.py --dataset data/processed/labeled_from_merged.csv --model-path models/xgb_undervalued.pkl
```

## Model Details

### Overview and limitations

The XGBoost classification model is the **fundamental analysis foundation** of the prediction system. It analyzes 10 key financial metrics (P/E, P/B, ROE, FCF yield, etc.) to identify potentially undervalued mid-cap stocks.

**Important**: this model achieves **50% precision** on its own, which may seem low but is reasonable for fundamental-only stock prediction. Stock prices are driven by many factors beyond fundamentals — sentiment, momentum, news catalysts, and market psychology all play major roles that financial ratios cannot capture. Professional quant funds with far more resources typically achieve 55–65% precision on similar tasks.

### Current performance

- **ROC AUC**: 0.610 (meaningfully better than random)
- **Training data**: 414 ticker-year combinations (2021–2024)
- **Calibration**: high-confidence picks (>0.7 probability) have a 50% success rate vs. 26% for low-confidence picks

The model excels at **ranking** stocks by fundamental quality but struggles to predict actual price movements on its own.

### Why this is a foundation, not the final product

The model is designed to work alongside **FinBERT sentiment analysis**:

- **XGBoost**: identifies fundamentally cheap, quality stocks.
- **FinBERT**: detects positive sentiment and catalysts.
- **Combined**: filters out "value traps" (cheap for a reason) and finds true opportunities.

By combining fundamental scores with sentiment analysis, we expect to push precision from 50% to **60–70%**, similar to professional quantitative strategies.

### Key takeaway

Think of the classifier as the "quality filter" that identifies financially sound, undervalued companies. The sentiment layer then helps determine _when_ the market is ready to recognize that value.
