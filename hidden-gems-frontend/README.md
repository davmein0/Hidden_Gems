# Hidden Gems Frontend

The active dashboard for [Hidden Gems](../README.md): a React 19 + Vite + Tailwind app for browsing mid-cap NASDAQ stocks, running undervaluation predictions, and reading the generated analysis.

> The top-level `frontend/` directory is a deprecated prototype. All frontend work happens here.

## Requirements

- Node.js 18 or newer
- The Hidden Gems backend running at `http://localhost:8000` (see the [root README](../README.md#backend-setup))

## Install

```bash
npm install
```

## Run the dev server

```bash
npm run dev
```

Vite serves the app at `http://localhost:5173` with hot module replacement.

## Build and preview

```bash
npm run build     # production build into dist/
npm run preview   # serve the built output locally
npm run lint      # eslint
```

## Backend connection

The app talks to the FastAPI backend with `axios`, using the hard-coded base URL `http://localhost:8000`. The backend enables permissive CORS, so no proxy configuration is needed for local development. Endpoints used:

| Endpoint                   | Used by                            |
| -------------------------- | ---------------------------------- |
| `GET /midcaps/`            | `components/MidcapGrid.jsx`        |
| `GET /watchlist/`          | `components/Watchlist.jsx`         |
| `GET /features/{ticker}`   | `MidcapGrid.jsx`, `Watchlist.jsx`  |
| `POST /predict/`           | `MidcapGrid.jsx`, `Watchlist.jsx`  |
| `GET /sentiment/{ticker}`  | `components/MidcapGrid.jsx`        |

If the backend runs elsewhere, update the URLs in `src/components/`.

## Structure

```
src/
  App.jsx                 app shell
  main.jsx                entrypoint
  pages/Dashboard.jsx     composes the dashboard
  components/
    MidcapGrid.jsx        mid-cap universe grid with predictions
    Watchlist.jsx         saved tickers
    AnalysisPanel.jsx     detail view for the selected ticker
  styles/, assets/
```
