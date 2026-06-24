# Turbofan Fault Monitor — Frontend

React + Vite dashboard for the NASA CMAPSS turbofan fault classifier.

## Features

- **3D engine viewer** — interactive Three.js turbofan model with fault-zone highlighting (HPC, HPT, Combustor)
- **Demo samples** — click any fault class (C0–C3) to load a synthetic sensor window and see the prediction
- **CSV upload** — upload a real or synthetic 30-cycle sensor window for live inference via the Flask API
- **Probability bars** — per-class confidence breakdown after every prediction
- **API status badge** — live health-check against the Flask backend

## Project structure

```
src/
  TurbofanFaultDashboard.jsx   # main dashboard (3D viewer + upload panel)
  App.jsx                      # root component
  main.jsx                     # Vite entry point
  App.css / index.css          # global styles
```

## Getting started

```bash
# Install dependencies
npm install

# Start the dev server (requires the Flask backend on :5000)
npm run dev
```

The app will be available at `http://localhost:5173`.

## Configuration

The API base URL is read from a Vite environment variable so it never needs
to be hard-coded in source.

| Variable        | Default                       | Description               |
|-----------------|-------------------------------|---------------------------|
| `VITE_API_BASE` | `http://localhost:5000/api`   | Flask backend API root    |

**Local dev** — the default in `.env` points to `localhost:5000` and works
out of the box.

**Staging / production** — copy `.env.example` to `.env.local` and set
`VITE_API_BASE` to your deployed backend URL:

```bash
cp .env.example .env.local
# edit .env.local
VITE_API_BASE=https://your-backend.example.com/api
```

`.env.local` is git-ignored and will override `.env` automatically.

## Building for production

```bash
npm run build   # outputs to dist/
npm run preview # preview the production build locally
```

## Backend dependency

The dashboard calls:

| Endpoint                  | Used for                              |
|---------------------------|---------------------------------------|
| `GET  /api/health`        | API status badge                      |
| `GET  /api/sample/<id>`   | Demo sample buttons (C0–C3)           |
| `POST /api/predict`       | CSV upload inference                  |
| `GET  /api/classes`       | Feature-column metadata               |

See `turbofan_app/backend/app.py` for the Flask API source.

## CSV upload format

The uploaded file must contain at minimum these 16 columns (extra columns
are ignored):

```
op1, op2, s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21
```

Any number of rows is accepted; the backend uses the last 30 rows as the
inference window (zero-padded if fewer than 30 rows are provided).
