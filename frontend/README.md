# IndianIntradayAI — Angular Trading Terminal

Real-time, reactive Angular frontend for IndianIntradayAI. Built with Angular 19, RxJS BehaviorSubjects, and reactive streaming.

## Features
- **Zero Full-Page Reloads:** Pure single-page application (SPA) architecture with RxJS streaming.
- **Real-Time Flashing Prices:** Green on upticks, red on downticks.
- **Live Option Greeks:** Delta, Gamma, Theta, Vega, IV, and POP display.
- **Autonomous Trade Signals:** BUY CALL / BUY PUT / NO TRADE with Entry, SL, and Target.

## Quick Start

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Run Locally
```bash
npm start
```
Navigate to `http://localhost:4200/`. The application will automatically stream real-time price updates without refreshing.

### 3. Production Build
```bash
npm run build
```
Build artifacts will be stored in `dist/indian-intraday-ai/`.
