export interface MarketTick {
  symbol: string;
  ltp: number;
  timestamp: string;
  volume: number;
  source: string;
}

export interface CandleEngineSummary {
  m1: number;
  m3: number;
  m5: number;
  m15: number;
  lastCompleted: string;
}

export interface DataQuality {
  state: 'LIVE' | 'DELAYED' | 'STALE' | 'MARKET_CLOSED' | 'DEMO' | 'ERROR';
  missingTicks: number;
  connectionErrors: number;
  validationErrors: number;
}

export interface OptionContract {
  strike: number;
  type: 'CE' | 'PE';
  marketPrice: number;
  modelPrice: number;
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  ivPct: number;
  popPct: number;
  moneyness: 'ITM' | 'ATM' | 'OTM';
}

export interface TradeAdvice {
  action: 'BUY_CALL' | 'BUY_PUT' | 'NO_TRADE';
  confidence: number;
  entryTrigger?: number;
  invalidationLevel?: number;
  firstObjective?: number;
  reasoning: string[];
  warnings: string[];
  recommendedContract?: OptionContract;
}
