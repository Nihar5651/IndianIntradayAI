import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, interval } from 'rxjs';
import { MarketTick, CandleEngineSummary, DataQuality, OptionContract, TradeAdvice } from '../models/market-data.model';

@Injectable({
  providedIn: 'root'
})
export class MarketDataService {
  private tickSubject = new BehaviorSubject<MarketTick>({
    symbol: 'NIFTY',
    ltp: 25030.35,
    timestamp: new Date().toLocaleTimeString('en-IN', { hour12: false }),
    volume: 1245000,
    source: 'SMARTAPI'
  });

  private qualitySubject = new BehaviorSubject<DataQuality>({
    state: 'MARKET_CLOSED',
    missingTicks: 0,
    connectionErrors: 0,
    validationErrors: 0
  });

  private candleSubject = new BehaviorSubject<CandleEngineSummary>({
    m1: 0,
    m3: 0,
    m5: 0,
    m15: 0,
    lastCompleted: '15:30:00'
  });

  private tradeAdviceSubject = new BehaviorSubject<TradeAdvice>({
    action: 'BUY_CALL',
    confidence: 85,
    entryTrigger: 25030.35,
    invalidationLevel: 24980.35,
    firstObjective: 25200.00,
    reasoning: [
      'Put OI materially exceeds Call OI (PCR 1.16)',
      'Largest PE OI support established near 24,950',
      'Strong gamma acceleration in ATM 25,050 CE contract'
    ],
    warnings: ['High IV rank requires trailing stop-loss discipline.'],
    recommendedContract: {
      strike: 25050,
      type: 'CE',
      marketPrice: 110.50,
      modelPrice: 108.20,
      delta: 0.48,
      gamma: 0.00032,
      theta: -12.40,
      vega: 18.50,
      ivPct: 14.2,
      popPct: 48.5,
      moneyness: 'ATM'
    }
  });

  public tick$: Observable<MarketTick> = this.tickSubject.asObservable();
  public quality$: Observable<DataQuality> = this.qualitySubject.asObservable();
  public candles$: Observable<CandleEngineSummary> = this.candleSubject.asObservable();
  public tradeAdvice$: Observable<TradeAdvice> = this.tradeAdviceSubject.asObservable();

  constructor() {
    this.startLiveStreamSimulation();
  }

  private startLiveStreamSimulation() {
    interval(1500).subscribe(() => {
      const current = this.tickSubject.value;
      const change = (Math.random() - 0.49) * 4;
      const newLtp = +(current.ltp + change).toFixed(2);

      this.tickSubject.next({
        ...current,
        ltp: newLtp,
        timestamp: new Date().toLocaleTimeString('en-IN', { hour12: false })
      });
    });
  }
}
