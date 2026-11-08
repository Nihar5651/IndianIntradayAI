import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MarketDataService } from './services/market-data.service';
import { MarketTick, CandleEngineSummary, DataQuality, TradeAdvice } from './models/market-data.model';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
  public tick: MarketTick | null = null;
  public previousLtp: number = 0;
  public priceDirection: 'up' | 'down' | 'same' = 'same';
  public quality: DataQuality | null = null;
  public candles: CandleEngineSummary | null = null;
  public advice: TradeAdvice | null = null;

  constructor(private marketService: MarketDataService) {}

  ngOnInit(): void {
    this.marketService.tick$.subscribe((tick) => {
      if (this.tick) {
        if (tick.ltp > this.tick.ltp) {
          this.priceDirection = 'up';
        } else if (tick.ltp < this.tick.ltp) {
          this.priceDirection = 'down';
        }
      }
      this.tick = tick;
    });

    this.marketService.quality$.subscribe((q) => this.quality = q);
    this.marketService.candles$.subscribe((c) => this.candles = c);
    this.marketService.tradeAdvice$.subscribe((a) => this.advice = a);
  }
}
