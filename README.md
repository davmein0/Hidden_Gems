# Hidden_Gems

An AI model that finds undervalued companies in the NASDAQ 500. We can determine key metrics for success for different industries (like stock data, industry tailwinds, market conditions, and news/political context), and compare that with current ratings or indicators of success, like stock prices. If time permits, we could create a dashboard web app, where users can input companies of their choice to determine their evaluation.

<!-- TABLE OF CONTENTS (outline)-->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## XGBoost Fundamental Analysis Model

### Overview & Limitations

This XGBoost classification model serves as the **fundamental analysis foundation** for our stock prediction system. It analyzes 8 key financial metrics (P/E, P/B, ROE, FCF Yield, etc.) to identify potentially undervalued mid-cap stocks.

**Important**: This model achieves **45% precision** on its own, which may seem low but is actually reasonable for fundamental-only stock prediction. Stock prices are driven by many factors beyond fundamentals—sentiment, momentum, news catalysts, and market psychology all play major roles that financial ratios cannot capture. Professional quant funds with far more resources typically achieve 55-65% precision on similar tasks.

### Current Performance

- **ROC AUC**: 0.627 (meaningfully better than random)
- **Training Data**: 505 ticker-year combinations (2020-2024)
- **Calibration**: High-confidence picks (>0.7 probability) have a 47% success rate vs. 21% for low-confidence picks

The model excels at **ranking** stocks by fundamental quality but struggles to predict actual price movements on its own.

### Why This is a Foundation, Not the Final Product

This model is designed to work alongside **FinBERT sentiment analysis**:

- **XGBoost**: Identifies fundamentally cheap, quality stocks
- **FinBERT** (to be added): Detects positive sentiment and catalysts
- **Combined**: Filters out "value traps" (cheap for a reason) and finds true opportunities

By combining fundamental scores with sentiment analysis, we expect to push precision from 45% to **60-70%**, similar to professional quantitative strategies.

### Key Takeaway

Think of this as the "quality filter" that identifies financially sound, undervalued companies. The sentiment layer will then help us determine *when* the market is ready to recognize that value.
