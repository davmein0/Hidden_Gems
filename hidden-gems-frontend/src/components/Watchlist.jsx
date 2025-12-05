import React, { useEffect, useState } from "react";
import axios from "axios";

export default function Watchlist({ setSelectedTicker, setAnalysis }) {
  const [tickers, setTickers] = useState([]);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await axios.get("http://localhost:8000/watchlist/");

        // ✅ FILTER: only keep valid items
        const valid = res.data.filter(
          (s) =>
            s &&
            typeof s.probability === "number" &&
            !Number.isNaN(s.probability)
        );

        setTickers(valid);
      } catch (err) {
        console.error(err);
      }
    };
    load();
  }, []);

  const clickTicker = async (ticker) => {
    setSelectedTicker(ticker);

    try {
      const feat = await axios.get(`http://localhost:8000/features/${ticker}`);
      const pred = await axios.post("http://localhost:8000/predict/", {
        ticker,
        ...feat.data,
      });

      // ❌ If prediction failed → completely hide analysis
      if (pred.data.error || isNaN(pred.data.undervalued_probability)) {
        setAnalysis(null);
        return;
      }

      setAnalysis(pred.data);
    } catch {
      setAnalysis(null); // Hide card
    }
  };

  return (
    <div>
      <div className="watchlist-title">Top Undervalued (Model Ranking)</div>

      {tickers.map((s) => (
        <div
          key={s.ticker}
          className="watchlist-item"
          onClick={() => clickTicker(s.ticker)}
        >
          <div className="watchlist-ticker">{s.ticker}</div>
          <div className="watchlist-prob">
            Prob: {(s.probability * 100).toFixed(1)}%
          </div>
        </div>
      ))}
    </div>
  );
}
