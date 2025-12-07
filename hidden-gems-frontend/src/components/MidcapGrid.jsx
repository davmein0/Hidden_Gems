import React, { useEffect, useState } from "react";
import axios from "axios";

export default function MidcapGrid({ setSelectedTicker, setAnalysis }) {
  const [companies, setCompanies] = useState([]);
  const [filtered, setFiltered] = useState([]);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadMidcaps = async () => {
      try {
        const res = await axios.get("http://localhost:8000/midcaps/");
        setCompanies(res.data);
        setFiltered(res.data);
      } catch (err) {
        console.error("ERROR loading midcaps:", err);
        setCompanies([]);
        setFiltered([]);
      } finally {
        setLoading(false);
      }
    };
    loadMidcaps();
  }, []);
 
  const handleSelect = async (ticker) => {
    setSelectedTicker(ticker);

    try {
      const featRes = await axios.get(`http://localhost:8000/features/${ticker}`);
      const features = featRes.data;

      const predRes = await axios.post("http://localhost:8000/predict/", {
        ticker,
        ...featRes.data,
      });

      // ❌ Hide analysis if prediction invalid
      if (
        predRes.data.error ||
        predRes.data.undervalued_probability === undefined ||
        isNaN(predRes.data.undervalued_probability)
      ) {
        setAnalysis(null);
        return;
      }

      const sentimentRes = await axios.get(
        `http://localhost:8000/sentiment/${ticker}`
      );
      const sentiment = sentimentRes.data;

      setAnalysis({ ...predRes.data, sentiment });
    } catch (err) {
      console.error("Prediction failed:", err);
      setAnalysis(null); // hide card, do NOT show "Prediction failed"
    }
  };

  if (loading) return <div>Loading midcaps...</div>;

  return (
    <div className="midcap-grid-container">
      <div className="midcap-search-bar-wrapper">
        <input
          type="text"
          className="midcap-search-input"
          placeholder="Search ticker or company name..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      <div className="midcap-grid">
        {filtered.map((stock) => (
          <div
            key={stock.Ticker}
            className="company-tile"
            onClick={() => handleSelect(stock.Ticker)}
          >
            <strong>{stock.Ticker}</strong>
            <div className="company-tile-name">{stock.Name}</div>
          </div>
        ))}

        {filtered.length === 0 && (
          <div className="no-results">No matching tickers or companies</div>
        )}
      </div>
    </div>
  );
}
