import React, { useEffect, useState } from "react";
import axios from "axios";

export default function MidcapGrid({ setSelectedTicker, setAnalysis }) {
  const [companies, setCompanies] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadMidcaps = async () => {
      try {
        const res = await axios.get("http://localhost:8000/midcaps/");
        setCompanies(res.data);
      } catch (err) {
        console.error("ERROR loading midcaps:", err);
        setCompanies([]);
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
        ...features,
      });

      setAnalysis(predRes.data);
    } catch (err) {
      console.error("Prediction failed:", err);
      setAnalysis({ error: "Prediction failed" });
    }
  };

  if (loading) return <div>Loading midcaps...</div>;

  return (
    <div className="midcap-grid">
      {companies.map((stock) => (
        <div
          key={stock.Ticker}
          className="company-tile"
          onClick={() => handleSelect(stock.Ticker)}
        >
          <strong>{stock.Ticker}</strong>
          <div className="company-tile-name">{stock.Name}</div>
        </div>
      ))}
    </div>
  );
}
