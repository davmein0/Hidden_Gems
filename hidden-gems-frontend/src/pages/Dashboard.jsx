import React, { useState } from "react";
import "../styles/dashboard.css";

import Watchlist from "../components/Watchlist";
import MidcapGrid from "../components/MidcapGrid";
import AnalysisPanel from "../components/AnalysisPanel";

export default function Dashboard() {
  const [selectedTicker, setSelectedTicker] = useState(null);
  const [analysis, setAnalysis] = useState(null);

  return (
    <div className="dashboard-container">

      <div className="watchlist-panel">
        <Watchlist setSelectedTicker={setSelectedTicker} setAnalysis={setAnalysis} />
      </div>

      {/* FIX: Remove wrapper div */}
      <MidcapGrid
        setSelectedTicker={setSelectedTicker}
        setAnalysis={setAnalysis}
      />

      <div className="analysis-panel">
        <AnalysisPanel selectedTicker={selectedTicker} analysis={analysis} />
      </div>

    </div>
  );
}
