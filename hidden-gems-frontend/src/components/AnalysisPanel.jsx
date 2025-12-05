export default function AnalysisPanel({ selectedTicker, analysis }) {

  // No ticker selected yet → show nothing
  if (!selectedTicker) return null;

  // While loading → optionally show nothing OR a spinner
  if (analysis === undefined) return null; 

  // Prediction failed → render nothing at all
  if (analysis === null || analysis?.error) return null;

  // SUCCESS → render the full card
  return (
    <div>
      <h2>{analysis.ticker}</h2>

      <p><strong>Probability:</strong> {(analysis.undervalued_probability * 100).toFixed(2)}%</p>
      <p><strong>Category:</strong> {analysis.confidence_category}</p>
      <p><strong>Recommendation:</strong> {analysis.recommendation}</p> 

      <h3 className="fin-info">Financial Info</h3>
      <div className="analysis-features">
        <p><strong>Market Cap:</strong> {analysis.features.MarketCap}</p>
        <p><strong>EV/EBITDA:</strong> {analysis.features.EV_EBITDA}</p>
        <p><strong>FCF Yield:</strong> {analysis.features.FCF_Yield}</p>
        <p><strong>PB Ratio:</strong> {analysis.features.PB_Ratio}</p>
      </div>
    </div>
  );
}
