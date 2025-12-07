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
      <p><strong>Model Version:</strong> {analysis.model_version}</p>

      <h3>Features</h3>
      <pre>{JSON.stringify(analysis.features, null, 2)}</pre>
    </div>
  );
}
