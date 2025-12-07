export default function AnalysisPanel({ selectedTicker, analysis }) {
  if (!selectedTicker) return <div>Select a stock</div>;
  if (!analysis) return <div>Loading...</div>;
  if (analysis.error) return <div>Error: {analysis.error}</div>;

  return (
    <div>
      <h2>{analysis.ticker}</h2>

      <p>
        <strong>Probability:</strong>{" "}
        {(analysis.undervalued_probability * 100).toFixed(2)}%
      </p>
      <p>
        <strong>Category:</strong> {analysis.confidence_category}
      </p>
      <p>
        <strong>Recommendation:</strong> {analysis.recommendation}
      </p>
      <p>
        <strong>Model Version:</strong> {analysis.model_version}
      </p>

      <h3>Features</h3>
      <pre>{JSON.stringify(analysis.features, null, 2)}</pre>
      <h3>Sentiment Analysis</h3>
      <pre>{JSON.stringify(analysis.sentiment, null, 2)}</pre>
    </div>
  );
}
