import React from 'react';

export default function PredictionResults({ result, error, isLoading }) {
  if (isLoading) {
    return (
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">
            <span className="icon">🎯</span>
            Predicted Ablation Zone
          </h2>
        </div>
        <div className="results-grid">
          <div className="result-metric border-default">
            <div className="skeleton" style={{ height: '50px', marginBottom: '10px' }}></div>
            <div className="skeleton" style={{ height: '20px', width: '50%', margin: '0 auto' }}></div>
          </div>
          <div className="result-metric border-default">
            <div className="skeleton" style={{ height: '50px', marginBottom: '10px' }}></div>
            <div className="skeleton" style={{ height: '20px', width: '50%', margin: '0 auto' }}></div>
          </div>
          <div className="result-metric border-default">
            <div className="skeleton" style={{ height: '50px', marginBottom: '10px' }}></div>
            <div className="skeleton" style={{ height: '20px', width: '50%', margin: '0 auto' }}></div>
          </div>
          <div className="result-metric border-default">
            <div className="skeleton" style={{ height: '50px', marginBottom: '10px' }}></div>
            <div className="skeleton" style={{ height: '20px', width: '50%', margin: '0 auto' }}></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card" style={{ border: '1px solid var(--accent-danger)' }}>
        <div className="card-header" style={{ borderBottomColor: 'var(--accent-danger-dim)' }}>
          <h2 className="card-title" style={{ color: 'var(--accent-danger)' }}>
            <span className="icon">⚠️</span>
            Prediction Error
          </h2>
        </div>
        <p style={{ color: 'var(--accent-danger)' }}>{error}</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="card empty-state">
        <div className="icon">🧬</div>
        <div className="title">Awaiting Parameters</div>
        <div className="description">
          Enter treatment power, time, and antenna type to generate an ML-driven ablation prediction.
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">
          <span className="icon">🎯</span>
          Predicted Ablation Zone
        </h2>
      </div>

      <div className="results-grid">
        <div className="result-metric highlight">
          <div className="label">Focus Diameter</div>
          <div className="value">
            {result.prediction.diameter_mm.toFixed(1)}
            <span className="unit"> mm</span>
          </div>
        </div>

        <div className="result-metric">
          <div className="label">Total Length</div>
          <div className="value">
            {result.prediction.length_mm.toFixed(1)}
            <span className="unit"> mm</span>
          </div>
        </div>

        <div className="result-metric">
          <div className="label">Est. Volume</div>
          <div className="value">
            {(result.prediction.estimated_volume_mm3 / 1000).toFixed(2)}
            <span className="unit"> cc</span>
          </div>
        </div>

        <div className="result-metric">
          <div className="label">Sphericity</div>
          <div className="value">
            {result.prediction.sphericity.toFixed(2)}
          </div>
        </div>
      </div>
    </div>
  );
}
