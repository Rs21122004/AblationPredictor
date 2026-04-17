import React from 'react';

export default function ModelInfo({ modelData, isLoading }) {
  if (isLoading || !modelData) return null;

  const { diameter_model, length_model } = modelData;

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">
          <span className="icon">🔬</span>
          ML Ensemble Architecture
        </h2>
      </div>

      <div className="form-row">
        {/* Diameter Model */}
        <div>
          <div style={{ marginBottom: '12px' }}>
            <span className="model-badge">Diameter Predictor</span>
          </div>
          <div className="model-info-row">
            <span className="model-info-label">Model</span>
            <span className="model-info-value">{diameter_model.name}</span>
          </div>
          <div className="model-info-row">
            <span className="model-info-label">Accuracy (R²)</span>
            <span className="model-info-value">{(diameter_model.test_r2 * 100).toFixed(1)}%</span>
          </div>
          <div className="model-info-row">
            <span className="model-info-label">Typ. Error (MAE)</span>
            <span className="model-info-value">{diameter_model.test_mae.toFixed(2)} mm</span>
          </div>
        </div>

        {/* Length Model */}
        <div>
          <div style={{ marginBottom: '12px' }}>
            <span className="model-badge">Length Predictor</span>
          </div>
          <div className="model-info-row">
            <span className="model-info-label">Model</span>
            <span className="model-info-value">{length_model.name}</span>
          </div>
          <div className="model-info-row">
            <span className="model-info-label">Accuracy (R²)</span>
            <span className="model-info-value">{(length_model.test_r2 * 100).toFixed(1)}%</span>
          </div>
          <div className="model-info-row">
            <span className="model-info-label">Typ. Error (MAE)</span>
            <span className="model-info-value">{length_model.test_mae.toFixed(2)} mm</span>
          </div>
        </div>
      </div>
    </div>
  );
}
