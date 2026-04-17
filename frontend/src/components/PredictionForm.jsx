import React, { useState } from 'react';

const ANTENNA_TYPES = [
  "Coaxial Half-Slot", "Cooled Antenna", "Dipole", "Directional",
  "Dual Slot", "Floating Sleeve", "Helical Dipole", "MRSA",
  "Monopole", "Multi Slot/Triple", "Other", "Single Slot",
  "Sliding Choke", "Slot Antenna", "Triaxial"
];

export default function PredictionForm({ onSubmit, isLoading }) {
  const [power, setPower] = useState(50);
  const [time, setTime] = useState(5);
  const [antennaType, setAntennaType] = useState('Other');

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      power: parseFloat(power),
      time: parseFloat(time),
      antenna_type: antennaType
    });
  };

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">
          <span className="icon">⚡</span>
          Treatment Parameters
        </h2>
      </div>
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label className="form-label">
            Power <span className="unit">(Watts)</span>
          </label>
          <input
            type="number"
            className="form-input"
            value={power}
            onChange={(e) => setPower(e.target.value)}
            min="1"
            max="200"
            step="0.1"
            required
            disabled={isLoading}
          />
        </div>

        <div className="form-group">
          <label className="form-label">
            Treatment Time <span className="unit">(Minutes)</span>
          </label>
          <input
            type="number"
            className="form-input"
            value={time}
            onChange={(e) => setTime(e.target.value)}
            min="0.1"
            max="60"
            step="0.1"
            required
            disabled={isLoading}
          />
        </div>

        <div className="form-group">
          <label className="form-label">Antenna Type</label>
          <select
            className="form-select"
            value={antennaType}
            onChange={(e) => setAntennaType(e.target.value)}
            disabled={isLoading}
          >
            {ANTENNA_TYPES.map(type => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
        </div>

        <div className="divider"></div>

        <button 
          type="submit" 
          className="btn btn-primary btn-full"
          disabled={isLoading}
        >
          {isLoading ? (
            <>
              <div className="spinner"></div>
              Processing ML Models...
            </>
          ) : (
            'Generate Prediction'
          )}
        </button>
      </form>
    </div>
  );
}
