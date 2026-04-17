import { useEffect, useState } from 'react';

import { getPredictionHistory } from '../services/api';

export default function PredictionHistoryPage() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    getPredictionHistory(1, 50).then((data) => setHistory(data.items || [])).catch(() => setHistory([]));
  }, []);

  return (
    <div className="card">
      <h2 className="card-title">Prediction History</h2>
      <table className="comparison-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Power</th>
            <th>Diameter</th>
            <th>Length</th>
            <th>Volume</th>
          </tr>
        </thead>
        <tbody>
          {history.map((row) => (
            <tr key={row.id}>
              <td>{new Date(row.created_at).toLocaleString()}</td>
              <td>{row.power}</td>
              <td>{row.predicted_diameter}</td>
              <td>{row.predicted_length}</td>
              <td>{row.predicted_volume}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
