import { useState } from 'react';

import ModelInfo from '../components/ModelInfo';
import PredictionForm from '../components/PredictionForm';
import PredictionResults from '../components/PredictionResults';
import PredictionTimeline from '../components/charts/PredictionTimeline';
import { predict } from '../services/api';

export default function DashboardPage() {
  const [predictionResult, setPredictionResult] = useState(null);
  const [predictionError, setPredictionError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [history, setHistory] = useState([]);

  const handlePredictionSubmit = async (params) => {
    setIsLoading(true);
    setPredictionError(null);
    try {
      const result = await predict(params);
      setPredictionResult(result);
      setHistory((prev) => [...prev, { ts: new Date().toLocaleTimeString(), value: result.prediction.estimated_volume_mm3 }]);
    } catch (err) {
      setPredictionError(err.message || 'Prediction failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="tab-panel">
      <div className="dashboard">
        <PredictionForm onSubmit={handlePredictionSubmit} isLoading={isLoading} />
        <div className="tab-panel">
          <PredictionResults result={predictionResult} error={predictionError} isLoading={isLoading} />
          <ModelInfo modelData={predictionResult} isLoading={isLoading} />
        </div>
      </div>
      <PredictionTimeline data={history} />
    </div>
  );
}
