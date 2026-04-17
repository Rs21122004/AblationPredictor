import { useEffect, useState } from 'react';

import FeatureImportanceChart from '../components/charts/FeatureImportanceChart';
import RadarChartPanel from '../components/charts/RadarChart';
import ScatterPlotPanel from '../components/charts/ScatterPlot';
import { getModels } from '../services/api';

export default function ModelExplorerPage() {
  const [models, setModels] = useState([]);

  useEffect(() => {
    getModels().then((data) => setModels(data.diameter_models || [])).catch(() => setModels([]));
  }, []);

  return (
    <div className="tab-panel">
      <RadarChartPanel models={models} />
      <FeatureImportanceChart />
      <ScatterPlotPanel models={models} />
    </div>
  );
}
