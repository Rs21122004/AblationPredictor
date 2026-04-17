import { PolarAngleAxis, PolarGrid, Radar, RadarChart, ResponsiveContainer } from 'recharts';

export default function RadarChartPanel({ models }) {
  const data = models.map((model) => ({ model: model.name, score: Math.max(model.test_r2 * 100, 0) }));
  if (!data.length) return null;
  return (
    <div className="card">
      <h3 className="card-title">Model Performance Radar</h3>
      <ResponsiveContainer width="100%" height={300}>
        <RadarChart data={data}>
          <PolarGrid />
          <PolarAngleAxis dataKey="model" />
          <Radar dataKey="score" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.4} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
