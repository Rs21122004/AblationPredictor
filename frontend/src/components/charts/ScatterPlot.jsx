import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts';

const demo = Array.from({ length: 20 }).map((_, index) => ({ x: index + 1, y: 8 + Math.random() * 10 }));

export default function ScatterPlotPanel() {
  return (
    <div className="card">
      <h3 className="card-title">Predicted vs Actual</h3>
      <ResponsiveContainer width="100%" height={260}>
        <ScatterChart>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="x" />
          <YAxis dataKey="y" />
          <Tooltip />
          <Scatter data={demo} fill="#8b5cf6" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
