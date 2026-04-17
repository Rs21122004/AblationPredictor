import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

export default function PredictionTimeline({ data }) {
  if (!data?.length) return null;
  return (
    <div className="card">
      <h3 className="card-title">Prediction Timeline</h3>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data}>
          <XAxis dataKey="ts" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="value" stroke="#06b6d4" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
