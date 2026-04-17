import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

const data = [
  { name: 'Power', importance: 0.42 },
  { name: 'Time', importance: 0.35 },
  { name: 'Power x Time', importance: 0.23 },
];

export default function FeatureImportanceChart() {
  return (
    <div className="card">
      <h3 className="card-title">Feature Importance</h3>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={data} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis type="category" dataKey="name" />
          <Tooltip />
          <Bar dataKey="importance" fill="#10b981" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
