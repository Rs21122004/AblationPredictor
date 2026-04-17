export default function AblationEllipse({ diameter = 20, length = 30 }) {
  const rx = Math.max(20, diameter * 1.5);
  const ry = Math.max(20, length * 1.5);
  return (
    <div className="card ablation-viz">
      <h3 className="card-title">Ablation Zone Geometry</h3>
      <svg viewBox="0 0 200 200" className="ablation-svg-container">
        <ellipse cx="100" cy="100" rx={rx / 2} ry={ry / 2} fill="rgba(6, 182, 212, 0.35)" stroke="#06b6d4" />
      </svg>
    </div>
  );
}
