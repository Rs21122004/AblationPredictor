import { NavLink } from 'react-router-dom';

const links = [
  { to: '/', label: 'Dashboard' },
  { to: '/models', label: 'Models' },
  { to: '/batch', label: 'Batch' },
  { to: '/history', label: 'History' },
  { to: '/login', label: 'Login' },
];

export default function Navbar() {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="header-brand">
          <div>
            <h1 className="header-title">Ablation AI</h1>
            <div className="header-subtitle">Production Prediction Suite</div>
          </div>
        </div>
        <nav className="header-nav">
          {links.map((link) => (
            <NavLink key={link.to} to={link.to} className="nav-tab">
              {link.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}
