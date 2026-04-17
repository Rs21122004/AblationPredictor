import { useState } from 'react';

import { login, register, setAccessToken } from '../services/api';

export default function AuthPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [message, setMessage] = useState('');

  const onRegister = async () => {
    await register({ email, password, full_name: fullName });
    setMessage('Registration successful. You can log in now.');
  };

  const onLogin = async () => {
    const result = await login({ email, password });
    setAccessToken(result.access_token);
    setMessage('Logged in successfully.');
  };

  return (
    <div className="card" style={{ maxWidth: 500 }}>
      <h2 className="card-title">Authentication</h2>
      <div className="form-group">
        <label className="form-label">Full Name</label>
        <input className="form-input" value={fullName} onChange={(e) => setFullName(e.target.value)} />
      </div>
      <div className="form-group">
        <label className="form-label">Email</label>
        <input className="form-input" value={email} onChange={(e) => setEmail(e.target.value)} />
      </div>
      <div className="form-group">
        <label className="form-label">Password</label>
        <input type="password" className="form-input" value={password} onChange={(e) => setPassword(e.target.value)} />
      </div>
      <div className="form-row">
        <button className="btn btn-secondary" onClick={onRegister}>Register</button>
        <button className="btn btn-primary" onClick={onLogin}>Login</button>
      </div>
      {message && <p style={{ marginTop: '1rem' }}>{message}</p>}
    </div>
  );
}
