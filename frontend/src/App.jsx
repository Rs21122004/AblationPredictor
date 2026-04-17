import { BrowserRouter, Route, Routes } from 'react-router-dom';

import Navbar from './components/Navbar';
import AuthPage from './pages/AuthPage';
import BatchUploadPage from './pages/BatchUploadPage';
import DashboardPage from './pages/DashboardPage';
import ModelExplorerPage from './pages/ModelExplorerPage';
import PredictionHistoryPage from './pages/PredictionHistoryPage';

export default function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <Navbar />
        <main className="app-main">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/models" element={<ModelExplorerPage />} />
            <Route path="/batch" element={<BatchUploadPage />} />
            <Route path="/history" element={<PredictionHistoryPage />} />
            <Route path="/login" element={<AuthPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
