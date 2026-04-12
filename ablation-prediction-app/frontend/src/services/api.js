/**
 * API Service — HTTP client for the Ablation Zone Prediction backend.
 * Wraps all API calls with error handling and response parsing.
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

/**
 * Generic fetch wrapper with error handling.
 */
async function apiFetch(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;

  try {
    const response = await fetch(url, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `API error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new Error('Cannot connect to prediction server. Is the backend running?');
    }
    throw error;
  }
}

/**
 * Check backend health status.
 */
export async function checkHealth() {
  return apiFetch('/health');
}

/**
 * Get list of available models with metrics.
 */
export async function getModels() {
  return apiFetch('/models');
}

/**
 * Single prediction.
 */
export async function predict(params) {
  return apiFetch('/predict', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

/**
 * Compare predictions across all models.
 */
export async function predictCompare(params) {
  return apiFetch('/predict/compare', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

/**
 * Batch prediction from CSV file.
 */
export async function batchPredict(file) {
  const url = `${API_BASE}/batch-predict`;
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(url, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Batch prediction failed: ${response.status}`);
  }

  return response.json();
}
