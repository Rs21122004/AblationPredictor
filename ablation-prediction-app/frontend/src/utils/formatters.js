/**
 * Formatting utilities for display values.
 */

/** Format a number to fixed decimal places. */
export function formatNumber(value, decimals = 2) {
  if (value === null || value === undefined) return '—';
  return Number(value).toFixed(decimals);
}

/** Format volume with appropriate units. */
export function formatVolume(mm3) {
  if (mm3 === null || mm3 === undefined) return '—';
  if (mm3 > 1000) {
    return `${(mm3 / 1000).toFixed(1)} cm³`;
  }
  return `${mm3.toFixed(1)} mm³`;
}

/** Format R² score as percentage string. */
export function formatR2(r2) {
  if (r2 === null || r2 === undefined) return '—';
  return `${(r2 * 100).toFixed(1)}%`;
}

/** Classify uncertainty level. */
export function getUncertaintyLevel(stdMm, meanMm) {
  if (!stdMm || !meanMm) return 'low';
  const relativeStd = stdMm / meanMm;
  if (relativeStd < 0.15) return 'low';
  if (relativeStd < 0.35) return 'medium';
  return 'high';
}

/** Uncertainty label text. */
export function getUncertaintyLabel(level) {
  const labels = {
    low: 'Low — Models agree closely',
    medium: 'Moderate — Some model disagreement',
    high: 'High — Significant model disagreement',
  };
  return labels[level] || labels.low;
}
