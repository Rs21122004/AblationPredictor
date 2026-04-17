import { useState } from 'react';

import { getBatchJob, startBatchJob } from '../services/api';

export default function BatchUploadPage() {
  const [job, setJob] = useState(null);

  const handleUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const started = await startBatchJob(file);
    const status = await getBatchJob(started.job_id);
    setJob(status);
  };

  return (
    <div className="card">
      <h2 className="card-title">Batch Upload</h2>
      <input type="file" accept=".csv" onChange={handleUpload} />
      {job && (
        <div style={{ marginTop: '1rem' }}>
          <div>Status: {job.status}</div>
          <div>Total: {job.total_rows}</div>
          <div>Successful: {job.successful}</div>
          <div>Failed: {job.failed}</div>
        </div>
      )}
    </div>
  );
}
