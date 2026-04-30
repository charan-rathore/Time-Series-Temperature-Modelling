import React, { useEffect, useState, useCallback, useRef } from 'react';
import {
  Play, RefreshCw, Database, Cpu, CheckCircle2, XCircle,
  Loader, Terminal, Download, Zap,
} from 'lucide-react';
import { api } from '../api';

export default function Pipeline() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [logs, setLogs] = useState([]);
  const [polling, setPolling] = useState(false);

  const [bfStart, setBfStart] = useState('');
  const [bfEnd, setBfEnd] = useState('');
  const [trainModels, setTrainModels] = useState({ sarima: true, lgbm: true, ensemble: true });
  const [skipMlflow, setSkipMlflow] = useState(false);

  const logRef = useRef(null);
  const pollRef = useRef(null);

  const loadStatus = useCallback(async () => {
    try {
      const [s, l] = await Promise.all([api.getStatus(), api.getLogs(200)]);
      setStatus(s);
      setLogs(l.lines);
    } catch { /* noop */ }
    setLoading(false);
  }, []);

  useEffect(() => { loadStatus(); }, [loadStatus]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  const startPolling = useCallback(() => {
    if (pollRef.current) return;
    setPolling(true);
    pollRef.current = setInterval(async () => {
      try {
        const [s, l] = await Promise.all([api.getStatus(), api.getLogs(200)]);
        setStatus(s);
        setLogs(l.lines);
        const anyRunning = Object.values(s.active_jobs || {}).some(j => j.status === 'running');
        if (!anyRunning) {
          clearInterval(pollRef.current);
          pollRef.current = null;
          setPolling(false);
        }
      } catch { /* noop */ }
    }, 2000);
  }, []);

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  const runBackfill = async () => {
    try {
      await api.runBackfill(bfStart || undefined, bfEnd || undefined);
      startPolling();
      loadStatus();
    } catch (e) {
      alert(e.message);
    }
  };

  const runDaily = async () => {
    try {
      await api.runDaily();
      startPolling();
      loadStatus();
    } catch (e) {
      alert(e.message);
    }
  };

  const runTraining = async () => {
    const models = Object.entries(trainModels).filter(([, v]) => v).map(([k]) => k);
    if (!models.length) return;
    try {
      await api.runTraining(models, skipMlflow);
      startPolling();
      loadStatus();
    } catch (e) {
      alert(e.message);
    }
  };

  const isJobRunning = (name) => status?.active_jobs?.[name]?.status === 'running';
  const anyRunning = status && Object.values(status.active_jobs || {}).some(j => j.status === 'running');

  const getJobBadge = (name) => {
    const job = status?.active_jobs?.[name];
    if (!job) return null;
    if (job.status === 'running') return <span className="badge badge-warning"><Loader style={{ width: 10, animation: 'spin 1s linear infinite' }} /> Running</span>;
    if (job.status === 'completed') return <span className="badge badge-success"><CheckCircle2 style={{ width: 10 }} /> Done</span>;
    if (job.status === 'failed') return <span className="badge badge-danger"><XCircle style={{ width: 10 }} /> Failed</span>;
    return null;
  };

  if (loading) {
    return <div className="loading-state"><div className="spinner" /><span>Loading pipeline…</span></div>;
  }

  const pipelineStep = !status?.data_available ? 0
    : status?.models_available?.length === 0 ? 1
    : 2;

  return (
    <>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2>Pipeline Control</h2>
          <p>Run data ingestion, training, and manage the full workflow</p>
        </div>
        <div className="btn-group">
          {polling && <span className="badge badge-warning"><span className="badge-dot" /> Live</span>}
          <button className="btn btn-secondary btn-sm" onClick={loadStatus}><RefreshCw /> Refresh</button>
        </div>
      </div>

      {/* Pipeline steps */}
      <div className="steps">
        <div className={`step ${pipelineStep === 0 ? 'active' : pipelineStep > 0 ? 'completed' : ''}`}>
          <div className="step-number">{pipelineStep > 0 ? '✓' : '1'}</div>
          <div className="step-text">Data Ingestion</div>
        </div>
        <div className={`step ${pipelineStep === 1 ? 'active' : pipelineStep > 1 ? 'completed' : ''}`}>
          <div className="step-number">{pipelineStep > 1 ? '✓' : '2'}</div>
          <div className="step-text">Model Training</div>
        </div>
        <div className={`step ${pipelineStep >= 2 ? 'completed' : ''}`}>
          <div className="step-number">{pipelineStep >= 2 ? '✓' : '3'}</div>
          <div className="step-text">Ready to Serve</div>
        </div>
      </div>

      {/* Status cards */}
      <div className="card-grid" style={{ marginBottom: 24 }}>
        <div className="card">
          <div className="card-header">
            <span className="card-title">Dataset</span>
            <Database style={{ width: 16, color: status?.data_available ? 'var(--success)' : 'var(--text-muted)' }} />
          </div>
          <div className="stat-value">{status?.data_rows?.toLocaleString() || 0}</div>
          <div className="stat-label">{status?.data_date_range || 'No data loaded'}</div>
        </div>
        <div className="card">
          <div className="card-header">
            <span className="card-title">Features</span>
            <Zap style={{ width: 16, color: status?.features_available ? 'var(--success)' : 'var(--text-muted)' }} />
          </div>
          <div className="stat-value">{status?.features_rows?.toLocaleString() || 0}</div>
          <div className="stat-label">{status?.features_available ? 'Feature matrix ready' : 'Not computed yet'}</div>
        </div>
        <div className="card">
          <div className="card-header">
            <span className="card-title">Models</span>
            <Cpu style={{ width: 16, color: (status?.models_available?.length || 0) > 0 ? 'var(--success)' : 'var(--text-muted)' }} />
          </div>
          <div className="stat-value">{status?.models_available?.length || 0}</div>
          <div className="stat-label">
            {status?.models_available?.length ? status.models_available.join(', ') : 'No models trained'}
          </div>
        </div>
      </div>

      <div className="card-grid-2">
        {/* Backfill */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">
              <Database style={{ width: 14, marginRight: 6, verticalAlign: -2 }} />
              Data Backfill
            </span>
            {getJobBadge('backfill')}
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: 16 }}>
            Fetch historical data from Open-Meteo and merge with legacy sensor readings.
          </p>
          <div className="form-row" style={{ marginBottom: 12 }}>
            <div className="form-group" style={{ flex: 1 }}>
              <label className="form-label">Start (optional)</label>
              <input type="date" className="form-input" value={bfStart} onChange={e => setBfStart(e.target.value)} />
            </div>
            <div className="form-group" style={{ flex: 1 }}>
              <label className="form-label">End (optional)</label>
              <input type="date" className="form-input" value={bfEnd} onChange={e => setBfEnd(e.target.value)} />
            </div>
          </div>
          <div className="btn-group">
            <button className="btn btn-primary" onClick={runBackfill} disabled={isJobRunning('backfill')}>
              {isJobRunning('backfill') ? <span className="spinner" /> : <Play />} Run Backfill
            </button>
            <button className="btn btn-secondary" onClick={runDaily} disabled={isJobRunning('daily') || !status?.data_available}>
              {isJobRunning('daily') ? <span className="spinner" /> : <Download />} Daily Update
            </button>
          </div>
        </div>

        {/* Training */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">
              <Cpu style={{ width: 14, marginRight: 6, verticalAlign: -2 }} />
              Model Training
            </span>
            {getJobBadge('train')}
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: 16 }}>
            Train SARIMA, LightGBM, and Ensemble models on your dataset.
          </p>
          <div style={{ display: 'flex', gap: 16, marginBottom: 16, flexWrap: 'wrap' }}>
            {['sarima', 'lgbm', 'ensemble'].map(m => (
              <label key={m} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.85rem', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={trainModels[m]}
                  onChange={e => setTrainModels(prev => ({ ...prev, [m]: e.target.checked }))}
                  style={{ accentColor: 'var(--accent)' }}
                />
                {m.toUpperCase()}
              </label>
            ))}
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.85rem', cursor: 'pointer', color: 'var(--text-muted)' }}>
              <input
                type="checkbox"
                checked={skipMlflow}
                onChange={e => setSkipMlflow(e.target.checked)}
                style={{ accentColor: 'var(--accent)' }}
              />
              Skip MLflow
            </label>
          </div>
          <button
            className="btn btn-success"
            onClick={runTraining}
            disabled={isJobRunning('train') || !status?.data_available}
          >
            {isJobRunning('train') ? <span className="spinner" /> : <Play />} Train Models
          </button>
          {!status?.data_available && (
            <p style={{ fontSize: '0.78rem', color: 'var(--warning)', marginTop: 8 }}>
              Run data backfill first before training.
            </p>
          )}
        </div>
      </div>

      {/* Logs */}
      <div className="card" style={{ marginTop: 24 }}>
        <div className="card-header">
          <span className="card-title">
            <Terminal style={{ width: 14, marginRight: 6, verticalAlign: -2 }} />
            Pipeline Logs
          </span>
          {anyRunning && <span className="badge badge-warning"><span className="badge-dot" /> Live</span>}
        </div>
        <div className="log-viewer" ref={logRef}>
          {logs.length === 0 ? (
            <div className="log-line" style={{ color: 'var(--text-muted)' }}>
              No logs yet. Run a pipeline action to see output here.
            </div>
          ) : (
            logs.map((line, i) => (
              <div
                key={i}
                className={`log-line ${
                  line.includes('Error') || line.includes('ERROR') ? 'error' :
                  line.includes('complete') || line.includes('Finished') || line.includes('saved') ? 'success' :
                  ''
                }`}
              >
                {line}
              </div>
            ))
          )}
        </div>
      </div>
    </>
  );
}
