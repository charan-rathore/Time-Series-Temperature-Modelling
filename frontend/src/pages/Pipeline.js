import React, { useEffect, useState, useCallback, useRef } from 'react';
import {
  Play, RefreshCw, Database, Cpu, CheckCircle2, XCircle,
  Loader, Terminal, Download, Zap, FlaskConical, Clock, Tag,
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

  const [mlflow, setMlflow] = useState(null);
  const [mlflowLoading, setMlflowLoading] = useState(false);
  const [mlflowExpanded, setMlflowExpanded] = useState(null);

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

  const loadMlflow = useCallback(async () => {
    setMlflowLoading(true);
    try {
      const data = await api.getMlflowRuns(20);
      setMlflow(data);
    } catch {
      setMlflow({ available: false });
    }
    setMlflowLoading(false);
  }, []);

  useEffect(() => { loadStatus(); loadMlflow(); }, [loadStatus, loadMlflow]);

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
          loadMlflow();
        }
      } catch { /* noop */ }
    }, 2000);
  }, [loadMlflow]);

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

  const formatRunTime = (startStr, endStr) => {
    if (!startStr) return '—';
    try {
      const start = new Date(startStr);
      const dateStr = start.toLocaleDateString();
      const timeStr = start.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      if (endStr) {
        const end = new Date(endStr);
        const durationSec = Math.round((end - start) / 1000);
        if (durationSec < 60) return `${dateStr} ${timeStr} (${durationSec}s)`;
        return `${dateStr} ${timeStr} (${Math.round(durationSec / 60)}m)`;
      }
      return `${dateStr} ${timeStr}`;
    } catch {
      return startStr;
    }
  };

  return (
    <>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2>Pipeline Control</h2>
          <p>Run data ingestion, training, and manage the full workflow</p>
        </div>
        <div className="btn-group">
          {polling && <span className="badge badge-warning"><span className="badge-dot" /> Live</span>}
          <button className="btn btn-secondary btn-sm" onClick={() => { loadStatus(); loadMlflow(); }}><RefreshCw /> Refresh</button>
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

      {/* MLflow Experiment Tracking */}
      <div className="card" style={{ marginTop: 24 }}>
        <div className="card-header">
          <span className="card-title">
            <FlaskConical style={{ width: 14, marginRight: 6, verticalAlign: -2 }} />
            MLflow Experiment Tracking
          </span>
          <div className="btn-group">
            {mlflow?.available && (
              <span className="badge badge-success" style={{ fontSize: '0.68rem' }}>
                <CheckCircle2 style={{ width: 10 }} /> Connected
              </span>
            )}
            <button className="btn btn-secondary btn-sm" onClick={loadMlflow} disabled={mlflowLoading}>
              <RefreshCw style={mlflowLoading ? { animation: 'spin 1s linear infinite' } : {}} />
            </button>
          </div>
        </div>

        {mlflowLoading && !mlflow ? (
          <div className="loading-state" style={{ padding: 24 }}><div className="spinner" /></div>
        ) : !mlflow?.available ? (
          <div style={{ padding: '16px 0' }}>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: 12 }}>
              MLflow is not available. Install it with <code style={{ background: 'var(--bg-input)', padding: '2px 6px', borderRadius: 4 }}>pip install mlflow</code> and 
              run training without the "Skip MLflow" checkbox to start tracking experiments.
            </p>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              MLflow tracks model parameters, metrics, and artifacts across training runs, enabling you to 
              compare experiments and reproduce results.
            </p>
          </div>
        ) : mlflow.runs.length === 0 ? (
          <div style={{ padding: '16px 0' }}>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              No experiment runs found. Train models with MLflow enabled to start tracking experiments.
            </p>
            {mlflow.tracking_uri && (
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 8 }}>
                Tracking URI: <code style={{ background: 'var(--bg-input)', padding: '2px 6px', borderRadius: 4 }}>{mlflow.tracking_uri}</code>
              </p>
            )}
          </div>
        ) : (
          <>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 16 }}>
              {mlflow.total_runs} training run{mlflow.total_runs !== 1 ? 's' : ''} tracked in 
              experiment "<strong>{mlflow.experiment_name}</strong>". Click a row to see full details.
            </p>

            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>Run</th>
                    <th>Model</th>
                    <th>Status</th>
                    <th>Time</th>
                    <th>Day-1 RMSE</th>
                    <th>Day-1 MAE</th>
                    <th>Day-2 RMSE</th>
                    <th>Day-3 RMSE</th>
                  </tr>
                </thead>
                <tbody>
                  {mlflow.runs.map((run, idx) => {
                    const isExpanded = mlflowExpanded === idx;
                    return (
                      <React.Fragment key={run.run_id}>
                        <tr
                          onClick={() => setMlflowExpanded(isExpanded ? null : idx)}
                          style={{ cursor: 'pointer' }}
                        >
                          <td style={{ fontSize: '0.78rem' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                              <Tag style={{ width: 12, height: 12, color: 'var(--text-muted)' }} />
                              <span className="mono" style={{ fontSize: '0.72rem' }}>
                                {run.run_name || run.run_id.slice(0, 8)}
                              </span>
                            </div>
                          </td>
                          <td>
                            <span className="badge badge-info">{run.model || run.params?.model || '—'}</span>
                          </td>
                          <td>
                            <span className={`badge ${run.status === 'FINISHED' ? 'badge-success' : run.status === 'FAILED' ? 'badge-danger' : 'badge-warning'}`}>
                              {run.status === 'FINISHED' ? 'Done' : run.status}
                            </span>
                          </td>
                          <td style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                            <Clock style={{ width: 11, height: 11, marginRight: 4, verticalAlign: -1 }} />
                            {formatRunTime(run.start_time, run.end_time)}
                          </td>
                          <td className="mono">{run.metrics?.day1_rmse?.toFixed(4) ?? '—'}</td>
                          <td className="mono">{run.metrics?.day1_mae?.toFixed(4) ?? '—'}</td>
                          <td className="mono">{run.metrics?.day2_rmse?.toFixed(4) ?? '—'}</td>
                          <td className="mono">{run.metrics?.day3_rmse?.toFixed(4) ?? '—'}</td>
                        </tr>
                        {isExpanded && (
                          <tr className="mlflow-detail-row">
                            <td colSpan={8} style={{ background: 'var(--bg-secondary)', padding: 16 }}>
                              <div className="mlflow-detail-grid">
                                <div>
                                  <h4 style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                    Parameters
                                  </h4>
                                  {Object.keys(run.params).length > 0 ? (
                                    <div className="mlflow-kv-list">
                                      {Object.entries(run.params).map(([k, v]) => (
                                        <div key={k} className="mlflow-kv">
                                          <span className="mlflow-kv-key">{k}</span>
                                          <span className="mlflow-kv-val">{String(v)}</span>
                                        </div>
                                      ))}
                                    </div>
                                  ) : <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>None</span>}
                                </div>
                                <div>
                                  <h4 style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                    All Metrics
                                  </h4>
                                  {Object.keys(run.metrics).length > 0 ? (
                                    <div className="mlflow-kv-list">
                                      {Object.entries(run.metrics)
                                        .sort(([a], [b]) => a.localeCompare(b))
                                        .map(([k, v]) => (
                                          <div key={k} className="mlflow-kv">
                                            <span className="mlflow-kv-key">{k}</span>
                                            <span className="mlflow-kv-val">{typeof v === 'number' ? v.toFixed(4) : String(v)}</span>
                                          </div>
                                        ))}
                                    </div>
                                  ) : <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>None</span>}
                                </div>
                                <div>
                                  <h4 style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                    Run Info
                                  </h4>
                                  <div className="mlflow-kv-list">
                                    <div className="mlflow-kv">
                                      <span className="mlflow-kv-key">Run ID</span>
                                      <span className="mlflow-kv-val mono" style={{ fontSize: '0.7rem' }}>{run.run_id}</span>
                                    </div>
                                    <div className="mlflow-kv">
                                      <span className="mlflow-kv-key">Started</span>
                                      <span className="mlflow-kv-val">{run.start_time ? new Date(run.start_time).toLocaleString() : '—'}</span>
                                    </div>
                                    <div className="mlflow-kv">
                                      <span className="mlflow-kv-key">Finished</span>
                                      <span className="mlflow-kv-val">{run.end_time ? new Date(run.end_time).toLocaleString() : '—'}</span>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 12 }}>
              For the full MLflow UI with charts and artifact browsing, run{' '}
              <code style={{ background: 'var(--bg-input)', padding: '2px 6px', borderRadius: 4 }}>mlflow ui --port 5000 --backend-store-uri {mlflow.tracking_uri}</code>
            </p>
          </>
        )}
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
