import React, { useEffect, useState, useCallback } from 'react';
import { CloudSun, Send, RefreshCw, CheckCircle2, Info } from 'lucide-react';
import { api } from '../api';

export default function Forecast() {
  const [forecast, setForecast] = useState(null);
  const [days, setDays] = useState(3);
  const [loading, setLoading] = useState(true);

  const [fbDate, setFbDate] = useState('');
  const [fbTemp, setFbTemp] = useState('');
  const [fbMsg, setFbMsg] = useState(null);
  const [fbLoading, setFbLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.getForecast(days);
      setForecast(data);
    } catch { /* noop */ }
    setLoading(false);
  }, [days]);

  useEffect(() => { load(); }, [load]);

  const submitFeedback = async (e) => {
    e.preventDefault();
    if (!fbDate || !fbTemp) return;
    setFbLoading(true);
    setFbMsg(null);
    try {
      const res = await api.postFeedback(fbDate, parseFloat(fbTemp));
      setFbMsg({ type: 'success', text: res.message });
      setFbDate('');
      setFbTemp('');
    } catch (err) {
      setFbMsg({ type: 'error', text: err.message });
    }
    setFbLoading(false);
  };

  return (
    <>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2>Forecast</h2>
          <p>Temperature predictions with confidence intervals</p>
        </div>
        <div className="btn-group">
          {[1, 2, 3].map(d => (
            <button
              key={d}
              className={`btn btn-sm ${days === d ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setDays(d)}
            >
              {d} Day{d > 1 ? 's' : ''}
            </button>
          ))}
          <button className="btn btn-secondary btn-sm" onClick={load}><RefreshCw /></button>
        </div>
      </div>

      {loading ? (
        <div className="loading-state"><div className="spinner" /></div>
      ) : forecast ? (
        <>
          <div style={{ marginBottom: 8, display: 'flex', gap: 12, alignItems: 'center' }}>
            <span className="badge badge-info"><span className="badge-dot" /> {forecast.model_used}</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              {forecast.location} · Generated {new Date(forecast.generated_at).toLocaleString()}
            </span>
          </div>

          {/* What do these values mean */}
          <div className="info-box" style={{ marginTop: 12, marginBottom: 20 }}>
            <Info style={{ width: 16, height: 16, flexShrink: 0, marginTop: 2 }} />
            <div>
              <strong>What do these temperatures mean?</strong>
              <p style={{ margin: '4px 0 0', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                Each forecast shows the <strong>predicted daily temperature</strong> for the given date. 
                The value corresponds to a <strong>9 PM local-time snapshot</strong> — the reference point 
                used across the system for daily temperature. It is <em>not</em> an hourly prediction or a 
                24-hour average, but a single representative reading for that calendar day. The range 
                below each value is a <strong>90% confidence interval</strong> — there is a 90% probability 
                the actual temperature falls within that band.
              </p>
            </div>
          </div>

          <div className="card-grid-3" style={{ marginTop: 16 }}>
            {forecast.forecasts.map(f => (
              <div key={f.horizon_days} className="card forecast-card">
                <CloudSun style={{ width: 32, height: 32, color: 'var(--warning)', marginBottom: 8 }} />
                <div className="forecast-day">Day {f.horizon_days}</div>
                <div className="forecast-date">{(() => { const [y, m, d] = (f.date || '').split('-'); return d && m && y ? `${d}-${m}-${y}` : f.date; })()}</div>
                <div className="forecast-temp">{f.predicted_temp_c}°</div>
                <div className="forecast-range">
                  {f.lower_bound_c}° – {f.upper_bound_c}°C
                </div>
                <div className="forecast-confidence">{f.confidence} confidence</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 6 }}>
                  Daily temp (9 PM snapshot)
                </div>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div className="empty-state"><p>No forecast data available. Train models first.</p></div>
      )}

      {/* Feedback form */}
      <div className="card" style={{ marginTop: 24 }}>
        <div className="card-header">
          <span className="card-title">Submit Actual Observation</span>
        </div>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: 16 }}>
          Record the real temperature for a past date to improve model accuracy over time.
          This actual reading will be used to compute prediction errors and retrain models.
        </p>
        <form onSubmit={submitFeedback}>
          <div className="form-row">
            <div className="form-group" style={{ flex: 1 }}>
              <label className="form-label">Date</label>
              <input
                type="date"
                className="form-input"
                value={fbDate}
                onChange={e => setFbDate(e.target.value)}
                max={new Date().toISOString().slice(0, 10)}
                required
              />
            </div>
            <div className="form-group" style={{ flex: 1 }}>
              <label className="form-label">Actual Temperature (°C)</label>
              <input
                type="number"
                step="0.1"
                className="form-input"
                value={fbTemp}
                onChange={e => setFbTemp(e.target.value)}
                placeholder="e.g. 28.5"
                required
              />
            </div>
            <button type="submit" className="btn btn-primary" disabled={fbLoading} style={{ marginBottom: 16 }}>
              {fbLoading ? <span className="spinner" /> : <Send />} Submit
            </button>
          </div>
        </form>
        {fbMsg && (
          <div style={{ marginTop: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
            <CheckCircle2 style={{ width: 16, color: fbMsg.type === 'success' ? 'var(--success)' : 'var(--danger)' }} />
            <span style={{ fontSize: '0.82rem', color: fbMsg.type === 'success' ? 'var(--success)' : 'var(--danger)' }}>
              {fbMsg.text}
            </span>
          </div>
        )}
      </div>
    </>
  );
}
