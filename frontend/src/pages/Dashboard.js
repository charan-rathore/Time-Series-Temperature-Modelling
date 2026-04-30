import React, { useEffect, useState, useCallback } from 'react';
import {
  Thermometer,
  Database,
  Cpu,
  TrendingUp,
  CloudSun,
  ArrowRight,
  RefreshCw,
  AlertCircle,
} from 'lucide-react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { api } from '../api';

export default function Dashboard({ onNavigate }) {
  const [status, setStatus] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [s, f, h] = await Promise.all([
        api.getStatus(),
        api.getForecast(3).catch(() => null),
        api.getHistory().catch(() => null),
      ]);
      setStatus(s);
      setForecast(f);
      setHistory(h);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  if (loading) {
    return <div className="loading-state"><div className="spinner" /><span>Loading dashboard…</span></div>;
  }

  if (error) {
    return (
      <div className="empty-state">
        <AlertCircle />
        <p>Failed to connect: {error}</p>
        <button className="btn btn-secondary btn-sm" onClick={load} style={{ marginTop: 12 }}>
          <RefreshCw /> Retry
        </button>
      </div>
    );
  }

  const chartData = history?.records?.slice(-30).map(r => ({
    date: r.date?.slice(5),
    temp: r.actual_temp_c,
    api: r.api_temp_c,
  })) || [];

  const bestModel = status?.models_available?.includes('ensemble')
    ? 'Ensemble'
    : status?.models_available?.find(m => m.startsWith('lgbm'))
      ? 'LightGBM'
      : status?.models_available?.includes('sarima')
        ? 'SARIMA'
        : 'None';

  return (
    <>
      <div className="page-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2>Dashboard</h2>
          <p>ThermoSense system overview — {status?.data_date_range || 'No data'}</p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={load}><RefreshCw /> Refresh</button>
      </div>

      <div className="card-grid">
        <div className="card">
          <div className="card-header">
            <span className="card-title">Data Points</span>
            <div className="stat-icon blue"><Database /></div>
          </div>
          <div className="stat-value">{status?.data_rows?.toLocaleString() || 0}</div>
          <div className="stat-label">{status?.data_available ? 'Dataset loaded' : 'No data — run backfill'}</div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title">Active Model</span>
            <div className="stat-icon green"><Cpu /></div>
          </div>
          <div className="stat-value">{bestModel}</div>
          <div className="stat-label">
            {status?.models_available?.length || 0} model file{status?.models_available?.length !== 1 ? 's' : ''} loaded
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title">Tomorrow</span>
            <div className="stat-icon orange"><Thermometer /></div>
          </div>
          <div className="stat-value">
            {forecast?.forecasts?.[0]?.predicted_temp_c
              ? `${forecast.forecasts[0].predicted_temp_c}°C`
              : '—'}
          </div>
          <div className="stat-label">
            {forecast?.forecasts?.[0]
              ? `${forecast.forecasts[0].lower_bound_c}° – ${forecast.forecasts[0].upper_bound_c}°`
              : 'No forecast available'}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title">Day-1 RMSE</span>
            <div className="stat-icon blue"><TrendingUp /></div>
          </div>
          <div className="stat-value">
            {status?.last_training_results?.ensemble?.day1?.rmse
              ? `${status.last_training_results.ensemble.day1.rmse.toFixed(3)}°C`
              : status?.last_training_results?.lgbm?.day1?.rmse
                ? `${status.last_training_results.lgbm.day1.rmse.toFixed(3)}°C`
                : '—'}
          </div>
          <div className="stat-label">Root mean squared error</div>
        </div>
      </div>

      {/* Forecast strip */}
      {forecast?.forecasts && (
        <div className="card" style={{ marginBottom: 24, cursor: 'pointer' }} onClick={() => onNavigate('forecast')}>
          <div className="card-header">
            <span className="card-title"><CloudSun style={{ width: 14, height: 14, marginRight: 6, verticalAlign: -2 }} />
              3-Day Forecast — {forecast.model_used}</span>
            <span style={{ fontSize: '0.78rem', color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: 4 }}>
              View details <ArrowRight style={{ width: 14 }} />
            </span>
          </div>
          <div className="card-grid-3">
            {forecast.forecasts.map(f => (
              <div key={f.horizon_days} className="forecast-card card">
                <div className="forecast-day">Day {f.horizon_days}</div>
                <div className="forecast-date">{f.date}</div>
                <div className="forecast-temp">{f.predicted_temp_c}°</div>
                <div className="forecast-range">{f.lower_bound_c}° – {f.upper_bound_c}°</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Temperature chart */}
      {chartData.length > 0 && (
        <div className="card">
          <div className="card-header">
            <span className="card-title">Temperature — Last 30 Days</span>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="tempGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="apiGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
                <XAxis dataKey="date" stroke="#6b7280" fontSize={11} />
                <YAxis stroke="#6b7280" fontSize={11} domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{ background: '#1e2130', border: '1px solid #2a2d3a', borderRadius: 8 }}
                  labelStyle={{ color: '#9aa0b0' }}
                />
                <Area type="monotone" dataKey="temp" name="Sensor °C" stroke="#3b82f6" fill="url(#tempGrad)" strokeWidth={2} dot={false} />
                <Area type="monotone" dataKey="api" name="API °C" stroke="#f59e0b" fill="url(#apiGrad)" strokeWidth={1.5} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </>
  );
}
