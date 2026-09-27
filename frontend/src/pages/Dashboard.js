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
  Info,
} from 'lucide-react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
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
    date: r.date,
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

  const formatDateRange = (range) => {
    if (!range) return 'No data';
    return range.replace(/(\d{4})-(\d{2})-(\d{2})/g, (_, y, m, d) => `${d}-${m}-${y}`);
  };

  return (
    <>
      <div className="page-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2>Dashboard</h2>
          <p>ThermoSense system overview - {formatDateRange(status?.data_date_range)}</p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={load}><RefreshCw /> Refresh</button>
      </div>

      {!status?.data_available && <div className="info-box" style={{ marginBottom: 20 }}>Demo host: no live dataset or trained model is loaded. Any forecast shown is Open-Meteo regional weather, not a hyperlocal model. Historical metrics below are not current sensor performance.</div>}
      <div className="card-grid">
        <div className="card">
          <div className="card-header">
            <span className="card-title">Data Points</span>
            <div className="stat-icon blue"><Database /></div>
          </div>
          <div className="stat-value">{status?.data_rows?.toLocaleString() || 0}</div>
          <div className="stat-label">{status?.data_available ? 'Dataset loaded' : 'No data - run backfill'}</div>
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
            <span className="card-title">Tomorrow's Forecast</span>
            <div className="stat-icon orange"><Thermometer /></div>
          </div>
          <div className="stat-value">
            {forecast?.forecasts?.[0]?.predicted_temp_c
              ? `${forecast.forecasts[0].predicted_temp_c}°C`
              : '-'}
          </div>
          <div className="stat-label">
            {forecast?.forecasts?.[0]
              ? forecast.availability === "regional_forecast_only" ? "Open-Meteo regional estimate; no validated interval" : `${forecast.forecasts[0].lower_bound_c}° - ${forecast.forecasts[0].upper_bound_c}°C range`
              : 'No forecast available'}
          </div>
          <div className="info-hint">
            <Info style={{ width: 12, height: 12 }} />
            9 PM local snapshot forecast; source shown below
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title">Day-1 RMSE</span>
            <div className="stat-icon blue"><TrendingUp /></div>
          </div>
          <div className="stat-value">
            {status?.models_available?.length && status?.data_available && status?.last_training_results?.lgbm?.day1?.rmse
                ? `${status.last_training_results.lgbm.day1.rmse.toFixed(3)}°C`
                : '-'}
          </div>
          <div className="stat-label">Root mean squared error</div>
          <div className="info-hint">
            <Info style={{ width: 12, height: 12 }} />
            No verified live accuracy score on this host
          </div>
        </div>
      </div>

      {/* Forecast strip */}
      {!!forecast?.forecasts?.length && (
        <div className="card" style={{ marginBottom: 24, cursor: 'pointer' }} onClick={() => onNavigate('forecast')}>
          <div className="card-header">
            <span className="card-title"><CloudSun style={{ width: 14, height: 14, marginRight: 6, verticalAlign: -2 }} />
              3-Day Forecast - {forecast.model_used}</span>
            <span style={{ fontSize: '0.78rem', color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: 4 }}>
              View details <ArrowRight style={{ width: 14 }} />
            </span>
          </div>
          <p className="chart-description">
            Predicted 9 PM local-time temperatures for the next 3 days. These are point-in-time snapshots, not daily averages.
          </p>
          <div className="card-grid-3">
            {forecast.forecasts.map(f => {
              const [y, m, d] = (f.date || '').split('-');
              const fmtDate = d && m && y ? `${d}-${m}-${y}` : f.date;
              return (
                <div key={f.horizon_days} className="forecast-card card">
                  <div className="forecast-day">Day {f.horizon_days}</div>
                  <div className="forecast-date">{fmtDate}</div>
                  <div className="forecast-temp">{f.predicted_temp_c}°</div>
                  {f.confidence !== "none" && <div className="forecast-range">{f.lower_bound_c}° - {f.upper_bound_c}°C</div>}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Temperature chart */}
      {chartData.length > 0 && (
        <div className="card">
          <div className="card-header">
            <span className="card-title">Temperature - Last 30 Days</span>
          </div>
          <p className="chart-description">
            Comparison of two temperature sources over the last 30 days.{' '}
            <strong style={{ color: '#3b82f6' }}>Observation</strong> = a local recorded value; older manual rows are not verified DHT22 readings.{' '}
            <strong style={{ color: '#f59e0b' }}>API</strong> = Open-Meteo grid-cell estimate for this location.
            Differences can reflect siting, time matching, measurement calibration, or local microclimate; they do not by themselves prove the cause.
          </p>
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
                <XAxis
                  dataKey="date"
                  stroke="#6b7280"
                  fontSize={11}
                  tickFormatter={v => {
                    if (!v) return '';
                    const parts = v.split('-');
                    return `${parts[2]}-${parts[1]}`;
                  }}
                  label={{ value: 'Date (DD-MM)', position: 'insideBottom', offset: -5, fill: '#6b7280', fontSize: 11 }}
                  height={50}
                />
                <YAxis
                  stroke="#6b7280"
                  fontSize={11}
                  domain={['auto', 'auto']}
                  label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 11, dx: -5 }}
                  width={65}
                />
                <Tooltip
                  contentStyle={{ background: '#1e2130', border: '1px solid #2a2d3a', borderRadius: 8 }}
                  labelStyle={{ color: '#9aa0b0' }}
                  labelFormatter={v => {
                    if (!v) return '';
                    const parts = v.split('-');
                    return `${parts[2]}-${parts[1]}-${parts[0]}`;
                  }}
                  formatter={(value, name) => [`${value}°C`, name]}
                />
                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
                <Area type="monotone" dataKey="temp" name="Sensor (local reading) °C" stroke="#3b82f6" fill="url(#tempGrad)" strokeWidth={2} dot={false} />
                <Area type="monotone" dataKey="api" name="API (Open-Meteo grid) °C" stroke="#f59e0b" fill="url(#apiGrad)" strokeWidth={1.5} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </>
  );
}
