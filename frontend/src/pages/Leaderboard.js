import React, { useEffect, useState, useCallback } from 'react';
import {
  RefreshCw,
  Trophy,
  Medal,
  TrendingUp,
  TrendingDown,
  AlertCircle,
  Calendar,
  Info,
  Award,
  Target,
  CheckCircle,
  XCircle,
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell,
  LineChart, Line,
} from 'recharts';
import { api } from '../api';

const SOURCE_COLORS = {
  thermosense: '#22c55e',
  open_meteo: '#3b82f6',
  openweathermap: '#f59e0b',
  accuweather: '#ef4444',
  google: '#8b5cf6',
};

const SOURCE_NAMES = {
  thermosense: 'ThermoSense',
  open_meteo: 'Open-Meteo',
  openweathermap: 'OpenWeatherMap',
  accuweather: 'AccuWeather',
  google: 'Google Weather',
};

function getRankIcon(rank) {
  if (rank === 1) return <Trophy style={{ color: '#fbbf24', width: 20, height: 20 }} />;
  if (rank === 2) return <Medal style={{ color: '#94a3b8', width: 18, height: 18 }} />;
  if (rank === 3) return <Medal style={{ color: '#d97706', width: 18, height: 18 }} />;
  return <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>{rank}</span>;
}

export default function Leaderboard() {
  const [data, setData] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [windowDays, setWindowDays] = useState(30);
  const [horizon, setHorizon] = useState(1);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [leaderboard, comp, stat] = await Promise.all([
        api.getLeaderboard(windowDays, horizon),
        api.getLeaderboardComparison(),
        api.getLeaderboardStatus(),
      ]);
      setData(leaderboard);
      setComparison(comp);
      setStatus(stat);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [windowDays, horizon]);

  useEffect(() => { load(); }, [load]);

  if (loading) {
    return <div className="loading-state"><div className="spinner" /><span>Loading leaderboard…</span></div>;
  }

  if (error) {
    return (
      <div className="empty-state">
        <AlertCircle />
        <p>Failed to load leaderboard: {error}</p>
        <button className="btn btn-secondary btn-sm" onClick={load} style={{ marginTop: 12 }}>
          <RefreshCw /> Retry
        </button>
      </div>
    );
  }

  const hasData = data?.rankings && data.rankings.length > 0;
  const isInitialized = status?.initialized;

  if (!isInitialized) {
    return (
      <div className="empty-state">
        <Target style={{ width: 48, height: 48 }} />
        <p>Baseline collection not initialized.</p>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: 8 }}>
          Run <code style={{ background: 'var(--surface-elevated)', padding: '2px 6px', borderRadius: 4 }}>
            python scripts/collect_baselines.py --init
          </code> to set up the database.
        </p>
      </div>
    );
  }

  if (!hasData) {
    return (
      <>
        <div className="page-header">
          <h2>Live Accuracy Leaderboard</h2>
          <p>Comparing ThermoSense vs commercial weather services</p>
        </div>

        <div className="card" style={{ textAlign: 'center', padding: 40 }}>
          <Award style={{ width: 48, height: 48, color: 'var(--text-secondary)', marginBottom: 16 }} />
          <h3 style={{ marginBottom: 8 }}>No Rankings Yet</h3>
          <p style={{ color: 'var(--text-secondary)', marginBottom: 24 }}>
            Need at least 3 days of forecasts AND actual sensor readings to compute rankings.
          </p>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, maxWidth: 400, margin: '0 auto' }}>
            <div className="card" style={{ padding: 16, textAlign: 'left' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                {status?.forecast_count > 0 ? 
                  <CheckCircle style={{ color: 'var(--success)', width: 16 }} /> : 
                  <XCircle style={{ color: 'var(--text-secondary)', width: 16 }} />
                }
                <span style={{ fontWeight: 600 }}>Forecasts</span>
              </div>
              <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{status?.forecast_count || 0}</div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>stored</div>
            </div>
            
            <div className="card" style={{ padding: 16, textAlign: 'left' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                {status?.actual_count > 0 ? 
                  <CheckCircle style={{ color: 'var(--success)', width: 16 }} /> : 
                  <XCircle style={{ color: 'var(--text-secondary)', width: 16 }} />
                }
                <span style={{ fontWeight: 600 }}>Actuals</span>
              </div>
              <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{status?.actual_count || 0}</div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>sensor readings</div>
            </div>
          </div>

          <p style={{ marginTop: 24, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Run <code style={{ background: 'var(--surface-elevated)', padding: '2px 6px', borderRadius: 4 }}>
              python scripts/collect_baselines.py
            </code> daily to collect forecasts.
          </p>
        </div>
      </>
    );
  }

  const thermosenseEntry = data.rankings.find(r => r.source === 'thermosense');
  const bestCommercial = data.rankings.find(r => r.source !== 'thermosense');
  const improvement = data.improvement_vs_best_commercial_pct;

  const chartData = data.rankings.map(r => ({
    source: SOURCE_NAMES[r.source] || r.source,
    rmse: r.rmse,
    mae: r.mae,
    fill: SOURCE_COLORS[r.source] || '#666',
  }));

  const comparisonChartData = comparison?.data?.slice(0, 14).reverse().map(d => {
    const row = { date: d.date };
    row.actual = d.actual_temp_c;
    Object.entries(d.forecasts || {}).forEach(([source, horizons]) => {
      if (horizons.day1) {
        row[source] = horizons.day1.predicted;
      }
    });
    return row;
  }) || [];

  return (
    <>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h2>Live Accuracy Leaderboard</h2>
          <p>Comparing ThermoSense vs commercial weather services · Updated daily</p>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <select 
            value={windowDays} 
            onChange={e => setWindowDays(Number(e.target.value))}
            className="form-select"
            style={{ width: 'auto' }}
          >
            <option value={7}>Last 7 days</option>
            <option value={14}>Last 14 days</option>
            <option value={30}>Last 30 days</option>
            <option value={60}>Last 60 days</option>
            <option value={90}>Last 90 days</option>
          </select>
          <select 
            value={horizon} 
            onChange={e => setHorizon(Number(e.target.value))}
            className="form-select"
            style={{ width: 'auto' }}
          >
            <option value={1}>Day 1 (Tomorrow)</option>
            <option value={2}>Day 2</option>
            <option value={3}>Day 3</option>
          </select>
          <button className="btn btn-secondary btn-sm" onClick={load}><RefreshCw /></button>
        </div>
      </div>

      {/* Improvement banner */}
      {thermosenseEntry && bestCommercial && improvement !== null && (
        <div 
          className="card" 
          style={{ 
            marginBottom: 24, 
            background: improvement > 0 ? 'var(--success-soft)' : 'var(--warning-soft)',
            borderColor: improvement > 0 ? 'var(--success)' : 'var(--warning)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            {improvement > 0 ? 
              <TrendingUp style={{ color: 'var(--success)', width: 32, height: 32 }} /> :
              <TrendingDown style={{ color: 'var(--warning)', width: 32, height: 32 }} />
            }
            <div>
              <div style={{ fontWeight: 700, fontSize: '1.1rem' }}>
                ThermoSense {improvement > 0 ? 'beats' : 'trails'} {SOURCE_NAMES[bestCommercial.source] || bestCommercial.source} by {Math.abs(improvement).toFixed(1)}%
              </div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Day-{horizon} RMSE: {thermosenseEntry.rmse}°C vs {bestCommercial.rmse}°C · 
                Based on {thermosenseEntry.n_days} days of parallel forecasts
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Stats cards */}
      <div className="card-grid" style={{ marginBottom: 24 }}>
        <div className="card">
          <div className="card-header">
            <span className="card-title">ThermoSense Rank</span>
            <div className="stat-icon green"><Trophy /></div>
          </div>
          <div className="stat-value" style={{ color: thermosenseEntry?.rank === 1 ? 'var(--success)' : 'inherit' }}>
            #{thermosenseEntry?.rank || '—'}
          </div>
          <div className="stat-label">of {data.rankings.length} forecasters</div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title">ThermoSense RMSE</span>
            <div className="stat-icon blue"><Target /></div>
          </div>
          <div className="stat-value">{thermosenseEntry?.rmse?.toFixed(3) || '—'}°C</div>
          <div className="stat-label">Day-{horizon} prediction error</div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title">Sample Size</span>
            <div className="stat-icon orange"><Calendar /></div>
          </div>
          <div className="stat-value">{thermosenseEntry?.n_days || 0}</div>
          <div className="stat-label">days of parallel comparison</div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title">Forecasts Stored</span>
            <div className="stat-icon blue"><TrendingUp /></div>
          </div>
          <div className="stat-value">{status?.forecast_count?.toLocaleString() || 0}</div>
          <div className="stat-label">total predictions collected</div>
        </div>
      </div>

      {/* Rankings table */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-header">
          <span className="card-title">
            <Trophy style={{ width: 14, height: 14, marginRight: 6, verticalAlign: -2 }} />
            Day-{horizon} Forecast Accuracy Rankings (Last {windowDays} Days)
          </span>
        </div>
        <p className="chart-description">
          Each forecaster's predictions are compared against actual sensor readings at 9 PM daily.
          RMSE (Root Mean Squared Error) penalizes large misses more heavily than MAE.
          Lower values indicate more accurate predictions.
        </p>
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th style={{ width: 60 }}>Rank</th>
                <th>Source</th>
                <th title="Root Mean Squared Error in °C">RMSE</th>
                <th title="Mean Absolute Error in °C">MAE</th>
                <th title="Mean bias (positive = overpredicting)">Bias</th>
                <th>Days</th>
              </tr>
            </thead>
            <tbody>
              {data.rankings.map((r, i) => (
                <tr 
                  key={r.source} 
                  style={r.source === 'thermosense' ? { background: 'var(--success-soft)' } : {}}
                >
                  <td style={{ textAlign: 'center' }}>{getRankIcon(r.rank)}</td>
                  <td>
                    <span style={{ 
                      fontWeight: 600, 
                      color: SOURCE_COLORS[r.source] || 'inherit',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                    }}>
                      <span style={{ 
                        width: 8, 
                        height: 8, 
                        borderRadius: '50%', 
                        background: SOURCE_COLORS[r.source] || '#666',
                      }} />
                      {SOURCE_NAMES[r.source] || r.source}
                      {r.source === 'thermosense' && (
                        <span className="badge badge-success" style={{ marginLeft: 4 }}>You</span>
                      )}
                    </span>
                  </td>
                  <td className="mono" style={{ fontWeight: r.rank === 1 ? 700 : 400 }}>
                    {r.rmse?.toFixed(3)}°C
                  </td>
                  <td className="mono">{r.mae?.toFixed(3)}°C</td>
                  <td className="mono" style={{ 
                    color: r.mean_error > 0 ? '#f59e0b' : r.mean_error < 0 ? '#3b82f6' : 'inherit' 
                  }}>
                    {r.mean_error > 0 ? '+' : ''}{r.mean_error?.toFixed(3)}°C
                  </td>
                  <td className="mono">{r.n_days}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Charts */}
      <div className="card-grid-2" style={{ marginBottom: 24 }}>
        <div className="card">
          <div className="card-header"><span className="card-title">RMSE Comparison</span></div>
          <p className="chart-description">
            Lower RMSE = more accurate. ThermoSense learns your location's microclimate bias,
            which commercial apps miss because they use regional grid averages.
          </p>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
                <XAxis type="number" stroke="#6b7280" fontSize={11} unit="°C" />
                <YAxis type="category" dataKey="source" stroke="#6b7280" fontSize={11} width={100} />
                <Tooltip
                  contentStyle={{ background: '#1e2130', border: '1px solid #2a2d3a', borderRadius: 8 }}
                  formatter={(value) => [`${value.toFixed(3)}°C`, 'RMSE']}
                />
                <Bar dataKey="rmse" radius={[0, 4, 4, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="card-header"><span className="card-title">MAE Comparison</span></div>
          <p className="chart-description">
            MAE (Mean Absolute Error) shows the average prediction miss in °C.
            Unlike RMSE, MAE doesn't penalize occasional large errors more heavily.
          </p>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
                <XAxis type="number" stroke="#6b7280" fontSize={11} unit="°C" />
                <YAxis type="category" dataKey="source" stroke="#6b7280" fontSize={11} width={100} />
                <Tooltip
                  contentStyle={{ background: '#1e2130', border: '1px solid #2a2d3a', borderRadius: 8 }}
                  formatter={(value) => [`${value.toFixed(3)}°C`, 'MAE']}
                />
                <Bar dataKey="mae" radius={[0, 4, 4, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Forecast vs Actual chart */}
      {comparisonChartData.length > 0 && (
        <div className="card">
          <div className="card-header">
            <span className="card-title">Forecast vs Actual (Last 14 Days)</span>
          </div>
          <p className="chart-description">
            Day-1 predictions from each source compared against actual sensor readings.
            The closer a forecaster's line is to the actual (black), the more accurate it is.
          </p>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={comparisonChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
                <XAxis 
                  dataKey="date" 
                  stroke="#6b7280" 
                  fontSize={11}
                  tickFormatter={v => {
                    if (!v) return '';
                    const parts = v.split('-');
                    return `${parts[2]}/${parts[1]}`;
                  }}
                />
                <YAxis 
                  stroke="#6b7280" 
                  fontSize={11} 
                  domain={['auto', 'auto']}
                  label={{ value: '°C', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 11 }}
                  width={40}
                />
                <Tooltip
                  contentStyle={{ background: '#1e2130', border: '1px solid #2a2d3a', borderRadius: 8 }}
                  labelFormatter={v => {
                    if (!v) return '';
                    const parts = v.split('-');
                    return `${parts[2]}-${parts[1]}-${parts[0]}`;
                  }}
                  formatter={(value, name) => [`${value?.toFixed(1)}°C`, SOURCE_NAMES[name] || name]}
                />
                <Legend 
                  wrapperStyle={{ fontSize: 12 }}
                  formatter={(value) => SOURCE_NAMES[value] || value}
                />
                <Line 
                  type="monotone" 
                  dataKey="actual" 
                  stroke="#ffffff" 
                  strokeWidth={3}
                  dot={{ fill: '#ffffff', r: 3 }}
                  name="Actual"
                />
                {Object.keys(SOURCE_COLORS).map(source => (
                  <Line 
                    key={source}
                    type="monotone" 
                    dataKey={source} 
                    stroke={SOURCE_COLORS[source]}
                    strokeWidth={2}
                    strokeDasharray={source === 'thermosense' ? '0' : '5 5'}
                    dot={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Collection status */}
      <div className="card" style={{ marginTop: 24 }}>
        <div className="card-header">
          <span className="card-title">
            <Info style={{ width: 14, height: 14, marginRight: 6, verticalAlign: -2 }} />
            Collection Status
          </span>
        </div>
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Source</th>
                <th>Last Collected</th>
                <th>Successes</th>
                <th>Failures</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(status?.sources || {}).map(([source, info]) => (
                <tr key={source}>
                  <td style={{ fontWeight: 600, color: SOURCE_COLORS[source] }}>
                    {SOURCE_NAMES[source] || source}
                  </td>
                  <td className="mono" style={{ fontSize: '0.8rem' }}>
                    {info.last_collected ? new Date(info.last_collected).toLocaleString() : '—'}
                  </td>
                  <td className="mono" style={{ color: 'var(--success)' }}>{info.successes}</td>
                  <td className="mono" style={{ color: info.failures > 0 ? 'var(--danger)' : 'inherit' }}>
                    {info.failures}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {status?.date_range && (
          <p style={{ marginTop: 12, fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            Data range: {status.date_range.earliest} to {status.date_range.latest}
          </p>
        )}
      </div>
    </>
  );
}
