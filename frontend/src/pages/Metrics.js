import React, { useEffect, useState, useCallback } from 'react';
import { RefreshCw, Trophy, Target, Info } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
} from 'recharts';
import { api } from '../api';

const MODEL_COLORS = {
  sarima: '#3b82f6',
  lgbm: '#22c55e',
  tft: '#a855f7',
  ensemble: '#f59e0b',
};

const METRIC_DEFS = {
  MAE: {
    label: 'MAE (Mean Absolute Error)',
    desc: 'Average size of prediction errors in °C. Lower is better. If MAE = 1.0, predictions are off by ~1°C on average.',
  },
  RMSE: {
    label: 'RMSE (Root Mean Squared Error)',
    desc: 'Like MAE but penalises large errors more. If RMSE is much higher than MAE, the model occasionally makes big mistakes.',
  },
  MAPE: {
    label: 'MAPE (Mean Absolute % Error)',
    desc: 'Prediction error as a percentage of the actual value. 3% means predictions are typically within 3% of the true temperature.',
  },
  Skill: {
    label: 'Skill Score',
    desc: 'How much better this model is vs a naive "use yesterday\'s temp" baseline. 1.0 = perfect, 0 = no better than naive, negative = worse.',
  },
  Coverage: {
    label: '90% Coverage',
    desc: 'Fraction of actual values that fell inside the predicted 90% confidence interval. Ideally ≥ 90%. Only available for models that produce intervals.',
  },
};

export default function Metrics() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.getMetrics(30);
      setData(res);
    } catch { /* noop */ }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  if (loading) {
    return <div className="loading-state"><div className="spinner" /><span>Loading metrics…</span></div>;
  }

  if (!data || !data.models || Object.keys(data.models).length === 0 ||
      data.models.no_models_trained) {
    return (
      <div className="empty-state">
        <Target style={{ width: 48, height: 48 }} />
        <p>No model metrics available. Train models first.</p>
      </div>
    );
  }

  const models = Object.entries(data.models);
  const horizons = ['day1', 'day2', 'day3'];

  const rmseData = horizons.map(h => {
    const row = { horizon: h.replace('day', 'Day ') };
    models.forEach(([name, hdata]) => {
      row[name] = hdata[h]?.rmse ?? 0;
    });
    return row;
  });

  const maeData = horizons.map(h => {
    const row = { horizon: h.replace('day', 'Day ') };
    models.forEach(([name, hdata]) => {
      row[name] = hdata[h]?.mae ?? 0;
    });
    return row;
  });

  const radarData = models.map(([name, hdata]) => {
    const d1 = hdata.day1 || {};
    return {
      model: name,
      MAE: d1.mae ? Math.max(0, 1 - d1.mae / 3) * 100 : 0,
      RMSE: d1.rmse ? Math.max(0, 1 - d1.rmse / 3) * 100 : 0,
      Skill: d1.skill_score ? Math.max(0, d1.skill_score) * 100 : 0,
      Coverage: d1.coverage_90pct != null ? d1.coverage_90pct * 100 : 50,
      MAPE: d1.mape ? Math.max(0, 1 - d1.mape / 10) * 100 : 0,
    };
  });

  const radarMetrics = ['MAE', 'RMSE', 'Skill', 'Coverage', 'MAPE'];

  const bestModel = models.reduce((best, [name, hdata]) => {
    const rmse = hdata.day1?.rmse ?? Infinity;
    return rmse < best.rmse ? { name, rmse } : best;
  }, { name: '', rmse: Infinity });

  return (
    <>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2>Model Metrics</h2>
          <p>{data.location} · {models.length} model{models.length !== 1 ? 's' : ''} evaluated on held-out test data</p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={load}><RefreshCw /></button>
      </div>

      {/* Metric glossary */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-header">
          <span className="card-title"><Info style={{ width: 14, height: 14, marginRight: 6, verticalAlign: -2 }} /> Metric Definitions</span>
        </div>
        <div className="metric-glossary">
          {Object.entries(METRIC_DEFS).map(([key, { label, desc }]) => (
            <div key={key} className="metric-glossary-item">
              <strong>{label}</strong>
              <span>{desc}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Best model banner */}
      <div className="card" style={{ marginBottom: 24, background: 'var(--success-soft)', borderColor: 'var(--success)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <Trophy style={{ color: 'var(--success)', width: 24 }} />
          <div>
            <div style={{ fontWeight: 700, fontSize: '0.95rem' }}>Best Model: {bestModel.name}</div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Day-1 RMSE: {bestModel.rmse.toFixed(4)}°C — lowest next-day prediction error on test data
            </div>
          </div>
        </div>
      </div>

      {/* Metric cards per model */}
      <div className="card-grid" style={{ marginBottom: 24 }}>
        {models.map(([name, hdata]) => {
          const d1 = hdata.day1 || {};
          return (
            <div className="card" key={name}>
              <div className="card-header">
                <span className="card-title" style={{ color: MODEL_COLORS[name] || 'var(--text-secondary)' }}>
                  {name}
                </span>
                {name === bestModel.name && <span className="badge badge-success">Best</span>}
              </div>
              <div className="metric-bar">
                <span className="metric-bar-label" title={METRIC_DEFS.MAE.desc}>MAE</span>
                <div className="metric-bar-track">
                  <div className="metric-bar-fill" style={{
                    width: `${Math.min(100, (d1.mae || 0) / 3 * 100)}%`,
                    background: MODEL_COLORS[name] || '#3b82f6',
                  }} />
                </div>
                <span className="metric-bar-value">{d1.mae?.toFixed(3) ?? '—'}°C</span>
              </div>
              <div className="metric-bar">
                <span className="metric-bar-label" title={METRIC_DEFS.RMSE.desc}>RMSE</span>
                <div className="metric-bar-track">
                  <div className="metric-bar-fill" style={{
                    width: `${Math.min(100, (d1.rmse || 0) / 3 * 100)}%`,
                    background: MODEL_COLORS[name] || '#3b82f6',
                  }} />
                </div>
                <span className="metric-bar-value">{d1.rmse?.toFixed(3) ?? '—'}°C</span>
              </div>
              <div className="metric-bar">
                <span className="metric-bar-label" title={METRIC_DEFS.MAPE.desc}>MAPE</span>
                <div className="metric-bar-track">
                  <div className="metric-bar-fill" style={{
                    width: `${Math.min(100, (d1.mape || 0) / 10 * 100)}%`,
                    background: MODEL_COLORS[name] || '#3b82f6',
                  }} />
                </div>
                <span className="metric-bar-value">{d1.mape?.toFixed(2) ?? '—'}%</span>
              </div>
              <div className="metric-bar">
                <span className="metric-bar-label" title={METRIC_DEFS.Skill.desc}>Skill</span>
                <div className="metric-bar-track">
                  <div className="metric-bar-fill" style={{
                    width: `${Math.min(100, Math.max(0, (d1.skill_score || 0)) * 100)}%`,
                    background: (d1.skill_score || 0) > 0 ? 'var(--success)' : 'var(--danger)',
                  }} />
                </div>
                <span className="metric-bar-value">{d1.skill_score?.toFixed(3) ?? '—'}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* RMSE by horizon */}
      <div className="card-grid-2">
        <div className="card">
          <div className="card-header"><span className="card-title">RMSE by Horizon</span></div>
          <p className="chart-description">
            Prediction error (RMSE) for each forecast horizon. Day 1 = next-day forecast, Day 3 = 3 days ahead. 
            Error typically grows with longer horizons.
          </p>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={rmseData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
                <XAxis dataKey="horizon" stroke="#6b7280" fontSize={12} />
                <YAxis
                  stroke="#6b7280"
                  fontSize={11}
                  label={{ value: 'RMSE (°C)', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 11 }}
                  width={55}
                />
                <Tooltip
                  contentStyle={{ background: '#1e2130', border: '1px solid #2a2d3a', borderRadius: 8 }}
                  formatter={(value) => [`${value.toFixed(4)}°C`]}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {models.map(([name]) => (
                  <Bar key={name} dataKey={name} fill={MODEL_COLORS[name] || '#666'} radius={[4, 4, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="card-header"><span className="card-title">MAE by Horizon</span></div>
          <p className="chart-description">
            Average absolute error (MAE) for each forecast horizon. Unlike RMSE, 
            MAE treats all errors equally without extra penalty for large misses.
          </p>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={maeData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
                <XAxis dataKey="horizon" stroke="#6b7280" fontSize={12} />
                <YAxis
                  stroke="#6b7280"
                  fontSize={11}
                  label={{ value: 'MAE (°C)', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 11 }}
                  width={55}
                />
                <Tooltip
                  contentStyle={{ background: '#1e2130', border: '1px solid #2a2d3a', borderRadius: 8 }}
                  formatter={(value) => [`${value.toFixed(4)}°C`]}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {models.map(([name]) => (
                  <Bar key={name} dataKey={name} fill={MODEL_COLORS[name] || '#666'} radius={[4, 4, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Radar chart */}
      {radarData.length > 0 && (
        <div className="card" style={{ marginTop: 24 }}>
          <div className="card-header"><span className="card-title">Model Comparison Radar (Day-1)</span></div>
          <p className="chart-description">
            Normalised Day-1 scores on a 0–100 scale (higher = better). For MAE/RMSE/MAPE, 
            the value is inverted so that lower error appears as a higher score. Allows quick visual 
            comparison of model strengths across different criteria.
          </p>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarMetrics.map(m => {
                const row = { metric: m };
                radarData.forEach(rd => { row[rd.model] = rd[m]; });
                return row;
              })}>
                <PolarGrid stroke="#2a2d3a" />
                <PolarAngleAxis dataKey="metric" stroke="#9aa0b0" fontSize={11} />
                <PolarRadiusAxis stroke="#2a2d3a" fontSize={10} domain={[0, 100]} />
                {radarData.map(rd => (
                  <Radar key={rd.model} name={rd.model} dataKey={rd.model}
                    stroke={MODEL_COLORS[rd.model] || '#666'}
                    fill={MODEL_COLORS[rd.model] || '#666'}
                    fillOpacity={0.15} strokeWidth={2}
                  />
                ))}
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Tooltip contentStyle={{ background: '#1e2130', border: '1px solid #2a2d3a', borderRadius: 8 }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Full comparison table */}
      <div className="card" style={{ marginTop: 24 }}>
        <div className="card-header"><span className="card-title">Full Results Table</span></div>
        <p className="chart-description" style={{ marginBottom: 12 }}>
          Complete metrics for every model and forecast horizon, evaluated on the held-out test set.
          All error values are in °C. Skill Score is relative to a naive baseline (yesterday's temp). 
          Coverage shows the fraction of test points inside the 90% prediction interval.
        </p>
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Horizon</th>
                <th title="Mean Absolute Error (°C)">MAE</th>
                <th title="Root Mean Squared Error (°C)">RMSE</th>
                <th title="Mean Absolute Percentage Error">MAPE %</th>
                <th title="Skill vs naive baseline (1.0 = perfect)">Skill Score</th>
                <th title="Fraction inside 90% prediction interval">Coverage 90%</th>
              </tr>
            </thead>
            <tbody>
              {models.flatMap(([name, hdata]) =>
                horizons.filter(h => hdata[h]).map(h => {
                  const m = hdata[h];
                  return (
                    <tr key={`${name}-${h}`}>
                      <td style={{ fontWeight: 600, color: MODEL_COLORS[name] }}>{name}</td>
                      <td>{h.replace('day', 'Day ')}</td>
                      <td className="mono">{m.mae?.toFixed(4)}</td>
                      <td className="mono">{m.rmse?.toFixed(4)}</td>
                      <td className="mono">{m.mape?.toFixed(2)}</td>
                      <td className="mono">{m.skill_score?.toFixed(4)}</td>
                      <td className="mono">{m.coverage_90pct != null ? (m.coverage_90pct * 100).toFixed(1) + '%' : '—'}</td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
