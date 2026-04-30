import React, { useEffect, useState, useCallback } from 'react';
import { RefreshCw, Download } from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Brush,
} from 'recharts';
import { api } from '../api';

export default function History() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [start, setStart] = useState('2024-06-01');
  const [end, setEnd] = useState(new Date().toISOString().slice(0, 10));

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.getHistory(start, end);
      setData(res);
    } catch { /* noop */ }
    setLoading(false);
  }, [start, end]);

  useEffect(() => { load(); }, [load]);

  const chartData = data?.records?.map(r => ({
    date: r.date,
    temp: r.actual_temp_c,
    api: r.api_temp_c,
    bias: r.api_bias,
    humidity: r.humidity_pct,
    pressure: r.pressure_hpa,
  })) || [];

  const exportCSV = () => {
    if (!chartData.length) return;
    const headers = ['date', 'temp_c', 'api_temp_c', 'api_bias', 'humidity_pct', 'pressure_hpa'];
    const rows = chartData.map(r => headers.map(h => r[h === 'temp_c' ? 'temp' : h] ?? '').join(','));
    const csv = [headers.join(','), ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `thermosense_history_${start}_${end}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2>Historical Data</h2>
          <p>{data ? `${data.total_records} records · ${data.location}` : 'Loading…'}</p>
        </div>
        <div className="btn-group">
          <button className="btn btn-secondary btn-sm" onClick={exportCSV}><Download /> Export CSV</button>
          <button className="btn btn-secondary btn-sm" onClick={load}><RefreshCw /></button>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <div className="form-row">
          <div className="form-group" style={{ flex: 1 }}>
            <label className="form-label">Start Date</label>
            <input type="date" className="form-input" value={start} onChange={e => setStart(e.target.value)} />
          </div>
          <div className="form-group" style={{ flex: 1 }}>
            <label className="form-label">End Date</label>
            <input type="date" className="form-input" value={end} onChange={e => setEnd(e.target.value)} />
          </div>
          <button className="btn btn-primary" onClick={load} style={{ marginBottom: 16 }}>
            <RefreshCw /> Load
          </button>
        </div>
      </div>

      {loading ? (
        <div className="loading-state"><div className="spinner" /><span>Fetching history…</span></div>
      ) : (
        <>
          {/* Temperature chart */}
          <div className="card" style={{ marginBottom: 24 }}>
            <div className="card-header"><span className="card-title">Temperature Timeline</span></div>
            <div className="chart-container chart-container-lg">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
                  <XAxis dataKey="date" stroke="#6b7280" fontSize={10} tickFormatter={v => v?.slice(5)} interval="preserveStartEnd" />
                  <YAxis stroke="#6b7280" fontSize={11} domain={['auto', 'auto']} />
                  <Tooltip
                    contentStyle={{ background: '#1e2130', border: '1px solid #2a2d3a', borderRadius: 8, fontSize: 12 }}
                    labelStyle={{ color: '#9aa0b0' }}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line type="monotone" dataKey="temp" name="Sensor °C" stroke="#3b82f6" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="api" name="API °C" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
                  <Line type="monotone" dataKey="bias" name="Bias °C" stroke="#ef4444" strokeWidth={1} dot={false} />
                  {chartData.length > 60 && <Brush dataKey="date" height={24} stroke="#3b82f6" fill="#14161e" />}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Weather conditions chart */}
          <div className="card" style={{ marginBottom: 24 }}>
            <div className="card-header"><span className="card-title">Weather Conditions</span></div>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
                  <XAxis dataKey="date" stroke="#6b7280" fontSize={10} tickFormatter={v => v?.slice(5)} interval="preserveStartEnd" />
                  <YAxis yAxisId="hum" stroke="#06b6d4" fontSize={11} orientation="left" domain={[0, 100]} />
                  <YAxis yAxisId="pres" stroke="#a78bfa" fontSize={11} orientation="right" domain={['auto', 'auto']} />
                  <Tooltip
                    contentStyle={{ background: '#1e2130', border: '1px solid #2a2d3a', borderRadius: 8, fontSize: 12 }}
                    labelStyle={{ color: '#9aa0b0' }}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line yAxisId="hum" type="monotone" dataKey="humidity" name="Humidity %" stroke="#06b6d4" strokeWidth={1.5} dot={false} />
                  <Line yAxisId="pres" type="monotone" dataKey="pressure" name="Pressure hPa" stroke="#a78bfa" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Data table */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Records (showing last 50)</span>
            </div>
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Temp °C</th>
                    <th>API °C</th>
                    <th>Bias</th>
                    <th>Humidity %</th>
                    <th>Pressure hPa</th>
                    <th>Source</th>
                  </tr>
                </thead>
                <tbody>
                  {(data?.records || []).slice(-50).reverse().map(r => (
                    <tr key={r.date}>
                      <td className="mono">{r.date}</td>
                      <td className="mono">{r.actual_temp_c ?? '—'}</td>
                      <td className="mono">{r.api_temp_c ?? '—'}</td>
                      <td className="mono" style={{ color: r.api_bias > 0 ? 'var(--danger)' : r.api_bias < 0 ? 'var(--accent)' : undefined }}>
                        {r.api_bias != null ? (r.api_bias > 0 ? '+' : '') + r.api_bias.toFixed(2) : '—'}
                      </td>
                      <td className="mono">{r.humidity_pct ?? '—'}</td>
                      <td className="mono">{r.pressure_hpa ?? '—'}</td>
                      <td>
                        <span className={`badge ${r.is_sensor_reading ? 'badge-success' : 'badge-info'}`}>
                          {r.is_sensor_reading ? 'Sensor' : 'API'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </>
  );
}
