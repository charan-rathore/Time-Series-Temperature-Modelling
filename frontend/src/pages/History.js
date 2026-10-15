import React, { useEffect, useState, useCallback } from "react";
import { RefreshCw, Download, Info } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Brush,
} from "recharts";
import { api } from "../api";

function getDefaultStart() {
  return "2024-06-12";
}

function getDefaultEnd() {
  return "2024-07-11";
}

export default function History() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [start, setStart] = useState(getDefaultStart);
  const [end, setEnd] = useState(getDefaultEnd);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getHistory(start, end);
      setData(res);
    } catch (e) {
      setError(e.message);
      setData(null);
    }
    setLoading(false);
  }, [start, end]);

  useEffect(() => {
    load();
  }, [load]);

  const chartData =
    data?.records?.map((r) => ({
      date: r.date,
      temp: r.actual_temp_c,
      api: r.api_temp_c,
      bias: r.api_bias,
      humidity: r.humidity_pct,
      pressure: r.pressure_hpa,
    })) || [];

  const exportCSV = () => {
    if (!chartData.length) return;
    const headers = [
      "date",
      "temp_c",
      "api_temp_c",
      "api_bias",
      "humidity_pct",
      "pressure_hpa",
    ];
    const rows = chartData.map((r) =>
      headers.map((h) => r[h === "temp_c" ? "temp" : h] ?? "").join(","),
    );
    const csv = [headers.join(","), ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `thermosense_history_${start}_${end}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <div
        className="page-header"
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div>
          <h2>Historical Data</h2>
          <p>
            {data
              ? `${data.total_records} records · ${data.location}`
              : "Loading…"}
          </p>
        </div>
        <div className="btn-group">
          <button className="btn btn-secondary btn-sm" onClick={exportCSV}>
            <Download /> Export CSV
          </button>
          <button
            className="btn btn-secondary btn-sm"
            onClick={load}
            aria-label="Refresh"
          >
            <RefreshCw />
          </button>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <div className="form-row">
          <div className="form-group" style={{ flex: 1 }}>
            <label className="form-label" htmlFor="history-start">
              Start date
            </label>
            <input
              id="history-start"
              type="date"
              className="form-input"
              value={start}
              onChange={(e) => setStart(e.target.value)}
            />
          </div>
          <div className="form-group" style={{ flex: 1 }}>
            <label className="form-label" htmlFor="history-end">
              End date
            </label>
            <input
              id="history-end"
              type="date"
              className="form-input"
              value={end}
              onChange={(e) => setEnd(e.target.value)}
            />
          </div>
          <button
            className="btn btn-primary"
            onClick={load}
            style={{ marginBottom: 16 }}
          >
            <RefreshCw /> Load
          </button>
        </div>
      </div>

      {error && (
        <div className="form-error" role="alert">
          History unavailable: {error}. Use Load to try again.
        </div>
      )}
      {loading ? (
        <div className="loading-state">
          <div className="spinner" />
          <span>Fetching history…</span>
        </div>
      ) : (
        <>
          {/* Temperature chart */}
          <div className="card" style={{ marginBottom: 24 }}>
            <div className="card-header">
              <span className="card-title">Temperature Timeline</span>
            </div>
            <p className="chart-description">
              Daily local/reference readings and Open-Meteo estimates, in °C.
              The dashed line is the API estimate; bias is the signed difference
              (reference minus API). Missing readings are not connected.
            </p>
            <div className="chart-container chart-container-lg">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#dce2d7" />
                  <XAxis
                    dataKey="date"
                    stroke="#647268"
                    fontSize={10}
                    tickFormatter={(v) => {
                      if (!v) return "";
                      const parts = v.split("-");
                      return `${parts[2]}-${parts[1]}`;
                    }}
                    interval="preserveStartEnd"
                    label={{
                      value: "Date (DD-MM)",
                      position: "insideBottom",
                      offset: -5,
                      fill: "#647268",
                      fontSize: 11,
                    }}
                    height={50}
                  />
                  <YAxis
                    stroke="#647268"
                    fontSize={11}
                    domain={["auto", "auto"]}
                    label={{
                      value: "Temperature (°C)",
                      angle: -90,
                      position: "insideLeft",
                      fill: "#647268",
                      fontSize: 11,
                      dx: -5,
                    }}
                    width={65}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#ffffff",
                      border: "1px solid #dce2d7",
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                    labelStyle={{ color: "#53645b" }}
                    labelFormatter={(v) => {
                      if (!v) return "";
                      const parts = v.split("-");
                      return `${parts[2]}-${parts[1]}-${parts[0]}`;
                    }}
                    formatter={(value, name) => {
                      if (value == null) return ["-", name];
                      return [`${value}°C`, name];
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
                  <Line
                    isAnimationActive={false}
                    type="linear"
                    dataKey="temp"
                    name="Local / reference °C"
                    stroke="#276349"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    isAnimationActive={false}
                    type="linear"
                    dataKey="api"
                    strokeDasharray="5 4"
                    name="API (Open-Meteo) °C"
                    stroke="#ab612c"
                    strokeWidth={1.5}
                    dot={false}
                  />
                  <Line
                    isAnimationActive={false}
                    type="linear"
                    dataKey="bias"
                    name="Bias (Sensor − API) °C"
                    stroke="#b14035"
                    strokeWidth={1}
                    dot={false}
                  />
                  {chartData.length > 60 && (
                    <Brush
                      dataKey="date"
                      height={24}
                      stroke="#276349"
                      fill="#f3f3ed"
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Weather conditions chart */}
          <div className="card" style={{ marginBottom: 24 }}>
            <div className="card-header">
              <span className="card-title">Weather Conditions</span>
            </div>
            <p className="chart-description">
              Open-Meteo weather context: humidity on the left axis (%) and
              pressure on the right (hPa). These use separate scales.
            </p>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#dce2d7" />
                  <XAxis
                    dataKey="date"
                    stroke="#647268"
                    fontSize={10}
                    tickFormatter={(v) => {
                      if (!v) return "";
                      const parts = v.split("-");
                      return `${parts[2]}-${parts[1]}`;
                    }}
                    interval="preserveStartEnd"
                    label={{
                      value: "Date (DD-MM)",
                      position: "insideBottom",
                      offset: -5,
                      fill: "#647268",
                      fontSize: 11,
                    }}
                    height={50}
                  />
                  <YAxis
                    yAxisId="hum"
                    stroke="#346c80"
                    fontSize={11}
                    orientation="left"
                    domain={[0, 100]}
                    label={{
                      value: "Humidity (%)",
                      angle: -90,
                      position: "insideLeft",
                      fill: "#346c80",
                      fontSize: 11,
                      dx: -5,
                    }}
                    width={60}
                  />
                  <YAxis
                    yAxisId="pres"
                    stroke="#626fa4"
                    fontSize={11}
                    orientation="right"
                    domain={["auto", "auto"]}
                    label={{
                      value: "Pressure (hPa)",
                      angle: 90,
                      position: "insideRight",
                      fill: "#626fa4",
                      fontSize: 11,
                      dx: 5,
                    }}
                    width={70}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#ffffff",
                      border: "1px solid #dce2d7",
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                    labelStyle={{ color: "#53645b" }}
                    labelFormatter={(v) => {
                      if (!v) return "";
                      const parts = v.split("-");
                      return `${parts[2]}-${parts[1]}-${parts[0]}`;
                    }}
                    formatter={(value, name) => {
                      if (value == null) return ["-", name];
                      if (name.includes("Humidity")) return [`${value}%`, name];
                      if (name.includes("Pressure"))
                        return [`${value} hPa`, name];
                      return [value, name];
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
                  <Line
                    isAnimationActive={false}
                    yAxisId="hum"
                    type="linear"
                    dataKey="humidity"
                    name="Humidity (%)"
                    stroke="#346c80"
                    strokeWidth={1.5}
                    dot={false}
                  />
                  <Line
                    isAnimationActive={false}
                    yAxisId="pres"
                    type="linear"
                    dataKey="pressure"
                    name="Pressure (hPa)"
                    stroke="#626fa4"
                    strokeWidth={1.5}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Data table */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Records (showing last 50)</span>
            </div>
            <div className="info-box" style={{ marginBottom: 16 }}>
              <Info
                style={{ width: 14, height: 14, flexShrink: 0, marginTop: 2 }}
              />
              <span style={{ fontSize: "0.78rem" }}>
                <strong>Bias</strong> = Sensor − API temperature. Positive bias
                (red) means the local sensor reads warmer than the API grid
                estimate. <strong>Source</strong> = "Sensor" if a local reading
                exists, "API" if only the Open-Meteo value is available for that
                date.
              </span>
            </div>
            <p className="quiet-label" style={{ marginBottom: 8 }}>
              Scroll the records table horizontally on a small screen.
            </p>
            <div
              className="table-wrapper"
              tabIndex={0}
              role="region"
              aria-label="Historical records"
            >
              <table>
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Sensor °C</th>
                    <th>API °C</th>
                    <th>Bias (°C)</th>
                    <th>Humidity %</th>
                    <th>Pressure hPa</th>
                    <th>Source</th>
                  </tr>
                </thead>
                <tbody>
                  {(data?.records || [])
                    .slice(-50)
                    .reverse()
                    .map((r) => (
                      <tr key={r.date}>
                        <td className="mono">
                          {r.date
                            ? (() => {
                                const [y, m, d] = r.date.split("-");
                                return `${d}-${m}-${y}`;
                              })()
                            : "-"}
                        </td>
                        <td className="mono">{r.actual_temp_c ?? "-"}</td>
                        <td className="mono">{r.api_temp_c ?? "-"}</td>
                        <td
                          className="mono"
                          style={{
                            color:
                              r.api_bias > 0
                                ? "var(--danger)"
                                : r.api_bias < 0
                                  ? "var(--accent)"
                                  : undefined,
                          }}
                        >
                          {r.api_bias != null
                            ? (r.api_bias > 0 ? "+" : "") +
                              r.api_bias.toFixed(2)
                            : "-"}
                        </td>
                        <td className="mono">{r.humidity_pct ?? "-"}</td>
                        <td className="mono">{r.pressure_hpa ?? "-"}</td>
                        <td>
                          <span
                            className={`badge ${r.is_sensor_reading ? "badge-success" : "badge-info"}`}
                          >
                            {r.is_sensor_reading ? "Sensor" : "API"}
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
