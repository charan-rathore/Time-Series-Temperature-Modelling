import React, { useEffect, useState, useCallback } from "react";
import { ArrowUpRight, RefreshCw, AlertCircle, ArrowRight } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { api } from "../api";
import { temperature, shortDate } from "../presentation";

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
  useEffect(() => {
    load();
  }, [load]);
  if (loading)
    return (
      <div className="loading-state" role="status">
        <div className="spinner" />
        Reading the station…
      </div>
    );
  if (error)
    return (
      <div className="empty-state" role="alert">
        <AlertCircle />
        <h2>Station unavailable</h2>
        <p>{error}</p>
        <button className="btn btn-secondary" onClick={load}>
          <RefreshCw />
          Try again
        </button>
      </div>
    );
  const tomorrow = forecast?.forecasts?.[0];
  const chartData = (history?.records || [])
    .slice(-30)
    .map((r) => ({ date: r.date, temp: r.actual_temp_c, api: r.api_temp_c }));
  const model = forecast?.model_used || "Unavailable";
  const untrained = !forecast || ["placeholder", "climatology", "fallback", "Unavailable"].includes(model);
  const modelLabel = model === "climatology" || model === "fallback" ? "last recorded reading" : model;
  // Show the error of the model with the strongest held-out evidence,
  // matching how the API picks what it serves.
  const training = status?.last_training_results || {};
  const rmse = Object.values(training)
    .map((m) => m?.day1)
    .filter((d) => typeof d?.rmse === "number" && (d?.n_observations ?? 0) >= 7)
    .sort((a, b) => a.rmse - b.rmse)[0]?.rmse;
  return (
    <>
      <header className="page-header">
        <div>
          <p className="eyebrow">The observation room</p>
          <h2>Station overview</h2>
          <p className="page-intro">
            ThermoSense pairs a small temperature sensor at home with the
            weather forecast and learns the difference between the two, so the
            number you see is for <em>your street</em>, not the nearest
            airport. For anyone whose plans depend on the weather right
            outside their window.
          </p>
          <p className="page-location">
            {forecast?.location ||
              history?.location ||
              "Local temperature station"}
          </p>
        </div>
        <button className="btn btn-secondary" onClick={load}>
          <RefreshCw />
          Refresh
        </button>
      </header>
      <div className="overview-grid">
        <section
          className="temperature-hero"
          aria-label="Tomorrow's temperature forecast"
        >
          <div className="hero-top">
            <span className="eyebrow">
              Tomorrow · {shortDate(tomorrow?.date)}
            </span>
            <span className={`source-tag ${untrained ? "source-tag-provisional" : "source-tag-trained"}`}>{modelLabel}</span>
          </div>
          <div className="hero-temperature">
            {temperature(tomorrow?.predicted_temp_c)}
            <span>°C</span>
          </div>
          <p className="hero-caption">Expected <em>tomorrow evening</em>, around 9 PM in Bangalore</p>
          <div className="hero-bottom">
            <span>
              {tomorrow
                ? `${temperature(tomorrow.lower_bound_c)} to ${temperature(tomorrow.upper_bound_c)} °C reported range`
                : "Forecast unavailable"}
            </span>
            <button
              className="text-button"
              onClick={() => onNavigate("forecast")}
            >
              Forecast details <ArrowUpRight size={16} />
            </button>
          </div>
          {untrained && (
            <p className="hero-caveat">
              {model === "climatology" || model === "fallback"
                ? "Not a trained forecast yet: this repeats the latest recorded temperature, shown until the pipeline has enough real readings to learn from."
                : "Placeholder estimate, not a trained model forecast."}
            </p>
          )}
        </section>
        <section className="station-ledger" aria-label="Station record">
          <span className="eyebrow">Station record</span>
          <dl>
            <div>
              <dt>Daily observations</dt>
              {status?.data_rows ? (
                <dd>{status.data_rows.toLocaleString()}</dd>
              ) : (
                <dd className="dd-pending">none yet</dd>
              )}
            </div>
            <div>
              <dt>Model artifacts</dt>
              {status?.models_available?.length ? (
                <dd>{status.models_available.length}</dd>
              ) : (
                <dd className="dd-pending">none yet</dd>
              )}
            </div>
            <div>
              <dt>Training RMSE · day 1</dt>
              <dd>
                {temperature(rmse)} <small>°C</small>
              </dd>
            </div>
          </dl>
          <p>
            {status?.data_available
              ? `Dataset: ${status.data_date_range || "date range unavailable"}`
              : "No readings stored yet. The daily pull fills this in; a backfill from Pipeline seeds it."}
          </p>
          <p className="ledger-note">
            Training scores are not a guarantee of future accuracy.
          </p>
        </section>
      </div>
      <section className="card trend-card">
        <div className="card-header">
          <div>
            <span className="eyebrow">Measured & estimated</span>
            <h3>Temperature record</h3>
          </div>
          <button className="text-button" onClick={() => onNavigate("history")}>
            Explore history <ArrowRight size={15} />
          </button>
        </div>
        <p className="chart-description">
          Latest {chartData.length} stored daily records. Local/reference
          temperature alongside the Open-Meteo grid estimate.
        </p>
        {chartData.length ? (
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={chartData}
                margin={{ top: 16, right: 14, left: 0, bottom: 4 }}
              >
                <CartesianGrid vertical={false} stroke="var(--border)" />
                <XAxis
                  dataKey="date"
                  tickFormatter={shortDate}
                  tickLine={false}
                  axisLine={false}
                  minTickGap={36}
                  tick={{ fill: "var(--text-muted)", fontSize: 12 }}
                />
                <YAxis
                  domain={["auto", "auto"]}
                  tickLine={false}
                  axisLine={false}
                  width={42}
                  tick={{ fill: "var(--text-muted)", fontSize: 12 }}
                  unit="°"
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--bg-card)",
                    border: "1px solid var(--border)",
                    borderRadius: 4,
                  }}
                  labelFormatter={shortDate}
                  formatter={(v, n) => [`${temperature(v)} °C`, n]}
                />
                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 18 }} />
                <Line
                  isAnimationActive={false}
                  type="linear"
                  dataKey="temp"
                  name="Local / reference"
                  stroke="var(--chart-local)"
                  strokeWidth={2.5}
                  dot={false}
                />
                <Line
                  isAnimationActive={false}
                  type="linear"
                  dataKey="api"
                  name="Open-Meteo estimate"
                  stroke="var(--chart-api)"
                  strokeDasharray="5 4"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="empty-state">
            <p>No history available. Check the dataset in Pipeline.</p>
          </div>
        )}
        <div className="chart-footnote">
          Daily reference: 9 PM local time. A gap in a line means a reading is
          missing, not zero.
        </div>
      </section>
      <section className="outlook-section">
        <div className="section-heading">
          <div>
            <span className="eyebrow">Looking ahead</span>
            <h3>The next three evenings</h3>
          </div>
          <span className="quiet-label">Not hourly or daily averages</span>
        </div>
        {forecast?.forecasts?.length ? (
          <div className="outlook-list">
            {forecast.forecasts.map((f) => (
              <button
                className="outlook-item"
                key={f.horizon_days}
                onClick={() => onNavigate("forecast")}
              >
                <span className="outlook-day">
                  Day {f.horizon_days}
                  <small>{shortDate(f.date)}</small>
                </span>
                <strong>
                  {temperature(f.predicted_temp_c)}
                  <span>°C</span>
                </strong>
                <span className="outlook-range">
                  {temperature(f.lower_bound_c)} to{" "}
                  {temperature(f.upper_bound_c)} °C
                </span>
                <ArrowUpRight size={18} />
              </button>
            ))}
          </div>
        ) : (
          <p className="chart-description">
            Forecast unavailable. Open Forecast to retry.
          </p>
        )}
      </section>
    </>
  );
}
