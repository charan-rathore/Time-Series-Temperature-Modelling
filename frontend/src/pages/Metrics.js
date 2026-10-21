import React, { useEffect, useState, useCallback } from "react";
import { RefreshCw, Target, AlertCircle } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { api } from "../api";
const COLORS = {
  sarima: "#346c80",
  lgbm: "#276349",
  tft: "#626fa4",
  ensemble: "#ab612c",
};
const HORIZONS = ["day1", "day2", "day3"];
const number = (v, d = 3) =>
  typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "Not available";
export default function Metrics() {
  const [data, setData] = useState(null),
    [loading, setLoading] = useState(true),
    [error, setError] = useState(null);
  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await api.getMetrics(30));
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
        Reading model results…
      </div>
    );
  if (error)
    return (
      <div className="empty-state" role="alert">
        <AlertCircle />
        <h2>Results unavailable</h2>
        <p>{error}</p>
        <button className="btn btn-secondary" onClick={load}>
          <RefreshCw />
          Try again
        </button>
      </div>
    );
  const models = Object.entries(data?.models || {}).filter(
    ([name]) => name !== "no_models_trained",
  );
  if (!models.length)
    return (
      <div className="empty-state">
        <Target />
        <h2>No model results yet</h2>
        <p>Train a model in Pipeline to see its recorded evaluation.</p>
      </div>
    );
  // A "best" badge needs evidence: at least a week of paired observations,
  // otherwise a lucky single day would wear the crown.
  const best = models.reduce(
    (b, [name, h]) =>
      typeof h.day1?.rmse === "number" &&
      Number.isFinite(h.day1.rmse) &&
      (h.day1?.n_observations ?? 0) >= 7 &&
      h.day1.rmse < b.rmse
        ? { name, rmse: h.day1.rmse }
        : b,
    { name: null, rmse: Infinity },
  );
  return (
    <>
      <header className="page-header">
        <div>
          <p className="eyebrow">Evaluation notebook</p>
          <h2>Model metrics</h2>
          <p>
            {data.location} · {models.length} models · how far off each one was
            on past days it had not seen. Lower is better.
          </p>
        </div>
        <button className="btn btn-secondary" onClick={load}>
          <RefreshCw />
          Refresh
        </button>
      </header>
      <div className="info-box" style={{ marginBottom: 24 }}>
        These are recorded model results, not a live accuracy guarantee. Legacy
        skill scores may be withheld. Use Leaderboard for stored
        forecast-to-observation comparisons.
      </div>
      <div className="info-box" style={{ marginBottom: 24 }}>
        Honest yardstick: in a stricter rolling backtest the best model is
        only level with guessing that tomorrow equals today - day-1 error
        1.14 °C vs 1.13 °C on average (MAE) and 1.50 °C vs 1.54 °C on
        squared error (RMSE), no real edge yet. Treat the figures below as
        recorded results, not proof of beating a weather app.
      </div>
      <div className="card-grid">
        {models.map(([name, h]) => (
          <section className="card" key={name}>
            <div className="card-header">
              <span className="card-title" style={{ color: COLORS[name] }}>
                {name}
              </span>
              {best.name === name && (
                <span className="badge badge-success">Lowest day-1 RMSE</span>
              )}
            </div>
            <div className="stat-value">
              {number(h.day1?.rmse, 2)}
              {typeof h.day1?.rmse === "number" && (
                <small style={{ fontSize: 16 }}> °C</small>
              )}
            </div>
            <p className="stat-label">Next-day root mean squared error</p>
            <div className="metric-pair">
              <span>
                MAE <strong>{number(h.day1?.mae, 2)} °C</strong>
              </span>
              <span>
                Paired observations{" "}
                <strong>{h.day1?.n_observations ?? "Not available"}</strong>
              </span>
            </div>
          </section>
        ))}
      </div>
      <div className="card-grid-2">
        {[
          ["rmse", "RMSE by forecast day"],
          ["mae", "MAE by forecast day"],
        ].map(([key, label]) => (
          <section className="card" key={key}>
            <div className="card-header">
              <h3>{label}</h3>
            </div>
            <p className="chart-description">
              Error in °C, lower is better. Missing values are not plotted as
              zero.
            </p>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={HORIZONS.map((h) => {
                    const row = { horizon: h.replace("day", "Day ") };
                    models.forEach(([name, m]) => {
                      row[name] = m[h]?.[key] ?? null;
                    });
                    return row;
                  })}
                  margin={{ top: 16, right: 8, left: 0, bottom: 0 }}
                >
                  <CartesianGrid vertical={false} stroke="var(--border)" />
                  <XAxis
                    dataKey="horizon"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: "var(--text-muted)", fontSize: 12 }}
                  />
                  <YAxis
                    axisLine={false}
                    tickLine={false}
                    width={38}
                    tick={{ fill: "var(--text-muted)", fontSize: 12 }}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "var(--bg-card)",
                      border: "1px solid var(--border)",
                      borderRadius: 4,
                    }}
                    formatter={(v) => [`${number(v)} °C`]}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  {models.map(([name]) => (
                    <Bar
                      isAnimationActive={false}
                      key={name}
                      dataKey={name}
                      fill={COLORS[name] || "#647268"}
                      maxBarSize={36}
                      radius={[2, 2, 0, 0]}
                    />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>
        ))}
      </div>
      <section className="card">
        <div className="card-header">
          <h3>The full record</h3>
          <span className="quiet-label">
            Actual units. No composite scores.
          </span>
        </div>
        <p className="chart-description">
          All models and horizons. MAPE and coverage are percentages; MAE and
          RMSE are °C. Scroll the table horizontally on a small screen.
        </p>
        <div
          className="table-wrapper"
          tabIndex={0}
          role="region"
          aria-label="All recorded metrics"
        >
          <table>
            <thead>
              <tr>
                <th scope="col">Model / day</th>
                <th scope="col">MAE °C</th>
                <th scope="col">RMSE °C</th>
                <th scope="col">MAPE %</th>
                <th scope="col">Skill score</th>
                <th scope="col">Coverage %</th>
                <th scope="col">Observations</th>
              </tr>
            </thead>
            <tbody>
              {models.flatMap(([name, h]) =>
                HORIZONS.filter((day) => h[day]).map((day) => {
                  const m = h[day];
                  return (
                    <tr key={`${name}-${day}`}>
                      <th scope="row">
                        {name} / {day.replace("day", "Day ")}
                      </th>
                      <td className="mono">{number(m.mae)}</td>
                      <td className="mono">{number(m.rmse)}</td>
                      <td className="mono">{number(m.mape, 2)}</td>
                      <td className="mono">
                        {number(m.skill_score)}
                        {m.skill_score == null && m.skill_score_status && (
                          <span className="cell-note">
                            {m.skill_score_status.replace(/_/g, " ")}
                          </span>
                        )}
                      </td>
                      <td className="mono">
                        {m.coverage_90pct != null
                          ? number(m.coverage_90pct * 100, 1)
                          : "Not available"}
                      </td>
                      <td className="mono">
                        {m.n_observations ?? "Not available"}
                        {typeof m.n_observations === "number" &&
                          m.n_observations < 7 && (
                            <span className="cell-note">
                              too few days to trust
                            </span>
                          )}
                      </td>
                    </tr>
                  );
                }),
              )}
            </tbody>
          </table>
        </div>
      </section>
      <details className="method-note">
        <summary>How to read these metrics</summary>
        <dl className="metric-glossary">
          <div className="metric-glossary-item">
            <dt>MAE</dt>
            <dd>Average absolute miss, in °C.</dd>
          </div>
          <div className="metric-glossary-item">
            <dt>RMSE</dt>
            <dd>Prediction error in °C, with extra weight on larger misses.</dd>
          </div>
          <div className="metric-glossary-item">
            <dt>MAPE</dt>
            <dd>
              Error relative to actual temperature. Percentage errors in Celsius
              need care near zero.
            </dd>
          </div>
          <div className="metric-glossary-item">
            <dt>Skill score</dt>
            <dd>
              Relative improvement over the evaluation's climatology reference,
              usually the mean observed temperature. Undefined for a zero-error
              reference.
            </dd>
          </div>
          <div className="metric-glossary-item">
            <dt>Coverage</dt>
            <dd>
              Observed fraction inside the reported 90% interval. A nominal 90%
              interval is not a guarantee of 90% observed coverage.
            </dd>
          </div>
        </dl>
      </details>
    </>
  );
}
