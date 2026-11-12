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
  const modelLabel = model === "climatology" ? "seasonal average"
    : model === "fallback" ? "last recorded reading"
    : model;
  // Show the day-1 error of the model that is actually served (never the best
  // of several), with how many days it was measured on.
  const training = status?.last_training_results || {};
  const served = training[model]?.day1;
  const rmseDays =
    typeof served?.rmse === "number" && (served?.n_observations ?? 0) >= 7
      ? served.n_observations
      : null;
  const rmse = rmseDays ? served.rmse : undefined;
  // The API marks output that is not a real forward forecast (data ended
  // before today) or whose target date it cannot verify.
  const stale = !!forecast && forecast.is_current === false;
  const synthetic = !!tomorrow?.synthetic;
  const stationRows = status?.sensor_reading_rows;
  const targetText = tomorrow?.date ? `for ${shortDate(tomorrow.date)}` : "target date unverified";
  const captionTarget = tomorrow?.date
    ? `its output for ${shortDate(tomorrow.date)}`
    : "its output for a date that cannot be verified";
  return (
    <>
      <header className="page-header">
        <div>
          <p className="eyebrow">Why this exists</p>
          <h2 className="why-headline">
            Cooling is about a tenth of India&rsquo;s electricity. On a hot
            night it can be <em>a third</em>.
          </h2>
          <p className="page-intro">
            Whether you run the AC or open a window depends on the temperature
            at <em>your street</em> around 9 PM, not on a grid estimate for
            the nearest airport. ThermoSense is built to learn that one number
            from a street-level sensor and keep score of who predicted it.
            {typeof stationRows === "number"
              ? ` Today it has ${stationRows.toLocaleString()} station readings; the rest of its history is grid data, and the live reading in the sidebar is the airport.`
              : ""}
          </p>
          <p className="why-source">
            Source:{" "}
            <a
              href="https://www.iea.org/commentaries/india-s-electricity-demand-grows-at-night-managing-rising-cooling-demand"
              target="_blank"
              rel="noreferrer"
            >
              IEA, India&rsquo;s electricity demand grows at night (June 2026)
            </a>
            . Location: {forecast?.location || history?.location || "Bangalore"}.
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
          aria-label={stale ? (synthetic ? "Placeholder value, not a forecast" : "Latest model output, not a current forecast")  : "Tomorrow's temperature forecast"}
        >
          <div className="hero-top">
            <span className="eyebrow">
              {synthetic
                ? "Placeholder · not a forecast"
                : stale
                ? `Latest model output · ${targetText}`
                : `Tomorrow evening · ${shortDate(tomorrow?.date)}`}
            </span>
            <span className={`source-tag ${untrained ? "source-tag-provisional" : "source-tag-trained"}`}>{modelLabel}</span>
          </div>
          <div className="hero-temperature">
            {temperature(tomorrow?.predicted_temp_c)}
            <span>°C</span>
          </div>
          {synthetic ? (
            <p className="hero-caption" data-testid="stale-forecast">
              This is a fixed placeholder value, not model output and not a forecast. No trained
              forecast is available right now.
            </p>
          ) : stale ? (
            <p className="hero-caption" data-testid="stale-forecast">
              Not a forecast for tomorrow. The data behind this model ends{" "}
              {forecast.data_through ? shortDate(forecast.data_through) : "on an unknown date"}, so
              this is {captionTarget}. Do not plan tonight around it.
            </p>
          ) : (
            <p className="hero-caption">What to expect outside at <em>9 PM</em>. Plan the window or the AC around this.</p>
          )}
          <div className="hero-bottom">
            <span>
              {tomorrow
                ? `${temperature(tomorrow.lower_bound_c)} to ${temperature(tomorrow.upper_bound_c)} °C likely range`
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
              {model === "climatology"
                ? "Not a trained forecast yet: this is the historical average for this date, shown until the pipeline has enough real readings to learn from."
                : model === "fallback"
                ? "Not a trained forecast yet: this repeats the latest recorded temperature, shown until the pipeline has enough real readings to learn from."
                : "Placeholder estimate, not a trained model forecast."}
            </p>
          )}
        </section>
        <section className="station-ledger" aria-label="Station record">
          <span className="eyebrow">What the station knows</span>
          <dl>
            <div>
              <dt>Station readings</dt>
              {status?.sensor_reading_rows != null ? (
                status.sensor_reading_rows > 0 ? (
                  <dd>{status.sensor_reading_rows.toLocaleString()}</dd>
                ) : (
                  <dd className="dd-pending">none yet</dd>
                )
              ) : status ? (
                /* Backend predates the sensor split: the count is unknown,
                   never the grid-history total wearing a station label. */
                <dd className="dd-pending">unknown</dd>
              ) : (
                <dd className="dd-pending">none yet</dd>
              )}
            </div>
            <div>
              <dt>Grid history days</dt>
              {status?.data_rows ? (
                <dd>{status.data_rows.toLocaleString()}</dd>
              ) : (
                <dd className="dd-pending">none yet</dd>
              )}
            </div>
            <div>
              <dt>Forecasting models</dt>
              {status?.models_available?.length ? (
                <dd>{status.models_available.length}</dd>
              ) : (
                <dd className="dd-pending">none yet</dd>
              )}
            </div>
            <div>
              <dt>Typical miss, next day</dt>
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
            {rmseDays
              ? `Error of the served model on ${rmseDays} days in its last training run, not independently audited. A guide, not a promise.`
              : "No error measurement available for the served model."}
          </p>
        </section>
      </div>
      <section className="card trend-card">
        <div className="card-header">
          <div>
            <span className="eyebrow">Measured & estimated</span>
            <h3>Measured vs. the grid estimate</h3>
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
            <span className="eyebrow">{stale ? "Preview, not a forecast" : "Looking ahead"}</span>
            <h3>{stale ? "Latest model output" : "The next three evenings"}</h3>
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
                  {f.synthetic ? "Placeholder" : f.is_current === false ? `Step ${f.horizon_days}` : `Day ${f.horizon_days}`}
                  <small>{f.date ? shortDate(f.date) : f.synthetic ? "not a forecast" : "date unverified"}</small>
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
