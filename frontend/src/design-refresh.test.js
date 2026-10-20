import React from "react";
import { render, screen, within } from "@testing-library/react";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import Leaderboard from "./pages/Leaderboard";
import Metrics from "./pages/Metrics";
import { api } from "./api";

jest.mock("./api", () => ({
  api: {
    getStatus: jest.fn(),
    getForecast: jest.fn(),
    getHistory: jest.fn(),
    getSensorLatest: jest.fn(),
    getLeaderboard: jest.fn(),
    getLeaderboardComparison: jest.fn(),
    getLeaderboardStatus: jest.fn(),
    getMetrics: jest.fn(),
  },
}));

// Same station-timezone formatter the component uses, so the mocked
// "today" is the Bangalore station date on any CI runner.
const fmt = (d) =>
  new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Kolkata",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(d);

beforeEach(() => jest.clearAllMocks());

it("reports a live sensor reading in plain words, never a fabricated one", async () => {
  api.getSensorLatest.mockResolvedValue({
    date: fmt(new Date()),
    sensor_temp_c: 24.6,
    sensor_humidity_pct: 61,
    recorded_at: new Date().toISOString(),
    source: "dht22_sensor",
  });
  render(<Sidebar activePage="dashboard" onNavigate={jest.fn()} />);
  expect(await screen.findByText(/24\.6°c this evening/)).toBeInTheDocument();
});

it("says it is waiting instead of showing a zero when no sensor has reported", async () => {
  api.getSensorLatest.mockRejectedValue(new Error("No sensor data available"));
  render(<Sidebar activePage="dashboard" onNavigate={jest.fn()} />);
  expect(
    await screen.findByText(/waiting for its first reading/),
  ).toBeInTheDocument();
});

it("labels an untrained climatology forecast as the last recorded reading", async () => {
  api.getStatus.mockResolvedValue({
    data_available: true,
    data_rows: 30,
    models_available: [],
    last_training_results: {},
    active_jobs: {},
  });
  api.getForecast.mockResolvedValue({
    location: "Bangalore",
    model_used: "climatology",
    generated_at: "2026-10-20T00:00:00Z",
    forecasts: [
      {
        date: "2026-10-21",
        horizon_days: 1,
        predicted_temp_c: 26,
        lower_bound_c: 24,
        upper_bound_c: 28,
      },
    ],
  });
  api.getHistory.mockResolvedValue({ records: [], total_records: 0 });
  render(<Dashboard onNavigate={jest.fn()} />);
  expect(await screen.findByText("last recorded reading")).toBeInTheDocument();
  expect(screen.getByText(/repeats the latest recorded temperature/)).toBeInTheDocument();
  expect(document.querySelector(".source-tag-provisional")).not.toBeNull();
});

it("shows 'none yet' instead of bare zeros when the station has no data", async () => {
  api.getStatus.mockResolvedValue({
    data_available: false,
    data_rows: 0,
    models_available: [],
    last_training_results: {},
    active_jobs: {},
  });
  api.getForecast.mockRejectedValue(new Error("offline"));
  api.getHistory.mockResolvedValue({ records: [], total_records: 0 });
  render(<Dashboard onNavigate={jest.fn()} />);
  await screen.findByRole("heading", { name: "Station overview" });
  const ledger = screen.getByRole("region", { name: "Station record" });
  expect(within(ledger).getAllByText("none yet")).toHaveLength(2);
});

it("marks fallback and unavailable forecasts as provisional, never trained", async () => {
  api.getStatus.mockResolvedValue({
    data_available: true, data_rows: 30, models_available: ["lgbm_h1"],
    last_training_results: {}, active_jobs: {},
  });
  api.getForecast.mockResolvedValue({
    location: "Bangalore", model_used: "fallback", generated_at: "2026-10-20T00:00:00Z",
    forecasts: [{ date: "2026-10-21", horizon_days: 1, predicted_temp_c: 26, lower_bound_c: 24, upper_bound_c: 28 }],
  });
  api.getHistory.mockResolvedValue({ records: [], total_records: 0 });
  const { unmount } = render(<Dashboard onNavigate={jest.fn()} />);
  expect(await screen.findByText("last recorded reading")).toBeInTheDocument();
  expect(document.querySelector(".source-tag-trained")).toBeNull();
  expect(document.querySelector(".source-tag-provisional")).not.toBeNull();
  unmount();

  api.getForecast.mockRejectedValue(new Error("network down"));
  render(<Dashboard onNavigate={jest.fn()} />);
  expect(await screen.findByText("Unavailable")).toBeInTheDocument();
  expect(document.querySelector(".source-tag-trained")).toBeNull();
});

it("labels the forecast target hour in the station timezone, not the viewer's", async () => {
  api.getStatus.mockResolvedValue({
    data_available: true, data_rows: 30, models_available: ["lgbm_h1"],
    last_training_results: { lgbm: { day1: { rmse: 0.7, n_observations: 30 } } }, active_jobs: {},
  });
  api.getForecast.mockResolvedValue({
    location: "Bangalore", model_used: "lgbm", generated_at: "2026-10-20T00:00:00Z",
    forecasts: [{ date: "2026-10-21", horizon_days: 1, predicted_temp_c: 22.7, lower_bound_c: 21.2, upper_bound_c: 24.2 }],
  });
  api.getHistory.mockResolvedValue({ records: [], total_records: 0 });
  render(<Dashboard onNavigate={jest.fn()} />);
  expect(await screen.findByText(/9 PM in Bangalore/)).toBeInTheDocument();
});

it("leaderboard shows the collect command instead of claiming automation that does not exist", async () => {
  api.getLeaderboard.mockResolvedValue({ rankings: [], window_days: 30, horizon_days: 1 });
  api.getLeaderboardComparison.mockResolvedValue({ sources: [] });
  api.getLeaderboardStatus.mockResolvedValue({ initialized: true, forecast_count: 0, actual_count: 0 });
  render(<Leaderboard />);
  expect(await screen.findByText("python -m src.data.baseline_collector --collect")).toBeInTheDocument();
  expect(screen.getByText("python scripts/record_reference_actual.py")).toBeInTheDocument();
  expect(screen.queryByText(/no action needed/)).toBeNull();
});

it("flags a one-day metric as too few days to trust instead of headlining it", async () => {
  api.getMetrics.mockResolvedValue({
    location: "Bangalore",
    models: {
      sarima: { day1: { rmse: 0.136, mae: 0.136, n_observations: 1 } },
      lgbm: { day1: { rmse: 0.76, mae: 0.637, n_observations: 30 } },
    },
  });
  render(<Metrics />);
  expect(await screen.findByText(/too few days to trust/)).toBeInTheDocument();
  // the badge exists exactly once and belongs to the evidenced model
  const badges = screen.getAllByText("Lowest day-1 RMSE");
  expect(badges).toHaveLength(1);
  expect(badges[0].closest("section").textContent).toMatch(/lgbm/);
  expect(badges[0].closest("section").textContent).not.toMatch(/sarima/);
});
