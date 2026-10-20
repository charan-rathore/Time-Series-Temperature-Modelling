import React from "react";
import { render, screen, within } from "@testing-library/react";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import { api } from "./api";

jest.mock("./api", () => ({
  api: {
    getStatus: jest.fn(),
    getForecast: jest.fn(),
    getHistory: jest.fn(),
    getSensorLatest: jest.fn(),
  },
}));

const fmt = (d) =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;

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

it("labels an untrained climatology forecast as the seasonal average", async () => {
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
  expect(await screen.findByText("seasonal average")).toBeInTheDocument();
  expect(screen.getByText(/historical average for this date/)).toBeInTheDocument();
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
