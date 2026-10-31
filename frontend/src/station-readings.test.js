import { render, screen } from "@testing-library/react";
import Dashboard from "./pages/Dashboard";

jest.mock("./api", () => ({
  api: {
    getStatus: jest.fn(),
    getForecast: jest.fn(),
    getMetrics: jest.fn(),
    getHistory: jest.fn(),
  },
}));

import { api } from "./api";

const forecast = {
  location: "Bangalore",
  model_used: "lgbm",
  generated_at: "2026-10-20T00:00:00Z",
  forecasts: [
    { date: "2026-10-21", horizon_days: 1, temp_c: 25.1 },
    { date: "2026-10-22", horizon_days: 2, temp_c: 25.4 },
    { date: "2026-10-23", horizon_days: 3, temp_c: 25.6 },
  ],
};

beforeEach(() => {
  jest.resetAllMocks();
  api.getForecast.mockResolvedValue(forecast);
  api.getMetrics.mockResolvedValue({ models: [] });
  api.getHistory.mockResolvedValue({ records: [], total_records: 0 });
});

it("separates real station readings from grid history days", async () => {
  api.getStatus.mockResolvedValue({
    data_available: true,
    data_rows: 871,
    sensor_reading_rows: 40,
    models_available: ["lgbm"],
    last_training_results: {},
    active_jobs: {},
  });
  render(<Dashboard onNavigate={jest.fn()} />);
  expect(await screen.findByText("Station readings")).toBeInTheDocument();
  expect(screen.getByText("Grid history days")).toBeInTheDocument();
  expect(screen.getByText("40")).toBeInTheDocument();
  expect(screen.getByText("871")).toBeInTheDocument();
  expect(screen.queryByText("Days recorded")).not.toBeInTheDocument();
});

it("reports station readings as unknown when the backend predates the sensor split", async () => {
  // Backward compatibility: an older backend omits sensor_reading_rows.
  // The dashboard must say the count is unknown - relabeling the grid
  // history total as station readings is exactly the false claim the split
  // exists to remove.
  api.getStatus.mockResolvedValue({
    data_available: true,
    data_rows: 30,
    models_available: ["lgbm"],
    last_training_results: {},
    active_jobs: {},
  });
  render(<Dashboard onNavigate={jest.fn()} />);
  expect(await screen.findByText("Station readings")).toBeInTheDocument();
  expect(screen.getByText("unknown")).toBeInTheDocument();
  // grid history still shows the total, exactly once
  expect(screen.getAllByText("30")).toHaveLength(1);
});
