import React from "react";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  within,
  cleanup,
} from "@testing-library/react";
import App from "./App";
import Dashboard from "./pages/Dashboard";
import Forecast from "./pages/Forecast";
import Metrics from "./pages/Metrics";
import Pipeline from "./pages/Pipeline";
import { api } from "./api";
jest.mock("./api", () => ({
  api: {
    getStatus: jest.fn(),
    getForecast: jest.fn(),
    getHistory: jest.fn(),
    getMetrics: jest.fn(),
    getLogs: jest.fn(),
    getMlflowRuns: jest.fn(),
    postFeedback: jest.fn(),
    getLeaderboard: jest.fn(),
    getLeaderboardComparison: jest.fn(),
    getLeaderboardStatus: jest.fn(),
    runBackfill: jest.fn(),
    runDaily: jest.fn(),
    runTraining: jest.fn(),
  },
}));
const forecast = {
  location: "Test station",
  model_used: "lgbm",
  generated_at: "2026-10-16T15:30:00Z",
  forecasts: [
    {
      date: "2026-10-17",
      horizon_days: 1,
      predicted_temp_c: 0,
      lower_bound_c: -1,
      upper_bound_c: 1,
      confidence: "90%",
    },
  ],
};
beforeEach(() => {
  jest.clearAllMocks();
  api.getStatus.mockResolvedValue({
    data_available: true,
    data_rows: 30,
    models_available: ["lgbm_h1"],
    last_training_results: { lgbm: { day1: { rmse: 0 } } },
    active_jobs: {},
  });
  api.getForecast.mockResolvedValue(forecast);
  api.getHistory.mockResolvedValue({ records: [], total_records: 0 });
  api.getLogs.mockResolvedValue({ lines: [] });
  api.getMlflowRuns.mockResolvedValue({ available: false });
});
afterEach(cleanup);
it("keeps labels and the current destination accessible through navigation", async () => {
  render(<App />);
  await screen.findByRole("heading", { name: "Station overview" });
  const nav = screen.getByRole("navigation", { name: "Main navigation" });
  expect(within(nav).getAllByRole("button")).toHaveLength(6);
  fireEvent.click(within(nav).getByRole("button", { name: "Forecast" }));
  expect(within(nav).getByRole("button", { name: "Forecast" })).toHaveAttribute(
    "aria-current",
    "page",
  );
  await screen.findByLabelText("Actual temperature (°C)");
  expect(api.runTraining).not.toHaveBeenCalled();
  expect(api.runBackfill).not.toHaveBeenCalled();
});
it("keeps a zero forecast and zero training error visible, not missing", async () => {
  render(<Dashboard onNavigate={jest.fn()} />);
  await screen.findByRole("heading", { name: "Station overview" });
  const hero = screen.getByRole("region", {
    name: "Tomorrow's temperature forecast",
  });
  expect(within(hero).getByText("0.0")).toBeInTheDocument();
  expect(screen.getByText("Training RMSE · day 1")).toBeInTheDocument();
  expect(
    within(screen.getByRole("region", { name: "Station record" })).getByText(
      "0.0",
    ),
  ).toBeInTheDocument();
});
it("does not invent a forecast on a partial API failure", async () => {
  api.getForecast.mockRejectedValue(new Error("offline"));
  render(<Dashboard onNavigate={jest.fn()} />);
  await screen.findByText("Forecast unavailable");
  expect(screen.getByText("Unavailable")).toBeInTheDocument();
});
it("renders a retryable station error", async () => {
  api.getStatus.mockRejectedValue(new Error("offline"));
  render(<Dashboard onNavigate={jest.fn()} />);
  expect(await screen.findByRole("alert")).toHaveTextContent("offline");
  api.getStatus.mockResolvedValue({ data_rows: 0 });
  fireEvent.click(screen.getByRole("button", { name: "Try again" }));
  await screen.findByRole("heading", { name: "Station overview" });
});
it("preserves forecast day selection and signals loading failures", async () => {
  render(<Forecast />);
  await screen.findByText("Test station", { exact: false });
  fireEvent.click(screen.getByRole("button", { name: "1 Day" }));
  await waitFor(() => expect(api.getForecast).toHaveBeenCalledWith(1));
  api.getForecast.mockRejectedValue(new Error("offline"));
  fireEvent.click(screen.getByRole("button", { name: "Refresh" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("offline");
});
it("keeps the observation form labelled and saves zero as an actual reading", async () => {
  api.postFeedback.mockResolvedValue({ message: "Observation recorded" });
  render(<Forecast />);
  await screen.findByText("Test station", { exact: false });
  fireEvent.change(screen.getByLabelText("Date"), {
    target: { value: "2024-07-11" },
  });
  fireEvent.change(screen.getByLabelText("Actual temperature (°C)"), {
    target: { value: "0" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Submit" }));
  await waitFor(() =>
    expect(api.postFeedback).toHaveBeenCalledWith("2024-07-11", 0),
  );
  expect(await screen.findByRole("status")).toHaveTextContent(
    "Observation recorded",
  );
});
it("shows rejected observations without claiming a successful save", async () => {
  api.postFeedback.mockRejectedValue(new Error("not recorded"));
  render(<Forecast />);
  fireEvent.change(screen.getByLabelText("Date"), {
    target: { value: "2024-07-11" },
  });
  fireEvent.change(screen.getByLabelText("Actual temperature (°C)"), {
    target: { value: "25" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Submit" }));
  expect(await screen.findByRole("status")).toHaveTextContent("not recorded");
});
it("retains missing metric states without a fabricated radar/composite score", async () => {
  api.getMetrics.mockResolvedValue({
    models: {
      lgbm: {
        day1: {
          rmse: 0,
          mae: 0,
          mape: null,
          skill_score: null,
          skill_score_status: "undefined_zero_baseline",
          n_observations: 1,
        },
      },
    },
  });
  render(<Metrics />);
  await screen.findByRole("heading", { name: "Model metrics" });
  const table = screen.getByRole("table");
  expect(within(table).getAllByText("0.000")).toHaveLength(2);
  expect(
    within(table).getByText("undefined zero baseline"),
  ).toBeInTheDocument();
  expect(screen.queryByText(/Radar/)).not.toBeInTheDocument();
});
it("reports unavailable evaluation instead of silently looking empty", async () => {
  api.getMetrics.mockRejectedValue(new Error("offline"));
  render(<Metrics />);
  expect(await screen.findByRole("alert")).toHaveTextContent("offline");
});
it("keeps pipeline controls and records action errors inline", async () => {
  api.runBackfill.mockRejectedValue(new Error("job busy"));
  render(<Pipeline />);
  await screen.findByLabelText("Start (optional)");
  fireEvent.click(screen.getByRole("button", { name: "Run Backfill" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("job busy");
  expect(api.runTraining).not.toHaveBeenCalled();
});
