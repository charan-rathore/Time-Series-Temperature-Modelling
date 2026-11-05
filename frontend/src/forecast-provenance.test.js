import { render, screen } from "@testing-library/react";
import Forecast from "./pages/Forecast";

jest.mock("./api", () => ({
  api: {
    getForecast: jest.fn(),
    getMetrics: jest.fn(),
    submitFeedback: jest.fn(),
  },
}));

import { api } from "./api";

const baseForecast = {
  location: "Bangalore",
  generated_at: "2026-10-31T19:00:00Z",
  model_used: "lgbm",
  forecasts: [
    { date: "2026-11-01", horizon_days: 1, predicted_temp_c: 25.1, lower_bound_c: 23.9, upper_bound_c: 26.3, confidence: "90%" },
  ],
};

beforeEach(() => {
  jest.clearAllMocks();
  api.getMetrics.mockResolvedValue({ models: [] });
});

it("shows the training window when the forecast carries it", async () => {
  api.getForecast.mockResolvedValue({
    ...baseForecast,
    trained_data_range: "2024-06-01 to 2026-10-19",
    model_trained_at: "2026-10-20T15:35:25+00:00",
  });
  render(<Forecast />);
  expect(
    await screen.findByTestId("training-provenance"),
  ).toHaveTextContent("Trained on 2024-06-01 to 2026-10-19 grid history");
});

it("shows no provenance line when the backend omits it", async () => {
  api.getForecast.mockResolvedValue(baseForecast);
  render(<Forecast />);
  await screen.findByText(/Day 1/);
  expect(screen.queryByTestId("training-provenance")).not.toBeInTheDocument();
});
