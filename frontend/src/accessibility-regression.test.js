import React from "react";
import { render, screen } from "@testing-library/react";
import Sidebar from "./components/Sidebar";
import Forecast from "./pages/Forecast";
import { api } from "./api";
jest.mock("./api", () => ({ api: { getForecast: jest.fn() } }));
it("marks the current navigation destination for assistive technology", () => {
  render(<Sidebar activePage="forecast" onNavigate={jest.fn()} />);
  expect(screen.getByRole("button", { name: "Forecast" })).toHaveAttribute(
    "aria-current",
    "page",
  );
});
it("connects observation field labels to their controls", async () => {
  api.getForecast.mockResolvedValue({
    forecasts: [],
    generated_at: "2026-10-16T00:00:00Z",
  });
  render(<Forecast />);
  await screen.findByText(/Generated/);
  expect(screen.getByLabelText(/Actual temperature/i)).toHaveAttribute(
    "type",
    "number",
  );
});
it("announces an unavailable forecast instead of silently hiding the failure", async () => {
  api.getForecast.mockRejectedValue(new Error("offline"));
  render(<Forecast />);
  expect(await screen.findByRole("alert")).toHaveTextContent("offline");
});
