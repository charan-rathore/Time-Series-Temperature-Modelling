import { StrictMode } from "react";
import { act, fireEvent, render, screen } from "@testing-library/react";
import History from "./pages/History";
import { api } from "./api";

jest.mock("./api", () => ({ api: { getHistory: jest.fn() } }));
jest.mock("recharts", () => ({
  ResponsiveContainer: ({ children }) => children,
  LineChart: ({ children }) => children,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
  Brush: () => null,
}));
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
const response = (name) => ({ records: [], total_records: 0, location: name });
async function finish(request, value, failed = false) {
  await act(async () => {
    if (failed) request.reject(new Error(value));
    else request.resolve(value);
    await request.promise.catch(() => {});
  });
}
function selectRange() {
  fireEvent.change(screen.getByLabelText("Start date"), { target: { value: "2026-10-01" } });
}
beforeEach(() => jest.resetAllMocks());

it("keeps the newest history when an older range finishes last", async () => {
  const old = deferred(), current = deferred();
  api.getHistory.mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise);
  render(<History />);
  selectRange();
  await finish(current, response("Selected range"));
  await finish(old, response("Obsolete range"));
  expect(screen.getByText(/Selected range/)).toBeInTheDocument();
  expect(screen.queryByText(/Obsolete range/)).not.toBeInTheDocument();
});

it("does not replace a newer range with an obsolete error", async () => {
  const old = deferred(), current = deferred();
  api.getHistory.mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise);
  render(<History />);
  selectRange();
  await finish(current, response("Selected range"));
  await finish(old, "obsolete failure", true);
  expect(screen.getByText(/Selected range/)).toBeInTheDocument();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

it("keeps the current range loading when an obsolete request completes", async () => {
  const old = deferred(), current = deferred();
  api.getHistory.mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise);
  render(<History />);
  selectRange();
  await finish(old, response("Obsolete range"));
  expect(screen.getByText("Fetching history…")).toBeInTheDocument();
  await finish(current, response("Selected range"));
  expect(screen.getByText(/Selected range/)).toBeInTheDocument();
});

it("preserves a current failure after an obsolete success", async () => {
  const old = deferred(), current = deferred();
  api.getHistory.mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise);
  render(<History />);
  selectRange();
  await finish(current, "current failure", true);
  await finish(old, response("Obsolete range"));
  expect(screen.getByRole("alert")).toHaveTextContent("current failure");
  expect(screen.queryByText(/Obsolete range/)).not.toBeInTheDocument();
});

it("permits recovery from a current failure", async () => {
  const initial = deferred(), retry = deferred();
  api.getHistory.mockReturnValueOnce(initial.promise).mockReturnValueOnce(retry.promise);
  render(<History />);
  await finish(initial, "current failure", true);
  expect(screen.getByRole("alert")).toHaveTextContent("current failure");
  fireEvent.click(screen.getByRole("button", { name: "Load", exact: true }));
  await finish(retry, response("Recovered range"));
  expect(screen.getByText(/Recovered range/)).toBeInTheDocument();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

it("does not revive an older refresh for the same date range", async () => {
  const old = deferred(), current = deferred();
  api.getHistory.mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise);
  render(<History />);
  fireEvent.click(screen.getByRole("button", { name: "Refresh", exact: true }));
  await finish(current, response("Current refresh"));
  await finish(old, response("Obsolete refresh"));
  expect(screen.getByText(/Current refresh/)).toBeInTheDocument();
  expect(screen.queryByText(/Obsolete refresh/)).not.toBeInTheDocument();
});

it("ignores the discarded StrictMode mount request", async () => {
  const old = deferred(), current = deferred();
  api.getHistory.mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise);
  render(<StrictMode><History /></StrictMode>);
  expect(api.getHistory).toHaveBeenCalledTimes(2);
  await finish(current, response("Current mount"));
  await finish(old, response("Discarded mount"));
  expect(screen.getByText(/Current mount/)).toBeInTheDocument();
  expect(screen.queryByText(/Discarded mount/)).not.toBeInTheDocument();
});

it("keeps a remounted page independent of an unmounted request", async () => {
  const old = deferred(), current = deferred();
  api.getHistory.mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise);
  const { unmount } = render(<History />);
  unmount();
  render(<History />);
  await finish(current, response("Current mount"));
  await finish(old, response("Unmounted range"));
  expect(screen.getByText(/Current mount/)).toBeInTheDocument();
  expect(screen.queryByText(/Unmounted range/)).not.toBeInTheDocument();
});
