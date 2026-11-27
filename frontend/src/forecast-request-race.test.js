import { Profiler, StrictMode } from "react";
import { act, fireEvent, render, screen } from "@testing-library/react";
import Forecast from "./pages/Forecast";
import { api } from "./api";

jest.mock("./api", () => ({
  api: { getForecast: jest.fn(), postFeedback: jest.fn() },
}));

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}
function result(days, model_used = "Current selection") {
  return {
    location: "Synthetic fixture",
    model_used,
    generated_at: "2026-11-22T06:00:00Z",
    forecasts: Array.from({ length: days }, (_, i) => ({
      date: `2026-11-${23 + i}`,
      horizon_days: i + 1,
      predicted_temp_c: 20 + i,
      lower_bound_c: 19 + i,
      upper_bound_c: 21 + i,
      confidence: "90%",
    })),
  };
}
async function finish(request, value, failed = false) {
  await act(async () => {
    if (failed) request.reject(new Error(value));
    else request.resolve(value);
    await request.promise.catch(() => {});
  });
}
function selectOne() {
  fireEvent.click(screen.getByRole("button", { name: "1 Day", exact: true }));
}
function expectCurrent(count = 1, name = "Current selection") {
  expect(screen.getByText(name)).toBeInTheDocument();
  expect(screen.getAllByText(/^Day [1-3]$/)).toHaveLength(count);
  expect(
    screen.getByRole("button", {
      name: `${count} Day${count > 1 ? "s" : ""}`,
      exact: true,
    }),
  ).toHaveClass("btn-primary");
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
}

beforeEach(() => jest.resetAllMocks());

it("keeps the selected horizon when an older successful request finishes last", async () => {
  const old = deferred(),
    current = deferred();
  api.getForecast
    .mockReturnValueOnce(old.promise)
    .mockReturnValueOnce(current.promise);
  render(<Forecast />);
  selectOne();
  await finish(current, result(1));
  expectCurrent();
  await finish(old, result(3, "Obsolete response"));
  expectCurrent();
  expect(screen.queryByText("Obsolete response")).not.toBeInTheDocument();
});

it("does not replace a newer success with an older request error", async () => {
  const old = deferred(),
    current = deferred();
  api.getForecast
    .mockReturnValueOnce(old.promise)
    .mockReturnValueOnce(current.promise);
  render(<Forecast />);
  selectOne();
  await finish(current, result(1));
  await finish(old, "obsolete failure", true);
  expectCurrent();
});

it.each([false, true])(
  "keeps loading the current request when an obsolete completion arrives (error=%s)",
  async (failed) => {
    const old = deferred(),
      current = deferred();
    api.getForecast
      .mockReturnValueOnce(old.promise)
      .mockReturnValueOnce(current.promise);
    render(<Forecast />);
    selectOne();
    await finish(
      old,
      failed ? "obsolete failure" : result(3, "Obsolete response"),
      failed,
    );
    expect(screen.queryAllByText(/^Day [1-3]$/)).toHaveLength(0);
    expect(
      screen.queryByText(/No forecast data available/),
    ).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    await finish(current, result(1));
    expectCurrent();
  },
);

it("still reports a current request failure and allows a successful refresh", async () => {
  const current = deferred(),
    retry = deferred();
  api.getForecast
    .mockReturnValueOnce(current.promise)
    .mockReturnValueOnce(retry.promise);
  render(<Forecast />);
  await finish(current, "current failure", true);
  expect(screen.getByRole("alert")).toHaveTextContent("current failure");
  fireEvent.click(screen.getByRole("button", { name: "Refresh", exact: true }));
  await finish(retry, result(3));
  expectCurrent(3);
});

it("ignores an older same-horizon refresh response", async () => {
  const old = deferred(),
    current = deferred();
  api.getForecast
    .mockReturnValueOnce(old.promise)
    .mockReturnValueOnce(current.promise);
  render(<Forecast />);
  fireEvent.click(screen.getByRole("button", { name: "Refresh", exact: true }));
  await finish(current, result(3));
  await finish(old, result(3, "Obsolete response"));
  expectCurrent(3);
});

it("does not revive the first request when the person switches back to that horizon", async () => {
  const first = deferred(),
    second = deferred(),
    third = deferred();
  api.getForecast
    .mockReturnValueOnce(first.promise)
    .mockReturnValueOnce(second.promise)
    .mockReturnValueOnce(third.promise);
  render(<Forecast />);
  selectOne();
  fireEvent.click(screen.getByRole("button", { name: "3 Days", exact: true }));
  await finish(third, result(3));
  await finish(second, result(1, "Obsolete one-day response"));
  await finish(first, result(3, "Obsolete three-day response"));
  expectCurrent(3);
});

it("invalidates pending completions when the page unmounts", async () => {
  const old = deferred(),
    current = deferred();
  api.getForecast
    .mockReturnValueOnce(old.promise)
    .mockReturnValueOnce(current.promise);
  const { unmount } = render(<Forecast />);
  unmount();
  render(<Forecast />);
  await finish(current, result(3));
  await finish(old, result(3, "Unmounted response"));
  expectCurrent(3);
});

it("ignores the discarded mount request when StrictMode replays the effect", async () => {
  const discarded = deferred(),
    current = deferred();
  api.getForecast
    .mockReturnValueOnce(discarded.promise)
    .mockReturnValueOnce(current.promise);
  render(
    <StrictMode>
      <Forecast />
    </StrictMode>,
  );
  expect(api.getForecast).toHaveBeenCalledTimes(2);
  await finish(current, result(3));
  await finish(discarded, result(3, "Discarded mount response"));
  expectCurrent(3);
});

it("never commits old cards under a newly selected horizon before passive effects", async () => {
  const old = deferred(),
    current = deferred(),
    committedCards = [];
  api.getForecast
    .mockReturnValueOnce(old.promise)
    .mockReturnValueOnce(current.promise);
  render(
    <Profiler
      id="forecast"
      onRender={() => {
        if (
          screen
            .queryByRole("button", { name: "1 Day", exact: true })
            ?.getAttribute("class")
            ?.includes("btn-primary")
        ) {
          committedCards.push(screen.queryAllByText(/^Day [1-3]$/).length);
        }
      }}
    >
      <Forecast />
    </Profiler>,
  );
  await finish(old, result(3));
  expectCurrent(3);
  selectOne();
  expect(committedCards.length).toBeGreaterThan(0);
  expect(committedCards.every((count) => count === 0)).toBe(true);
  await finish(current, result(1));
  expectCurrent();
});

it("preserves a latest failure when an obsolete success arrives afterward", async () => {
  const old = deferred(),
    current = deferred();
  api.getForecast
    .mockReturnValueOnce(old.promise)
    .mockReturnValueOnce(current.promise);
  render(<Forecast />);
  selectOne();
  await finish(current, "current failure", true);
  await finish(old, result(3, "Obsolete response"));
  expect(screen.getByRole("alert")).toHaveTextContent("current failure");
  expect(screen.queryByText("Obsolete response")).not.toBeInTheDocument();
});
