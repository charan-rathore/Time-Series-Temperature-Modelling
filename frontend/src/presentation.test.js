import { comparisonSeries, temperature, shortDate } from "./presentation";
it("preserves zero, negative and finite measurement precision", () => {
  expect(temperature(0)).toBe("0.0");
  expect(temperature(-3.25)).toBe("-3.3");
  expect(temperature(28.599999999999998)).toBe("28.6");
});
it.each([null, undefined, NaN, Infinity, "0"])(
  "does not coerce %s into a temperature",
  (v) => expect(temperature(v)).toBe("–"),
);
it("formats calendar dates without moving to the previous timezone day", () =>
  expect(shortDate("2026-10-17")).toBe("17 Oct"));

it("draws the latest 14 comparison dates in ascending order without mutating API records", () => {
  const records = Array.from({ length: 20 }, (_, i) => ({
    date: `2026-10-${String(i + 1).padStart(2, "0")}`,
    actual_temp_c: 0,
    forecasts: { thermosense: { day1: { predicted: 0 } } },
  }));
  const forward = comparisonSeries(records),
    reverse = comparisonSeries([...records].reverse());
  expect(forward).toEqual(reverse);
  expect(forward).toHaveLength(14);
  expect(forward[0].date).toBe("2026-10-07");
  expect(forward[13].date).toBe("2026-10-20");
  expect(records[0].date).toBe("2026-10-01");
  expect(forward[0].thermosense).toBe(0);
});
