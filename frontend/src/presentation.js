/** Display precision only. Never coerce missing values into a measurement. */
export function temperature(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(1)
    : "–";
}
export function shortDate(value) {
  if (!value) return "Date unavailable";
  const parts = value.split("-");
  if (parts.length !== 3) return value;
  // Parse at noon without timezone conversion changing the calendar day.
  const date = new Date(`${value}T12:00:00`);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleDateString("en-GB", { day: "numeric", month: "short" });
}
/** Select the latest two weeks, then draw in calendar order regardless of API ordering. */
export function comparisonSeries(records = []) {
  return [...records]
    .sort((a, b) => a.date.localeCompare(b.date))
    .slice(-14)
    .map((record) => {
      const row = { date: record.date, actual: record.actual_temp_c };
      Object.entries(record.forecasts || {}).forEach(([source, horizons]) => {
        if (horizons.day1) row[source] = horizons.day1.predicted;
      });
      return row;
    });
}
