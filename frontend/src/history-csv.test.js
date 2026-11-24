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

const header = "date,temp_c,api_temp_c,api_bias,humidity_pct,pressure_hpa";
const record = {
  date: "2026-11-20",
  actual_temp_c: 24.5,
  api_temp_c: 23.25,
  api_bias: 1.25,
  humidity_pct: 67,
  pressure_hpa: 1008.5,
  is_sensor_reading: true,
};
const response = (records) => ({
  records,
  total_records: records.length,
  location: "Synthetic fixture",
});
let createURL, revokeURL, click;

beforeEach(() => {
  jest.resetAllMocks();
  createURL = jest.fn().mockReturnValue("blob:history-export-fixture");
  revokeURL = jest.fn();
  Object.defineProperty(URL, "createObjectURL", {
    configurable: true,
    value: createURL,
  });
  Object.defineProperty(URL, "revokeObjectURL", {
    configurable: true,
    value: revokeURL,
  });
  click = jest
    .spyOn(HTMLAnchorElement.prototype, "click")
    .mockImplementation(() => {});
});
afterEach(() => click.mockRestore());

async function open(records) {
  api.getHistory.mockResolvedValue(response(records));
  render(<History />);
  await screen.findByText("Temperature Timeline");
}
function readBlob(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsText(blob);
  });
}
async function exportCSV() {
  fireEvent.click(
    screen.getByRole("button", { name: "Export CSV", exact: true }),
  );
  expect(createURL).toHaveBeenCalledTimes(1);
  return readBlob(createURL.mock.calls[0][0]);
}

it("exports every populated weather value under its existing column header", async () => {
  await open([record]);
  const csv = await exportCSV();
  expect(csv).toBe(`${header}\n2026-11-20,24.5,23.25,1.25,67,1008.5`);
});

it("keeps zero and negative values rather than replacing them with blanks", async () => {
  await open([
    {
      ...record,
      actual_temp_c: -2.5,
      api_temp_c: 0,
      api_bias: -2.5,
      humidity_pct: 0,
      pressure_hpa: 0,
    },
  ]);
  expect(await exportCSV()).toBe(`${header}\n2026-11-20,-2.5,0,-2.5,0,0`);
});

it("keeps missing values blank without writing null or undefined", async () => {
  await open([
    {
      date: record.date,
      actual_temp_c: null,
      api_temp_c: null,
      api_bias: undefined,
      humidity_pct: null,
      pressure_hpa: undefined,
    },
  ]);
  expect(await exportCSV()).toBe(`${header}\n2026-11-20,,,,,`);
});

it("exports all records in source order, not only the last fifty displayed rows", async () => {
  const records = Array.from({ length: 61 }, (_, i) => ({
    ...record,
    date: new Date(Date.UTC(2026, 8, i + 1)).toISOString().slice(0, 10),
    api_temp_c: i,
    api_bias: -i,
  }));
  await open(records);
  const rows = (await exportCSV()).split("\n");
  expect(rows).toHaveLength(62);
  expect(rows[1]).toBe("2026-09-01,24.5,0,0,67,1008.5");
  expect(rows[61]).toBe("2026-10-31,24.5,60,-60,67,1008.5");
});

it("preserves the selected-date filename, CSV MIME type and object URL cleanup", async () => {
  await open([record]);
  await exportCSV();
  const [start, end] = api.getHistory.mock.calls[0];
  expect(click.mock.instances[0]).toHaveAttribute(
    "download",
    `thermosense_history_${start}_${end}.csv`,
  );
  expect(createURL.mock.calls[0][0].type).toBe("text/csv");
  expect(revokeURL).toHaveBeenCalledWith("blob:history-export-fixture");
});

it("does not create a download for an empty history", async () => {
  await open([]);
  fireEvent.click(
    screen.getByRole("button", { name: "Export CSV", exact: true }),
  );
  expect(createURL).not.toHaveBeenCalled();
  expect(click).not.toHaveBeenCalled();
});

it("does not create a download before the first request has completed", async () => {
  let resolve;
  api.getHistory.mockReturnValue(
    new Promise((done) => {
      resolve = done;
    }),
  );
  render(<History />);
  fireEvent.click(
    screen.getByRole("button", { name: "Export CSV", exact: true }),
  );
  expect(createURL).not.toHaveBeenCalled();
  await act(async () => resolve(response([])));
});

it("preserves fractional numbers without adding display rounding", async () => {
  await open([
    {
      ...record,
      actual_temp_c: 24.123456789,
      api_temp_c: 0.000001,
      api_bias: -0.125,
      humidity_pct: 12.34567,
      pressure_hpa: 1008.56789,
    },
  ]);
  expect(await exportCSV()).toBe(
    `${header}\n2026-11-20,24.123456789,0.000001,-0.125,12.34567,1008.56789`,
  );
});

it("keeps a deliberately shuffled source sequence unchanged", async () => {
  await open([
    { ...record, date: "2026-11-21", api_temp_c: 21 },
    { ...record, date: "2026-11-19", api_temp_c: 19 },
    { ...record, date: "2026-11-20", api_temp_c: 20 },
  ]);
  expect(await exportCSV()).toBe(
    `${header}\n2026-11-21,24.5,21,1.25,67,1008.5\n2026-11-19,24.5,19,1.25,67,1008.5\n2026-11-20,24.5,20,1.25,67,1008.5`,
  );
});

it("does not create a download when the records property is absent", async () => {
  api.getHistory.mockResolvedValue({
    total_records: 0,
    location: "Synthetic fixture",
  });
  render(<History />);
  await screen.findByText("Temperature Timeline");
  fireEvent.click(
    screen.getByRole("button", { name: "Export CSV", exact: true }),
  );
  expect(createURL).not.toHaveBeenCalled();
});
