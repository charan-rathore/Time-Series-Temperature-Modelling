import { act, fireEvent, render, screen } from "@testing-library/react";
import History from "./pages/History";
import { api } from "./api";
jest.mock("./api", () => ({ api: { getHistory: jest.fn() } }));
jest.mock("recharts", () => ({ ResponsiveContainer: ({children}) => children, LineChart: ({children}) => children, Line: () => null, XAxis: () => null, YAxis: () => null, CartesianGrid: () => null, Tooltip: () => null, Legend: () => null, Brush: () => null }));
let click;
beforeEach(() => {
  jest.resetAllMocks();
  Object.defineProperty(URL, "createObjectURL", { configurable:true, value:jest.fn().mockReturnValue("blob:fixture") });
  Object.defineProperty(URL, "revokeObjectURL", { configurable:true, value:jest.fn() });
  click=jest.spyOn(HTMLAnchorElement.prototype,"click").mockImplementation(()=>{});
});
afterEach(()=>click.mockRestore());
const fixture={records:[{date:"2026-11-20",actual_temp_c:24}],total_records:1,location:"Fixture"};
it("names a retained-history download with the range that actually loaded", async()=>{
  let finish;
  api.getHistory.mockResolvedValueOnce(fixture).mockImplementationOnce(()=>new Promise(resolve=>{finish=resolve}));
  render(<History/>);
  await screen.findByText("Temperature Timeline");
  const [loadedStart,loadedEnd]=api.getHistory.mock.calls[0];
  fireEvent.change(screen.getByLabelText("Start date"),{target:{value:"2026-10-01"}});
  fireEvent.click(screen.getByRole("button",{name:"Export CSV",exact:true}));
  expect(click.mock.instances[0]).toHaveAttribute("download",`thermosense_history_${loadedStart}_${loadedEnd}.csv`);
  await act(async()=>finish(fixture));
});

it("uses the new range after its successful response", async()=>{
  let finish;
  api.getHistory.mockResolvedValueOnce(fixture).mockImplementationOnce(()=>new Promise(resolve=>{finish=resolve}));
  render(<History/>);
  await screen.findByText("Temperature Timeline");
  fireEvent.change(screen.getByLabelText("Start date"),{target:{value:"2026-10-01"}});
  const [,selectedEnd]=api.getHistory.mock.calls[1];
  await act(async()=>finish({...fixture,location:"New fixture"}));
  fireEvent.click(screen.getByRole("button",{name:"Export CSV",exact:true}));
  expect(click.mock.instances[0]).toHaveAttribute("download",`thermosense_history_2026-10-01_${selectedEnd}.csv`);
});

it("does not relabel retained rows when an obsolete range finishes",async()=>{
  let obsolete,current;
  api.getHistory.mockResolvedValueOnce(fixture)
    .mockImplementationOnce(()=>new Promise(resolve=>{obsolete=resolve}))
    .mockImplementationOnce(()=>new Promise(resolve=>{current=resolve}));
  render(<History/>);
  await screen.findByText("Temperature Timeline");
  const [originalStart,originalEnd]=api.getHistory.mock.calls[0];
  fireEvent.change(screen.getByLabelText("Start date"),{target:{value:"2026-10-01"}});
  fireEvent.change(screen.getByLabelText("Start date"),{target:{value:"2026-09-01"}});
  await act(async()=>obsolete({...fixture,location:"Obsolete"}));
  fireEvent.click(screen.getByRole("button",{name:"Export CSV",exact:true}));
  expect(click.mock.instances[0]).toHaveAttribute("download",`thermosense_history_${originalStart}_${originalEnd}.csv`);
  await act(async()=>current({...fixture,location:"Newest"}));
  fireEvent.click(screen.getByRole("button",{name:"Export CSV",exact:true}));
  expect(click.mock.instances[1]).toHaveAttribute("download",`thermosense_history_2026-09-01_${originalEnd}.csv`);
});

it("preserves existing CSV bytes, MIME and cleanup on unchanged range",async()=>{
  api.getHistory.mockResolvedValueOnce(fixture);
  render(<History/>);await screen.findByText("Temperature Timeline");
  const [start,end]=api.getHistory.mock.calls[0];
  fireEvent.click(screen.getByRole("button",{name:"Export CSV",exact:true}));
  const blob=URL.createObjectURL.mock.calls[0][0];
  const text=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result);r.onerror=reject;r.readAsText(blob)});
  expect(text).toBe("date,temp_c,api_temp_c,api_bias,humidity_pct,pressure_hpa\n2026-11-20,24,,,,");
  expect(blob.type).toBe("text/csv");expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:fixture");
  expect(click.mock.instances[0]).toHaveAttribute("download",`thermosense_history_${start}_${end}.csv`);
});

it("does not export when the current range failed and cleared its data",async()=>{
  api.getHistory.mockResolvedValueOnce(fixture).mockRejectedValueOnce(new Error("range failure"));
  render(<History/>);await screen.findByText("Temperature Timeline");
  fireEvent.change(screen.getByLabelText("Start date"),{target:{value:"2026-10-01"}});
  await screen.findByRole("alert");
  fireEvent.click(screen.getByRole("button",{name:"Export CSV",exact:true}));
  expect(click).not.toHaveBeenCalled();expect(URL.createObjectURL).not.toHaveBeenCalled();
});
