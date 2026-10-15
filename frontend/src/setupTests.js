import "@testing-library/jest-dom";
// Recharts needs a measurable container; jsdom has no layout engine.
jest.mock("recharts", () => {
  const actual = jest.requireActual("recharts");
  return {
    ...actual,
    ResponsiveContainer: ({ children }) =>
      require("react").cloneElement(children, { width: 600, height: 260 }),
  };
});
