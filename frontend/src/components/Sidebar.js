import React from "react";
import { BookOpen } from "lucide-react";
import SensorStatus from "./SensorStatus";

/*
 * Hand-drawn marks, drawn once and kept imperfect on purpose.
 * Round caps, slightly wandering strokes - a pencil, not an icon font.
 */
const stroke = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.7,
  strokeLinecap: "round",
  strokeLinejoin: "round",
};

const GaugeIcon = () => (
  <svg viewBox="0 0 20 20" {...stroke} aria-hidden="true">
    <path d="M3.4 14.3a6.9 6.9 0 0 1 13.4-.6" />
    <path d="M10 13.9l3.3-3.2" />
    <circle cx="10" cy="14" r="1.05" />
    <path d="M4.8 9.3l.9.8M15.3 9.2l-.9.9" />
  </svg>
);
const SunCloudIcon = () => (
  <svg viewBox="0 0 20 20" {...stroke} aria-hidden="true">
    <circle cx="7" cy="6.8" r="2.55" />
    <path d="M7 1.7v1.2M2.6 3.5l.9.8M11.4 3.7l-.9.7M1.9 7h1.2" />
    <path d="M6.3 16.9h7.4a2.9 2.9 0 0 0 .6-5.7 3.7 3.7 0 0 0-7.1-.8 2.6 2.6 0 0 0-.9 5.1z" />
  </svg>
);
const ClockIcon = () => (
  <svg viewBox="0 0 20 20" {...stroke} aria-hidden="true">
    <path d="M16.9 10a6.9 6.9 0 1 1-2-4.9" />
    <path d="M15.2 2.4l-.2 3-3-.3" />
    <path d="M10 6.4v3.8l2.6 1.3" />
  </svg>
);
const BarsIcon = () => (
  <svg viewBox="0 0 20 20" {...stroke} aria-hidden="true">
    <path d="M2.9 17h14.3" />
    <path d="M5.8 17v-4.4" />
    <path d="M10 17V7.1" />
    <path d="M14.3 17v-6.8" />
  </svg>
);
const FlagIcon = () => (
  <svg viewBox="0 0 20 20" {...stroke} aria-hidden="true">
    <path d="M5.1 17.2V3.3" />
    <path d="M5.1 4.2c2.4-1.4 4.5 1 7-.4 1-.6 2-.8 2.8-.7v6.2c-.8-.1-1.8.1-2.8.7-2.5 1.4-4.6-1-7 .4" />
  </svg>
);
const FlaskIcon = () => (
  <svg viewBox="0 0 20 20" {...stroke} aria-hidden="true">
    <path d="M7.6 3.1h4.8" />
    <path d="M8.6 3.1v4.7l-3.9 6.8a1.9 1.9 0 0 0 1.7 2.9h7.2a1.9 1.9 0 0 0 1.7-2.9l-3.9-6.8V3.1" />
    <path d="M6.2 13.1c1.3-.6 2.4.4 3.7 0 1.3-.3 2.5-.5 3.5.1" />
  </svg>
);
const ThermoDoodle = () => (
  <svg
    viewBox="0 0 16 24"
    className="brand-doodle"
    {...stroke}
    aria-hidden="true"
  >
    <path d="M8 2.1a2.6 2.6 0 0 1 2.6 2.6v8.8a4.5 4.5 0 1 1-5.2.1V4.7A2.6 2.6 0 0 1 8 2.1z" />
    <path d="M8 8.4v9.4" />
    <circle cx="8" cy="17.7" r="1.9" fill="currentColor" stroke="none" />
    <path d="M13 5.8h1.5M13 9.4h1.5" />
  </svg>
);
const Spark = () => (
  <svg viewBox="0 0 12 12" className="nav-spark" {...stroke} aria-hidden="true">
    <path d="M6 1.4v9.1M1.7 3.7l8.5 4.5M10.3 3.8L1.8 8.3" />
  </svg>
);

const NAV = [
  { id: "dashboard", label: "Dashboard", icon: GaugeIcon },
  { id: "forecast", label: "Forecast", icon: SunCloudIcon },
  { id: "history", label: "History", icon: ClockIcon },
  { id: "metrics", label: "Metrics", icon: BarsIcon },
  { id: "leaderboard", label: "Leaderboard", icon: FlagIcon },
  { id: "pipeline", label: "Pipeline", icon: FlaskIcon },
];

export default function Sidebar({ activePage, onNavigate }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <ThermoDoodle />
        <div>
          <h1>
            ThermoSense
            <svg
              viewBox="0 0 120 8"
              className="brand-underline"
              aria-hidden="true"
            >
              <path
                d="M2 5.4C22 2.4 44 6.6 66 4.1 86 2 104 3.1 118 4.9"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
              />
            </svg>
          </h1>
          <p>local temperature, understood.</p>
        </div>
      </div>
      <nav className="sidebar-nav" aria-label="Main navigation">
        {NAV.map(({ id, label, icon: Icon }) => {
          const active = activePage === id;
          return (
            <button
              key={id}
              className={`nav-link ${active ? "active" : ""}`}
              aria-current={active ? "page" : undefined}
              onClick={() => onNavigate(id)}
            >
              <Icon aria-hidden="true" />
              <span className="nav-label">{label}</span>
              {active && <Spark />}
            </button>
          );
        })}
      </nav>
      <div className="sidebar-footer">
        <SensorStatus />
        <div className="sidebar-note">
          <p>the field notebook</p>
          <span>observe. compare. refine.</span>
        </div>
        <a href="/docs" target="_blank" rel="noreferrer">
          <BookOpen size={15} aria-hidden="true" />
          API documentation <span aria-hidden="true">↗</span>
        </a>
      </div>
    </aside>
  );
}
