import React from "react";
import {
  LayoutDashboard,
  CloudSun,
  History,
  BarChart3,
  Settings2,
  BookOpen,
  Trophy,
  Thermometer,
} from "lucide-react";
const NAV = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "forecast", label: "Forecast", icon: CloudSun },
  { id: "history", label: "History", icon: History },
  { id: "metrics", label: "Metrics", icon: BarChart3 },
  { id: "leaderboard", label: "Leaderboard", icon: Trophy },
  { id: "pipeline", label: "Pipeline", icon: Settings2 },
];
export default function Sidebar({ activePage, onNavigate }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-symbol">
          <Thermometer aria-hidden="true" />
        </div>
        <div>
          <h1>
            ThermoSense<span className="brand-period">.</span>
          </h1>
          <p>Local temperature, understood.</p>
        </div>
      </div>
      <div className="nav-section-label">Workspace</div>
      <nav className="sidebar-nav" aria-label="Main navigation">
        {NAV.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            className={`nav-link ${activePage === id ? "active" : ""}`}
            aria-current={activePage === id ? "page" : undefined}
            onClick={() => onNavigate(id)}
          >
            <Icon aria-hidden="true" />
            <span>{label}</span>
            <span className="nav-index" aria-hidden="true">
              {String(NAV.findIndex((n) => n.id === id) + 1).padStart(2, "0")}
            </span>
          </button>
        ))}
      </nav>
      <div className="sidebar-footer">
        <p>THE FIELD NOTEBOOK</p>
        <span>Observe. Compare. Refine.</span>
        <a href="/docs" target="_blank" rel="noreferrer">
          <BookOpen size={15} aria-hidden="true" />
          API documentation <span aria-hidden="true">↗</span>
        </a>
      </div>
    </aside>
  );
}
