import React from 'react';
import {
  LayoutDashboard,
  CloudSun,
  History,
  BarChart3,
  Settings2,
  BookOpen,
  Trophy,
} from 'lucide-react';

const NAV = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'forecast', label: 'Forecast', icon: CloudSun },
  { id: 'history', label: 'History', icon: History },
  { id: 'metrics', label: 'Metrics', icon: BarChart3 },
  { id: 'leaderboard', label: 'Leaderboard', icon: Trophy },
  { id: 'pipeline', label: 'Pipeline', icon: Settings2 },
];

export default function Sidebar({ activePage, onNavigate }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <h1>ThermoSense</h1>
        <p>Temperature Intelligence</p>
      </div>

      <nav className="sidebar-nav">
        {NAV.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            className={`nav-link ${activePage === id ? 'active' : ''}`}
            onClick={() => onNavigate(id)}
          >
            <Icon />
            <span>{label}</span>
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <a href="/docs" target="_blank" rel="noreferrer">
          <BookOpen style={{ width: 14, height: 14, marginRight: 6, verticalAlign: -2 }} />
          API Docs
        </a>
      </div>
    </aside>
  );
}
