import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Forecast from './pages/Forecast';
import History from './pages/History';
import Metrics from './pages/Metrics';
import Leaderboard from './pages/Leaderboard';
import Pipeline from './pages/Pipeline';

const PAGES = {
  dashboard: Dashboard,
  forecast: Forecast,
  history: History,
  metrics: Metrics,
  leaderboard: Leaderboard,
  pipeline: Pipeline,
};

export default function App() {
  const [page, setPage] = useState('dashboard');
  const Page = PAGES[page] || Dashboard;

  return (
    <div className="app-layout">
      <Sidebar activePage={page} onNavigate={setPage} />
      <main className="main-content">
        <Page onNavigate={setPage} />
      </main>
    </div>
  );
}
