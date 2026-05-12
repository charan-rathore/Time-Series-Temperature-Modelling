const BASE = '/api';

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  health: () => request('/health'),

  getForecast: (days = 3) => request(`/forecast?days=${days}`),
  postFeedback: (date, actual_temp_c) =>
    request('/forecast/feedback', {
      method: 'POST',
      body: JSON.stringify({ date, actual_temp_c }),
    }),

  getHistory: (start, end) => {
    const params = new URLSearchParams();
    if (start) params.set('start', start);
    if (end) params.set('end', end);
    return request(`/history?${params}`);
  },

  getMetrics: (windowDays = 30) => request(`/metrics?window_days=${windowDays}`),

  getStatus: () => request('/pipeline/status'),
  runBackfill: (startDate, endDate) =>
    request('/pipeline/backfill', {
      method: 'POST',
      body: JSON.stringify({ start_date: startDate || null, end_date: endDate || null }),
    }),
  runDaily: () => request('/pipeline/daily', { method: 'POST' }),
  runTraining: (models, skipMlflow = false) =>
    request('/pipeline/train', {
      method: 'POST',
      body: JSON.stringify({ models, skip_mlflow: skipMlflow }),
    }),
  getLogs: (tail = 200) => request(`/pipeline/logs?tail=${tail}`),
  getMlflowRuns: (limit = 20) => request(`/pipeline/mlflow?limit=${limit}`),

  getLeaderboard: (windowDays = 30, horizon = 1) =>
    request(`/leaderboard?window_days=${windowDays}&horizon=${horizon}`),
  getLeaderboardComparison: (startDate, endDate) => {
    const params = new URLSearchParams();
    if (startDate) params.set('start_date', startDate);
    if (endDate) params.set('end_date', endDate);
    return request(`/leaderboard/comparison?${params}`);
  },
  getLeaderboardStatus: () => request('/leaderboard/status'),
};
