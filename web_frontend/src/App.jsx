import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine
} from 'recharts';
import {
  TrendingUp, TrendingDown, Minus, Activity, Wind, Search,
  BrainCircuit, RefreshCw, Loader2, CheckCircle2, AlertTriangle,
  BarChart3, Cpu, GitBranch, Sparkles
} from 'lucide-react';
import './App.css';

const API_BASE = 'http://localhost:8000/api';

const SIGNAL = {
  UP:       { text: 'text-emerald-400', dot: 'bg-emerald-400', stroke: '#34d399', pill: 'up',       icon: TrendingUp,    label: 'BULLISH' },
  DOWN:     { text: 'text-rose-400',    dot: 'bg-rose-400',    stroke: '#fb7185', pill: 'down',     icon: TrendingDown,  label: 'BEARISH' },
  SIDEWAYS: { text: 'text-slate-400',   dot: 'bg-slate-400',   stroke: '#94a3b8', pill: 'sideways', icon: Minus,         label: 'NEUTRAL' },
};

const Sig = ({ signal, size = 'sm' }) => {
  const s = SIGNAL[signal] || SIGNAL.SIDEWAYS;
  const dotSize = size === 'lg' ? 'w-3 h-3' : 'w-2 h-2';
  return <span className={`inline-block ${dotSize} rounded-full ${s.dot} shadow-lg`} style={{ boxShadow: `0 0 8px ${s.stroke}40` }} />;
};

const StatCard = ({ icon: Icon, label, value, sub, accent }) => (
  <div className="glow-card stat-shine p-5 flex flex-col gap-2">
    <div className="flex items-center gap-2 text-[11px] uppercase tracking-widest text-slate-500 font-semibold">
      <Icon className="w-3.5 h-3.5" style={{ color: accent }} /> {label}
    </div>
    <div className="text-2xl font-light text-white font-mono tracking-tight">{value}</div>
    {sub && <div className="text-xs text-slate-500 font-mono">{sub}</div>}
  </div>
);

function App() {
  const [tickers, setTickers] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTicker, setSelectedTicker] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [overview, setOverview] = useState([]);
  const [overviewLoading, setOverviewLoading] = useState(true);
  const [trainStatus, setTrainStatus] = useState({ status: 'idle', message: '', log: [] });
  const [trainBtnLoading, setTrainBtnLoading] = useState(false);
  const [showTrainLog, setShowTrainLog] = useState(false);

  useEffect(() => {
    axios.get(`${API_BASE}/tickers`).then(r => {
      const t = r.data.tickers || [];
      setTickers(t);
      if (t.length && !selectedTicker) setSelectedTicker(t[0]);
    }).catch(() => setTickers([]));
  }, []);

  const fetchOverview = useCallback(async () => {
    setOverviewLoading(true);
    try {
      const r = await axios.get(`${API_BASE}/overview`);
      setOverview(r.data.rows || []);
    } catch { /* ignore */ }
    setOverviewLoading(false);
  }, []);

  useEffect(() => { fetchOverview(); }, [fetchOverview]);

  useEffect(() => {
    if (trainStatus.status !== 'running') return;
    const id = setInterval(async () => {
      try {
        const r = await axios.get(`${API_BASE}/train/status`);
        setTrainStatus(r.data);
        if (r.data.status !== 'running') { clearInterval(id); fetchOverview(); }
      } catch { /* ignore */ }
    }, 2500);
    return () => clearInterval(id);
  }, [trainStatus.status, fetchOverview]);

  const triggerTrain = async () => {
    setTrainBtnLoading(true);
    try {
      const r = await axios.post(`${API_BASE}/train`);
      setTrainStatus(r.data);
      setShowTrainLog(true);
    } catch (e) {
      setTrainStatus({ status: 'error', message: e.response?.data?.detail || 'Failed', log: [] });
      setShowTrainLog(true);
    }
    setTrainBtnLoading(false);
  };

  useEffect(() => {
    if (!selectedTicker) return;
    setLoading(true);
    setError(null);
    axios.get(`${API_BASE}/predict/${selectedTicker}`)
      .then(r => setPredictionData(r.data))
      .catch(e => { setError(e.response?.data?.detail || 'Failed'); setPredictionData(null); })
      .finally(() => setLoading(false));
  }, [selectedTicker]);

  const chartData = (() => {
    if (!predictionData) return [];
    const d = [];
    predictionData.history.dates.forEach((dt, i) =>
      d.push({ date: dt.substring(5), Historical: predictionData.history.prices[i], Forecast: null })
    );
    const last = predictionData.history.prices[predictionData.history.prices.length - 1];
    if (d.length) d[d.length - 1].Forecast = last;
    predictionData.forecast.dates.forEach((dt, i) =>
      d.push({ date: dt.substring(5), Historical: null, Forecast: predictionData.forecast.prices[i] })
    );
    return d;
  })();

  const filtered = tickers.filter(t => t.toLowerCase().includes(searchTerm.toLowerCase())).slice(0, 100);
  const sigMap = {};
  overview.forEach(r => { sigMap[r.ticker] = r.forecast_signal; });

  return (
    <div className="min-h-screen relative">
      {/* Background layers */}
      <div className="mesh-bg" />
      <div className="noise-overlay" />

      {/* Navbar */}
      <nav className="navbar-glass sticky top-0 z-50">
        <div className="max-w-[1400px] mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 via-purple-500 to-fuchsia-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <Activity className="text-white w-5 h-5" />
            </div>
            <div>
              <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-white via-indigo-200 to-purple-300 bg-clip-text text-transparent">
                CausalFolio
              </span>
              <span className="ml-2 text-[10px] font-mono text-indigo-400/60 tracking-widest">v3.2</span>
            </div>
            <span className="ml-3 px-2.5 py-1 rounded-lg text-[10px] font-bold tracking-wider bg-indigo-500/10 text-indigo-300/80 border border-indigo-500/20">
              GNN + TCN
            </span>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-xs text-emerald-400/80">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              LIVE
            </div>
            <button
              onClick={triggerTrain}
              disabled={trainStatus.status === 'running' || trainBtnLoading}
              className="btn-glow flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold bg-gradient-to-r from-indigo-500 to-purple-600 text-white border border-indigo-400/30 shadow-lg shadow-indigo-500/20 disabled:opacity-60 disabled:cursor-wait"
            >
              {trainStatus.status === 'running' ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : trainStatus.status === 'success' ? (
                <CheckCircle2 className="w-4 h-4" />
              ) : (
                <Cpu className="w-4 h-4" />
              )}
              {trainStatus.status === 'running' ? 'Training...' : 'Train Model'}
            </button>
          </div>
        </div>
      </nav>

      {/* Training status banner */}
      {trainStatus.status !== 'idle' && (
        <div className={`max-w-[1400px] mx-auto px-6 mt-4`}>
          <div className={`rounded-2xl border p-4 flex items-center justify-between ${
            trainStatus.status === 'running' ? 'bg-indigo-500/10 border-indigo-500/20' :
            trainStatus.status === 'success' ? 'bg-emerald-500/10 border-emerald-500/20' :
            'bg-rose-500/10 border-rose-500/20'
          }`}>
            <div className="flex items-center gap-3">
              {trainStatus.status === 'running' ? <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" /> :
               trainStatus.status === 'success' ? <CheckCircle2 className="w-4 h-4 text-emerald-400" /> :
               <AlertTriangle className="w-4 h-4 text-rose-400" />}
              <span className={`text-sm font-medium ${
                trainStatus.status === 'running' ? 'text-indigo-300' :
                trainStatus.status === 'success' ? 'text-emerald-300' : 'text-rose-300'
              }`}>
                {trainStatus.message || (trainStatus.status === 'running' ? 'Daily model update in progress...' : '')}
              </span>
            </div>
            <div className="flex items-center gap-3">
              {trainStatus.started_at && <span className="text-[10px] text-slate-500 font-mono">{trainStatus.started_at}</span>}
              <button onClick={() => setShowTrainLog(v => !v)} className="text-[11px] text-slate-400 hover:text-white px-2.5 py-1 rounded-lg bg-white/5 border border-white/10 transition-colors">
                {showTrainLog ? 'Hide Log' : 'View Log'}
              </button>
            </div>
          </div>
          {showTrainLog && trainStatus.log?.length > 0 && (
            <pre className="mt-2 train-log custom-scrollbar">{trainStatus.log.join('\n')}</pre>
          )}
        </div>
      )}

      {/* Main layout */}
      <main className="max-w-[1400px] mx-auto px-6 py-6 flex gap-6 min-h-[calc(100vh-4rem)]">

        {/* ── Left: Sidebar ── */}
        <aside className="w-72 flex-shrink-0 flex flex-col gap-4">
          {/* Search */}
          <div className="glow-card p-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-600" />
              <input
                type="text"
                placeholder="Search symbol..."
                className="w-full bg-[#0a0a0c] border border-white/8 rounded-xl pl-10 pr-4 py-2.5 text-sm focus:outline-none focus:border-indigo-500/60 focus:ring-1 focus:ring-indigo-500/30 transition-all text-white placeholder-slate-600"
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
              />
            </div>
          </div>

          {/* Ticker list */}
          <div className="glow-card flex-1 overflow-hidden flex flex-col">
            <div className="px-4 pt-4 pb-2 flex items-center justify-between">
              <span className="text-[11px] uppercase tracking-widest text-slate-500 font-semibold">Universe</span>
              <span className="text-[10px] text-slate-600 font-mono">{filtered.length}</span>
            </div>
            <div className="flex-1 overflow-y-auto custom-scrollbar px-2 pb-2">
              {filtered.map((t, idx) => {
                const sig = sigMap[t];
                const active = selectedTicker === t;
                return (
                  <button
                    key={t}
                    onClick={() => setSelectedTicker(t)}
                    className={`ticker-row w-full text-left px-3 py-2.5 rounded-lg flex items-center gap-2.5 ${
                      active ? 'active bg-indigo-500/10 text-white' : 'text-slate-400 hover:text-slate-200'
                    }`}
                    style={{ animationDelay: `${idx * 15}ms` }}
                  >
                    {sig && <Sig signal={sig} />}
                    <span className="font-mono text-sm font-medium">{t.replace('.BO', '')}</span>
                  </button>
                );
              })}
              {filtered.length === 0 && (
                <div className="empty-state py-12 text-sm">No symbols found</div>
              )}
            </div>
          </div>
        </aside>

        {/* ── Right: Dashboard ── */}
        <div className="flex-1 flex flex-col gap-5 min-w-0">

          {/* Overview table */}
          <div className="glow-card overflow-hidden">
            <div className="px-5 pt-5 pb-3 flex items-center justify-between">
              <h2 className="text-[11px] uppercase tracking-widest text-slate-500 font-semibold flex items-center gap-2">
                <BarChart3 className="w-3.5 h-3.5" /> Market Overview
              </h2>
              <button onClick={fetchOverview} className="btn-glow flex items-center gap-1.5 text-[11px] text-slate-400 hover:text-white px-2.5 py-1.5 rounded-lg bg-white/5 border border-white/8 transition-colors">
                <RefreshCw className={`w-3 h-3 ${overviewLoading ? 'animate-spin' : ''}`} /> Refresh
              </button>
            </div>
            <div className="overflow-x-auto max-h-[380px] overflow-y-auto custom-scrollbar overview-table">
              <table className="w-full text-sm">
                <thead className="sticky top-0 z-10" style={{ background: 'var(--cf-card)' }}>
                  <tr className="text-[10px] uppercase tracking-widest text-slate-600 border-b border-white/5">
                    <th className="px-5 py-3 text-left font-semibold">Symbol</th>
                    <th className="px-4 py-3 text-right font-semibold">Price</th>
                    <th className="px-4 py-3 text-center font-semibold">Signal</th>
                    <th className="px-4 py-3 text-right font-semibold">Move</th>
                    <th className="px-4 py-3 text-right font-semibold">Volatility</th>
                  </tr>
                </thead>
                <tbody>
                  {overview.map((r, i) => (
                    <tr
                      key={r.ticker}
                      onClick={() => setSelectedTicker(r.ticker)}
                      className={`table-row-enter cursor-pointer border-b border-white/3 ${
                        selectedTicker === r.ticker ? 'bg-indigo-500/8' : 'hover:bg-white/3'
                      }`}
                      style={{ animationDelay: `${i * 20}ms` }}
                    >
                      <td className="px-5 py-2.5">
                        <div className="flex items-center gap-2.5">
                          <Sig signal={r.forecast_signal} />
                          <span className="font-mono font-medium text-slate-200">{r.ticker.replace('.BO', '')}</span>
                        </div>
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-slate-300">
                        ₹{r.current_price.toLocaleString()}
                      </td>
                      <td className="px-4 py-2.5 text-center">
                        <span className={`signal-pill ${SIGNAL[r.forecast_signal]?.pill || 'sideways'}`}>
                          {SIGNAL[r.forecast_signal]?.label || 'NEUTRAL'}
                        </span>
                      </td>
                      <td className={`px-4 py-2.5 text-right font-mono font-semibold ${
                        r.expected_move_pct > 0 ? 'text-emerald-400' : r.expected_move_pct < 0 ? 'text-rose-400' : 'text-slate-500'
                      }`}>
                        {r.expected_move_pct > 0 ? '+' : ''}{r.expected_move_pct}%
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-slate-500 text-xs">
                        {r.volatility.toFixed(3)}
                      </td>
                    </tr>
                  ))}
                  {overview.length === 0 && !overviewLoading && (
                    <tr><td colSpan={5} className="empty-state py-10 text-sm">No predictions loaded</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Loading / Error / Detail */}
          {loading ? (
            <div className="glow-card flex-1 flex items-center justify-center min-h-[400px]">
              <div className="flex flex-col items-center gap-4">
                <div className="w-12 h-12 rounded-full border-4 border-indigo-500/20 border-t-indigo-500 animate-spin" />
                <p className="text-slate-500 text-sm font-medium animate-pulse flex items-center gap-2">
                  <GitBranch className="w-4 h-4" /> Running GNN Inference...
                </p>
              </div>
            </div>
          ) : error ? (
            <div className="glow-card flex-1 flex items-center justify-center min-h-[400px] border-rose-500/20">
              <div className="text-center">
                <AlertTriangle className="w-10 h-10 text-rose-400/60 mx-auto mb-3" />
                <p className="text-rose-400 text-sm">{error}</p>
              </div>
            </div>
          ) : predictionData ? (
            <>
              {/* Stat cards */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <StatCard
                  icon={Activity}
                  label="Current Price"
                  value={`₹${predictionData.current_price.toLocaleString()}`}
                  sub={predictionData.ticker}
                  accent="#6366f1"
                />
                <StatCard
                  icon={SIGNAL[predictionData.forecast_signal]?.icon || Minus}
                  label="AI Forecast (5D)"
                  value={
                    <span className={SIGNAL[predictionData.forecast_signal]?.text || 'text-slate-400'}>
                      {predictionData.forecast_signal}
                    </span>
                  }
                  sub={`${predictionData.expected_move_pct > 0 ? '+' : ''}${predictionData.expected_move_pct}% expected`}
                  accent={SIGNAL[predictionData.forecast_signal]?.stroke || '#94a3b8'}
                />
                <StatCard
                  icon={Sparkles}
                  label="Sentiment"
                  value={`${predictionData.sentiment_score > 0 ? '+' : ''}${predictionData.sentiment_score}`}
                  sub="FinBERT / momentum proxy"
                  accent={predictionData.sentiment_score > 0 ? '#34d399' : '#fb7185'}
                />
                <StatCard
                  icon={Wind}
                  label="Volatility"
                  value={predictionData.volatility.toFixed(4)}
                  sub={`Betti-0: ${predictionData.tda_betti_0}`}
                  accent="#a855f7"
                />
              </div>

              {/* Chart */}
              <div className="glow-card p-6 flex-1 min-h-[420px] dot-pattern">
                <div className="flex items-center justify-between mb-5">
                  <div>
                    <h3 className="text-xl font-bold text-white flex items-center gap-3">
                      {predictionData.ticker}
                      <span className={`signal-pill ${SIGNAL[predictionData.forecast_signal]?.pill || 'sideways'}`}>
                        {SIGNAL[predictionData.forecast_signal]?.label || 'NEUTRAL'}
                      </span>
                    </h3>
                    <p className="text-xs text-slate-500 mt-1 font-mono">GNN + TCN Spatial-Temporal Projection</p>
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={360}>
                  <LineChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                    <XAxis dataKey="date" stroke="#475569" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis domain={['auto', 'auto']} stroke="#475569" tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={v => `₹${v}`} axisLine={false} tickLine={false} width={75} />
                    <Tooltip
                      contentStyle={{ background: '#15151a', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '12px', color: '#e2e8f0', boxShadow: '0 12px 40px rgba(0,0,0,0.5)' }}
                      itemStyle={{ fontFamily: 'monospace', fontSize: 12 }}
                      labelStyle={{ color: '#64748b', marginBottom: 4 }}
                    />
                    <ReferenceLine
                      x={predictionData.history.dates[predictionData.history.dates.length - 1]?.substring(5)}
                      stroke="rgba(99,102,241,0.3)"
                      strokeDasharray="4 4"
                      label={{ position: 'top', value: 'TODAY', fill: '#6366f1', fontSize: 10, fontWeight: 600 }}
                    />
                    <Line type="monotone" dataKey="Historical" stroke="#6366f1" strokeWidth={2.5} dot={false} activeDot={{ r: 5, fill: '#6366f1', stroke: '#fff', strokeWidth: 2 }} />
                    <Line
                      type="monotone"
                      dataKey="Forecast"
                      stroke={SIGNAL[predictionData.forecast_signal]?.stroke || '#94a3b8'}
                      strokeWidth={3}
                      strokeDasharray="8 4"
                      dot={{ r: 4, strokeWidth: 2 }}
                      className={predictionData.forecast_signal === 'UP' ? 'forecast-glow' : predictionData.forecast_signal === 'DOWN' ? 'forecast-glow down' : ''}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </>
          ) : (
            <div className="glow-card flex-1 flex items-center justify-center min-h-[400px]">
              <div className="empty-state">
                <Activity className="w-12 h-12 text-slate-700 mb-3" />
                <p className="text-slate-600 text-sm">Select a ticker to view predictions</p>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;