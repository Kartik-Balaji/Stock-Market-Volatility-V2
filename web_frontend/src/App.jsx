import { useState, useEffect } from 'react';
import axios from 'axios';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import { TrendingUp, TrendingDown, Minus, Activity, Wind, Search } from 'lucide-react';
import './App.css';

// The URL where FastAPI is running locally
const API_BASE = 'http://localhost:8000/api';

function App() {
  const [tickers, setTickers] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTicker, setSelectedTicker] = useState('RELIANCE.BO');
  const [predictionData, setPredictionData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch available tickers on load
  useEffect(() => {
    const fetchTickers = async () => {
      try {
        const res = await axios.get(`${API_BASE}/tickers`);
        setTickers(res.data.tickers || []);
      } catch (err) {
        console.error("Failed to load tickers:", err);
        // Fallback for development if backend isn't running yet
        setTickers(['RELIANCE.BO', 'TCS.BO', 'INFY.BO', 'HDFCBANK.BO']);
      }
    };
    fetchTickers();
  }, []);

  // Fetch prediction when a ticker is selected
  useEffect(() => {
    if (!selectedTicker) return;

    const fetchPrediction = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await axios.get(`${API_BASE}/predict/${selectedTicker}`);
        setPredictionData(res.data);
      } catch (err) {
        setError(err.response?.data?.detail || "Failed to fetch prediction");
        setPredictionData(null);
      } finally {
        setLoading(false);
      }
    };
    
    fetchPrediction();
  }, [selectedTicker]);

  // Combine history and forecast into one array for Recharts
  const formatChartData = (data) => {
    if (!data) return [];
    
    const chartData = [];
    
    // Add history
    data.history.dates.forEach((date, i) => {
      chartData.push({
        date: date.substring(5), // e.g. "03-05"
        Historical: data.history.prices[i],
        Forecast: null // Null so the line doesn't connect
      });
    });
    
    // To make the lines connect perfectly, add the last historical point as the first forecast point
    const lastHistPrice = data.history.prices[data.history.prices.length - 1];
    const lastHistDate = data.history.dates[data.history.dates.length - 1].substring(5);
    
    // We update the last element to have both
    chartData[chartData.length - 1].Forecast = lastHistPrice;

    // Add forecast dates
    data.forecast.dates.forEach((date, i) => {
      chartData.push({
        date: date.substring(5),
        Historical: null,
        Forecast: data.forecast.prices[i]
      });
    });

    return chartData;
  };

  const filteredTickers = tickers.filter(t => 
    t.toLowerCase().includes(searchTerm.toLowerCase())
  ).slice(0, 100); // Limit to 100 for performance

  return (
    <div className="min-h-screen bg-[#0a0a0c] text-slate-200 font-sans selection:bg-indigo-500/30">
      
      {/* Top Navbar */}
      <nav className="border-b border-white/10 bg-[#0f0f13]/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                <Activity className="text-white w-5 h-5" />
              </div>
              <span className="font-bold text-xl tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
                CausalFolio
              </span>
              <span className="ml-2 px-2 py-0.5 rounded-full text-[10px] font-medium bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                v3.0 Topological
              </span>
            </div>
            <div className="flex items-center space-x-4">
               <div className="flex items-center gap-2 text-sm text-emerald-400 bg-emerald-400/10 px-3 py-1.5 rounded-full border border-emerald-400/20">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                  </span>
                  API Connected
               </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col lg:flex-row gap-8">
        
        {/* Sidebar Structure - Ticker Selection */}
        <div className="w-full lg:w-80 flex flex-col gap-4 flex-shrink-0">
          <div className="bg-[#15151a] border border-white/5 rounded-2xl p-4 shadow-xl shadow-black/40">
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
              <Search className="w-4 h-4" /> Market Universe
            </h2>
            
            <div className="relative mb-4">
              <input 
                type="text" 
                placeholder="Search symbol (e.g. TCS)"
                className="w-full bg-[#0a0a0c] border border-white/10 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all text-white placeholder-slate-600"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            
            <div className="space-y-1 h-[600px] overflow-y-auto pr-2 custom-scrollbar">
              {filteredTickers.map((ticker) => (
                <button
                  key={ticker}
                  onClick={() => setSelectedTicker(ticker)}
                  className={`w-full text-left px-4 py-3 rounded-xl transition-all duration-200 flex items-center justify-between ${
                    selectedTicker === ticker 
                      ? 'bg-gradient-to-r from-indigo-500/20 to-purple-500/10 border border-indigo-500/30 text-white' 
                      : 'hover:bg-white/5 text-slate-400 hover:text-slate-200 border border-transparent'
                  }`}
                >
                  <span className="font-medium font-mono text-sm">{ticker.replace('.BO', '')}</span>
                  <span className="text-[10px] text-slate-500 px-2 py-0.5 rounded-md bg-black/30 bg-clip-padding">.BO</span>
                </button>
              ))}
              {filteredTickers.length === 0 && (
                <div className="text-center py-8 text-slate-500 text-sm">
                  No symbols found.
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Predictive Dashboard Area */}
        <div className="flex-1 flex flex-col gap-6">
          
          {loading ? (
            <div className="h-full flex items-center justify-center min-h-[500px] bg-[#15151a] border border-white/5 rounded-3xl">
              <div className="flex flex-col items-center gap-4">
                <div className="w-10 h-10 border-4 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin"></div>
                <p className="text-slate-400 text-sm animate-pulse">Running GNN Inference...</p>
              </div>
            </div>
          ) : error ? (
            <div className="h-full flex items-center justify-center min-h-[500px] bg-red-500/5 border border-red-500/20 rounded-3xl">
               <p className="text-red-400">{error}</p>
            </div>
          ) : predictionData ? (
            <>
              {/* Top Banner Stats */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                 
                 {/* Current Price */}
                 <div className="bg-[#15151a] border border-white/5 rounded-2xl p-5 relative overflow-hidden group">
                    <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                      <Activity className="w-16 h-16" />
                    </div>
                    <p className="text-slate-500 text-xs font-medium uppercase tracking-wider mb-1">Current Price</p>
                    <div className="flex items-baseline gap-1">
                      <span className="text-3xl font-light text-white font-mono">
                        ₹{predictionData.current_price.toLocaleString()}
                      </span>
                    </div>
                 </div>

                 {/* Directional Forecast */}
                 <div className={`border rounded-2xl p-5 relative overflow-hidden group ${
                    predictionData.forecast_signal === 'UP' ? 'bg-emerald-500/10 border-emerald-500/20' :
                    predictionData.forecast_signal === 'DOWN' ? 'bg-rose-500/10 border-rose-500/20' :
                    'bg-slate-500/10 border-slate-500/20'
                 }`}>
                    <p className="text-slate-500 text-xs font-medium uppercase tracking-wider mb-1">AI Forecast (5D)</p>
                    <div className="flex items-center gap-2 mt-1">
                      {predictionData.forecast_signal === 'UP' && <TrendingUp className="w-8 h-8 text-emerald-400" />}
                      {predictionData.forecast_signal === 'DOWN' && <TrendingDown className="w-8 h-8 text-rose-400" />}
                      {predictionData.forecast_signal === 'SIDEWAYS' && <Minus className="w-8 h-8 text-slate-400" />}
                      <span className={`text-2xl font-bold tracking-tight ${
                        predictionData.forecast_signal === 'UP' ? 'text-emerald-400' :
                        predictionData.forecast_signal === 'DOWN' ? 'text-rose-400' :
                        'text-slate-400'
                      }`}>
                        {predictionData.forecast_signal}
                      </span>
                    </div>
                 </div>

                 {/* FinBERT Sentiment */}
                 <div className="bg-[#15151a] border border-white/5 rounded-2xl p-5">
                    <p className="text-slate-500 text-xs font-medium uppercase tracking-wider mb-1">FinBERT Sentiment</p>
                    <div className="mt-2">
                       <div className="flex justify-between text-sm mb-1 font-mono">
                         <span className="text-white text-xl">{(predictionData.sentiment_score > 0 ? '+' : '')}{predictionData.sentiment_score}</span>
                       </div>
                       <div className="w-full bg-slate-800 rounded-full h-1.5 mt-3 overflow-hidden flex">
                          {/* Neutral bar base */}
                          <div className="w-1/2 flex justify-end">
                            <div className="h-full bg-rose-500" style={{ width: `${predictionData.sentiment_score < 0 ? Math.abs(predictionData.sentiment_score) * 100 : 0}%` }}></div>
                          </div>
                          <div className="w-1/2 flex justify-start">
                            <div className="h-full bg-emerald-500" style={{ width: `${predictionData.sentiment_score > 0 ? predictionData.sentiment_score * 100 : 0}%` }}></div>
                          </div>
                       </div>
                    </div>
                 </div>

                 {/* TDA Features */}
                 <div className="bg-[#15151a] border border-white/5 rounded-2xl p-5">
                    <p className="text-slate-500 text-xs font-medium uppercase tracking-wider mb-1">Topological Phase</p>
                    <div className="flex items-center gap-3 mt-1">
                      <Wind className="w-7 h-7 text-indigo-400" />
                      <div>
                        <span className="text-2xl font-light text-white">Betti-0: {predictionData.tda_betti_0}</span>
                      </div>
                    </div>
                 </div>

              </div>

              {/* Main Chart */}
              <div className="bg-[#15151a] border border-white/5 rounded-3xl p-6 flex-1 min-h-[450px] shadow-2xl relative">
                
                {/* Header inside chart */}
                <div className="flex justify-between items-center mb-6">
                  <div>
                    <h3 className="text-xl font-bold text-white flex items-center gap-3">
                      {predictionData.ticker}
                      <span className="px-2.5 py-1 rounded-md text-xs font-medium bg-white/5 border border-white/10 text-slate-300">
                        BSE Quant Model
                      </span>
                    </h3>
                    <p className="text-sm text-slate-500 mt-1">GNN + TCN Spatial-Temporal Projection</p>
                  </div>
                </div>

                <div className="w-full h-[350px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={formatChartData(predictionData)} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                      <defs>
                        <linearGradient id="colorHist" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2a2a35" vertical={false} />
                      <XAxis 
                        dataKey="date" 
                        stroke="#64748b" 
                        tick={{fill: '#64748b', fontSize: 12}}
                        tickMargin={10}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis 
                        domain={['auto', 'auto']} 
                        stroke="#64748b"
                        tick={{fill: '#64748b', fontSize: 12}}
                        tickFormatter={(val) => `₹${val}`}
                        axisLine={false}
                        tickLine={false}
                        width={80}
                      />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#1e1e24', 
                          border: '1px solid rgba(255,255,255,0.1)',
                          borderRadius: '12px',
                          color: '#fff',
                          boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.5)'
                        }}
                        itemStyle={{ fontFamily: 'monospace' }}
                        labelStyle={{ color: '#94a3b8', marginBottom: '4px' }}
                      />
                      
                      {/* Vertical line separating history from future */}
                      <ReferenceLine 
                        x={predictionData.history.dates[predictionData.history.dates.length - 1].substring(5)} 
                        stroke="#475569" 
                        strokeDasharray="3 3" 
                        label={{ position: 'top', value: 'TODAY', fill: '#94a3b8', fontSize: 10 }} 
                      />

                      <Line 
                        type="monotone" 
                        dataKey="Historical" 
                        stroke="#6366f1" 
                        strokeWidth={3}
                        dot={false}
                        activeDot={{ r: 6, fill: '#6366f1', stroke: '#fff', strokeWidth: 2 }}
                        name="Actual Price"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="Forecast" 
                        stroke={
                          predictionData.forecast_signal === 'UP' ? '#34d399' : 
                          predictionData.forecast_signal === 'DOWN' ? '#fb7185' : '#94a3b8'
                        } 
                        strokeWidth={4}
                        strokeDasharray="5 5"
                        dot={{ r: 4, strokeWidth: 2 }}
                        activeDot={{ r: 7 }}
                        name="Model Forecast"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          ) : null}
        </div>

      </main>
    </div>
  );
}

export default App;
