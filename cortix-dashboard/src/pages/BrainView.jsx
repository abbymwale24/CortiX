import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, LineChart, Line,
} from "recharts";
import { Brain, Zap, Activity, Eye } from "lucide-react";

const API_BASE = "http://localhost:8000";

function BrainView() {
  const [weights, setWeights] = useState(null);
  const [heatmap, setHeatmap] = useState(null);
  const [history, setHistory] = useState([]);
  const [selectedModule, setSelectedModule] = useState(0);

  const fetchWeights = useCallback(() => {
    axios.get(`${API_BASE}/api/brain/weights`)
      .then((res) => setWeights(res.data))
      .catch(() => {});
  }, []);

  const fetchHeatmap = useCallback(() => {
    axios.get(`${API_BASE}/api/brain/weights/heatmap?module=${selectedModule}`)
      .then((res) => setHeatmap(res.data))
      .catch(() => {});
  }, [selectedModule]);

  const fetchHistory = useCallback(() => {
    axios.get(`${API_BASE}/api/brain/weights/history`)
      .then((res) => setHistory(res.data))
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetchWeights();
    fetchHeatmap();
    fetchHistory();
    const interval = setInterval(() => {
      fetchWeights();
      fetchHeatmap();
      fetchHistory();
    }, 5000);
    return () => clearInterval(interval);
  }, [fetchWeights, fetchHeatmap, fetchHistory]);

  // Prepare histogram data for the selected module
  const histogramData = weights?.modules?.[selectedModule]
    ? weights.modules[selectedModule].histogram.map((count, i) => ({
        bin: weights.modules[selectedModule].histogram_bins[i].toFixed(4),
        count,
      }))
    : [];

  // Prepare history chart data
  const historyData = history.map((snap) => {
    const entry = { time: new Date(snap.timestamp * 1000).toLocaleTimeString() };
    snap.modules.forEach((m) => {
      entry[`M${m.module_id} mean`] = m.weight_mean;
    });
    return entry;
  });

  const moduleColors = ["#ef4444", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6"];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "28px" }}>
      {/* Header */}
      <div>
        <h2 style={{ fontSize: "28px", fontWeight: "bold", margin: "0 0 8px 0" }}>
          Brain State Visualization
        </h2>
        <p style={{ color: "#9ca3af", margin: 0 }}>
          Live synaptic weight evolution across the Hebbian SNN ensemble.
        </p>
      </div>

      {/* Summary Stats Cards */}
      {weights && (
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
          gap: "16px",
        }}>
          <StatCard
            icon={<Brain size={22} />}
            label="EVENTS PROCESSED"
            value={weights.total_events_processed.toLocaleString()}
            color="#3b82f6"
          />
          <StatCard
            icon={<Zap size={22} />}
            label="LEARNING RATE (η)"
            value={weights.metaplasticity.current_eta.toExponential(3)}
            color="#f59e0b"
          />
          <StatCard
            icon={<Activity size={22} />}
            label="ADAPTATION RATIO"
            value={`${(weights.metaplasticity.adaptation_ratio * 100).toFixed(1)}%`}
            color="#10b981"
          />
          <StatCard
            icon={<Eye size={22} />}
            label="AVG SPARSITY"
            value={`${(weights.modules.reduce((a, m) => a + m.sparsity, 0) / weights.modules.length * 100).toFixed(1)}%`}
            color="#8b5cf6"
          />
        </div>
      )}

      {/* Module Selector */}
      <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
        {[0, 1, 2, 3, 4].map((m) => (
          <button
            key={m}
            onClick={() => setSelectedModule(m)}
            style={{
              padding: "10px 20px",
              borderRadius: "8px",
              border: selectedModule === m ? `2px solid ${moduleColors[m]}` : "1px solid #374151",
              backgroundColor: selectedModule === m ? `${moduleColors[m]}20` : "#11141e",
              color: selectedModule === m ? moduleColors[m] : "#9ca3af",
              cursor: "pointer",
              fontWeight: selectedModule === m ? "bold" : "normal",
              fontSize: "13px",
              transition: "all 0.2s",
            }}
          >
            Module {m}
            {weights?.modules?.[m] && (
              <span style={{ marginLeft: "8px", fontSize: "11px", opacity: 0.7 }}>
                μ={weights.modules[m].weight_mean.toFixed(4)}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Two-column layout: Histogram + Heatmap */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px" }}>
        {/* Weight Distribution Histogram */}
        <div style={panelStyle}>
          <h3 style={panelTitle}>
            Weight Distribution — Module {selectedModule}
          </h3>
          <div style={{ width: "100%", height: 280 }}>
            <ResponsiveContainer>
              <BarChart data={histogramData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="bin"
                  stroke="#6b7280"
                  tick={{ fontSize: 10 }}
                  interval="preserveStartEnd"
                />
                <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#11141e", borderColor: "#374151", fontSize: "12px" }}
                />
                <Bar
                  dataKey="count"
                  fill={moduleColors[selectedModule]}
                  radius={[4, 4, 0, 0]}
                  name="Synapse Count"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Module detail stats */}
          {weights?.modules?.[selectedModule] && (
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(3, 1fr)",
              gap: "12px",
              marginTop: "16px",
              fontSize: "12px",
              color: "#9ca3af",
            }}>
              <div>
                <div style={{ color: "#6b7280" }}>Top Synapses</div>
                <div style={{ color: "#10b981", fontWeight: "bold", fontSize: "16px" }}>
                  {weights.modules[selectedModule].top_synapses}
                </div>
              </div>
              <div>
                <div style={{ color: "#6b7280" }}>Dead Synapses</div>
                <div style={{ color: "#ef4444", fontWeight: "bold", fontSize: "16px" }}>
                  {weights.modules[selectedModule].dead_synapses}
                </div>
              </div>
              <div>
                <div style={{ color: "#6b7280" }}>Weight Range</div>
                <div style={{ fontWeight: "bold", fontSize: "14px" }}>
                  [{weights.modules[selectedModule].weight_min.toFixed(4)},
                  {weights.modules[selectedModule].weight_max.toFixed(4)}]
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Synaptic Weight Heatmap */}
        <div style={panelStyle}>
          <h3 style={panelTitle}>
            Synaptic Weight Heatmap — Module {selectedModule}
          </h3>
          {heatmap?.data ? (
            <div style={{
              display: "grid",
              gridTemplateColumns: `repeat(${heatmap.size}, 1fr)`,
              gap: "1px",
              width: "100%",
              aspectRatio: "1",
              borderRadius: "8px",
              overflow: "hidden",
            }}>
              {heatmap.data.flat().map((val, i) => (
                <div
                  key={i}
                  style={{
                    backgroundColor: heatmapColor(val),
                    aspectRatio: "1",
                  }}
                />
              ))}
            </div>
          ) : (
            <div style={{ color: "#6b7280", padding: "40px", textAlign: "center" }}>
              Loading heatmap...
            </div>
          )}
          <div style={{
            display: "flex",
            justifyContent: "space-between",
            marginTop: "12px",
            fontSize: "11px",
            color: "#6b7280",
          }}>
            <span>Weak (cold)</span>
            <div style={{
              flex: 1,
              height: "8px",
              margin: "0 12px",
              borderRadius: "4px",
              background: "linear-gradient(to right, #1e1b4b, #3b0764, #7c2d12, #dc2626, #fbbf24)",
            }} />
            <span>Strong (hot)</span>
          </div>
        </div>
      </div>

      {/* Weight Evolution Over Time */}
      <div style={panelStyle}>
        <h3 style={panelTitle}>Weight Mean Evolution Over Time</h3>
        <div style={{ width: "100%", height: 250 }}>
          <ResponsiveContainer>
            <LineChart data={historyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#6b7280" tick={{ fontSize: 11 }} />
              <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: "#11141e", borderColor: "#374151", fontSize: "12px" }}
              />
              {moduleColors.map((color, i) => (
                <Line
                  key={i}
                  type="monotone"
                  dataKey={`M${i} mean`}
                  stroke={color}
                  strokeWidth={2}
                  dot={false}
                  name={`Module ${i}`}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

/* ── Helper Components ── */

function StatCard({ icon, label, value, color }) {
  return (
    <div style={{
      backgroundColor: "#11141e",
      borderRadius: "12px",
      border: "1px solid #1f2937",
      padding: "18px",
      display: "flex",
      alignItems: "center",
      gap: "14px",
      boxShadow: `0 2px 12px rgba(0, 0, 0, 0.3), inset 0 0 8px ${color}10`,
    }}>
      <div style={{
        padding: "10px",
        borderRadius: "10px",
        backgroundColor: `${color}15`,
        color: color,
        display: "flex",
      }}>
        {icon}
      </div>
      <div>
        <div style={{ fontSize: "11px", color: "#6b7280", letterSpacing: "0.5px" }}>{label}</div>
        <div style={{ fontSize: "20px", fontWeight: "bold", marginTop: "2px" }}>{value}</div>
      </div>
    </div>
  );
}

/* ── Heatmap Color Function ── */

function heatmapColor(value) {
  // Map 0..1 to a perceptual color scale: dark purple → red → yellow
  const v = Math.max(0, Math.min(1, value));
  if (v < 0.25) {
    // Dark indigo to purple
    const t = v / 0.25;
    return `rgb(${Math.round(30 + 29 * t)}, ${Math.round(27 + 0 * t)}, ${Math.round(75 + 25 * t)})`;
  } else if (v < 0.5) {
    // Purple to dark red
    const t = (v - 0.25) / 0.25;
    return `rgb(${Math.round(59 + 65 * t)}, ${Math.round(27 - 12 * t)}, ${Math.round(100 - 82 * t)})`;
  } else if (v < 0.75) {
    // Dark red to red
    const t = (v - 0.5) / 0.25;
    return `rgb(${Math.round(124 + 96 * t)}, ${Math.round(15 + 30 * t)}, ${Math.round(18 + 0 * t)})`;
  } else {
    // Red to amber/yellow
    const t = (v - 0.75) / 0.25;
    return `rgb(${Math.round(220 + 31 * t)}, ${Math.round(45 + 146 * t)}, ${Math.round(18 + 18 * t)})`;
  }
}

/* ── Styles ── */

const panelStyle = {
  backgroundColor: "#11141e",
  borderRadius: "16px",
  border: "1px solid #1f2937",
  padding: "24px",
};

const panelTitle = {
  fontSize: "18px",
  fontWeight: "bold",
  margin: "0 0 20px 0",
};

export default BrainView;
