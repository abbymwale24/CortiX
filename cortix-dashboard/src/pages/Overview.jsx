import React, { useState, useEffect } from "react";
import axios from "axios";
import { Shield, AlertOctagon, Activity, Wifi } from "lucide-react";

function Overview({ liveAlerts }) {
  const [stats, setStats] = useState(null);
  const [threats, setThreats] = useState([]);

  useEffect(() => {
    // Fetch recent threats
    axios.get("http://localhost:8000/api/threats?limit=5")
      .then((res) => setThreats(res.data))
      .catch(() => {});

    // Fetch real summary stats
    axios.get("http://localhost:8000/api/metrics/summary")
      .then((res) => setStats(res.data))
      .catch(() => {});
  }, [liveAlerts]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "32px" }}>
      {/* Page Header */}
      <div>
        <h2 style={{ fontSize: "28px", fontWeight: "bold", margin: "0 0 8px 0" }}>Security Center</h2>
        <p style={{ color: "#9ca3af", margin: 0 }}>Neuro-inspired active firewall telemetry and brain state overview.</p>
      </div>

      {/* Summary Cards Row */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
        gap: "24px"
      }}>
        <div style={cardStyle("#ef4444")}>
          <div style={iconWrapperStyle("#fef2f2", "#ef4444")}>
            <AlertOctagon size={24} />
          </div>
          <div>
            <div style={{ color: "#9ca3af", fontSize: "14px" }}>TOTAL EVENTS</div>
            <div style={{ fontSize: "28px", fontWeight: "bold", margin: "4px 0" }}>
              {stats ? stats.total_events_processed.toLocaleString() : "—"}
            </div>
          </div>
        </div>

        <div style={cardStyle("#eab308")}>
          <div style={iconWrapperStyle("#fef9c3", "#eab308")}>
            <Shield size={24} />
          </div>
          <div>
            <div style={{ color: "#9ca3af", fontSize: "14px" }}>FALSE POSITIVE RATE</div>
            <div style={{ fontSize: "28px", fontWeight: "bold", margin: "4px 0" }}>
              {stats ? `${(stats.avg_fpr * 100).toFixed(3)}%` : "—"}
            </div>
          </div>
        </div>

        <div style={cardStyle("#3b82f6")}>
          <div style={iconWrapperStyle("#eff6ff", "#3b82f6")}>
            <Activity size={24} />
          </div>
          <div>
            <div style={{ color: "#9ca3af", fontSize: "14px" }}>p50 BRAIN LATENCY</div>
            <div style={{ fontSize: "28px", fontWeight: "bold", margin: "4px 0" }}>
              {stats ? `${stats.p50_latency_ms.toFixed(1)} ms` : "—"}
            </div>
          </div>
        </div>

        <div style={cardStyle("#10b981")}>
          <div style={iconWrapperStyle("#ecfdf5", "#10b981")}>
            <Wifi size={24} />
          </div>
          <div>
            <div style={{ color: "#9ca3af", fontSize: "14px" }}>THROUGHPUT</div>
            <div style={{ fontSize: "28px", fontWeight: "bold", margin: "4px 0" }}>
              {stats ? `${stats.avg_throughput_pps.toFixed(0)} pps` : "—"}
            </div>
          </div>
        </div>
      </div>

      {/* Live Threat Feed Table */}
      <div style={{
        backgroundColor: "#11141e",
        borderRadius: "16px",
        border: "1px solid #1f2937",
        padding: "24px"
      }}>
        <h3 style={{ fontSize: "20px", fontWeight: "bold", margin: "0 0 20px 0" }}>Recent Intrusion Alerts</h3>
        {threats.length === 0 ? (
          <div style={{ color: "#6b7280", textAlign: "center", padding: "40px" }}>
            No threat events recorded yet. The SNN is learning...
          </div>
        ) : (
          <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid #1f2937", color: "#6b7280" }}>
                <th style={{ padding: "12px 16px" }}>TIMESTAMP</th>
                <th style={{ padding: "12px 16px" }}>SOURCE IP</th>
                <th style={{ padding: "12px 16px" }}>ATTACK CLASS</th>
                <th style={{ padding: "12px 16px" }}>CONFIDENCE</th>
                <th style={{ padding: "12px 16px" }}>SNN Z-SCORE</th>
                <th style={{ padding: "12px 16px" }}>ACTION</th>
              </tr>
            </thead>
            <tbody>
              {threats.map((threat, idx) => (
                <tr key={idx} style={{ borderBottom: "1px solid #1f2937", color: "#d1d5db" }}>
                  <td style={{ padding: "16px" }}>{new Date(threat.timestamp || Date.now()).toLocaleTimeString()}</td>
                  <td style={{ padding: "16px", fontWeight: "bold" }}>{threat.src_ip}</td>
                  <td style={{ padding: "16px" }}>
                    <span style={badgeStyle(threat.attack_class)}>{threat.attack_class}</span>
                  </td>
                  <td style={{ padding: "16px" }}>{(threat.confidence * 100).toFixed(1)}%</td>
                  <td style={{ padding: "16px" }}>{threat.z_score.toFixed(2)}</td>
                  <td style={{ padding: "16px", color: "#ef4444", fontWeight: "bold" }}>{threat.action_taken}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

const cardStyle = (glowColor) => ({
  backgroundColor: "#11141e",
  borderRadius: "16px",
  border: "1px solid #1f2937",
  padding: "24px",
  display: "flex",
  alignItems: "center",
  gap: "20px",
  boxShadow: `0 4px 20px rgba(0, 0, 0, 0.4), inset 0 0 10px ${glowColor}15`
});

const iconWrapperStyle = (bg, color) => ({
  padding: "16px",
  borderRadius: "12px",
  backgroundColor: bg,
  color: color,
  display: "flex",
  justifyContent: "center",
  alignItems: "center"
});

const badgeStyle = (cls) => {
  const colors = {
    BENIGN: { bg: "#ecfdf5", text: "#065f46" },
    DoS: { bg: "#fef2f2", text: "#991b1b" },
    DDoS: { bg: "#fff5f5", text: "#c53030" },
    PortScan: { bg: "#fffbeb", text: "#92400e" },
    BruteForce: { bg: "#f3e8ff", text: "#6b21a8" },
    WebAttack: { bg: "#fff7ed", text: "#9a3412" },
    Infiltration: { bg: "#fdf4ff", text: "#86198f" },
    Botnet: { bg: "#f0fdf4", text: "#166534" },
    ZeroDay: { bg: "#fef2f2", text: "#dc2626" },
  };
  const val = colors[cls] || { bg: "#eff6ff", text: "#1e40af" };
  return {
    padding: "6px 12px",
    borderRadius: "20px",
    fontSize: "12px",
    fontWeight: "600",
    backgroundColor: val.bg,
    color: val.text
  };
};

export default Overview;
