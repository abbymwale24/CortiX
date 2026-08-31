import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";
import PageHeader from "../components/PageHeader";

const API_BASE = "http://localhost:8000";

function Threats({ liveAlerts }) {
  const [threats, setThreats] = useState([]);
  const [limit, setLimit] = useState(50);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState(new Date());

  const fetchThreats = useCallback(async () => {
    setIsRefreshing(true);
    try {
      const res = await axios.get(`${API_BASE}/api/threats?limit=${limit}`);
      setThreats(res.data);
      setLastRefreshed(new Date());
    } catch (err) {
      console.error("Failed to fetch threats:", err);
    } finally {
      setIsRefreshing(false);
    }
  }, [limit]);

  useEffect(() => {
    fetchThreats();
    const interval = setInterval(fetchThreats, 5000);
    return () => clearInterval(interval);
  }, [fetchThreats]);

  // Re-fetch when WS live alert arrives
  useEffect(() => {
    if (liveAlerts && liveAlerts.length > 0) {
      fetchThreats();
    }
  }, [liveAlerts, fetchThreats]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "28px" }}>
      <PageHeader
        title="Threat Activity Feed"
        subtitle="Detailed log of all anomalous events flagged by the Spiking Neural Network and LSTM-CNN model."
        onRefresh={fetchThreats}
        isRefreshing={isRefreshing}
        lastRefreshed={lastRefreshed}
      />

      {/* Controls Bar */}
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        backgroundColor: "#11141e",
        padding: "16px 24px",
        borderRadius: "12px",
        border: "1px solid #1f2937"
      }}>
        <div style={{ fontSize: "14px", color: "#9ca3af" }}>
          Showing <strong style={{ color: "#ffffff" }}>{threats.length}</strong> recorded threat logs
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <label style={{ fontSize: "13px", color: "#9ca3af" }}>Rows limit:</label>
          <select
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            style={{
              backgroundColor: "#161b26",
              color: "#ffffff",
              border: "1px solid #374151",
              borderRadius: "6px",
              padding: "6px 12px",
              fontSize: "13px"
            }}
          >
            <option value={25}>25</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
            <option value={200}>200</option>
          </select>
        </div>
      </div>

      <div style={{
        backgroundColor: "#11141e",
        borderRadius: "16px",
        border: "1px solid #1f2937",
        padding: "24px"
      }}>
        {threats.length === 0 ? (
          <div style={{ color: "#6b7280", textAlign: "center", padding: "40px" }}>
            No threat events recorded yet. Start the daemon to begin monitoring.
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid #1f2937", color: "#6b7280" }}>
                  <th style={{ padding: "12px 16px" }}>TIMESTAMP</th>
                  <th style={{ padding: "12px 16px" }}>SOURCE IP</th>
                  <th style={{ padding: "12px 16px" }}>DESTINATION</th>
                  <th style={{ padding: "12px 16px" }}>PROTOCOL</th>
                  <th style={{ padding: "12px 16px" }}>ATTACK CLASS</th>
                  <th style={{ padding: "12px 16px" }}>CONFIDENCE</th>
                  <th style={{ padding: "12px 16px" }}>SNN Z-SCORE</th>
                  <th style={{ padding: "12px 16px" }}>ACTION TAKEN</th>
                </tr>
              </thead>
              <tbody>
                {threats.map((threat, idx) => (
                  <tr key={threat.id || idx} style={{ borderBottom: "1px solid #1f2937", color: "#d1d5db" }}>
                    <td style={{ padding: "16px" }}>{new Date(threat.timestamp || Date.now()).toLocaleString()}</td>
                    <td style={{ padding: "16px", fontWeight: "bold" }}>{threat.src_ip}</td>
                    <td style={{ padding: "16px" }}>{threat.dst_ip}:{threat.dst_port}</td>
                    <td style={{ padding: "16px" }}>{threat.protocol}</td>
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
          </div>
        )}
      </div>
    </div>
  );
}

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

export default Threats;
