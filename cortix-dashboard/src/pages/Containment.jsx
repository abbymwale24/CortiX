import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";
import { Cpu } from "lucide-react";
import PageHeader from "../components/PageHeader";

const API_BASE = "http://localhost:8000";

function Containment({ liveAlerts }) {
  const [actions, setActions] = useState([]);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState(new Date());

  const fetchActions = useCallback(async () => {
    setIsRefreshing(true);
    try {
      const res = await axios.get(`${API_BASE}/api/containment`);
      setActions(res.data);
      setLastRefreshed(new Date());
    } catch (err) {
      console.error("Failed to fetch containment actions:", err);
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchActions();
    const interval = setInterval(fetchActions, 5000);
    return () => clearInterval(interval);
  }, [fetchActions]);

  // Re-fetch when WS live alert arrives
  useEffect(() => {
    if (liveAlerts && liveAlerts.length > 0) {
      fetchActions();
    }
  }, [liveAlerts, fetchActions]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "28px" }}>
      <PageHeader
        title="Active Containment Center"
        subtitle="Deep Reinforcement Learning (DQN) executed firewall rules and quarantine actions."
        onRefresh={fetchActions}
        isRefreshing={isRefreshing}
        lastRefreshed={lastRefreshed}
      />

      <div style={{
        backgroundColor: "#11141e",
        borderRadius: "16px",
        border: "1px solid #1f2937",
        padding: "24px"
      }}>
        {actions.length === 0 ? (
          <div style={{ color: "#6b7280", textAlign: "center", padding: "40px" }}>
            No active containment rules. The network is currently operating nominally.
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid #1f2937", color: "#6b7280" }}>
                  <th style={{ padding: "12px 16px" }}>TIMESTAMP</th>
                  <th style={{ padding: "12px 16px" }}>TARGET IP</th>
                  <th style={{ padding: "12px 16px" }}>DQN ACTION CODE</th>
                  <th style={{ padding: "12px 16px" }}>STATUS</th>
                  <th style={{ padding: "12px 16px" }}>EXPIRATION</th>
                </tr>
              </thead>
              <tbody>
                {actions.map((action, idx) => (
                  <tr key={action.id || idx} style={{ borderBottom: "1px solid #1f2937", color: "#d1d5db" }}>
                    <td style={{ padding: "16px" }}>{new Date(action.timestamp || Date.now()).toLocaleString()}</td>
                    <td style={{ padding: "16px", fontWeight: "bold" }}>{action.target_ip}</td>
                    <td style={{ padding: "16px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <Cpu size={16} color="#3b82f6" />
                        {action.action_name}
                      </div>
                    </td>
                    <td style={{ padding: "16px" }}>
                      <span style={statusBadgeStyle(action.status)}>{action.status}</span>
                    </td>
                    <td style={{ padding: "16px", color: "#9ca3af" }}>
                      {action.expires_at ? new Date(action.expires_at).toLocaleTimeString() : "Never (Hard Block)"}
                    </td>
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

const statusBadgeStyle = (status) => {
  const isActive = status === "ACTIVE";
  return {
    padding: "4px 8px",
    borderRadius: "4px",
    fontSize: "11px",
    fontWeight: "bold",
    backgroundColor: isActive ? "#fef2f2" : "#f3f4f6",
    color: isActive ? "#991b1b" : "#6b7280"
  };
};

export default Containment;
