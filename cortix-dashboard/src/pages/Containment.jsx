import React, { useState, useEffect } from "react";
import axios from "axios";
import { Shield, Lock, ShieldAlert, Cpu } from "lucide-react";

function Containment() {
  const [actions, setActions] = useState([]);

  useEffect(() => {
    axios.get("http://localhost:8000/api/containment")
      .then((res) => setActions(res.data))
      .catch(() => {});
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "32px" }}>
      <div>
        <h2 style={{ fontSize: "28px", fontWeight: "bold", margin: "0 0 8px 0" }}>Active Containment Center</h2>
        <p style={{ color: "#9ca3af", margin: 0 }}>Deep Reinforcement Learning (DQN) executed firewall rules and quarantine actions.</p>
      </div>

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
                <tr key={idx} style={{ borderBottom: "1px solid #1f2937", color: "#d1d5db" }}>
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
