import React, { useState, useEffect, useRef, useCallback } from "react";
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from "react-router-dom";
import { Shield, AlertTriangle, Users, BarChart3, ShieldAlert, Activity, BrainCircuit } from "lucide-react";
import axios from "axios";
import Overview from "./pages/Overview";
import Threats from "./pages/Threats";
import Attackers from "./pages/Attackers";
import Metrics from "./pages/Metrics";
import Containment from "./pages/Containment";
import BrainView from "./pages/BrainView";

const API_BASE = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws/live";
const MAX_RECONNECT_DELAY = 30000;

function App() {
  const [liveAlerts, setLiveAlerts] = useState([]);
  const [wsStatus, setWsStatus] = useState("disconnected");
  const [systemStats, setSystemStats] = useState({
    avg_fpr: 0.0,
    p50_latency_ms: 0.0,
    total_events_processed: 0,
  });

  const wsRef = useRef(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimerRef = useRef(null);

  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsStatus("connected");
        reconnectAttemptRef.current = 0;
        console.log("[CortiX WS] Connected to live event stream");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.event === "THREAT_ALERT") {
            setLiveAlerts((prev) => [data, ...prev].slice(0, 50));
          }
        } catch (err) {
          console.debug("[CortiX WS] Parse error:", err);
        }
      };

      ws.onclose = () => {
        setWsStatus("reconnecting");
        const attempt = reconnectAttemptRef.current;
        const delay = Math.min(1000 * Math.pow(2, attempt), MAX_RECONNECT_DELAY);

        reconnectTimerRef.current = setTimeout(() => {
          reconnectAttemptRef.current += 1;
          connectWebSocket();
        }, delay);
      };

      ws.onerror = () => {
        ws.close();
      };
    } catch (err) {
      setWsStatus("disconnected");
      console.debug("[CortiX WS] Connection failed:", err);
    }
  }, []);

  const fetchSystemStats = useCallback(() => {
    axios.get(`${API_BASE}/api/metrics/summary`)
      .then((res) => setSystemStats(res.data))
      .catch(() => {});
  }, []);

  useEffect(() => {
    connectWebSocket();
    fetchSystemStats();

    const metricsInterval = setInterval(fetchSystemStats, 15000);

    return () => {
      clearInterval(metricsInterval);
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connectWebSocket, fetchSystemStats]);

  const statusConfig = {
    connected: { color: "#10b981", glow: "#10b981", label: "SYSTEM ACTIVE" },
    reconnecting: { color: "#f59e0b", glow: "#f59e0b", label: "RECONNECTING..." },
    disconnected: { color: "#ef4444", glow: "#ef4444", label: "OFFLINE" },
  };
  const status = statusConfig[wsStatus];

  return (
    <Router>
      <div style={{
        display: "flex",
        minHeight: "100vh",
        backgroundColor: "#0d0f14",
        color: "#f3f4f6",
        fontFamily: "'Outfit', sans-serif"
      }}>
        {/* Navigation Sidebar */}
        <aside style={{
          width: "280px",
          backgroundColor: "#11141e",
          borderRight: "1px solid #1f2937",
          padding: "24px",
          display: "flex",
          flexDirection: "column"
        }}>
          {/* Logo Branding */}
          <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "40px" }}>
            <div style={{
              background: "linear-gradient(135deg, #ef4444, #b91c1c)",
              padding: "10px",
              borderRadius: "12px",
              boxShadow: "0 0 15px rgba(239, 68, 68, 0.4)"
            }}>
              <ShieldAlert size={28} color="#ffffff" />
            </div>
            <div>
              <h1 style={{ fontSize: "22px", fontWeight: "bold", margin: 0, letterSpacing: "0.05em", color: "#ffffff" }}>CORTIX</h1>
              <span style={{ fontSize: "11px", color: "#6b7280", letterSpacing: "1px" }}>NEURO DEFENDER</span>
            </div>
          </div>

          {/* Nav List */}
          <NavLinks />

          {/* Footer stats badge with live connection status */}
          <div style={{
            marginTop: "auto",
            backgroundColor: "#161b26",
            padding: "16px",
            borderRadius: "12px",
            border: "1px solid #374151",
            display: "flex",
            alignItems: "center",
            gap: "12px"
          }}>
            <div style={{
              width: "12px",
              height: "12px",
              borderRadius: "50%",
              backgroundColor: status.color,
              boxShadow: `0 0 8px ${status.glow}`,
              animation: wsStatus === "reconnecting" ? "pulse 1.5s infinite" : "none",
            }} />
            <div>
              <div style={{ fontSize: "12px", fontWeight: "bold" }}>{status.label}</div>
              <div style={{ fontSize: "10px", color: "#6b7280" }}>
                FPR: {(systemStats.avg_fpr * 100 || 0).toFixed(2)}% | p50: {(systemStats.p50_latency_ms || 0).toFixed(1)}ms
              </div>
            </div>
          </div>
        </aside>

        {/* Main Content Area */}
        <main style={{ flex: 1, padding: "40px", overflowY: "auto" }}>
          <Routes>
            <Route path="/" element={<Overview liveAlerts={liveAlerts} />} />
            <Route path="/threats" element={<Threats liveAlerts={liveAlerts} />} />
            <Route path="/attackers" element={<Attackers liveAlerts={liveAlerts} />} />
            <Route path="/metrics" element={<Metrics />} />
            <Route path="/containment" element={<Containment liveAlerts={liveAlerts} />} />
            <Route path="/brain" element={<BrainView />} />
          </Routes>
        </main>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>
    </Router>
  );
}

function NavLinks() {
  const location = useLocation();

  const links = [
    { to: "/", label: "Overview", icon: <Shield size={20} /> },
    { to: "/threats", label: "Threat Feed", icon: <AlertTriangle size={20} /> },
    { to: "/attackers", label: "Attacker Maps", icon: <Users size={20} /> },
    { to: "/metrics", label: "Brain Metrics", icon: <BarChart3 size={20} /> },
    { to: "/containment", label: "Firewall Rules", icon: <Activity size={20} /> },
    { to: "/brain", label: "Brain State", icon: <BrainCircuit size={20} /> },
  ];

  return (
    <nav style={{ display: "flex", flexDirection: "column", gap: "8px", flex: 1 }}>
      {links.map((link) => {
        const isActive = location.pathname === link.to ||
          (link.to !== "/" && location.pathname.startsWith(link.to));

        return (
          <Link
            key={link.to}
            to={link.to}
            style={{
              ...navItemStyle,
              backgroundColor: isActive ? "#1e293b" : "transparent",
              color: isActive ? "#ffffff" : "#9ca3af",
              borderLeft: isActive ? "3px solid #ef4444" : "3px solid transparent",
            }}
          >
            {link.icon} {link.label}
          </Link>
        );
      })}
    </nav>
  );
}

const navItemStyle = {
  display: "flex",
  alignItems: "center",
  gap: "12px",
  padding: "12px 16px",
  borderRadius: "8px",
  textDecoration: "none",
  fontWeight: "500",
  transition: "all 0.2s ease-in-out",
};

export default App;
