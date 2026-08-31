import React from "react";
import { RefreshCw, Radio } from "lucide-react";

function PageHeader({ title, subtitle, onRefresh, isRefreshing, lastRefreshed }) {
  return (
    <div style={{
      display: "flex",
      justifyContent: "space-between",
      alignItems: "flex-start",
      flexWrap: "wrap",
      gap: "16px",
      marginBottom: "8px"
    }}>
      <div>
        <h2 style={{ fontSize: "28px", fontWeight: "bold", margin: "0 0 8px 0", color: "#ffffff" }}>
          {title}
        </h2>
        {subtitle && (
          <p style={{ color: "#9ca3af", margin: 0, fontSize: "14px" }}>
            {subtitle}
          </p>
        )}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
        {/* Live Auto-Refresh Badge */}
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          backgroundColor: "#161b26",
          padding: "6px 12px",
          borderRadius: "20px",
          border: "1px solid #1f2937",
          fontSize: "12px",
          color: "#10b981",
          fontWeight: "600"
        }}>
          <Radio size={14} className="pulse-icon" />
          <span>LIVE AUTO (5s)</span>
        </div>

        {/* Last Refreshed Time */}
        {lastRefreshed && (
          <span style={{ fontSize: "12px", color: "#6b7280" }}>
            Updated: {lastRefreshed.toLocaleTimeString()}
          </span>
        )}

        {/* Manual Refresh Button */}
        {onRefresh && (
          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              backgroundColor: isRefreshing ? "#1f2937" : "#1e293b",
              color: "#ffffff",
              border: "1px solid #374151",
              borderRadius: "8px",
              padding: "8px 16px",
              fontSize: "13px",
              fontWeight: "600",
              cursor: isRefreshing ? "not-allowed" : "pointer",
              transition: "all 0.2s ease-in-out",
              boxShadow: "0 2px 8px rgba(0,0,0,0.3)"
            }}
            onMouseEnter={(e) => {
              if (!isRefreshing) e.currentTarget.style.backgroundColor = "#334155";
            }}
            onMouseLeave={(e) => {
              if (!isRefreshing) e.currentTarget.style.backgroundColor = "#1e293b";
            }}
          >
            <RefreshCw
              size={15}
              style={{
                animation: isRefreshing ? "spin 1s linear infinite" : "none"
              }}
            />
            <span>{isRefreshing ? "Refreshing..." : "Refresh"}</span>
          </button>
        )}
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        @keyframes pulse-dot {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
        .pulse-icon {
          animation: pulse-dot 1.5s infinite;
        }
      `}</style>
    </div>
  );
}

export default PageHeader;
