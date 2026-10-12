"""
CortiX Module 4 — Containment Action Executor

Executes active blocks and rate limits on the local Linux system
using iptables, ip rule, and tc (traffic control) commands.
"""

import os
import re
import subprocess
import logging
from typing import Optional

logger = logging.getLogger("cortix.containment.executor")

# Linux interface names: max 15 chars (IFNAMSIZ - 1), no whitespace or '/'.
# Restricting the charset also keeps the name safe to interpolate into shell commands.
_IFACE_NAME_RE = re.compile(r"^[A-Za-z0-9_.:\-]{1,15}$")


def _parse_default_routes(output: str) -> Optional[str]:
    """
    Parse `ip route show default` output and return the interface of the
    preferred (lowest-metric) default route, or None if there is none.

    Example line:
        default via 192.168.8.1 dev wlan0 proto dhcp src 192.168.8.100 metric 600
    """
    best_iface, best_metric = None, None
    for line in output.splitlines():
        tokens = line.split()
        if not tokens or tokens[0] != "default" or "dev" not in tokens:
            continue
        dev_idx = tokens.index("dev")
        if dev_idx + 1 >= len(tokens):
            continue
        iface = tokens[dev_idx + 1]
        metric = 0  # kernel treats a missing metric as 0
        if "metric" in tokens:
            m_idx = tokens.index("metric")
            if m_idx + 1 < len(tokens) and tokens[m_idx + 1].isdigit():
                metric = int(tokens[m_idx + 1])
        if best_metric is None or metric < best_metric:
            best_iface, best_metric = iface, metric
    return best_iface


def detect_default_interface() -> Optional[str]:
    """
    Detect the network interface carrying the system's default route.

    Queries `ip route show default` (IPv4 first, then IPv6) and picks the
    lowest-metric route. Returns None if no default route exists or the
    `ip` tool is unavailable (e.g. non-Linux dev environments).
    """
    for family in (["-4"], ["-6"]):
        try:
            res = subprocess.run(
                ["ip", *family, "route", "show", "default"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
        except (subprocess.SubprocessError, FileNotFoundError, OSError) as exc:
            logger.debug("Could not run 'ip route' for interface detection: %s", exc)
            return None
        if res.returncode != 0:
            logger.debug("'ip %s route show default' failed: %s", family[0], res.stderr.strip())
            continue
        iface = _parse_default_routes(res.stdout)
        if iface and _IFACE_NAME_RE.match(iface):
            return iface
    return None


class ContainmentExecutor:
    """
    Applies network level blocks and rate limits on Linux systems.
    Runs commands safely with fallback/no-op on non-supported OS (Windows/Mac).
    """

    def __init__(self, interface: Optional[str] = None):
        """
        Args:
            interface: Explicit interface to apply tc rules on. If omitted,
                the CORTIX_INTERFACE env var is used if set, otherwise the
                default-route interface is detected at runtime.
        """
        if interface:
            source = "constructor argument"
        elif os.getenv("CORTIX_INTERFACE"):
            interface, source = os.getenv("CORTIX_INTERFACE"), "CORTIX_INTERFACE env var"
        else:
            interface, source = detect_default_interface(), "default-route detection"

        if interface is not None and not _IFACE_NAME_RE.match(interface):
            raise ValueError(f"Invalid network interface name: {interface!r}")

        self.interface: Optional[str] = interface
        if self.interface:
            logger.info("ContainmentExecutor using interface %s (from %s)", self.interface, source)
        else:
            logger.warning(
                "No network interface detected (no default route?). "
                "Interface-level actions (tc rate limiting) will be skipped."
            )

    def apply_action(self, action_id: int, src_ip: str) -> bool:
        """
        Apply a containment action mapping to the local OS network subsystem.
        
        Args:
            action_id: 0..5 (ALLOW, RATE_LIMIT, TEMP_BLOCK, QUARANTINE, HARD_BLOCK, HONEYPOT_REDIRECT)
            src_ip: Target source IP address
            
        Returns:
            bool indicating execution success
        """
        action_names = {
            0: "ALLOW",
            1: "RATE_LIMIT",
            2: "TEMP_BLOCK",
            3: "QUARANTINE",
            4: "HARD_BLOCK",
            5: "HONEYPOT_REDIRECT",
        }
        
        action = action_names.get(action_id, "ALLOW")
        logger.info("Executing containment action: %s on IP: %s", action, src_ip)

        try:
            if action_id == 0:
                # ALLOW - remove any temporary blocks/limits
                self.remove_blocks(src_ip)
                return True
                
            elif action_id == 1:
                # RATE_LIMIT (tc qdisc throttling to 10% / 100kbit)
                if not self.interface:
                    logger.warning("RATE_LIMIT skipped: no network interface available")
                    return False
                return self._run_linux_cmd(
                    f"sudo tc qdisc add dev {self.interface} root tbf rate 100kbit burst 1mbit latency 70ms"
                )
                
            elif action_id == 2:
                # TEMP_BLOCK (iptables temporary drop)
                return self._run_linux_cmd(
                    f"sudo iptables -I INPUT 1 -s {src_ip} -j DROP"
                )
                
            elif action_id == 3:
                # QUARANTINE (redirect to vlan table)
                return self._run_linux_cmd(
                    f"sudo ip rule add from {src_ip} table honeypot_table"
                )
                
            elif action_id == 4:
                # HARD_BLOCK (permanent drop)
                return self._run_linux_cmd(
                    f"sudo iptables -A INPUT -s {src_ip} -j DROP"
                )
                
            elif action_id == 5:
                # HONEYPOT_REDIRECT (Port forward/Route to honeypot interface)
                # Redirect port traffic from IP to local docker honeypot container
                return self._run_linux_cmd(
                    f"sudo iptables -t nat -A PREROUTING -s {src_ip} -p tcp -j DNAT --to-destination 172.17.0.2"
                )
                
        except Exception as exc:
            logger.error("Failed to execute network block: %s", exc)
            
        return False

    def remove_blocks(self, src_ip: str) -> bool:
        """Undo blocks or rate-limits for the specified IP."""
        logger.info("Restoring normal traffic permissions for IP: %s", src_ip)
        success = True
        
        # Best effort cleanup of any iptables/tc rules for this IP
        self._run_linux_cmd(f"sudo iptables -D INPUT -s {src_ip} -j DROP")
        self._run_linux_cmd(f"sudo iptables -t nat -D PREROUTING -s {src_ip} -p tcp -j DNAT --to-destination 172.17.0.2")
        if self.interface:
            self._run_linux_cmd(f"sudo tc qdisc del dev {self.interface} root")
        self._run_linux_cmd(f"sudo ip rule del from {src_ip} table honeypot_table")

        return success

    def _run_linux_cmd(self, cmd: str) -> bool:
        """Run terminal command. Log standard exceptions."""
        try:
            # We use shell=True for complex piping but run under safe system controls
            res = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            if res.returncode != 0:
                logger.debug("Linux firewall command failed/not supported: %s | %s", cmd, res.stderr.strip())
                return False
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            # Safe ignore if executing on non-Linux OS dev environments
            logger.debug("Platform does not support command execution: %s", cmd)
            return False
