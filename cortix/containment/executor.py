"""
CortiX Module 4 — Containment Action Executor

Executes active blocks and rate limits on the local Linux system
using iptables, ip rule, and tc (traffic control) commands.
"""

import subprocess
import logging
from cortix.config import config

logger = logging.getLogger("cortix.containment.executor")


class ContainmentExecutor:
    """
    Applies network level blocks and rate limits on Linux systems.
    Runs commands safely with fallback/no-op on non-supported OS (Windows/Mac).
    """

    def __init__(self, interface: str = None):
        self.interface = interface or config.CAPTURE_INTERFACE

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
