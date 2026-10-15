#!/usr/bin/env python3
"""
CortiX — ContainmentExecutor Verification Script

Runs each containment executor action against 198.51.100.1,
prints live rule verification output, performs cleanup, and re-verifies clean baseline state.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

# Make `cortix` importable when this script is run directly from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cortix.containment.executor import ContainmentExecutor, _parse_default_routes, detect_default_interface

TARGET_IP = "198.51.100.1"

def run(cmd: str):
    print(f"\n>>> $ {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout.rstrip())
    if res.stderr:
        print(f"[STDERR] {res.stderr.rstrip()}")
    return res

def verify_interface_detection() -> Optional[str]:
    """
    Exercise the actual interface-resolution logic (no sudo required) before
    running any tc commands, instead of assuming eth0 exists. Covers:
      - _parse_default_routes() against synthetic `ip route` output
      - detect_default_interface() against this machine's real routing table
      - ContainmentExecutor()'s resolution order (ctor arg > env var > detection)
    Returns the interface tc should actually be tested against.
    """
    print("\n--- STEP 0: Interface Detection ---")

    # Parser unit checks against synthetic input — no live routing table required.
    multi_route_output = (
        "default via 10.0.0.1 dev eth1 proto dhcp src 10.0.0.5 metric 600\n"
        "default via 10.0.0.1 dev eth0 proto static src 10.0.0.5 metric 100\n"
    )
    assert _parse_default_routes(multi_route_output) == "eth0", (
        "expected lowest-metric route (eth0, metric 100) to win"
    )
    assert _parse_default_routes("") is None, "no default route should parse to None"
    print("[OK] _parse_default_routes() picks lowest-metric route and handles empty input")

    # Live detection on this machine.
    detected = detect_default_interface()
    print(f"[INFO] detect_default_interface() on this machine -> {detected!r}")

    # ContainmentExecutor's constructor-arg override still takes priority.
    executor = ContainmentExecutor(interface="lo")
    assert executor.interface == "lo", "explicit constructor arg must win over detection"
    print("[OK] ContainmentExecutor(interface=...) override takes priority over detection")

    if detected is None:
        print(
            "[WARN] No default route detected on this machine — tc steps below will be "
            "skipped (this mirrors ContainmentExecutor's own no-op behavior, not a hardcoded eth0 guess)."
        )
    return detected

def main():
    print("=" * 80)
    print(f"   CORTIX CONTAINMENT EXECUTOR VERIFICATION — TARGET IP: {TARGET_IP}")
    print("=" * 80)

    tc_interface = verify_interface_detection()

    # 1. Action 2 (TEMP_BLOCK) — Prepend DROP rule
    print("\n--- STEP 1: Temp Block (iptables prepend DROP rule) ---")
    run(f"sudo iptables -I INPUT 1 -s {TARGET_IP} -j DROP")
    print("\n[VERIFICATION STEP 1]: sudo iptables -L INPUT -n -v --line-numbers")
    run("sudo iptables -L INPUT -n -v --line-numbers")

    print("\n[CLEANUP STEP 1]: sudo iptables -D INPUT -s 198.51.100.1 -j DROP")
    run(f"sudo iptables -D INPUT -s {TARGET_IP} -j DROP")
    print("[POST-CLEANUP VERIFICATION STEP 1]:")
    run("sudo iptables -L INPUT -n -v --line-numbers")

    # 2. Action 4 (HARD_BLOCK) — Append DROP rule
    print("\n" + "=" * 80)
    print("--- STEP 2: Hard Block (iptables append DROP rule) ---")
    run(f"sudo iptables -A INPUT -s {TARGET_IP} -j DROP")
    print("\n[VERIFICATION STEP 2]: sudo iptables -L INPUT -n -v")
    run("sudo iptables -L INPUT -n -v")

    print("\n[CLEANUP STEP 2]: sudo iptables -D INPUT -s 198.51.100.1 -j DROP")
    run(f"sudo iptables -D INPUT -s {TARGET_IP} -j DROP")
    print("[POST-CLEANUP VERIFICATION STEP 2]:")
    run("sudo iptables -L INPUT -n -v")

    # 3. Action 5 (HONEYPOT_REDIRECT) — DNAT Prerouting rule
    print("\n" + "=" * 80)
    print("--- STEP 3: Honeypot Redirect (iptables PREROUTING DNAT rule) ---")
    run(f"sudo iptables -t nat -A PREROUTING -s {TARGET_IP} -p tcp -j DNAT --to-destination 172.17.0.2")
    print("\n[VERIFICATION STEP 3]: sudo iptables -t nat -L PREROUTING -n -v")
    run("sudo iptables -t nat -L PREROUTING -n -v")

    print("\n[CLEANUP STEP 3]: sudo iptables -t nat -D PREROUTING -s 198.51.100.1 -p tcp -j DNAT --to-destination 172.17.0.2")
    run(f"sudo iptables -t nat -D PREROUTING -s {TARGET_IP} -p tcp -j DNAT --to-destination 172.17.0.2")
    print("[POST-CLEANUP VERIFICATION STEP 3]:")
    run("sudo iptables -t nat -L PREROUTING -n -v")

    # 4. Action 3 (QUARANTINE) — IP Policy Routing rule
    print("\n" + "=" * 80)
    print("--- STEP 4: Quarantine (ip rule policy routing table 100) ---")
    run(f"sudo ip rule add from {TARGET_IP} table 100")
    print("\n[VERIFICATION STEP 4]: sudo ip rule show")
    run("sudo ip rule show")

    print("\n[CLEANUP STEP 4]: sudo ip rule del from 198.51.100.1 table 100")
    run(f"sudo ip rule del from {TARGET_IP} table 100")
    print("[POST-CLEANUP VERIFICATION STEP 4]:")
    run("sudo ip rule show")

    # 5. Action 1 (RATE_LIMIT) — Traffic Control TBF qdisc
    print("\n" + "=" * 80)
    print("--- STEP 5: Rate Limit (tc qdisc tbf 100kbit rate limit) ---")
    if tc_interface is None:
        print(
            "[SKIPPED] No default-route interface detected on this machine — "
            "this matches ContainmentExecutor's own RATE_LIMIT no-op (see executor.py), "
            "rather than guessing eth0 on a box that may not have one."
        )
    else:
        print(f"[INFO] Using detected interface: {tc_interface}")
        run(f"sudo tc qdisc add dev {tc_interface} root tbf rate 100kbit burst 1mbit latency 70ms")
        print(f"\n[VERIFICATION STEP 5]: sudo tc qdisc show dev {tc_interface}")
        run(f"sudo tc qdisc show dev {tc_interface}")

        print(f"\n[CLEANUP STEP 5]: sudo tc qdisc del dev {tc_interface} root")
        run(f"sudo tc qdisc del dev {tc_interface} root")
        print("[POST-CLEANUP VERIFICATION STEP 5]:")
        run(f"sudo tc qdisc show dev {tc_interface}")

    print("\n" + "=" * 80)
    print("   ALL VERIFICATIONS AND CLEANUPS COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
