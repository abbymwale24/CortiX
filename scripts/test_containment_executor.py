#!/usr/bin/env python3
"""
CortiX — ContainmentExecutor Verification Script

Runs each containment executor action against 198.51.100.1,
prints live rule verification output, performs cleanup, and re-verifies clean baseline state.
"""

import subprocess

TARGET_IP = "198.51.100.1"

def run(cmd: str):
    print(f"\n>>> $ {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout.rstrip())
    if res.stderr:
        print(f"[STDERR] {res.stderr.rstrip()}")
    return res

def main():
    print("=" * 80)
    print(f"   CORTIX CONTAINMENT EXECUTOR VERIFICATION — TARGET IP: {TARGET_IP}")
    print("=" * 80)

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
    run("sudo tc qdisc add dev eth0 root tbf rate 100kbit burst 1mbit latency 70ms")
    print("\n[VERIFICATION STEP 5]: sudo tc qdisc show dev eth0")
    run("sudo tc qdisc show dev eth0")

    print("\n[CLEANUP STEP 5]: sudo tc qdisc del dev eth0 root")
    run("sudo tc qdisc del dev eth0 root")
    print("[POST-CLEANUP VERIFICATION STEP 5]:")
    run("sudo tc qdisc show dev eth0")

    print("\n" + "=" * 80)
    print("   ALL VERIFICATIONS AND CLEANUPS COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
