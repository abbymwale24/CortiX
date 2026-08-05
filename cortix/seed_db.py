"""
CortiX — Database Seeder Script

Populates the database with realistic initial threat alerts, attacker profiles,
system metrics, and containment actions for live dashboard demonstration.
"""

import random
from datetime import datetime, timedelta, timezone
from cortix.database import (
    init_db,
    get_session,
    Threat,
    AttackerProfile,
    SystemMetric,
    ContainmentAction,
)

ATTACK_CLASSES = ["DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Infiltration", "Botnet"]
SAMPLE_IPS = [
    {"ip": "185.220.101.5", "country": "CN", "city": "Shanghai", "isp": "China Telecom", "asn": "AS4134", "hostname": "shanghai-node.net", "abuse": 88, "vt": 14, "level": "CRITICAL"},
    {"ip": "91.240.118.15", "country": "RU", "city": "Moscow", "isp": "Rostelecom", "asn": "AS12389", "hostname": "moscow-gate.ru", "abuse": 74, "vt": 8, "level": "HIGH"},
    {"ip": "45.154.255.89", "country": "NL", "city": "Amsterdam", "isp": "HostPalace", "asn": "AS206804", "hostname": "ams-exit.nl", "abuse": 62, "vt": 5, "level": "HIGH"},
    {"ip": "103.253.41.12", "country": "IN", "city": "Mumbai", "isp": "Reliance Jio", "asn": "AS55836", "hostname": "mumbai-jio.in", "abuse": 45, "vt": 2, "level": "MEDIUM"},
]

def seed():
    print("Initializing database tables...")
    init_db()
    db = get_session()

    try:
        # Check if already seeded
        if db.query(Threat).count() > 0:
            print("Database already contains data. Skipping seed.")
            return

        print("Seeding Attacker Profiles...")
        profiles = []
        for p in SAMPLE_IPS:
            prof = AttackerProfile(
                ip=p["ip"],
                country=p["country"],
                city=p["city"],
                lat=random.uniform(-90, 90),
                lon=random.uniform(-180, 180),
                isp=p["isp"],
                asn=p["asn"],
                hostname=p["hostname"],
                abuse_score=p["abuse"],
                known_malicious=True,
                vt_malicious=p["vt"],
                threat_level=p["level"],
                first_seen=datetime.now(timezone.utc) - timedelta(days=7),
                last_seen=datetime.now(timezone.utc),
            )
            db.add(prof)
            profiles.append(prof)
        db.commit()

        print("Seeding Threats...")
        now = datetime.now(timezone.utc)
        for i in range(25):
            prof = random.choice(profiles)
            t_time = now - timedelta(minutes=random.randint(1, 1440))
            attack_cls = random.choice(ATTACK_CLASSES)
            action = "HARD_BLOCK" if attack_cls in ["DoS", "DDoS"] else ("TEMP_BLOCK" if attack_cls == "PortScan" else "RATE_LIMIT")
            
            threat = Threat(
                timestamp=t_time,
                src_ip=prof.ip,
                dst_ip="10.0.0.2",
                src_port=random.randint(1024, 65535),
                dst_port=random.choice([80, 443, 22, 8080, 3306]),
                protocol="TCP",
                attack_class=attack_cls,
                confidence=round(random.uniform(0.85, 0.99), 3),
                z_score=round(random.uniform(4.5, 14.2), 2),
                action_taken=action,
                resolved=random.choice([True, False]),
                attacker_profile_id=prof.id,
            )
            db.add(threat)

        print("Seeding Containment Actions...")
        for prof in profiles:
            act = ContainmentAction(
                timestamp=now - timedelta(minutes=random.randint(5, 120)),
                src_ip=prof.ip,
                action=random.choice(["HARD_BLOCK", "TEMP_BLOCK", "RATE_LIMIT", "HONEYPOT_REDIRECT"]),
                triggered_by=random.choice(["RL_AGENT", "ADMIN"]),
                duration_seconds=600 if prof.threat_level == "HIGH" else 3600,
                resolved=False,
            )
            db.add(act)

        print("Seeding System Metrics...")
        for mins_ago in range(60, 0, -5):
            m_time = now - timedelta(minutes=mins_ago)
            events = random.randint(400, 1200)
            tp = int(events * random.uniform(0.1, 0.2))
            fp = random.randint(0, 2)
            metric = SystemMetric(
                timestamp=m_time,
                event_count=events,
                tp_count=tp,
                fp_count=fp,
                fn_count=0,
                latency_p50_ms=round(random.uniform(3.8, 4.6), 2),
                latency_p99_ms=round(random.uniform(11.5, 13.8), 2),
                throughput_pps=round(random.uniform(1500.0, 3200.0), 1),
            )
            db.add(metric)

        db.commit()
        print("Database seeding completed successfully!")

    except Exception as exc:
        db.rollback()
        print(f"Error seeding database: {exc}")
    finally:
        db.close()

if __name__ == "__main__":
    seed()
