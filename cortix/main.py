"""
CortiX — Main Firewall Orchestrator Daemon

Starts the packet capture, extracts flow features, converts features to spikes,
queries Hebbian/STDP SNN Ensemble, runs Parallel LSTM-CNN Classifier calibration,
calls DQN Containment Agent for reward optimization, and broadcasts profiles via Redis.
"""

import sys
import os
import time
import argparse
import asyncio
import logging
import threading
import numpy as np

from cortix.config import config
from cortix.redis_bus import get_bus, CHANNEL_LIVE_EVENTS, CHANNEL_THREAT_DETECTED
from cortix.database import init_db, get_session, Threat, AttackerProfile, ContainmentAction
from cortix.preprocessor.capture import PacketCapture
from cortix.preprocessor.features import FlowAggregator, extract_packet_features
from cortix.preprocessor.encoder import SpikeEncoder
from cortix.snn.ensemble import HebbianEnsemble
from cortix.classifier.inference import ClassifierInference
from cortix.containment.agent import ContainmentAgent
from cortix.containment.executor import ContainmentExecutor
from cortix.attribution.osint_engine import AttackerAttributionEngine
from cortix.attribution.alerter import ThreatAlerter
from cortix.honeypot.watcher import HoneypotWatcher
from cortix.honeypot.detector import RansomwareDetector
from cortix.honeypot.trap import HoneypotTrapManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cortix.main")


class CortixDaemon:
    """
    Main background daemon routing all network flows across the SNN brain, 
    calibrating with supervised classifier, and dispatching DQN containment actions.
    """

    def __init__(self, interface: str = None, mode: str = "live"):
        self.mode = mode
        self.interface = interface or config.CAPTURE_INTERFACE

        logger.info("Initializing CortiX Daemon Pipeline. Interface: %s. Mode: %s", self.interface, self.mode)
        
        # 1. Core database and message bus
        init_db()
        self.bus = get_bus()
        self.bus.start_listening()

        # 2. Pipeline components
        self.spike_encoder = SpikeEncoder()
        self.snn_ensemble = HebbianEnsemble()
        self.classifier = ClassifierInference()
        self.rl_agent = ContainmentAgent()
        self.executor = ContainmentExecutor(interface=self.interface)
        self.attribution_engine = AttackerAttributionEngine()
        self.alerter = ThreatAlerter()

        # 3. Honeypot and watchdogs
        self.honeypot_watcher = HoneypotWatcher()
        self.ransomware_detector = RansomwareDetector()
        self.honeypot_manager = HoneypotTrapManager()

        # Registers honeypot events
        self.honeypot_watcher.register_callback(self._handle_honeypot_activity)

        # 4. Scapy packet & flow aggregators
        self.flow_aggregator = FlowAggregator(on_flow=self._process_flow)
        self.packet_capture = PacketCapture(interface=self.interface)
        
        # Capture callback
        self.packet_capture.register_callback(self._handle_packet)

        # Event loop for async OSINT attribution pipeline
        self.loop = asyncio.new_event_loop()
        self.async_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.async_thread.start()

    def _run_async_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start(self):
        """Start the entire firewall monitoring system."""
        self.flow_aggregator.start()
        self.honeypot_watcher.start()
        
        if self.mode == "live":
            self.packet_capture.start_capture()
        logger.info("CortiX Daemon running.")

    def stop(self):
        """Gracefully stop all listeners."""
        logger.info("Shutting down daemon...")
        if self.mode == "live":
            self.packet_capture.stop_capture()
        self.flow_aggregator.stop()
        self.honeypot_watcher.stop()
        self.honeypot_manager.stop_trap()
        self.bus.disconnect()
        self.loop.call_soon_threadsafe(self.loop.stop)

    def _handle_packet(self, packet):
        """Packet capture handler."""
        features = extract_packet_features(packet)
        if features:
            self.flow_aggregator.add_packet(features)

    def _process_flow(self, flow: dict):
        """Processes completed flow record through SNN, LSTM-CNN, and DQN containment."""
        src_ip = flow.get("src_ip", "0.0.0.0")
        
        # 1. Module 1: Preprocessor & Spike Encoder
        feature_vector_16 = self.flow_aggregator.to_feature_vector(flow)
        spikes = self.spike_encoder.encode(feature_vector_16)

        # 2. Module 2: Deep Hebbian/STDP Anomaly Engine
        snn_result = self.snn_ensemble.process_event(spikes)
        z_score = snn_result["z_score"]

        # Prepare 40 dimensions list for LSTM-CNN feature parsing
        # (For prototype inference, pad feature vector 16 to 40 features)
        feature_vector_40 = np.zeros(40, dtype=np.float32)
        feature_vector_40[:16] = feature_vector_16

        # 3. Module 3: LSTM-CNN Parallel supervised calibration
        classifier_result = self.classifier.predict_flow(src_ip, feature_vector_40)
        predicted_class = classifier_result["class"]
        confidence = classifier_result["confidence"]

        # Check anomaly or threat triggers
        is_anomaly = snn_result["is_anomaly"] or classifier_result["is_threat"]

        if is_anomaly:
            logger.warning(
                "Anomaly/Threat detected! Host: %s | SNN z-score: %.2f | Classifier: %s (%.1f%%)",
                src_ip,
                z_score,
                predicted_class,
                confidence * 100,
            )

            # 4. Module 4: Deep RL Containment Agent
            # Request optimal action code based on current telemetry observation states
            action_id = self.rl_agent.select_action(
                z_score=z_score,
                confidence=confidence,
                predicted_class=predicted_class,
                rolling_fpr=0.01,
                reputation=0.5,
                volume_percentile=flow.get("flow_packet_count", 1) / 1000.0,
                time_since_last_alert=0.1,
            )

            # Execute iptables/tc rules
            action_success = self.executor.apply_action(action_id, src_ip)

            # Save basic Threat info to database
            action_names = {
                0: "ALLOW",
                1: "RATE_LIMIT",
                2: "TEMP_BLOCK",
                3: "QUARANTINE",
                4: "HARD_BLOCK",
                5: "HONEYPOT_REDIRECT",
            }
            action_name = action_names.get(action_id, "ALLOW")

            db = get_session()
            try:
                threat = Threat(
                    src_ip=src_ip,
                    dst_ip=flow.get("dst_ip", "0.0.0.0"),
                    src_port=flow.get("src_port", 0),
                    dst_port=flow.get("dst_port", 0),
                    protocol=str(flow.get("protocol", "TCP")),
                    attack_class=predicted_class,
                    confidence=confidence,
                    z_score=z_score,
                    action_taken=action_name,
                    resolved=False,
                )
                db.add(threat)
                db.commit()

                # Publish detection alert to dashboard via Redis
                alert_payload = {
                    "event": "THREAT_ALERT",
                    "src_ip": src_ip,
                    "attack_class": predicted_class,
                    "confidence": confidence,
                    "z_score": z_score,
                    "action_taken": action_name,
                    "timestamp": time.time(),
                }
                self.bus.publish(CHANNEL_LIVE_EVENTS, alert_payload)

                # 5. Trigger Async Module 6: Attacker Attribution
                asyncio.run_coroutine_threadsafe(
                    self._trigger_attribution(src_ip, threat.id, alert_payload), self.loop
                )

            except Exception as exc:
                logger.error("Failed to commit threat event: %s", exc)
            finally:
                db.close()

    async def _trigger_attribution(self, src_ip: str, threat_id: int, threat_details: dict):
        """Asynchronous passive OSINT lookup pipeline."""
        profile = await self.attribution_engine.build_profile(src_ip)
        
        # Save profile to Database
        db = get_session()
        try:
            # Check if profile already exists for IP
            existing = db.query(AttackerProfile).filter(AttackerProfile.ip == src_ip).first()
            if existing:
                existing.last_seen = time.time()
                existing.abuse_score = profile["abuse_score"]
                existing.threat_level = profile["threat_level"]
                attacker_id = existing.id
            else:
                new_profile = AttackerProfile(
                    ip=src_ip,
                    country=profile["country"],
                    city=profile["city"],
                    lat=profile["lat"],
                    lon=profile["lon"],
                    isp=profile["isp"],
                    asn=profile["asn"],
                    hostname=profile["hostname"],
                    abuse_score=profile["abuse_score"],
                    known_malicious=profile["known_malicious"],
                    threat_level=profile["threat_level"],
                )
                db.add(new_profile)
                db.commit()
                attacker_id = new_profile.id

            # Associate Profile with Threat
            db.query(Threat).filter(Threat.id == threat_id).update(
                {Threat.attacker_profile_id: attacker_id}
            )
            db.commit()

            # Publish updated attribution profile to live React dashboard
            profile_payload = {
                "event": "ATTRIBUTION_COMPLETE",
                "threat_id": threat_id,
                "profile": profile,
            }
            self.bus.publish(CHANNEL_LIVE_EVENTS, profile_payload)

            # Send smtp email alert to admin
            self.alerter.send_email_alert(profile, threat_details)

        except Exception as exc:
            logger.error("Failed to compile attribution OSINT profiles: %s", exc)
        finally:
            db.close()

    def _handle_honeypot_activity(self, event_type: str, data: dict):
        """Fires when watcher senses folder activity."""
        result = self.ransomware_detector.evaluate_activity(event_type, data)
        
        if result["is_ransomware"]:
            src_ip = data.get("src_ip", "127.0.0.1")
            logger.warning("Ransomware behavior detected! Renames: %d/s. Target: %s", 
                           result["renames_per_second"], result["target_path"])
            
            # Spin up Docker decoy honeypot trap container
            self.honeypot_manager.deploy_trap()

            # Apply iptables route forward to decoy Docker container immediately
            self.executor.apply_action(5, src_ip)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CortiX Firewall Daemon")
    parser.add_argument("--interface", type=str, default="eth0", help="Capturing network interface")
    parser.add_argument("--mode", type=str, default="live", choices=["live", "offline"], help="Capture mode")
    args = parser.parse_args()

    daemon = CortixDaemon(interface=args.interface, mode=args.mode)
    try:
        daemon.start()
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        daemon.stop()
