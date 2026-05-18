"""
CortiX Module 6 — Alerter Engine

Constructs and sends SMTP email alerts to the administrator 
when a high-threat intrusion is confirmed.
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from cortix.config import config

logger = logging.getLogger("cortix.attribution.alerter")


class ThreatAlerter:
    """
    SMTP Alerter Engine.
    """

    def send_email_alert(self, profile: dict, threat_details: dict) -> bool:
        """
        Send comprehensive intrusion attribution report to configured admin email.
        """
        admin_email = config.ADMIN_EMAIL
        smtp_user = config.SMTP_USER
        smtp_password = config.SMTP_PASSWORD

        if not smtp_user or not smtp_password:
            logger.debug("SMTP credentials not configured. Skipping email dispatch.")
            return False

        try:
            # Format subject & body templates
            subject = f"[CortiX ALERT] {profile.get('threat_level', 'HIGH')} threat from {profile.get('country', 'US')} — {profile.get('ip')}"
            
            body = f"""
Threat detected at {threat_details.get('timestamp')}
Attack type: {threat_details.get('attack_class', 'Anomaly')} (confidence: {threat_details.get('confidence', 0)*100:.1f}%)
Containment Action: {threat_details.get('action_taken', 'TEMP_BLOCK')}

ATTACKER PROFILE:
  IP Address: {profile.get('ip')}
  Location: {profile.get('city', 'Unknown')}, {profile.get('country', 'US')} ({profile.get('lat')}, {profile.get('lon')})
  ISP/ASN: {profile.get('isp', 'Unknown')} / {profile.get('asn', 'Unknown')}
  Hostname: {profile.get('hostname', 'Unknown')}
  Abuse Score: {profile.get('abuse_score', 0)}/100
  VirusTotal: {profile.get('vt_malicious', 0)} malicious verdicts
  Shodan open ports: {profile.get('shodan_ports', [])}

THREAT DETAILS:
  Z-Score: {threat_details.get('z_score', 0.0):.2f}
  Classifier confidence: {threat_details.get('confidence', 0.0)*100:.1f}%

Action taken: {threat_details.get('action_taken')}
Review dashboard: http://localhost:{config.DASHBOARD_PORT}

-- CortiX Adaptive Firewall v1.0
"""

            msg = MIMEMultipart()
            msg["From"] = smtp_user
            msg["To"] = admin_email
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            # Dispatch SMTP
            with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, admin_email, msg.as_string())

            logger.info("Intrusion alert successfully sent to %s", admin_email)
            return True

        except Exception as exc:
            logger.error("Failed to dispatch threat SMTP alert: %s", exc)

        return False
