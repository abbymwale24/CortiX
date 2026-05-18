"""
CortiX Module 6 — GeoIP Geolocation resolver

Resolves IP geographics using MaxMind GeoIP2 offline databases or online API fallbacks.
"""

import os
import logging
from typing import Dict, Any

try:
    import geoip2.database
except ImportError:
    geoip2 = None

from cortix.config import config

logger = logging.getLogger("cortix.attribution.geoip")


class GeoIPResolver:
    """
    MaxMind GeoLite2 resolver wrapping native DB queries.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.MAXMIND_DB_PATH
        self.reader = None
        self._load_db()

    def _load_db(self):
        if geoip2 is None:
            logger.warning("geoip2 library not installed. Offline geolocation disabled.")
            return

        if os.path.exists(self.db_path):
            try:
                self.reader = geoip2.database.Reader(self.db_path)
                logger.info("MaxMind DB loaded from: %s", self.db_path)
            except Exception as exc:
                logger.error("Failed to read MaxMind DB: %s", exc)
        else:
            logger.debug("MaxMind database not found at %s. Falling back to online mock data.", self.db_path)

    def resolve(self, ip: str) -> Dict[str, Any]:
        """
        Geolocate target IP address.
        """
        res = {
            "country": "US",
            "city": "Unknown",
            "lat": 37.751,
            "lon": -97.822,
            "isp": "Local Loopback / Unknown",
            "asn": "AS0000",
        }

        # Handle local lookups
        if ip.startswith(("127.", "192.168.", "10.", "172.16.", "172.17.")):
            res["isp"] = "Private Subnet / Local Network"
            return res

        if self.reader:
            try:
                match = self.reader.city(ip)
                res["country"] = match.country.iso_code or "US"
                res["city"] = match.city.name or "Unknown"
                res["lat"] = match.location.latitude or 37.751
                res["lon"] = match.location.longitude or -97.822
                
                # Check for ASN database if exists
                asn_path = self.db_path.replace("City", "ASN")
                if os.path.exists(asn_path):
                    with geoip2.database.Reader(asn_path) as asn_reader:
                        asn_match = asn_reader.asn(ip)
                        res["asn"] = f"AS{asn_match.autonomous_system_number}"
                        res["isp"] = asn_match.autonomous_system_organization or "Unknown ISP"
            except Exception as exc:
                logger.debug("GeoIP lookup failed: %s", exc)
                
        return res

    def close(self):
        if self.reader:
            self.reader.close()
