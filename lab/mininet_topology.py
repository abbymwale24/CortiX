"""
CortiX Mininet Laboratory Network Topology

Builds an isolated Software-Defined Network (SDN) topology containing:
- h1: Attacker Node
- h2: Victim Node
- h3: Cortix Firewall Node routing internal traffic
- s1: OpenFlow Switch connecting nodes
"""

import logging

try:
    from mininet.topo import Topo
    from mininet.net import Mininet
    from mininet.node import OVSController
    from mininet.cli import CLI
except ImportError:
    Topo = object
    Mininet = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortix.lab.mininet_topology")


class CortixLabTopology(Topo):
    """
    Isolated 3-host lab topology for active threat testing.
    """

    def build(self):
        # Add Switch
        s1 = self.addSwitch("s1")

        # Add Attacker Node
        h1 = self.addHost("h1", ip="10.0.0.1/24")
        # Add Victim Node
        h2 = self.addHost("h2", ip="10.0.0.2/24")
        # Add Cortix Firewall Node
        h3 = self.addHost("h3", ip="10.0.0.3/24")

        # Add duplex links
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s1)


def start_lab():
    if Mininet is None:
        logger.warning("Mininet library not installed. Lab topologies cannot start.")
        return

    logger.info("Starting CortiX Mininet Lab Network topology...")
    topo = CortixLabTopology()
    net = Mininet(topo=topo, controller=OVSController)
    net.start()
    
    logger.info("Lab hosts deployed: h1 (attacker), h2 (victim), h3 (firewall)")
    CLI(net)
    net.stop()


if __name__ == "__main__":
    start_lab()
