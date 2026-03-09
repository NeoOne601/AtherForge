import asyncio
import structlog
import socket
from typing import Callable

from zeroconf import IPVersion, ServiceBrowser, ServiceInfo, ServiceStateChange, Zeroconf

logger = structlog.get_logger("aetherforge.sync.discovery")

class AetherForgeDiscovery:
    """
    Handles Zero-Config (mDNS/Bonjour) discovery for local LAN P2P sync.
    Advertises this node's presence and maintains a list of remote peers.
    """
    SERVICE_TYPE = "_aetherforge._tcp.local."

    def __init__(self, node_id: str, port: int, on_peer_joined: Callable[[str, str, int], None] | None = None, on_peer_left: Callable[[str], None] | None = None):
        self.node_id = node_id
        self.port = port
        self.zeroconf: Zeroconf | None = None
        self.service_info: ServiceInfo | None = None
        self.browser: ServiceBrowser | None = None
        self._active_peers: dict[str, dict[str, int | str]] = {}
        
        self.on_peer_joined = on_peer_joined
        self.on_peer_left = on_peer_left

    def get_local_ip(self) -> str:
        """Get the primary local IP address of the machine."""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            try:
                # Doesn't have to be reachable, just forces socket resolution
                s.connect(('10.255.255.255', 1))
                ip = s.getsockname()[0]
            except Exception:
                ip = '127.0.0.1'
        return ip

    async def start(self) -> None:
        """Start advertising as an AetherForge node and browsing for others."""
        self.zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
        
        # 1. Advertise ourselves
        local_ip = self.get_local_ip()
        logger.info(f"Starting mDNS Beacon: Node {self.node_id} on {local_ip}:{self.port}")
        
        self.service_info = ServiceInfo(
            type_=self.SERVICE_TYPE,
            name=f"{self.node_id}.{self.SERVICE_TYPE}",
            addresses=[socket.inet_aton(local_ip)],
            port=self.port,
            properties=b"",
            server=f"{self.node_id}.local.",
        )
        # Offload blocking call to background thread to prevent EventLoopBlocked
        try:
            await asyncio.to_thread(self.zeroconf.register_service, self.service_info)
        except Exception as e:
            # Handle name collisions gracefully (common if previous run didn't unregister)
            from zeroconf import NonUniqueNameException
            if isinstance(e, NonUniqueNameException) or "NonUniqueNameException" in str(e):
                logger.warning(f"mDNS name collision for {self.node_id}, retrying with suffix...")
                import time
                suffix = int(time.time()) % 1000
                self.service_info = ServiceInfo(
                    type_=self.SERVICE_TYPE,
                    name=f"{self.node_id}-{suffix}.{self.SERVICE_TYPE}",
                    addresses=[socket.inet_aton(local_ip)],
                    port=self.port,
                    properties=b"",
                    server=f"{self.node_id}-{suffix}.local.",
                )
                await asyncio.to_thread(self.zeroconf.register_service, self.service_info)
            else:
                raise e

        # 2. Browse for peers
        self.browser = ServiceBrowser(self.zeroconf, self.SERVICE_TYPE, handlers=[self._on_service_state_change])

    async def stop(self) -> None:
        """Stop advertising and shutdown mdns cleanly."""
        if self.zeroconf:
            if self.service_info:
                await asyncio.to_thread(self.zeroconf.unregister_service, self.service_info)
            self.zeroconf.close()
            self.zeroconf = None
        logger.info("mDNS Beacon stopped.")

    def get_active_peers(self) -> list[dict[str, str | int]]:
        return list(self._active_peers.values())

    def _on_service_state_change(self, zeroconf: Zeroconf, service_type: str, name: str, state_change: ServiceStateChange) -> None:
        """Callback from Zeroconf when a peer is discovered or lost."""
        if state_change is ServiceStateChange.Added:
            # New Node Discovered
            info = zeroconf.get_service_info(service_type, name)
            if info:
                # Ignore self
                if name.startswith(self.node_id):
                    return
                
                ip = socket.inet_ntoa(info.addresses[0]) if info.addresses else "127.0.0.1"
                port = info.port
                peer_node_id = name.split(".")[0]
                
                logger.info(f"Discovered peer: {peer_node_id} at {ip}:{port}")
                self._active_peers[peer_node_id] = {"id": peer_node_id, "ip": ip, "port": port}
                
                if self.on_peer_joined:
                    self.on_peer_joined(peer_node_id, ip, int(port))
                    
        elif state_change is ServiceStateChange.Removed:
            # Node Disconnected
            peer_node_id = name.split(".")[0]
            if peer_node_id in self._active_peers:
                logger.info(f"Peer lost: {peer_node_id}")
                del self._active_peers[peer_node_id]
                
                if self.on_peer_left:
                    self.on_peer_left(peer_node_id)
