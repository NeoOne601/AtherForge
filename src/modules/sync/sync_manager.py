import asyncio
import json
from typing import Any

import structlog
import websockets
from fastapi import WebSocket, WebSocketDisconnect

from src.modules.sync.discovery import AetherForgeDiscovery
from src.modules.sync.event_log import EventLog

logger = structlog.get_logger("aetherforge.sync.manager")


class SyncManager:
    """
    Orchestrates the P2P Zero-Knowledge Sync.
    1. Manages the SQLite Event Log.
    2. Runs the mDNS discovery beacon.
    3. Handles incoming FastAPI WebSockets from peers.
    4. Connects outward to discovered peers to merge CRDT event logs.
    """

    def __init__(self, node_id: str, port: int, event_log: EventLog):
        self.node_id = node_id
        self.port = port
        self.event_log = event_log
        self.discovery = AetherForgeDiscovery(
            node_id=node_id,
            port=port,
            on_peer_joined=self._on_peer_joined,
            on_peer_left=self._on_peer_left,
        )
        # In memory storage of authorized peer keys (node_id -> AES_KEY_B64)
        # In a real app, these would be loaded from a secure persistent keychain
        self.authorized_peers: dict[str, str] = {}
        self._active_connections: dict[str, WebSocket] = {}

    async def start(self):
        """Starts mDNS discovery."""
        await self.discovery.start()
        logger.info("SyncManager started.")

    async def stop(self):
        """Stops mDNS discovery."""
        await self.discovery.stop()
        logger.info("SyncManager stopped.")

    def authorize_peer(self, peer_node_id: str, key_b64: str):
        """Authorizes a peer by storing its symmetric E2EE key."""
        self.authorized_peers[peer_node_id] = key_b64
        logger.info(f"Authorized peer: {peer_node_id}")

    # --- INCOMING CONNECTIONS (Server) ---

    async def handle_websocket(self, websocket: WebSocket, peer_node_id: str):
        """
        FastAPI endpoint handler for incoming P2P sync connections.
        Expected path: /api/v1/sync/ws/{peer_node_id}
        """
        await websocket.accept()

        if peer_node_id not in self.authorized_peers:
            logger.warning(f"Rejected unauthorized sync attempt from {peer_node_id}")
            await websocket.close(code=4003, reason="Unauthorized")
            return

        self._active_connections[peer_node_id] = websocket
        logger.info(f"Accepted inbound sync connection from {peer_node_id}")

        try:
            # 1. Wait for the peer to send their high-water mark (last seen HLC)
            data = await websocket.receive_json()
            peer_high_water = data.get("last_hlc", "0")

            # 2. Send them all events they are missing
            my_events = self.event_log.get_events_since(peer_high_water)
            for event in my_events:
                # Event is already encrypted in DB, just proxy it
                await websocket.send_json({"type": "event", "data": event})

            # 3. Enter listen loop to receive streaming events from them
            while True:
                msg = await websocket.receive_json()
                if msg.get("type") == "event":
                    ev_data = msg["data"]
                    # Merge into our CRDT WAL
                    merged = self.event_log.merge_foreign_event(ev_data)
                    if merged:
                        logger.debug(f"Merged inbound event {ev_data['id']} from {peer_node_id}")

        except WebSocketDisconnect:
            logger.info(f"Peer {peer_node_id} disconnected.")
        except Exception as e:
            logger.error(f"Sync error with {peer_node_id}: {e}")
        finally:
            self._active_connections.pop(peer_node_id, None)

    # --- OUTGOING CONNECTIONS (Client) ---

    def _on_peer_joined(self, peer_node_id: str, ip: str, port: int):
        """Triggered by mDNS when a peer appears on the network."""
        logger.info(f"Peer joined: {peer_node_id} on {ip}:{port}")

        # If we trust this peer, initiate a connection to pull their data
        if peer_node_id in self.authorized_peers:

            def _handle_done(t: asyncio.Task[Any]):
                try:
                    t.result()
                except Exception as e:
                    logger.error(f"Outgoing sync task failed for {peer_node_id}: {e}")

            task = asyncio.create_task(self._connect_to_peer(peer_node_id, ip, port))
            task.add_done_callback(_handle_done)

    def _on_peer_left(self, peer_node_id: str):
        """Triggered by mDNS when a peer drops off the network."""
        logger.info(f"Peer left: {peer_node_id}")

    async def _connect_to_peer(self, peer_node_id: str, ip: str, port: int):
        """Connects outwardly to a discovered peer to pull events."""
        uri = f"ws://{ip}:{port}/api/v1/sync/ws/{self.node_id}"
        logger.info(f"Connecting outpatient sync to {uri}...")

        try:
            async with websockets.connect(uri) as ws:
                # 1. Ask for everything since the dawn of time (or our latest HLC)
                # In a real impl, we'd query SELECT MAX(hlc_timestamp) FROM sync_events WHERE origin_device_id = peer_node_id
                await ws.send(json.dumps({"last_hlc": "0"}))

                # 2. Receive missing events
                while True:
                    msg_str = await ws.recv()
                    msg = json.loads(msg_str)
                    if msg.get("type") == "event":
                        ev_data = msg["data"]
                        merged = self.event_log.merge_foreign_event(ev_data)
                        if merged:
                            logger.debug(f"Merged pulled event {ev_data['id']} from {peer_node_id}")
        except Exception as e:
            logger.warning(f"Failed to connect to peer {peer_node_id}: {e}")
