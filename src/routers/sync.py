from fastapi import APIRouter, WebSocket, Request, HTTPException
from pydantic import BaseModel
import structlog

router = APIRouter(prefix="/api/v1/sync", tags=["Sync"])
logger = structlog.get_logger("aetherforge.sync_router")

class PairRequest(BaseModel):
    uri: str

@router.get("/info")
async def sync_info(request: Request):
    state = request.app.state.app_state
    if not state.sync_manager:
        raise HTTPException(status_code=404, detail="SyncEngine offline")
    
    # Render discovery status
    return {
        "status": "online",
        "node_id": state.sync_manager.node_id,
        "pairing_uri": state.sync_manager.discovery.get_pairing_uri(state.sync_manager.event_log.get_private_key()) if hasattr(state.sync_manager.event_log, 'get_private_key') else "aetherforge://sync?ip=127.0.0.1&port=8765&node=local&key=demo",
        "peers": [{"id": p, "ip": ip, "port": port} for p, (ip, port) in state.sync_manager.discovery.available_peers.items()] if hasattr(state.sync_manager.discovery, 'available_peers') else [],
        "authorized": list(state.sync_manager.authorized_peers.keys())
    }

@router.post("/pair")
async def sync_pair(payload: PairRequest, request: Request):
    state = request.app.state.app_state
    if not state.sync_manager:
        raise HTTPException(status_code=404, detail="SyncEngine offline")
    
    try:
        from src.modules.sync.crypto import SyncCrypto
        peer_id, ip, port, key = SyncCrypto.parse_pairing_uri(payload.uri)
        state.sync_manager.authorize_peer(peer_id, key)
        return {"status": "success", "peer_id": peer_id}
    except Exception as e:
        logger.warning("Pairing failed", error=str(e))
        raise HTTPException(status_code=400, detail=f"Invalid pairing URI: {str(e)}")

@router.websocket("/ws/{peer_node_id}")
async def sync_websocket(websocket: WebSocket, peer_node_id: str):
    state = websocket.app.state.app_state
    if not state.sync_manager:
        await websocket.close(code=1011, reason="SyncEngine offline")
        return
    await state.sync_manager.handle_websocket(websocket, peer_node_id)
