from __future__ import annotations
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/v1/policies", tags=["Guardrails"])

@router.get("")
async def get_policies(request: Request) -> JSONResponse:
    state = request.app.state.app_state
    policy_text = await state.colosseum.get_policy_source()
    return JSONResponse({"policy": policy_text})

@router.post("")
async def update_policy(request: Request, body: dict[str, str]) -> JSONResponse:
    state = request.app.state.app_state
    new_policy = body.get("policy", "")
    result = await state.colosseum.update_policy(new_policy)
    return JSONResponse(result)
