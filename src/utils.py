import asyncio
import structlog
from typing import Any, Optional

logger = structlog.get_logger("aetherforge.utils")

def safe_create_task(coro: Any, name: Optional[str] = None) -> asyncio.Task[Any]:
    """
    Wrapper for asyncio.create_task that logs unobserved exceptions.
    Prevents background tasks from failing silently.
    """
    task = asyncio.create_task(coro, name=name)
    
    def _handle_task_result(t: asyncio.Task[Any]) -> None:
        try:
            t.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Background task '{name or t.get_name()}' failed: {e}", exc_info=True)
            
    task.add_done_callback(_handle_task_result)
    return task
