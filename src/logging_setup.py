import logging
import asyncio
import structlog
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

# Global queue for UI log streaming
LOG_QUEUE: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)

class LogQueueHandler(logging.Handler):
    """
    Custom logging handler that pushes records to an asyncio.Queue.
    The Logger module (via WebSocket) consumes from this queue.
    """
    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        super().__init__()
        self.loop = loop

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Check if this record has 'structlog' context
            # ProcessorFormatter adds 'event' and other fields to the message 
            # or record actually.
            entry = {
                "timestamp": record.created,
                "level": record.levelname,
                "module": record.module,
                "message": record.getMessage(),
            }
            
            # If it's a structlog record, it might have been formatted already
            if hasattr(record, "msg") and isinstance(record.msg, str) and record.msg.startswith("{"):
                # Potential JSON log from structlog
                pass

            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(LOG_QUEUE.put_nowait, entry)
        except Exception:
            pass

def setup_logging(env: str = "development"):
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if env == "production":
        processors = shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ]
    else:
        processors = shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Standard logging bridge — console handler
    handler = logging.StreamHandler(sys.stdout)
    
    if env == "production":
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
        )
    else:
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True),
        )
    
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # ── File handler for data/logs/backend.log ──────────────
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "backend.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=False),
    )
    file_handler.setFormatter(file_formatter)
    root.addHandler(file_handler)

    # Add our custom queue handler for the UI
    queue_handler = LogQueueHandler()
    root.addHandler(queue_handler)
    
    # Silence some noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return queue_handler
