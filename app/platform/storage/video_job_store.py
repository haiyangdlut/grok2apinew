"""MySQL-backed video job persistence.

Replaces the in-memory ``_VIDEO_JOBS`` dict so video jobs survive restarts.
Uses the project's existing SQLAlchemy + aiomysql infrastructure.
"""

import json
import time
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.platform.logging.logger import logger

_TBL = "video_jobs"

metadata = sa.MetaData()

video_jobs_table = sa.Table(
    _TBL,
    metadata,
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("created_at", sa.BigInteger, nullable=False),
    sa.Column("status", sa.String(20), nullable=False, default="queued"),
    sa.Column("model", sa.String(128), nullable=False),
    sa.Column("progress", sa.Integer, nullable=False, default=0),
    sa.Column("prompt", sa.Text, nullable=False),
    sa.Column("seconds", sa.String(10), nullable=False, default=""),
    sa.Column("size", sa.String(20), nullable=False, default=""),
    sa.Column("quality", sa.String(20), nullable=False, default="standard"),
    sa.Column("completed_at", sa.BigInteger),
    sa.Column("content_path", sa.Text),
    sa.Column("video_url", sa.Text),
    sa.Column("error", sa.Text),
    sa.Column("remixed_from_video_id", sa.String(64)),
    sa.Column("updated_at", sa.BigInteger, nullable=False),
)


def _job_to_row(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job["id"],
        "created_at": job["created_at"],
        "status": job["status"],
        "model": job["model"],
        "progress": job["progress"],
        "prompt": job["prompt"],
        "seconds": job["seconds"],
        "size": job["size"],
        "quality": job["quality"],
        "completed_at": job.get("completed_at"),
        "content_path": job.get("content_path"),
        "video_url": job.get("video_url"),
        "error": json.dumps(job["error"]) if job.get("error") else None,
        "remixed_from_video_id": job.get("remixed_from_video_id"),
        "updated_at": job.get("updated_at", 0),
    }


def _row_to_job(row: Any) -> dict[str, Any]:
    d = dict(row._mapping)
    error = d.pop("error", None)
    return {
        "id": d["id"],
        "created_at": d["created_at"],
        "status": d["status"],
        "model": d["model"],
        "progress": d["progress"],
        "prompt": d["prompt"],
        "seconds": d["seconds"],
        "size": d["size"],
        "quality": d["quality"],
        "completed_at": d.get("completed_at"),
        "content_path": d.get("content_path"),
        "video_url": d.get("video_url"),
        "error": json.loads(error) if error else None,
        "remixed_from_video_id": d.get("remixed_from_video_id"),
    }


class VideoJobStore:
    """Async MySQL persistence for video jobs."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._session = async_sessionmaker(engine, expire_on_commit=False)
        self._initialized = False

    async def ensure_table(self) -> None:
        if self._initialized:
            return
        async with self._engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
        self._initialized = True
        logger.info("video_jobs table ready")

    async def insert_job(self, job: dict[str, Any]) -> None:
        await self.ensure_table()
        row = _job_to_row(job)
        async with self._engine.begin() as conn:
            await conn.execute(video_jobs_table.insert().values(**row))

    async def update_job(self, job: dict[str, Any]) -> None:
        await self.ensure_table()
        row = _job_to_row(job)
        async with self._engine.begin() as conn:
            await conn.execute(
                video_jobs_table.update()
                .where(video_jobs_table.c.id == job["id"])
                .values(**row)
            )

    async def get_job(self, video_id: str) -> dict[str, Any] | None:
        await self.ensure_table()
        async with self._engine.connect() as conn:
            result = await conn.execute(
                sa.select(video_jobs_table).where(video_jobs_table.c.id == video_id)
            )
            row = result.fetchone()
            if row is None:
                return None
            return _row_to_job(row)

    async def delete_job(self, video_id: str) -> None:
        await self.ensure_table()
        async with self._engine.begin() as conn:
            await conn.execute(
                video_jobs_table.delete().where(video_jobs_table.c.id == video_id)
            )

    async def find_stuck_in_progress_jobs(
        self, cutoff_unix: int | None = None
    ) -> list[dict[str, Any]]:
        """Return in_progress jobs created after *cutoff_unix* (default: now - 3600)."""
        await self.ensure_table()
        if cutoff_unix is None:
            cutoff_unix = int(time.time()) - 3600
        async with self._engine.connect() as conn:
            result = await conn.execute(
                sa.select(video_jobs_table).where(
                    sa.and_(
                        video_jobs_table.c.status == "in_progress",
                        video_jobs_table.c.created_at >= cutoff_unix,
                    )
                )
            )
            rows = result.fetchall()
            return [_row_to_job(row) for row in rows]

    async def close(self) -> None:
        await self._engine.dispose()


__all__ = ["VideoJobStore"]
