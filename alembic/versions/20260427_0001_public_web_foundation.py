"""Bootstrap public web foundation columns and fresh-schema setup."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "20260427_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    table_names = set(inspector.get_table_names())

    if not table_names:
        from asr_viz.db.base import Base
        from asr_viz import models  # noqa: F401

        Base.metadata.create_all(bind=bind)
        return

    _add_column_if_missing(
        inspector,
        "media_assets",
        sa.Column("owner_user_id", sa.String(length=255), nullable=False, server_default="local-dev-user"),
    )
    _add_column_if_missing(inspector, "media_assets", sa.Column("size_bytes", sa.Integer(), nullable=True))
    _add_column_if_missing(
        inspector,
        "processing_jobs",
        sa.Column("owner_user_id", sa.String(length=255), nullable=False, server_default="local-dev-user"),
    )
    _add_column_if_missing(
        inspector,
        "transcripts",
        sa.Column("owner_user_id", sa.String(length=255), nullable=False, server_default="local-dev-user"),
    )
    _add_column_if_missing(
        inspector,
        "stream_ingestion_sessions",
        sa.Column("owner_user_id", sa.String(length=255), nullable=False, server_default="local-dev-user"),
    )
    _add_column_if_missing(
        inspector,
        "live_sessions",
        sa.Column("owner_user_id", sa.String(length=255), nullable=False, server_default="local-dev-user"),
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    for table_name, column_name in (
        ("live_sessions", "owner_user_id"),
        ("stream_ingestion_sessions", "owner_user_id"),
        ("transcripts", "owner_user_id"),
        ("processing_jobs", "owner_user_id"),
        ("media_assets", "size_bytes"),
        ("media_assets", "owner_user_id"),
    ):
        if table_name not in inspector.get_table_names():
            continue
        if column_name not in {column["name"] for column in inspector.get_columns(table_name)}:
            continue
        with op.batch_alter_table(table_name) as batch_op:
            batch_op.drop_column(column_name)


def _add_column_if_missing(inspector, table_name: str, column: sa.Column) -> None:
    if table_name not in inspector.get_table_names():
        return
    existing_columns = {existing["name"] for existing in inspector.get_columns(table_name)}
    if column.name in existing_columns:
        return
    with op.batch_alter_table(table_name) as batch_op:
        batch_op.add_column(column)
