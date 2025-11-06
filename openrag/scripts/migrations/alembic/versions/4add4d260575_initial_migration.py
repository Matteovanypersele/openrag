"""Initial migration

Revision ID: 4add4d260575
Revises:
Create Date: 2025-10-27 12:32:52.093810

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


def table_exists(table_name: str) -> bool:
    """Check if a table exists in the database."""
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def index_exists(index_name: str, table_name: str) -> bool:
    """Check if an index exists on a table."""
    bind = op.get_bind()
    inspector = inspect(bind)
    indexes = inspector.get_indexes(table_name)
    return any(idx["name"] == index_name for idx in indexes)


# revision identifiers, used by Alembic.
revision: str = "4add4d260575"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create partitions table if it doesn't exist
    if not table_exists("partitions"):
        op.create_table(
            "partitions",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("partition", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )

    # Create indexes for partitions table if they don't exist
    if table_exists("partitions"):
        if not index_exists("ix_partitions_created_at", "partitions"):
            op.create_index(
                op.f("ix_partitions_created_at"),
                "partitions",
                ["created_at"],
                unique=False,
            )
        if not index_exists("ix_partitions_partition", "partitions"):
            op.create_index(
                op.f("ix_partitions_partition"),
                "partitions",
                ["partition"],
                unique=True,
            )

    # Create files table if it doesn't exist
    if not table_exists("files"):
        op.create_table(
            "files",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("file_id", sa.String(), nullable=False),
            sa.Column("partition_name", sa.String(), nullable=False),
            sa.Column("file_metadata", sa.JSON(), nullable=True),
            sa.ForeignKeyConstraint(
                ["partition_name"],
                ["partitions.partition"],
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "file_id", "partition_name", name="uix_file_id_partition"
            ),
        )

    # Create indexes for files table if they don't exist
    if table_exists("files"):
        if not index_exists("ix_files_file_id", "files"):
            op.create_index(
                op.f("ix_files_file_id"), "files", ["file_id"], unique=False
            )
        if not index_exists("ix_files_partition_name", "files"):
            op.create_index(
                op.f("ix_files_partition_name"),
                "files",
                ["partition_name"],
                unique=False,
            )
        if not index_exists("ix_partition_file", "files"):
            op.create_index(
                "ix_partition_file",
                "files",
                ["partition_name", "file_id"],
                unique=False,
            )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes and tables if they exist
    if table_exists("files"):
        if index_exists("ix_partition_file", "files"):
            op.drop_index("ix_partition_file", table_name="files")
        if index_exists("ix_files_partition_name", "files"):
            op.drop_index(op.f("ix_files_partition_name"), table_name="files")
        if index_exists("ix_files_file_id", "files"):
            op.drop_index(op.f("ix_files_file_id"), table_name="files")
        op.drop_table("files")

    if table_exists("partitions"):
        if index_exists("ix_partitions_partition", "partitions"):
            op.drop_index(op.f("ix_partitions_partition"), table_name="partitions")
        if index_exists("ix_partitions_created_at", "partitions"):
            op.drop_index(op.f("ix_partitions_created_at"), table_name="partitions")
        op.drop_table("partitions")
