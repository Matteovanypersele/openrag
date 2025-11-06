"""create users memberships tables

Revision ID: cd642e4502d8
Revises: 4add4d260575
Create Date: 2025-10-27 15:00:40.022871

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
revision: str = "cd642e4502d8"
down_revision: Union[str, Sequence[str], None] = "4add4d260575"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create users table if it doesn't exist
    if not table_exists("users"):
        op.create_table(
            "users",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("external_user_id", sa.String(), nullable=True),
            sa.Column("display_name", sa.String(), nullable=True),
            sa.Column("token", sa.String(), nullable=True),
            sa.Column("is_admin", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )

    # Create indexes for users table if they don't exist
    if table_exists("users"):
        if not index_exists("ix_users_external_user_id", "users"):
            op.create_index(
                op.f("ix_users_external_user_id"),
                "users",
                ["external_user_id"],
                unique=True,
            )
        if not index_exists("ix_users_token", "users"):
            op.create_index(op.f("ix_users_token"), "users", ["token"], unique=True)

    # Create partition_memberships table if it doesn't exist
    if not table_exists("partition_memberships"):
        op.create_table(
            "partition_memberships",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("partition_name", sa.String(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("role", sa.String(), nullable=False),
            sa.Column("added_at", sa.DateTime(), nullable=False),
            sa.CheckConstraint(
                "role IN ('owner','editor','viewer')", name="ck_membership_role"
            ),
            sa.ForeignKeyConstraint(
                ["partition_name"], ["partitions.partition"], ondelete="CASCADE"
            ),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("partition_name", "user_id", name="uix_partition_user"),
        )

    # Create indexes for partition_memberships table if they don't exist
    if table_exists("partition_memberships"):
        if not index_exists(
            "ix_partition_memberships_partition_name", "partition_memberships"
        ):
            op.create_index(
                op.f("ix_partition_memberships_partition_name"),
                "partition_memberships",
                ["partition_name"],
                unique=False,
            )
        if not index_exists(
            "ix_partition_memberships_user_id", "partition_memberships"
        ):
            op.create_index(
                op.f("ix_partition_memberships_user_id"),
                "partition_memberships",
                ["user_id"],
                unique=False,
            )
        if not index_exists("ix_user_partition", "partition_memberships"):
            op.create_index(
                "ix_user_partition",
                "partition_memberships",
                ["user_id", "partition_name"],
                unique=False,
            )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes and tables if they exist
    if table_exists("partition_memberships"):
        if index_exists("ix_user_partition", "partition_memberships"):
            op.drop_index("ix_user_partition", table_name="partition_memberships")
        if index_exists("ix_partition_memberships_user_id", "partition_memberships"):
            op.drop_index(
                op.f("ix_partition_memberships_user_id"),
                table_name="partition_memberships",
            )
        if index_exists(
            "ix_partition_memberships_partition_name", "partition_memberships"
        ):
            op.drop_index(
                op.f("ix_partition_memberships_partition_name"),
                table_name="partition_memberships",
            )
        op.drop_table("partition_memberships")

    if table_exists("users"):
        if index_exists("ix_users_token", "users"):
            op.drop_index(op.f("ix_users_token"), table_name="users")
        if index_exists("ix_users_external_user_id", "users"):
            op.drop_index(op.f("ix_users_external_user_id"), table_name="users")
        op.drop_table("users")
