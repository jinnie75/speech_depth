from sqlalchemy import inspect

from asr_viz.db.base import Base
from asr_viz.db.session import engine
from asr_viz import models  # noqa: F401


def _assert_schema_compatibility() -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())

    if "analysis_results" in table_names:
        analysis_columns = {column["name"] for column in inspector.get_columns("analysis_results")}
        legacy_analysis_columns = {"topic_label_raw", "topic_label_normalized", "topic_shift_probability"}
        if analysis_columns.intersection(legacy_analysis_columns):
            raise RuntimeError(
                "Database schema is from the old scene/topic version. "
                "Delete or migrate `asr_viz.db` before starting the app."
            )

    legacy_scene_tables = {"scene_clusters", "scene_cluster_members"}
    if table_names.intersection(legacy_scene_tables):
        raise RuntimeError(
            "Database schema still contains legacy scene tables. "
            "Delete or migrate `asr_viz.db` before starting the app."
        )


def init_db() -> None:
    _assert_schema_compatibility()
    Base.metadata.create_all(bind=engine)
