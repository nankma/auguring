"""
Tests for vector_store/__init__.py's backend-selection dispatch -- the
actual "type: yaml_files vs type: sqlite_vec" seam this refactor exists
for. Mirrors tests/test_storage.py's own shape for storage.get_storage().
"""

import pytest
from trailsign import Settings

import app_settings
import vector_store
from vector_store.sqlite_vec_store import SqliteVecStore
from vector_store.yaml_files import YamlFilesStore


@pytest.fixture
def _restore_settings_and_vector_store():
    """Saves whatever Settings conftest.py's base fixture installed, and
    restores exactly that object afterward -- forcing a disk reload
    instead would risk picking up a real settings.yml for whichever test
    runs next. Mirrors test_storage.py's own fixture."""
    original = app_settings.get_settings()
    yield
    app_settings.reset_settings_for_tests(original)
    vector_store.reset_vector_store_for_tests(None)


def _settings_with_backend(backend: dict) -> Settings:
    return Settings({"news_cache": {"backend": backend}})


def test_get_vector_store_dispatches_to_yaml_files_by_default(tmp_path, _restore_settings_and_vector_store):
    # No news_cache.backend.type at all -- YamlFilesStore.__init__ still
    # needs storage.news_cache_dir.path, a separate, pre-existing
    # required setting, not this seam's concern.
    app_settings.reset_settings_for_tests(Settings({
        "storage": {"news_cache_dir": {"path": str(tmp_path)}},
    }))
    vector_store.reset_vector_store_for_tests(None)
    assert type(vector_store.get_vector_store()) is YamlFilesStore


def test_get_vector_store_dispatches_to_yaml_files_when_explicit(tmp_path, _restore_settings_and_vector_store):
    app_settings.reset_settings_for_tests(Settings({
        "news_cache": {"backend": {"type": "yaml_files"}},
        "storage": {"news_cache_dir": {"path": str(tmp_path)}},
    }))
    vector_store.reset_vector_store_for_tests(None)
    assert type(vector_store.get_vector_store()) is YamlFilesStore


def test_get_vector_store_dispatches_to_sqlite_vec(tmp_path, _restore_settings_and_vector_store):
    app_settings.reset_settings_for_tests(_settings_with_backend({
        "type": "sqlite_vec", "sqlite_vec": {"path": str(tmp_path / "vec.db")},
    }))
    vector_store.reset_vector_store_for_tests(None)
    assert type(vector_store.get_vector_store()) is SqliteVecStore


def test_get_vector_store_unknown_type_raises(_restore_settings_and_vector_store):
    app_settings.reset_settings_for_tests(_settings_with_backend({"type": "lancedb"}))
    vector_store.reset_vector_store_for_tests(None)
    with pytest.raises(ValueError):
        vector_store.get_vector_store()


def test_get_vector_store_is_a_singleton_within_one_reset_cycle(tmp_path, _restore_settings_and_vector_store):
    app_settings.reset_settings_for_tests(_settings_with_backend({
        "type": "sqlite_vec", "sqlite_vec": {"path": str(tmp_path / "vec.db")},
    }))
    vector_store.reset_vector_store_for_tests(None)
    assert vector_store.get_vector_store() is vector_store.get_vector_store()
