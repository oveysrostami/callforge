import pytest


@pytest.fixture(autouse=True)
def isolated_callforge_home(tmp_path, monkeypatch):
    """Tests must never use the user's active workspace/runtime registration."""
    monkeypatch.setenv("CALLFORGE_HOME", str(tmp_path / "user-state"))
    monkeypatch.delenv("CALLFORGE_ROOT", raising=False)
