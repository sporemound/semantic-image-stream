"""Small local reference/model files must not allocate their whole size limit."""
from pathlib import Path

import pytest

from semantic_image_stream.learned_relevance import AdapterError, load_adapter
from semantic_image_stream.story_references import load_story_references


@pytest.mark.parametrize("filename", ["story_examples.json", "relevance-model.json"])
@pytest.mark.parametrize("memory_failure", [False, True])
def test_readers_use_actual_file_size_and_control_memory_failure(tmp_path, monkeypatch, filename, memory_failure):
    bundled = Path(__file__).parents[1] / "src" / "semantic_image_stream" / "data" / "training" / filename
    target = tmp_path / filename
    target.write_bytes(bundled.read_bytes())
    size = target.stat().st_size
    original = Path.open
    requests = []

    class Reader:
        def __init__(self, source):
            self.source = source

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.source.close()

        def fileno(self):
            return self.source.fileno()

        def read(self, requested=-1):
            requests.append(requested)
            if memory_failure or requested > size + 1:
                raise MemoryError("private diagnostic must stay local")
            return self.source.read(requested)

    def open_local(path, *args, **kwargs):
        source = original(path, *args, **kwargs)
        return Reader(source) if path == target and args == ("rb",) else source

    monkeypatch.setattr(Path, "open", open_local)
    if filename == "story_examples.json":
        pack = load_story_references(tmp_path)
        assert pack.status == ("invalid" if memory_failure else "ready")
    elif memory_failure:
        with pytest.raises(AdapterError) as error:
            load_adapter(target, "openai/clip-vit-base-patch32")
        assert "private diagnostic" not in str(error.value)
    else:
        assert load_adapter(target, "openai/clip-vit-base-patch32").artifact["report"]["train_ready"]
    assert requests[0] == size + 1
    assert target.stat().st_size == size
