import copy
import json
from types import SimpleNamespace

import pytest

from semantic_image_stream import story_references as references


def example(index=0, source="The wooden ship sailed past a stone tower.", answer="stone tower"):
    return {"id": f"human-train-{index}", "source": source,
            "question": "What building did the wooden ship pass?", "answer": answer, "attribute": "setting",
            "provenance": {"dataset": references.DATASET, "revision": "a" * 40,
                           "split": "train", "license": "apache-2.0", "story_name": f"ship-voyage-{index}",
                           "row_index": index, "answer_type": "explicit",
                           "license_url": "https://github.com/uci-soe/FairytaleQAData/blob/main/LICENSE"}}


def write_pack(tmp_path, rows=None):
    data = {"version": 1, "examples": rows if rows is not None else [example()]}
    path = tmp_path / "story_examples.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_verified_human_training_examples_are_local_bounded_and_detached(tmp_path):
    rows = [example(index) for index in range(3)]
    write_pack(tmp_path, rows)
    pack = references.load_story_references(tmp_path)
    assert pack.status == "ready" and len(pack.examples) == 3
    assert len(pack.identity) == 64
    selected = pack.select("The wooden ship was near a tower.")
    assert len(selected) == 2
    assert len(references._encoded(selected)) <= 1200
    assert selected[0] == rows[0]
    selected[0]["provenance"]["split"] = "test"
    assert pack.examples[0]["provenance"]["split"] == "train"
    assert pack.select("The avalanche buried a snowy hillside.") == []


def test_selection_uses_shared_content_not_an_incidental_character_name(tmp_path):
    row = example(source="Zelda passed a stone tower.")
    row["question"] = "Where did Zelda go?"
    write_pack(tmp_path, [row])
    pack = references.load_story_references(tmp_path)
    assert pack.select("Zelda") == []
    assert pack.select("A tower stood beside the road.")


def test_long_source_is_excerpted_literally_without_truncating_human_answer(tmp_path):
    source = "The wooden ship sailed quietly through the narrow channel. " * 15 + "A stone tower rose beside it."
    row = example(source=source)
    write_pack(tmp_path, [row])
    selected = references.load_story_references(tmp_path).select("The wooden ship passed a tower.")
    assert len(selected) == 1
    assert len(references._encoded(selected)) <= 1200
    assert selected[0]["source"] in source
    assert row["answer"] in selected[0]["source"]
    assert selected[0]["question"] == row["question"]
    assert selected[0]["answer"] == row["answer"]
    assert selected[0]["provenance"] == row["provenance"]


def test_json_escape_lengths_are_included_in_prompt_budget(tmp_path):
    source = ('"The wooden ship passed a stone tower."\n' * 24)[:1200]
    write_pack(tmp_path, [example(source=source)])
    pack = references.load_story_references(tmp_path)
    selected = pack.select("The wooden ship passed a tower.")
    assert selected and len(references._encoded(selected)) <= references.MAX_PROMPT_CHARS
    assert selected[0]["source"] in source
    assert selected[0]["answer"] in selected[0]["source"]


@pytest.mark.parametrize("mutate", [
    lambda row: row["provenance"].update(split="test"),
    lambda row: row["provenance"].update(split="validation"),
    lambda row: row["provenance"].update(answer_type="implicit"),
    lambda row: row["provenance"].update(dataset="synthetic/story-generator"),
    lambda row: row["provenance"].update(license="unknown"),
    lambda row: row["provenance"].update(revision="main"),
    lambda row: row["provenance"].update(row_index=True),
    lambda row: row.update(attribute="prediction"),
    lambda row: row.update(answer="A palace absent from the source."),
    lambda row: row.update(source="x" * 1201),
    lambda row: row.update(question="x" * 241),
    lambda row: row.update(answer="x" * 301),
    lambda row: row.update(extra="unverified"),
])
def test_unverified_or_held_out_pack_is_rejected_as_a_whole(tmp_path, mutate):
    invalid = example(1)
    mutate(invalid)
    write_pack(tmp_path, [example(), invalid])
    pack = references.load_story_references(tmp_path)
    assert pack.status == "invalid" and pack.examples == () and pack.identity == ""
    assert pack.select("The wooden ship passed a tower.") == []


@pytest.mark.parametrize("invalid", [{"version": True, "examples": []}, {"version": 2, "examples": []},
                                    {"version": 1, "examples": []}, ["unverified"],
                                    {"version": 1, "examples": [example(), example()]}])
def test_malformed_or_duplicate_pack_stays_unused(tmp_path, invalid):
    (tmp_path / "story_examples.json").write_text(json.dumps(invalid), encoding="utf-8")
    assert references.load_story_references(tmp_path).status == "invalid"


def test_missing_malformed_and_oversized_files_have_safe_status(tmp_path):
    assert references.load_story_references(tmp_path).status == "missing"
    path = tmp_path / "story_examples.json"
    path.write_text('{"private text": invalid}', encoding="utf-8")
    config = SimpleNamespace(training_dir=str(tmp_path), use_story_examples=True)
    status = references.inspect_story_examples(config)
    assert status["status"] == "invalid" and status["examples"] == 0
    assert "private text" not in status["message"]
    path.write_bytes(b" " * (references.MAX_PACK_BYTES + 1))
    assert references.load_story_references(tmp_path).status == "invalid"


def test_disabled_examples_preserve_file_and_do_not_load_it(tmp_path, monkeypatch):
    path = write_pack(tmp_path)
    before = path.read_bytes()
    config = SimpleNamespace(training_dir=str(tmp_path), use_story_examples=False)

    def forbidden(*args):
        raise AssertionError("Disabled references must not be loaded")

    monkeypatch.setattr(references, "load_story_references", forbidden)
    status = references.inspect_story_examples(config)
    assert status["status"] == "disabled" and status["examples"] == 0
    assert path.read_bytes() == before


def test_pack_identity_tracks_actual_example_changes_and_ready_status(tmp_path):
    row = example()
    write_pack(tmp_path, [row])
    initial = references.load_story_references(tmp_path).identity
    changed = copy.deepcopy(row)
    changed["question"] = "What tower stood beside the ship?"
    write_pack(tmp_path, [changed])
    assert references.load_story_references(tmp_path).identity != initial
    status = references.inspect_story_examples(SimpleNamespace(training_dir=str(tmp_path)))
    assert status["status"] == "ready" and status["examples"] == 1
