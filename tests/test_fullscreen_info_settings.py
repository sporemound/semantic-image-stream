"""Fullscreen information choices persist without changing recording settings."""
from dataclasses import asdict

import pytest

from semantic_image_stream.configuration import Settings


FULLSCREEN_FIELDS = ("fullscreen_narrative", "fullscreen_scene", "fullscreen_subjects",
                     "fullscreen_title", "fullscreen_credits", "fullscreen_transcript")


def test_default_and_older_configuration_enable_each_fullscreen_information_section(tmp_path):
    default = Settings.load()
    assert all(getattr(default, field) is True for field in FULLSCREEN_FIELDS)
    path = tmp_path / "older-config.toml"
    path.write_text('chunk_seconds = 3.0\naudio_mode = "microphone"\nwhisper_model = "tiny"\n', encoding="utf-8")
    restored = Settings.load(path)
    assert all(getattr(restored, field) is True for field in FULLSCREEN_FIELDS)
    assert restored.chunk_seconds == 3.0 and restored.audio_mode == "microphone" and restored.whisper_model == "tiny"


@pytest.mark.parametrize("choices", [
    dict.fromkeys(FULLSCREEN_FIELDS, False),
    {field: index % 2 == 0 for index, field in enumerate(FULLSCREEN_FIELDS)},
    dict.fromkeys(FULLSCREEN_FIELDS, True),
])
def test_fullscreen_choices_roundtrip_with_recording_and_other_preferences_intact(tmp_path, choices):
    settings = Settings(**choices, chunk_seconds=3.0, audio_mode="microphone", audio_device=7,
                        silence_seconds=.6, audio_queue_size=6, whisper_model="tiny", language="en",
                        semantic_interval=2.5, book_context="A voyage across an ice field.",
                        book_guidance_strength=.3, retain_context=False, slideshow_interval=11.0,
                        providers=["archive", "wikimedia"], transition="cut",
                        cache_dir=str(tmp_path / "cache")).validate()
    path = tmp_path / "config.toml"
    settings.save(path)
    restored = Settings.load(path)
    assert asdict(restored) == asdict(settings)
    for field, value in choices.items():
        assert getattr(restored, field) is value
    assert restored.chunk_seconds == 3.0


@pytest.mark.parametrize("field", FULLSCREEN_FIELDS)
@pytest.mark.parametrize("invalid", [0, 1, "true", "false", None, [], {}])
def test_fullscreen_choices_reject_values_that_are_not_booleans(field, invalid):
    with pytest.raises(ValueError, match=f"{field} must be true or false"):
        Settings(**{field: invalid}).validate()


@pytest.mark.parametrize("field", FULLSCREEN_FIELDS)
def test_false_fullscreen_choice_is_valid_and_does_not_change_unrelated_settings(field):
    default = asdict(Settings().validate())
    selected = asdict(Settings(**{field: False}).validate())
    assert selected.pop(field) is False
    default.pop(field)
    assert selected == default


def test_loading_numeric_fullscreen_choice_rejects_invalid_toml_type(tmp_path):
    path = tmp_path / "bad-config.toml"
    path.write_text("fullscreen_transcript = 0\nchunk_seconds = 3.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fullscreen_transcript must be true or false"):
        Settings.load(path)
