"""Profiles may tune narration without replacing the user's working capture/model setup."""
from dataclasses import asdict

import pytest

from semantic_image_stream.configuration import Settings
from semantic_image_stream.presets import AUDIOBOOK_PRESETS, apply_audiobook_preset


@pytest.mark.parametrize("key", AUDIOBOOK_PRESETS)
def test_audiobook_profile_preserves_working_audio_and_model_choices(tmp_path, key):
    original = Settings(audio_mode="microphone", audio_device=141, whisper_model="turbo",
                        whisper_device="cuda", whisper_compute_type="float16", language="fr",
                        silence_rms=0.02, clip_device="cpu", cache_dir=str(tmp_path),
                        attribution_visible=False, display_index=2, transition="zoom").validate()
    before = asdict(original)
    profile = apply_audiobook_preset(original, key)
    assert asdict(original) == before
    for field in ("audio_mode", "audio_device", "audio_file", "whisper_model", "whisper_device",
                  "whisper_compute_type", "language", "silence_rms", "chunk_seconds", "cache_dir",
                  "clip_model", "clip_device", "attribution_visible", "display_index", "transition"):
        assert getattr(profile, field) == getattr(original, field)
    assert profile.transcript_window_seconds > original.transcript_window_seconds
    assert profile.semantic_provider == "ollama"
    assert profile.ranking == "context_clip"
    profile.providers.append("duckduckgo")
    assert original.providers == before["providers"]
    assert "duckduckgo" not in AUDIOBOOK_PRESETS[key].providers


def test_unknown_profile_cannot_change_settings(tmp_path):
    settings = Settings(cache_dir=str(tmp_path))
    with pytest.raises(ValueError, match="supported audiobook"):
        apply_audiobook_preset(settings, "missing")
    assert settings.transcript_window_seconds == 45
