"""Tests for filename-based track auto-discovery."""

import dnd_pipeline as dp


def touch(d, name):
    (d / name).write_bytes(b"")


def test_explicit_tracks_returned_unchanged():
    explicit = [{"file": "a.wav", "speaker": "S", "character": "C", "priority": 50}]
    out = dp.discover_tracks(None, {"session_name": "s", "tracks": explicit})
    assert out is explicit


def test_discovers_all_files_regardless_of_count(tmp_path):
    for n in ["dnd_2-Антернер.wav", "dnd_2-Шиян.wav", "dnd_2-Гай Гексан.wav"]:
        touch(tmp_path, n)
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2"})
    assert len(out) == 3


def test_five_vs_six_files(tmp_path):
    for n in ["dnd_2-A.wav", "dnd_2-B.wav", "dnd_2-C.wav", "dnd_2-D.wav", "dnd_2-E.wav"]:
        touch(tmp_path, n)
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2"})
    assert len(out) == 5


def test_speaker_character_parsed_from_dotted_name(tmp_path):
    touch(tmp_path, "dnd_2-Дима. Ангрон.wav")
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2"})
    assert out[0]["speaker"] == "Дима"
    assert out[0]["character"] == "Ангрон"


def test_single_name_sets_speaker_equals_character(tmp_path):
    touch(tmp_path, "dnd_2-Антернер.wav")
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2"})
    assert out[0]["speaker"] == "Антернер"
    assert out[0]["character"] == "Антернер"


def test_prefix_stripped(tmp_path):
    touch(tmp_path, "dnd_2-Шиян.wav")
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2"})
    assert out[0]["speaker"] == "Шиян"  # not "dnd_2-Шиян"


def test_master_mix_excluded(tmp_path):
    touch(tmp_path, "dnd_2-001.wav")
    touch(tmp_path, "dnd_2-Шиян.wav")
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2", "master_mix": "dnd_2-001.wav"})
    files = [t["file"] for t in out]
    assert "dnd_2-001.wav" not in files
    assert "dnd_2-Шиян.wav" in files


def test_exclude_list_honored(tmp_path):
    touch(tmp_path, "dnd_2-skip.wav")
    touch(tmp_path, "dnd_2-keep.wav")
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2", "exclude": ["dnd_2-skip.wav"]})
    files = [t["file"] for t in out]
    assert files == ["dnd_2-keep.wav"]


def test_dm_speaker_gets_priority_100(tmp_path):
    touch(tmp_path, "dnd_2-Данжен Мастер.wav")
    touch(tmp_path, "dnd_2-Дима. Ангрон.wav")
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2", "dm_speaker": "Данжен Мастер"})
    by_speaker = {t["speaker"]: t["priority"] for t in out}
    assert by_speaker["Данжен Мастер"] == 100
    assert by_speaker["Дима"] == 50


def test_non_audio_files_ignored(tmp_path):
    touch(tmp_path, "dnd_2-Шиян.wav")
    touch(tmp_path, "notes.txt")
    touch(tmp_path, "config.json")
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2"})
    assert [t["file"] for t in out] == ["dnd_2-Шиян.wav"]


def test_extension_match_is_case_insensitive(tmp_path):
    touch(tmp_path, "dnd_2-Шиян.WAV")
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2"})
    assert len(out) == 1


def test_results_sorted_by_filename(tmp_path):
    for n in ["dnd_2-C.wav", "dnd_2-A.wav", "dnd_2-B.wav"]:
        touch(tmp_path, n)
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2"})
    assert [t["file"] for t in out] == ["dnd_2-A.wav", "dnd_2-B.wav", "dnd_2-C.wav"]


def test_two_dot_separators_split_on_first_only(tmp_path):
    touch(tmp_path, "dnd_2-Дима. Ангрон. Прочее.wav")
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2"})
    assert out[0]["speaker"] == "Дима"
    assert out[0]["character"] == "Ангрон. Прочее"


def test_file_without_prefix_kept_intact(tmp_path):
    touch(tmp_path, "Безпрефикса.wav")
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2"})
    assert out[0]["speaker"] == "Безпрефикса"
    assert out[0]["character"] == "Безпрефикса"


def test_transcribe_all_uses_discovery_when_no_tracks(tmp_path, monkeypatch):
    # No `tracks` in config → transcribe_all should discover the two files and
    # call transcribe_track once per discovered track.
    (tmp_path / "dnd_2-Шиян.wav").write_bytes(b"")
    (tmp_path / "dnd_2-Дима. Ангрон.wav").write_bytes(b"")

    seen = []

    def fake_transcribe_track(model, session_dir, track, config, paths):
        seen.append(track["speaker"])
        return []

    monkeypatch.setattr(dp, "transcribe_track", fake_transcribe_track)
    cfg = {"session_name": "dnd_2", "transcription_backend": "mlx"}
    # Avoid loading a real MLX model: stub the model-load path.
    monkeypatch.setattr(dp, "mlx_whisper", object())
    paths = dp.build_paths(tmp_path, "dnd_2")
    dp.ensure_dirs(paths)

    dp.transcribe_all(tmp_path, cfg, paths)
    assert sorted(seen) == ["Дима", "Шиян"]


def test_empty_discovery_logs_warning(tmp_path, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="dnd_pipeline"):
        out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2"})
    assert out == []
    assert any("Не найдено аудиодорожек" in r.message and r.levelno == logging.WARNING
               for r in caplog.records)


def test_nonempty_discovery_no_warning(tmp_path, caplog):
    import logging
    (tmp_path / "dnd_2-Шиян.wav").write_bytes(b"")
    with caplog.at_level(logging.WARNING, logger="dnd_pipeline"):
        dp.discover_tracks(tmp_path, {"session_name": "dnd_2"})
    assert not any("Не найдено аудиодорожек" in r.message for r in caplog.records)
