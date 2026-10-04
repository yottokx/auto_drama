"""Run bounded CPU loop analysis in the prepared music environment."""
import json
import subprocess
import tempfile
from pathlib import Path

from packages.contracts.music import MusicResult

from .m2_service import ROOT


def normalize_music(data, scene_id):
    if not data or len(data) > 32 * 1024 * 1024 or not isinstance(scene_id, str) or len(scene_id) > 128:
        raise ValueError("32MiB以内のBGMを指定してください。")
    config = json.loads((ROOT / "config/m2-generation.json").read_text(encoding="utf-8"))
    python = Path(config.get("music", {}).get("python", "services/worker/runtimes/stable_audio3/.venv/Scripts/python.exe"))
    python = python if python.is_absolute() else ROOT / python
    if not python.is_file():
        raise ValueError("先にStable Audioの専用環境を準備してください。")
    with tempfile.TemporaryDirectory(prefix="auto-drama-import-") as directory:
        temporary = Path(directory)
        wav_header = data[:4] in (b"RIFF", b"RF64") and data[8:12] == b"WAVE"
        source = temporary / ("input.wav" if wav_header else "input.mp3")
        source.write_bytes(data)
        request = temporary / "request.json"
        request.write_text(json.dumps({"scene_id": scene_id, "source_audio": str(source),
                           "prompt": "Imported instrumental background music.", "loop_target_seconds": 60,
                           "crossfade_seconds": 0.5}), encoding="utf-8")
        output = temporary / "output"
        try:
            process = subprocess.run([str(python), "-X", "utf8", "-m", "services.worker.generation.music_runner",
                                      "--import-request", str(request), "--output-dir", str(output)],
                                     cwd=ROOT, capture_output=True, timeout=180, check=False,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired as exc:
            raise ValueError("BGMの自動ループ探索が時間内に完了しませんでした。") from exc
        if process.returncode or not (output / "result.json").is_file():
            raise ValueError("BGMを取り込めませんでした。無音・破損音源や長さを確認してください。")
        report = json.loads((output / "result.json").read_text(encoding="utf-8"))
        if not report.get("ok"):
            raise ValueError("BGMの品質検査・ループ探索に失敗しました。")
        result = MusicResult.model_validate(report["result"]).model_dump(mode="json")
        if result["scene_id"] != scene_id:
            raise ValueError("BGMの場面が一致しません。")
        return {name: (output / name).read_bytes() for name in ("music.mp3", "source.mp3")}, result
