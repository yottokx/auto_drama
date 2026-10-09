"""Record the app-owned Tyrano player in headless Chromium, including Web Audio."""

from __future__ import annotations

import argparse
import base64
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
RECORDING_FADE_MILLISECONDS = 400
# Reader-only controls and indicators. The window, speaker name, page counter and
# wait marks stay, since they are part of how the story is shown.
RECORDING_STYLE = ("#adn-foot,#adn-gauge,#adn-voice,#adn-toast,#adn-end,#adn-log,#adn-config"
                   "{visibility:hidden!important}")

# The viewing screen reads its settings from this browser-wide key before it starts.
SETTINGS_KEY = "adn_settings_v1"
TEXT_SIZES = {"small": 21, "standard": 24, "large": 28}


class RecordingStopped(RuntimeError):
    """The user asked to stop; what was captured so far is kept and still joined."""

    def __init__(self, path: Path):
        super().__init__(f"停止要求を受け付けました。 途中の録画: {path}")
        self.path = path


def viewing_settings(args) -> dict:
    """Recorder options in the viewing screen's own units."""
    return {"size": TEXT_SIZES[args.text_size], "face": args.typeface, "speed": args.text_speed,
            "wait": round(args.auto_wait * 10), "voice": args.voice_volume, "bgm": args.bgm_volume,
            "alpha": args.window_opacity}


def settings_script(args) -> str:
    """Runs before the page's own scripts in each chapter's fresh browser profile."""
    key, value = json.dumps(SETTINGS_KEY), json.dumps(json.dumps(viewing_settings(args)))
    return f"try {{ window.localStorage.setItem({key}, {value}); }} catch (_) {{}}"


PLAYER_PROGRESS = """() => {
  const k = window.TYRANO.kag;
  let current = 0, total = 0;
  k.ftag.array_tag.forEach((tag, index) => {
    if (tag.name === 'label' && tag.pm.label_name.startsWith('utterance_')) {
      total++;
      if (index <= k.ftag.current_order_index) current = total;
    }
  });
  return {utterance_current: current, utterance_total: total};
}"""

AUDIO_START = """async () => {
  const h = window.Howler;
  // Autoplay is enabled for this isolated browser. Avoid Howler's mobile unlock
  // recreating the context on the first Howl after we attach the recording mix.
  if (h) h.autoUnlock = false;
  h?.volume(); // Howler creates its AudioContext lazily through this public accessor.
  if (!h?.usingWebAudio || !h.ctx || !h.masterGain)
    throw Error('Web Audio / Howler is required for recording');
  h.autoSuspend = false;
  await h.ctx.resume();
  const destination = h.ctx.createMediaStreamDestination();
  const recordingGain = h.ctx.createGain();
  recordingGain.connect(destination);
  let musicTap = null, musicInput = null, cleaned = false;
  const cleanup = () => {
    if (cleaned) return;
    cleaned = true;
    try { musicInput?.disconnect(); } catch (_) {}
    // The page remains open while FFmpeg encodes. Keep its speakers silent.
    try { musicTap?.release({restoreSpeakers: false}); } catch (_) {}
    try { h.masterGain.disconnect(recordingGain); } catch (_) {}
    try { recordingGain.disconnect(); } catch (_) {}
    destination.stream.getTracks().forEach(track => track.stop());
  };
  try {
    h.masterGain.connect(recordingGain);
    h.masterGain.disconnect(h.ctx.destination);
    musicTap = window.AutoDramaMusic?.captureOutput?.() ?? null;
    if (musicTap) {
      // BGM arrives after its own authored/user gain. Mix it directly into the
      // recording destination, so the voice master gain cannot change music.
      musicInput = h.ctx.createMediaStreamSource(musicTap.stream);
      musicInput.connect(recordingGain);
    }
    let fading = false;
    window.fadeRecordingAudio = milliseconds => {
      if (!Number.isInteger(milliseconds) || milliseconds < 0 || milliseconds > 2000)
        throw Error('Invalid recording fade duration');
      if (fading || cleaned) return;
      fading = true;
      // Only the captured mix fades. Reader playback, its loop position and
      // authored/user gains are untouched by this recording-only envelope.
      const param = recordingGain.gain, now = h.ctx.currentTime;
      param.cancelScheduledValues?.(now);
      param.setValueAtTime(param.value, now);
      if (milliseconds && typeof param.linearRampToValueAtTime === 'function')
        param.linearRampToValueAtTime(0, now + milliseconds / 1000);
      else param.setValueAtTime(0, now);
    };
    const recorder = new MediaRecorder(destination.stream, {mimeType: 'audio/webm;codecs=opus'});
    let pending = Promise.resolve(), recordingError = null, finished = null;
    const stopped = new Promise(resolve => { recorder.onstop = resolve; });
    recorder.onerror = event => {
      recordingError = Error(event.error?.message || 'Audio recording failed');
    };
    recorder.ondataavailable = event => {
      if (!event.data.size) return;
      pending = pending.then(async () => {
        const bytes = new Uint8Array(await event.data.arrayBuffer());
        let text = '';
        for (let i = 0; i < bytes.length; i += 8192)
          text += String.fromCharCode(...bytes.subarray(i, i + 8192));
        await window.recordAudioChunk(btoa(text));
      });
      pending.catch(() => {}); // The finish call reports a failed file write.
    };
    window.finishRecordingAudio = () => {
      if (!finished) finished = (async () => {
        try {
          if (recorder.state !== 'inactive') recorder.stop();
          await stopped;
          await pending;
          if (recordingError) throw recordingError;
        } finally { cleanup(); }
      })();
      return finished;
    };
    recorder.start(1000);
  } catch (error) { cleanup(); throw error; }
}"""


def get_json(url: str) -> dict:
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def player_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("鑑賞URLには http:// または https:// のURLを指定してください。")
    if parsed.query or parsed.fragment:
        raise ValueError("鑑賞URLにクエリやフラグメントは指定できません。")
    return value


def resolve_url(args) -> str:
    if args.url:
        return player_url(args.url)
    base = player_url(args.base_url).rstrip("/")
    project = get_json(base + "/api/m3/projects/" + quote(args.project_id, safe=""))
    production = project.get("production") or {}
    path = production.get("player_url")
    if not path:
        raise ValueError("この作品には鑑賞できる公開章がありません。")
    return urljoin(base, path)


def successor(result: dict, build_id: str, chapter_number: int) -> str | None:
    if result.get("build_id") != build_id:
        raise ValueError("次章の作品識別情報が一致しません。")
    status = result.get("status")
    if status in {"complete", "waiting"}:
        return None
    next_build = result.get("next_build") or {}
    identifier = next_build.get("id", "")
    if (status != "ready" or not identifier or identifier == build_id
            or not all(c.isascii() and (c.isalnum() or c in "_-") for c in identifier)
            or next_build.get("chapter_number") != chapter_number + 1):
        raise ValueError("次章の識別情報または章順序が不正です。")
    return identifier


def encode(command: list[str], log: Path) -> None:
    with log.open("ab") as stream:
        subprocess.run(command, stdout=stream, stderr=stream, check=True)


def record_chapter(browser, url: str, directory: Path, args, progress) -> Path:
    directory.mkdir()
    context = browser.new_context(viewport={"width": args.width, "height": args.height},
                                  device_scale_factor=1)
    # Each chapter opens in a fresh browser profile, so seed the settings before any page script.
    context.add_init_script(settings_script(args))
    page = context.new_page()
    encoder = None
    try:
        page.goto(url, wait_until="load", timeout=60000)
        page.wait_for_function("document.getElementById('adn-begin')?.disabled === false", timeout=30000)
        page.locator("#adn-begin").wait_for()
        progress(phase="loading", chapter_recorded_seconds=0, **page.evaluate(PLAYER_PROGRESS))
        page.add_style_tag(content=RECORDING_STYLE)
        with (directory / "audio.webm").open("wb") as audio, (directory / "ffmpeg.log").open("wb") as log:
            page.expose_function("recordAudioChunk", lambda data: audio.write(base64.b64decode(data)))
            # Start before clicking play so the very first utterance is included.
            first = page.screenshot(type="jpeg", quality=90)
            encoder = subprocess.Popen([
                args.ffmpeg, "-nostdin", "-n", "-loglevel", "warning", "-f", "image2pipe",
                "-vcodec", "mjpeg", "-framerate", str(args.fps), "-i", "pipe:0",
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", str(directory / "video.mp4"),
            ], stdin=subprocess.PIPE, stdout=log, stderr=log)
            page.evaluate(AUDIO_START)
            started = time.monotonic()
            frames = 0
            previous = first
            reason = None
            stopped = False
            fade_deadline = None
            last_report = started
            progress(phase="recording", chapter_recorded_seconds=0)
            try:
                page.locator("#adn-begin").click()
                # Use the same handler as the visible Auto button, without exposing the menu.
                page.wait_for_function("window.AutoDramaPlayer.state().started")
                page.locator("#adn-m-auto").dispatch_event("click")
                while True:
                    elapsed = time.monotonic() - started
                    if fade_deadline is None:
                        if (args.output_dir / "stop.request").exists():
                            reason, stopped = "停止要求を受け付けました。", True
                        elif elapsed >= args.max_seconds:
                            reason = "章の最大録画時間に達しました。"
                        ended = page.locator("#adn-end").evaluate("element => !element.hidden")
                        if reason or ended:
                            page.evaluate("milliseconds => window.fadeRecordingAudio(milliseconds)",
                                          RECORDING_FADE_MILLISECONDS)
                            fade_deadline = time.monotonic() + RECORDING_FADE_MILLISECONDS / 1000
                            progress(phase="recording", chapter_recorded_seconds=frames / args.fps,
                                     finishing=True, fade_tail_seconds=RECORDING_FADE_MILLISECONDS / 1000,
                                     **page.evaluate(PLAYER_PROGRESS))
                    current = page.screenshot(type="jpeg", quality=90)
                    target = max(1, math.ceil((time.monotonic() - started) * args.fps))
                    # Duplicate the preceding frame when capture is slow, preserving audio timing.
                    while frames < target:
                        encoder.stdin.write(current if frames == target - 1 else previous)
                        frames += 1
                    previous = current
                    if fade_deadline is not None or time.monotonic() - last_report >= 1:
                        progress(phase="recording", chapter_recorded_seconds=frames / args.fps,
                                 **page.evaluate(PLAYER_PROGRESS))
                        last_report = time.monotonic()
                    # A looping chapter outro must never keep the recording
                    # alive. Capture this finite tail for audio/video alignment;
                    # explicit stops and the time limit allow the same 400ms.
                    if fade_deadline is not None and time.monotonic() >= fade_deadline:
                        break
                    page.wait_for_timeout(max(1, (frames / args.fps - (time.monotonic() - started)) * 1000))
            finally:
                progress(phase="encoding", chapter_recorded_seconds=frames / args.fps, finishing=False)
                page.evaluate("() => window.finishRecordingAudio()")
                encoder.stdin.close()
                encoder.wait(timeout=120)
            if encoder.returncode:
                raise RuntimeError(f"映像エンコードに失敗しました: {directory / 'ffmpeg.log'}")
        output = directory / ("partial.mp4" if reason else "chapter.mp4")
        progress(phase="muxing_chapter")
        encode([args.ffmpeg, "-nostdin", "-n", "-loglevel", "warning",
                "-i", str(directory / "video.mp4"), "-i", str(directory / "audio.webm"),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                # Chromium may omit audio packets while the Web Audio mix is silent.
                # Materialize those timestamp gaps before AAC encoding; apad alone
                # only pads the end and leaves narration gaps that players can skip.
                "-af", "aresample=async=1:first_pts=0,apad", "-shortest",
                "-movflags", "+faststart", str(output)],
               directory / "ffmpeg.log")
        if stopped:
            raise RecordingStopped(output)
        if reason:
            raise RuntimeError(f"{reason} 途中の録画: {output}")
        if not args.keep_raw:
            (directory / "video.mp4").unlink()
            (directory / "audio.webm").unlink()
        return output
    finally:
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        context.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="作品を画面非表示・音声付きでオート録画します。")
    target = result.add_mutually_exclusive_group(required=True)
    target.add_argument("--project-id", help="作品ID（現在選択中の公開版から開始）")
    target.add_argument("--url", help="鑑賞URL。公開版を固定する場合はこちらを使用")
    result.add_argument("--base-url", default="http://127.0.0.1:8000")
    result.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs" / ("recording-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")))
    result.add_argument("--browser", choices=["msedge", "chrome", "chromium"], default="msedge")
    result.add_argument("--ffmpeg", default="ffmpeg")
    result.add_argument("--fps", type=int, default=15)
    result.add_argument("--width", type=int, default=1280)
    result.add_argument("--height", type=int, default=720)
    result.add_argument("--max-seconds", type=float, default=7200, help="1章あたりの上限秒数")
    result.add_argument("--single-chapter", action="store_true")
    view = result.add_argument_group("鑑賞設定", "録画する画面の表示と音量。鑑賞画面の「設定」と同じ項目です。")
    view.add_argument("--text-size", choices=list(TEXT_SIZES), default="standard", help="文字の大きさ")
    view.add_argument("--typeface", choices=["gothic", "mincho"], default="gothic", help="書体")
    view.add_argument("--text-speed", type=int, default=62, help="文字の速さ 0〜100（98以上は一括表示）")
    view.add_argument("--auto-wait", type=float, default=1.4, help="オートの待ち時間（秒）0.3〜4.0")
    view.add_argument("--voice-volume", type=int, default=100, help="音声の音量 0〜100")
    view.add_argument("--bgm-volume", type=int, default=100, help="BGMの音量 0〜100")
    view.add_argument("--window-opacity", type=int, default=80, help="ウィンドウの濃さ 30〜100")
    result.add_argument("--keep-raw", action="store_true", help="合成前の映像・音声も保持")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if (not 1 <= args.fps <= 60 or min(args.width, args.height) < 2
            or args.width % 2 or args.height % 2
            or not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        raise SystemExit("fpsは1〜60、幅・高さは正の偶数、上限秒数は正の有限値にしてください。")
    if (not 0 <= args.text_speed <= 100 or not 0 <= args.voice_volume <= 100
            or not 0 <= args.bgm_volume <= 100 or not 30 <= args.window_opacity <= 100
            or not math.isfinite(args.auto_wait) or not .3 <= args.auto_wait <= 4):
        raise SystemExit("文字の速さと音量は0〜100、ウィンドウの濃さは30〜100、"
                         "オートの待ち時間は0.3〜4.0秒にしてください。")
    if not shutil.which(args.ffmpeg):
        raise SystemExit("FFmpegが見つかりません。--ffmpeg に実行ファイルを指定してください。")
    from playwright.sync_api import sync_playwright

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run_started = time.monotonic()
    state = {"status": "running", "phase": "starting", "chapters": [],
             "output_dir": str(args.output_dir), "recorded_seconds": 0,
             "viewing_settings": viewing_settings(args),
             "started_at": datetime.now(UTC).isoformat()}

    def save(**updates):
        state.update(updates)
        state["elapsed_seconds"] = time.monotonic() - run_started
        state["updated_at_epoch"] = time.time()
        (args.output_dir / "status.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    try:
        url = resolve_url(args)
        seen = set()
        stopped = False
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                channel=args.browser, headless=True,
                args=["--autoplay-policy=no-user-gesture-required",
                      "--disable-background-timer-throttling", "--disable-renderer-backgrounding"])
            try:
                while url:
                    if url in seen:
                        raise ValueError("章の循環参照を検出しました。")
                    seen.add(url)
                    index = len(state["chapters"]) + 1
                    save(current_url=url, phase="loading", current_chapter=index,
                         chapter_recorded_seconds=0, utterance_current=0, utterance_total=None)
                    recorded_before = state["recorded_seconds"]

                    def chapter_progress(recorded_before=recorded_before, **updates):
                        if "chapter_recorded_seconds" in updates:
                            updates["recorded_seconds"] = recorded_before + updates["chapter_recorded_seconds"]
                        save(**updates)

                    if (args.output_dir / "stop.request").exists():
                        stopped = True  # Asked between chapters: nothing of this one is recorded.
                        break
                    print(f"録画中: {url}", flush=True)
                    try:
                        path = record_chapter(browser, url, args.output_dir / f"chapter-{index:03d}",
                                              args, chapter_progress)
                    except RecordingStopped as stop:
                        path, stopped = stop.path, True
                    state["chapters"].append({"url": url, "file": str(path),
                                              "recorded_seconds": state["chapter_recorded_seconds"],
                                              "utterances": state["utterance_total"]})
                    if stopped:
                        break
                    save(phase="checking_next")
                    if args.single_chapter:
                        state["end_reason"] = "single_chapter"
                        break
                    identity = get_json(urljoin(url, "player-context.json"))
                    if identity.get("mode") != "live":
                        state["end_reason"] = "static"
                        break
                    result = get_json(urljoin(url, identity["next_url"]))
                    next_id = successor(result, identity["build_id"], identity["chapter_number"])
                    if not next_id:
                        state["end_reason"] = result["status"]
                        break
                    url = urljoin(url, "/player/" + next_id + "/")
            finally:
                browser.close()
        if stopped:
            state["end_reason"] = "stopped"
            if not state["chapters"]:
                state.update(status="stopped", phase="stopped", output=None)
                print("録画を停止しました。保存した映像はありません。", flush=True)
                return 0
        listing = args.output_dir / "chapters.txt"
        # A stopped run ends with the current chapter's partial file; join it like the rest.
        listing.write_text("".join(
            f"file '{Path(chapter['file']).relative_to(args.output_dir).as_posix()}'\n"
            for chapter in state["chapters"]), encoding="utf-8")
        output = args.output_dir / "recording.mp4"
        save(phase="joining")
        encode([args.ffmpeg, "-nostdin", "-n", "-loglevel", "warning", "-f", "concat",
                "-safe", "1", "-i", str(listing), "-c", "copy", "-movflags", "+faststart",
                str(output)], args.output_dir / "ffmpeg.log")
        final = "stopped" if stopped else "completed"
        state.update(status=final, phase=final, output=str(output))
        print(("停止しました。ここまでを保存しました: " if stopped else "保存しました: ") + str(output), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 -- persist any browser/encoder failure for background runs
        state.update(status="failed", phase="failed", error=str(exc))
        print(str(exc), file=sys.stderr, flush=True)
        return 1
    finally:
        save()


if __name__ == "__main__":
    raise SystemExit(main())
