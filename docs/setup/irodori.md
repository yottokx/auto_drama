# Irodori-TTS の独立環境

公式の [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) をワーカー配下へ `git clone` しています。2026-09-22 の導入時の main HEAD `89f9d8fbd4d51ea019867ee1197725ede1df13c5` と公式 `uv.lock` を固定し、通常のセットアップ再実行で自動更新しません。既存 PC の Irodori 本体、仮想環境、モデルはコピー・共有していません。

## セットアップ

プロジェクトルートで実行します。Windows x64、Git、uv、Windows 標準 tar、CUDA 12.8 対応 NVIDIA ドライバーが必要です。

```powershell
powershell -ExecutionPolicy Bypass -File .\runtimes\installers\setup-irodori.ps1
```

Python 3.11.14 は uv で runtime 内へ導入します。公式 lock の `cu128` extra を使用し、PyTorch 2.10.0+cu128 をインストールします。依存のインストールは `UV_LINK_MODE=copy` により、uv cache を消しても独立した環境として残ります。グローバル Python shim と既存環境は変更しません。

| 内容 | プロジェクト内の配置 |
| --- | --- |
| ワーカー用ランタイム | `services/worker/runtimes/irodori/` |
| 公式 Git clone | `services/worker/runtimes/irodori/source/` |
| uv 管理 Python | `services/worker/runtimes/irodori/.python/` |
| Python 仮想環境 | `services/worker/runtimes/irodori/.venv/` |
| FFmpeg shared | `services/worker/runtimes/irodori/ffmpeg/` |
| 今後のモデル・生成 cache | `services/worker/runtimes/irodori/cache/` |
| ダウンロード一時領域 | `services/worker/runtimes/irodori/downloads/` |
| 依存のダウンロード cache | `.cache/uv/` |
| 固定した取得元・版 | `runtimes/manifests/irodori.json` |

TorchCodec の音声読み込みには FFmpeg の共有 DLL が必要です。[FFmpeg 公式が掲載している Windows 配布元](https://ffmpeg.org/download.html)の [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) から 8.1.2 full shared を新規取得し、公開 SHA256 と照合して runtime 内に置いています。公式 lock の TorchCodec 0.10 が対応する FFmpeg は最大 major 8 のため、この版を使用します。ランチャーはこのプロセスの PATH だけに DLL の場所を追加します。

## 起動・診断

```powershell
# モデルを読み込まない診断
& .\services\worker\runtimes\irodori\run.ps1 -EntryPoint verify

# 推論 CLI の引数確認
& .\services\worker\runtimes\irodori\run.ps1 -EntryPoint infer.py -ScriptArguments '--help'
```

診断は Irodori と依存ライブラリの import、CLI、CUDA 行列演算、参照音声の WAV デコードを確認し、runtime の `.runtime-state.json` に結果を保存します。Python 本体・依存ライブラリがこの runtime 内から読まれていることも検査します。

ランチャーの `EntryPoint` は `infer.py`、`gradio_app.py`、`gradio_app_voicedesign.py`、`verify` です。HF_HOME/HF_HUB_CACHE、TORCH_HOME、Gradio の一時領域を runtime 内に閉じ、既定で Hugging Face のオンライン取得を無効にします。モデル導入を明示的に行う場合のみ `-AllowModelDownload` を付けてください。これはモデルを固定・登録する M0 検証時に行います。

**今回モデル重みは未導入です。** 音声の実生成、声デザイン、クローン品質、モデル切替時の GPU 解放は M0 の別検証です。source のコードや仮想環境、モデルを Git の本体リポジトリへ追加せず、インストーラ・manifest・ランチャー・診断を版管理します。別 PC では同じインストーラで再構築してください。仮想環境の絶対パスを含むため、既存 `.venv` を別の絶対パスへ移動する運用は避けます。
