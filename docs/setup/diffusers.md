# Diffusers / Anima の独立環境

画像用Python、依存パッケージ、公式Gitソースはすべて `services/worker/runtimes/diffusers/` に置きます。制御サーバー、Irodori-TTS、PC内の既存環境には依存しません。ワーカーへ含める対象です。現状のロックは Windows x64 向けです。

## 固定構成

| 項目 | 採用値 |
| --- | --- |
| Python | uv管理 CPython 3.12.13 |
| Diffusers | 公式main取得時の最新commit `7263f3317f6b392d62f41e9d75ed9d7e21fc5a5c` / `0.41.0.dev0` |
| PyTorch | `2.10.0+cu128` |
| torchvision | `0.25.0+cu128` |
| Transformers | `5.17.0` |
| Accelerate | `1.15.0` |
| 背景除去候補 | `rembg[cpu]==2.0.85` |

Diffusersは公式Gitから新規cloneしたソースを `source/` に保持します。Pythonパッケージの導入も同じGit commitを指定しています。`uv.lock` は間接依存も固定します。通常の再実行で最新版へ追従しません。

採用commitのAnima APIは `AnimaModularPipeline` / `ModularPipeline` と `AnimaAutoBlocks` です。`AnimaPipeline` というクラス名を前提に実装しないでください。Qwen3エンコーダー、Qwen Image VAE、AnimaTextConditionerを使う構成です。

## 再構築と診断

プロジェクトルートのPowerShellから実行します。`uv` と `git`、対応するNVIDIAドライバーが必要です。Python自体もワーカー配下へ導入するため、既存のPython環境をactivateする必要はありません。

```powershell
.\runtimes\installers\setup-diffusers.ps1
```

GPUがないPCで依存とimportだけを確認する場合は `-SkipGpuCheck` を指定します。NVIDIA CUDA Toolkitのシステムインストールには依存せず、PyTorchのCUDA 12.8 wheelを利用します。

診断だけの実行:

```powershell
.\services\worker\runtimes\diffusers\.venv\Scripts\python.exe -I .\services\worker\runtimes\diffusers\verify_environment.py
```

診断はモデルをダウンロードせず、次を確認します。

- runtime内のPythonと固定Git commitの一致
- Anima API、Transformers、背景除去ライブラリのimport
- 公式 `source/scripts/convert_anima_to_diffusers.py --help` の正常終了
- ONNX RuntimeのCPUExecutionProvider
- CUDA上のbfloat16行列演算とGPU情報

背景除去のライブラリは候補として導入しました。アニメ立ち絵の髪や小物に対する品質、採用する背景除去モデルは未選定です。Anima、テキストエンコーダー、VAE、トークナイザー、背景除去モデルの重みは今回導入していません。画像生成、背景除去の実推論、実チェックポイント変換の検証はM0で行います。

## ローカルファイル

| 場所 | 内容 |
| --- | --- |
| `.python/` | このruntime専用のuv管理Python |
| `.venv/` | このruntime専用のPython依存 |
| `source/` | 公式Diffusers cloneと変換スクリプト |
| `cache/huggingface/` | 将来のモデル取得キャッシュ |
| `cache/numba/` | 背景除去関連のローカルコンパイルキャッシュ |
| `models/rembg/` | 将来の背景除去モデル |
| `pyproject.toml`, `uv.lock`, `.python-version` | Git管理する再現情報 |

uvのパッケージ取得キャッシュだけはプロジェクト内の `.cache/uv/` を共用し、PC既存のuvキャッシュは使用しません。各venvへはwheelから通常インストールします。既存環境のコピーは行いません。

ワーカーが推論サブプロセスを起動するときは、上記ローカル保存先を `HF_HOME`、`U2NET_HOME`、`NUMBA_CACHE_DIR` へ指定してください。診断スクリプトはこの設定とオフライン動作を自身で行います。セットアップ用PowerShellは一時的な環境変数を終了時に戻します。

venvとuv管理Pythonは別PCへそのまま移動する形式ではありません。別PCのワーカーでは、このmanifest・lock・インストーラーから再構築してください。

## 参照

- [Diffusers公式ソース（固定commit）](https://github.com/huggingface/diffusers/tree/7263f3317f6b392d62f41e9d75ed9d7e21fc5a5c)
- [採用commitのAnimaドキュメント](https://github.com/huggingface/diffusers/blob/7263f3317f6b392d62f41e9d75ed9d7e21fc5a5c/docs/source/en/api/pipelines/anima.md)
- [採用commitのAnima変換スクリプト](https://github.com/huggingface/diffusers/blob/7263f3317f6b392d62f41e9d75ed9d7e21fc5a5c/scripts/convert_anima_to_diffusers.py)
- [PyTorch公式バージョン別インストール手順](https://pytorch.org/get-started/previous-versions/)
- [rembg公式リポジトリ](https://github.com/danielgatis/rembg)
