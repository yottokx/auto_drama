# AIオートドラマ 開発ガイド

[実装計画](implementation-plan.md)に基づき、M0〜M4を実装しています。ローカルモデルで設定・透過立ち絵・基準音声を生成し、修正・固定・リテイクと初期承認を行えます。STEP4で全体プロットと予定サブキャラを確認・直接編集・AI修正し、構成の承認後にSTEP5で指定した全章を順番に制作します。完成した章から鑑賞・書き出しできます。詳しくは [STEP4の使い方](docs/setup/planning.md)、[M4の使い方](docs/setup/m4.md)、[M4の設計](docs/architecture/m4.md)、[M3の使い方](docs/setup/m3.md)、[M2の使い方](docs/setup/m2.md)、[M1の使い方と検証](docs/setup/m1.md) を参照してください。

2026-09-24にM4の複数章・状態引き継ぎ・人物別知識と伏線の検査・章単位公開・停止/中断/再開・次章待機と閲覧保存を実装しました。旧M3の公開済み第1章は自動継続せず、明示的な制作再開で後続章へ進みます。模擬ワーカーと固定データによる検証を追加していますが、M4の実モデルによるGPU生成・複数章の読解と試聴は未実施です。

2026-09-22のユーザー判断でM2は一旦完了です。生成結果の多様性、メインキャラ向け発案の強化、用途別LLMパラメータ・reasoningと処理時間の調整は継続課題です。サブキャラには重い候補探索を一律適用しません。次の作業では [M2完了時の引継ぎ](docs/setup/m2.md#m2完了時の引継ぎと継続課題) を確認してください。

## 配置

メインキャラは最大3人です。自己紹介・代表台詞と全組の関係性を生成し、自己紹介の基準音声から自由入力の台詞をボイスクローンで試せます。人物の生成中も完成済みキャラを閲覧でき、立ち絵はクリックで拡大できます。

WebのルートURLはM2の設定・承認とM4の制作・鑑賞入口です。作品・設定・生成結果・承認は制御サーバーへ保存されます。外観だけのサンプルは `http://127.0.0.1:5173/?demo=results`、既存M1機能は `http://127.0.0.1:5173/?view=m1` で利用できます。

| 用途 | 配置 |
|---|---|
| 制御サーバーのuv環境 | `.venv/`（Python本体は `.python/`） |
| React / TypeScript / Vite | `apps/web/` |
| ワーカー専用uv環境 | `services/worker/.venv/` |
| Irodori-TTS専用uv環境・公式clone | `services/worker/runtimes/irodori/` |
| Diffusers専用uv環境・公式clone | `services/worker/runtimes/diffusers/` |
| llama.cpp公式clone・実行ファイル | `services/worker/runtimes/llama_cpp/` |
| ティラノスクリプト公式clone | `tyranoscript/`（全体をGit対象外） |
| バージョン・取得情報 | `runtimes/manifests/` |
| 環境設定の雛形 | `config/environment.example.toml` |

既存のPC内のIrodori-TTS・llama.cpp環境やモデルはコピーしません。各公式Gitリポジトリの取得時点の最新コミットを記録して利用します。Python仮想環境も独立しており、インストール済みの別環境を参照しません。

## 環境を再構築する

Windows、PowerShell 7、Git、uv、Node.js（20.19以上または22.12以上）、対応するNVIDIAドライバーを前提とします。プロジェクトルートのPowerShell 7（pwsh）で実行します。

```powershell
.\scripts\setup.ps1
.\scripts\check-environment.ps1
```

必要な部分だけ実行する場合:

```powershell
.\scripts\setup.ps1 -Components core,worker,web
.\scripts\setup.ps1 -Components irodori,llama,diffusers,tyrano
```

依存は各 `uv.lock`、`package-lock.json` とランタイムのコミット情報で固定します。再実行は同じ版を再現し、自動で上流の最新版へ更新しません。uvのダウンロードキャッシュは `.cache/uv/` に置きます。これはインストール時のキャッシュで、各仮想環境の共有ではありません。

## 開発時の起動

別々のPowerShellで起動します。

```powershell
.\scripts\start-coordinator.ps1
.\scripts\start-worker.ps1
.\scripts\start-web.ps1
```

- UI: http://127.0.0.1:5173
- API疎通: http://127.0.0.1:8000/api/health
- API仕様: http://127.0.0.1:8000/docs

UIで「新しい物語」を作成し、世界観の生成・確定からキャラクターの生成へ進みます。ワーカーが起動していれば設定・画像・音声の生成が進みます。M1画面の「サンプルを作成して変換」は引き続き固定脚本からのZIP出力に使えます。保存先の既定値は `data/` です。変更する場合は `start-coordinator.ps1 -DataDir <保存先>` を指定します。ワーカーは同じAPIからデータを取得し、DBを直接開きません。

過去のSTEPへ戻り、同じ確定ボタンを押して後続工程をやり直せます。世界観の再確定はメインキャラ以降（固定した項目は維持）、メインキャラの再確定は全体計画以降、全体計画の再確定は本編を新しく制作します。再確定前の成果物は削除せず「変更履歴・元に戻す」から復元でき、作品ごとのCG上限設定も戻ります。復元だけでは生成を自動再開しません。生成が動作中の場合は完了を待つか、本編制作画面で停止・中断してから再確定します。

ティラノ公式サンプルの配信:

```powershell
.\scripts\start-tyrano.ps1
```

http://127.0.0.1:8080 をブラウザーで開きます。サーバーの終了は各ターミナルの `Ctrl+C` です。

## モデルと次の工程

M0用のモデル取得情報は `config/m0-models-tts.json`、`config/m0-models-image.json`、`config/m0-llm.json` に固定しています。配置済みのGemma GGUFは `private/model_download/` から直接参照し、画像・音声モデルはワーカー配下へ取得済みです。CLIヘルプやパッケージのimportによる確認と、実際のモデル生成の確認は区別します。

- [Irodori-TTSの環境](docs/setup/irodori.md)
- [画像生成の環境](docs/setup/diffusers.md)
- [Animaによる立ち絵差分の調査と設計方針](docs/research/anima-portrait-variants-20261004.md)（差分生成・自動マスクは未実装）
- [Qwen-Image-2.1によるイベントCGの実装計画](docs/qwen-event-cg-plan.md)、[イベントCGの利用と検証](docs/setup/event-cg.md)（任意機能・既定で無効）
- [llama.cpp・ティラノの環境](docs/setup/native-runtimes.md)

ワーカー用のランタイムはすべて `services/worker/` 配下です。別PCではこの配下と依存・ランタイムの構築手順から再構築します。Windowsのvenvには絶対パスが入るため、フォルダー移動後はそのPCでuv環境を再作成します。ワーカー配布のパッケージ化は今後実装します。

`tyranoscript/` はルート直下の独立したGitリポジトリで、このプロジェクトのGit追跡・ワーカー配布には含めません。M0では生成した素材と本文から検証用の一場面を組み立て、指定したティラノ環境を参照して配信します。M1のZIPも本体を含まず、固定脚本と素材だけを書き出します。M3・M4の公開章は制御サーバーの `/player/<build ID>/` から鑑賞できます。ユーザー指定のティラノ本体を読み取り専用で参照し、作品ZIPには同梱しません。M4のまとめ書き出しAPIは章別ZIPと順序・終端のmanifestを返し、静的プレイヤーの自動連結は行いません。

## M0の接続検証

検証結果と既知の制約は [M0 接続検証](docs/setup/m0.md) を参照してください。固定モデル32ファイルの取得・整合性検査、LLMの本文・JSON・乱数Tool・推論無効化、Animaの画像生成・背景除去・単体モデル変換と再ロード、Irodoriの声デザインと3感情のクローン生成を確認済みです。画像→TTS→LLMの切り替え後にGPUメモリが戻ることも確認しました。ティラノでの一場面の再生とユーザーの試聴結果を受け、M0は既知制約ありで完了です。話し方の差は出るものの、今回の本文では喜びの感情は再現できていません。感情は台詞・場面に合わせて指定する方針とし、声質一致や読みの個別の品質評価は今後も残します。

モデル約14.96GBの取得・整合性検査と、配置済みGGUFのSHA256照合・最初のローカルLLM検証の記録は `private/m0/latest.json` から確認できます。画像・音声・変換の検証記録は別の `private/m0/latest-generation.json` から確認します。進捗は各実行先の `progress.log`、工程別の結果とGPUメモリ記録は `status.json`、詳細は各工程のログに保存します。監視用のWebサーバーは使用しません。

生成検証を再実行する場合は、モデル取得・LLM検証が成功した記録を入力にして、新しい実行ディレクトリを作成します。既存の実行結果は上書きしません。次のコマンドは前面で動作し、画像・変換済みモデル・音声を新しく生成します。

```powershell
$downloadRun = Get-Content private/m0/latest.json -Raw | ConvertFrom-Json
$generationDir = Join-Path $PWD.Path ('private/m0/generation-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $generationDir | Out-Null
@{ run_dir = $generationDir; download_run = $downloadRun.run_dir } |
    ConvertTo-Json | Set-Content private/m0/latest-generation.json -Encoding utf8
& .\services\worker\.venv\Scripts\python.exe -I -u -X utf8 `
    .\scripts\m0\run_generation_checks.py --run-dir $generationDir --download-run $downloadRun.run_dir `
    2>&1 | Tee-Object -FilePath (Join-Path $generationDir 'progress.log')
```

別のPowerShellで生成検証のログを確認する場合:

```powershell
$run = Get-Content private/m0/latest-generation.json -Raw | ConvertFrom-Json
Get-Content -LiteralPath (Join-Path $run.run_dir 'progress.log') -Tail 20 -Wait
```

別のPowerShellで開いたログ表示だけを終了するにはCtrl+Cを押します。生成処理は継続します。生成処理を中止したい場合は、対象の実行ディレクトリに `stop.request` という空ファイルを作成します。モデル取得を中断した場合の途中ファイルは `.part` として保持し、取得処理の次回実行で再開します。

実機検証に使用するスクリプト:

- `scripts/m0/image_smoke.py`: Anima画像生成と背景除去
- `scripts/m0/convert_anima_smoke.py`: 公式単体チェックポイントの変換・再ロード
- `scripts/m0/tts_smoke.py`: 声デザイン→クローン、感情入力、WAV検査
- `scripts/m0/llm_smoke.py`: 本文・JSON・乱数Tool・推論無効化
- `scripts/m0/build_tyrano_scene.py`: 生成本文・画像・音声から検証用の一場面を組み立て
- `scripts/m0/serve_scene.py`: 指定したティラノ環境と生成した一場面を配信

`run_generation_checks.py` が生成・変換・組み立ての各スクリプトを対象ランタイムの `.venv/Scripts/python.exe` で順番に起動します。モデルの同時GPUロードを避け、画像→TTS→LLMの順でプロセス終了後の解放を確認します。配信とブラウザーでの確認は別に実施します。生成検証の終了状態 `generation_completed_review_pending` はM0全体の完了を意味しません。目視・試聴・実再生の結果と、許容した制約や残る品質評価を記録して最終判定します。今回の記録は `completed_with_known_limitations` です。外部LLM APIの接続検証はユーザー指定により後回しです。

## LLM共通設定

[モデル・生成パラメータの設定とWorkerのモデル配置](docs/setup/llm-settings.md)を参照してください。
