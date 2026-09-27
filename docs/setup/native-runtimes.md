# llama.cpp とティラノスクリプト

2026-09-22 に公式 Git リポジトリの最新 `master` を新規 clone した。PC に既存の環境からのコピー、リンク、Git サブモジュールは使用していない。取得コミットは `runtimes/manifests/` に固定し、通常のセットアップ再実行では最新版に追従しない。

## 配置

| 用途 | プロジェクトルートからのパス | 固定情報 |
|---|---|---|
| llama.cpp 独立 Git clone | `services/worker/runtimes/llama_cpp/source/` | `58367713a6935c0810103378144008df32e3d5db` |
| llama.cpp 実行ファイル・CUDA DLL | `services/worker/runtimes/llama_cpp/bin/` | 公式 `b11093` / `fb34fc262c1b43f1832c7472429fb2247d650493`、Windows x64 CUDA 13.4 |
| llama.cpp ライセンス | `services/worker/runtimes/llama_cpp/licenses/`、`source/LICENSE`、`bin/LICENSE-LLVM-OpenMP` | MIT / 同梱ランタイムの原文を保持 |
| ティラノスクリプト独立 Git clone | `tyranoscript/` | `c8dbfd492afd3d79b0954fcf4477236f5c6c4830` |
| ティラノの起点・利用規約 | `tyranoscript/index.html`、`tyranoscript/LICENCE.txt` | 公式 clone の原文を保持 |

llama.cpp のソースは最新 `master` の clone、実行版は取得時の公式最新リリースを使用しており、**両者のコミットは異なる**。取得した `master` コミット向けの公式 Release ワークフローはセットアップ時点で pending だった。ローカルに CUDA Toolkit がないためソースからの CUDA ビルドは行わず、公式 CUDA 配布を補助の実行環境として採用した。NVIDIA ドライバーやシステムの PATH は変更していない。

ワーカーの配布範囲は `services/worker/` 配下で完結する構成。llama.cpp を移送する場合は実行ファイルだけを抜き出さず `bin/` の DLL とライセンスを保持する。`.downloads/` は取得用キャッシュで、実行に不要。ティラノはルート直下の独立フォルダーで、ワーカーには含めず、プロジェクトの Git からも除外する。ティラノ自身の `.git/` は独立 clone の管理用に保持する。

## セットアップと確認

PowerShell 7 と Git を使用する。プロジェクトルートで実行する。

```powershell
pwsh -NoProfile -File runtimes/installers/setup-llama.ps1
pwsh -NoProfile -File runtimes/installers/setup-tyrano.ps1
```

初回は公式 Git から clone し、manifest のコミットを取得する。llama.cpp の公式リリース ZIP は manifest の SHA-256 と一致してから展開する。再実行では既存 clone の取得元とコミットを確認し、ソースの強制リセットや自動更新を行わない。取得済み clone を手動更新した場合は不一致を報告して停止する。

llama.cpp の起動確認は以下で行える。

```powershell
& ./services/worker/runtimes/llama_cpp/bin/llama-server.exe --version
& ./services/worker/runtimes/llama_cpp/bin/llama-server.exe --list-devices
```

セットアップ時に以下を実測し、両 installer の再実行も成功した。

- `version: 0.4.1-dev (build 11093, commit fb34fc262)`
- `CUDA0: NVIDIA GeForce RTX 5090 (32606 MiB, ...)`
- NVIDIA ドライバー `591.86` をそのまま利用
- ティラノの `index.html`、エンジン JavaScript、`Config.tjs`、利用規約の存在と取得コミット一致

LLM モデルは導入していない。GPU 列挙と実行ファイル起動までを確認済みで、GGUF を読み込む推論の検証はモデル登録時に行う。ティラノの利用は `scripts/start-tyrano.ps1` でローカル HTTP 配信する。

## 取得元

- [llama.cpp 公式リポジトリ](https://github.com/ggml-org/llama.cpp)
- [固定した llama.cpp ソース](https://github.com/ggml-org/llama.cpp/tree/58367713a6935c0810103378144008df32e3d5db)
- [公式 CUDA 実行版 b11093](https://github.com/ggml-org/llama.cpp/releases/tag/b11093)
- [ティラノスクリプト公式リポジトリ](https://github.com/ShikemokuMK/tyranoscript)
- [固定したティラノソース](https://github.com/ShikemokuMK/tyranoscript/tree/c8dbfd492afd3d79b0954fcf4477236f5c6c4830)
