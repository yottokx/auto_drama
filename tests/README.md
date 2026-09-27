# 検証

M1・M2・M3の軽量テストはGPUや実モデルを起動しません。リポジトリルートから実行します。

フロントエンドの3段階の状態遷移・旧保存値の移行・未確定への戻しは `npm --prefix apps/web test` で検査します。既存のViteを使ってTypeScriptをメモリ内で変換するため、新たなテストランタイムは不要です。`integration/test_m2_combined_brief.py` は世界観・人物・関係性の一括保存、世界観確定後の自動生成、重複防止、変更時の再確認と既存承認版の保持を検査します。

```powershell
& .\.venv\Scripts\python.exe -m pytest -q --basetemp (Join-Path 'tmp' ('pytest-' + [guid]::NewGuid().ToString('N')))
```

一時ディレクトリをリポジトリ内に限定し、実行ごとに新しい名前を付けます。これによりWindowsで既存のOS一時ディレクトリのACLに影響されません。テスト結果は `tmp/` に残り、Git対象外です。

- `unit/test_contracts_export.py`: schema・参照・パス・本文のエスケープ・ZIP再現性。Node.jsとローカルティラノがある場合はネイティブのパーサー・タグ処理も検証し、ない環境ではその検査だけskipします。
- `unit/test_worker.py`: HTTPワーカーの取得・ハッシュ検査・heartbeat・通信失敗・古いリース・完了の再送。
- `integration/test_coordinator.py`: マイグレーション、再起動、ワーカー実行、依存関係、同時取得、途中ファイル、試行上限、遅延・重複結果、成果物の破損。
- `integration/test_m2.py`: ウィザード全体、入力と結果の分離、固定と部分修正、素材リテイク、承認版の不変性、再起動、競合、失敗・再試行、不正な生成物の拒否。最大3人、全組の関係性と再生成、自己紹介と音声の整合性、旧データ互換、クローン音声の参照元検査も含む。
- `unit/test_m2_generation.py`: 階層候補・採用、4種の乱数Tool、引数と回数制限、部分修正、完全性検証、同ジョブのキャッシュ、GPUプロセスの実行順と終了。実候補ID・関係性IDへの出力制約とIrodoriの参照音声によるクローンも含む。
- `unit/test_m2_worker.py`: M2能力申告、ジョブの振り分け、持続キャッシュ、完了ZIPの再送、リース失効、検証拒否時の失敗処理。クローン参照音声の取得・再利用、ID・ハッシュ・PCM WAVの検証も含む。
- `unit/test_voice_process_session.py` / `unit/test_voice_runner_session.py`: 連続音声のプロセス・モデル再利用、参照音声の分離、設定変更・エラー・タイムアウト・終了時の解放。GPUを使わない模擬プロセスとランタイムで検証する。
- `unit/test_image_runner_prompts.py`: 画像ランナーのモード別プロンプト、明示的なネガティブ上書きと空文字による無効化。M2からの品質タグ・ネガティブ伝播、生成記録と再試行時の設定保持は `unit/test_m2_generation.py` で検証する。
- `unit/test_m3_narrative.py`: 原文と構造化結果の完全一致、重要イベントの根拠、長文の継続、段階別リトライ、章数を維持した分割生成、素材生成の入力と再利用。
- `unit/test_m3_source_repair.py`: 未知の話者・人物IDの誤記の修正、正常な本文の保持、文法制約付きの継続、再試行回数と失敗位置の保持。
- `unit/test_m3_player.py`: プレイヤー用ZIPと配信の境界、原文表示、既読バックログ、保存・復元。ローカルのティラノを使い、オート停止時のイベント伝播も検証する。
- `unit/test_portrait_layout.py`: 立ち絵の縦横比、透過余白・ノイズ、上半身／全身の構図、人物ごとの明示指定と配置移動。
- `integration/test_m3_rebuild.py`: 採用本文・素材を再生成しない表示更新、旧版の保持、同じ出力の重複防止と失敗時のロールバック。
- `integration/test_m3.py`: 承認から公開まで、凍結した設定の保持、不正素材・古いリースの拒否、再起動時の復旧、完成素材を保った組み立て再試行。
- `integration/test_m3_review.py`: 実際のワーカークライアントを介した模擬生成、応答喪失後の再送、下書き変更後の承認素材再利用、ZIP内の原文照合。
- `fixtures/m1-script.json`: 版付き固定脚本。対応素材は `packages.tyrano_export.demo_content()` で再現できます。

環境構築は `scripts/check-environment.ps1`、実モデルによる検証は [M0](../docs/setup/m0.md) と [M2](../docs/setup/m2.md)、M1のブラウザー確認は [M1](../docs/setup/m1.md) を参照してください。

`scripts/m2/check_cast.py --output-dir <保存先>` は実モデルで3人分の設定・立ち絵・自己紹介音声、全3組の関係性、承認、任意台詞のボイスクローンを検証します。保存先にはJSON・PNG・WAVを残します。GPUを使用し、新規の検証作品を作成するため軽量テストには含めません。

M3の実モデル確認は [M3の手順](../docs/setup/m3.md) と `scripts/m3/check_local.py` を参照してください。原文と発話の完全一致、承認からの自動開始、素材不足・破損時の非公開、再起動・リース・再送、プレイヤーの経路制限を軽量テストで確認します。
