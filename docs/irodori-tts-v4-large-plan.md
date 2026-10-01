# Irodori-TTS v4 Large 対応とTTS設定の実装計画

日付：2026-09-30

状態：設定UI・モデル取得・音声生成への接続を実装済み。利用者が取得したv4 Large・bf16でボイスデザイン／クローン、Large→Small→Largeの切り替えを実機確認した。fp32／int8／int4の選択・ロード分岐は実装済みで、量子化版の実ウェイト生成検証は未実施。

現在の操作手順は[モデルのダウンロードと設定画面](setup/tts-settings.md)を参照。Small／Largeそれぞれのfp32／bf16／int8／int4を取得できる。用途別設定は取得・検証済みで実行可能な組み合わせだけを保存でき、新規の音声生成に適用する。以下は当初の計画と受け入れ条件を記録したもの。

設定画面をLLMとTTSのタブに分け、Irodori-TTS v4.1 Small／v4 Largeとfp32／bf16／int8／int4を選択できるようにする。ボイスデザインとボイスクローンには、それぞれ独立したモデルと精度を設定する。モデルの取得・検証は実行先Workerが担当し、将来ほかのTTSを追加できる共通インターフェースを設ける。

既存のSmall＋fp32を移行時の初期値にする。設定の保存とダウンロードは別操作にし、保存前に必要なウェイトと対応Workerの準備を確認する。本編は制作開始時のTTS設定を後続章まで引き継ぐ。

## 対応範囲

| 項目 | 今回の対応 |
| --- | --- |
| 設定画面 | サイドバーの入口を「設定」に変更し、LLM／TTSタブを配置 |
| モデル | Irodori-TTS v4.1 Small、v4 Large |
| 精度 | fp32、bf16、int8、int4 |
| 用途 | ボイスデザイン、ボイスクローンのモデル・精度を独立選択 |
| 取得 | 選択したモデルと必要な共通依存のダウンロード、進捗、中断、再開、再試行、整合性検査 |
| 実行 | M2の基準音声・試聴、M3／M4の基準音声・台詞音声 |
| 拡張 | Provider登録と能力情報により、別TTSを追加可能な設計 |

今回の実モデル実装はIrodoriに絞る。別TTSの接続、LoRA、Speaker Inversion、FP8、動的INT8、ユーザーによるローカル量子化は追加機能として扱う。LLMタブは現在の設定項目と保存APIを引き継ぎ、今回のウェイト取得機能はTTSを対象とする。

## 公式仕様と精度の扱い

fp32／bf16は通常ウェイトの計算精度であり、int8／int4は別の量子化済みチェックポイントである。画面は4択とし、内部ではウェイト形式と計算精度を分離する。量子化版の非量子化層と計算にはbf16を使用する。[公式推論・量子化仕様](https://github.com/Aratako/Irodori-TTS#quantization)

| 画面の選択 | 取得するウェイト | 計算精度 |
| --- | --- | --- |
| fp32 | 通常のmodel.safetensors | fp32 |
| bf16 | fp32と共通のmodel.safetensors | bf16 |
| int8 | Quantizedリポジトリのint8-weight-only | bf16 |
| int4 | Quantizedリポジトリのint4-weight-only | bf16 |

通常ウェイトはモデルごとに1回取得し、fp32／bf16で共用する。デザインとクローンが同じウェイトを使う場合も重複保存しない。

Small・Largeとも公式量子化版が公開済みである。Largeの通常版カードには「量子化版は予定」との記載が残っているが、取得可否は実際の量子化リポジトリとファイルを基準にする。INT4はCUDA compute capability 8.0以上を必要とする。[Small量子化版](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-Quantized)、[Large量子化版](https://huggingface.co/Aratako/Irodori-TTS-v4-Large-Quantized)

Largeは約3.29Bで、テキスト・声の説明文のエンコーダーにT5Gemma 2を使用する。通常ウェイトは約13.2GBである。チェックポイントの容量を実行時VRAMの必要量として表示せず、GPU上のピーク使用量は実機検証で測定する。[Largeモデル仕様](https://huggingface.co/Aratako/Irodori-TTS-v4-Large)、[通常ウェイトのファイル一覧](https://huggingface.co/Aratako/Irodori-TTS-v4-Large/tree/main)

現環境の固定済み上流コミット`89f9d8fbd4d51ea019867ee1197725ede1df13c5`にはT5Gemma 2とtorchaoの量子化ロード処理が既にある。まずこのランタイムを検証し、ロード互換性に問題がある場合だけ、理由と検証結果を記録してコミット・lockを更新する。

公式量子化版の検証記載はNVIDIA CUDAである。この計画の調査では、既存Windows環境のPyTorch 2.10.0+cu128／torchao 0.16.0／RTX 5090で、合成したLinear層のINT8／INT4量子化、safetensors用の変換・復元、CUDA実行が成功した。これは実際のIrodoriチェックポイントでの生成確認ではない。実装開始時に実モデル推論を検証し、対応できない形式は理由を表示して選択不可にする。4形式の実機検証が揃うまでは全形式対応完了と扱わない。

## 現在の実装から変更する箇所

| 現在の実装 | 変更内容 |
| --- | --- |
| `apps/web/src/LLMSettings.tsx`がDialog全体を所有 | 共通SettingsDialogとLLM設定内容へ分離し、TTSタブを追加 |
| `config/m2-generation.json`のvoice設定は1つ | 用途別TTS設定への互換変換を用意 |
| `voice_runner.py`がSmallのパス・モデル名・revisionを固定 | 選択したウェイトと依存を解決し、実際に使った情報を記録 |
| `voice_session.py`の常駐プロセス識別はコマンド＋精度 | Provider・モデル・ウェイト・版・計算精度を含めて識別 |
| `tts_smoke.py`のverify_modelsはmanifest全体を検証 | 選択ウェイトとその依存だけを検証 |
| CoordinatorはTTSをジョブ種別だけでWorkerへ割り当て | モデル・精度・用途・版の適合性も判定 |
| TTS設定はWorker実行時にローカルconfigから取得 | Coordinatorのジョブ作成時に設定を固定 |
| M0の取得スクリプトに再開・整合性検査がある | 取得処理を共通化し、Workerの非同期管理処理から利用 |

既存のLLM設定はDB保存、Workerの候補申告、ジョブへの設定固定を実装済みである。この仕組みを参考にするが、TTS情報は既存LLMの`profile`と混在させず、独立した`tts_profile`として保存する。

## 設定画面の構成

TTSタブには「用途別の設定」と「モデルのダウンロード」を配置する。

用途別の設定には、ボイスデザインとボイスクローンそれぞれのTTSエンジン、モデル、精度を表示する。初期のエンジンはIrodori-TTSのみで、モデルはv4.1 Small／v4 Large、精度はfp32／bf16／int8／int4を選択できる。「クローンにも同じ設定を使う」は入力補助として設け、保存データには両用途の設定を明示する。

ダウンロード欄には、取得先Worker、モデル、必要なウェイト、依存込みの取得容量、状態と進捗を表示する。複数Workerの場合は取得先を選択し、1台の場合も対象名を表示する。未取得のモデルも候補として閲覧でき、取得・検証が完了すると用途別の設定へ適用できる。

状態は「未取得」「待機中」「ダウンロード中」「中断」「検証中」「利用可能」「失敗」「このWorkerでは非対応」を区別する。取得完了とGPU上での利用可否は別に扱う。GPU検証待ちはその理由を表示し、画面を閉じても処理は継続する。

LLMとTTSの編集中の値はタブ移動でも保持し、タブごとに保存する。失敗理由と再試行操作を表示し、Worker情報の更新が未保存の入力を上書きしないようにする。設定保存で音声を自動再生成せず、基準音声を変更したい場合は既存のリテイク操作を使う。

タブはtablist／tab／tabpanelと選択状態・関連付けを持ち、矢印キーで移動できるようにする。進捗はprogressbarで示し、読み上げ通知は状態の変化・完了・失敗を中心にする。Dialogのフォーカス復帰と狭い画面での操作も確認する。

## 設定とモデル管理のデータ

公開設定はProvider、モデルID、精度を用途別に保持する。次は設定例であり、初期値は両用途ともSmall＋fp32とする。

```json
{
  "schema_version": 1,
  "voice_design": {
    "provider_id": "irodori",
    "model_id": "irodori-v4-large",
    "precision": "bf16"
  },
  "voice_clone": {
    "provider_id": "irodori",
    "model_id": "irodori-v4.1-small",
    "precision": "int8"
  }
}
```

アプリが配布するカタログには、Provider、モデルID、対応用途、精度からウェイトへの対応、Hugging Faceリポジトリ・固定revision・subfolder、必要ファイルのサイズ・ハッシュ、共通依存、必要ランタイム、ライセンス情報を記録する。推論時に`main`から最新版を自動取得しない。

Workerのローカル登録には実ファイルの場所、取得記録、検証状態を持たせる。Coordinatorへはパスを送らず、ID、版、用途、精度、利用可否を申告する。カタログにある取得可能モデルと、各Workerで利用可能なモデルを別の情報として扱う。

ジョブの`tts_profile`には、Provider・モデルIDに加えて、ウェイトID、固定revision、ウェイトのハッシュ、量子化方式、計算精度、ランタイム版、codec等の依存ID、実際の生成パラメータを固定する。カタログ更新後も、再試行時に同じIDが別ウェイトへ解決されないようにする。

## Workerでのダウンロード

取得はCoordinatorから対象Workerへ管理要求を送り、Workerのモデル配置先に保存する。通常のHTTPリクエスト内で大きなウェイトを取得せず、操作IDを返して進捗を取得する。管理要求には永続状態とWorkerのleaseを持たせ、Worker再起動後の再開と二重処理防止を行う。

`scripts/m0/run_background.py`の取得処理から、Rangeによる再開、リトライ、空き容量確認、サイズ・SHA256照合、`.part`から検証後の確定を共通モジュールへ抽出する。既存M0の画像・TTS一括取得とLLM検証も、このモジュールを利用できるようにする。

取得単位は選択ウェイトと必要依存の組とする。Small／Largeの通常版は各モデルのtokenizerを含め、量子化版は選択したsubfolderと対応tokenizerを取得する。DACVAEとSilentCipherは同じ固定版を共有し、既に検証済みのファイルは再取得しない。モデルカタログ全体を一括で要求しない。TTSの精度設定でcodecの精度は変更せず、既存のfp32を引き継ぐ。

ダウンロード中はGPUを占有しない。I/O処理は生成処理と独立して動かす。ロード検査や短い推論検査はWorkerのGPU作業としてスケジュールし、常駐音声プロセスと保持中のGPU leaseを先に解放してから、既存GPUロックを取得する。取得用の別スレッドからロック取得だけを行う構成にはしない。中断では部分ファイルを保持し、同じ版の再開だけに使用する。同一ウェイトの重複要求は既存操作にまとめ、ディレクトリへの排他を行う。

サイズ・ハッシュ・必要なtokenizerと依存の検査後、ランタイムとGPUの対応を確認する。未確認の組み合わせには短いロード・生成検査を行い、成功後にWorkerの利用可能モデルへ登録する。未完了ファイルを推論へ渡さず、生成プロセスは引き続きオフラインで動かす。

SmallとLargeのライセンス情報はモデルごとに表示する。LargeはGemma由来の条件を持つため、配布されているNOTICE・規約等も取得記録に保持する。[Large量子化版のライセンス情報](https://huggingface.co/Aratako/Irodori-TTS-v4-Large-Quantized#license--ethical-restrictions)

## TTSエンジンの拡張

共通契約として`TTSRequest`、`TTSResult`、`TTSProvider`とProvider登録を設ける。Providerは対応モデル・用途・精度の申告、必要ウェイトの解決、ランタイム検査、生成を担当する。取得の進捗・再開・整合性検査は共通モデル管理へ委譲する。

共通の生成要求は本文、声の説明、参照音声、参照原文、感情、seed、用途別設定を持つ。Irodoriの`SamplingRequest`、絵文字による感情指定、tokenizerと文字数制約、watermarkの初期化はIrodori adapter内に置く。別TTSを追加したときに、共通pipelineへProviderごとの条件分岐を増やさない。

能力情報にはvoice_design／voice_clone／emotion等を明示する。デザイン非対応のエンジンをデザイン候補へ出さず、非対応パラメータは黙って無視しない。Irodori固有のステップ数等はProvider設定として拡張可能にし、初期値は既存の40ステップを引き継ぐ。

基準音声はWAV・正確な原文・ハッシュを共通の受け渡し形式とする。LargeでデザインしたWAVをSmallでクローンする場合も、クローン側のadapterで参照を読み込む。Provider固有のspeaker embeddingを共通参照として扱わない。将来異なるエンジンを組み合わせる場合は、参照形式・音声長・サンプルレートの互換性を受け側で検査する。

出力は既存のPCM16 WAV契約に合わせ、サンプルレートと実際のモデル・精度・watermark等をprovenanceへ記録する。Irodoriでは現在のwatermark必須方針を引き継ぐ。量子化ウェイトは公式のmetadataとtorchaoによる復元処理を利用し、単純な通常tensorのロードや一括dtype変換に置き換えない。

## モデル切り替えと設定の適用

常駐音声プロセスの識別にはProvider、モデル、ウェイトrevision／ハッシュ、量子化方式、計算精度、ランタイム版、全依存のfingerprint、deviceを含める。codecだけでなくtokenizer・SilentCipherの版やハッシュ変更も再起動対象にする。Small→Largeや精度変更では旧プロセスを終了してGPUメモリを解放し、新しいプロセスを起動する。同じ識別ならデザインとクローン間でも再利用できるが、captionと参照音声は毎回新しい要求として渡す。

Coordinatorは選択用途・モデル・版・精度を実行できるWorkerだけへジョブを割り当てる。対応Workerの不在や切断は待機理由として表示し、別モデルへ自動変更しない。取得対象Workerへの管理要求と、利用可能モデルによる生成ジョブ割り当てを区別する。

`tts_models`を申告しない旧Workerは、新しい`tts_profile`付きジョブを取得できない。旧ジョブだけを従来の条件で扱う。新しいプロファイル付きジョブでは、結果採用時も実際のProvider・ウェイトrevision／ハッシュ・量子化方式・計算精度・ランタイム版を要求と照合し、不一致の結果を採用しない。

M2の新規音声ジョブには作成時の設定を保存し、デザインと試聴クローンにそれぞれの設定を使う。M3／M4は制作開始時に両用途の設定を保存し、サブキャラの基準音声、台詞音声、後続章、素材再試行へ引き継ぐ。リトライと中断・再開は保存済み設定を使用する。

移行前のジョブ・本編にはSmallを使う互換プロファイルを設ける。実行済みWorkerの`job-request.json`等に精度・設定が保存されていればそちらを優先する。未実行ジョブなど保存履歴がないものには、移行時に既存voice設定を一度だけ保存した互換既定値を使い、過去の設定が復元できたとは扱わない。実行のたびに変更可能なconfigを読み直さない。

互換情報は既存payloadを書き換えずに別途保存する。これにより保存済みの入力fingerprintとの一致を維持する。既存基準音声と承認済み素材は保持し、新規ジョブのモデル変更で上書きしない。

## APIと主な変更ファイル

APIの案は次のとおり。実装時は現在のPOSTによる設定保存とWorkerのpoll方式に合わせる。

| API | 用途 |
| --- | --- |
| GET／POST `/api/settings/tts` | 用途別設定の取得・保存 |
| GET `/api/tts/models` | カタログとWorkerごとの利用状態 |
| POST `/api/workers/{worker_id}/tts-downloads` | 固定ウェイトIDの取得開始、操作IDを返す |
| GET `/api/tts-downloads/{operation_id}` | 進捗・検証結果・失敗理由 |
| POST `/api/tts-downloads/{operation_id}/cancel` | 中断して部分ファイルを保持 |
| POST `/api/tts-downloads/{operation_id}/resume` | 同じ版で再開・再試行 |

Worker登録・能力更新APIには後方互換な`tts_models`とランタイム対応情報を追加し、管理要求の取得・進捗報告を設ける。ブラウザーから任意URLやローカルパスを渡さず、カタログのIDで操作する。

| 層 | 主な対象 |
| --- | --- |
| Web | `LiveWizardApp.tsx`、`LLMSettings.tsx`、新規`SettingsDialog.tsx`・`TTSSettings.tsx`、関連CSS・API型 |
| 契約 | 新規`packages/contracts/tts_settings.py`、TTS要求・結果・能力契約 |
| Coordinator | 新規`tts_settings.py`・モデル管理API、`app.py`、`service.py`、新規DB migration |
| 設定固定 | `m2_service.py`、`m3_service.py`、`m4_service.py`、制作スナップショット |
| Worker | `__main__.py`、`client.py`、新規TTSカタログ・ローカル登録・モデル取得管理 |
| 生成 | `pipeline.py`、`m3_pipeline.py`、`voice_runner.py`、`voice_session.py`、新規TTS adapter |
| 配置と検証 | TTSカタログ、`config/m0-models-tts.json`の互換扱い、`tts_smoke.py`、`run_background.py`、runtime診断 |
| 文書 | `docs/setup/irodori.md`の実際のモデル導入状況の整理、新規TTS設定ガイド |

DB migrationにはTTS設定、Worker能力、ダウンロード管理要求と状態を追加する。既存のLLM・retry用migrationと衝突しない次の番号を実装時に決める。既存の作業中ファイルを前提として差分を重ねる。

## 実装順序

1. **ランタイム互換性を確認する。** 固定済み環境でSmall／Large、INT8／INT4、bf16対応を確認し、各取得元のrevision・ファイルハッシュ・tokenizer・依存を確定する。Windowsでの量子化実行とオフラインロードを先に確認する。
2. **契約とカタログを作る。** 用途別設定、精度の内部変換、Provider能力、固定ウェイトID、ローカル登録、旧voice設定の互換解決を実装する。
3. **Irodoriの固定参照を解消する。** Provider adapter、選択依存だけの検証、モデル切替時のプロセス終了、動的provenanceを実装し、既存Smallの動作を確認する。
4. **Coordinatorの保存と割り当てを実装する。** DB、API、Worker申告、用途別のジョブ設定固定、M3／M4の制作開始時の固定、Worker適合判定を実装する。
5. **非同期ダウンロードを実装する。** 共通取得処理、Worker管理要求、進捗、中断再開、整合性検査、利用可能状態への更新を実装する。GPU検証を生成と排他にする。
6. **設定画面を実装する。** LLM／TTSタブ、用途別選択、取得先Worker、取得・検証状態、保存、未保存入力の保持を接続する。
7. **統合と実機検証を行う。** 全形式の音声生成、異なるモデル間のクローン、モデル切替、制作設定の固定、既存作品の互換動作を確認し、配置手順と測定結果を記録する。

2以降の契約・画面部品の準備は並行できるが、実行対応の確定は1の検証結果に従う。3・4の後に5・6を統合し、7まで通過して完了とする。

## 受け入れ条件

| 分類 | 確認内容 |
| --- | --- |
| 設定 | LLM／TTSタブ、両用途の独立選択、DB保存・再読み込み、LLM設定の回帰、キーボード・狭い画面・フォーカス復帰 |
| 精度 | 2モデル×4精度×2用途の16通りで実生成。fp32／bf16の取得共有を確認 |
| 組み合わせ | Largeデザイン→Smallクローンと逆方向を試聴し、参照元と原文・ハッシュを確認 |
| 切り替え | 同精度Small→Large、精度変更、依存版変更、用途切替、別キャラの参照漏洩防止、GPU検証へ移る際の常駐プロセス・lease解放 |
| ダウンロード | 中断再開、Worker再起動、重複要求、破損・容量不足・通信失敗、未取得モデルによる既存推論への影響なし |
| 割り当て | 同じkindでもモデル・版・精度・用途が不適合なWorkerはclaimできず、旧Workerは新プロファイル付きジョブを取得できない |
| 結果採用 | 要求と実際のモデル・ウェイト・精度・ランタイムが不一致なら採用しない |
| 制作 | 保存後の新規ジョブに反映し、待機中・実行中・再試行・後続章は保存済み設定を保持 |
| 互換性 | 旧Worker能力・旧ジョブ・旧本編を扱え、既存payloadのfingerprint、基準音声、承認を保持し、互換既定値を再試行でも固定 |
| 拡張 | テスト用Providerを登録し、Irodoriのpipeline分岐を増やさず能力申告・設定・生成へ接続できる |

自動テストは契約・精度変換・依存解決・切替識別・取得の状態遷移・Worker割り当て・スナップショットを中心に追加する。実機では初回ロード／再利用時の時間、ピークVRAM、音声長、読みの欠落・繰り返し、声質、感情、watermarkを確認する。数値・WAVの構造検査と、人の試聴結果は分けて記録する。

Largeを全用途の初期値にはしない。公式評価では声デザインの改善が示される一方、読みの誤り率はv4.1 Smallよりやや高い。用途ごとに選択できる今回の要件を活かし、初期値の変更は当プロジェクトの比較試聴後に判断する。[Largeの公式評価とその制約](https://huggingface.co/Aratako/Irodori-TTS-v4-Large#benchmarks)

計画作成時は量子化カーネルの基礎検査のみを行った。その後、利用者によるLarge通常ウェイト取得を経て生成接続を実装した。Large・bf16の実生成検証は完了し、全16通りの実ウェイト検証と人による比較試聴は残っている。
