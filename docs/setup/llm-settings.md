# LLM共通設定

サイドバーの「LLM共通設定」で、接続先、モデル、temperature、top_p、推論（reasoning_effort）、
コンテキスト長を設定する。用途別の設定欄は設けない。
保存先はCoordinatorのDBであり、ブラウザーのローカルストレージには保存しない。

- temperature: 0〜2。
- top_p: 0より大きく1以下。
- コンテキスト長: 最低16,384。指定値をそのままllama-serverの `--ctx-size` に渡す。
  既存の16,384や32,768を固定上限として扱わない。ロード後は実際のコンテキスト容量を確認する。
- 接続先: ローカルと、Workerの設定の `api_providers` に書いた接続先（その `id` で出る）のうち、
  接続中のWorkerに使えるモデルがあるものだけを出す。
  モデルは選んだ接続先のものだけを、接続先の部分を除いた名前で出す。
- 推論（reasoning_effort）: そのモデルをWorkerが受け付ける強さ（なし・低・中・高・とても高い・最大）から選ぶ。
  受け付けないモデルに切り替えると「なし」に戻す。
- 総生成上限（本文＋推論）は16,384を基準に自動拡張する。
  既存の処理別上限・本文の予算計算を基準値として、
  `floor((ctx_size - 16384) / 2)` トークンを加える。
  コンテキストの拡張分を入力と生成に1:1で配分し、端数1トークンは入力側へ回す。
  16,384では従来と同じ。32,768なら生成枠に8,192、65,536なら24,576を加える。
  章割り当ての基準値2,048は32,768設定で10,240になる。
  本文の目標文字数は増やさず、reasoning_effortの値による比率変更や数値換算もしない。
  推論を無効にした場合も同じ配分を使う。用途別の比率や上限のUIは設けない。
  推論専用の打ち切り機能には依存せず、本文用の残量を保証するものではない。

入力枠は `実効コンテキスト長 - 総生成上限 - 安全余白` で確認する。
モデルのロード後は実際の容量でも検証し、収まらない場合は既存の入力選択ルールを使う。
必須資料や総生成枠を黙って切り捨てない。
Workerの既存 `max_output_tokens` は16k時の基準許容量として保持し、
送信・入力容量確認時に同じ拡張分を加える。ローカルパスや保存済みモデル設定は変更しない。
保存済みプロファイルや `task_profile` の `max_tokens` は基準値、
実リクエストの `max_tokens` と容量確認の `output_tokens` は拡張後の総生成上限を表す。
台本の本文枠は従来の文字数見積もりに拡張分を一度だけ加え、実際の送信値を保存する。
推論ありの拡張実行では、推論を含む消費トークン総数を本文文字数の学習に流用しない。
同じ実行の中断・再開や自動再試行では、保存済みの場面生成枠を維持する。
明示的な再試行で設定を変更した場合、執筆が未完了の場面は新しい設定で生成枠を計算し、元の枠も履歴に残す。
本文生成が済んだ場面は、その原文と回答枠を保持して残りの工程を続ける。
この自動拡張はLLM共通設定を持つジョブに適用し、従来設定だけの実験CLIは変更しない。

## Workerのモデル配置

`services/worker/config/models.example.json` を同じディレクトリの
`models.local.json` にコピーし、モデルを置くフォルダを指定する。
このファイルはGit管理外。GGUFはコピーせず、その場所から直接読む。
フォルダは絶対パス、またはリポジトリルート基準の相対パスを指定できる。

```json
{
  "schema_version": 1,
  "model_dirs": ["private/model_download"]
}
```

指定フォルダとサブフォルダの `.gguf` を自動スキャンする。
モデルごとの登録やメタデータファイルの作成は不要。
ファイル名（拡張子を除く）をモデルIDとして扱い、同じ名前のファイルが
複数フォルダにある場合は曖昧な選択を避けるため設定エラーにする。
ファイルサイズ・GGUFヘッダーとサーバー実行ファイルの存在を確認する。
既知モデルのメタデータと推論設定は既存設定から引き継ぎ、その他は
コンテキスト16,384、推論 `none` で候補に表示する。候補への表示は実推論の動作確認を意味しない。
サーバーの場所は `server_executable` で指定できる。
コンテキスト長はUIで保存した値を使用し、Worker configの旧 `max_context_size` では制限しない。
既存の個別 `models` 登録も互換性のため使用できるが、自動検出では不要。

Workerは起動時とジョブ取得前（最短10秒間隔）に再スキャンする。
追加・削除後はWorkerの待機中に自動反映される。実行中のジョブは中断しない。
UIの「Worker情報を更新」で最新の申告内容を取得できる。
初回のコード更新時だけCoordinatorとWorkerを再起動する。
登録情報にはローカルパスを含めず、直近60秒以内に通信したWorkerの候補を表示する。

CoordinatorはジョブのモデルIDに対応するWorkerへ
処理を割り当てる。同じモデルIDを複数Workerで使う場合は、同じモデル・量子化を配置する。

## 外部APIのモデル

`models.local.json` の `api_providers` に接続先（API種別・エンドポイント・キーの環境変数名）を、
`api_models` にその接続先で使うモデルと入力上限を書く。

```json
{
  "schema_version": 1,
  "model_dirs": ["private/model_download"],
  "api_providers": [
    {"id": "openai", "type": "openai_responses", "api_key_env": "OPENAI_API_KEY"},
    {"id": "anthropic", "type": "anthropic_messages", "api_key_env": "ANTHROPIC_API_KEY"},
    {"id": "openrouter", "type": "openai_chat", "endpoint": "https://openrouter.ai/api/v1",
     "api_key_env": "OPENROUTER_API_KEY",
     "request_fields": {"provider": {"require_parameters": true, "data_collection": "deny"}}}
  ],
  "api_models": [
    {"provider": "openai", "model": "gpt-6-luna", "max_input_tokens": 272000},
    {"provider": "anthropic", "model": "claude-haiku-5-5", "max_input_tokens": 1000000},
    {"provider": "openrouter", "model": "google/gemma-4-31b-it", "max_input_tokens": 262144}
  ]
}
```

`api_providers` の項目:

| 項目 | 内容 |
|---|---|
| `id` | 接続先の名前。英数字と `.` `_` `-`。設定画面の「接続先」にこの名前で出る。`local` は使えない。 |
| `type` | APIの種別。`openai_responses`（OpenAI Responses API）、`anthropic_messages`（Anthropic Messages API）、`openai_chat`（OpenAI互換のChat Completions。OpenRouter・vLLMなど）。 |
| `endpoint` | APIのURL（`/v1` まで）。`openai_responses` と `anthropic_messages` は省略すると公式のURL。`openai_chat` は必須。 |
| `api_key_env` | APIキーを入れた環境変数の名前。キーそのものは書かない。省略すると認証なしで送る（同じPCのvLLMなど）。 |
| `request_fields` | 毎回の依頼に足すフィールド（任意）。例: OpenRouterの提供元の選び方 `provider`。依頼が元から持つフィールドは置き換えられない。 |

`api_models` の項目:

| 項目 | 内容 |
|---|---|
| `provider` | `api_providers` の `id`。 |
| `model` | その接続先でのモデル名（例: OpenRouterの `google/gemma-4-31b-it`）。 |
| `max_input_tokens` | そのモデルの入力上限。最低16,384。 |
| `reasoning_efforts` | 受け付ける推論の強さ（任意、`none` を含む）。省略するとAPI種別の既定（`openai_chat` は `none` のみ）。 |

- アプリの中のモデルIDは `接続先のid/モデル名`（例: `openrouter/google/gemma-4-31b-it`）。
- 接続先の `type`・`endpoint`・`request_fields` とモデル名はモデルの識別に入るので、
  変えると応答キャッシュは別扱いになる。キーの環境変数名は識別に入らない。
- キーの環境変数が設定されていない接続先のモデルは候補に出ない（Workerのログに理由が出る）。
  キーはCoordinator・DB・ジョブ・ログ・応答キャッシュには保存しない。
- ユーザー環境変数を設定した後は、環境変数が見える新しいターミナル（またはエクスプローラー）から
  Workerを起動し直す。
- 外部モデルでは llama-server を起動せず、GPUのモデルも読み込まない。
- コンテキスト長の段は、章数・登場キャラなどの上限と出力トークン上限にだけ使う。
  入力が収まるかは `max_input_tokens` と比べる。`openai_responses` と `anthropic_messages` は
  各社のトークン数を数えるAPIで数え、`openai_chat` は数えるAPIがないので送る文字数を上限の見積もりとする。
  同じ文章でもトークン数はモデルで違う（Gemmaの1.2〜1.9倍）ため。
- 温度・top_p は、受け付けるAPI種別と推論の組み合わせでだけ送り、設定画面にも出す
  （`anthropic_messages` は送らない。`openai_responses` は推論なしのときだけ。`openai_chat` は常に送る）。
- 推論なし（`none`）は、`openai_responses` に `reasoning.effort: "none"`、`anthropic_messages` に思考なし、
  `openai_chat` には推論の指定を付けずに送る。推論ありは `openai_chat` に `reasoning.effort` を付ける。
- 本文の形式（`話者ID: 台詞`）はgrammarで強制できないので、話せる人物IDを依頼文で伝え、生成後に検証する。
- 中止は、送信中のリクエストの応答を待ってから効く。

### OpenAI互換のChat Completions（`openai_chat`）

- ローカルで使うのと同じ公開モデル（例: Gemma 4 31B）を、GPUなしで使うための経路。
  依頼はllama-server向けの形のまま送り、JSONスキーマも変えずに渡す（提供元が文法で守らせる）。
- OpenRouterは複数の提供元に振り分ける。上の例の `request_fields` は、送った指定（スキーマなど）を
  すべて守れて、作品を学習に使わない提供元だけに送る指定。`"quantizations": ["bf16"]` などを足せる。
- 応答キャッシュには、実際に応答した提供元（`upstream_provider`）も残る。提供元で量子化が違い、
  同じ依頼でも応答が変わりうるため。
- `parallel_tool_calls` は送らない（OpenRouterでは対応を表明している提供元がなく、振り分け先がなくなる）。
- 各社が受け付けないJSONスキーマの制約は送る前に外し、応答は元のスキーマで検証する
  （`docs/experiments/external-llm-api-probe-20261025.md`）。
- 構造化出力の依頼には、スキーマにある件数（`minItems` / `maxItems` / `uniqueItems`）と数値の範囲を、
  依頼文の末尾に一覧で書き足す。ローカルモデルではgrammarが黙って止めていた制約で、
  Anthropic は強制せず、OpenAI は強制するがモデルが知らないまま止められると回答が崩れるため。

## 設定変更の適用範囲

新しい生成ジョブを作成するときと、失敗したジョブの「再試行」を明示的に選んだときに、
現在の共通設定を取得する。全体計画や本編の後続章も、その時点の設定を使う。
コンテキスト長だけでなく、モデル、temperature、top_p、reasoning_effortも反映する。
物語全体の開始時設定を継承する必要はなく、容量不足で失敗した章はコンテキスト長を変更して再試行できる。

ジョブへ取得した設定はその実行のスナップショットとして保存する。キュー内・実行中のジョブと、
通信切断などからの自動再試行ではこのスナップショットを使い、処理途中の設定変更を反映しない。
明示的な再試行では新しい設定を記録し、既存の生成結果と過去の実行設定を保持する。
完成済みの章やシーンは再生成せず、未完了の工程から続ける。
コンテキストを拡張すると、未実行の工程の総生成枠も上記の配分に従って増える。
生成済みの本文に対応するシーン回答枠は維持する。未完了のシーンの回答枠は、
元の見積りを履歴へ残して新設定で再計算する。変更後のモデルで古いLLM応答を
新しい応答として再利用しない。

既存の未移行ジョブと実験CLIの再現性のため、従来の `config/m2-generation.json` と
その参照設定は残している。共通設定を初めて保存するまでは従来設定を使用する。
更新を反映するにはCoordinatorとWorkerを再起動し、Webを再読み込みする。
