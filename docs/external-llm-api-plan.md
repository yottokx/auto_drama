# 外部LLM API（OpenAI・Anthropic）対応の実装計画

GPT-6 Luna（`gpt-6-luna`）と Claude Haiku 5.5（`claude-haiku-5-5`）が台本生成に使えるかを、製品と同じ生成の流れで確かめる。試験用に別の経路を作らず、LLM呼び出しの基盤（`LocalLLM`）に外部APIを足す。既存の試験スクリプト（`check_text_long.py` など）も製品の生成も、同じ基盤を通るので、そのまま外部APIで動く。

2026-10-25 時点で、両社のキーはユーザー環境変数に設定済み。最小の呼び出し（短い日本語の会話を3行）は両モデルとも成功した。`gpt-6.1-luna` はまだ提供されていない（`model_not_found`）。

## 1. 決定事項

| 項目 | 決めたこと |
|---|---|
| 対象 | `LocalLLM` / `RoutedLLM` を通るテキストLLMの呼び出しすべて。画像・音楽・音声の生成モデルはローカルのまま。画像の生成ジョブや音楽ジョブの中にあるLLMの工程（立ち絵の指示文、衣装の案など）も `LocalLLM` を通るので、共通設定のモデルに従う |
| 対象外 | 画像編集GUI・音声GUIのLLM呼び出し（`scripts/image_edit/llm_runner.py`、`scripts/audio/llm_runtime.py`）。外部APIに対応させるかは未定 |
| 差し替える場所 | `LocalLLM` の送受信だけ。内部のリクエストは今の chat completions 形のまま作り、送る直前に各社の形式へ変換し、受け取った応答を `choices[0].message` / `finish_reason` / `usage` の形へ戻す。呼び出し側・応答キャッシュの形式・trace は変えない |
| 使うAPI | OpenAI は Responses API（`store: false`）、Anthropic は Messages API。公式SDK（`openai`、`anthropic`）をWorkerのvenvにバージョン固定で入れる。429・5xxの再送はSDKの既定に任せる（通信の再送で、出力の検査・やり直しではない） |
| APIキー | Workerのプロセスの環境変数 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` だけを読む。Coordinator・DB・ジョブのpayload・設定のスナップショット・trace・キャッシュ・ログ・エラーメッセージには出さない。`.env` の読み込みは今回は入れない |
| モデルの申告 | `services/worker/config/models.local.json` に `api_models` を書く（例: `["openai/gpt-6-luna", "anthropic/claude-haiku-5-5"]`）。キーがあるプロバイダーのものだけをWorkerが候補として申告する。モデルIDの `/` はファイル名に使えない文字なので、GGUFのファイル名から作るIDとは重ならない |
| コンテキストの段 | 16k / 32k / 64k の段はそのまま選ぶ。外部モデルでは、段は作品の上限（章数・キャラ数など）と総生成枠の拡張にだけ使い、入力が収まるかの確認はモデルの実際の入力上限で行う（入力トークンがモデルで Gemma の1.2〜1.9倍になるため。送る資料の量は段で変わらないので、費用も変わらない。2026-10-25 決定）。llama-server は起動しない |
| 推論 | `reasoning_effort` の文字列を、OpenAI は `reasoning.effort` へ（`none`〜`max`）、Anthropic は `none` なら `thinking: {type: "disabled"}`、それ以外は `thinking: {type: "adaptive"}` と effort（`low`〜`max`）へ渡す。両社とも何も指定しないと推論するので、`none` は必ず送る（Phase 0で確認） |
| 本文の形式 | 外部APIでは grammar（`話者ID: 台詞` を強制する仕組み）を使えない。今の推論ありの経路と同じく、生成後に `parse_scene_text` で検証する。モデルに検査させてやり直す仕組みは足さない。形式の崩れが多ければ、Phase 2の結果を見て指示文や入力の構造で直す |
| GPU | 外部APIのジョブは llama-server を起動せず、常駐ローカルLLMの再利用（`llm_session`）の対象にもしない。GPUロックの取り方は今の呼び出し側のまま変えない |
| 再現性 | seed は送らない（Responses API・Messages API とも保証がない）。中断からの再開と再試行は、今の応答キャッシュで同じ応答を再生する。キャッシュの識別には provider とモデルIDを含め、応答が返したモデル名（実際のスナップショット）を trace に残す |
| 既存の作品 | 保存済みの `provider: local` のスナップショットはそのまま動く。外部モデルは、共通設定で選んだ後の新しいジョブと、明示的な再試行から使われる（今の設定変更のルールどおり） |

## 2. 今の仕組みと外部APIでの扱い

| 今の仕組み（llama-server） | 外部APIでの扱い |
|---|---|
| `/v1/chat/completions` | OpenAI: `responses.create`。system は instructions へ、user/assistant/tool は input へ変換。Anthropic: `messages.create`。system は `system` 引数へ分ける |
| `max_tokens` | OpenAI: `max_output_tokens`、Anthropic: `max_tokens`。値は今の総生成枠（推論を含む）をそのまま使う |
| `finish_reason` | OpenAI: `status: completed` → `stop`、`incomplete`（`max_output_tokens`）→ `length`。Anthropic: `end_turn` → `stop`、`max_tokens` → `length`、`tool_use` → `tool_calls`。どちらも拒否（refusal）は理由を残して失敗にする |
| `response_format: json_schema`（strict） | OpenAI: `text.format` の json_schema（strict）、Anthropic: `output_config.format`。送る前に変換し（OpenAI: `uniqueItems` を外し、全項目を `required` に。Anthropic: `minimum` / `maximum` / `maxItems` / `uniqueItems` を外し、2以上の `minItems` を1に、null を含む `enum` を anyOf に）、受け取った後は今の `validate_schema` / pydantic の検証で元の制約を確かめる（Phase 0で確認） |
| `tools`（M2のランダム道具） | 各社の関数ツール形式へ変換する。Anthropic は `strict` を付けない（`maxItems` を受け付けないため）。tool_calls と tool の結果のメッセージも変換する |
| `grammar` | 送らない（`_source_options` は外部モデルでは空を返す） |
| `chat_template_kwargs`、`reasoning_budget_tokens` | 送らない。Qwen専用の設定なので、外部モデルの経路では指定できないことを検証で弾く |
| `/apply-template` + `/tokenize` で入力トークンを数える | OpenAI: 入力トークン数を数えるAPI、Anthropic: `count_tokens`。数えた値で今の `context_budget` をそのまま確かめる |
| `/health`、`/props`、モデルの読み込み | なし。`server_context_size` は選んだ段の値にする |
| `temperature`、`top_p` | Anthropic には送らない（`2026-10-21` では送るだけで400）。OpenAI には推論なしのときだけ送る（推論ありでは温度1だけ可）。受け付けないモデル・推論の組み合わせでは、設定画面に温度・top_p の欄を出さない（2026-10-25 決定） |
| `usage.completion_tokens` | 出力トークン（推論を含む）を入れる。推論トークンは別に記録し、本文の文字数の学習には推論なしの実行だけを使う今のルールを守る |
| `timings` | なし。trace には usage だけを残す |

## 3. 作業の段階

### Phase 0: APIの確認（製品のコードは変えない）

スクラッチのスクリプトで、次を両モデルについて確かめる。結果は `docs/experiments/external-llm-api-probe-YYYYMMDD.md` に残す。

1. 今の構造化出力のスキーマ（`narrative.py`、`script_*.py`、`planning_run.py` などで使うもの）を全部、各社の structured outputs に送って受け付けられるか。受け付けないキーワードの一覧。
2. 推論なし（`none`）を指定できるか。指定できない場合の最小の推論の量。
3. `temperature` / `top_p` の範囲と、両方を同時に指定できるか。
4. 入力トークン数を数えるAPIが、実際の `usage` の入力トークン数と一致するか。
5. 8,192トークン程度の出力上限で、上限に達したときの応答（`length` に当たるもの）の形。

### Phase 1: 基盤

| 変更する場所 | 内容 |
|---|---|
| `services/worker/generation/llm_providers.py`（新規） | 各社への変換と送受信。SDKのクライアントは環境変数から作り、キーを引数や属性に保存しない |
| `services/worker/generation/llm.py` | `__enter__` / `request` / `chat` / `check_context` で、外部モデルなら llama-server を起動せず `llm_providers` へ送る。`provider != "local"` を弾く検証を外部モデル用に分ける |
| `services/worker/model_config.py` | `api_models` の読み込みと、キーがあるものだけの申告。`select_config` / `resolve_entry` に外部モデルの分岐を足す（GGUF・サーバーの確認をしない） |
| `packages/contracts/llm_settings.py`、`services/coordinator/llm_settings.py` | `profile()` の `provider` をモデルIDから決める。Workerの申告に、受け付ける温度の範囲と最大コンテキストを含める |
| `narrative.py`、`pipeline.py`、`model_routing.py`、`llm_session.py` | `provider == "local"` の決め打ちを外す。`_source_options` は外部モデルで grammar を返さない。常駐セッションは外部モデルを扱わない |
| テスト | 各社のSDKの応答を偽物に差し替えた単体テスト（通信しない）。キーが trace・キャッシュ・エラーに出ないことのテスト。実際のAPIを呼ぶ確認は `AUTO_DRAMA_API_SMOKE=1` のときだけ動かす |

### Phase 2: 品質の確認

`check_text_long.py --model-id openai/gpt-6-luna`（と `anthropic/claude-haiku-5-5`）で、Gemma 4 31B（推論なし）で使ってきたのと同じ入力を生成し、計画と台本の質を比べる。入力の数・章数・推論の条件は、実行の前にユーザーと決める。結果は `docs/experiments/` に残す。

### Phase 3: 製品で選べるようにする

Phase 1でWorkerが外部モデルを申告すると、LLM共通設定のモデル一覧に出る。ここでは、設定画面で選べる値（温度の範囲、コンテキストの段）がモデルの申告に従うことを確かめ、`docs/setup/llm-settings.md` に外部APIの使い方（環境変数、`api_models`、Workerを環境変数が見えるターミナルから起動すること）を足す。

## 4. あとで決めること

- 画像編集GUI・音声GUIのLLMを外部APIに対応させるか。
- 費用の見せ方（今はtraceと実験記録にトークン数を残すだけ）。
- Anthropic のプロンプトキャッシュなど、費用を下げる仕組みを使うか（品質の確認の後）。
- GPUを使わない外部APIのLLMジョブを、画像の生成と同時に動かすか（今は1ジョブずつのまま）。
- `.env` からのキーの読み込み。
