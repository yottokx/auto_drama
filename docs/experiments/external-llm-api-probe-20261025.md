# 外部LLM APIの確認（Phase 0、2026-10-25）

`docs/external-llm-api-plan.md` のPhase 0。製品のコードは変えず、使い捨てのスクリプトで両社のAPIを直接呼んで確かめた。

- モデル: OpenAI `gpt-6-luna`（Responses API）、Anthropic `claude-haiku-5-5`（Messages API、`anthropic-version: 2026-10-21`）
- キーはユーザー環境変数から子プロセスにだけ渡した。値は表示・保存していない
- 入力に使ったのは、製品の保存済みリクエスト（`services/worker/cache/m2/jobs/` と `outputs/` の要求キャッシュ）

## 1. 構造化出力のスキーマ

### 集めたスキーマ

保存済みの要求キャッシュ約1.8万件から、`json_schema` の応答形式を3,547種類（名前33種）集めた。2026-10-16以降に製品で使われた名前は13種（`m2_result`、`script-allocation`、`script-cast`、`script-outline`、`script-plan`、`script-voices`、`staging`、`script-staging`、`script-handoff`、`speech_separation`、`voices`、`character`、`chapter-count`）。使われているキーワードは `$ref`、`additionalProperties`、`anyOf`、`const`、`default`、`description`、`enum`、`items`、`maxItems`、`maxLength`、`maximum`、`minItems`、`minLength`、`minimum`、`pattern`、`required`、`title`、`type`、`uniqueItems`。

### キーワードごとの受け付け（小さなスキーマで1つずつ）

| キーワード | OpenAI | Anthropic |
|---|---|---|
| `title`、`description`、`enum`、`const`、`default`、`anyOf`（null との組み合わせ）、`$ref` / `$defs` | 可 | 可 |
| `minLength`、`maxLength`、`pattern` | 可 | 可（HTTP 200。出力で守られるかは下の生成の結果を参照） |
| `minimum`、`maximum` | 可 | 不可 |
| `minItems` 1 | 可 | 可 |
| `minItems` 2以上 | 可 | 不可（0か1だけ） |
| `maxItems` | 可 | 不可 |
| `uniqueItems` | 不可 | 不可 |
| `required` に入っていない項目（省略可能な項目） | 不可（全項目を `required` に入れる必要がある） | 可 |

### 実際のスキーマ

上の結果から、送る前の変換を次のようにした。

- OpenAI: `uniqueItems` を外す。すべての object で全項目を `required` にする。
- Anthropic: `minimum`、`maximum`、`maxItems`、`uniqueItems` を外し、2以上の `minItems` を1にする。`type: ["null", "string"]` に null を含む `enum` は、`anyOf: [{"type": "string", "enum": [...]}, {"type": "null"}]` に書き換える。

13種の名前ごとに最新と最大のもの（`m2_result` は項目の組み合わせごとに最大のもの、計62個）を送った結果。

- OpenAI: 変換後はすべて受け付けた。変換なしで通らなかったのは、省略可能な項目があるものと `uniqueItems` があるもの。
- Anthropic: 変換後、3つを除いて受け付けた。
  - `m2_result` の CG 計画（`cgs`）: null を含む `enum` で失敗。anyOf への書き換えで通った。
  - `script-outline`（13,273字、2026-10-11の実験）と `script-plan`（11,360字、2026-09-26の実験）の最大のもの: 「文法が大きすぎる」で失敗。`minLength` / `maxLength` / `pattern` を外しても、全項目を必須にしても通らなかった。
  - 2026-10-16以降に使われた `script-outline`（1,589〜7,028字）と `script-plan`（4,636〜7,444字）の全94種類は、変換後すべて通った。どの大きさから通らなくなるか（7,444字と11,360字の間）と、何で大きくなるかは調べていない。

### 道具（M2のランダム道具4つ）

- OpenAI: 関数ツール（`strict: true` でも false でも）で受け付け、`random_integer` を正しい引数で呼んだ。
- Anthropic: `strict: true` は `maxItems` のため不可。`strict` なしなら受け付け、正しい引数で呼んだ。

## 2. 推論

| | OpenAI（`reasoning.effort`） | Anthropic |
|---|---|---|
| 受け付ける値 | `none`、`low`、`medium`、`high`、`xhigh`、`max`（`minimal` は不可） | `thinking: {type: "disabled"}`（effort は `high` 以下のときだけ）、`thinking: {type: "adaptive"}` + effort `low`〜`max` |
| 推論なし | `none` で推論トークン0 | `disabled` で思考トークン0 |
| 何も指定しないと | 推論する（短い依頼で31トークン） | 推論する（同じ依頼で87トークン） |

推論なしは両社とも明示すれば使える。何も指定しないと推論するので、`none` は必ず送る。

## 3. 温度・top_p

| | OpenAI | Anthropic |
|---|---|---|
| 推論なし（`none`） | `temperature`、`top_p`、両方の同時指定とも可 | 不可（`2026-10-21` では `temperature` / `top_p` を送るだけで400。`2023-06-01` でも既定値以外は400） |
| 推論あり | `temperature` は1だけ可（0.65は400） | 同上 |

Anthropic のモデルでは、LLM共通設定の温度と top_p を使えない。OpenAI のモデルでは、推論ありにすると温度を変えられない。

## 4. 入力トークン数

製品の保存済みリクエスト（`script-plan` 1件、場面の本文1件）を、数えるAPIと実際の生成の両方に送った。

| リクエスト | Gemma 4 31B（llama-server の実測） | GPT-6 Luna | Haiku 5.5 |
|---|---|---|---|
| `script-plan` | 5,648 | 7,173（数えるAPIと実際が一致） | 10,794（一致） |
| 場面の本文 | 8,044 | 9,481 | 11,618 |

- 数えるAPI（OpenAI `POST /v1/responses/input_tokens`、Anthropic `POST /v1/messages/count_tokens`）は、スキーマを含めて実際の `usage` と同じ値を返した。
- 同じリクエストでも、入力トークンは Gemma より GPT-6 Luna で1.2〜1.3倍、Haiku で1.4〜1.9倍になる。今のコンテキストの段（16k / 32k / 64k）と作品の上限は、Gemmaのトークン数で決めている。

## 5. 実際の生成

| | GPT-6 Luna | Haiku 5.5 |
|---|---|---|
| `script-plan`（推論なし、上限14,336） | 完了、1,557トークン。元のスキーマ（外した制約を含む）の違反0 | 完了、1,413トークン。違反0 |
| 場面の本文（grammarなし、推論なし、上限11,264） | 完了、1,292トークン・34行。形式の崩れた行0 | 完了、1,806トークン・36行。崩れた行0 |
| 場面の本文（上限300） | `status: incomplete`、`incomplete_details.reason: max_output_tokens` | `stop_reason: max_tokens` |

- 同じ場面の Gemma（grammarあり）の出力は1,186トークン。
- 外した制約（文字数、要素数、数値の範囲）は、この1件では両モデルとも守っていた。1件だけなので、製品では今どおり受け取った後に元のスキーマで検証する。
- 場面の本文は1件ずつで、質は見ていない（Phase 2で見る）。
- OpenAI は同じ入力の2回目で `cached_tokens` が付いた（自動のプロンプトキャッシュ）。

## 6. モデルの情報

- Haiku 5.5: Models API で入力上限1,000,000トークン、出力上限128,000トークン。
- GPT-6 Luna: Models API はコンテキストの上限を返さない。

## 7. 計画への反映

- スキーマの変換は上の規則でよい。受け取った後は、今どおり元のスキーマ（と pydantic）で検証する。
- 推論なしは、OpenAI に `reasoning.effort: "none"`、Anthropic に `thinking: {type: "disabled"}` を必ず送る。
- 温度と top_p は、Anthropic には送らない。OpenAI には推論なしのときだけ送る。
- Anthropic の道具は `strict` を付けない。
- 入力トークンがモデルで1.2〜1.9倍違うので、外部モデルでは段を作品の上限と総生成枠にだけ使い、入力が収まるかはモデルの実際の入力上限で確かめる（2026-10-25 決定）。
- Anthropic で通らなかった大きなスキーマ2つは古い実験のもので、今の製品は作らない。追わない（2026-10-25）。製品で起きた場合は、そのジョブを失敗にして理由を残す。
- 温度・top_p を受け付けないモデル・推論の組み合わせでは、設定画面に欄を出さない（2026-10-25 決定）。
