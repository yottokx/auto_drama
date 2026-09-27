# 画像・音声なしの作劇実験

`scripts/story/check_text.py` は画像・音声の推論なしで台本生成を試すCLIです。`--policy script_continuation_v1` は人物ID付き台本・発話分離・演出を維持し、固定のテスト画像と音声未設定で既存のScript・ティラノソースまで生成します。coordinatorのDBや既存作品の公開状態は変更しません。従来の `causal` / `legacy` 経路は通常制作と同じ `m3_narrative` worker経路で生成・検査し、ティラノ組み立ては行いません。

2026-09-25 改訂：小説形式の `story_draft_v1` は、必要な台本形式と出力機能まで外しており、要求を満たしていないため続行対象から外しました。以下は実装済み `script_continuation_v1` の手順です。[章間改善計画](../story-first-poc-plan.md)は、未来を具体的な全体プロットから、過去の事実を実台本から構成して章計画で接続する方式を実装しました。通常UIへはまだ組み込まず、この独立した実験経路で確認します。

このコマンドを実行すると、設定済みのローカル LLM サーバーを起動し、終了時に解放します。モデルは自動ダウンロードしません。worker の Python 環境と `config/m2-generation.json`、その参照先のローカル LLM 設定・モデルが必要です。coordinator と通常 worker の起動は不要です。GPU ロックは通常生成と共有するので、別の制作が実行中なら待機します。

## 台本形式を維持する実験（script_continuation_v1）

現在はprompt 14 / implementation 15 / contract revision 13。[r13レビューを受けた修正](../experiments/script-continuation-r14-implementation-20260926.md)に対応しています。3章ではプロット前にサブキャラ・関係性・日常の接点を設計し、`cast-plan.json` に保存します。予定キャストは過去の登場実績と区別し、実際に登場させる人物だけ素材要求へ登録します。章計画はM3型の直接ScenePlanで、行動番号の対応表は作りません。旧revisionの出力は再開せず、新しい出力先を指定してください。比較設定は引き続き「境界線のエクリプス」・seed 1です。

プロット生成の同じ応答内で、`resolution_basis`に`earlier_experience`（先の行動と獲得した結果）と`later_application`（その結果を後でどう使うか）を最大2組記し、出来事列へ反映します。危機の深刻化だけを準備にせず、外的な決着では障害側の何が変わるかも具体化します。下書きは`draft-state.json`の構造化応答に保存し、`story-chain.json`以降へ重複して渡しません。章ごとの会話材料は`topic`と`exchange`で本人の用事・相手の都合や好み・返答を記します。旧`relationship_aspect`の保存データは読み取り可能ですが、新しい生成schemaには使いません。

プロット・章配分・章計画・本文には工程別の短い作例を任意資料として用います。対象作品の設定・実績・予定とは区別し、発話分離・演出・履歴メモには渡しません。容量不足時は作例を先に外し、その後にr12の未来詳細・履歴の容量調整を行います。`requests/*.json`の`selection.example`に版・hash・採否・理由、`selection.budget`に実要求の計測を保存します。作例のカタログと工程への割当も実験の同一性に含め、変更後の旧実行への混入を防ぎます。

章配分の修正：出来事が3個なら各章が1個ずつ担当するため、境界はコードで `[1, 2, 3]` に確定します。LLMは `chapter_1`〜`chapter_3` の章題・役割・会話の余地と伏線だけを出力します。4個以上では出来事番号と総数を明示して境界を選ばせ、順序・重複・全件配分を引き続き検証します。

Gemma単独で構成・章計画・実台本の引き継ぎ・執筆・発話分離・演出を行います。既定は16k、reasoningなしです。Qwenのレビュー設定がなくても実行でき、今回のフローではQwenをロードしません。論理整合性を求める処理に限定して将来利用できる既存の仕組みは残します。

[revision 2の通読レビュー](../experiments/script-continuation-r2-review-20260925.md)を受け、履歴メモを作業依頼から分離しました。メモは対象章の全文のみから作り、全体構成・過去メモ・次章の提案を混ぜません。後で容量調整が必要になった場合に備えて保存しますが、原文が全文入る計画・執筆要求には投入しません。

詳細プロットは `plot.json` に核心（中心課題・外的決着・人物関係の着地・各人物の転機）と章ごとの因果（開始条件・試み・結果・選択・次の状態）を保存します。既存の `outline.json` は同じ内容から機械的に作る表示用データです。3章構成では、まず `story-chain.json` に章境界のない3〜9個の主要な出来事と人物本人の行動を生成します。この数は今回の小規模実験の出力上限であり、長編の展開密度の基準ではありません。次に `chapter-allocation.json` で連続した出来事を3章へ配分します。出来事を一度ずつ順番通りに割り当て、`plot.json` の各章へ同じ内容を保存します。章ごとに出来事を生成し直しません。配分後の場面数はこの出来事数と同じとは限りません。3章以外は従来の計画経路を保持し、8章超の最大8章ずつの分割も従来のままです。可変章数と長編の設計は今回の対象外です。

3章のプロット生成では最初の `opening_condition` と出来事の行動・結果を生成します。後続出来事の開始条件は直前の最後の結果から組み立て、独立した開始事情を追加させません。保存用 `StoryChain` と表示用routeは維持します。計画入力では各章の出来事を渡し、同じ出来事をまとめた互換用routeは重複投入しません。

章計画は場面の場所・人物・目的・開始状態・required_events・終了状態を直接生成します。第2章以降の `continuation` は前章末尾を受けた最初の新しい行動を1〜2文・240文字以内で記し、第1場面の最初のrequired_eventへ直接組み込みます。後続場面には同じ冒頭指示を再掲しません。作業依頼は履歴・作例より後ろに配置します。未発生の前提が必要なら、その経緯を場面に含めるよう指示します。

第2章以降の初場面では、前章末尾と冒頭が完全一致する6発話以上・本文300文字以上の転載だけを除きます。空行と改行コード以外の表記差は一致と見なしません。`.raw.txt`に原応答、`.effective.txt`に除去後、`.overlap.json`に範囲・hash・除去前後の量を保存し、その後の発話分離・演出・履歴・出力・本文量には除去後を使います。全量コピーなら証跡を保存して停止し、再生成しません。意味の似た反復や短い台詞は対象外です。

執筆には当場面の場所定義と関係する人物の詳細を渡します。会話中に参照する不在人物や関係も残し、無関係の人物はID・名前・役割の表示へ絞ります。元のキャスト・場所・素材情報は保存データに残ります。当場面の終了状態・場所定義の重複を除き、後続場面の担当は残します。

容量不足時は遠い未来を章の目的・終点の表示へ切り替え、その後に古い完了章を本文由来メモへ置換します。完了章の原文とメモを両方省くことはありません。同章の既出本文は全文、直前の完了章の末場面は原文で保持します。空・取得失敗・出典不一致のメモは代替に使いません。必須資料が収まらなければ送信前に停止し、入力範囲・代替理由・出典hash・不足量を `requests/` と `preflight_failures` に記録します。

意味の合否検査は設けません。3章の全体計画はキャスト設計・出来事の連鎖・章への配分の3工程です。独立した意味検査・修復工程は追加しません。話者ID、演出、Script、ティラノ出力の契約は維持します。

主筋を進めない交流場面も計画できます。場面間の相対的な `length_weight` から各話の本文・台詞目安を配分し、数値を内部の `scene_sizes` に保存します。会話・日常の量を計画し、短い結果要約や反復で尺を埋めない方針です。

入力ラッパーの `script_options` で分量の補助目安を指定できます。既定値は以下です。各話への目安であり、字数による合否判定や再生時間の保証ではありません。変更すると実験の同一性が変わるので、新しい出力先を使います。

```json
{
  "approval_snapshot": { "...": "承認済み設定" },
  "script_options": {
    "target_episode_minutes": 10,
    "target_body_characters": 4000,
    "target_dialogue_characters": 2000
  }
}
```

`report.json` の `content_metrics` に本文・台詞・人物別文字数、ナレーション比率、場面数を記録します。音声なしのため再生秒数は未計測です。`cast-plan.json` は本文に登場する前の全体設計、`plot.md` は成立した全体プロットの読みやすい表示です。

```powershell
.\services\worker\.venv\Scripts\python.exe -I -u -X utf8 .\scripts\story\check_text.py `
  --input .\tests\fixtures\story-debug-relationship.json `
  --output-dir .\outputs\script-relationship-seed1 `
  --workflow causal --policy script_continuation_v1 --seed 1 --chapter-limit 1
```

キャストと3章プロットだけを生成して本文へ進まない場合は、`--plot-only` を指定します。検証済みのユーザー設定を使う例：

```powershell
.\services\worker\.venv\Scripts\python.exe -I -u -X utf8 .\scripts\story\check_text.py `
  --input .\private\story-inputs\eclipse-m3-approved-20260926.json `
  --output-dir .\outputs\script-eclipse-seed1-r13-new `
  --workflow causal --policy script_continuation_v1 --seed 1 --plot-only
```

`--plot-only --resume` は保存プロットを検証して終了し、本文を生成しません。本文を開始するには同じ入力・出力先で `--plot-only` を外し、`--resume --chapter-limit 1` を指定します。実行到達点は作品同一性と分離し、人物・プロットを再生成しません。承認済み3章構成を保ったまま、第1章の技術出力完了で止まります。続きは同一設定・出力先で `--resume` を加え、`--chapter-limit` を増やすか外します。今回の実装確認では3章通し生成の前に報告します。

章を跨ぐ時は前章の実台本を渡し、実際に起きた結果から次章を計画します。任意の引き継ぎメモが失敗した場合は理由を記録して原文から続行します。細かな意味レビュー・詳細台帳・引用ID照合は実行せず、`not_evaluated` と記録します。話者・原文・演出・素材参照、前章のhashと順序、Script・ティラノソースの技術検証は維持します。

コンテキストは設定に応じて拡張可能です。`profiles.script_planning`、`script_handoff`、`script_writer`、`script_speech`、`script_staging` で工程別の枠を指定できます。全体の `profile` や `--context-size` も利用可能です。原文が入る間は全文を渡し、不足時のみ古い章のメモへの置換や境界を保った抜粋を行います。必要な直前場面・対象台本・回答枠が入らなければ容量不足として保存して停止します。

章説明は `role`（最大200文字）と `conversation_topics`（最大4件）です。各話題は登録済み人物2〜3人、具体的な話題と関係の側面（各最大120文字）。これらは計画項目の上限であり、本文の上限ではありません。本文約4,000字・台詞約2,000字は章計画で場面へ配分します。

回答枠の既定は `script-allocation` 2,048、`script-plan` 6,144 tokensです。`script-scene` は場面の目標量・話者IDと改行・同モデルの既出場面の実測から見積もります。最小2,048、本文1文字あたり1.25 tokens、ラベル1文字あたり0.5 tokens、1行40文字、固定128 tokens、安全係数1.2、256 tokens刻みを初期値とし、`script_options.scene_tokens` で変更できます。これらの係数は経験的な初期値です。短い実出力を理由に分量目標や見積もりを引き下げず、実測比率が大きければ回答枠を増やします。

場面の既定profile上限は8,192ですが固定上限ではありません。`profiles.script_writer.max_tokens` または `profiles.script-scene.max_tokens` とworker/modelの `max_output_tokens` を許容範囲内で設定すれば、それ以上も使えます。個別purposeの設定はカテゴリより優先します。見積もりがprofile上限に制限された場合は `limited_by_profile` を記録します。入力に収めるために回答枠を自動縮小しません。見積もり・利用した実測・採用値は生成前に保存し、再開時にも同じ値を使用します。続筆は最大1回で、途中原稿・接続文字列・grammarを含めて再計測します。

同じ回答枠を送信前の実tokenizer計測と実要求に使用します。`scene_output_budgets` と各要求の `output_budget`、`selection.budget` で確認できます。profileのmax_tokensは許容上限、要求のmax_tokensは実際の予約です。回答枠の削減をVRAM削減率とはみなしません。コンテキストは16k固定ではなく設定で32k・64kへ拡張できますが、`allow_context_expansion` は許容最大容量でサーバーを起動する方式です。16k分のVRAMだけで開始して動的に増える方式ではありません。設定を変える場合は新規実験にします。

途中で切れた構造化計画は未完了として停止し、同じ保存先の再開で追加生成せず、新しい実験の回答予算を見直します。

`report.json.planning_metrics` に会話項目数と文字数を記録します。送信前の容量不足は `preflight_failures` に入力・回答予約・余裕・上限とともに保存し、LLMの修正回数を消費しません。受信した構造不備は `requests/*.validation.json` に応答と具体的な例外を保存し、修正は最大1回。送信済み・送信不明の要求と使用量は再開時も保持します。

主な保存物：

- `story.md` / `story.html`：全章を同じ人物ID付き台本のまま表示する通読用出力。
- `outline.json`、`chapters/*.plan.json`、`notes/`：全体構成・当章計画・引き継ぎ。
- `sources/`：執筆直後と発話分離後の台本。演出工程で停止しても残ります。
- `exports/chapter-NNN/`：`narrative.json`、`script.json`、`asset-requirements.json`、検証済み `tyrano-source.zip`、展開済み `player/<hash>/`。本来の画像・音声要求とテスト用代替を区別します。
- `experiment.json`、`draft-state.json`、`requests/`、`llm/`、`report.json`：実験同一性、場面ごとの途中再開、入力と回答、使用量、停止理由。意味的な合格を捏造しません。

技術的な再試行は各工程・各演出バッチで最大1回、出力上限からの続筆も1回です。再開で回数や使用量をリセットせず、完了した台本・演出・章を再生成しません。同じモデルとコンテキストの工程は同一実行内で保持し、起動・解放と切り替えを別に記録します。画像推論と音声合成は実行しません。

保存済み資料を使った容量診断には次の専用スクリプトを使います。旧設定と一致するGemma16kを起動し、tokenizerだけで既存の全場面と第2・3章計画を測ります。保存済み計画・回答枠を優先し、作例の採否も記録します。物語は生成せず、既存実験は書き換えません。`--output-dir` は存在しない新規ディレクトリにします。[実装時の計測結果](../experiments/script-continuation-r14-implementation-20260926.md)を参照してください。

```powershell
.\.venv\Scripts\python.exe -I -u -X utf8 .\scripts\story\check_script_context.py `
  --input .\private\story-inputs\eclipse-m3-approved-20260926.json `
  --baseline .\outputs\script-eclipse-seed1-r13-full-20260926 `
  --output-dir .\outputs\script-r14-context-probe-new
```

作劇の局所比較には`probe_story_craft.py`を使います。成立根拠の有無、章計画の指示、本文作例の有無を各1回・最大6要求で比較し、失敗・打ち切り時はその場で停止します。保存済みr12のキャスト・プロット・第1場面を固定した診断であり、新プロットからの通し生成ではありません。比較対象の本文だけを出し、画像・音声・演出生成は行いません。

```powershell
.\.venv\Scripts\python.exe -I -u -X utf8 .\scripts\story\probe_story_craft.py `
  --input .\private\story-inputs\eclipse-m3-approved-20260926.json `
  --baseline .\outputs\script-eclipse-seed1-r12-full-20260926 `
  --output-dir .\outputs\script-r13-craft-probe-new
```

章境界だけの局所検証には`probe_script_boundary.py`を使います。保存済み第1章までの材料で、第2章計画と冒頭1場面を各1回生成します。旧プロットを固定するので、新しいプロット設計や3章全体の品質の検証にはなりません。打ち切り・構造エラー等ではその場で停止します。画像・音声・演出は生成しません。

```powershell
.\.venv\Scripts\python.exe -I -u -X utf8 .\scripts\story\probe_script_boundary.py `
  --input .\private\story-inputs\eclipse-m3-approved-20260926.json `
  --baseline .\outputs\script-eclipse-seed1-r13-full-20260926 `
  --output-dir .\outputs\script-r14-boundary-probe-new
```

## 撤回済み試行の履歴（story_draft_v1）

以下は撤回した小説生成経路の再現用記録です。次の台本検証には使いません。この実装ではQwenが構成と依頼、Gemmaが小説本文を作り、次章へ引き継ぎます。3章の場合は通常計6回の生成ですが、必要な台本処理を維持した新計画の回数目標ではありません。

```powershell
.\services\worker\.venv\Scripts\python.exe -I -u -X utf8 .\scripts\story\check_text.py `
  --input .\tests\fixtures\story-debug-relationship.json `
  --output-dir .\outputs\story-draft-relationship-seed1 `
  --workflow causal --policy story_draft_v1 --seed 1
```

まず第1章だけ確認する場合は、同じコマンドに `--chapter-limit 1` を加えます。承認済みの章数や全体構成は変えません。続きは同じ入力・設定・出力先で `--resume` を指定し、章数上限を外すか増やします。

```powershell
.\services\worker\.venv\Scripts\python.exe -I -u -X utf8 .\scripts\story\check_text.py `
  --input .\tests\fixtures\story-debug-relationship.json `
  --output-dir .\outputs\story-draft-relationship-seed1 `
  --workflow causal --policy story_draft_v1 --seed 1 --resume
```

この経路には途中のLLM合否レビュー、詳細なシーン・人物・場所の登録、引用IDや実績台帳との照合を置きません。章本文は通常の日本語で保存し、予定からずれた場合も実本文の結果を次章へ渡します。任意の引き継ぎメモを取得できなければ、その旨を記録し、前章本文と全体構成から続けます。通信・起動・保存の失敗、必須出力が空、入力容量や使用量の上限など、生成を進められない場合に停止します。技術的な再試行は1工程につき最大1回、出力上限による続筆も保存済み原稿の直後から最大1回です。

Qwenの全体構成はlow、章間の引き継ぎはnone、Gemmaの執筆はreasoningなしです。初期設定はQwen32k／Gemma16kで、各モデルの入力と回答枠を実トークン数で計測します。原文が収まる間は全文を渡し、容量が足りなくなった場合だけ古い章をメモへ置き換えます。さらに必要なら古いメモを省き、直前章は終盤を優先します。保持・省略した原文の範囲を記録し、必須設定や執筆依頼、最近の本文と回答枠を確保できなければ容量不足として停止します。

モデルごとの変更には入力の `profiles.draft_outline`、`profiles.draft_handoff`、`profiles.draft_writer` を使います。保存jobを入力にする場合は `payload.profiles` です。例えば現在の初期サイズを明示する指定は次のとおりです。

```json
"profiles": {
  "draft_outline": {"context_size": 32768},
  "draft_handoff": {"context_size": 32768},
  "draft_writer": {"context_size": 16384}
}
```

拡張時は `config/m2-generation.json` の各モデルの許可上限・確認済み上限も合わせて設定します。Qwen側は `model_routing.review.llm`、Gemma側は `llm` です。全体の `profile` とCLIの `--context-size` はGemma側に適用されます。同じ出力先で設定を変えて再開せず、新しい実験として開始します。コンテキストを16kに固定する実装ではありません。

保存できた章から `story.md` / `story.html` を更新します。これらは下書きの通読用であり、意味検査済み・公開済みの作品ではありません。章ごとの本文、構成と引き継ぎメモ、`experiment.json`、入出力ログ、コンテキストの選択記録も残します。`report.json` では保存章数・停止理由・時間・トークン・モデル切り替えを確認できます。再開時は実験設定と保存本文の一致を確認し、完了済み章を再生成せず、使用量の上限をリセットしません。通常の3章実験は5回のモデル切り替えを伴います。

この旧経路の単体テストと、小説の第1章の生成・保存は実施しました（[訂正付きの確認記録](../experiments/story-draft-smoke-20260925.md)）。台本生成やティラノ出力の成功ではなく、章間品質も未評価です。以下の既存経路の説明にある採用検査・詳細台帳・`result.zip` を使わないという旧実装の事実は、新計画で既存機能を外す理由にはしません。

## 実行

リポジトリのルートから、worker の Python を使います。

```powershell
.\services\worker\.venv\Scripts\python.exe -I -u -X utf8 .\scripts\story\check_text.py `
  --input .\tmp\story-input.json `
  --output-dir .\outputs\story-causal-seed1 `
  --workflow causal --seed 1
```

入力は次のいずれかです。UTF-8（BOM あり・なし）で保存します。

- 承認スナップショット JSON：`world.result`、`characters[].result`、`relationships.result` を持つ承認済み設定。画像・音声の参照が残っていても、それらのファイルは読み込みません。world / character の結果を `result` で囲まないテキスト専用形式も受け付けます。
- 保存済み `m3_narrative` job の `job-request.json`：`payload.approval_snapshot` と `payload.profile`、`payload.seed` を取り出します。実験は独立した第1章から始めるため、元の job の章番号や前章成果物は引き継ぎません。

`--workflow legacy` は従来の M4 本文生成経路で baseline を作ります。既定は `causal` です。別モードの比較は別の出力ディレクトリを使い、同じ承認設定と seed を指定してください。サブキャラや場所を減らす特別な簡略プロンプトは使いません。

Qwenの章構成・章末編集とGemmaの連続執筆を試す場合は、独立した出力ディレクトリで `--workflow causal --policy chapter_editor_v1` を指定します。最初は `--chapter-limit 1` などで少数章を検証してください。`--policy` の有無はprotocol・再開時照合・予算の識別に含まれ、従来の結果を新方式として再利用しません。通常UIの制作経路には適用されません。新方式の場面レビューには「章末へ委譲」と記録し、採用には原文と記憶に結び付いた章末レビューを要求します。

```powershell
.\services\worker\.venv\Scripts\python.exe -I -u -X utf8 .\scripts\story\check_text.py `
  --input .\tests\fixtures\story-debug-relationship.json `
  --output-dir .\outputs\relationship-editor-seed1 `
  --workflow causal --policy chapter_editor_v1 --seed 1 --chapter-limit 1
```

新方式はQwen側を32k、Gemma本文を16kで開始します。Qwenの構成にはlow、章末編集にはnoneを既定にし、プロファイル上書きで比較できます。出力が途中で切れた場合は同じ長い依頼を自動反復せず、保存済み下書きと停止理由を確認します。章規模での実モデル回答取得率・作品品質は未確認です。

章全体の原文がQwenの入力枠を超えた場合は、各場面の原文を個別に検査し、その対象hashと実績記録から章全体を判定します。各場面の検査が揃わなければ章は採用しません。実モデルの短いJSON応答は確認済みですが、実際の章生成と通し品質評価はまだ行っていません。

現在の通常UIは従来方式のままです。新方式はこのCLIによる本文PoCとして提供し、通読評価を経てから通常制作・素材生成へ統合します。進捗と失敗を含む記録は[PoC記録](../experiments/story-workflow-poc.md)を参照してください。

新方式の目的別profileには `story_blueprint`、`chapter_intent`、`scene_planning`、`story_extraction`、`information_extraction`、`continuity_review` を使えます。既存の `scene_text`、`staging`、`quality_review` も使用します。保存済みjobを入力にすると `payload.profile` と `payload.profiles` を両方引き継ぎます。新方式の既定回答枠は状態抽出が4,096、情報開示抽出が6,144 tokens（workerの出力上限以内）で、それ以外は生成設定に従います。明示したprofileの値は既定より優先します。実際に使用した値は各requestとtraceへ保存します。

### コンテキストの拡張と計測

`--context-size` は実験の `profile.context_size` を指定します。未指定なら入力jobのprofile、次いで `config/m2-generation.json` の `llm.context_size` に従います。現在の設定は16,384で維持しています。固定上限ではありませんが、CLIの指定だけでworkerが許可する範囲やモデルの確認済み範囲を超えることはできません。

例えば、使用するモデル・ランタイムと手元のメモリで32,768まで利用できることを確認した環境では、同ファイルの `llm` 内に次の設定を置けます。他の既存項目は保持します。以下は設定例であり、この手順書の追加で実際の設定は変更していません。

```json
{
  "context_size": 16384,
  "max_context_size": 32768,
  "model_context_size": 32768,
  "allow_context_expansion": false,
  "context_margin_tokens": 512
}
```

| 項目 | 意味 |
| --- | --- |
| `context_size` | 通常の目標サイズ。章jobの起動前に `--context-size 32768` などで明示的に変更できる |
| `max_context_size` | このworkerで許可する資源上の上限。job側のprofileからworkerの上限を引き上げることはできない |
| `model_context_size` | 使用するモデル・ランタイムで確認した上限。モデル名だけから大きな値を推定しない |
| `allow_context_expansion` | `true` なら目標サイズを超えるリクエストでも許可上限内で入力予算を拡張する。workerが `false` の場合、jobだけで自動拡張を有効にはできない |
| `context_margin_tokens` | 入力と回答枠に加えて確保する余裕。既定は512 |

上の例は自動拡張を無効にしたまま、必要な実験で `--context-size 32768` を明示して起動する設定です。`max_context_size` と `model_context_size` を省略すると、どちらも設定済みの `context_size` になります。したがって現在の初期設定のまま32kを指定すると、モデル起動前に範囲外として拒否します。回答の `max_tokens` はコンテキストと別の設定であり、コンテキストを広げただけで回答枠を自動的に増やしません。

`allow_context_expansion: true` は、必要になった瞬間にモデルを再起動する機能ではありません。job開始時に許可上限のコンテキストを持つサーバーを起動し、各リクエストの実測量に応じて入力予算を目標サイズから上限へ広げます。そのため大きいコンテキストのメモリ確保は起動時から必要です。最初は明示指定で比較し、環境に合った上限を決めてから自動拡張を有効にできます。設定やprofileを変えた比較実験は、新しい出力ディレクトリで開始します。

実行中はサーバーの `/props` からスロットごとの実コンテキスト長を確認し、workerの許可上限・モデルの確認済み上限との小さい方を使用します。古いサーバーでこのAPIまたはコンテキスト長の項目が未提供の場合は、自分で渡した起動設定を根拠にし、その区別をtraceの `server_context.source` に記録します。接続エラーや不正な値は黙認しません。LLMへの各新規リクエストは `/apply-template` と `/tokenize` でschema・toolsを含む実際の入力を測り、`入力 + 回答枠 + 余裕` が収まるか検査します。

`input_token_limit` は予算の表示であり、原文をその長さに切って渡す指定ではありません。入力超過時に `max_input` 相当の長さで採用本文・根拠・必要な知識を黙って切り捨てたり、回答枠を勝手に縮めたりしません。許可された拡張でも収まらない工程は予算エラーとして止まり、`last-context-budget.json` に値を残します。工程側での分割・取得範囲の改善が必要なケースと、拡張設定で解決できるケースを分けて判断します。

ロード時間の比較には、各章の `jobs/chapter-NNN/llm-v*/llm-metrics.json` を使います。`model_load.elapsed_seconds` はサーバー起動から準備完了まで、`model_release.elapsed_seconds` は解放、`llm_generation` は工程ごとのHTTP生成時間・cache利用・トークン情報です。対応する応答には `server_timings` の入力処理時間とトークン生成時間も記録します。章job全体の実時間は `report.json` の `attempts[].metrics.elapsed_seconds` にあり、GPUロック待ち等も含みます。これらの時間には包含関係があるため、全項目を単純に足して総時間にしません。

同じjobの中ではモデルを維持し、次の章ではロード・解放が再び発生します。失敗時もLLMの計測ファイルと `generation-trace.json` が残るため、検査や修復に費やした時間を調べられます。保存済み `result.zip` を再利用した試行では、応答に含まれるtraceは元の生成時の記録です。現在の試行で新たにロードした時間とは区別してください。

`--chapter-limit 1` はまず第1章を読むための実験上限です。承認済みの `chapterCount` と全体構成は書き換えません。後で `--resume` を付け、上限を外すか増やせば、採用済みの章から続きを生成します。

### 独立した評価用設定で試す

[story-debug-relationship.json](../../tests/fixtures/story-debug-relationship.json) は、既存作品を流用せず用意した三章・メイン二人の設定です。町の写真展を閉館までに仕上げる共同作業を題材とし、二人は三年前からの仕事仲間です。初期の状況と関係だけを固定し、各章の解決や最終的な展示の形は指定していません。画像・音声ファイルは不要です。

```powershell
.\services\worker\.venv\Scripts\python.exe -I -u -X utf8 .\scripts\story\check_text.py `
  --input .\tests\fixtures\story-debug-relationship.json `
  --output-dir .\outputs\relationship-causal-seed1 `
  --workflow causal --seed 1
```

このfixtureは評価入力であり、合格済みの本文サンプルではありません。本文を読んで、前章の判断が次章の作業や関係に影響するか、残り時間や配置・説明札の状態が継続するか、既知の関係を初対面へ戻していないかを確認します。

題材を変えた評価用設定も同じ入力形式で使えます。いずれも既存作品から切り出したものではなく、初期条件を与えた独立した入力です。章ごとの展開や答え、到達すべき結末は指定していません。

| 入力 | 規模・題材 | 通読時に確認する点 |
| --- | --- | --- |
| [story-debug-relationship.json](../../tests/fixtures/story-debug-relationship.json) | 三章・二人、写真展の共同作業 | 既知の関係と作業の進み具合を引き継ぎ、毎章同じ衝突と和解に戻らないか |
| [story-debug-mystery.json](../../tests/fixtures/story-debug-mystery.json) | 三章・二人、資料館の封筒と記録の調査 | 原因の推測と確認済み事実を分けるか。片方だけが知る記録漏れを、もう片方が説明前から知っていないか。回想を用いた場合、過去の負傷を現在へ戻さないか。寄託者への連絡の約束が放置されないか |
| [story-debug-journey.json](../../tests/fixtures/story-debug-journey.json) | 五章・二人、楽器箱を運ぶ旅 | 場所・移動時間・荷物の持ち主・疲れや靴ずれを引き継ぐか。休息や手当の結果を次章で忘れないか。土地の情報は得た後に知識となるか。道中で交わした約束と人物の登場・退場に理由があるか |

例えば調査劇を試す場合は、上のコマンドの `--input` を `tests/fixtures/story-debug-mystery.json`、`--output-dir` を `outputs/mystery-causal-seed1` へ変更します。旅の設定には `story-debug-journey.json` と専用の出力ディレクトリを使います。各作品の本文を最後まで読む評価と、`author-notes.md` の原文根拠・台帳・検査の照合を併用し、検査が通ったという理由だけで品質を合格にしません。

### UIで承認済みの設定を取り出す

現行UIには承認snapshotだけを保存する専用ボタンはありません。既に承認した作品なら、起動中のcoordinatorへ次の読み取りAPIを使って取得できます。まず作品一覧から、UIで表示している作品名に対応する `id` を選びます。

```powershell
$storyCoordinator = 'http://127.0.0.1:8000'
(Invoke-RestMethod -Uri "$storyCoordinator/api/m2/projects").projects | Format-Table id, title
$storyProjectId = '一覧に表示された作品ID'
$storyDetail = Invoke-RestMethod -Uri "$storyCoordinator/api/m2/projects/$storyProjectId"
$storyApprovalId = $storyDetail.draft.approval.artifactId
if (-not $storyApprovalId) { throw 'この作品には承認済みスナップショットがありません。' }
New-Item -ItemType Directory -Path .\tmp -Force | Out-Null
Invoke-WebRequest -UseBasicParsing `
  -Uri "$storyCoordinator/api/artifacts/$storyApprovalId/content" `
  -OutFile .\tmp\story-input.json
```

JSON本体を直接ファイルに保存するため、PowerShellの文字列変換で本文が変わりません。これらは取得のみの操作で、承認や制作開始を実行しません。未承認の作品を実験するためだけにUIの承認ボタンを押すと通常制作も始まるので、その場合は独立したテキスト設定を入力に使います。取得後のCLI実験は別の出力ディレクトリで第1章から始まり、元作品の本文やDBは変更しません。

## 出力と再開

| ファイル | 内容 |
| --- | --- |
| `story.html` / `story.md` | 採用済み本文だけの通読用資料。章・シーン境界を表示し、作者用の秘密・未来の計画を混ぜない |
| `author-notes.md` | 各章の計画、実績・状態、検査と trace、profile / seed 等。先の展開を含む |
| `experiment.json` | 実行モード、承認スナップショット、seed、profile、入力 fingerprint。DB の制作系列とは独立 |
| `chapters/chapter-NNN.json` | 章ごとの採用記録。通常検査に合格した result、内容ハッシュ、前章参照、終了状態ハッシュ |
| `report.json` | `text_verified` / `chapter_limit_reached` / `failed` 等の状態、試行ごとの時間・トークン・cache 指標 |
| `jobs/chapter-NNN/` | 通常 worker の request cache、応答、ログ。失敗時は `draft.md` の冒頭に保存済み未採用本文（変換前後）、後段に検査応答をまとめる |

```powershell
.\services\worker\.venv\Scripts\python.exe -I -u -X utf8 .\scripts\story\check_text.py `
  --input .\tmp\story-input.json `
  --output-dir .\outputs\story-causal-seed1 `
  --workflow causal --seed 1 --resume
```

再開時は承認設定・seed・workflow・profile・生成設定の一致を確認し、採用済みの全章の内容と lineage を検証します。context を変える場合も新しい出力ディレクトリを使います。失敗した章の request cache は通常 worker の再試行処理へ戻し、採用済みの章は生成し直しません。中断後に未採用の応答が残っていても、章の採用記録と検査を通るまで次章には渡しません。同じ出力ディレクトリでの同時実行は拒否します。

`text_verified` は生成処理の本文検査が終了した意味です。作品の公開完了や、人による作劇品質の合格判定を表しません。`chapter_limit_reached` は途中までの実験終了であり、作品全体の完結ではありません。

## 再開してもリセットしない使用量予算

`causal` の実験には、章ごとの呼出・修復回数に加え、章と作品全体のトークン・稼働時間の上限があります。`legacy` は比較用の旧経路なので、この新しい使用量予算の対象外です。

| `workflow_limits` の項目 | 既定 | 対象 |
| --- | --- | --- |
| `max_calls` | 160 | 1章の新規LLM呼出回数 |
| `max_repairs` | 従来6、新編集方式2 | 1章の修復回数。新編集方式は本文系1回・記録訂正1回を個別にも制限 |
| `max_tokens` | 4,000,000 | 1章の実測入力＋出力トークン |
| `max_elapsed_seconds` | 14,400（4時間） | 1章の累積稼働時間 |
| `max_story_tokens` | 有効な章トークン上限×承認章数 | 作品全体の実測入力＋出力トークン |
| `max_story_elapsed_seconds` | 有効な章時間上限×承認章数 | 作品全体の累積稼働時間 |

既定は16k・3章のPoCで修復を観察できる余裕を持たせています。値は正の整数で、入力JSONの `workflow_limits` に指定します。保存jobを入力にする場合は `payload.workflow_limits` です。承認設定は同じJSONの `approval_snapshot` に置けます。例えば3章の試行で上限を明示する部分は次のとおりです。

```json
"workflow_limits": {
  "max_calls": 160,
  "max_repairs": 6,
  "max_tokens": 4000000,
  "max_elapsed_seconds": 14400,
  "max_story_tokens": 12000000,
  "max_story_elapsed_seconds": 43200
}
```

指定値は実験の入力fingerprintに含まれ、全章へ渡されます。同じ出力ディレクトリで値を変えて再開することはできません。上限を変える場合は新しい実験にします。`max_tokens` は累積使用量の予算で、生成profileの `max_tokens`（1応答の回答枠）とは別です。

`jobs/causal-resource-budgets/<系列のhash>.json` が作品共通の使用量記録です。LLM呼出の直前にpending記録を保存し、成功・失敗のいずれでも計測できたトークンと所要時間を保存します。章内の `causal-stages/budget.json` は従来の呼出・修復回数と使用量のスナップショットを持ちます。最新の作品集計は共通記録と `report.json.resource_budget` を参照してください。

request cache・stage cache・完成済みbundleの再利用で、元の生成トークンや元の時間を再度加算しません。デバッグ実行の時間予算は各章jobの実行区間を計り、ロード・GPU待ち・解放を含めます。その内側のLLM呼出時間は重複加算しません。cache読込のために今回実際に使った時間は計上します。停止から再開までの放置時間は課金対象にしません。通常のworkerから直接実行する場合は、正確に計測できるLLM呼出区間の時間を使用します。

上限は新規呼出前と応答後、章の採用前に検査します。進行中のHTTP要求を上限到達の瞬間に強制終了する方式ではなく、既存timeoutを使用します。1要求の完了によって上限を超えた場合は、超過分も記録したうえで応答・章を採用せず停止します。

HTTP障害で応答usageを得られなかった場合や、プロセス強制終了でpendingが残った場合は、未計測のトークン・時間を0とみなして再開しません。`unmeasured_requests` / `unfinished_attempts` に未確定記録を示し、成功採用を止めます。表示上のトークン合計は判明分の小計であり、このリストが空でなければ完全な実測値ではありません。旧protocolの実験は再開時に拒否し、呼出数だけある旧budgetも計測済みとは扱いません。使用量不明を無言で移行・初期化する処理はありません。

## 計測の読み方

`elapsed_seconds` は job 全体の実時間で、GPU ロック待ち・サーバーロード・推論・検査を含みます。保存済み response の `usage` が存在する場合だけトークン数を集計し、未計測値は `null` とします。`*_new` はその試行で新しく保存した応答、`*_saved` は失敗試行や再利用 cache も含む保存済み応答の集計です。cache 数を新規推論回数と同一視しません。LLM が記録する時間 trace と retry seed 区間も保持します。

章間のモデル再ロードは残ります。実時間の内訳を測り、ロードが支配的ならモデル常駐化を別途検討します。画像の同一性、音声品質、画像切替、鑑賞・保存再開はこのモードでは評価できません。[再設計計画](../story-workflow-redesign.md)の本文評価後に、素材を含む別の受け入れ確認を行います。

## GemmaとQwenの使い分け

causal implementation 14以降は、`config/m2-generation.json` の `model_routing.review` で検査モデルを指定します。本文生成・修復・計画・抽出・演出はGemma（思考なし、16,384）、意味検査はQwen3.8 27B（low、32,768、総出力上限8,192）です。`config/qwen-review-llm.json` はユーザー提供GGUFのローカルhash、template hash、確認済みサーバーbuildを記録しています。

全体の `profile` およびCLIの `--context-size` はGemma側の指定です。Qwenの設定を変更する場合は、入力JSONの `profiles.continuity_review` / `profiles.quality_review`、または具体的な検査stageへ指定します。例えば両方に `{"reasoning_level":"medium"}` を指定すれば、検査をmediumで試せます。xhighは受け付けず、不合格を理由とする自動effort変更も行いません。Qwenの総出力上限は思考と最終回答の合計であり、lowでも打ち切りがなくなる保証はありません。

モデルごとに入力・出力枠を実tokenizerで測り、入力＋総出力枠＋余裕が収まるか確認します。今回の設定はGemma16k／Qwen32kですが、コードはモデル別の `max_context_size` / `model_context_size` / `allow_context_expansion` による拡張にも対応します。拡張する際はモデル適合性とVRAMを確認し、設定を更新した新しい実験を使います。

同一モデルの連続呼出はプロセスを再利用します。切替時は一方を解放してから他方をロードし、二モデルを同時常駐させません。章の終了時には解放します。`report.json` の試行別 `metrics.models_new` にモデル別の新規応答数・token数、`model_loads_this_run` / `model_switches_this_run` にロード回数・切替回数を記録します。最初のロードは切替回数に含めません。`llm-v7/llm-metrics.json` に失敗・最終解放を含む統合traceを保存します。

両モデルの設定内容がjobと実験の識別に含まれます。同じ設定ファイル名でもモデル・template・検査profileが変われば以前のcacheは再利用しません。過去の実験はそのまま保存し、implementation 14は新しい出力ディレクトリで検証します。
