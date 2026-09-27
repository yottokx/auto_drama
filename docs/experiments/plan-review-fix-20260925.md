# 計画レビューと本文レビューの分離（2026-09-25）

## 修正対象

通し生成 `outputs/relationship-editor-seed1-full-20260925` は、構成案レビューが未執筆の第1〜3章の本文を要求し、`insufficient_evidence` で停止した。その資料不足を構成案の欠陥として再生成したため、同じ停止を繰り返し、意味修復予算も消費した。

- 全体構成・全体構成の改訂・章計画・場面計画の4経路に、計画専用のsystemと検査規則を適用。予定された因果・制約・進展を評価し、未執筆本文を要求しない。
- 採用済み過去の原文取得は維持。計画内の問題は章番号・項目・具体的な食い違いで説明し、本文の発話IDを創作しない。
- 構成案の再生成は、資料不足がなく、指摘先が構成案である具体的な `fail` に限定。`insufficient_evidence`、`context`、不足資料を併記した `fail` では再生成しない。
- 章計画から原因のある全体構成へ戻す既存経路、本文の根拠検査、修復回数制限を維持。
- 検査規則とsystemを段階キャッシュ識別に含め、両causalプロトコルを更新。旧実験の出力先は再開互換ではないため、次の実験には新しい出力先を使う。既存成果物は変更しない。

## 自動テスト

`tests/unit/test_causal_plan_reviews.py` に17件の回帰テストを追加。検査を常に合格にするmockではなく、実際の `_gate` → `_structured` → メッセージ生成を通して指示・資料取得・修復経路を確認する。LLMの意味判断自体は模擬応答であり、以下の実モデル確認と区別する。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/unit -q --tb=short --basetemp tmp/pytest-plan-review-final-20260925
.\.venv\Scripts\ruff.exe check services/worker/generation/causal_narrative.py services/worker/generation/workflow_version.py tests/unit/test_causal_plan_reviews.py tests/unit/test_causal_narrative.py
```

- 単体テスト: **1,043 passed**。既存ライブラリの非推奨警告2件。
- 変更したPythonファイルのRuff: **All checks passed**。

## Qwenによる小規模確認

本番と同じ構成案レビュー経路をQwenの `reasoning_effort=none` で呼び、保存された初稿・再生成稿・単純な正常対照案だけを検査した。構成案の生成・修復、Gemmaの本文生成、章の採用は行っていない。

| 入力 | 初回 | 時系列照合指示の補強後 | 評価 |
|---|---|---|---|
| 保存された初稿 | pass | pass | 未執筆本文の要求は解消 |
| 保存された再生成稿 | pass | pass | 「第2章で動かせない→第3章で移動」の問題を見逃した |
| 単純な正常対照案 | pass | pass | 本文なしでも計画として合格可能 |

初回は計3リクエスト、28.204秒。補強後は計3リクエスト、25.406秒。いずれもロード・解放を含み、構造化再試行と意味修復は0回、全応答の終了理由は `stop`。

補強後もQwenは再生成稿の物理的制約を人物の判断・合意として読み替えて合格とした。**停止を招いた検査段階の混同は改善したが、構成上の矛盾の検出精度は未解決。** 今回の少数例から、通し生成の成功や創作品質を保証しない。

記録:

- `outputs/plan-review-fix-smoke-20260925/report.json`
- `outputs/plan-review-fix-smoke-20260925-r2/report.json`
- 各ディレクトリの `*-input.json` と `llm/` に入力・応答・計測を保存。
- 検査用スクリプト: `tmp/probe_plan_review_fix.py`。出力先が既存の場合は上書きせず停止する。

ユーザーの指示どおり、通し生成は実施せず、コード修正と小規模確認までで終了。
