# 作品の変更履歴と復元

履歴は、操作単位の状態スナップショットと不変の成果物・公開buildへの参照で構成する。復元は既存データの選択を切り替える処理であり、生成ジョブの再投入、素材の複製、章の再コンパイルを行わない。

主な実装は [`project_history.py`](../../services/coordinator/project_history.py)、APIは [`m2_routes.py`](../../services/coordinator/m2_routes.py)、テーブル定義は [`005_project_history.sql`](../../services/coordinator/migrations/005_project_history.sql) にある。利用手順は [変更履歴・元に戻す](../setup/history.md) を参照。

## 保存構造

| テーブル | 用途 |
|---|---|
| `project_revision` | 作品内で単調増加する番号、操作名、操作種別、状態JSON、復元元の履歴ID、保存日時を追記する。既存行の更新はトリガーで拒否する。 |
| `project_history_state` | 現在の履歴ID、競合検出用version、進行中の操作、選択中の制作・build・表示設定成果物、凍結した制作状態を保持する。 |

スナップショットは `schema_version`、M2の `state`、M3の `production_state`、選択先の `selection` からなる。

- `state` は入力・採用設定・関係性・固定状態・承認・素材参照などを含む。実行用の `queue` は空、`activeJobId` はnullとして保存する。
- `production_state` は採用本文の成果物ID、エラー、ジョブ状況の記録、必要素材と成果物の対応を保持する。ジョブ記録は過去の状態を表示するための情報であり、実行ジョブの巻き戻しには使わない。
- `selection` は `production_id`、`build_id`、`portrait_settings_id` を保持する。公開済みbuildと表示設定も作品全体の復元対象に含める。

画像、音声、原文、脚本、ZIPは既存のartifact/buildを参照する。履歴登録・復元による成果物ファイルのコピーはなく、保存容量の増加は状態JSONと参照情報が中心となる。通常の生成・リテイク・rebuildで作成した新しい成果物は、その処理自体の出力として別に保持する。

初回の `ensure()` で現在の状態をbaselineとして保存する。既存作品の導入前の操作履歴は推定しない。移行は制御サーバーの再起動で読み込み、baselineは履歴利用・操作開始時に必要に応じて作る。

## 操作のまとまり

`begin()` と `finish()` で、一回のユーザー操作と、その操作から続く生成パイプラインをまとめる。直接編集は反映後、生成を伴う処理は操作全体の完了または失敗後に履歴を追加する。内容に変化がない操作と画面移動だけでは履歴を増やさない。

M2とM3は独立した操作グループを持つ。本編制作中に別の設定操作が完了しても、本編の操作を途中で閉じない。反対に、素材一件の結果を採用して次のジョブを登録するまでの短い空白も、操作の完了とはみなさない。M2のキュー、M3の公開build・失敗状態を含めて判定し、ジョブごとの進捗履歴を作らない。

操作前の状態が直近の履歴と異なる場合は、その状態もcheckpointとして保存する。比較ではUIの表示ステップと `draft.revision` を除外する。

## 復元と競合保護

復元は一つのDBトランザクションで以下を行う。

1. 履歴の `version` と現在の `draft.revision` を照合する。作品にpending/runningのジョブがあれば拒否する。
2. 対象履歴が同じ作品に属すること、参照先の制作・build・成果物が有効であること、必要な成果物を読み出せることを検証する。
3. 復元直前の保存済み状態が履歴に未収録ならcheckpointを追加する。
4. 設定と成果物の参照を復元し、実行キューとactiveJobを空にする。`draft.revision` は過去の値を使わず、現在値から1増やす。
5. 選択中の制作・build・表示設定を切り替え、制作状態を凍結する。復元操作も `restored_from_id` 付きの新しい履歴として追記する。

履歴は線形で、復元先より後の行も保持する。その後の状態へ再度復元できる。公開buildと成果物を削除・上書きしないため、旧版の鑑賞URLとZIPも維持される。必要な参照が不足している場合は復元を拒否し、現在の状態を保持する。

未完成の制作を復元した場合も、状態取得や自動復旧処理だけでジョブを再投入しない。`production_frozen` と `production_snapshot` により復元時点の本文・素材対応・進捗を表示する。同じ制作の本文・必要素材がその後変更されている場合は、古い途中状態からの再開を拒否し、新しい履歴を選ぶよう案内する。

M3の作品状態APIは `production.history_frozen` を返す。公開済みbuildがない凍結状態を、UIは動作中の制作として表示しない。完成済みbuildを選択している場合は通常の鑑賞・書き出しを利用できる。

## API

### 履歴一覧

`GET /api/m2/projects/{project_id}/history`

```json
{
  "project_id": "project-id",
  "version": 7,
  "current_revision_id": "revision-2",
  "busy": false,
  "entries": [
    {
      "id": "revision-2",
      "number": 2,
      "label": "立ち絵の表示を調整",
      "created_at": "2026-09-24T01:10:00+00:00",
      "restored_from_id": null,
      "current": true
    }
  ],
  "pending_operation": null
}
```

`entries` は新しい番号から返す。`pending_operation` がある場合は `label` と `status` を持つ。UIのstatus契約は `pending` / `running` / `failed`。進行中の複数グループはラベルをまとめて表示し、失敗して終了した操作は失敗を含む履歴名として記録する。`busy` は作品全体の実行待ち・実行中ジョブに基づく。

### 復元

`POST /api/m2/projects/{project_id}/history/{revision_id}/restore`

```json
{
  "expected_version": 7,
  "expected_revision": 18
}
```

両方とも1以上の整数を要求する。成功時は通常のM2作品詳細と同じ `M2Detail`（`project` / `draft` / `jobs`）を返す。生成ジョブは登録しない。

| ステータス | 主な条件 |
|---|---|
| `409` | 作品または履歴の版が変化した、生成ジョブがpending/runningで復元できない。 |
| `404` | 作品に属する履歴など、指定した対象が存在しない。 |
| `422` | 必要な設定・成果物を検証できない、または要求形式が不正。 |

## フロントエンド

全STEP下部の `ChangeHistory` は初期状態で折りたたむ。開いている間だけ一覧を取得・定期更新し、設定変更と立ち絵表示の保存を反映する。確認時の履歴versionとdraft revisionを固定し、確認中に更新された場合は再確認を求める。未保存入力や処理中は復元を無効にする。

`useM2Api.restore()` は通常のmutation保護を通し、実行中に届く古いpoll結果を反映しない。成功した復元だけで `restoreEpoch` を増やし、`ProductionPanel` を再マウントする。通常のpollでは再マウントせず、編集中の立ち絵枠を保持する。APIエラー時は確認対象と入力を残す。

履歴の分岐、個別素材だけの復元、STEP2での作品複製はこの設計に含めない。
