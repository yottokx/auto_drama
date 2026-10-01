# Gemma 31B medium: 第3章の場所参照エラー

確認日: 2026-09-29。対象作品「鉄と血の叙事詩」。

## 分類

直接原因は、モデルの第3章計画が使用場所の定義を落としたこと。加えて、その直前の初回応答には、ハーネスの人物IDと場面IDを混同しやすい指示が誤答を誘発した可能性がある。

第3章の場面計画が不受理となっており、第3章本文は未着手。第1・2章の本文は保持されている。空回答、反復による生成上限到達、コンテキスト不足ではない。

## 保存応答

対象ジョブ: `960491c501b443868239ed6ffdd572f8`

基点: `services/worker/cache/m2/jobs/63767aa5615dabad/960491c501b443868239ed6ffdd572f8/script/`

| 要求 | 結果 | 入力tokens | 生成tokens | 終了 |
| --- | --- | --- | --- | --- |
| 52、plan-003初回 | character_idsにs1/s2等を出力し、未知人物ID | 13692 | 3088 | stop |
| 53、plan-003修復 | scene-4のloc-castle-gateに対応するlocations定義なし | 13835 | 3898 | stop |

両要求の生成上限は14336。保存応答・検証結果は`llm/52-script-plan-ea6be1ecf03507da.json`、`llm/53-script-plan-e614ee09c1b4b6b5.json`、`requests/plan-003-1.validation.json`、`requests/plan-003-2.validation.json`。

修復応答のlocationsはloc-castle-city、loc-castle-hall、loc-isolde-roomの3件。scene-4はloc-castle-gateを使うが、定義が含まれない。

loc-castle-gate（山城正門前）は前章ジョブ`27dea9c623854e9292e69ce25d7d9236`で登録済み。第3章の保存状態にも保持され、初回・修復の実送信資料にも既存IDと名前が載っている。引継ぎや入力構築で消えたものではない。

## 背景の自動補完を行わない理由

章のlocationsは、当章で使う全背景の今回の状態を持つ。過去のID/name登録だけで検証を通すと、本文生成資料や章成果物に必要な背景定義が欠ける。

前章の城門は夕刻、敵軍に包囲され、絶望的な軍事的圧迫感のある背景。今回scene-4は翌朝、奪還後の領民への再興宣言。前章のLocationを丸ごと補完すると背景状態を誤るため、実施していない。

## 修正

`services/worker/generation/script_continuation.py`の章計画指示だけを明確化。

- 旧「各sceneのcharacter_idsは登場人物を最大3人、idはs1,s2など短くします。」を、短くする対象はscene.id（場面ID）であると明示。人物IDは資料またはnew_charactersのIDをそのまま使う。
- 既存の場所を含め、当章で使用する全場所をlocationsへ定義することを明示。背景状態は当章に合わせる。

スキーマ、参照検証、再試行回数、保存された回答・本文は変更しない。保存済み第2章checkpointを継続できるようプロトコル識別子は変更しない。明示的な再試行では失敗工程だけが新しい指示で生成される。新たな実モデル生成は実施していない。

## 確認と反映

- 既存の場所参照・不正参照拒否・再試行・章間引継ぎの関連テスト32件成功。
- 対象ファイルのruff成功。
- 稼働ジョブがないことを確認し、Workerだけを再起動。新Workerの登録と直近heartbeatを確認。
- 対象ジョブはfailed、retry_generation=0のまま。自動再試行は行っていない。

新Worker ID: `ed11fb089e384d21b027db8655ce81bd`。起動親PID: `145636`。

Workerログ: `tmp/worker-plan-refs-34d5b4a7e12a44caadfdce374bab5a38.err.log`。
