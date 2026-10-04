# ティラノソース書き出しと M3 プレイヤー

```python
from packages.tyrano_export import compile_bundle, demo_content, validate_bundle

script, assets = demo_content()
bundle = compile_bundle(script, assets)
manifest = validate_bundle(bundle, script, assets)
```

`assets` は `Script.assets[].id` をキーにした完成済みのバイト列です。
`artifact_id` は制御サーバー内の不変な成果物を指します。デモの `demo-*` は登録時に
制御サーバーが実際の成果物 ID に差し替えます。入力の参照先と SHA-256 が一致しない場合は
書き出しません。JSON Schema は `packages/contracts/script.schema.json` にあります。

ZIP には `script.json`、`manifest.json`、`data/scenario/first.ks` と `make.ks`、参照素材、
自作の `index.html`、設定、鑑賞操作・バックログ用の固定コードが入ります。ティラノ本体や
ローカルの絶対パス、接続設定は含みません。同じ入力からは ZIP のメタデータも含めて
同じバイト列を生成します。採用側の `validate_bundle` は期待する出力と完全一致を確認し、
受信 ZIP の解凍は行いません。

制御サーバーの `/player/{build_id}/` は公開済み ZIP と、ユーザー指定のティラノ本体を
読み取り専用で配信します。`AUTO_DRAMA_TYRANO_DIR` に制御PCのインストール先を指定します。
未指定時はリポジトリ直下の `tyranoscript/` を参照し、対応する V520 の必要ファイルを確認します。
原本・サンプル作品は書き換えません。スタンドアロンで再生する場合は、エンジンの
**作業用コピー**へ ZIP の内容を配置して HTTP で配信します。鑑賞設定は 960×640 です。
サンプルの画像は接続確認用の幾何学図形で、音声は含みません。

再生開始、オート、ブラウザー内の途中保存・再開、既読の台詞と音声のバックログ、
文字サイズ・音量を用意しています。保存領域は脚本のハッシュで分離します。
`documents=` で公開設定 `approval.json`、中間本文 `narrative.json`、
原文 `sources/{scene_id}.txt` を追加できます。呼び出し側で公開項目だけを渡し、
接続設定、APIキー、モデルの実行パスや来歴を含めないでください。

本文・題名・話者表示はすべてティラノのリテラル本文として出力します。
話者の内部 ID と表示名を分離し、任意の入力をタグ属性や JavaScript に挿入しません。
タグは背景切替・登退場・左右中央の移動・話者強調・暗転・間の列挙型から生成します。
演出は発話 ID と `before` / `start` / `after` に結び付き、同じ位置の演出は配列順です。
`after` は本文表示と音声の完了後、次のクリック待ちより前に実行します。
暗転は `duration_ms` を片道の所要時間とする黒へのフェードと復帰です。

`Script.scene_transitions` は台本確定後の場面入口の演出です。空なら従来の台本をそのまま
コンパイルします。指定された入口では背景・人物の到達状態を
`data/others/auto_drama_stages.json` にまとめ、固定タグ `ad_transition` で先読み・音楽フェード・
画面交換を一度だけ実行します。同じ舞台の不要な消去と再表示や、入口の重複暗転を整理します。
本文と発話IDは変えず、入口以外の演出は従来どおり処理します。`none` / `cut` は即時表示、
`dissolve` は旧画面の減衰、`fade` は黒への減衰と復帰です。音楽のフェード時間は画面と独立し、
`continue` は現在の曲・音量・A/Bループ位相を引き継ぎます。

`Script.event_cg_segments` は採用済みイベントCGの表示区間です。
`start_utterance_id` の直前から `end_utterance_id` の直前まで表示し、終了がnullなら章末まで
保持します。`variants` は区間内の台詞直前に切り替える採用画像です。最後に含む台詞の
クリック待ち中もCGを表示し、次へ進むと通常表示へ戻ります。空配列は保存時に省略します。
CG素材は `event_cg` 種別として `data/cgimage/` に同梱し、960×640の舞台へ縦横比を保った
contain表示を行います。異なる縦横比の余白は黒で埋め、背景・立ち絵は表示しません。
CG中の通常演出は論理状態として更新するため、終了時はその時点の背景・人物へ戻ります。
CG画像を読み込めなかった場合も、その時点の通常表示で本文を続けます。
保存・再開は採用済み台詞アンカーからCGを復元し、CG切替だけで音声・BGMを再開始しません。

`music.js` と `transitions.js` は調整画面のつなぎ目試聴からも利用します。試聴が渡す外部
AudioContext は本編の録画出力へ登録しません。転換中の途中保存を防ぎ、読込・画面破棄では
保留中の処理を取り消します。準備失敗時は本文が進まなくなる状態を避けて通知します。

`tests/unit/test_contracts_export.py` は再現性、参照不整合、途中 ZIP、不正パス、
タグ注入とバックログの HTML エスケープを検証します。
`tests/unit/test_m3_player.py` は公開境界、パス、HTTP Range、ハッシュ検査、鑑賞操作を確認します。
ローカルのティラノ本体と Node.js がある場合、実際のパーサーによる互換性検証も実行します。
