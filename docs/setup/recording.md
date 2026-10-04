# 作品のバックグラウンド録画

鑑賞プレイヤーを画面非表示のEdgeでオート再生し、音声付きMP4に保存します。PCの画面やマイクではなく、プレイヤーの映像・音声・BGMを取得します。スピーカーに音は出しません。既定は1280×720、15fpsです。録画には再生時間と同程度の時間がかかります。

採用済みのBGMは、公開版に保存された音量とループ位置で台詞の音声に混ぜて収録します。イントロを一度再生した後、指定されたA〜Bの範囲を繰り返します。音声とBGMの音量は独立しています。BGMを含まない以前の公開版も、そのまま録画できます。

場面転換のある公開版では、曲の継続・フェードアウト／インと暗転・ディゾルブも本編の再生処理を通して録画します。フェード後のBGM出力を収録するため、鑑賞時の音量変化がMP4にも入ります。場面転換の時間も動画の再生時間に含まれます。

鑑賞中の章末ではBGMをループしたまま次の操作を待ちます。録画は章末に達すると、録画用の混合音声だけを約0.4秒でフェードアウトして終了します。停止要求・録画時間の上限でも同じ短い終端を収録し、章末の待機を無期限に録画しません。

## GUIで録画する

プロジェクトルートの `record-gui.bat` をダブルクリックします。制御サーバーが起動していれば作品一覧を取得します。プルダウンで作品を選び、保存先フォルダーを指定して「録画開始」を押してください。「更新」で作品一覧を再取得できます。録画は別プロセスで続き、GUIを閉じても継続します。

保存先の中に `recording-日時-識別子/` を作り、完成した動画を `recording.mp4` として保存します。公開済みの章がない作品を選ぶと、録画は開始せずメッセージを表示します。GUIには進捗と「保存先を開く」ボタンがあります。終了や失敗の詳細は `status.json` とログで確認できます。

録画中は「録画時間 00:02:30　実行時間 00:02:40」と「録画1章目　台詞・地の文（10/100）」を表示します。録画時間は各章を合計した動画の長さ、実行時間は起動・読み込み・音声合成・結合を含む経過時間です。台詞数は地の文も含めた現在の章内の位置で、次章では分母と位置が切り替わります。進捗は約1秒ごとに更新します。録画が終わってからも処理中の場合は「章の音声を合成中」「動画を結合中」などと表示します。

## 準備

制御サーバーとティラノ本体を通常どおり利用できる状態にしてください。対象の章は公開済みである必要があります。WindowsのMicrosoft Edge、PATH上のFFmpegが必要です。

初回のみ、プロジェクトルートで依存を追加します。この作業環境には導入済みです。

```powershell
uv pip install --python .venv/Scripts/python.exe -r scripts/record/requirements.txt
```

## 実行

鑑賞画面のURLを指定すると、その公開版から録画します。

```powershell
.\scripts\record-player.ps1 -Url 'http://127.0.0.1:8000/player/BUILD_ID/'
```

作品IDで指定する場合は、現在選択中の公開版から開始します。通常の制作状況APIを使うため、そのAPIが持つ制作状態の復旧処理も実行されます。公開版を固定する場合はURLを使ってください。

```powershell
.\scripts\record-player.ps1 -ProjectId 'PROJECT_ID'
```

コマンドはバックグラウンドプロセスのPIDと保存先を表示して戻ります。録画が終わるまでPCをスリープさせないでください。

公開済みの次章があれば順に録画し、最後にまとめて `recording.mp4` に保存します。次章が制作中の場合は公開済みの章までで終了し、`status.json` の `end_reason` が `waiting` になります。特定の章だけなら `-SingleChapter` を指定します。

```powershell
.\scripts\record-player.ps1 -Url 'http://127.0.0.1:8000/player/BUILD_ID/' `
    -SingleChapter -OutputDir 'C:\Python\envs\auto_drama\outputs\my-recording'
```

保存先は新しいフォルダーを指定してください。既存フォルダーや動画は上書きしません。既定は `outputs/recording-日時/` です。

## 状態確認・停止

- `status.json`: `running` / `completed` / `failed`、録画した章、エラー
- `recording.mp4`: 正常終了時の全章結合動画
- `chapter-001/chapter.mp4` など: 章ごとの動画
- 保存先フォルダー名に `.stdout.log` / `.stderr.log` を付けたファイル: 実行ログ

停止する場合は、保存先に空の `stop.request` ファイルを作ります。

```powershell
New-Item -ItemType File -Path '.\outputs\my-recording\stop.request'
```

停止要求や時間上限では、録画中の章を `chapter-XXX/partial.mp4` に保存して `failed` で終了します。完了した章は残ります。読み込み・合成処理の途中では即時停止しません。ブラウザーや通信の異常時は、合成前の音声・映像が残る場合があります。

`-MaxSeconds 7200` が1章の上限秒数（既定2時間）、`-Fps 15` がフレームレートです。描画が遅い場合はフレームを複製して再生時間を保ちます。映像・音声の時刻は別々に取得するため、フレーム単位の厳密な同期は保証しません。

`-Foreground` でターミナルの前面実行になります。ブラウザーは非表示のままです。Chromeは `-Browser chrome`、Playwright付属Chromiumは `-Browser chromium` で選べます。後者は別途 `python -m playwright install chromium` が必要です。

Python版 `scripts/record/record_player.py --help` には `--width` / `--height`、`--ffmpeg`、`--keep-raw` もあります。独自のHTML5音声や外部ゲームには対応していません。静的な書き出しをHTTP配信したURLは `--single-chapter` と組み合わせてください。

## 検証

軽量テストは `tests/unit/test_record_player.py`。実ブラウザーの検証は次のとおりです。短い固定シーンとテスト音を使い、作品生成やモデル起動は行いません。

```powershell
$env:AUTO_DRAMA_RECORD_SMOKE = '1'
.\.venv\Scripts\python.exe -m pytest tests/integration/test_record_player_browser.py -q
```

参考: [Playwright Python](https://playwright.dev/python/docs/library)、[Howler公開API](https://github.com/goldfire/howler.js)。
