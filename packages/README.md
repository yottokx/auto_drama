# 共通パッケージ

- `contracts/`: Pydanticの中間脚本schemaと、配布用JSON Schema。
- `tyrano_export/`: 同じ脚本・素材から同じバイト列を作るティラノソースZIP変換器と固定サンプル。

制御サーバーとワーカーはこのパッケージを共有します。`narrative/` と `providers/` はM2以降で追加します。
