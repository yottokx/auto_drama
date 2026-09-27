"""Small, optional craft examples; never part of the story's facts or plot."""

from .causal_runtime import digest

EXAMPLES = {
    "causal-setup": {
        "version": "2",
        "text": (
            "説明だけの案：例Aが最後に故障した荷車を直し、荷物を届ける。\n"
            "earlier_experience：例Aが荷車を試すと段差で車輪が外れ、例Bが留め具の割れを見つける。"
            "手持ちのひもで仮留めすると少量なら運べるが、荷を増すと緩むことを二人で確かめる。\n"
            "later_application：少量なら運べた結果を使い、例Bが荷を分け、例Aが留め直して往復する。"
            "一度で運ぼうとして止まっていた配達を、荷の重さと回数を変えて完了する。\n"
            "出来事列にも、この試行で得る結果と、後の方法にそれを使う行動を置く。"
            "この筋や道具を流用せず、今の作品の設定に合う経験と利用を具体化する。"
        ),
        "sources": [],
        "adaptation": "New generic example of a prior observation informing a later method.",
    },
    "conversation-exchange": {
        "version": "2",
        "text": (
            "抽象的な案：例Aと例Bが互いの優しさを知る。\n"
            "会話材料の例：topicは『例Aは用意した飲み物が冷める前に休憩したいが、例Bは箱の修理を終えて帰りたい』。"
            "exchangeは『例Aが温かい飲み物を勧めると、例Bは片手がふさがると断る。"
            "例Aが箱を支えると申し出て、例Bが砂糖なしを頼み、空いた手で受け取る』。"
            "場面のobjectivesには双方の用事、required_eventsにはこの応答を置く。"
            "返答や距離感は人物ごとに選べる。和解や同じ往復数は必要ない。"
        ),
        "sources": ["docs/experiments/m3-r7-length-comparison-20260925.md"],
        "adaptation": "Generic exchange; no M3 names, setting, events or resolution reused.",
    },
    "script-exchange": {
        "version": "1",
        "text": (
            "sample-a: 休憩にしない？　温かいの、二つ用意したんだけど。\n"
            "sample-b: 今この手を離したら、さっき合わせたところがずれる。先に飲んでて。\n"
            "NARRATOR: Aは二つのカップを台の端に置き、Bの横から箱を支えた。\n"
            "sample-a: ここなら離さなくていいでしょ。持ってるから、ひと口くらい。\n"
            "sample-b: ……ありがとう。でもそっちは甘い方？　砂糖なしがいいんだけど。\n"
            "sample-a: あれ、いつも甘いもの買ってるのに。昨日も一つ多く選んでたよね。\n"
            "sample-b: あれは持って帰る分。家で待ってるのがいるから。\n"
            "NARRATOR: Bは空いた手で甘くない方を取り、湯気が落ち着くまで待った。\n"
            "sample-a: じゃあ帰りを遅くできないね。こっちの箱は私が運ぶよ。\n"
            "sample-b: 助かる。持ち上げる前に言ってね、そっちの底だけ少し弱いから。"
        ),
        "sources": [
            "outputs/m3-r7-length-comparison-20260925/m3-03.md",
            "outputs/m3-r7-length-comparison-20260925/m3-05.md",
        ],
        "adaptation": (
            "New generic dialogue inspired by M3's different personal concerns and concrete replies. "
            "Replaced characters, family details, work, setting, wording and plot; retained no story facts."
        ),
    },
}

PURPOSE_EXAMPLES = {
    "script-outline": "causal-setup",
    "script-allocation": "conversation-exchange",
    "script-plan": "conversation-exchange",
    "script-scene": "script-exchange",
}


def example_for(purpose: str) -> dict | None:
    """Return a fresh candidate; context fitting may omit it before story material."""
    example_id = PURPOSE_EXAMPLES.get(purpose)
    if example_id is None:
        return None
    row = EXAMPLES[example_id]
    return {"id": example_id, "version": row["version"],
            "sha256": digest(row["text"]), "text": row["text"]}


def example_catalog_identity() -> dict:
    """Freeze examples and their stage assignment as part of experiment identity."""
    return {"purposes": dict(PURPOSE_EXAMPLES), "examples": {
        key: {"version": row["version"], "sha256": digest(row["text"]),
              "sources": list(row["sources"]), "adaptation": row["adaptation"]}
        for key, row in EXAMPLES.items()}}
