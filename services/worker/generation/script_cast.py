"""Persisted ensemble design and non-gating episode size guidance."""

from collections import defaultdict

from pydantic import Field, model_validator

from packages.contracts.m2 import CharacterResult
from packages.contracts.script import Contract, Identifier

from .script_budget import SceneTokenPolicy


class ScriptOptions(Contract):
    target_episode_minutes: int = Field(default=10, ge=1, le=120, strict=True)
    target_body_characters: int = Field(default=4000, ge=1, le=100000, strict=True)
    target_dialogue_characters: int = Field(default=2000, ge=1, le=100000, strict=True)
    scene_tokens: SceneTokenPolicy = Field(default_factory=SceneTokenPolicy)

    @model_validator(mode="after")
    def dialogue_fits_body(self):
        if self.target_dialogue_characters > self.target_body_characters:
            raise ValueError("Dialogue target is part of the body target.")
        return self

    def guidance(self):
        return (f"1話（1章）あたり約{self.target_episode_minutes}分を目指すビジュアルノベルです。"
                f"各話の本文約{self.target_body_characters}文字、うち台詞約{self.target_dialogue_characters}文字を"
                "補助目安にします。全章合計の量ではありません。文字数から時間を保証するものでもありません。"
                "会話、試行、反応、日常の掛け合いを計画から十分に用意します。"
                "他の章の長さで短い章を補わず、機械的な字数合わせ・同義反復・待機で水増ししません。")

    def for_stage(self, stage):
        scale = f"完成作品は会話中心のビジュアルノベル、1話（1章）約{self.target_episode_minutes}分を目指します。"
        if stage in {"cast", "chain", "allocation"}:
            return scale + {
                "cast": "今回は人物と関係の設定だけを短い項目で作ります。本文や全台詞は書きません。",
                "chain": "今回は出来事の因果を計画します。各行動・結果は具体的な1〜2文で、全台詞や演出論は書きません。",
                "allocation": "今回は章の役割と少数の会話材料だけを短く記します。本文の分量をこの回答へ要求していません。",
            }[stage]
        if stage == "plan":
            return self.guidance() + "今回は本文ではなく短い場面計画です。上記の本文量をlength_weightで場面へ配分し、計画欄自体を長くしません。"
        if stage == "writer":
            return self.guidance() + "今回は台本本文です。当場面に配分された量を会話・応答・体験で描きます。"
        raise ValueError(f"Unknown script instruction stage: {stage}")


class CastConnection(Contract):
    character_ids: list[Identifier] = Field(min_length=2, max_length=2)
    relationship: str = Field(min_length=1, description="間柄、双方の呼び方・距離感・相手に見せる顔。")


class CastLife(Contract):
    character_id: Identifier
    personal_concern: str = Field(min_length=1, description="本人自身の関心や普段の用事。")
    contact: str = Field(min_length=1, description="接点になる場所や状況。登場予定であり実績ではない。")
    initial_knowledge: str = Field(min_length=1, description="設定上知ることと知らないこと。未来の出来事は知らない。")


class CastPlan(Contract):
    supporting_characters: list[CharacterResult] = Field(max_length=8)
    everyday_context: list[CastLife] = Field(max_length=8)
    connections: list[CastConnection] = Field(max_length=30)


def check_cast(plan: CastPlan, main_ids: set[str]):
    ids = [row.id for row in plan.supporting_characters]
    if len(set(ids)) != len(ids) or set(ids) & (main_ids | {"NARRATOR"}):
        raise ValueError("Planned supporting characters need unique unused IDs.")
    life_ids = [row.character_id for row in plan.everyday_context]
    if len(life_ids) != len(set(life_ids)) or set(life_ids) != set(ids):
        raise ValueError("Every planned supporting character needs one everyday context.")
    pairs = set()
    for row in plan.connections:
        pair = frozenset(row.character_ids)
        if (len(pair) != 2 or not pair <= main_ids | set(ids)
                or pair <= main_ids or pair in pairs):
            raise ValueError("Cast relationships must reference distinct known people without replacing approved pairs.")
        pairs.add(pair)


def story_character(character: dict):
    # Full appearance/voice reference data stays in the registry for asset work.
    return {key: character[key] for key in
            ("id", "name", "age", "gender", "role", "freeform", "settings", "voice") if key in character}


def script_metrics(text: str):
    speakers = defaultdict(lambda: {"lines": 0, "characters": 0})
    run = longest = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        speaker, separator, body = line.partition(":")
        if not separator:
            continue
        speaker, body = speaker.strip(), body.strip()
        speakers[speaker]["lines"] += 1
        speakers[speaker]["characters"] += len(body)
        run = run + 1 if speaker == "NARRATOR" else 0
        longest = max(longest, run)
    body = sum(row["characters"] for row in speakers.values())
    narrator = speakers.get("NARRATOR", {"lines": 0, "characters": 0})
    return {"body_characters": body, "dialogue_characters": body - narrator["characters"],
            "narrator_characters": narrator["characters"],
            "narrator_percent": round(narrator["characters"] * 100 / body, 2) if body else 0,
            "lines": sum(row["lines"] for row in speakers.values()),
            "dialogue_lines": sum(row["lines"] for key, row in speakers.items() if key != "NARRATOR"),
            "longest_narrator_run": longest, "speakers": dict(speakers),
            "playback_seconds": None, "acceptance": "not_evaluated"}


CAST_INSTRUCTION = (
    "全体プロットの前に、承認メインを中心にサブキャラを含むキャストと関係を設計します。"
    "メインキャラの再生成はせず、supporting_charactersには新しい人物の完全な設定だけを記します。"
    "全3章では1〜2人を目安に世界・メイン人数へ合わせます。最大8は容量であり目標ではありません。"
    "登場人物を限定するユーザー設定は優先し、その場合は空配列も可能です。"
    "主筋に不可欠な機能がなくても、掛け合い、生活感、親しみ、相手の別の面を見せる人物を作れます。"
    "本人の用事・関心・話し方を持たせ、解説係や問題を代わりに解く賢者だけにしません。"
    "全員に独立した大事件や成長の筋は不要です。雰囲気に合わせ、全作品を喜劇にはしません。"
    "everyday_contextに各サブの生活上の接点と初期知識を記し、connectionsにメインや他のサブとの間柄を記します。"
    "承認されたメイン同士の関係を書き換えません。初登場と初対面は別で、既知の間柄も作れます。"
    "人物IDは英数字の短い未使用IDを使い、予約話者NARRATORは使いません。"
    "ここで決める接点は設定・将来の可能性であり、本文で実際に登場した記録ではありません。"
)
