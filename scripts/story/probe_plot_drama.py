"""STEP4's plot with each person's aims and what they know read before the events.

The product plans a long-form plot as chain of events -> allocation -> each chapter's added steps,
and the chain writes the core (who wins and how) first. Its steps are "action -> result", where the
planner writes whatever result the ending needs, and several rules hold conflict back for every
work. Here the cast plan and chapter count stay the product's, and only the plot is made:

  base  the product's plot as is
  a     forces first: per person, what they want in this story, what they will and will not do,
        what they do not know or hide, and with whom they are at odds (empty when no one);
        steps carry the aim apart from the result, which is how the other side answers from its
        own wants and knowledge, and who goes on not knowing it; conflict rules tie to the forces
        (none added that the forces lack, none dropped that they hold); results are observable
  b     a, but the core comes after the events: nothing decides the ending before the chain
  a2    a, and each person's last resort in the forces; a conflict is not settled by someone's
        realization, persuasion or stepping back alone, but by their last resort failing against
        the other side, and the loser keeps the value they hold; people working together write
        where their wants differ, and someone carrying out another's plan has their own aim
  a3l   a2, and the losing side's landing says what they lose and what they keep, not that they
        admit the other side was right or see their limits
  a3i   a2, and in a conflict with a winner, the losing side once gets a move through that reads
        and beats the other side's plan before the end, which the winner recovers from
  a3s   a2, and the same as a planned field after the core (setbacks: who reads which plan, the
        move, what the winning side loses), which the events play before the last chapter and
        may not take back as something the winner foresaw
  a4    a3l with setbacks, but the forces come first in a call of their own, each with the kind of
        conflict (none, crossed wishes, win or lose); the chain gets them in its brief, and its
        answer format has the setbacks field only when some conflict is win or lose

Nothing in the product changes. --export writes a work's input from the app database (read only).
"""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import sys
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic.fields import FieldInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from packages.contracts.script import Contract, Identifier
from services.worker.generation import pipeline, script_continuation, script_expansion, script_plot
from services.worker.generation.script_cast import CastPlan
from services.worker.generation.text_debug import run_text_debug
from services.worker.model_config import select_config

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("base", "a", "b", "a2", "a3l", "a3i", "a3s", "a4", "a5", "a3r")
LAST_RESORT_ARMS = ("a2", "a3l", "a3i", "a3s", "a4", "a5", "a3r")
PROFILE = {"provider": "local", "model_id": "gemma-4-31B-it-UD-Q4_K_XL", "context_size": 32768,
           "temperature": 0.65, "top_p": 0.95, "reasoning_level": "none", "common_settings_version": 1}
OPTIONS = {"planned_cast_appears": True, "open_scene_cast": True, "long_form": True,
           "chapter_scope_guard": True, "opening_split": True}


def replaced(text: str, pairs: list[tuple[str, str]]) -> str:
    for old, new in pairs:
        if text.count(old) != 1:
            raise ValueError(f"The product wording differs from the expected text: {old}")
        text = text.replace(old, new)
    return text


class Force(Contract):
    character_id: Identifier
    want: str = Field(min_length=1, max_length=160,
        description="この物語の間に本人が得たい・守りたい・避けたいこと。大きさは作品なり。")
    means: str = Field(min_length=1, max_length=160, description="そのために本人がすること・しないこと。")
    unknown: str = Field(max_length=160, description="本人が知らないこと、または誰かに隠していること。なければ空文字。")
    at_odds: str = Field(max_length=160, description="誰の望み・手段と、どこで食い違うか。なければ空文字。")


FORCES_RULE = (
    "forcesには、承認設定と計画キャストから読み取れる力関係を、物語を動かす人物ごとに記します（承認メインは全員）。"
    "wantはこの物語の間に本人が得たい・守りたい・避けたいことで、大きさは作品なりです。日常の小さな望みでも構いません。"
    "meansはそのために本人がすること・しないこと、unknownは本人が知らないことや誰かに隠していること、"
    "at_oddsは誰の望み・手段とどこで食い違うかです。食い違いがなければ空文字にし、作品にない対立を作りません。")
DRAMA_RULE = (
    "各人物はforcesの望みと手段で動きます。食い違う人物を、相手の都合に合わせて引き下がる・従うだけの役にせず、"
    "自分の目的で先に仕掛けることもあります。相手が知っていることと望みから、狙いが通る・外れる・読まれる・"
    "断られる・別の形になるが決まります。外れる回数の決まりはありません。"
    "forcesにない対立や事件を足さず、forcesにある食い違いは省きません。"
    "食い違いが小さい作品では、それに見合う小さなすれ違いや譲り合いとして描きます。")
STEP_RULE = (
    "stepsは登録人物のaim、action、具体的なresult、hidden_fromの順にします。aimは本人の狙い、"
    "resultは相手や状況が自分の望みと知っていることから応じ、実際に起きたことです。"
    "その結果を本人が知る・経験して次の行動を選びます。resultは誰が何をした・言った・選んだかで書き、"
    "内心の評価だけで済ませません。hidden_fromの人物はその結果を知らないまま動き、知らない情報で判断しません。"
    "伝える・報告する・確認するだけの手順は、それで相手の選択が変わるときだけ置きます。")

A_CHAIN = replaced(script_plot.CHAIN_INSTRUCTION, [
    ("出力はcore、resolution_basis、opening_condition、eventsの順です。",
     "出力はforces、core、resolution_basis、opening_condition、eventsの順です。" + FORCES_RULE),
    ("まずcoreで外的な決着と人物・関係の着地を、本人の行動とその結果として定めます。",
     "まずcoreで外的な決着と人物・関係の着地を、本人の行動とその結果として定めます。"
     "決着はforcesの望みと手段から導き、食い違う人物がいるなら、その人物の手段と抵抗を経た結果として書きます。"),
    ("全動作の細分化や、毎回失敗・訓練・成功する型は不要です。",
     "全動作の細分化や、毎回失敗・訓練・成功する決まった型は不要です。" + DRAMA_RULE),
    ("stepsは登録人物のactionと具体的なresultの順にし、その結果を本人が知る・経験して次の行動を選びます。",
     STEP_RULE),
])


def _b_chain() -> str:
    text = replaced(A_CHAIN, [
        ("出力はforces、core、resolution_basis、opening_condition、eventsの順です。",
         "出力はforces、opening_condition、events、coreの順です。"),
        ("その下書きを受けてeventsを作ります。", ""),
        ("下書きに書いたこと自体は出来事の実施を意味しません。", ""),
        ("最後はcoreで約束した行動の結果まで描きます。準備や約束だけで完了扱いにしません。",
         "決着は準備や約束だけで済ませず、行動の結果まで描きます。最後にcoreで、eventsで実際に起きた外的な決着と"
         "人物・関係の着地を、本人の行動とその結果として書きます。外的な決着は、その行動が障害や相手側の"
         "どの条件を変えたかまで示します。"),
    ])
    # The core and the resolution_basis scaffold decide the ending first; b has neither up front.
    start = text.index("まずcoreで外的な決着")
    end = text.index("承認設定と矛盾する力や条件を")
    if text.count("まずcoreで外的な決着") != 1 or end < start:
        raise ValueError("The chain wording differs from the expected text.")
    return (text[:start] + "結末を先に決めて逆算せず、eventsをforcesの望みと手段から、各人物がそれぞれの目的で"
            "動いた結果として前へ進めます。最後の出来事で、中心の食い違いや問いに決着をつけます。" + text[end:])


B_CHAIN = _b_chain()


class LastResortForce(Contract):
    character_id: Identifier
    want: str = Field(min_length=1, max_length=160,
        description="この物語の間に本人が得たい・守りたい・避けたいこと。大きさは作品なり。")
    means: str = Field(min_length=1, max_length=160, description="そのために本人がすること・しないこと。")
    last_resort: str = Field(max_length=160,
        description="追い込まれたとき本人が最後に使う手段や越える一線。食い違いに見合う大きさ。なければ空文字。")
    unknown: str = Field(max_length=160, description="本人が知らないこと、または誰かに隠していること。なければ空文字。")
    at_odds: str = Field(max_length=160,
        description="誰の望み・手段と、どこで食い違うか。協力する相手との違いも含む。なければ空文字。")


A2_CHAIN = replaced(A_CHAIN, [
    ("meansはそのために本人がすること・しないこと、",
     "meansはそのために本人がすること・しないこと、last_resortは追い込まれたときに本人が最後に使う手段や"
     "越える一線で、大きさは食い違いに見合うものです（小さな食い違いなら意地や頼み込み程度）。"),
    ("at_oddsは誰の望み・手段とどこで食い違うかです。",
     "at_oddsは誰の望み・手段とどこで食い違うかで、協力する人物同士の望みの違いも含みます。"),
    ("食い違う人物がいるなら、その人物の手段と抵抗を経た結果として書きます。",
     "食い違う人物がいるなら、その人物がlast_resortまで使った手が通らなくなる結果として書きます。"
     "人物・関係の着地でも、負けた側が守る価値観を捨てたことにしません。"),
    ("食い違いが小さい作品では、それに見合う小さなすれ違いや譲り合いとして描きます。",
     "食い違いが小さい作品では、それに見合う小さなすれ違いや譲り合いとして描きます。"
     "勝ち負けのある食い違いの決着を、本人の気づき・説得・自発的な撤退だけで済ませません。引き下がる・負けるなら、"
     "本人がlast_resortまで使い、それが相手の手で通らなくなった結果にします。負けた人物も守る価値観は捨てず、"
     "その価値観のまま退く・負ける形にします。協力する人物も、自分のwantのために動く手順を持ちます。"
     "誰かの指示や計画を実行する手順でも、aimには本人自身の狙いを書き、resultには本人が足した・変えたことを含めます。"),
])
EXPANSION_RULE = (
    "加える手順も、briefのforcesの望み・手段・知らないことに沿い、aim・action・result・hidden_fromを"
    "骨組みと同じ意味で書きます。食い違う人物を、相手に合わせて従うだけの役にしません。")
LANDING_RULE = ("負ける側の着地は、何を失い、何を守ったままかで書き、"
                "相手の正しさを認める・限界を悟るという宣言で代用しません。")
OUTWIT_RULE = ("勝ち負けのある食い違いでは、負ける側も決着の前に一度は、相手の計画を読んで上回る手を通します。"
               "勝つ側の計画は少なくとも一度その手で崩れ、立て直して決着へ進みます。")
A3L_CHAIN = replaced(A2_CHAIN, [("人物・関係の着地でも、負けた側が守る価値観を捨てたことにしません。",
                                 "人物・関係の着地でも、負けた側が守る価値観を捨てたことにしません。" + LANDING_RULE)])
READER_RULE = (
    "readerは、その手順を読者がいつ知るかです。見せるはその場で描く手順、伏せるは起きたことを読者にもまだ見せず、"
    "後の手順で初めて明かす手順、明かすは伏せた手順の中身を読者が初めて知る手順です。"
    "仕掛けや隠しごとには、読者も先に知って当事者が気づくかを見守る形（見せる）と、読者にも伏せておき結果で驚かせる形"
    "（伏せる→明かす）があり、作品と場面に合わせて選びます。全部を見せる作品もあり、伏せる数を埋めるために作りません。"
    "伏せた手順は後の明かす手順で必ず回収し、明かすときは伏せていた準備や狙いも含めて見せます。"
    "伏せる仕掛けについて、その相談・指示・報告だけの手順を明かす前に置きません。")
A3R_CHAIN = replaced(A3L_CHAIN, [
    ("具体的なresult、hidden_fromの順にします。", "具体的なresult、hidden_from、readerの順にします。"),
    ("伝える・報告する・確認するだけの手順は、それで相手の選択が変わるときだけ置きます。",
     "伝える・報告する・確認するだけの手順は、それで相手の選択が変わるときだけ置きます。" + READER_RULE),
])
READER_EXPANSION_RULE = ("加える手順のreaderも骨組みと同じ意味で書きます。骨組みで伏せた手順の中身を、"
                         "それを明かす手順より前の手順で読者に見せません。")
A3I_CHAIN = replaced(A2_CHAIN, [("その価値観のまま退く・負ける形にします。",
                                 "その価値観のまま退く・負ける形にします。" + OUTWIT_RULE)])


class Setback(Contract):
    character_id: Identifier
    move: str = Field(min_length=1, max_length=200,
        description="負ける側が相手のどの計画をどう読み、何をするか。")
    loss: str = Field(min_length=1, max_length=160,
        description="それで勝つ側が実際に失うもの・崩れる計画。")


SETBACK_RULE = (
    "setbacksには、勝ち負けのある食い違いで、負ける側が決着の前に相手の計画を読んで上回り、通す手を最大2件記します。"
    "character_idはその手を打つ人物、moveは相手のどの計画をどう読み、何をするか、"
    "lossはそれで勝つ側が実際に失うもの・崩れる計画です。勝ち負けのある食い違いがなければ空配列です。"
    "eventsでは、setbacksを最終章より前に実際に起こし、勝つ側はその損失を抱えたまま立て直して決着へ進みます。"
    "勝つ側があらかじめ読んでいた・わざと乗ったことにして、setbacksを後から取り消しません。")
A3S_CHAIN = replaced(A2_CHAIN, [
    ("出力はforces、core、resolution_basis、opening_condition、eventsの順です。",
     "出力はforces、core、setbacks、resolution_basis、opening_condition、eventsの順です。"),
    ("食い違いがなければ空文字にし、作品にない対立を作りません。",
     "食い違いがなければ空文字にし、作品にない対立を作りません。" + SETBACK_RULE),
])
SETBACK_EXPANSION_RULE = "briefのsetbacksは計画済みの逆転です。それを勝つ側があらかじめ読んでいたことにする手順を加えません。"


class KindForce(LastResortForce):
    conflict_kind: Literal["なし", "すれ違い", "勝ち負け"] = Field(
        description="食い違いの種類。なし、すれ違い（勝ち負けを決めない望みや好みの違い）、"
                    "勝ち負け（一方が目的を達すると他方が達せない）。")


class ForcesAnswer(Contract):
    forces: list[KindForce] = Field(min_length=1, max_length=12,
        description="物語を動かす人物ごとの望み・手段・最後の手・知らないこと・食い違いとその種類。承認メインは全員。")


def _forces_text() -> str:
    start, end = A2_CHAIN.index("forcesには、"), A2_CHAIN.index("作品にない対立を作りません。")
    return A2_CHAIN[start:end + len("作品にない対立を作りません。")]


FORCES_TEXT = _forces_text()
FORCES_TASK = (
    "会話中心のビジュアルノベルのプロットを作る前に、承認設定と計画キャストから力関係だけを読み取ります。"
    "出来事・結末・章の区切りはまだ書きません。" + FORCES_TEXT
    + "conflict_kindは食い違いの種類です。食い違いがなければなし、望みや好みの違いで勝ち負けを決めないものはすれ違い、"
    "一方が目的を達すると他方が達せないものは勝ち負けです。作品にない勝負を作らず、ある勝負は省きません。")
A4_FORCES_GIVEN = ("briefのforcesは先に読み取った人物ごとの力関係で、望み・手段・最後の手・知らないこと・"
                   "食い違いとその種類（conflict_kind）です。")
A4_SETBACK_RULE = (
    "setbacksには、forcesで種類が勝ち負けの食い違いについて、負ける側が決着の前に相手の計画を読んで上回り、"
    "通す手を記します。相手側の手段と知っていることから成り立つものだけを0〜2件書き、件数を埋めるために作りません。"
    "character_idはその手を打つ人物、moveは相手のどの計画をどう読み、何をするか、"
    "lossはそれで勝つ側が実際に失うもの・崩れる計画です。"
    "eventsでは、setbacksを最終章より前に実際に起こし、勝つ側はその損失を抱えたまま立て直して決着へ進みます。"
    "勝つ側があらかじめ読んでいた・わざと乗ったことにして、setbacksを後から取り消しません。")
A4_WITH_SETBACKS = ("出力はcore、setbacks、resolution_basis、opening_condition、eventsの順です。"
                    + A4_FORCES_GIVEN + A4_SETBACK_RULE)
A4_WITHOUT_SETBACKS = "出力はcore、resolution_basis、opening_condition、eventsの順です。" + A4_FORCES_GIVEN
A4_CHAIN = replaced(A3L_CHAIN, [(FORCES_TEXT, ""), (
    "出力はforces、core、resolution_basis、opening_condition、eventsの順です。", A4_WITH_SETBACKS)])
A2_EXPANSION_RULE = "誰かの指示や計画を実行する人物の手順でも、aimは本人自身の狙いにします。"
SETBACKS_BRIEF = "\nsetbacks（計画済みの逆転）:\n"


class Reversal(Contract):
    victim: Identifier = Field(description="決着で勝つ側のうち、この手で実際に失う人物。")
    loss: str = Field(min_length=1, max_length=160,
        description="失うもの。味方・手札・拠点・信用・計画の一部など、後で無傷に戻らない具体物。")
    by: Identifier = Field(description="その手を打つ相手側の人物。")
    move: str = Field(min_length=1, max_length=200,
        description="byが本人の手段と知っていることから、victimのどの計画をどう読み、何をしたか。")
    recovery: str = Field(min_length=1, max_length=160,
        description="victimが失ったものなしで、どの別の手に切り替えて進むか。")


class ContestAnswer(Contract):
    reason: str = Field(min_length=1, max_length=200, description="判断の根拠を承認設定から1文で。")
    contest: bool = Field(description="主筋が勝ち負けを決める争いで進むか。")


CONTEST_TASK = (
    "会話中心のビジュアルノベルのプロットを作る前に、この作品の主筋の種類だけを判断します。出来事や結末はまだ書きません。"
    "contestは、主筋が、人物同士が相手の目的を阻み自分の目的を通そうとする争い（抗争・知略戦・戦い・勝負）で進むならtrue、"
    "日常・交流・すれ違い・個人の葛藤や選択が中心で、勝ち負けを決める争いが主筋でなければfalseです。"
    "脇の小さな張り合いや口論だけではtrueにしません。reasonはその判断の根拠を承認設定から1文で先に書きます。")
REVERSAL_RULE = (
    "setbacksには、主筋の争いで、決着の前に勝つ側が相手の手で実際に失うものを1〜2件記します。"
    "victimは失う側の人物、lossは失うもので、味方・手札・拠点・信用・計画の一部など、後で無傷に戻らない具体物です。"
    "byはその手を打つ相手側の人物、moveはbyが本人の手段と知っていることから、victimのどの計画をどう読み、何をしたかです。"
    "recoveryはvictimが失ったものなしで、どの別の手に切り替えて進むかです。"
    "eventsでは、setbacksを最終章より前に実際に起こし、lossは決着までそのまま残ります。"
    "victimがあらかじめ読んでいた・わざと乗った・実は無傷だったことにしません。決着はrecoveryの手から導き、失ったものに頼りません。"
    "決着に近づくほど、勝つ側の残る手は相手の対応で減っていき、最初の手のまま勝ち切りません。")
A3L_ORDER = "出力はforces、core、resolution_basis、opening_condition、eventsの順です。"
A5_ORDER = "出力はforces、core、setbacks、resolution_basis、opening_condition、eventsの順です。"
A5_CHAIN = replaced(A3L_CHAIN, [
    (A3L_ORDER, A5_ORDER),
    ("食い違いがなければ空文字にし、作品にない対立を作りません。",
     "食い違いがなければ空文字にし、作品にない対立を作りません。" + REVERSAL_RULE),
])
A5_EXPANSION_RULE = ("briefのsetbacksは計画済みの逆転で、lossは決着まで残ります。victimがあらかじめ読んでいた・"
                     "わざと乗った・実は無傷だったことにする手順や、失ったものを取り戻す手順を加えません。")
FORCES_BRIEF = "\n力関係forces（人物ごとの望み・手段・知らないこと・食い違い。全て予定の前提）:\n"


def add_step_fields() -> None:
    """aim before the action and hidden_from after the result, in every step the plot holds."""
    from pydantic import BaseModel

    aim = FieldInfo.from_annotated_attribute(str, Field(min_length=1, max_length=200,
        description="この行動で本人が狙うこと。結果と一致するとは限らない。"))
    hidden = FieldInfo.from_annotated_attribute(list[Identifier], Field(max_length=6,
        description="この結果を知らないまま次に動く当事者。知れば行動が変わる人物だけ。なければ空配列。"))
    result = FieldInfo.from_annotated_attribute(str, Field(min_length=1,
        description="相手や状況が自分の望みと知っていることから応じ、実際に起きたこと。狙いどおりとは限らない。"
                    "誰が何をした・言った・選んだかで書く。次の行動の根拠になる。"))
    for cls in (script_plot.ChainStep, script_expansion.AddedStep):
        fields = {}
        for key, value in cls.model_fields.items():
            if key in ("aim", "hidden_from"):
                # The product's own optional fields (ScriptOptions.story_forces); these replace them.
                continue
            if key == "action":
                fields["aim"] = aim
            fields[key] = result if key == "result" else value
            if key == "result":
                fields["hidden_from"] = hidden
        cls.model_fields.clear()
        cls.model_fields.update(fields)
        cls.model_rebuild(force=True)
    for module in (script_plot, script_expansion):
        for value in list(vars(module).values()):
            if (isinstance(value, type) and issubclass(value, BaseModel) and value.__module__ == module.__name__
                    and value not in (script_plot.ChainStep, script_expansion.AddedStep)):
                value.model_rebuild(force=True)


def add_reader_field() -> None:
    """reader after hidden_from: when the reader learns the step (a3r)."""
    from pydantic import BaseModel

    reader = FieldInfo.from_annotated_attribute(Literal["見せる", "伏せる", "明かす"], Field(
        description="読者がこの手順をいつ知るか。見せる＝その場で描く、伏せる＝読者にもまだ見せない、"
                    "明かす＝伏せた手順の中身を読者が初めて知る。"))
    for cls in (script_plot.ChainStep, script_expansion.AddedStep):
        fields = dict(cls.model_fields)
        fields["reader"] = reader
        cls.model_fields.clear()
        cls.model_fields.update(fields)
        cls.model_rebuild(force=True)
    for module in (script_plot, script_expansion):
        for value in list(vars(module).values()):
            if (isinstance(value, type) and issubclass(value, BaseModel) and value.__module__ == module.__name__
                    and value not in (script_plot.ChainStep, script_expansion.AddedStep)):
                value.model_rebuild(force=True)


def drama_models(arm: str):
    """chapter_models with forces in the chain draft (and, for b, the core after the events)."""
    original = script_plot.chapter_models

    def models(count: int):
        chain_model, draft_model, allocation_model, fixed = original(count)

        force = LastResortForce if arm in LAST_RESORT_ARMS else Force

        class ForcesDraft(draft_model):
            forces: list[force] = Field(min_length=1, max_length=12,
                description="物語を動かす人物ごとの望み・手段・知らないこと・食い違い。承認メインは全員。")

        class SetbackDraft(ForcesDraft):
            setbacks: list[Setback] = Field(max_length=2,
                description="負ける側が決着の前に相手の計画を読んで通す手。勝ち負けのある食い違いがなければ空配列。")

        class CoreLastDraft(ForcesDraft):
            resolution_basis: list[script_plot.ResolutionBasis] = Field(default_factory=list, max_length=0)

        class KindDraft(draft_model):
            setbacks: list[Setback] = Field(default_factory=list, max_length=2,
                description="勝ち負けの食い違いで、負ける側が決着の前に相手の計画を読んで通す手。0〜2件。")

        class ReversalDraft(ForcesDraft):
            setbacks: list[Reversal] = Field(default_factory=list, max_length=2,
                description="主筋の争いで、決着の前に勝つ側が相手の手で実際に失うもの。1〜2件。")

        draft = {"b": CoreLastDraft, "a3s": SetbackDraft, "a4": KindDraft, "a5": ReversalDraft}.get(arm, ForcesDraft)
        return chain_model, draft, allocation_model, fixed

    return models


def ordered(schema: dict, order: list[str]) -> None:
    properties = schema["properties"]
    schema["properties"] = {key: properties[key] for key in order}
    schema["required"] = [key for key in order if key in schema.get("required", order)]


def use_arm(arm: str, cast: CastPlan | None) -> None:
    sc = script_continuation
    structured = sc.ScriptRun.structured

    def saved_cast(self, key, purpose, number, prompt, context, model, validate=None, schema=None):
        if key == "cast-plan" and cast is not None:
            prepared = validate(cast) if validate else None
            return prepared if prepared is not None else cast
        if arm != "base" and "story_forces" in self.state and (key == "chapter-allocation" or key.startswith("expand-")):
            context = {**context, "brief": context["brief"] + FORCES_BRIEF
                       + json.dumps(self.state["story_forces"], ensure_ascii=False)
                       + (SETBACKS_BRIEF + json.dumps(self.state["story_setbacks"], ensure_ascii=False)
                          if self.state.get("story_setbacks") else "")}
        return structured(self, key, purpose, number, prompt, context, model, validate, schema)

    sc.ScriptRun.structured = saved_cast
    if arm == "base":
        return
    add_step_fields()
    if arm == "a3r":
        add_reader_field()
    sc.chapter_models = drama_models(arm)
    chain = {"a": A_CHAIN, "b": B_CHAIN, "a2": A2_CHAIN, "a3l": A3L_CHAIN, "a3i": A3I_CHAIN, "a3s": A3S_CHAIN,
             "a4": A4_CHAIN, "a5": A5_CHAIN, "a3r": A3R_CHAIN}[arm]
    sc.chain_instruction = lambda count: replaced(chain, [
        ("全3章の分量", f"全{count}章の分量"),
        ("3〜9個の出来事", f"{count}〜{count * script_plot.EVENTS_PER_CHAPTER}個の出来事"),
        ("3章の各区間で", f"{count}章の各区間で")])
    expansion_prompt, expansion_schema = sc.expansion_prompt, sc.expansion_schema
    rule = (EXPANSION_RULE + (A2_EXPANSION_RULE if arm in LAST_RESORT_ARMS else "")
            + (SETBACK_EXPANSION_RULE if arm in ("a3s", "a4") else "")
            + (A5_EXPANSION_RULE if arm == "a5" else "")
            + (READER_EXPANSION_RULE if arm == "a3r" else ""))
    sc.expansion_prompt = lambda count, number: expansion_prompt(count, number) + rule

    def drama_expansion_schema(available, steps, last):
        schema, room = expansion_schema(available, steps, last)
        schema["$defs"]["AddedStep"]["properties"]["hidden_from"]["items"]["enum"] = sorted(available)
        return schema, room

    sc.expansion_schema = drama_expansion_schema
    structured_core = sc.ScriptRun.structured_core

    def forces_core(self, key, purpose, number, prompt, context, model, validate, schema):
        if key != "story-chain":
            return structured_core(self, key, purpose, number, prompt, context, model, validate, schema)
        available = schema["$defs"]["ChainStep"]["properties"]["character_id"]["enum"]
        schema["$defs"]["ChainStep"]["properties"]["hidden_from"]["items"]["enum"] = available
        if arm == "a4":
            return kind_core(self, key, purpose, number, prompt, context, model, validate, schema, available)
        schema["$defs"][LastResortForce.__name__ if arm in LAST_RESORT_ARMS else Force.__name__][
            "properties"]["character_id"]["enum"] = available
        if arm == "b":
            schema["properties"].pop("resolution_basis")
            ordered(schema, ["forces", "opening_condition", "events", "core"])
        elif arm == "a3s":
            schema["$defs"]["Setback"]["properties"]["character_id"]["enum"] = available
            ordered(schema, ["forces", "core", "setbacks", "resolution_basis", "opening_condition", "events"])
        elif arm == "a5" and contest_of(self, purpose, number, context, available):
            for field in ("victim", "by"):
                schema["$defs"]["Reversal"]["properties"][field]["enum"] = available
            schema["required"] = [*schema.get("required", []), "setbacks"]
            ordered(schema, ["forces", "core", "setbacks", "resolution_basis", "opening_condition", "events"])
        elif arm == "a5":
            # Not a contest: the answer is the a3l answer, with no setbacks field to fill.
            schema["properties"].pop("setbacks")
            schema["$defs"].pop("Reversal")
            ordered(schema, ["forces", "core", "resolution_basis", "opening_condition", "events"])
            prompt = replaced(prompt, [(A5_ORDER, A3L_ORDER), (REVERSAL_RULE, "")])
        else:
            ordered(schema, ["forces", "core", "resolution_basis", "opening_condition", "events"])
        mains = set(self.setting(planned=True)["main_character_ids"])

        def check(value):
            validate(value)
            if not mains <= {row.character_id for row in value.forces}:
                raise ValueError("forces must hold every approved main character.")

        draft = structured_core(self, key, purpose, number, prompt, context, model, check, schema)
        self.state["story_forces"] = [row.model_dump(mode="json") for row in draft.forces]
        self.persist()
        sc.write_json(self.output / "story-forces.json", self.state["story_forces"])
        if hasattr(draft, "setbacks"):
            self.state["story_setbacks"] = [row.model_dump(mode="json") for row in draft.setbacks]
            self.persist()
            sc.write_json(self.output / "story-setbacks.json", self.state["story_setbacks"])
        return draft

    def contest_of(self, purpose, number, context, available):
        """Once per work: is the main plot a contest? Only then does the chain plan setbacks."""
        if "story_contest" not in self.state:
            answer = self.structured("story-contest", purpose, number, CONTEST_TASK, context, ContestAnswer)
            self.state["story_contest"] = answer.model_dump(mode="json")
            self.persist()
            sc.write_json(self.output / "story-contest.json", self.state["story_contest"])
        return self.state["story_contest"]["contest"]

    def kind_core(self, key, purpose, number, prompt, context, model, validate, schema, available):
        mains = set(self.setting(planned=True)["main_character_ids"])
        if "story_forces" not in self.state:
            forces_schema = ForcesAnswer.model_json_schema()
            forces_schema["$defs"]["KindForce"]["properties"]["character_id"]["enum"] = available

            def check_forces(value):
                if not mains <= {row.character_id for row in value.forces}:
                    raise ValueError("forces must hold every approved main character.")

            answer = self.structured("story-forces", purpose, number, FORCES_TASK, context, ForcesAnswer,
                                     check_forces, forces_schema)
            self.state["story_forces"] = [row.model_dump(mode="json") for row in answer.forces]
            self.persist()
            sc.write_json(self.output / "story-forces.json", self.state["story_forces"])
        forces = self.state["story_forces"]
        context = {**context, "brief": context["brief"] + FORCES_BRIEF + json.dumps(forces, ensure_ascii=False)}
        if any(row["conflict_kind"] == "勝ち負け" for row in forces):
            schema["$defs"]["Setback"]["properties"]["character_id"]["enum"] = available
            ordered(schema, ["core", "setbacks", "resolution_basis", "opening_condition", "events"])
        else:
            # No win-or-lose conflict: the answer has no setbacks field to fill.
            schema["properties"].pop("setbacks")
            schema["$defs"].pop("Setback")
            ordered(schema, ["core", "resolution_basis", "opening_condition", "events"])
            prompt = replaced(prompt, [(A4_WITH_SETBACKS, A4_WITHOUT_SETBACKS)])
        draft = structured_core(self, key, purpose, number, prompt, context, model, validate, schema)
        self.state["story_setbacks"] = [row.model_dump(mode="json") for row in draft.setbacks]
        self.persist()
        sc.write_json(self.output / "story-setbacks.json", self.state["story_setbacks"])
        return draft

    sc.ScriptRun.structured_core = forces_core


def names(value: dict, plan: dict | None) -> dict:
    found = {row["result"]["id"]: row["result"]["name"] for row in value["approval_snapshot"]["characters"]}
    for row in (plan or {}).get("supporting_characters", []):
        found[row["id"]] = row["name"]
    return found


def setback_line(row: dict, who: dict) -> str:
    if "victim" not in row:
        return f"- **{who.get(row['character_id'], row['character_id'])}** {row['move']} → 失うもの: {row['loss']}"
    return (f"- **{who.get(row['by'], row['by'])}** {row['move']} → {who.get(row['victim'], row['victim'])}が失う: "
            f"{row['loss']} / 立て直し: {row['recovery']}")


def plan_markdown(output: Path, who: dict) -> str:
    plot = json.loads((output / "plot.json").read_text(encoding="utf-8"))
    forces_path = output / "story-forces.json"
    lines = []
    if forces_path.exists():
        lines.append("## 力関係")
        for row in json.loads(forces_path.read_text(encoding="utf-8")):
            lines.append(f"- **{who.get(row['character_id'], row['character_id'])}** 望み: {row['want']} / "
                         f"手段: {row['means']} / " + (f"最後の手: {row['last_resort'] or '—'} / " if "last_resort" in row else "")
                         + f"知らない・隠す: {row['unknown'] or '—'} / 食い違い: {row['at_odds'] or '—'}"
                         + (f"（{row['conflict_kind']}）" if "conflict_kind" in row else ""))
    setbacks_path = output / "story-setbacks.json"
    if setbacks_path.exists():
        lines += ["", "## 計画した逆転"] + [
            setback_line(row, who) for row in json.loads(setbacks_path.read_text(encoding="utf-8"))] or ["- なし"]
    contest_path = output / "story-contest.json"
    if contest_path.exists():
        contest = json.loads(contest_path.read_text(encoding="utf-8"))
        lines += ["", f"## 主筋の種類: {'争い' if contest['contest'] else '争いではない'}", f"- {contest['reason']}"]
    core = plot["core"]
    lines += ["", "## 核心", f"- 中心の問い: {core['central_question']}", f"- 外的な決着: {core['external_resolution']}",
              f"- 人物・関係の着地: {core['relationship_resolution']}"]
    for chapter in plot["chapters"]:
        lines += ["", f"## 第{chapter['number']}章 {chapter['title']}", f"*{chapter['role']}*", ""]
        for event in chapter["events"]:
            for step in event["steps"]:
                aim = f"［狙い: {step['aim']}］ " if step.get("aim") else ""
                hidden = ("（知らない: " + "、".join(who.get(c, c) for c in step["hidden_from"]) + "）"
                          if step.get("hidden_from") else "")
                mark = {"伏せる": "〔伏せる〕", "明かす": "〔明かす〕"}.get(step.get("reader"), "")
                lines.append(f"- {mark}**{who.get(step['character_id'], step['character_id'])}** {aim}{step['action']}"
                             f" → {step['result']}{hidden}")
    return "\n".join(lines) + "\n"


def export(project: str, name: str, works: Path) -> Path:
    """A work's approval snapshot, cast plan and chapter count from the app database (read only)."""
    db = sqlite3.connect(f"file:{ROOT / 'data' / 'auto_drama.db'}?mode=ro", uri=True)
    title = db.execute("SELECT title FROM project WHERE id=?", (project,)).fetchone()[0]
    job = db.execute("SELECT payload FROM job WHERE project_id=? AND kind='m3_plan' ORDER BY created_at DESC LIMIT 1",
                     (project,)).fetchone()
    if job:
        snapshot = json.loads(job[0])["approval_snapshot"]
    else:
        key = db.execute("SELECT a.storage_key FROM m2_approval m JOIN artifact a ON a.id=m.artifact_id "
                         "WHERE m.project_id=? ORDER BY m.created_at DESC LIMIT 1", (project,)).fetchone()[0]
        snapshot = json.loads((ROOT / "data" / "artifacts" / key).read_text(encoding="utf-8"))
    draft = db.execute("SELECT content FROM planning_draft WHERE project_id=? AND content IS NOT NULL "
                       "ORDER BY updated_at DESC LIMIT 1", (project,)).fetchone()
    content = json.loads(draft[0]) if draft else {}
    count = (content.get("chapter_count") or len(content.get("plot", {}).get("chapters", []))
             or snapshot["world"]["result"]["chapterCount"])
    snapshot = copy.deepcopy(snapshot)
    snapshot["world"]["result"]["chapterCount"] = count
    value = {"title": title, "project_id": project, "approval_snapshot": snapshot, "profile": PROFILE,
             "script_options": {**OPTIONS, "cast_limit": 8}}
    if content.get("cast_plan"):
        value["cast_plan"] = content["cast_plan"]
    path = works / "inputs" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", nargs=2, metavar=("PROJECT_ID", "NAME"))
    parser.add_argument("--works", type=Path, help="With --export: the experiment folder")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--cast-from", type=Path, help="A run whose cast-plan.json this run reuses")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--script", action="store_true", help="Go on from the plan to the chapter scripts")
    args = parser.parse_args()
    if args.export:
        print(export(*args.export, args.works))
        return 0
    if not (args.input and args.output_dir and args.arm):
        parser.error("--input, --output-dir and --arm are needed to plan.")
    value = json.loads(args.input.read_text(encoding="utf-8"))
    plan = value.pop("cast_plan", None)
    if args.cast_from:
        plan = json.loads((args.cast_from / "cast-plan.json").read_text(encoding="utf-8"))["plan"]
    use_arm(args.arm, CastPlan.model_validate(plan) if plan else None)
    profile = value["profile"]
    load_config = pipeline.load_config
    pipeline.load_config = lambda: select_config(load_config(), {"profile": profile}, pipeline.ROOT)
    report = run_text_debug({key: value[key] for key in ("approval_snapshot", "profile", "script_options")},
                            args.output_dir, workflow="causal", resume=args.resume, seed=args.seed,
                            policy="script_continuation_v1", plot_only=not args.script)
    if (args.output_dir / "plot.json").exists():
        cast_path = args.output_dir / "cast-plan.json"
        cast_plan = json.loads(cast_path.read_text(encoding="utf-8"))["plan"] if cast_path.exists() else plan
        markdown = f"# {value['title']}（{args.arm}）\n\n" + plan_markdown(args.output_dir, names(value, cast_plan))
        (args.output_dir / "drama-plan.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"status": report["status"], "plan": str(args.output_dir.resolve() / "drama-plan.md")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
