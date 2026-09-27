"""Future-only plot material, projected into the existing export contract."""

from pydantic import AliasChoices, Field

from packages.contracts.m3 import CharacterArc, Foreshadowing, OutlineChapter, StoryOutline
from packages.contracts.script import Contract, Identifier


class ChainCharacter(Contract):
    character_id: Identifier
    initial_behavior: str = Field(min_length=1)
    enduring_value: str = Field(min_length=1)
    final_behavior: str = Field(min_length=1, description="着地を示す本人の具体的な行動。")


class PlotCharacter(ChainCharacter):
    turning_experience: str = Field(min_length=1, description="出来事に含まれる経験の要約。")


class PlotCore(Contract):
    central_question: str = Field(min_length=1)
    external_resolution: str = Field(min_length=1)
    relationship_resolution: str = Field(min_length=1)
    characters: list[PlotCharacter] = Field(min_length=1, max_length=10)
    foreshadowing: list[Foreshadowing] = Field(max_length=30)


class CausalRoute(Contract):
    start_condition: str = Field(min_length=1, description="予定する開始条件。過去の実績ではない。")
    attempt: str = Field(min_length=1, description="人物が具体的に試すこと・起こる出来事。")
    consequence: str = Field(min_length=1, description="試みで分かること、相手の反応、判断を変える結果。")
    choice: str = Field(min_length=1, description="その結果を受けて誰がどう選ぶか。")
    next_state: str = Field(min_length=1, description="選択の結果、次章へ何が残るか。")
    core_progress: str = Field(min_length=1, description="核心の人物・関係の着地へどの転機を担うか。")


class ChainStep(Contract):
    character_id: Identifier
    action: str = Field(min_length=1, description="直前の状況・結果を受けて本人が実際に取る行動・選択。")
    result: str = Field(min_length=1, description="その行動で得る具体的な結果・発見・相手の反応。次の行動の根拠になる。")


class ChainEvent(Contract):
    start_condition: str = Field(min_length=1, description="直前の出来事から引き継ぐ状態。未実施の発見や解決を先取りしない。")
    steps: list[ChainStep] = Field(min_length=1, max_length=6,
        description="行動→結果→それを受けた次の行動、という順序。転機もここで実際に起こす。")

    def as_route(self) -> CausalRoute:
        return CausalRoute(start_condition=self.start_condition,
            attempt=self.steps[0].character_id + ": " + self.steps[0].action,
            consequence=self.steps[0].result,
            choice=" → ".join(step.character_id + ": " + step.action + " → " + step.result
                              for step in self.steps[1:]) or self.steps[0].action,
            next_state=self.steps[-1].result, core_progress=self.steps[-1].result)


class ChainCore(Contract):
    central_question: str = Field(min_length=1)
    external_resolution: str = Field(min_length=1,
        description="誰が何をした結果、障害や相手側のどの条件が変わり、外的な課題が決着するか。")
    relationship_resolution: str = Field(min_length=1, description="着地を表す当人たちの行動。感謝や理解の宣言で代用しない。")
    characters: list[ChainCharacter] = Field(min_length=1, max_length=10)


class StoryChain(Contract):
    """Persisted/export view; later starts are derived before saving new chains."""

    core: ChainCore
    events: list[ChainEvent] = Field(min_length=3, max_length=9,
        description="3章に収まる主要な出来事の連鎖。章境界はまだ決めない。各出来事は場面とは別。")


class ChainEventDraft(Contract):
    steps: list[ChainStep] = Field(min_length=1, max_length=6,
        description="直前までの行動の結果から続く行動→結果の順序。必要な移動・発見・関係の変化もここで起こす。")


class ResolutionBasis(Contract):
    """A small generative scaffold, never a second source of planned events."""

    earlier_experience: str = Field(min_length=1, max_length=180,
        description="先に誰が何を行い、どんな知識・方法・共有経験を得る予定か。行動と具体的な結果を1〜2文。")
    later_application: str = Field(min_length=1, max_length=180,
        description="先に得た結果の何を後の判断・行動に使い、どの状況を変える予定か。獲得とのつながりを1〜2文。")


class StoryChainDraft(Contract):
    """Generated chain with one opening, not independently invented later starts."""

    core: ChainCore
    resolution_basis: list[ResolutionBasis] = Field(max_length=2,
        description="先の経験と後の利用を対にする下書き。重要な決着や判断を最大2件、承認設定だけで成立し準備が不要なら空。下流へは出来事だけを渡す。")
    opening_condition: str = Field(min_length=1,
        description="作品冒頭の開始状態。承認設定を起点とし、これから描く出来事を実施済みにしない。")
    events: list[ChainEventDraft] = Field(min_length=3, max_length=9,
        description="3章に収まる主要な出来事の連鎖。全て予定であり、章境界はまだ決めない。")

    def as_chain(self) -> StoryChain:
        """Keep planned steps, not their scaffold; derive starts from results."""
        events = []
        condition = self.opening_condition
        for event in self.events:
            events.append(ChainEvent(start_condition=condition, steps=event.steps))
            condition = event.steps[-1].result
        return StoryChain(core=self.core, events=events)


class ConversationTopic(Contract):
    character_ids: list[Identifier] = Field(min_length=2, max_length=3)
    topic: str = Field(min_length=1, max_length=120,
        description="その場で本人が済ませたい用事・試したいことから始まる具体的な会話のきっかけ。1文。")
    exchange: str = Field(min_length=1, max_length=120,
        validation_alias=AliasChoices("exchange", "relationship_aspect"),
        description="用事を持ちかけた相手が自身の都合・好みから返し、それを受けて何をするか。全台詞ではなく1〜2文。")


class ChapterPresentation(Contract):
    title: str = Field(min_length=1, max_length=200)
    role: str = Field(min_length=1, max_length=200, description="当章で新しく経験することと、その結果変わる到達状態を1〜2文で記す。")
    conversation_topics: list[ConversationTopic] = Field(default_factory=list, max_length=4,
        description="会話の材料を2〜3件を目安に短く記す。主筋の会話だけで十分なら空配列。")


class ChapterBoundary(ChapterPresentation):
    number: int = Field(ge=1, le=3, strict=True)
    last_event: int = Field(ge=1, le=9, strict=True, description="当章の最後に割り当てる出来事の位置（1始まり）。")


class ChapterAllocation(Contract):
    chapters: list[ChapterBoundary] = Field(min_length=3, max_length=3)
    foreshadowing: list[Foreshadowing] = Field(max_length=30)


class FixedChapterAllocation(Contract):
    chapter_1: ChapterPresentation
    chapter_2: ChapterPresentation
    chapter_3: ChapterPresentation
    foreshadowing: list[Foreshadowing] = Field(max_length=30)

    def as_allocation(self) -> ChapterAllocation:
        # With three events and three nonempty chapters there is no boundary choice.
        return ChapterAllocation(chapters=[ChapterBoundary(
            number=number, last_event=number,
            **getattr(self, f"chapter_{number}").model_dump()) for number in (1, 2, 3)],
            foreshadowing=self.foreshadowing)


class PlotChapter(ChapterPresentation):
    number: int = Field(ge=1, le=100, strict=True)
    route: CausalRoute
    events: list[ChainEvent] = Field(default_factory=list, max_length=9)


class PlotBatch(Contract):
    chapters: list[PlotChapter] = Field(min_length=1, max_length=8)


class DetailedPlot(Contract):
    core: PlotCore
    chapters: list[PlotChapter] = Field(min_length=1, max_length=100)

    def as_outline(self) -> StoryOutline:
        """A view of the same plan; no second LLM version of the plot is generated."""
        return StoryOutline(
            ending=("【中心課題】" + self.core.central_question
                    + "\n【外的な決着】" + self.core.external_resolution
                    + "\n【人物・関係の着地】" + self.core.relationship_resolution),
            character_arcs=[CharacterArc(character_id=row.character_id, change=(
                "【開始時】" + row.initial_behavior + "【守る価値観】" + row.enduring_value
                + "【転機】" + row.turning_experience + "【着地の行動】" + row.final_behavior))
                for row in self.core.characters],
            chapters=[OutlineChapter(number=row.number, title=row.title, role=row.role,
                summary="\n".join(route.attempt + " → " + route.consequence + " → "
                                  + route.choice + " → " + route.next_state
                                  for route in ([event.as_route() for event in row.events]
                                                if row.events else [row.route]))) for row in self.chapters],
            foreshadowing=self.core.foreshadowing,
        )


class AttributedView(Contract):
    character_id: Identifier
    view: str = Field(min_length=1, description="その人物が述べた評価・推測・主張。内容が客観的に正しいとは限らない。")


class CurrentFacts(Contract):
    situation: str = Field(min_length=1, description="実台本の現在地。初章のみ承認済みの開始条件。")
    decisions: str = Field(min_length=1, description="実際の選択・受容・拒否。なければ未決とする。")
    unresolved: str = Field(min_length=1, description="本文に残る問題。不明な内心や予定で補完しない。")
    attributed_views: list[AttributedView] = Field(max_length=20,
        description="次の判断に関わる人物の評価・推測を発言者付きで分離する。なければ空配列。")


class RouteAdjustment(Contract):
    chapter_number: int = Field(ge=1, le=100, strict=True)
    reason_from_source: str = Field(min_length=1)
    route: CausalRoute
    events: list[ChainEvent] = Field(default_factory=list, max_length=9)


class PlotCastError(ValueError):
    """The core's character rows do not match the approved main-character slots."""


def bind_core_cast(schema: dict, characters: set[str], *, core: str, character: str):
    """Bind output cardinality as well as IDs; events may still use supporting cast."""
    ids = sorted(characters)
    definition = schema if schema.get("title") == core else schema["$defs"][core]
    definition["properties"]["characters"].update(
        minItems=len(ids), maxItems=len(ids),
        description=(f"承認メイン{len(ids)}人のみ、各IDを1件ずつ: {', '.join(ids)}。"
                     "同じ人物の複数の側面は1件にまとめ、サブキャラの変化はeventsで描く。"))
    schema["$defs"][character]["properties"]["character_id"]["enum"] = ids


def check_main_cast(rows, characters: set[str]):
    ids = [row.character_id for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != characters:
        duplicates = sorted({cid for cid in ids if ids.count(cid) > 1})
        raise PlotCastError("Plot core must identify the approved main cast exactly once. "
            f"Expected IDs (one row each): {sorted(characters)}; received: {ids}; "
            f"missing: {sorted(characters - set(ids))}; duplicate: {duplicates}; "
            f"unexpected: {sorted(set(ids) - characters)}. "
            "各メインの複数の側面を1件にまとめ、サブキャラの変化をメインのIDへ割り当てないでください。")


def check_core(core: PlotCore, characters: set[str], count: int):
    check_main_cast(core.characters, characters)
    if any(row.setup_chapter > count or row.payoff_chapter > count for row in core.foreshadowing):
        raise ValueError("Plot foreshadowing exceeds approved chapters.")


def check_chapters(chapters: list[PlotChapter], numbers: list[int]):
    if [row.number for row in chapters] != numbers:
        raise ValueError("Detailed plot must preserve the requested chapter count and order.")


def check_chain(chain: StoryChain, characters: set[str], available: set[str] | None = None):
    check_main_cast(chain.core.characters, characters)
    for event in chain.events:
        if any(step.character_id not in (available if available is not None else characters) for step in event.steps):
            raise ValueError("Chain steps must identify registered characters.")


def allocate_chain(chain: StoryChain, allocation: ChapterAllocation) -> DetailedPlot:
    """Partition once, in order, without rewriting or duplicating planned events."""
    if [row.number for row in allocation.chapters] != [1, 2, 3]:
        raise ValueError("Chapter allocation must preserve the three approved chapters.")
    ends = [row.last_event for row in allocation.chapters]
    if ends != sorted(set(ends)) or ends[-1] != len(chain.events):
        raise ValueError("Chapter boundaries must partition every event exactly once in order.")
    # Derive the legacy/export synopsis from events, never ask the model for a
    # second, independent promise of a turning point that may not be dramatized.
    core_data = chain.core.model_dump(exclude={"characters"})
    characters = [PlotCharacter(**row.model_dump(), turning_experience="\n".join(
        f"出来事{index}: {step.character_id}: {step.action} → {step.result}"
        for index, event in enumerate(chain.events, 1)
        for step in event.steps if step.character_id == row.character_id
    ) or "本人の転機は予定していない。") for row in chain.core.characters]
    core = PlotCore(**core_data, characters=characters, foreshadowing=allocation.foreshadowing)
    check_core(core, {row.character_id for row in chain.core.characters}, 3)
    chapters, start = [], 0
    for boundary in allocation.chapters:
        events = chain.events[start:boundary.last_event]
        # The route is an export-compatible view. Generation receives the exact
        # events instead, avoiding a second competing synopsis of the same story.
        routes = [event.as_route() for event in events]
        route = CausalRoute(start_condition=routes[0].start_condition,
            next_state=routes[-1].next_state,
            **{key: " → ".join(getattr(route, key) for route in routes)
               for key in ("attempt", "consequence", "choice", "core_progress")})
        chapters.append(PlotChapter(number=boundary.number, title=boundary.title,
            role=boundary.role, route=route, events=events, conversation_topics=boundary.conversation_topics))
        start = boundary.last_event
    return DetailedPlot(core=core, chapters=chapters)


def future_material(plot: DetailedPlot, plot_hash: str, number: int,
                    committed_plans: list[dict]) -> dict:
    """Apply only recorded future-route changes, never update the immutable core."""
    routes = {row.number: {**row.model_dump(mode="json"), "version": "plot:" + plot_hash}
              for row in plot.chapters}
    for record in sorted(committed_plans, key=lambda row: row["number"]):
        if record["number"] >= number:
            continue
        for change in record["plan"]["future_route_changes"]:
            target = change["chapter_number"]
            if target < number:
                continue
            routes[target] = {**routes[target], "route": change["route"],
                              "events": change.get("events", []),
                              "version": "chapter-plan:" + record["sha256"],
                              "reason_from_source": change["reason_from_source"]}
    for row in routes.values():
        if row["events"]:
            del row["route"]
            for event in row["events"]:
                event["planned_start_condition"] = event.pop("start_condition")
        else:
            del row["events"]
            row["route"]["planned_start_condition"] = row["route"].pop("start_condition")
    # Completed planned experiences must not come back through a core synopsis
    # as either accomplished facts or instructions to repeat an earlier event.
    core = plot.core.model_dump(mode="json", exclude={"characters": {"__all__": {"turning_experience"}}})
    return {"core": core, "core_version": plot_hash,
            "current_chapter": number,
            "routes": [routes[n] for n in sorted(routes) if n >= number]}


CHAIN_INSTRUCTION = (
    "会話中心のビジュアルノベルです。全3章の分量を支える試行・応答と、日常の掛け合いを用意します。"
    "承認設定と計画キャストの関係を使い、サブ自身の用事や関心も交流に生かします。"
    "core.charactersは承認メインのみ、stepsの行動主体は利用可能な全キャストです。"
    "出力はcore、resolution_basis、opening_condition、eventsの順です。"
    "まずcoreで外的な決着と人物・関係の着地を、本人の行動とその結果として定めます。"
    "外的な決着は、その行動が障害や相手側のどの条件を変えるかまで示します。"
    "次にresolution_basisで、先に得る経験と後の利用を対にして最大2件だけ具体化します。"
    "earlier_experienceは誰のどんな行動で何を得るか、later_applicationはその結果の何を"
    "後の判断・行動に使い、何を変えるかです。各1〜2文・180文字以内。"
    "危機の発生条件だけで済ませず、試した方法や、共に過ごして知った相手の事情などをつなぎます。"
    "承認設定だけで手段や判断が成立し、先行経験が不要なら空配列です。"
    "承認設定と矛盾する力や条件を便利な決着のために追加せず、作者だけが知る情報を本人の判断根拠にしません。"
    "その下書きを受けてeventsを作ります。先に行動して具体的な結果を得るstepと、"
    "後でその結果を使うstepを出来事列に含めます。下書きに書いたこと自体は出来事の実施を意味しません。"
    "主要な出来事の正本はevents.stepsだけです。全体をまだ章ごとに分けず、3〜9個の出来事へまとめます。"
    "この個数は場面数や均等配分を指定するものではありません。"
    "opening_conditionは承認設定に基づく作品冒頭の状態だけです。後続の開始事情は別に作りません。"
    "stepsは登録人物のactionと具体的なresultの順にし、その結果を本人が知る・経験して次の行動を選びます。"
    "後続の表示用開始状態は前eventの最後のresultから作られますが、それ以前の所在・約束・結果も継承します。"
    "後の目的に必要な移動・情報入手・関係変化は、経緯をstepsで描きます。未発生の事件を前提へ埋めません。"
    "3章の各区間で新しく経験することと到達状態が変わる連鎖にします。同じ状態の再確認を重ねて章数を埋めません。"
    "全動作の細分化や、毎回失敗・訓練・成功する型は不要です。"
    "重要な関係や喪失には、先に読者も共有できる約束・習慣・遊び・小さな失敗などを少数用意できます。"
    "日常の全要素を伏線にせず、主筋を進めない交流も場面化できる余地を残します。"
    "問い、提案、返答、選び直しも行動です。本人の判断が方法・結果・関係をどう変えるかを示し、"
    "『理解した』『成長した』という評価や他者の感謝だけで代用しません。"
    "既知の苦しみや協力約束を再演せず、その後の試行や相手の反応へ進みます。"
    "全員の行動量・成長・対立・和解は均等にせず、守る価値観と変わらない選択も保ちます。"
    "最後はcoreで約束した行動の結果まで描きます。準備や約束だけで完了扱いにしません。"
    "各stepは1〜2文。全台詞や細かな演出はまだ書きません。すべて未実施の予定です。"
)

PRESENTATION_INSTRUCTION = (
    "今回の回答は短い章説明です。roleは当章で新しく経験することと変わる到達状態を1〜2文・200文字以内です。"
    "conversation_topicsは会話の材料を2〜3件を目安に最大4件、主筋の会話で十分なら空配列です。"
    "各項目は登録済み人物2〜3人のcharacter_ids、具体的な会話のきっかけtopic、働きかけと反応exchangeです。"
    "topicはその場で本人が済ませたい用事・試したいことを1文、exchangeはそれを持ちかけた相手が"
    "自身の都合・好みから返し、その返答を受けて何をするかを1〜2文で記します。それぞれ120文字以内です。"
    "『設定を説明して相手が動揺する』『信頼が深まる』だけで済ませず、具体的な用事と応答を材料にします。"
    "サブ自身の用事や日常の遊びも交流の材料です。主筋への不可欠性だけで会話相手を選びません。"
    "全会話に反対・合意・成長を置かず、固定した三往復や全台詞の予定表にもせず、場面で応答を広げる余地を残します。"
    "既知の苦しみや協力約束の言い換えを繰り返さず、その後の試行や相手の反応を材料にします。全員分の日常のノルマはありません。"
    "台詞全文、細かな所作、光や視線の演出、読者の感情を指示する解説は書きません。"
    "主筋の選択・結果・結末は出来事の正本にだけ置き、会話材料へ別案や追加事件を作りません。"
    "完成本文の分量は章計画と執筆で確保し、この回答を長くして代用しません。"
    "foreshadowingは連鎖に既にある仕込みと結果だけを章番号へ対応させ、なければ空配列です。"
)

ALLOCATION_INSTRUCTION = PRESENTATION_INSTRUCTION + (
    "主筋の出来事を変更・重複させず、同じ結論の言い換え以外の応答や人物の接点を考えます。"
    "保存済みの出来事の連鎖を、順番と内容を保って全3章へ配分します。"
    "各出来事の結果や選択が一区切りになる箇所で章を区切ります。各章は連続した一つ以上の出来事を担当します。"
    "last_eventは入力のeventsの1始まりの位置です。全出来事を一度ずつ、抜け・重複・並べ替えなしで割り当てます。"
    "章数を増減せず、出来事を均等に三分割する必要もありません。"
    "『対立→一部合意→最終合意』などの型を先に当てはめず、実際の出来事から各章の役割と題名を付けます。"
    "章末の引きを作るために合意を取り消したり、未決事項・事件を追加したりしません。"
    "foreshadowingは連鎖に既にある仕込みと結果だけを章番号へ対応させ、なければ空配列です。"
)

PLOT_INSTRUCTION = (
    "未来の全体プロットを、核心coreから章ごとの因果chaptersへトップダウンで設計します。"
    "coreには中心課題、外的な決着、人物・関係の着地を分け、各主要人物について開始時の行動、"
    "守る価値観、行動を変える経験、最後に実際に取る行動を記します。"
    "各章のrouteは、開始条件→具体的な試み→結果や発見・相手の反応→誰の選択→次章へ残る状態です。"
    "何を試したから何が分かり、その結果なぜ選択が変わるのかを、各項目1〜2文で踏み込んで描きます。"
    "『対立する、説得する、成長する』だけで済ませず、判断を生む具体的な出来事を用意します。"
    "core_progressに当章が核心へ担う転機を記し、同じ説明・対立・合意を各章でやり直しません。"
    "合意後はその実行と結果を描けます。対立の激化、事件の追加、全員の成長や和解は必須ではありません。"
    "章ごとに独立した話へ戻らず、前章で予定した結果から次章が必要になる形にします。"
    "全台詞や細かな所作は決めず、場面で具体化する余地を残します。これらはすべて未実施の予定です。"
)

PLOT_PRIORITY = (
    "過去の事実の根拠は保存済み台本、未来の目的の根拠は全体プロットの核心と当章の到達点です。"
    "本文で起きたことを保ちつつ、未来は核心に定めた人物の行動・関係の着地へ向けて組み立てます。"
    "直前の本文の勢いをそのまま未来の方針にせず、まだ必要な転機を具体的な行動と結果で接続します。"
    "予定通り進めるために過去を改変したり、未実施の出来事を実績にしたりしません。"
    "planned_start_conditionは予定の開始条件です。実台本と異なる場合は実台本から出発し、必要な経緯を当章の場面で描くか、"
    "現在の実態から同じ目的へ進む手段を選びます。未発生の大事件を場面の開始状態へ埋めて済ませません。"
    "既に達成したことは再演せず、残る目標へ進めます。核心は変更せず、手段や順序だけを調整します。"
    "事実と核心を両立させるために、起きていない過去を捏造したり別の核心へ置換したりしません。"
)
