"""
Entity selection strategies for the VG-Cap ablation study (Table 3, panel (b)).

Every selector is built on the SAME candidate set E as the heuristic scorer --
spaCy NER over the article, unioned with the LLM keyword list, then
substring-aware deduplication -- and differs only in how that set is ranked and
cut.  That is exactly what panel (b) needs to isolate: whether the gain comes
from the scoring design or merely from restricting how many entities reach the
generator.

Budget matching
---------------
A fixed K_e=10 is NOT a matched budget: article-order and random always return
10 entities, whereas VG-Cap returns 6.76 on average because tau also applies.
`budget_map` fixes that by giving each baseline exactly as many entities as
VG-Cap selected for that same query, so no baseline can win by being more
verbose.  Build it with `load_budget_map(...)` from a reference run.
"""
from __future__ import annotations

import json
import random
from typing import Dict, List, Optional, Sequence, Tuple

from step_2_entity_scorer import Entity, EntityHeuristicScorer

CAPTION_PRESENCE_FEATURES: Tuple[str, ...] = (
    "exact_match_crawl",
    "exact_match_generated",
    "partial_match_crawl",
    "partial_match_generated",
)
CONTEXT_PRESENCE_FEATURES: Tuple[str, ...] = CAPTION_PRESENCE_FEATURES + (
    "substring_in_qa",
    "substring_in_summary",
)
TYPE_FEATURES: Tuple[str, ...] = (
    "type_person",
    "type_gpe",
    "type_org",
    "type_loc",
    "type_date",
    "type_event",
    "type_other",
)


def build_signals(query_data: dict) -> dict:
    """Mirror of the signal dict assembled inside EntityHeuristicScorer.score_query."""
    return {
        "crawl_caption": query_data.get("crawl_caption", ""),
        "generated_caption": query_data.get("generated_caption", ""),
        "question_answer": query_data.get("question_answer", ""),
        "summary": query_data.get("article_summary", {}).get("fact_summary", ""),
    }


def first_mention(entity: Entity, article: str) -> int:
    """Character offset of the first mention; len(article)+1 when not found.

    spaCy entities carry `start`, but entities parsed from the LLM keyword
    string always have start=0, so they would all look like position 0.  We
    look those up in the article text instead.
    """
    if entity.start and entity.start > 0:
        return entity.start
    idx = article.lower().find(entity.text.lower())
    return idx if idx >= 0 else len(article) + 1


class PaperSpanPenaltyScorer(EntityHeuristicScorer):
    """DEPRECATED -- kept so older configs keep working.

    The span penalty now lives in the base scorer, driven by
    `fixes.span_penalty` ("off" | "code" | "paper").  This subclass only
    translates the old `thresholds.length_penalty_excess_only` flag into the new
    switch; it adds no behaviour of its own, and in particular it must NOT set
    the `length_penalty` feature any more -- the base class would then count the
    penalty twice.
    """

    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path=config_path)
        if self.thresholds.get("length_penalty_excess_only"):
            self.fixes["span_penalty"] = "paper"


def make_scorer(config_path: Optional[str] = None) -> EntityHeuristicScorer:
    """Return the scorer class implied by the config.

    `fixes.span_penalty` is the supported route; `length_penalty_excess_only`
    is the legacy spelling of `fixes.span_penalty = "paper"`.
    """
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("thresholds", {}).get("length_penalty_excess_only"):
            return PaperSpanPenaltyScorer(config_path=config_path)
    return EntityHeuristicScorer(config_path=config_path)


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------
class BaseSelector:
    """Common candidate construction and budget handling."""

    name = "base"
    #: whether `select` applies the admission threshold tau
    applies_threshold = False

    def __init__(
        self,
        scorer: EntityHeuristicScorer,
        budget: int = 10,
        budget_map: Optional[Dict[str, int]] = None,
        seed: int = 0,
    ):
        self.scorer = scorer
        self.budget = budget
        self.budget_map = budget_map
        self.seed = seed

    # -- helpers ------------------------------------------------------------
    def _candidates(self, query_data: dict) -> Tuple[List[Entity], str]:
        article = query_data.get("article", "")
        spacy_entities = self.scorer.extract_entities_from_article(article)
        parsed = self.scorer.parse_entity_string(query_data.get("name_entity_keyword", ""))
        return self.scorer.deduplicate_entities(spacy_entities, parsed), article

    def _scored(self, query_data: dict) -> Tuple[List[Entity], str]:
        entities, article = self._candidates(query_data)
        scored = self.scorer.score_entities(entities, build_signals(query_data), article)
        return scored, article

    def _budget(self, query_data: dict) -> int:
        if self.budget_map is not None:
            qid = query_data.get("query_id")
            if qid in self.budget_map:
                return int(self.budget_map[qid])
        return self.budget

    def _pack(
        self,
        chosen: Sequence[Entity],
        candidates: Sequence[Entity],
        budget: int,
    ) -> Tuple[List[str], dict]:
        selected_ids = {id(e) for e in chosen}
        debug = {
            "selector": self.name,
            "budget_used": budget,
            "threshold_used": self.scorer.thresholds["min_score"] if self.applies_threshold else None,
            "total_entities_extracted": len(candidates),
            "entities_selected": len(chosen),
            "top_entities": [e.text for e in chosen],
            "rejected_entities": [e.text for e in candidates if id(e) not in selected_ids],
            "scored_entities": [
                {
                    "entity": e.text,
                    "label": e.label,
                    "score": round(e.score, 2),
                    # per-feature breakdown: Appendix D and make_qualitative.py need it
                    "features": {k: round(v, 2) for k, v in e.features.items()},
                    "selected": id(e) in selected_ids,
                }
                for e in candidates
            ],
        }
        return [e.text for e in chosen], debug

    # -- interface ----------------------------------------------------------
    def select(self, query_data: dict) -> Tuple[List[str], dict]:
        raise NotImplementedError


class HeuristicSelector(BaseSelector):
    """VG-Cap. Reproduces EntityHeuristicScorer.filter_and_rank exactly.

    The sort is stable and keyed on score alone, as in the shipped code, so ties
    keep candidate order -- do not 'improve' this tie-break or the row stops
    reproducing the 0.6219 submission.

    Scorer repairs (keyword parsing, word-boundary matching, span penalty,
    substring dedup, first-mention position, candidate filtering) are driven by
    the `fixes` block of the config, so this class needs no change to run either
    the published or the repaired scorer.
    """

    name = "heuristic"
    applies_threshold = True

    def select(self, query_data):
        scored, _ = self._scored(query_data)
        budget = self._budget(query_data)
        keep = [e for e in scored if e.score >= self.scorer.thresholds["min_score"]]
        ranked = sorted(keep, key=lambda e: e.score, reverse=True)
        return self._pack(ranked[:budget], scored, budget)


class ArticleOrderSelector(BaseSelector):
    """First N entities by first mention -- the journalistic front-loading prior alone."""

    name = "article_order"

    def select(self, query_data):
        scored, article = self._scored(query_data)
        budget = self._budget(query_data)
        ranked = sorted(scored, key=lambda e: first_mention(e, article))
        return self._pack(ranked[:budget], scored, budget)


class RandomSelector(BaseSelector):
    """Lower bound: N entities drawn uniformly from the candidate set.

    The RNG is keyed on (seed, query_id) so a draw is reproducible and the five
    draws of Table 3(b) are independent.
    """

    name = "random"

    def select(self, query_data):
        scored, _ = self._scored(query_data)
        budget = self._budget(query_data)
        rng = random.Random(f"{self.seed}:{query_data.get('query_id', '')}")
        chosen = rng.sample(scored, min(budget, len(scored)))
        return self._pack(chosen, scored, budget)


class PresenceOnlySelector(BaseSelector):
    """Caption match only: an entity must actually occur in the editorial or
    generated caption. Isolates the strongest single signal and doubles as the
    'presence-required' variant promised in Section 3.4."""

    name = "presence_only"

    def select(self, query_data):
        scored, _ = self._scored(query_data)
        budget = self._budget(query_data)
        keep = [
            e for e in scored
            if any(e.features.get(f, 0.0) > 0.0 for f in CAPTION_PRESENCE_FEATURES)
        ]
        ranked = sorted(keep, key=lambda e: e.score, reverse=True)
        return self._pack(ranked[:budget], scored, budget)


class TypeOnlySelector(BaseSelector):
    """Depictability prior only: rank by entity type, ignore all evidence.

    tau is deliberately NOT applied here -- under type weights alone only PERSON
    reaches 3.0, so a threshold would collapse the selection to people and the
    row would no longer measure the type prior as a *ranking* signal.
    """

    name = "type_only"

    def select(self, query_data):
        scored, article = self._scored(query_data)
        budget = self._budget(query_data)

        def type_score(entity: Entity) -> float:
            return sum(
                self.scorer.weights.get(f, 0.0) * entity.features.get(f, 0.0)
                for f in TYPE_FEATURES
            )

        ranked = sorted(scored, key=lambda e: (-type_score(e), first_mention(e, article)))
        return self._pack(ranked[:budget], scored, budget)


class PresenceScoreSelector(BaseSelector):
    """Contextual presence only (Table 3(b) of new_main.tex): rank by the
    presence-family score alone -- editorial/generated caption, QA pairs,
    summary -- ignoring position, type and span.

    Entities with no presence evidence at all are not admitted; otherwise the
    remaining slots would be filled in candidate order and the row would partly
    measure article order instead.  tau is not applied, mirroring
    TypeOnlySelector.  Ties are broken by first mention.
    """

    name = "presence_score"

    def select(self, query_data):
        scored, article = self._scored(query_data)
        budget = self._budget(query_data)

        def presence_score(entity: Entity) -> float:
            return sum(
                self.scorer.weights.get(f, 0.0) * entity.features.get(f, 0.0)
                for f in CONTEXT_PRESENCE_FEATURES
            )

        keep = [e for e in scored if presence_score(e) > 0.0]
        ranked = sorted(keep, key=lambda e: (-presence_score(e), first_mention(e, article)))
        return self._pack(ranked[:budget], scored, budget)


SELECTORS = {
    cls.name: cls
    for cls in (
        HeuristicSelector,
        ArticleOrderSelector,
        RandomSelector,
        PresenceOnlySelector,
        PresenceScoreSelector,
        TypeOnlySelector,
    )
}


def build_selector(
    name: str,
    config_path: Optional[str] = None,
    budget: int = 10,
    budget_map: Optional[Dict[str, int]] = None,
    seed: int = 0,
    coefficients_path: Optional[str] = None,
) -> "BaseSelector":
    if name not in SELECTORS:
        raise ValueError(f"unknown selector '{name}'; choose from {sorted(SELECTORS)}")
    kwargs = dict(scorer=make_scorer(config_path), budget=budget,
                  budget_map=budget_map, seed=seed)
    if name == "learned":
        kwargs["coefficients_path"] = coefficients_path
    return SELECTORS[name](**kwargs)


def load_budget_map(path: str) -> Dict[str, int]:
    """Per-query entity budget taken from a reference run.

    Accepts either runs/<tag>/selected_entities.json ({qid: [entities]}) or
    assemble_result/entity_heuristic_scores.json ({qid: {entities_selected: n}}).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    budget_map: Dict[str, int] = {}
    for qid, value in data.items():
        if isinstance(value, list):
            budget_map[qid] = len(value)
        elif isinstance(value, dict) and "entities_selected" in value:
            budget_map[qid] = int(value["entities_selected"])
        elif isinstance(value, dict) and "top_entities" in value:
            budget_map[qid] = len(value["top_entities"])
        else:
            raise ValueError(f"cannot read a budget for query '{qid}' from {path}")
    return budget_map


class LearnedSelector(BaseSelector):
    """Appendix B: logistic regression over the same binary features.

    Tests whether the hand-set weights sit near a data-driven optimum.  Load
    coefficients produced by scripts/train_learned_selector.py:

        {"intercept": -2.1, "coefficients": {"exact_match_crawl": 3.4, ...},
         "threshold": 0.5}
    """

    name = "learned"

    def __init__(self, scorer, budget=10, budget_map=None, seed=0, coefficients_path=None):
        super().__init__(scorer, budget, budget_map, seed)
        if not coefficients_path:
            raise ValueError("LearnedSelector needs --learned_coefficients")
        with open(coefficients_path, "r", encoding="utf-8") as f:
            model = json.load(f)
        self.intercept = float(model["intercept"])
        self.coefficients = model["coefficients"]
        self.prob_threshold = float(model.get("threshold", 0.5))

    def _probability(self, entity: Entity) -> float:
        import math
        z = self.intercept + sum(
            w * entity.features.get(f, 0.0) for f, w in self.coefficients.items()
        )
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def select(self, query_data):
        scored, _ = self._scored(query_data)
        budget = self._budget(query_data)
        for entity in scored:
            entity.score = self._probability(entity)
        keep = [e for e in scored if e.score >= self.prob_threshold]
        ranked = sorted(keep, key=lambda e: e.score, reverse=True)
        return self._pack(ranked[:budget], scored, budget)


SELECTORS[LearnedSelector.name] = LearnedSelector


def build_learned_selector(config_path=None, budget=10, budget_map=None, seed=0,
                           coefficients_path=None) -> LearnedSelector:
    return LearnedSelector(
        scorer=make_scorer(config_path), budget=budget, budget_map=budget_map,
        seed=seed, coefficients_path=coefficients_path)
