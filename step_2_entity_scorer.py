"""
Entity Heuristic Scoring System
Dánh giá mức dộ liên quan của entities với visual content.

LEGACY vs FIXED
---------------
The weights and thresholds below reproduce the published VG-Cap run (Overall
0.6219).  That run contains four defects, documented one by one in `DEFAULTS`
below.  Each is repaired by a switch in the optional `fixes` block of the
config, and EVERY switch defaults to the legacy behaviour, so a config without
a `fixes` block scores exactly as the submission did.

    configs/entity_scoring_config.json        as-run   (all fixes off)
    configs/entity_scoring_config_fixed.json  repaired (all fixes on)

Switches
    parse_keyword_markdown  the LLM keyword list is a markdown bullet list, but
                            parse_entity_string split it on commas, so the whole
                            list entered E as 1-3 unsplit blobs of ~300 chars.
    word_boundary_match     presence matching used raw substring containment, so
                            'One' matched "someone" and outscored the person in
                            the photograph.  Switch to word-boundary matching.
    span_penalty            'off'   P_span is never applied: compute_score looks
                                    up the feature key 'length_penalty' while the
                                    weights declare 'length_penalty_factor'.
                            'code'  -0.1 * |e|          for |e| > 3   (what the
                                    code structure intended)
                            'paper' -0.1 * max(0,|e|-3) (Eq. (2) of the paper)
    substring_dedup         deduplication merged longer surface forms only
                            between the keyword list and spaCy, never between two
                            spaCy entities, so 'Michelle Payne' and 'Payne' both
                            survived.
    first_mention_position  start == 0 was used as the sentinel for 'no position
                            information', which silently denied the position
                            bonus to every keyword-derived candidate and to any
                            entity starting at character 0.
    drop_non_depictable     candidates whose spaCy label is non-depictable
                            (CARDINAL, ORDINAL, ...) or which carry no letter at
                            all ('1', '10') are not entity candidates.
"""

import json
import re
import string
from typing import Dict, List, Optional, Tuple


#: Legacy behaviour of every switch.  Do not change these: a config without a
#: `fixes` block must keep reproducing the published submission.
DEFAULT_FIXES: Dict[str, object] = {
    "parse_keyword_markdown": False,
    "word_boundary_match": False,
    "span_penalty": "off",            # off | code | paper
    "substring_dedup": False,
    "first_mention_position": False,
    "drop_non_depictable": False,
    "excluded_labels": ["CARDINAL", "ORDINAL", "PERCENT", "MONEY",
                        "QUANTITY", "TIME", "LANGUAGE"],
}

SPAN_PENALTY_MODES = ("off", "code", "paper")

#: markdown decorations the LLM wraps its keyword list in
_BULLET = re.compile(r"^\s*(?:[-*•·]|\d+[.)])\s+")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_TRAILING_PAREN = re.compile(r"\s*\([^()]*\)\s*$")


class Entity:
    """Entity data structure"""
    def __init__(self, text: str, label: str = "", start: int = 0):
        self.text = text.strip()
        self.label = label  # PERSON, LOC, ORG, etc.
        self.start = start  # Position in article
        self.score = 0.0
        self.features = {}
        self.selected = False

    def __repr__(self):
        return f"Entity('{self.text}', {self.label}, score={self.score:.2f})"


class EntityHeuristicScorer:
    """
    Rule-based scoring system cho entity relevance.
    spaCy is imported lazily so that the parsing / matching / scoring logic can
    be unit-tested without the model installed.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: Path to JSON config file with weights/thresholds/fixes
        """
        self._nlp = None  # lazy: see .nlp

        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.weights = config['weights']
            self.thresholds = config['thresholds']
            fixes = config.get('fixes', {})
        else:
            self.weights = self._get_default_weights()
            self.thresholds = self._get_default_thresholds()
            fixes = {}

        self.fixes = dict(DEFAULT_FIXES)
        self.fixes.update(fixes)

        mode = self.fixes.get("span_penalty", "off")
        if mode not in SPAN_PENALTY_MODES:
            raise ValueError(
                f"fixes.span_penalty must be one of {SPAN_PENALTY_MODES}, got {mode!r}")
        self.excluded_labels = set(self.fixes.get("excluded_labels") or [])

    # -- spaCy ---------------------------------------------------------------
    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def _get_default_weights(self) -> dict:
        """
        Default heuristic weights
        Dựa trên phân tích ở phần design
        """
        return {
            # Presence features (cao nhất - direct evidence)
            'exact_match_crawl': 10.0,      # Entity trong crawl_caption
            'exact_match_generated': 8.0,    # Entity trong generated_caption
            'partial_match_crawl': 6.0,      # Partial match trong crawl
            'partial_match_generated': 4.0,  # Partial match trong generated
            'substring_in_qa': 3.0,          # Trong Q&A pairs
            'substring_in_summary': 1.5,     # Trong article summary

            # Position features (entity position trong article)
            'early_position_bonus': 2.0,     # Trong 20% đầu article
            'mid_position_bonus': 1.0,       # Trong 20-50% article

            # Type features (entity type priority)
            'type_person': 3.0,              # PERSON entities
            'type_gpe': 2.5,                 # Geo-political entities (cities, countries)
            'type_org': 2.0,                 # Organizations
            'type_loc': 2.0,                 # Locations
            'type_date': 1.0,                # Dates
            'type_event': 2.5,               # Events
            'type_other': 0.5,               # Other types

            # Length penalty (avoid very long entities)
            # NOTE: only read when fixes.span_penalty != "off"
            'length_penalty_factor': -0.1,
        }

    def _get_default_thresholds(self) -> dict:
        """Default filtering thresholds"""
        return {
            'min_score': 3.0,        # Minimum score to be selected
            'top_k': 10,             # Maximum entities to select
            'min_entity_length': 2,  # Minimum characters in entity
        }

    # -- candidate construction ---------------------------------------------
    def extract_entities_from_article(self, article: str) -> List[Entity]:
        """
        Extract entities từ article text using spaCy
        """
        doc = self.nlp(article)
        entities = []

        for ent in doc.ents:
            if not self._is_admissible(ent.text, ent.label_):
                continue
            entities.append(Entity(text=ent.text, label=ent.label_, start=ent.start_char))

        return entities

    def _is_admissible(self, text: str, label: str) -> bool:
        """Reject candidates that cannot denote something depictable.

        Legacy behaviour accepts everything, which is how '1', '10' and 'One'
        reached the top of E*.
        """
        if not self.fixes.get("drop_non_depictable"):
            return True
        text = text.strip()
        if len(text) < int(self.thresholds.get('min_entity_length', 2)):
            return False
        if label in self.excluded_labels:
            return False
        if not any(ch.isalpha() for ch in text):      # '1', '10', '2015'
            return False
        return True

    def parse_entity_string(self, entity_string: str) -> List[str]:
        """
        Parse the LLM keyword field.

        Legacy: split on commas.  The field is in fact a markdown bullet list
        with a preamble, so the comma split returns 1-3 unsplit blobs of several
        hundred characters, which then score as if they were entities.
        Enable fixes.parse_keyword_markdown for the real parse.
        """
        if not entity_string:
            return []

        if not self.fixes.get("parse_keyword_markdown"):
            entities = [e.strip() for e in entity_string.split(',')]
            return [e for e in entities
                    if len(e) >= self.thresholds['min_entity_length']]

        return self._parse_markdown_entities(entity_string)

    def _parse_markdown_entities(self, entity_string: str) -> List[str]:
        """Markdown bullet list -> list of entity surface forms.

        Handles the shape llama3 actually emits:

            Here is the list of named entities ...:

            **People**:

            * Michelle Payne
            * Stevie Payne (Michelle's brother)

            **Organizations**:

            * CNN
        """
        min_len = int(self.thresholds.get('min_entity_length', 2))
        out: List[str] = []

        for raw in entity_string.splitlines():
            line = raw.strip()
            if not line:
                continue

            bulleted = bool(_BULLET.match(line))
            line = _BULLET.sub("", line, count=1)

            # a type header such as '**People**:' -- never an entity
            stripped = _BOLD.sub(r"\1", line).strip()
            if stripped.endswith(":"):
                continue
            # the conversational preamble ('Here is the list ...')
            if not bulleted and len(stripped.split()) > 8:
                continue

            # A bulleted line holds exactly one entity, so commas inside it
            # belong to the entity ('November 3, 2015').  Only an unbulleted
            # line is treated as a comma-separated list.
            pieces = [stripped] if bulleted else stripped.split(",")
            for piece in pieces:
                piece = _BOLD.sub(r"\1", piece).strip()
                piece = _TRAILING_PAREN.sub("", piece).strip()
                piece = piece.strip(string.whitespace + '"“”')
                piece = piece.rstrip(".;")
                if len(piece) < min_len:
                    continue
                if not any(ch.isalpha() for ch in piece):
                    continue
                out.append(piece)

        # order-preserving dedup
        seen, unique = set(), []
        for item in out:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def deduplicate_entities(self, entities: List[Entity],
                             parsed_strings: List[str]) -> List[Entity]:
        """
        Merge entities từ spaCy và parsed strings.
        Handle duplicates như "Biden" vs "Joe Biden".
        """
        if self.fixes.get("substring_dedup"):
            return self._deduplicate_substring_aware(entities, parsed_strings)

        entity_map = {}

        # Add spaCy entities
        for ent in entities:
            key = ent.text.lower()
            if key not in entity_map:
                entity_map[key] = ent

        # Add parsed entities (if not already covered)
        for text in parsed_strings:
            key = text.lower()

            # Check if this is a substring of existing entity
            is_duplicate = False
            for existing_key in list(entity_map.keys()):
                if key in existing_key or existing_key in key:
                    # Prefer longer entity
                    if len(key) > len(existing_key):
                        del entity_map[existing_key]
                        entity_map[key] = Entity(text=text, label="PARSED")
                    is_duplicate = True
                    break

            if not is_duplicate:
                entity_map[key] = Entity(text=text, label="PARSED")

        return list(entity_map.values())

    def _deduplicate_substring_aware(self, entities: List[Entity],
                                     parsed_strings: List[str]) -> List[Entity]:
        """Substring-aware merge over ALL candidates, keeping the longest
        surface form -- what Section 3.4 describes.

        The surviving entity inherits the earliest character offset and the most
        informative label seen among the forms it absorbed, so merging never
        destroys position or type evidence.
        """
        candidates: List[Entity] = list(entities)
        for text in parsed_strings:
            if self._is_admissible(text, "PARSED"):
                candidates.append(Entity(text=text, label="PARSED"))

        # longest first, so a short form is always absorbed by its longer form
        candidates.sort(key=lambda e: (-len(e.text), e.start))

        kept: List[Entity] = []
        for cand in candidates:
            key = cand.text.lower()
            if not key:
                continue
            host = None
            for existing in kept:
                ekey = existing.text.lower()
                if key == ekey or self._is_word_subsequence(key, ekey):
                    host = existing
                    break
            if host is None:
                kept.append(cand)
                continue
            # merge evidence into the longer form we already keep
            if cand.start and (not host.start or cand.start < host.start):
                host.start = cand.start
            if host.label in ("", "PARSED") and cand.label not in ("", "PARSED"):
                host.label = cand.label

        return kept

    @staticmethod
    def _is_word_subsequence(short: str, long: str) -> bool:
        """True when `short` occurs inside `long` on word boundaries.

        Prevents 'Ann' from being merged into 'Hannah' while still merging
        'Payne' into 'Michelle Payne'.
        """
        if len(short) >= len(long):
            return False
        return re.search(rf"(?<!\w){re.escape(short)}(?!\w)", long) is not None

    # -- features ------------------------------------------------------------
    def _contains(self, haystack: str, needle: str) -> bool:
        """Containment test used by every presence feature.

        Legacy: raw substring, so 'one' matches 'someone'.
        Fixed:  word-boundary match.
        """
        if not needle or not haystack:
            return False
        if not self.fixes.get("word_boundary_match"):
            return needle in haystack
        return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None

    def calculate_presence_features(self, entity: Entity, signals: dict) -> dict:
        """
        Calculate presence-based features

        Args:
            entity: Entity object
            signals: Dict with keys: crawl_caption, generated_caption, question_answer, summary
        """
        features = {}
        entity_lower = entity.text.lower()
        entity_words = set(entity_lower.split())

        # Exact match in crawl_caption (highest signal)
        crawl = signals.get('crawl_caption', '').lower()
        features['exact_match_crawl'] = 1.0 if self._contains(crawl, entity_lower) else 0.0

        # Partial match in crawl (any word matches)
        if features['exact_match_crawl'] == 0.0:
            features['partial_match_crawl'] = 1.0 if any(
                self._contains(crawl, word) for word in entity_words if len(word) > 2) else 0.0
        else:
            features['partial_match_crawl'] = 0.0

        # Exact match in generated_caption
        generated = signals.get('generated_caption', '').lower()
        features['exact_match_generated'] = 1.0 if self._contains(generated, entity_lower) else 0.0

        # Partial match in generated
        if features['exact_match_generated'] == 0.0:
            features['partial_match_generated'] = 1.0 if any(
                self._contains(generated, word) for word in entity_words if len(word) > 2) else 0.0
        else:
            features['partial_match_generated'] = 0.0

        # Substring in Q&A
        qa = signals.get('question_answer', '').lower()
        features['substring_in_qa'] = 1.0 if self._contains(qa, entity_lower) else 0.0

        # Substring in summary
        summary = signals.get('summary', '').lower()
        features['substring_in_summary'] = 1.0 if self._contains(summary, entity_lower) else 0.0

        return features

    def first_mention(self, entity: Entity, article: str) -> int:
        """Character offset of the first mention, or -1 when absent.

        Legacy used `start == 0` as the sentinel for 'unknown', which denied the
        position bonus to every keyword-derived candidate.
        """
        if entity.start and entity.start > 0:
            return entity.start
        if not article:
            return -1
        idx = article.lower().find(entity.text.lower())
        return idx if idx >= 0 else -1

    def calculate_position_features(self, entity: Entity, article: str) -> dict:
        """
        Calculate position-based features
        Entity position trong article text
        """
        features = {'early_position_bonus': 0.0, 'mid_position_bonus': 0.0}

        if self.fixes.get("first_mention_position"):
            start = self.first_mention(entity, article)
            if start < 0:
                return features
        else:
            if entity.start == 0:
                # Không có position info (parsed entity)
                return features
            start = entity.start

        article_length = len(article)
        position_ratio = start / article_length if article_length > 0 else 0.5

        if position_ratio < 0.2:    # First 20%
            features['early_position_bonus'] = 1.0
        elif position_ratio < 0.5:  # 20-50%
            features['mid_position_bonus'] = 1.0

        return features

    def calculate_type_features(self, entity: Entity) -> dict:
        """
        Calculate entity type-based features
        """
        features = {}

        # Map entity label to feature
        type_map = {
            'PERSON': 'type_person',
            'GPE': 'type_gpe',
            'ORG': 'type_org',
            'LOC': 'type_loc',
            'DATE': 'type_date',
            'EVENT': 'type_event',
        }

        feature_name = type_map.get(entity.label, 'type_other')

        # Set all to 0, then activate the matching one
        for key in type_map.values():
            features[key] = 0.0
        features['type_other'] = 0.0

        features[feature_name] = 1.0

        # Length penalty.  Legacy keeps the raw word count here and the generic
        # weighted sum silently drops it (no 'length_penalty' weight exists);
        # when fixes.span_penalty is on, _span_penalty() owns the term instead
        # and this feature is zeroed so it cannot be counted twice.
        word_count = len(entity.text.split())
        if self.fixes.get("span_penalty", "off") != "off":
            features['length_penalty'] = 0.0
        elif word_count > 3:
            features['length_penalty'] = word_count
        else:
            features['length_penalty'] = 0

        return features

    def _span_penalty(self, entity: Entity) -> float:
        """P_span(e); zero unless fixes.span_penalty is 'code' or 'paper'."""
        mode = self.fixes.get("span_penalty", "off")
        if mode == "off":
            return 0.0
        weight = self.weights.get(
            'length_penalty_factor', self.weights.get('length_penalty', 0.0))
        n_words = len(entity.text.split())
        excess = max(0, n_words - 3) if mode == "paper" else (n_words if n_words > 3 else 0)
        return weight * excess

    def compute_score(self, features: dict) -> float:
        """
        Compute weighted score từ features
        """
        score = 0.0

        for feature_name, feature_value in features.items():
            weight = self.weights.get(feature_name, 0.0)
            score += weight * feature_value

        return max(0.0, score)  # Non-negative scores

    def score_entities(self, entities: List[Entity], signals: dict,
                       article: str) -> List[Entity]:
        """
        Score all entities and update their scores
        """
        for entity in entities:
            all_features = {}

            all_features.update(self.calculate_presence_features(entity, signals))
            all_features.update(self.calculate_position_features(entity, article))
            all_features.update(self.calculate_type_features(entity))

            penalty = self._span_penalty(entity)
            entity.features = dict(all_features)
            if penalty:
                entity.features['span_penalty_applied'] = round(penalty, 4)
            entity.score = max(0.0, self.compute_score(all_features) + penalty)

        return entities

    def filter_and_rank(self, entities: List[Entity]) -> List[Entity]:
        """
        Filter entities by threshold and rank by score
        """
        filtered = [e for e in entities if e.score >= self.thresholds['min_score']]
        ranked = sorted(filtered, key=lambda e: e.score, reverse=True)
        top_k = ranked[:self.thresholds['top_k']]

        for ent in top_k:
            ent.selected = True

        return ranked  # Return all ranked, but mark selected

    def score_query(self, query_data: dict) -> Tuple[List[str], dict]:
        """
        Main scoring function for a single query
        """
        article = query_data.get('article', '')

        signals = {
            'crawl_caption': query_data.get('crawl_caption', ''),
            'generated_caption': query_data.get('generated_caption', ''),
            'question_answer': query_data.get('question_answer', ''),
            'summary': query_data.get('article_summary', {}).get('fact_summary', '')
        }

        spacy_entities = self.extract_entities_from_article(article)
        parsed_entities = self.parse_entity_string(query_data.get('name_entity_keyword', ''))
        all_entities = self.deduplicate_entities(spacy_entities, parsed_entities)
        scored_entities = self.score_entities(all_entities, signals, article)
        ranked_entities = self.filter_and_rank(scored_entities)

        selected_texts = [e.text for e in ranked_entities if e.selected]

        debug_info = {
            'scored_entities': [
                {
                    'entity': e.text,
                    'score': round(e.score, 2),
                    'label': e.label,
                    'features': {k: round(v, 2) for k, v in e.features.items()},
                    'selected': e.selected
                }
                for e in ranked_entities
            ],
            'top_entities': selected_texts,
            'rejected_entities': [e.text for e in scored_entities if not e.selected],
            'threshold_used': self.thresholds['min_score'],
            'total_entities_extracted': len(all_entities),
            'entities_selected': len(selected_texts),
            'signals_used': list(signals.keys()),
            'fixes': self.fixes,
        }

        return selected_texts, debug_info


def save_scoring_results(results: dict, output_path: str):
    """Save entity scoring results to JSON for debugging"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Entity scoring results saved to: {output_path}")


def load_scoring_results(input_path: str) -> dict:
    """Load previously saved scoring results"""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)
