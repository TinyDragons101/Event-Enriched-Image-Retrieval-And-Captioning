"""
Entity Heuristic Scoring System
Đánh giá mức độ liên quan của entities với visual content
"""

import spacy
import json
from typing import List, Dict, Tuple
from collections import defaultdict
import re


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
    Rule-based scoring system cho entity relevance
    Dựa trên multiple heuristics từ different sources
    """

    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: Path to JSON config file with weights
        """
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")

        # Load or use default weights
        if config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.weights = config['weights']
            self.thresholds = config['thresholds']
        else:
            self.weights = self._get_default_weights()
            self.thresholds = self._get_default_thresholds()

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
            'length_penalty_factor': -0.1,   # Per word in entity
        }

    def _get_default_thresholds(self) -> dict:
        """Default filtering thresholds"""
        return {
            'min_score': 3.0,        # Minimum score to be selected
            'top_k': 10,             # Maximum entities to select
            'min_entity_length': 2,  # Minimum characters in entity
        }

    def extract_entities_from_article(self, article: str) -> List[Entity]:
        """
        Extract entities từ article text using spaCy
        """
        doc = self.nlp(article)
        entities = []

        for ent in doc.ents:
            entity = Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char
            )
            entities.append(entity)

        return entities

    def parse_entity_string(self, entity_string: str) -> List[str]:
        """
        Parse name_entity_keyword string
        Format có thể là: "Biden, White House, flood"
        """
        if not entity_string:
            return []

        # Split by comma
        entities = [e.strip() for e in entity_string.split(',')]

        # Filter empty
        entities = [e for e in entities if len(e) >= self.thresholds['min_entity_length']]

        return entities

    def deduplicate_entities(self, entities: List[Entity], parsed_strings: List[str]) -> List[Entity]:
        """
        Merge entities từ spaCy và parsed strings
        Handle duplicates như "Biden" vs "Joe Biden"
        """
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
        features['exact_match_crawl'] = 1.0 if entity_lower in crawl else 0.0

        # Partial match in crawl (any word matches)
        if features['exact_match_crawl'] == 0.0:
            features['partial_match_crawl'] = 1.0 if any(word in crawl for word in entity_words if len(word) > 2) else 0.0
        else:
            features['partial_match_crawl'] = 0.0

        # Exact match in generated_caption
        generated = signals.get('generated_caption', '').lower()
        features['exact_match_generated'] = 1.0 if entity_lower in generated else 0.0

        # Partial match in generated
        if features['exact_match_generated'] == 0.0:
            features['partial_match_generated'] = 1.0 if any(word in generated for word in entity_words if len(word) > 2) else 0.0
        else:
            features['partial_match_generated'] = 0.0

        # Substring in Q&A
        qa = signals.get('question_answer', '').lower()
        features['substring_in_qa'] = 1.0 if entity_lower in qa else 0.0

        # Substring in summary
        summary = signals.get('summary', '').lower()
        features['substring_in_summary'] = 1.0 if entity_lower in summary else 0.0

        return features

    def calculate_position_features(self, entity: Entity, article: str) -> dict:
        """
        Calculate position-based features
        Entity position trong article text
        """
        features = {}

        if entity.start == 0:
            # Không có position info (parsed entity)
            features['early_position_bonus'] = 0.0
            features['mid_position_bonus'] = 0.0
        else:
            article_length = len(article)
            position_ratio = entity.start / article_length if article_length > 0 else 0.5

            if position_ratio < 0.2:  # First 20%
                features['early_position_bonus'] = 1.0
                features['mid_position_bonus'] = 0.0
            elif position_ratio < 0.5:  # 20-50%
                features['early_position_bonus'] = 0.0
                features['mid_position_bonus'] = 1.0
            else:
                features['early_position_bonus'] = 0.0
                features['mid_position_bonus'] = 0.0

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

        # Length penalty (longer entities are less likely to be single objects)
        word_count = len(entity.text.split())
        if word_count > 3:
            features['length_penalty'] = word_count
        else:
            features['length_penalty'] = 0

        return features

    def compute_score(self, features: dict) -> float:
        """
        Compute weighted score từ features
        """
        score = 0.0

        for feature_name, feature_value in features.items():
            weight = self.weights.get(feature_name, 0.0)

            if feature_name == 'length_penalty':
                # Penalty is negative
                score += weight * feature_value
            else:
                score += weight * feature_value

        return max(0.0, score)  # Non-negative scores

    def score_entities(
        self,
        entities: List[Entity],
        signals: dict,
        article: str
    ) -> List[Entity]:
        """
        Score all entities and update their scores

        Args:
            entities: List of Entity objects
            signals: Dict with visual/context signals
            article: Full article text

        Returns:
            List of Entity objects with scores populated
        """
        for entity in entities:
            # Combine all features
            all_features = {}

            all_features.update(self.calculate_presence_features(entity, signals))
            all_features.update(self.calculate_position_features(entity, article))
            all_features.update(self.calculate_type_features(entity))

            # Compute final score
            entity.features = all_features
            entity.score = self.compute_score(all_features)

        return entities

    def filter_and_rank(self, entities: List[Entity]) -> List[Entity]:
        """
        Filter entities by threshold and rank by score
        """
        # Filter by min score
        filtered = [e for e in entities if e.score >= self.thresholds['min_score']]

        # Sort by score descending
        ranked = sorted(filtered, key=lambda e: e.score, reverse=True)

        # Take top-K
        top_k = ranked[:self.thresholds['top_k']]

        # Mark as selected
        for ent in top_k:
            ent.selected = True

        return ranked  # Return all ranked, but mark selected

    def score_query(self, query_data: dict) -> Tuple[List[str], dict]:
        """
        Main scoring function for a single query

        Args:
            query_data: Dict with query information

        Returns:
            Tuple of (selected_entities: List[str], debug_info: dict)
        """
        # Extract article and signals
        article = query_data.get('article', '')

        signals = {
            'crawl_caption': query_data.get('crawl_caption', ''),
            'generated_caption': query_data.get('generated_caption', ''),
            'question_answer': query_data.get('question_answer', ''),
            'summary': query_data.get('article_summary', {}).get('fact_summary', '')
        }

        # Step 1: Extract entities
        spacy_entities = self.extract_entities_from_article(article)
        parsed_entities = self.parse_entity_string(query_data.get('name_entity_keyword', ''))

        # Step 2: Deduplicate
        all_entities = self.deduplicate_entities(spacy_entities, parsed_entities)

        # Step 3: Score entities
        scored_entities = self.score_entities(all_entities, signals, article)

        # Step 4: Filter and rank
        ranked_entities = self.filter_and_rank(scored_entities)

        # Step 5: Extract selected entity texts
        selected_texts = [e.text for e in ranked_entities if e.selected]

        # Step 6: Create debug info
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
            'threshold_used': self.thresholds['min_score'],
            'total_entities_extracted': len(all_entities),
            'entities_selected': len(selected_texts),
            'signals_used': list(signals.keys())
        }

        return selected_texts, debug_info


def save_scoring_results(results: dict, output_path: str):
    """
    Save entity scoring results to JSON for debugging

    Args:
        results: Dict mapping query_id -> debug_info
        output_path: Path to save JSON
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Entity scoring results saved to: {output_path}")


def load_scoring_results(input_path: str) -> dict:
    """Load previously saved scoring results"""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)
