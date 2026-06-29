"""
Debug and analyze entity heuristic scoring results
"""

import json
from collections import defaultdict


def analyze_scoring(scores_path="assemble_result/entity_heuristic_scores.json"):
    """
    Analyze entity scoring results to understand:
    - Selection rates
    - Most common features
    - Score distributions
    """
    print("=" * 60)
    print("ENTITY SCORING ANALYSIS")
    print("=" * 60)

    with open(scores_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Count selected vs rejected
    total_extracted = sum(item['total_entities_extracted'] for item in data.values())
    total_selected = sum(item['entities_selected'] for item in data.values())

    print(f"\n📊 Overall Statistics:")
    print(f"  Total queries: {len(data)}")
    print(f"  Total entities extracted: {total_extracted}")
    print(f"  Total entities selected: {total_selected}")
    print(f"  Selection rate: {total_selected/total_extracted*100:.1f}%")
    print(f"  Avg entities per query: {total_extracted/len(data):.1f}")
    print(f"  Avg selected per query: {total_selected/len(data):.1f}")

    # Score distribution
    all_scores = []
    selected_scores = []
    rejected_scores = []

    for query_data in data.values():
        for entity in query_data['scored_entities']:
            score = entity['score']
            all_scores.append(score)
            if entity['selected']:
                selected_scores.append(score)
            else:
                rejected_scores.append(score)

    print(f"\n📈 Score Distribution:")
    print(f"  All entities:")
    print(f"    Min: {min(all_scores):.2f}, Max: {max(all_scores):.2f}, Avg: {sum(all_scores)/len(all_scores):.2f}")
    print(f"  Selected entities:")
    print(f"    Min: {min(selected_scores):.2f}, Max: {max(selected_scores):.2f}, Avg: {sum(selected_scores)/len(selected_scores):.2f}")
    if rejected_scores:
        print(f"  Rejected entities:")
        print(f"    Min: {min(rejected_scores):.2f}, Max: {max(rejected_scores):.2f}, Avg: {sum(rejected_scores)/len(rejected_scores):.2f}")

    # Top scoring features
    feature_contributions = defaultdict(list)
    feature_counts = defaultdict(int)

    for query_data in data.values():
        for entity in query_data['scored_entities']:
            if entity['selected']:
                for feature, value in entity['features'].items():
                    if value > 0:
                        feature_contributions[feature].append(value)
                        feature_counts[feature] += 1

    print(f"\n🔥 Most Active Features (in selected entities):")
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, count) in enumerate(sorted_features[:15], 1):
        avg_value = sum(feature_contributions[feature]) / len(feature_contributions[feature])
        usage_rate = count / total_selected * 100
        print(f"  {i:2d}. {feature:30s}: used {count:4d} times ({usage_rate:5.1f}%), avg={avg_value:.2f}")

    # Entity type distribution
    type_counts = defaultdict(int)
    selected_type_counts = defaultdict(int)

    for query_data in data.values():
        for entity in query_data['scored_entities']:
            entity_type = entity['label']
            type_counts[entity_type] += 1
            if entity['selected']:
                selected_type_counts[entity_type] += 1

    print(f"\n🏷️  Entity Type Distribution:")
    print(f"  {'Type':<15} {'Extracted':<12} {'Selected':<12} {'Selection Rate'}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*15}")
    for entity_type in sorted(type_counts.keys(), key=lambda x: type_counts[x], reverse=True):
        extracted = type_counts[entity_type]
        selected = selected_type_counts[entity_type]
        rate = selected / extracted * 100 if extracted > 0 else 0
        print(f"  {entity_type:<15} {extracted:<12} {selected:<12} {rate:>6.1f}%")

    # Examples of high-scoring entities
    print(f"\n⭐ Examples of High-Scoring Selected Entities:")
    examples_shown = 0
    for query_id, query_data in list(data.items())[:5]:  # First 5 queries
        if query_data['scored_entities']:
            top_entity = query_data['scored_entities'][0]
            if top_entity['selected']:
                print(f"\n  Query: {query_id}")
                print(f"    Entity: '{top_entity['entity']}' (score: {top_entity['score']:.2f})")
                print(f"    Type: {top_entity['label']}")
                print(f"    Top features:")
                sorted_features = sorted(top_entity['features'].items(), key=lambda x: x[1], reverse=True)
                for feat, val in sorted_features[:5]:
                    if val > 0:
                        print(f"      - {feat}: {val:.2f}")
                examples_shown += 1
                if examples_shown >= 3:
                    break

    # Examples of rejected entities
    print(f"\n❌ Examples of Rejected Entities (low scores):")
    examples_shown = 0
    for query_id, query_data in data.items():
        rejected = [e for e in query_data['scored_entities'] if not e['selected']]
        if rejected:
            low_score_entity = rejected[-1]  # Last one (lowest score)
            print(f"\n  Query: {query_id}")
            print(f"    Entity: '{low_score_entity['entity']}' (score: {low_score_entity['score']:.2f})")
            print(f"    Type: {low_score_entity['label']}")
            print(f"    Features:")
            for feat, val in low_score_entity['features'].items():
                if val > 0:
                    print(f"      - {feat}: {val:.2f}")
            examples_shown += 1
            if examples_shown >= 3:
                break

    print("\n" + "=" * 60)


def compare_queries(scores_path="assemble_result/entity_heuristic_scores.json", query_ids=None):
    """
    Compare entity scoring for specific queries
    """
    with open(scores_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if query_ids is None:
        query_ids = list(data.keys())[:3]

    print("\n" + "=" * 60)
    print("QUERY COMPARISON")
    print("=" * 60)

    for query_id in query_ids:
        if query_id not in data:
            print(f"\n⚠️  Query {query_id} not found")
            continue

        query_data = data[query_id]
        print(f"\n📋 Query ID: {query_id}")
        print(f"   Entities extracted: {query_data['total_entities_extracted']}")
        print(f"   Entities selected: {query_data['entities_selected']}")
        print(f"   Threshold: {query_data['threshold_used']}")

        print(f"\n   Top selected entities:")
        for i, entity_text in enumerate(query_data['top_entities'], 1):
            # Find the entity in scored_entities
            entity_data = next((e for e in query_data['scored_entities'] if e['entity'] == entity_text), None)
            if entity_data:
                print(f"     {i}. {entity_text:<30s} (score: {entity_data['score']:>6.2f}, type: {entity_data['label']})")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        scores_path = sys.argv[1]
    else:
        scores_path = "assemble_result/entity_heuristic_scores.json"

    try:
        analyze_scoring(scores_path)

        # Compare first 3 queries
        compare_queries(scores_path)

    except FileNotFoundError:
        print(f"❌ Error: File not found: {scores_path}")
        print(f"\nUsage: python debug_entity_scoring.py [path_to_scores.json]")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
