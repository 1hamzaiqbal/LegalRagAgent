"""Build NLP entity graph from barexam corpus (zero LLM calls).

Extracts noun phrases via spaCy, builds co-occurrence graph,
creates inverted index for entity-based passage lookup.

Usage:
  python utils/build_entity_graph.py              # Full build
  python utils/build_entity_graph.py --max 10000  # Test on subset
  python utils/build_entity_graph.py --status      # Check existing graph
"""

import argparse
import json
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict

import pandas as pd

CORPUS_CSV = "datasets/barexam_qa/barexam_qa_train.csv"
OUTPUT_DIR = "datasets/barexam_qa/entity_graph"

# Legal stopwords to filter out common but unhelpful noun phrases
LEGAL_STOPWORDS = {
    'court', 'case', 'plaintiff', 'defendant', 'party', 'parties',
    'state', 'united states', 'person', 'people', 'time', 'fact',
    'question', 'issue', 'matter', 'order', 'part', 'section',
    'action', 'claim', 'right', 'law', 'rule', 'act',
    'evidence', 'trial', 'jury', 'judge', 'witness',
}


def extract_noun_phrases_regex(text: str) -> list[str]:
    """Fast noun phrase extraction using regex patterns for legal text.

    Extracts:
    - Multi-word capitalized phrases (case names, doctrines)
    - Legal Latin phrases
    - Statute references
    - Quoted terms
    """
    phrases = set()

    # Multi-word capitalized sequences (case names, proper nouns)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text):
        phrase = m.group(1).lower()
        if len(phrase) > 5 and phrase not in LEGAL_STOPWORDS:
            phrases.add(phrase)

    # Legal Latin and common doctrine names
    latin = re.findall(r'\b(res ipsa loquitur|habeas corpus|mens rea|actus reus|'
                       r'prima facie|bona fide|de facto|de jure|stare decisis|'
                       r'nolo contendere|voir dire|amicus curiae|'
                       r'due process|equal protection|commerce clause|'
                       r'free exercise|establishment clause|'
                       r'strict liability|negligence per se|proximate cause|'
                       r'consideration|promissory estoppel|specific performance|'
                       r'hearsay|confrontation clause|best evidence|'
                       r'adverse possession|easement|covenant|eminent domain|'
                       r'felony murder|manslaughter|larceny|robbery|burglary|arson)\b',
                       text, re.IGNORECASE)
    for term in latin:
        phrases.add(term.lower())

    # Statute references (e.g., "Section 1983", "Rule 403", "Article III")
    for m in re.finditer(r'\b((?:Section|Rule|Article|Amendment|Clause)\s+\w+)\b', text):
        phrases.add(m.group(1).lower())

    # Quoted legal terms
    for m in re.finditer(r'"([^"]{3,50})"', text):
        phrase = m.group(1).lower().strip()
        if len(phrase.split()) <= 5:
            phrases.add(phrase)

    return list(phrases)


def extract_noun_phrases_spacy(text: str, nlp) -> list[str]:
    """Extract noun phrases using spaCy NLP pipeline."""
    doc = nlp(text[:5000])  # limit length for speed
    phrases = set()

    # Noun chunks
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()
        if len(phrase) > 3 and phrase not in LEGAL_STOPWORDS and len(phrase.split()) <= 5:
            phrases.add(phrase)

    # Named entities
    for ent in doc.ents:
        if ent.label_ in ('LAW', 'ORG', 'PERSON', 'GPE', 'EVENT', 'WORK_OF_ART'):
            phrase = ent.text.lower().strip()
            if len(phrase) > 3:
                phrases.add(phrase)

    return list(phrases)


def build_graph(passages_df: pd.DataFrame, use_spacy: bool = False):
    """Build entity co-occurrence graph and inverted index."""
    nlp = None
    if use_spacy:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            nlp.max_length = 10000
            print("  Using spaCy for NP extraction")
        except (ImportError, OSError):
            print("  spaCy not available, falling back to regex extraction")
            use_spacy = False

    # Extract entities from each passage
    inverted_index = defaultdict(set)  # entity → set of passage idx
    passage_entities = {}  # passage idx → set of entities
    entity_counts = Counter()

    t0 = time.time()
    for i, (_, row) in enumerate(passages_df.iterrows()):
        idx = str(row['idx'])
        text = str(row['text'])

        if use_spacy and nlp:
            entities = extract_noun_phrases_spacy(text, nlp)
        else:
            entities = extract_noun_phrases_regex(text)

        passage_entities[idx] = set(entities)
        for e in entities:
            inverted_index[e].add(idx)
            entity_counts[e] += 1

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(passages_df) - i - 1) / rate
            print(f"  {i+1:>8,}/{len(passages_df):,} ({(i+1)/len(passages_df)*100:.1f}%) | "
                  f"{rate:.0f} passages/sec | ETA {eta/60:.0f}min | "
                  f"entities: {len(inverted_index):,}")

    total_time = time.time() - t0
    print(f"\n  Extraction done: {total_time/60:.1f}min")
    print(f"  Unique entities: {len(inverted_index):,}")
    print(f"  Avg entities/passage: {sum(len(v) for v in passage_entities.values())/max(len(passage_entities),1):.1f}")

    # Filter: keep entities that appear in 3+ passages but <10% of corpus
    max_freq = len(passages_df) * 0.1
    filtered_index = {e: pids for e, pids in inverted_index.items()
                      if 3 <= len(pids) <= max_freq}
    print(f"  After filtering (3 ≤ freq ≤ {max_freq:.0f}): {len(filtered_index):,} entities")

    # Build co-occurrence edges (entities that appear in same passage)
    print("  Building co-occurrence graph...")
    edges = Counter()
    for idx, ents in passage_entities.items():
        ents_list = [e for e in ents if e in filtered_index]
        for i, e1 in enumerate(ents_list):
            for e2 in ents_list[i+1:]:
                pair = tuple(sorted([e1, e2]))
                edges[pair] += 1

    # Keep edges with weight >= 2
    strong_edges = {pair: w for pair, w in edges.items() if w >= 2}
    print(f"  Co-occurrence edges (weight ≥ 2): {len(strong_edges):,}")

    # Top entities
    print(f"\n  Top 20 entities:")
    for entity, count in entity_counts.most_common(20):
        if entity in filtered_index:
            print(f"    {entity:40s} {count:>6,} passages")

    # Community detection (Louvain clustering)
    communities = {}
    try:
        import networkx as nx
        import community as community_louvain

        print("\n  Running Louvain community detection...")
        G = nx.Graph()
        for (e1, e2), w in strong_edges.items():
            G.add_edge(e1, e2, weight=w)
        print(f"    Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        partition = community_louvain.best_partition(G, resolution=1.0)
        n_communities = max(partition.values()) + 1 if partition else 0
        print(f"    Communities found: {n_communities}")

        # Build community → entities mapping
        comm_to_entities = defaultdict(list)
        for entity, comm_id in partition.items():
            comm_to_entities[comm_id].append(entity)

        # Label each community by its top entities
        for comm_id, entities in sorted(comm_to_entities.items()):
            top = sorted(entities, key=lambda e: entity_counts.get(e, 0), reverse=True)[:5]
            communities[str(comm_id)] = {
                'entities': entities,
                'top_terms': top,
                'size': len(entities),
            }
            if comm_id < 10:
                print(f"    Community {comm_id} ({len(entities)} entities): {', '.join(top[:5])}")

        # Map passages to communities via their entities
        passage_communities = {}
        for idx, ents in passage_entities.items():
            comms = set()
            for e in ents:
                if e in partition:
                    comms.add(partition[e])
            if comms:
                passage_communities[idx] = list(comms)

        print(f"    Passages with community labels: {len(passage_communities):,}/{len(passage_entities):,}")

    except ImportError:
        print("  SKIP: community detection (install python-louvain + networkx)")
        passage_communities = {}

    return {
        'inverted_index': {e: list(pids) for e, pids in filtered_index.items()},
        'entity_counts': dict(entity_counts.most_common(10000)),
        'edges': {f"{p[0]}|||{p[1]}": w for p, w in strong_edges.most_common(50000)},
        'communities': communities,
        'passage_communities': passage_communities,
        'n_passages': len(passages_df),
        'n_entities': len(filtered_index),
        'n_edges': len(strong_edges),
        'n_communities': len(communities),
    }


def main():
    parser = argparse.ArgumentParser(description="Build NLP entity graph from barexam corpus")
    parser.add_argument("--max", type=int, default=0, help="Max passages (0=all)")
    parser.add_argument("--spacy", action="store_true", help="Use spaCy (slower, better)")
    parser.add_argument("--status", action="store_true", help="Check existing graph")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.status:
        graph_path = os.path.join(OUTPUT_DIR, "entity_graph.json")
        if os.path.exists(graph_path):
            with open(graph_path) as f:
                g = json.load(f)
            print(f"Entity graph: {g['n_entities']:,} entities, {g['n_edges']:,} edges, {g['n_passages']:,} passages")
        else:
            print("No graph built yet")
        return

    print(f"Reading corpus from {CORPUS_CSV}...")
    df = pd.read_csv(CORPUS_CSV)
    if args.max > 0:
        df = df.head(args.max)
    print(f"  {len(df):,} passages")

    print("Extracting entities and building graph...")
    graph = build_graph(df, use_spacy=args.spacy)

    # Save
    graph_path = os.path.join(OUTPUT_DIR, "entity_graph.json")
    with open(graph_path, 'w') as f:
        json.dump(graph, f)
    print(f"\nSaved to {graph_path}")

    # Also save inverted index separately for fast loading
    idx_path = os.path.join(OUTPUT_DIR, "inverted_index.json")
    with open(idx_path, 'w') as f:
        json.dump(graph['inverted_index'], f)
    print(f"Inverted index: {idx_path}")


if __name__ == "__main__":
    main()
