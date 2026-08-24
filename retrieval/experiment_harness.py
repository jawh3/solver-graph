import json
import logging
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np

from knowledge_graph.models import Node, NodeType
from knowledge_graph.store import KnowledgeGraphStore
from retrieval.embedder import MockEmbedder
from retrieval.retriever import ContextRetriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def run_sampling_sweep(n_nodes: int = 100, m_sample: int = 5, temperatures: List[float] = [0.1, 0.5, 1.0, 2.0], iterations: int = 1000):
    """
    Sweep across different exploration rates (temperatures) to measure sampling diversity.
    """
    log.info(f"Generating mock graph of size N={n_nodes}...")
    store = KnowledgeGraphStore()
    embedder = MockEmbedder(dim=128, seed=42)
    
    # We create nodes with simple string texts so the embedder hashes them
    for i in range(n_nodes):
        text = f"node text {i}"
        node = Node(id=f"node_{i}", type=NodeType.TECHNIQUE, attributes={"text": text})
        store.add_node(node, embedder.embed(text))
        
    retriever = ContextRetriever(store, embedder)
    
    # Target query
    query = "node text 0" # This will strongly match node_0, and others will be random noise cosine sim
    
    log.info(f"Retrieving Top-N candidates (N={n_nodes})...")
    candidates = retriever.get_top_n_candidates(query, n=n_nodes)
    
    # Verify candidates
    top_candidate = candidates[0]
    log.info(f"Top candidate: {top_candidate[0].id} with sim={top_candidate[1]:.4f}")
    
    results = {}
    
    for temp in temperatures:
        log.info(f"--- Sampling M={m_sample} for T={temp} ({iterations} iterations) ---")
        
        sampled_counts = Counter()
        
        for _ in range(iterations):
            sampled = retriever.sample_top_m(candidates, m=m_sample, exploration_rate=temp)
            for node in sampled:
                sampled_counts[node.id] += 1
                
        # Calculate empirical probability distribution
        total_samples = iterations * m_sample
        empirical_probs = {node_id: count / total_samples for node_id, count in sampled_counts.items()}
        
        # Calculate entropy of the empirical distribution (diversity metric)
        entropy = -sum(p * np.log(p) for p in empirical_probs.values() if p > 0)
        
        # How often was the absolute #1 best candidate picked?
        top_1_id = candidates[0][0].id
        top_1_prob = empirical_probs.get(top_1_id, 0.0)
        
        stats = {
            "temperature": temp,
            "entropy": float(entropy),
            "top_1_selection_prob": float(top_1_prob),
            "unique_nodes_sampled": len(sampled_counts),
        }
        
        log.info(f"Stats for T={temp}: {json.dumps(stats, indent=2)}")
        results[f"T_{temp}"] = stats
        
    # Save results
    eval_dir = Path("evaluation/eval_results")
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = eval_dir / "retrieval_sweep.json"
    with open(save_path, "w") as f:
        json.dump({
            "sweep_params": {
                "n_nodes": n_nodes,
                "m_sample": m_sample,
                "iterations": iterations
            },
            "results": results
        }, f, indent=2)
        
    log.info(f"Saved sweep results to {save_path}")

if __name__ == "__main__":
    run_sampling_sweep()
