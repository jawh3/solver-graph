#!/usr/bin/env python3
import argparse
import json
import logging
import time
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# Add project root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_graph.models import Node, NodeType, Edge
from knowledge_graph.store import KnowledgeGraphStore
from knowledge_graph.embedding_mock import MockEmbeddingGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

def generate_mock_graph(store: KnowledgeGraphStore, embedder: MockEmbeddingGenerator, n_nodes: int):
    """Generate a random graph of size n_nodes with random edges."""
    nodes = []
    log.info(f"Generating {n_nodes} nodes...")
    for i in range(n_nodes):
        # 10% categories, 20% tools, 30% techniques, 40% challenges
        import random
        n_type = random.choices(
            [NodeType.CATEGORY, NodeType.TOOL, NodeType.TECHNIQUE, NodeType.CHALLENGE],
            weights=[0.1, 0.2, 0.3, 0.4],
            k=1
        )[0]
        
        node_id = f"{n_type.value}_{i}"
        nodes.append(node_id)
        
        node = Node(id=node_id, type=n_type, attributes={"index": i, "generated": True})
        vec = embedder.generate(node_id)
        
        store.add_node(node, vec)
        
    log.info(f"Generating edges for {n_nodes} nodes...")
    # Add roughly 3 edges per node
    for i in range(n_nodes * 3):
        src = nodes[np.random.randint(0, n_nodes)]
        dst = nodes[np.random.randint(0, n_nodes)]
        if src != dst:
            edge = Edge(source=src, target=dst, weight=np.random.rand())
            try:
                store.add_edge(edge)
            except KeyError:
                pass

def run_experiment(config: dict):
    """Run the serialization scalability sweep."""
    seed = config.get("seed", 42)
    embedding_dim = config.get("embedding_dim", 128)
    scales = config.get("scales", [100, 1000, 10000])
    
    np.random.seed(seed)
    
    results = {
        "metadata": {
            "start_time": datetime.utcnow().isoformat(),
            "seed": seed,
            "embedding_dim": embedding_dim,
            "scales": scales,
            "environment": dict(os.environ.items())
        },
        "runs": []
    }
    
    base_out = Path(config.get("out_dir", "evaluation/eval_results/kg_storage"))
    base_out.mkdir(parents=True, exist_ok=True)
    
    for scale in scales:
        log.info(f"--- Running sweep for N={scale} ---")
        
        store = KnowledgeGraphStore()
        embedder = MockEmbeddingGenerator(dim=embedding_dim, seed=seed)
        
        # 1. Measure Build Time
        t0 = time.perf_counter()
        generate_mock_graph(store, embedder, scale)
        t_build = time.perf_counter() - t0
        
        # 2. Measure Save Time
        save_path = base_out / f"kg_mock_{scale}"
        t0 = time.perf_counter()
        store.save(save_path)
        t_save = time.perf_counter() - t0
        
        # Size on disk
        json_size = (save_path.with_suffix(".json")).stat().st_size
        npy_size = (save_path.with_suffix(".npy")).stat().st_size
        
        # 3. Measure Load Time
        t0 = time.perf_counter()
        loaded_store = KnowledgeGraphStore.load(save_path)
        t_load = time.perf_counter() - t0
        
        # Verify
        ids, matrix = loaded_store.get_all_embeddings()
        
        run_stats = {
            "scale": scale,
            "build_time_sec": round(t_build, 4),
            "save_time_sec": round(t_save, 4),
            "load_time_sec": round(t_load, 4),
            "json_size_bytes": json_size,
            "npy_size_bytes": npy_size,
            "total_size_mb": round((json_size + npy_size) / (1024 * 1024), 2),
            "loaded_nodes": len(ids),
            "loaded_matrix_shape": list(matrix.shape)
        }
        
        results["runs"].append(run_stats)
        log.info(f"Stats for N={scale}: {json.dumps(run_stats, indent=2)}")
        
    results["metadata"]["end_time"] = datetime.utcnow().isoformat()
    
    # Save final structured results
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_file = Path("evaluation/eval_results") / f"kg_storage_sweep_{timestamp}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
        
    log.info(f"Sweep complete. Results saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KG Storage Scalability Sweep")
    parser.add_argument("--config", type=str, help="Path to config JSON")
    args = parser.parse_args()
    
    config = {
        "seed": 42,
        "embedding_dim": 1536, # Typical vLLM dense dim
        "scales": [100, 1000, 10000]
    }
    
    if args.config:
        with open(args.config, "r") as f:
            config.update(json.load(f))
            
    run_experiment(config)
