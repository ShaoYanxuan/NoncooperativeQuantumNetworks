# TODO


import json
import os
from pathlib import Path
from qroute.routing.allocator import QuantumRoutingAllocator, _make_edge_key
from graph_tool.all import load_graph

from qroute.routing.entanglement_manipulation import werner_param_from_fidelity
from qroute.routing.flow_allocator import FlowAllocator


def open_graph_as_quantum_allocator(file_path, F0, CAP=1000):
    # Load the graph from the compressed XML file
    graph = load_graph(file_path)

    # Map each edge to its capacity and Werner parameter
    edge_keys = [_make_edge_key(e) for e in graph.edges()]

    ent = graph.edge_properties["entanglement"]
    edge_werner_params = {}
    for edge in graph.edges():
        f = F0 if ent[edge] else 1
        edge_werner_params[_make_edge_key(edge)] = werner_param_from_fidelity(f)

    edge_capacities = dict(zip(edge_keys, [CAP] * len(edge_keys)))

    return QuantumRoutingAllocator(graph, edge_werner_params, edge_capacities, [])

def open_graph_as_quantum_with_fidelities(file_path, fidelities, CAP=1000):
    # Load graph
    graph = load_graph(file_path)

    edge_keys = [_make_edge_key(e) for e in graph.edges()]
    if len(fidelities) != len(edge_keys):
        raise ValueError(f"Number of fidelities ({len(fidelities)}) does not match number of edges ({len(edge_keys)}).")

    # Map fidelity list to edge Werner parameters
    edge_werner_params = {
        edge_key: werner_param_from_fidelity(f)
        for edge_key, f in zip(edge_keys, fidelities)
    }

    edge_capacities = dict(zip(edge_keys, [CAP] * len(edge_keys)))

    return QuantumRoutingAllocator(graph, edge_werner_params, edge_capacities, [])



def open_graph_as_flow_allocator(file_path, F0, CAP=1.0):
    # Load the graph from the compressed XML file
    graph = load_graph(file_path)

    # Map each edge to its capacity and Werner parameter
    edge_keys = [_make_edge_key(e) for e in graph.edges()]

    ent = graph.edge_properties["entanglement"]
    edge_werner_params = {}
    for edge in graph.edges():
        f = F0 if ent[edge] else 1
        edge_werner_params[_make_edge_key(edge)] = werner_param_from_fidelity(f)

    edge_capacities = dict(zip(edge_keys, [CAP] * len(edge_keys)))

    return FlowAllocator(graph, edge_werner_params, edge_capacities, [])


def save_graph_from_quantum_allocator(
    qalloc: QuantumRoutingAllocator, file_path: str | Path
):
    """
    Save the internal graph of a QuantumRoutingAllocator to disk,
    encoding entanglement presence (p0 < 1) as a boolean edge property.
    Does nothing if the file already exists.

    Args:
        qalloc (QuantumRoutingAllocator): Allocator with graph and Werner params.
        file_path (str or Path): Path to save the .gt.xml.gz graph file.
    """
    file_path = Path(file_path)
    if file_path.exists():
        print("File already exists, skipping save.")
        return  # Skip saving if file already exists

    g = qalloc.graph
    ent_prop = g.new_edge_property("bool")

    for e in g.edges():
        k = qalloc.edge_keys[e]
        ent_prop[e] = qalloc.edge_p0[k] < 1.0

    g.edge_properties["entanglement"] = ent_prop
    g.save(str(file_path))


def save_sample(folder, f0: float, sample_index: int, file_suffix: str, **data_dicts):
    """
    Save sample fidelity data with customizable file naming and content.

    Args:
        folder (str or Path): Folder where the JSON file should be saved.
        f0 (float): Initial fidelity parameter.
        sample_index (int): Index of the sample.
        file_suffix (str): Custom string to append in the filename (e.g., 'edges_3', 'pairs_100').
        **data_dicts: Arbitrary number of dictionaries to include in the JSON,
                     e.g., fidelity_before=..., optimal_fidelity_after_removal=...
    """
    file_name = f"sample_{sample_index}_{file_suffix}_f0_{f0:.3f}".replace(".", "p")
    file_path = Path(folder) / f"{file_name}.json"

    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert tuple keys to strings
    json_data = {
        key: {json.dumps(pair_set): val for pair_set, val in val_dict.items()}
        for key, val_dict in data_dicts.items()
    }

    with open(file_path, "w") as f:
        json.dump(json_data, f, indent=2)


def get_completed_samples_per_fidelity(folder, suffix_key: str, suffix_val, f0_grid):
    """
    Returns a dict mapping each f0 in f0_grid to the highest completed sample index.

    Args:
        folder (str or Path): Directory containing the .json files.
        suffix_key (str): Identifier like 'edges' or 'pairs'.
        suffix_val (int): Value to look for in filenames, e.g., 3 for '_edges_3_'.
        f0_grid (Iterable[float]): List of f0 values to check.

    Returns:
        dict[float, int or None]: Highest sample index for each f0, or None if not found.
    """
    results = {round(f0, 3): [] for f0 in f0_grid}
    files = os.listdir(folder)

    suffix_token = f"_{suffix_key}_{suffix_val}_"

    for fname in files:
        if not fname.endswith(".json"):
            continue
        if suffix_token not in fname:
            continue
        try:
            sample_idx = int(fname.split("_")[1])
            f0_str = fname.split("_f0_")[1].replace(".json", "").replace("p", ".")
            f0_val = round(float(f0_str), 3)
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue

        if f0_val in results:
            results[f0_val].append(sample_idx)

    return {f0: (max(samples) if samples else None) for f0, samples in results.items()}
