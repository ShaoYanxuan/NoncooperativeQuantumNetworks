from itertools import combinations
from math import comb
import random

# all_pairs = list(combinations(range(num_nodes), 2))


def generate_random_unique_pair_sets(n, num_pairs=2, max_pairsets=200, disjoint=False):
    """
    Generate a list of sets of `num_pairs` node pairs from n nodes.

    - If disjoint=True, each set contains disjoint pairs (no shared nodes).
    - If disjoint=False, pairs can overlap.
    """
    nodes = list(range(n))
    all_pairs = list(combinations(nodes, 2))

    if disjoint:
        if 2 * num_pairs > n:
            return []

        # Estimate total number of disjoint sets (upper bound, not exact)
        total_disjoint_sets = comb(n, 2 * num_pairs)

        if total_disjoint_sets < 2 * max_pairsets:
            # Exhaustive mode
            pair_sets = set()
            for node_subset in combinations(nodes, 2 * num_pairs):
                for pair_comb in combinations(combinations(node_subset, 2), num_pairs):
                    flat = [x for pair in pair_comb for x in pair]
                    if len(set(flat)) == 2 * num_pairs:
                        pair_set = tuple(
                            sorted(tuple(sorted(pair)) for pair in pair_comb)
                        )
                        pair_sets.add(pair_set)
                        if len(pair_sets) >= max_pairsets:
                            return list(pair_sets)
            return list(pair_sets)

        else:
            # Lazy random sampling mode
            pair_sets = set()
            attempts = 0
            max_attempts = 10 * max_pairsets
            while len(pair_sets) < max_pairsets and attempts < max_attempts:
                sampled_nodes = random.sample(nodes, 2 * num_pairs)
                pairs = list(combinations(sampled_nodes, 2))
                random.shuffle(pairs)
                chosen = []
                used = set()
                for a, b in pairs:
                    if a not in used and b not in used:
                        chosen.append(tuple(sorted((a, b))))
                        used.update([a, b])
                        if len(chosen) == num_pairs:
                            break
                if len(chosen) == num_pairs:
                    pair_sets.add(tuple(sorted(chosen)))
                attempts += 1
            return list(pair_sets)

    else:
        total_combinations = comb(len(all_pairs), num_pairs)

        if total_combinations < 2 * max_pairsets:
            pair_sets = list(combinations(all_pairs, num_pairs))
            if len(pair_sets) > max_pairsets:
                pair_sets = random.sample(pair_sets, max_pairsets)
            return pair_sets

        pair_sets = set()
        while len(pair_sets) < max_pairsets:
            random_set = tuple(sorted(random.sample(all_pairs, num_pairs)))
            pair_sets.add(random_set)
        return list(pair_sets)


def generate_random_unique_edge_sets(edge_list, num_edges_removed):
    edge_sets = list(combinations(edge_list, num_edges_removed))
    return edge_sets
