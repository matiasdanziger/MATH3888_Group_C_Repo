"""
NOTES TO MATIAS: 

1) ANY QUESTIONS ASK JEREMY
2) MAY NEED TO CHANGE FILE NAME FOR THE 4932 FILES FOR THE CONSISTENT VERSION, MINE USES 12.0


NOTES TO MARKER FOR SECTION 6

1) Required packages: networkx, pandas
2) To run: python section5_analysis.py --inputs inputs --output results
3) The endpoints PBI2, PKR1 and IMP2 are taken for granted from previous sections. 
4) The primary graph uses score >=750; >750 only for sensitivity checks.
5) Outputs to look for: shortest paths, shared intermediates, deletion effects,
threshold comparisons, target-background comparisons and matched controls.
6) Convention of distances mean the number of edges. 
"""

from pathlib import Path
from collections import Counter
from itertools import combinations
import argparse
import math
import pandas as pd
import networkx as nx

TARGETS = {"HSC82": "4932.YMR186W", "SSA1": "4932.YAL005C", "HSF1": "4932.YGL073W"}
ENDPOINTS = {"PBI2": "4932.YNL015W", "PKR1": "4932.YMR123W", "IMP2": "4932.YMR035W"}
FOCAL = {"IMP21": "4932.YIL154C", "CYT1": "4932.YOR065W", "TFS1": "4932.YLR178C"}


def save(rows, folder, name):
    pd.DataFrame(rows).to_csv(folder / (name + ".csv"), index=False)


def show(title, table):
    print("\n" + title, flush=True)
    if table.empty:
        print("No rows.", flush=True)
    else:
        print(table.to_string(index=False, na_rep="NA", float_format=lambda x: f"{x:.3g}"), flush=True)


def principal_pairs(table):
    # Focus terminal summaries on the three delivered candidate-endpoint pairs.
    return table.loc[
        ((table["endpoint"] == "PBI2") & (table["candidate"] == "TFS1"))
        | ((table["endpoint"] == "IMP2") & table["candidate"].isin(["CYT1", "IMP21"]))
    ]


def load_inputs(folder):
    info = pd.read_csv(folder / "4932.protein.info.v12.0.txt", sep="\t").fillna("")
    links = pd.read_csv(folder / "4932.protein.links.v12.0.txt.gz", sep=r"\s+")
    names = dict(zip(info.iloc[:, 0], info.preferred_name))
    for group in [TARGETS, ENDPOINTS, FOCAL]:
        names.update({protein: name for name, protein in group.items()})
    dubious = set(
        info.loc[
            info.annotation.str.contains("dubious open reading frame", case=False),
            info.columns[0],
        ]
    )
    table = pd.read_csv(
        folder / "Data - List of essential proteins of saccharomyces cerevisiae.csv",
        header=None,
        dtype=str,
    )
    essential = {
        "4932." + value.removeprefix("4932.")
        for value in table.iloc[:, 1].dropna().str.strip()
    }
    return links, names, dubious, essential


def build_graph(links, dubious, cutoff, strict=False):
    if strict:
        selected = links.loc[links.combined_score > cutoff]
    else:
        selected = links.loc[links.combined_score >= cutoff]
    graph = nx.from_pandas_edgelist(selected, "protein1", "protein2")
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    graph.remove_nodes_from(dubious)
    return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()


def without(graph, removed):
    # A view avoids copying the whole graph. Keep all later components.
    return nx.subgraph_view(graph, filter_node=lambda protein: protein not in removed)


def shortest_data(graph, source):
    """
    Process nodes in increasing distance. A predecessor is a neighbour
    one step closer to the target protein set. Add its path count to count paths for faster computation.
    Union finds nodes on some paths and an intersection finds nodes on all.
    """
    if source not in graph:
        return {}, {}, {}, {}
    distances = dict(nx.single_source_shortest_path_length(graph, source))
    counts = {source: 1}
    some = {source: {source}}
    every = {source: {source}}
    for node in sorted(distances, key=distances.get):
        if node == source:
            continue
        predecessors = [
            p for p in graph[node] if distances.get(p) == distances[node] - 1
        ]
        counts[node] = sum(counts[p] for p in predecessors)
        some[node] = set.union(*(some[p] for p in predecessors)) | {node}
        every[node] = set.intersection(*(every[p] for p in predecessors)) | {node}
    return distances, counts, some, every


def route(graph, source, endpoint):
    if (
        source not in graph
        or endpoint not in graph
        or not nx.has_path(graph, source, endpoint)
    ):
        return [], Counter()
    paths = list(nx.all_shortest_paths(graph, source, endpoint))
    through = Counter(node for path in paths for node in path[1:-1])
    return paths, through


def deletion_effect(graph, source, endpoint, candidate):
    before = nx.shortest_path_length(graph, source, endpoint)
    changed = without(graph, {candidate})
    if not nx.has_path(changed, source, endpoint):
        return before, None, math.inf, 0, []
    distance, counts, _, _ = shortest_data(changed, source)
    example = nx.shortest_path(changed, source, endpoint)
    return (
        before,
        distance[endpoint],
        distance[endpoint] - before,
        counts[endpoint],
        example,
    )


def analyse_routes(graph, targets, names, essential, folder):
    folder.mkdir(parents=True, exist_ok=True)
    routes, paths_table, shared_table, deletions, replacement_paths = [], [], [], [], []
    original_distances = {
        name: dict(nx.single_source_shortest_path_length(graph, source))
        for name, source in targets.items()
    }
    for endpoint_name, endpoint in ENDPOINTS.items():
        if endpoint not in graph:
            continue
        experiments = {}
        for target_name, source in targets.items():
            removed = set(targets.values()) - {source}
            view = without(graph, removed)
            paths, through = route(view, source, endpoint)
            experiments[target_name] = (view, paths, through)
            routes.append(
                dict(
                    endpoint=endpoint_name,
                    target=target_name,
                    distance=len(paths[0]) - 1 if paths else None,
                    path_count=len(paths),
                )
            )
            for number, path in enumerate(paths, 1):
                paths_table.append(
                    dict(
                        endpoint=endpoint_name,
                        target=target_name,
                        path_number=number,
                        gene_path=" -> ".join(names[p] for p in path),
                        systematic_path=" -> ".join(path),
                    )
                )
        shared = set.intersection(*(set(data[2]) for data in experiments.values()))
        for node in sorted(shared):
            shared_table.append(
                dict(endpoint=endpoint_name, protein_id=node, gene=names[node])
            )
        # Checking candidates that are not shared intermediates.
        for candidate in sorted(shared | set(FOCAL.values())):
            if candidate not in graph:
                continue
            for target_name, source in targets.items():
                view, paths, through = experiments[target_name]
                if not paths:
                    continue
                before, after, increase, count, example = deletion_effect(
                    view, source, endpoint, candidate
                )
                row = dict(
                    endpoint=endpoint_name,
                    target=target_name,
                    candidate=names[candidate],
                    candidate_id=candidate,
                    shared_all_targets=candidate in shared,
                    paths_through=through[candidate],
                    total_shortest_paths=len(paths),
                    path_fraction=through[candidate] / len(paths),
                    distance_before=before,
                    distance_after=after,
                    distance_increase=increase,
                    disconnected=after is None,
                    shortest_paths_after=count,
                    essential_listed=candidate in essential,
                    degree=graph.degree(candidate),
                    example_after=" -> ".join(names[p] for p in example),
                )
                for name in targets:
                    row["original_distance_" + name] = original_distances[name].get(
                        candidate
                    )
                deletions.append(row)
                if (
                    endpoint_name == "IMP2"
                    and candidate in {FOCAL["IMP21"], FOCAL["CYT1"]}
                ) or (endpoint_name == "PBI2" and candidate == FOCAL["TFS1"]):
                    changed = without(view, {candidate})
                    if after is not None:
                        for number, path in enumerate(
                            nx.all_shortest_paths(changed, source, endpoint), 1
                        ):
                            replacement_paths.append(
                                dict(
                                    endpoint=endpoint_name,
                                    target=target_name,
                                    candidate=names[candidate],
                                    path_number=number,
                                    gene_path=" -> ".join(names[p] for p in path),
                                )
                            )
    for name, rows in [
        ("routes", routes),
        ("shortest_paths", paths_table),
        ("shared_intermediates", shared_table),
        ("candidate_deletions", deletions),
        ("replacement_paths", replacement_paths),
    ]:
        save(rows, folder, name)


    label = folder.parent.name + " / " + folder.name
    show(label + ": baseline routes", pd.DataFrame(routes))
    shared_frame = pd.DataFrame(shared_table, columns=["endpoint", "protein_id", "gene"])
    shared_summary = shared_frame.groupby("endpoint")["gene"].agg(
        count="size", proteins=lambda values: ", ".join(sorted(values)))
    show(label + ": shared intermediates", shared_summary.reset_index())
    effects = pd.DataFrame(deletions)
    if not effects.empty:
        columns = ["endpoint", "target", "candidate", "paths_through", "total_shortest_paths",
                   "path_fraction", "distance_before", "distance_after", "disconnected", "shortest_paths_after"]
        show(label + ": named deletion effects", principal_pairs(effects)[columns])
        hits = effects.loc[effects["shared_all_targets"] & (effects["distance_increase"] > 0)]
        show(label + ": shared intermediates whose deletion increases distance or disconnects",
             hits[["endpoint", "target", "candidate", "distance_increase", "disconnected", "essential_listed"]])


def endpoint_metrics(endpoint, data, eligible):
    if any(endpoint not in result[0] for result in data.values()):
        return None
    excluded = set(TARGETS.values()) | {endpoint}
    some = [result[2][endpoint] - excluded for result in data.values()]
    every = [result[3][endpoint] - excluded for result in data.values()]
    shared = set.intersection(*some)
    required_once = shared & set.union(*every)
    required_twice = {
        node for node in shared if sum(node in group for group in every) >= 2
    }
    return dict(
        shared_count=len(shared),
        qualifying_any_count=len(required_once),
        qualifying_two_count=len(required_twice),
        eligible_any_count=len(required_once & eligible),
        eligible_two_count=len(required_twice & eligible),
    )


def matched_controls(graph, names, essential, folder):
    folder.mkdir(parents=True, exist_ok=True)
    # Step finds comparable nodes with similar degree and distance to the three targets with the other two targets deleted.
    data = {
        name: shortest_data(without(graph, set(TARGETS.values()) - {source}), source)
        for name, source in TARGETS.items()
    }
    original = {
        name: dict(nx.single_source_shortest_path_length(graph, source))
        for name, source in TARGETS.items()
    }
    eligible = {
        node
        for node in graph
        if node not in essential
        and all(3 <= distances.get(node, -1) <= 5 for distances in original.values())
    }
    excluded = set(TARGETS.values()) | set(ENDPOINTS.values()) | set(FOCAL.values())
    census = []
    for node in sorted(graph):
        metrics = endpoint_metrics(node, data, eligible)
        if metrics is not None:
            row = dict(
                protein_id=node,
                gene=names[node],
                degree=graph.degree(node),
                excluded_from_controls=node in excluded,
                **metrics,
            )
            for name in TARGETS:
                row["distance_" + name] = data[name][0][node]
            census.append(row)
    members, comparisons, observed, focal_results = [], [], [], []
    for endpoint_name, endpoint in ENDPOINTS.items():
        obs = next(row for row in census if row["protein_id"] == endpoint)
        observed.append(dict(endpoint=endpoint_name, **obs))
        controls = [
            row
            for row in census
            if not row["excluded_from_controls"]
            and row["degree"] == obs["degree"]
            and all(
                row["distance_" + name] == obs["distance_" + name] for name in TARGETS
            )
        ]
        members.extend(dict(matched_to=endpoint_name, **row) for row in controls)
        for metric in [
            "shared_count",
            "qualifying_any_count",
            "qualifying_two_count",
            "eligible_any_count",
            "eligible_two_count",
        ]:
            values = [row[metric] for row in controls]
            comparisons.append(
                dict(
                    endpoint=endpoint_name,
                    metric=metric,
                    observed=obs[metric],
                    n_controls=len(values),
                    limited_controls=len(values) < 20,
                    control_min=min(values) if values else None,
                    control_median=pd.Series(values).median() if values else None,
                    control_max=max(values) if values else None,
                    n_equal=sum(value == obs[metric] for value in values),
                    n_strictly_greater=sum(value > obs[metric] for value in values),
                    upper_tail_fraction=sum(value >= obs[metric] for value in values)
                    / len(values)
                    if values
                    else None,
                )
            )
        for target_name, source in TARGETS.items():
            base = without(graph, set(TARGETS.values()) - {source})
            before = data[target_name]
            for candidate_name, candidate in FOCAL.items():
                after = shortest_data(without(base, {candidate}), source)
                suffix = shortest_data(base, candidate)
                for control in controls:
                    node = control["protein_id"]
                    through = 0
                    if (
                        candidate in before[0]
                        and node in suffix[0]
                        and before[0][candidate] + suffix[0][node] == before[0][node]
                    ):
                        through = before[1][candidate] * suffix[1][node]
                    distance_after = after[0].get(node)
                    focal_results.append(
                        dict(
                            matched_to=endpoint_name,
                            control_id=node,
                            control_gene=names[node],
                            target=target_name,
                            candidate=candidate_name,
                            paths_before=before[1][node],
                            paths_through=through,
                            coverage=through / before[1][node],
                            distance_before=before[0][node],
                            distance_after=distance_after,
                            delta=distance_after - before[0][node]
                            if distance_after is not None
                            else math.inf,
                        )
                    )
    for name, rows in [
        ("endpoint_census", census),
        ("matched_control_members", members),
        ("matched_comparisons", comparisons),
        ("observed_endpoint_metrics", observed),
        ("focal_control_deletions", focal_results),
    ]:
        save(rows, folder, name)


    show("Matched controls: endpoint metrics versus control pools", pd.DataFrame(comparisons)[[
        "endpoint", "metric", "observed", "n_controls", "limited_controls",
        "control_min", "control_median", "control_max", "upper_tail_fraction"]])
    focal = pd.DataFrame(focal_results)
    focal["uses_candidate"] = focal["coverage"] > 0
    focal["distance_increases"] = focal["delta"] > 0
    summary = focal.groupby(["matched_to", "candidate", "target"]).agg(
        controls=("control_id", "size"), using_candidate=("uses_candidate", "sum"),
        distance_increases=("distance_increases", "sum")).reset_index()
    summary = summary.rename(columns={"matched_to": "endpoint"})
    show("Matched controls: named candidates on the principal endpoint comparisons",
         principal_pairs(summary))


def target_backgrounds(graph, cutoff, folder):
    # Robustnest checks with no targets deleted, one target deleted and two targets deleted (already done in prev step)
    rows = []
    for target_name, source in TARGETS.items():
        others = [name for name in TARGETS if name != target_name]
        for size in range(3):
            for subset in combinations(others, size):
                base = without(graph, {TARGETS[name] for name in subset})
                before = shortest_data(base, source)
                for candidate_name, candidate in FOCAL.items():
                    after = shortest_data(without(base, {candidate}), source)
                    suffix = shortest_data(base, candidate)
                    for endpoint_name, endpoint in ENDPOINTS.items():
                        first, second = before[0].get(endpoint), after[0].get(endpoint)
                        through = 0
                        if (
                            candidate in before[0]
                            and endpoint in suffix[0]
                            and first is not None
                            and before[0][candidate] + suffix[0][endpoint] == first
                        ):
                            through = before[1][candidate] * suffix[1][endpoint]
                        status = "ok"
                        if endpoint not in graph:
                            status = "endpoint_absent"
                        elif first is None:
                            status = "baseline_unreachable"
                        elif second is None:
                            status = "disconnected_after"
                        rows.append(
                            dict(
                                threshold=cutoff,
                                target=target_name,
                                deleted_targets=";".join(subset),
                                n_deleted_targets=len(subset),
                                endpoint=endpoint_name,
                                candidate=candidate_name,
                                status=status,
                                distance_before=first,
                                distance_after=second,
                                delta=(
                                    second - first if second is not None else math.inf
                                )
                                if first is not None
                                else None,
                                paths_before=before[1].get(endpoint, 0),
                                paths_through=through,
                                coverage=through / before[1][endpoint]
                                if first is not None
                                else None,
                                paths_after=after[1].get(endpoint, 0),
                            )
                        )
    save(rows, folder, "target_backgrounds")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=Path("inputs"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    links, names, dubious, essential = load_inputs(args.inputs)
    metadata, all_backgrounds = [], []
    for label, cutoff, strict in [
        ("ge750", 750, False),
        ("gt750", 750, True),
        ("ge700", 700, False),
        ("ge800", 800, False),
        ("ge850", 850, False),
        ("ge900", 900, False),
    ]:
        graph = build_graph(links, dubious, cutoff, strict)
        folder = args.output / label
        folder.mkdir(exist_ok=True)
        metadata.append(
            dict(run=label, nodes=len(graph), edges=graph.number_of_edges())
        )
        if cutoff == 750:
            analyse_routes(graph, TARGETS, names, essential, folder / "three")
            paired = {name: TARGETS[name] for name in ["HSC82", "SSA1"]}
            analyse_routes(graph, paired, names, essential, folder / "paired")
        if not strict:
            all_backgrounds.extend(target_backgrounds(graph, cutoff, folder))
        if label == "ge750":
            matched_controls(graph, names, essential, folder / "controls")
        print("Finished", label, flush=True)
    save(metadata, args.output, "graphs")
    save(all_backgrounds, args.output, "threshold_backgrounds")


    show("Graph sizes", pd.DataFrame(metadata))
    backgrounds = principal_pairs(pd.DataFrame(all_backgrounds))
    baseline = backgrounds.loc[backgrounds["n_deleted_targets"] == 2].copy()
    baseline["result"] = baseline.apply(
        lambda row: str(row["delta"]) if row["status"] == "ok" else row["status"], axis=1)
    sensitivity = baseline.pivot(index=["endpoint", "candidate", "target"],
                                 columns="threshold", values="result").reset_index()
    show("Threshold comparison: distance increases with both other targets removed", sensitivity)
    reference = backgrounds.loc[backgrounds["threshold"] == 750].copy()
    reference["outcome"] = reference.apply(
        lambda row: f"{row['distance_before']} -> {row['distance_after']} ({row['status']})", axis=1)
    comparison = reference.groupby(["endpoint", "candidate", "target"]).agg(
        backgrounds=("outcome", "size"), distinct_outcomes=("outcome", "nunique"),
        outcomes=("outcome", lambda values: "; ".join(sorted(set(values))))).reset_index()
    show("Target-background check at >=750: all four backgrounds per retained target", comparison)
    print("\nFull tables and path lists saved to", args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
