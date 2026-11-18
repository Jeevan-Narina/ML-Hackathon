"""Bounded directed routes between positions on original arcs; no references."""
from collections import OrderedDict
from dataclasses import dataclass
import heapq
import math
import networkx as nx


@dataclass(frozen=True)
class Route:
    distance_m: float
    arcs: tuple
    status: str = 'ok'

    @property
    def feasible(self):
        return self.status == 'ok'


class DirectedRouter:
    """One bounded Dijkstra tree per cached source, retaining parallel arc IDs.

    Search-budget failure is distinct from proof of topological unreachability.
    Equal distances retain first discovery: sorted adjacency and node heap order.
    """
    def __init__(self, arcs, cutoff_m=350., max_settled=4000, cache_size=1024):
        if cutoff_m <= 0 or max_settled < 1 or cache_size < 1:
            raise ValueError('Invalid routing budget')
        if not arcs.index.is_unique:
            raise ValueError('Duplicate original arc IDs')
        self.cutoff_m, self.max_settled, self.cache_size = cutoff_m, max_settled, cache_size
        self.arcs = {}; self.adj = {}; graph = nx.DiGraph()
        for aid, row in arcs.sort_index().iterrows():
            if row.get('geometry_status', 'valid') not in ['valid', 'degenerate_segment']:
                continue
            length = float(row.arc_length_m)
            if not math.isfinite(length) or length < 0:
                continue
            u, v, aid = int(row.from_node_id), int(row.to_node_id), int(aid)
            self.arcs[aid] = (u, v, length)
            self.adj.setdefault(u, []).append((aid, v, length))
            graph.add_edge(u, v)
        self.dag = nx.condensation(graph)
        self.component = self.dag.graph['mapping']
        self.cache = OrderedDict(); self.reach_cache = OrderedDict()
        self.stats = dict(searches=0, cache_hits=0, settled_nodes=0, budget_exhaustions=0)

    def reachable(self, source, target):
        if source not in self.component or target not in self.component:
            return False
        a, b = self.component[source], self.component[target]
        if a == b:
            return True
        if a not in self.reach_cache:
            self.reach_cache[a] = nx.descendants(self.dag, a)
            if len(self.reach_cache) > self.cache_size:
                self.reach_cache.popitem(last=False)
        else:
            self.reach_cache.move_to_end(a)
        return b in self.reach_cache[a]

    def _tree(self, source):
        if source in self.cache:
            self.stats['cache_hits'] += 1; self.cache.move_to_end(source)
            return self.cache[source]
        self.stats['searches'] += 1
        dist = {source: 0.}; parent = {}; done = set(); heap = [(0., source)]
        while heap and len(done) < self.max_settled:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for aid, v, length in self.adj.get(u, []):
                nd = d + length
                if nd <= self.cutoff_m and nd < dist.get(v, math.inf):
                    dist[v] = nd; parent[v] = (u, aid)
                    heapq.heappush(heap, (nd, v))
        exhausted = bool(heap)
        self.stats['settled_nodes'] += len(done)
        self.stats['budget_exhaustions'] += int(exhausted)
        result = ({u: dist[u] for u in done}, parent, exhausted)
        self.cache[source] = result
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result

    def between_nodes(self, source, target):
        if source == target:
            return Route(0., ())
        if not self.reachable(source, target):
            return Route(math.inf, (), 'directed_unreachable')
        distances, parent, exhausted = self._tree(source)
        if target not in distances:
            return Route(math.inf, (), 'search_budget' if exhausted else 'distance_bound')
        path = []; node = target
        while node != source:
            node, aid = parent[node]; path.append(aid)
        return Route(distances[target], tuple(reversed(path)))

    def route(self, a, fraction_a, b, fraction_b):
        if a not in self.arcs or b not in self.arcs:
            return Route(math.inf, (), 'unknown_arc')
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in [fraction_a, fraction_b]):
            return Route(math.inf, (), 'invalid_position')
        _, end, la = self.arcs[a]; start, _, lb = self.arcs[b]
        if a == b and fraction_b >= fraction_a:
            return Route((fraction_b - fraction_a) * la, (int(a),))
        middle = self.between_nodes(end, start)
        if not middle.feasible:
            return middle
        distance = (1 - fraction_a) * la + middle.distance_m + fraction_b * lb
        if distance > self.cutoff_m:
            return Route(distance, (), 'distance_bound')
        return Route(distance, (int(a), *middle.arcs, int(b)))
