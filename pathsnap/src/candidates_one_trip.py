"""Step 1 worked example: candidate road generation for ONE trip.

Usage: python candidates_one_trip.py "<path to trip folder>" [radius_m] [out_dir]
e.g.   python candidates_one_trip.py "Student folder/train/00000048" 50 out

What it does
1. Loads .track / .nodes / .arcs / .route (all headerless, whitespace-separated).
2. Projects lon/lat to a local metric CRS (UTM zone picked from the trip centre),
   so distances are in metres instead of degrees.
3. Builds one LineString per arc and a spatial index (STRtree).
4. For every GPS point, finds arcs within `radius_m` metres = its candidate set.
5. Reports candidate miss rate (reference arc not in candidate set).
6. Writes a Folium map of the raw GPS, reference route and candidate counts.
"""
import sys, os
import numpy as np
import folium
from pyproj import Transformer
from shapely import STRtree, LineString, Point


def load_trip(folder):
    tid = os.path.basename(os.path.normpath(folder))
    p = os.path.join(folder, tid)
    track = np.loadtxt(p + ".track")               # lon, lat, t
    nodes = np.loadtxt(p + ".nodes")               # lon, lat (row index = node id)
    arcs = np.loadtxt(p + ".arcs", dtype=int)      # from_node, to_node (row index = arc id)
    route = np.loadtxt(p + ".route", dtype=int, ndmin=1)  # reference arc ids
    return track, nodes, arcs, route


def make_projector(lon, lat):
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return Transformer.from_crs(4326, epsg, always_xy=True)


def main(folder, radius=50.0, out="out"):
    track, nodes, arcs, route = load_trip(folder)
    os.makedirs(out, exist_ok=True)

    tr = make_projector(track[:, 0].mean(), track[:, 1].mean())
    nx_, ny_ = tr.transform(nodes[:, 0], nodes[:, 1])
    nxy = np.column_stack([nx_, ny_])
    gx, gy = tr.transform(track[:, 0], track[:, 1])
    gxy = np.column_stack([gx, gy])

    lines = [LineString([nxy[a], nxy[b]]) for a, b in arcs]
    tree = STRtree(lines)

    pts = [Point(xy) for xy in gxy]
    # predicate "dwithin" returns all arcs within radius of each point
    pt_idx, arc_idx = tree.query(pts, predicate="dwithin", distance=radius)
    counts = np.bincount(pt_idx, minlength=len(pts))

    cand = [set() for _ in pts]
    for p, a in zip(pt_idx, arc_idx):
        cand[p].add(int(a))

    # Reference arc per GPS point is NOT given directly (route is only an ordered arc list),
    # so as a simple proxy: a point is a "miss" if no arc of the reference route is in
    # its candidate set. This is a route-level proxy, not an exact per-point label.
    route_set = set(route.tolist())
    miss = np.array([len(c & route_set) == 0 for c in cand])

    print(f"trip: {os.path.basename(os.path.normpath(folder))}")
    print(f"GPS points: {len(track)}  nodes: {len(nodes)}  arcs: {len(arcs)}  route arcs: {len(route)}")
    print(f"radius: {radius} m")
    print(f"candidates per point: min {counts.min()}  median {np.median(counts):.0f}  "
          f"mean {counts.mean():.1f}  max {counts.max()}")
    print(f"points with 0 candidates: {(counts == 0).sum()}")
    print(f"proxy candidate miss rate (no reference-route arc nearby): {miss.mean():.3%}")

    # Folium map (lat, lon order!)
    m = folium.Map(location=[track[:, 1].mean(), track[:, 0].mean()], zoom_start=14)
    for aid in route:
        a, b = arcs[aid]
        folium.PolyLine([(nodes[a][1], nodes[a][0]), (nodes[b][1], nodes[b][0])],
                        color="green", weight=4, opacity=0.8).add_to(m)
    for i, (lon, lat, _) in enumerate(track):
        col = "red" if miss[i] else "blue"
        folium.CircleMarker((lat, lon), radius=2, color=col, fill=True,
                            popup=f"pt {i}: {counts[i]} candidates").add_to(m)
    path = os.path.join(out, "trip_candidates.html")
    m.save(path)
    print("map saved:", path)


if __name__ == "__main__":
    folder = sys.argv[1]
    radius = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
    out = sys.argv[3] if len(sys.argv) > 3 else "out"
    main(folder, radius, out)
