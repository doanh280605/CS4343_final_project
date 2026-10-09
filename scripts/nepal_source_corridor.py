"""Replay a cached bounded OSM extract into a fresh candidate AOI directory."""

import argparse
import heapq
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
import numpy as np
from rasterio.warp import transform

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", default="data/nepal/discovery-20261008")
parser.add_argument("--output", required=True)
args = parser.parse_args()
folder = Path(args.source)
output = Path(args.output)
output.mkdir(parents=True, exist_ok=False)
raw = json.loads((folder / "osm-rivers.json").read_text())
positions = {}
for way in raw["elements"]:
    if way["type"] == "way":
        for node, point in zip(way["nodes"], way["geometry"], strict=True):
            positions[node] = (point["lon"], point["lat"])
ids = list(positions)
x, y = transform(
    "EPSG:4326", "EPSG:32645", [positions[n][0] for n in ids], [positions[n][1] for n in ids]
)
projected = dict(zip(ids, zip(x, y, strict=True), strict=True))
adjacency = {n: [] for n in ids}
for way in raw["elements"]:
    if way["type"] != "way":
        continue
    for a, b in zip(way["nodes"][:-1], way["nodes"][1:], strict=True):
        length = math.dist(projected[a], projected[b])
        adjacency[a].append((b, length, way["id"]))
        adjacency[b].append((a, length, way["id"]))
main_ids = set()
for way in raw["elements"]:
    names = " ".join(way.get("tags", {}).get(k, "") for k in ("name", "name:en", "alt_name:en"))
    if way["type"] == "way" and any(
        k.lower() in names.lower() for k in ("Trishuli", "Bhote", "Lende", "Lhende")
    ):
        main_ids.update(way["nodes"])
anchors = []
for label, node_id in [
    ("Rasuwagadhi", 992955542),
    ("Timure", 2553894641),
    ("Betrawati", 268864226),
]:
    node = ET.fromstring((folder / f"osm-node-{node_id}.xml").read_text()).find("node")
    point = [float(node.attrib["lon"]), float(node.attrib["lat"])]
    px, py = transform("EPSG:4326", "EPSG:32645", [point[0]], [point[1]])
    nearest = min(main_ids, key=lambda n: math.dist(projected[n], (px[0], py[0])))
    distance = math.dist(projected[nearest], (px[0], py[0]))
    if distance > 1000:
        raise ValueError(f"{label} snap exceeds 1km: {distance}")
    anchors.append(
        {
            "name": label,
            "coordinates": point,
            "source": f"https://www.openstreetmap.org/node/{node_id}",
            "osm_node_id": node_id,
            "version": node.attrib["version"],
            "river_node": nearest,
            "snap_distance_m": distance,
        }
    )
start, end = anchors[0]["river_node"], anchors[2]["river_node"]
queue = [(0, start)]
distance = {start: 0}
previous = {}
while queue:
    cost, node = heapq.heappop(queue)
    if cost != distance[node]:
        continue
    if node == end:
        break
    for next_node, length, way_id in adjacency[node]:
        proposal = cost + length
        if proposal < distance.get(next_node, float("inf")):
            distance[next_node] = proposal
            previous[next_node] = (node, way_id)
            heapq.heappush(queue, (proposal, next_node))
if end not in distance:
    raise ValueError("OSM river graph disconnected; do not invent connecting lines")
path = [end]
ways = []
while path[-1] != start:
    p, w = previous[path[-1]]
    path.append(p)
    ways.append(w)
path.reverse()
ways.reverse()
near_timure = min(
    range(len(path)),
    key=lambda i: math.dist(projected[path[i]], projected[anchors[1]["river_node"]]),
)
print(
    "Timure route distance m:",
    math.dist(projected[path[near_timure]], projected[anchors[1]["river_node"]]),
)
if math.dist(projected[path[near_timure]], projected[anchors[1]["river_node"]]) > 1000:
    raise ValueError("Route misses Timure")
selected_ids = list(dict.fromkeys(ways))
metadata = {
    "status": "sourced candidate; human AOI review pending",
    "buffer_m": 2000,
    "crs": "EPSG:32645",
    "anchors": anchors,
    "routing": "shortest continuous path using only shared OSM river nodes; no synthetic gap edges",
    "length_m": distance[end],
    "river_way_ids": selected_ids,
    "attribution": "© OpenStreetMap contributors",
    "license": "ODbL 1.0",
    "source_snapshot": raw.get("osm3s"),
    "ways": [
        {
            "id": w["id"],
            "version": w["version"],
            "timestamp": w["timestamp"],
            "tags": w.get("tags", {}),
        }
        for w in raw["elements"]
        if w["type"] == "way" and w["id"] in selected_ids
    ],
}
for name, subset in [("corridor", path), ("pilot", path[: near_timure + 1])]:
    feature = {
        "type": "Feature",
        "properties": {"status": metadata["status"], "source": "OpenStreetMap", "buffer_m": 2000},
        "geometry": {"type": "LineString", "coordinates": [positions[n] for n in subset]},
    }
    (output / f"{name}.geojson").write_text(json.dumps(feature, indent=2))
(output / "corridor-provenance.json").write_text(json.dumps(metadata, indent=2))
fig, ax = plt.subplots(figsize=(8, 10))
for way in raw["elements"]:
    if way["type"] == "way":
        pts = np.array([projected[n] for n in way["nodes"]])
        ax.plot(pts[:, 0] / 1000, pts[:, 1] / 1000, color="#dddddd", lw=0.6)
pts = np.array([projected[n] for n in path])
ax.plot(
    pts[:, 0] / 1000, pts[:, 1] / 1000, color="#2563eb", lw=2, label="Sourced candidate corridor"
)
pilot = np.array([projected[n] for n in path[: near_timure + 1]])
ax.plot(
    pilot[:, 0] / 1000, pilot[:, 1] / 1000, color="#e76f51", lw=3, label="Initial feasibility pilot"
)
for a in anchors:
    xx, yy = projected[a["river_node"]]
    ax.scatter(xx / 1000, yy / 1000, color="#111827")
    ax.annotate(a["name"], (xx / 1000, yy / 1000), xytext=(6, 5), textcoords="offset points")
ax.set_xlim(pts[:, 0].min() / 1000 - 3, pts[:, 0].max() / 1000 + 3)
ax.set_ylim(pts[:, 1].min() / 1000 - 3, pts[:, 1].max() / 1000 + 3)
ax.set_aspect("equal")
ax.set_xlabel("UTM 45N easting (km)")
ax.set_ylabel("UTM 45N northing (km)")
ax.set_title(
    "Candidate Rasuwagadhi–Timure–Betrawati river path\nHuman review pending; not a flood extent"
)
ax.legend()
fig.tight_layout()
fig.savefig(output / "corridor-preview.png", dpi=140)
plt.close(fig)
print("Candidate length km:", round(distance[end] / 1000, 2))
print("Source ways:", selected_ids)
print("Anchor snap distances m:", [round(a["snap_distance_m"]) for a in anchors])
