"""
Сравнение пространственного графа и графа схожести на одном фрагменте города.

Input:  data/interim/gnn_features_step1.csv
Output: docs/graph_comparison.png

Usage:
    python src/visualization/visualize_graphs.py
"""

import csv
import random
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree

# ── конфигурация ───────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
CSV_FILE  = ROOT / 'data/processed/gnn_node_features-b2-final.csv'
OUT_FILE  = ROOT / "docs/graph_comparison.png"

DISTRICT      = "Василеостровский район"   # достаточно компактный
N_SAMPLE      = 120                        # узлов для показа
K             = 4                          # соседей в каждом графе
RANDOM_SEED   = 42

SIM_COLS = ['area_m2', 'perimeter_m',
    'year_built', 'floors', 'last_modified_year',
    'distance_to_water_m', 'distance_to_major_road_m',
    'distance_to_metro_m', 'distance_to_tram_rail_m',
    'amenity_count_300m', 'distance_to_park_m', 'distance_to_pedestrian_m',
    'dist_Василеостровский район', 'dist_Петроградский район', 'dist_Центральный район',
]

# ── загрузка данных ────────────────────────────────────────────────────────
def load_district(path, district, sim_cols):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["district_name"] != district:
                continue
            try:
                lat = float(row["centroid_lat"])
                lon = float(row["centroid_lon"])
                feats = [float(row[c]) for c in sim_cols]
                rows.append({"lat": lat, "lon": lon, "feats": feats})
            except (ValueError, KeyError):
                continue
    return rows


def build_spatial_edges(coords_rad, k):
    tree = BallTree(coords_rad, metric="haversine")
    _, idx = tree.query(coords_rad, k=k + 1)
    src = np.repeat(np.arange(len(coords_rad)), k)
    dst = idx[:, 1:].flatten()
    return list(zip(src.tolist(), dst.tolist()))


def build_similarity_edges(feats, k):
    X = StandardScaler().fit_transform(feats)
    tree = BallTree(X, metric="euclidean")
    _, idx = tree.query(X, k=k + 1)
    src = np.repeat(np.arange(len(feats)), k)
    dst = idx[:, 1:].flatten()
    return list(zip(src.tolist(), dst.tolist()))


# ── построение ────────────────────────────────────────────────────────────
rows = load_district(CSV_FILE, DISTRICT, SIM_COLS)
print(f"Зданий в районе: {len(rows)}")

random.seed(RANDOM_SEED)
rows = random.sample(rows, min(N_SAMPLE, len(rows)))
n = len(rows)

lats = np.array([r["lat"] for r in rows])
lons = np.array([r["lon"] for r in rows])
feats = np.array([r["feats"] for r in rows])

coords_rad = np.deg2rad(np.column_stack([lats, lons]))

spatial_edges    = build_spatial_edges(coords_rad, K)
similarity_edges = build_similarity_edges(feats, K)

# медианная длина рёбер для справки
def haversine_km(i, j):
    R = 6371
    dlat = math.radians(lats[j] - lats[i])
    dlon = math.radians(lons[j] - lons[i])
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lats[i])) * \
        math.cos(math.radians(lats[j])) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

sp_km  = [haversine_km(i, j) for i, j in spatial_edges]
sim_km = [haversine_km(i, j) for i, j in similarity_edges]
print(f"Пространственный граф — медиана длины ребра: {np.median(sp_km):.2f} км")
print(f"Граф схожести        — медиана длины ребра: {np.median(sim_km):.2f} км")

# ── отрисовка ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                         facecolor="#1a1a2e")

EDGE_SPATIAL    = "#4fc3f7"
EDGE_SIMILARITY = "#f48fb1"
NODE_COLOR      = "#ffffff"
BG_COLOR        = "#1a1a2e"

for ax in axes:
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)

def draw_panel(ax, edges, edge_color, title, subtitle):
    for i, j in edges:
        ax.plot([lons[i], lons[j]], [lats[i], lats[j]],
                color=edge_color, linewidth=0.8, alpha=0.55, zorder=1)

    ax.scatter(lons, lats, s=22, c=NODE_COLOR,
               edgecolors="none", zorder=2)

    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=10)
    ax.text(0.5, -0.03, subtitle, transform=ax.transAxes,
            color="#aaaaaa", fontsize=9, ha="center", va="top")

    # масштабная линейка ~200 м
    lon_c   = lons.mean()
    lat_c   = lats.min() - (lats.max() - lats.min()) * 0.06
    deg_200m = 0.200 / (111.32 * math.cos(math.radians(lats.mean())))
    ax.plot([lon_c - deg_200m/2, lon_c + deg_200m/2],
            [lat_c, lat_c], color="#888888", linewidth=2, solid_capstyle="round")
    ax.text(lon_c, lat_c - (lats.max()-lats.min())*0.012,
            "200 м", color="#888888", fontsize=8, ha="center", va="top")


sp_median  = np.median(sp_km) * 1000
sim_median = np.median(sim_km) * 1000

draw_panel(axes[0], spatial_edges, EDGE_SPATIAL,
           "Пространственный граф",
           f"K={K} ближайших по расстоянию · медиана ребра {sp_median:.0f} м")

draw_panel(axes[1], similarity_edges, EDGE_SIMILARITY,
           "Граф схожести",
           f"K={K} ближайших по признакам (год, тип, площадь…) · медиана {sim_median:.0f} м")

fig.suptitle(f"{DISTRICT} · {n} зданий",
             color="white", fontsize=12, y=0.98)

plt.tight_layout(rect=[0, 0.02, 1, 0.97])
OUT_FILE.parent.mkdir(exist_ok=True)
fig.savefig(OUT_FILE, dpi=160, bbox_inches="tight",
            facecolor=BG_COLOR)
print(f"\nСохранено: {OUT_FILE}")