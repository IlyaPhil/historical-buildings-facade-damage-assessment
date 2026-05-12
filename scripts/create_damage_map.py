"""
Interactive Folium map of building facade damage predictions.

Key design:
- Single GeoJson layer (not 4 copies) → file ~4x smaller than v1
- JavaScript-based dynamic styling: radio buttons switch score, checkboxes toggle
- Labeled buildings shown in red; unlabeled colored by selected damage score
- Per-score color scales matching project histogram colors

Inputs:
  - data/interim/spb_4districts_buildings.gpkg
  - data/interim/gnn_predictions_similarity.csv

Output:
  - docs/map.html

Usage:
  python src/visualization/create_damage_map.py
  python src/visualization/create_damage_map.py --simplify 0.00005  # ~5m, default
  python src/visualization/create_damage_map.py --simplify 0.0003   # rougher/faster
"""

import argparse
import json
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd

ROOT      = Path(__file__).resolve().parents[2]
GPKG_FILE = ROOT / "data/interim/spb_4districts_buildings.gpkg"
PRED_FILE = ROOT / "data/processed/gnn_predictions_all.csv"
OUT_DIR   = ROOT / "docs"
OUT_HTML  = OUT_DIR / "map.html"

PRED_COLS = [
    "coating_deterioration_score",
    "masonry_degradation_score",
    "moisture_bio_damage_score",
    "vandalism_score",
]
META_COLS = ["building_id", "district_name", "r_year_int", "r_floors", "is_labeled"]

# Color scales per score — light → dark, matching project histogram colors
SCORE_CFG = {
    "coating_deterioration_score": {"label": "Повреждения покрытия",  "low": "#c6dbef", "high": "#08306b"},
    "masonry_degradation_score":   {"label": "Повреждения кладки",     "low": "#c7e9c0", "high": "#00441b"},
    "moisture_bio_damage_score":   {"label": "Сырость / биопоражения", "low": "#dadaeb", "high": "#3f007d"},
    "vandalism_score":             {"label": "Вандализм",              "low": "#feedde", "high": "#7f2704"},
}

POPUP_ALIASES = {
    "building_id":                "ID здания",
    "district_name":              "Район",
    "is_labeled":                 "Размечено",
    "coords":                     "Координаты",
    "coating_deterioration_score": "Покрытие (GNN)",
    "masonry_degradation_score":   "Кладка (GNN)",
    "moisture_bio_damage_score":   "Влажность (GNN)",
    "vandalism_score":             "Вандализм (GNN)",
}

# ── JavaScript injected into HTML ──────────────────────────────────────────────
JS_TEMPLATE = """
<script>
(function() {{
  var scoreConfigs = {score_configs};
  var currentScore  = "{first_score}";
  var showLabeled   = true;
  var showUnlabeled = true;

  function hexToRgb(h) {{
    var r = /^#?([a-f\\d]{{2}})([a-f\\d]{{2}})([a-f\\d]{{2}})$/i.exec(h);
    return r ? [parseInt(r[1],16), parseInt(r[2],16), parseInt(r[3],16)] : [160,160,160];
  }}
  function lerpColor(val, vmin, vmax, lo, hi) {{
    var t = Math.max(0, Math.min(1, (val - vmin) / (vmax - vmin)));
    var a = hexToRgb(lo), b = hexToRgb(hi);
    return "rgb("+Math.round(a[0]+t*(b[0]-a[0]))+","+
                  Math.round(a[1]+t*(b[1]-a[1]))+","+
                  Math.round(a[2]+t*(b[2]-a[2]))+")";
  }}

  function getStyle(feature) {{
    var p = feature.properties;
    var labeled = (p.is_labeled == 1 || p.is_labeled === true);
    if (labeled  && !showLabeled)   return {{opacity:0, fillOpacity:0, weight:0}};
    if (!labeled && !showUnlabeled) return {{opacity:0, fillOpacity:0, weight:0}};
    if (labeled) return {{fillColor:"#e74c3c", fillOpacity:0.85, color:"white", weight:2.0}};
    var cfg = scoreConfigs[currentScore];
    var val = parseFloat(p[currentScore]);
    if (isNaN(val)) return {{fillColor:"#d0d0d0", fillOpacity:0.5, color:"#666", weight:0.3}};
    return {{fillColor: lerpColor(val,cfg.vmin,cfg.vmax,cfg.colorLow,cfg.colorHigh),
             fillOpacity:0.8, color:"#555", weight:0.3}};
  }}

  function updateLegend() {{
    var cfg = scoreConfigs[currentScore];
    var el = document.getElementById("leg-grad");
    if (!el) return;
    el.style.background = "linear-gradient(to right,"+cfg.colorLow+","+cfg.colorHigh+")";
    document.getElementById("leg-title").innerText = cfg.label;
    document.getElementById("leg-min").innerText   = cfg.vmin.toFixed(3);
    document.getElementById("leg-max").innerText   = cfg.vmax.toFixed(3);
  }}

  function updateMap() {{
    var lyr = window["{layer_var}"];
    if (lyr) lyr.setStyle(getStyle);
    updateLegend();
  }}

  // wait for layer to be available, then wire controls
  var attempts = 0;
  var timer = setInterval(function() {{
    attempts++;
    if (window["{layer_var}"] || attempts > 80) {{
      clearInterval(timer);
      document.querySelectorAll('input[name="score-rb"]').forEach(function(rb) {{
        rb.addEventListener("change", function() {{ currentScore = this.value; updateMap(); }});
      }});
      document.getElementById("cb-labeled").addEventListener("change", function() {{
        showLabeled = this.checked; updateMap();
      }});
      document.getElementById("cb-unlabeled").addEventListener("change", function() {{
        showUnlabeled = this.checked; updateMap();
      }});
      updateMap();
    }}
  }}, 120);
}})();
</script>
"""

CONTROL_PANEL = """
<div id="map-ctrl" style="
    position:fixed;top:70px;right:10px;z-index:9999;
    background:white;padding:13px 16px;border-radius:8px;
    border:1px solid #ccc;font-size:13px;min-width:235px;
    box-shadow:0 2px 10px rgba(0,0,0,.18);">

  <div style="font-weight:bold;margin-bottom:6px;">Индекс повреждений</div>
  {radio_buttons}

  <hr style="margin:10px 0 8px;">
  <div style="font-weight:bold;margin-bottom:5px;">Показать здания</div>
  <label style="display:block;margin:2px 0;">
    <input type="checkbox" id="cb-labeled" checked>
    <span style="display:inline-block;width:12px;height:12px;
           background:#e74c3c;border:2px solid white;
           vertical-align:middle;margin:0 4px 1px;"></span>
    Размеченные здания
  </label>
  <label style="display:block;margin:2px 0;">
    <input type="checkbox" id="cb-unlabeled" checked>
    <span style="display:inline-block;width:12px;height:12px;
           background:#aaa;border:0.5px solid #555;
           vertical-align:middle;margin:0 4px 1px;"></span>
    Неразмеченные здания
  </label>

  <hr style="margin:10px 0 8px;">
  <div id="leg-title" style="font-weight:bold;margin-bottom:5px;"></div>
  <div id="leg-grad" style="height:14px;border-radius:3px;margin-bottom:4px;"></div>
  <div style="display:flex;justify-content:space-between;font-size:11px;color:#666;">
    <span>меньше (<span id="leg-min"></span>)</span>
    <span>больше (<span id="leg-max"></span>)</span>
  </div>

  <hr style="margin:10px 0 6px;">
  <div style="font-size:11px;color:#888;">Нажмите на здание для деталей</div>
</div>
"""


def main(simplify_tolerance: float) -> None:
    OUT_DIR.mkdir(exist_ok=True)

    # ── load polygons ──────────────────────────────────────────────────────────
    print("Loading building polygons...")
    gdf = gpd.read_file(GPKG_FILE)
    print(f"  {len(gdf):,} buildings  |  CRS: {gdf.crs}")

    if "building_id" not in gdf.columns:
        cands = [c for c in gdf.columns if "id" in c.lower()]
        if not cands:
            raise ValueError(f"building_id not found. Columns: {list(gdf.columns)}")
        gdf = gdf.rename(columns={cands[0]: "building_id"})
        print(f"  Renamed '{cands[0]}' → 'building_id'")

    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    gdf["centroid_lat"] = gdf.geometry.centroid.y.round(6)
    gdf["centroid_lon"] = gdf.geometry.centroid.x.round(6)
    gdf["coords"] = gdf["centroid_lat"].astype(str) + ", " + gdf["centroid_lon"].astype(str)

    gdf["geometry"] = gdf.simplify(simplify_tolerance, preserve_topology=True)
    print(f"  Simplified (tolerance={simplify_tolerance})")

    # ── join predictions ───────────────────────────────────────────────────────
    print("Loading GNN predictions...")
    pred = pd.read_csv(PRED_FILE)
    keep = [c for c in META_COLS + PRED_COLS if c in pred.columns]
    pred = pred[keep]
    for c in PRED_COLS:
        if c in pred.columns:
            pred[c] = pred[c].round(4)

    gdf["building_id"] = gdf["building_id"].astype(str)
    pred["building_id"] = pred["building_id"].astype(str)
    gdf = gdf.merge(pred, on="building_id", how="left")

    n_matched = gdf[PRED_COLS[0]].notna().sum() if PRED_COLS[0] in gdf.columns else 0
    print(f"  Matched: {n_matched:,} / {len(gdf):,}")

    # drop columns not needed in GeoJSON (reduces file size)
    keep_geo = (["geometry"]
                + [c for c in META_COLS + PRED_COLS if c in gdf.columns]
                + ["centroid_lat", "centroid_lon", "coords"])
    gdf = gdf[keep_geo]

    # ── JS score config ────────────────────────────────────────────────────────
    score_configs = {}
    for col in PRED_COLS:
        if col not in gdf.columns:
            continue
        vals = gdf[col].dropna()
        cfg  = SCORE_CFG[col]
        score_configs[col] = {
            "label":    cfg["label"],
            "vmin":     round(float(vals.quantile(0.02)), 4),
            "vmax":     round(float(vals.quantile(0.98)), 4),
            "colorLow": cfg["low"],
            "colorHigh":cfg["high"],
        }

    # ── build map ──────────────────────────────────────────────────────────────
    cx = gdf.geometry.centroid.x.median()
    cy = gdf.geometry.centroid.y.median()
    m  = folium.Map(location=[cy, cx], zoom_start=14, tiles="CartoDB positron")

    popup_fields = [f for f in POPUP_ALIASES if f in gdf.columns]
    popup_labels = [POPUP_ALIASES[f] for f in popup_fields]

    gj = folium.GeoJson(
        gdf.__geo_interface__,
        style_function=lambda f: {"fillColor": "#aaa", "fillOpacity": 0.6,
                                  "color": "#555", "weight": 0.3},
        popup=folium.GeoJsonPopup(
            fields=popup_fields,
            aliases=popup_labels,
            localize=True,
            max_width=310,
        ),
        tooltip=folium.GeoJsonTooltip(
            fields=[f for f in ["building_id", "district_name"] if f in gdf.columns],
            aliases=["ID", "Район"][: sum(1 for f in ["building_id", "district_name"] if f in gdf.columns)],
            sticky=False,
        ),
    ).add_to(m)

    layer_var = gj.get_name()
    print(f"  Layer variable: {layer_var}")

    # radio buttons
    radio_html = ""
    for col in PRED_COLS:
        if col not in score_configs:
            continue
        checked = "checked" if col == PRED_COLS[0] else ""
        label   = score_configs[col]["label"]
        lo, hi  = SCORE_CFG[col]["low"], SCORE_CFG[col]["high"]
        swatch  = (f'<span style="display:inline-block;width:30px;height:10px;'
                   f'background:linear-gradient(to right,{lo},{hi});'
                   f'vertical-align:middle;margin:0 5px 1px;border-radius:2px;"></span>')
        radio_html += (
            f'<label style="display:block;margin:3px 0;">'
            f'<input type="radio" name="score-rb" value="{col}" {checked}>'
            f'{swatch}{label}</label>\n'
        )

    panel = CONTROL_PANEL.replace("{radio_buttons}", radio_html)
    m.get_root().html.add_child(folium.Element(panel))

    js = JS_TEMPLATE.format(
        score_configs=json.dumps(score_configs, ensure_ascii=False),
        first_score=PRED_COLS[0],
        layer_var=layer_var,
    )
    m.get_root().html.add_child(folium.Element(js))

    # ── save ───────────────────────────────────────────────────────────────────
    m.save(str(OUT_HTML))
    size_mb = OUT_HTML.stat().st_size / 1_048_576
    print(f"\nSaved: {OUT_HTML}  ({size_mb:.1f} MB)")
    print("\nGitHub Pages:")
    print("  git add docs/map.html && git commit -m 'Add damage map' && git push")
    print("  Settings → Pages → Source: master / /docs")
    print("  URL: https://neuralist.github.io/building-facades-damage-prediction/map.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simplify", type=float, default=0.00001,
                        help="Simplification tolerance in degrees (~5 m, default 0.00005)")
    args = parser.parse_args()
    main(args.simplify)