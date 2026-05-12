"""
Map of 402 labeled buildings (point markers).

Input:  data/interim/400-buildings/402_buildings_addresses_coords.csv
Output: docs/labeled_buildings_map.html

Usage:
    python src/visualization/create_labeled_buildings_map.py
"""

from pathlib import Path

import folium
import pandas as pd

ROOT     = Path(__file__).resolve().parents[2]
CSV_FILE = ROOT / "data/interim/400-buildings/402_buildings_addresses_coords.csv"
OUT_DIR  = ROOT / "docs"
OUT_HTML = OUT_DIR / "labeled_buildings_map.html"

PALETTE = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad", "#e67e22", "#16a085", "#c0392b", "#7f8c8d"]


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(CSV_FILE)
    print(f"Loaded: {len(df)} buildings")
    print(f"Columns: {list(df.columns)}")

    lat_col = find_col(df, ["centroid_lat", "lat", "latitude", "Latitude"])
    lon_col = find_col(df, ["centroid_lon", "lon", "longitude", "Longitude"])
    if lat_col is None or lon_col is None:
        raise ValueError(
            f"Coordinate columns not found. Available columns: {list(df.columns)}"
        )

    addr_col     = find_col(df, ["address", "addr", "адрес"])
    district_col = find_col(df, ["district_name", "district", "район"])
    id_col       = find_col(df, ["building_id", "id"])

    df = df.dropna(subset=[lat_col, lon_col])
    print(f"Buildings with coordinates: {len(df)}")

    district_colors: dict[str, str] = {}
    if district_col:
        for i, name in enumerate(sorted(df[district_col].dropna().unique())):
            district_colors[name] = PALETTE[i % len(PALETTE)]
        print(f"Districts: {district_colors}")

    cx = df[lon_col].median()
    cy = df[lat_col].median()
    m  = folium.Map(location=[cy, cx], zoom_start=12, tiles="CartoDB positron")

    for _, row in df.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]

        district = row[district_col] if district_col else None
        color    = district_colors.get(district, PALETTE[0]) if district else PALETTE[0]

        # popup content
        lines = []
        if id_col:
            lines.append(f"<b>ID:</b> {row[id_col]}")
        if addr_col:
            lines.append(f"<b>Адрес:</b> {row[addr_col]}")
        if district_col:
            lines.append(f"<b>Район:</b> {district}")
        lines.append(f"<b>Координаты:</b> {lat:.5f}, {lon:.5f}")
        popup_html = "<br>".join(lines)

        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color="white",
            weight=1.2,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=row[addr_col] if addr_col else f"{lat:.4f}, {lon:.4f}",
        ).add_to(m)

    # legend
    if district_col and df[district_col].notna().any():
        legend_items = ""
        for name, clr in district_colors.items():
            legend_items += (
                    f'<div style="margin:3px 0;">'
                    f'<span style="display:inline-block;width:24px;height:24px;'
                    f'border-radius:50%;background:{clr};border:1.5px solid white;'
                    f'box-shadow:0 0 0 1px #999;vertical-align:middle;margin-right:6px;"></span>'
                    f'{name}</div>\n'
                )
        n_districts = df[district_col].nunique()
        legend_html = f"""
<div style="position:fixed;top:15px;left:10px;z-index:9999;
     background:white;padding:11px 10px;border-radius:8px;
     border:1px solid #ccc;font-size:13px;
     box-shadow:0 2px 8px rgba(0,0,0,.18);">
  <div style="font-weight:bold;margin-bottom:6px;">
    Размеченные здания ({len(df)})
  </div>
  {legend_items}
</div>"""
        m.get_root().html.add_child(folium.Element(legend_html))

    m.save(str(OUT_HTML))
    size_kb = OUT_HTML.stat().st_size / 1024
    print(f"\nSaved: {OUT_HTML}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()