"""
Конвертирует индексные (grayscale) маски датасета в RGB для визуального контроля.

Запуск:
    python export_rgb_masks.py datasets/tiles_800x800_505_imgs-v3
    python export_rgb_masks.py datasets/tiles_800x800_505_imgs-v3 --split val
    python export_rgb_masks.py datasets/tiles_800x800_505_imgs-v3 --out my_output_folder
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

CLASS_COLORS = {
    0: (30,  30,  30),   # background
    1: (54,  21,  217),  # coating_deterioration
    2: (250, 50,  183),  # cracks
    3: (42,  125, 209),  # masonry_degradation
    4: (38,  221, 98),   # moisture_bio_damage
    5: (222, 28,  123),  # vandalism
}


def index_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        rgb[mask == cls] = color
    unknown = ~np.isin(mask, list(CLASS_COLORS.keys()))
    if unknown.any():
        rgb[unknown] = (255, 255, 0)  # жёлтый — неизвестное значение
    return rgb


def export(dataset_dir: Path, split: str, out_dir: Path) -> None:
    masks_dir = dataset_dir / "masks" / split
    if not masks_dir.exists():
        print(f"Папка не найдена: {masks_dir}")
        return

    files = sorted(masks_dir.glob("*.png"))
    if not files:
        print(f"PNG-файлов не найдено в {masks_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Конвертирую {len(files)} масок из {masks_dir}  ->  {out_dir}")

    unknown_values: dict[str, list] = {}

    for f in tqdm(files, unit="mask"):
        mask = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  Не удалось прочитать: {f.name}")
            continue

        # Предупреждение о неизвестных значениях пикселей
        unique_vals = np.unique(mask)
        bad = [int(v) for v in unique_vals if v not in CLASS_COLORS]
        if bad:
            unknown_values[f.name] = bad

        rgb = index_to_rgb(mask)
        # cv2 пишет BGR, поэтому конвертируем из RGB
        cv2.imwrite(str(out_dir / f.name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    if unknown_values:
        print("\nВНИМАНИЕ: найдены неизвестные значения пикселей (выделены жёлтым):")
        for fname, vals in unknown_values.items():
            print(f"  {fname}: значения {vals}")

    print(f"\nГотово. {len(files)} файлов сохранено в: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path,
                        help="data/processed/505-imgs-4-classes)")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both",
                        help="Какой сплит обрабатывать (default: both)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Папка для сохранения (default: <dataset>/masks_rgb)")
    args = parser.parse_args()

    dataset_dir = args.dataset
    if not dataset_dir.exists():
        print(f"Датасет не найден: {dataset_dir}")
        return

    out_base = args.out or (dataset_dir / "masks_rgb")

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        export(dataset_dir, split, out_base / split)


if __name__ == "__main__":
    main()