import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_MASKS_DIR = BASE_DIR / "data/processed/505-imgs-4-classes/masks/train"

NUM_CLASSES = 5
CLASS_NAMES = {
    0: "background",
    1: "coating_deterioration",
    2: "masonry_degradation",
    3: "moisture_bio_damage",
    4: "vandalism"
}

print(TRAIN_MASKS_DIR)

def calculate_class_weights(masks_dir):
    pixel_counts = {i: 0 for i in range(NUM_CLASSES)}
    total_pixels = 0
    total_images = 0

    for mask_path in sorted(masks_dir.glob("*.png")):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        unique, counts = np.unique(mask, return_counts=True)

        for val, count in zip(unique, counts):
            if val in pixel_counts:
                pixel_counts[val] += count

        total_pixels += mask.size
        total_images += 1

    print(f"Обработано масок: {total_images}")
    print("\nСтатистика по классам:")
    for cls, count in pixel_counts.items():
        pct = count / total_pixels * 100
        print(f"  {CLASS_NAMES[cls]}: {count} пикселей ({pct:.4f}%)")

    # Median Frequency Balancing: weight_i = median_freq / freq_i
    # для того, чтобы редкие классы получили большой вес, а частые — маленький
    valid_counts = [c for c in pixel_counts.values() if c > 0]
    median_freq = np.median([c / total_pixels for c in valid_counts])

    weights = []
    print("\nВеса классов:")
    for cls in range(NUM_CLASSES):
        count = pixel_counts[cls]
        if count == 0:
            w = 0.0
        else:
            w = median_freq / (count / total_pixels)
        weights.append(w)
        print(f"  {CLASS_NAMES[cls]}: {w:.4f}")

    weights_str = ", ".join([f"{w:.4f}" for w in weights])
    print(f"\nweights = [{weights_str}]")


if __name__ == "__main__":
    calculate_class_weights(TRAIN_MASKS_DIR)