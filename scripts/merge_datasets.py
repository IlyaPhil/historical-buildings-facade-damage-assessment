import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Корневая папка
BASE_DIR = Path(__file__).resolve().parent.parent

# Датасет из CVAT из первого аккаунта (205 патчей, 800x800, PNG)
DS1_RAW = BASE_DIR / "data/raw/1"
DS1_MASKS = BASE_DIR / "data/raw/1/SegmentationClass"

# Датасет из CVAT из второга аккаунта (300 патчей, 800x800, JPG)
DS2_RAW = BASE_DIR / "data/raw/2"
DS2_MASKS = BASE_DIR / "data/raw/2/SegmentationClass"

# Папка для объединенного датасета
OUT_DIR = BASE_DIR / "data/processed/unified_dataset"
OUT_IMAGES = OUT_DIR / "images"
OUT_MASKS = OUT_DIR / "masks"

TARGET_SIZE = (800, 800)

# Цветовые карты в BGR для cv2
# Два датасета размечались в разных аккаунтах CVAT, поэтому цвета классов различаются
BGR_COLOR_MAP_1 = {
    (0, 0, 0):      0,  # background
    (217, 21, 54):  1,  # coating_deterioration
    (183, 50, 250): 1,  # cracks (объединяем с coating_deterioration)
    (209, 125, 42): 2,  # masonry_degradation
    (98, 221, 38):  3,  # moisture_bio_damage
    (123, 28, 222): 4   # vandalism
}

BGR_COLOR_MAP_2 = {
    (0, 0, 0):       0,  # background
    (224, 69, 94):   1,  # coating_deterioration
    (152, 89, 219):  1,  # cracks (объединяем с coating_deterioration)
    (209, 125, 42):  2,  # masonry_degradation
    (160, 244, 184): 3,  # moisture_bio_damage
    (116, 86, 241):  4   # vandalism
}

# Приоритет классов для стратификации с таким расчетом, чтобы самый редкий класс (vandalism) был равномерно представлен в train и val
CLASS_PRIORITY = {4: 4, 2: 2, 3: 3, 1: 1, 0: 0}


def rgb_to_index(mask_bgr, color_map):
    """Конвертирует цветную BGR-маску в одноканальную маску с индексами классов."""
    mask_idx = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)
    for bgr_color, class_idx in color_map.items():
        matches = np.all(mask_bgr == bgr_color, axis=-1)
        mask_idx[matches] = class_idx
    return mask_idx


def get_priority_class(mask_path, is_ds1):
    """Возвращает наиболее приоритетный класс в маске — используется для стратификации сплита."""
    mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    color_map = BGR_COLOR_MAP_1 if is_ds1 else BGR_COLOR_MAP_2
    mask_idx = rgb_to_index(mask_bgr, color_map)

    present_classes = np.unique(mask_idx)
    return max(present_classes, key=lambda c: CLASS_PRIORITY.get(c, 0))


def process_dataset(raw_dir, masks_dir, img_ext, resize_needed=False, is_ds1=True):
    """
    Собирает список пар (изображение, маска) из одного датасета.
    Возвращает список словарей с путями и метаданными для последующей обработки.
    """
    data_list = []

    for mask_file in os.listdir(masks_dir):
        if not mask_file.endswith('.png'):
            continue

        base_name = os.path.splitext(mask_file)[0]
        mask_path = masks_dir / mask_file

        # Ищем изображение с подходящим расширением
        img_path = None
        for ext in [img_ext, '.jpg', '.JPG', '.png', '.jpeg']:
            p = raw_dir / (base_name + ext)
            if p.exists():
                img_path = p
                break

        if img_path is None:
            print(f"Для маски {mask_file} не найдено изображение")
            continue

        strata_class = get_priority_class(mask_path, is_ds1)
        data_list.append({
            'img_path':  img_path,
            'mask_path': mask_path,
            'basename':  base_name,
            'strata':    strata_class,
            'resize':    resize_needed,
            'is_ds1':    is_ds1
        })

    return data_list


def copy_and_process(data_subset, split_name):
    """Копирует изображения и маски в выходную директорию и приводит их к одному размеру"""
    for item in tqdm(data_subset, desc=f"Обработка {split_name}"):
        img_out = OUT_IMAGES / split_name / f"{item['basename']}.png"
        mask_out = OUT_MASKS  / split_name / f"{item['basename']}.png"

        img = cv2.imread(str(item['img_path']))
        mask_bgr = cv2.imread(str(item['mask_path']), cv2.IMREAD_COLOR)

        color_map = BGR_COLOR_MAP_1 if item['is_ds1'] else BGR_COLOR_MAP_2
        mask = rgb_to_index(mask_bgr, color_map)

        if item['resize']:
            img = cv2.resize(img,  TARGET_SIZE, interpolation=cv2.INTER_AREA)            
            mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(img_out),  img)
        cv2.imwrite(str(mask_out), mask)


def main():   

    # DS1: 205 патчей, 800x800, PNG
    data1 = process_dataset(DS1_RAW, DS1_MASKS, img_ext='.png', resize_needed=False, is_ds1=True)
    # DS2: 300 патчей, 1200x1200, JPG (нужен ресайз до 800x800)
    data2 = process_dataset(DS2_RAW, DS2_MASKS, img_ext='.JPG', resize_needed=True,  is_ds1=False)

    all_data = data1 + data2
    print(f"Найдено пар: DS1={len(data1)}, DS2={len(data2)}, всего={len(all_data)}")

    y_strata = [item['strata'] for item in all_data]  

    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42, stratify=y_strata)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    for split in ['train', 'val']:
        (OUT_IMAGES / split).mkdir(parents=True, exist_ok=True)
        (OUT_MASKS  / split).mkdir(parents=True, exist_ok=True)

    copy_and_process(train_data, 'train')
    copy_and_process(val_data,   'val')

    print("Датасет сохранён в:", OUT_DIR)


if __name__ == "__main__":
    main()