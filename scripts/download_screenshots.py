from playwright.sync_api import sync_playwright
import pandas as pd
from pathlib import Path


def screenshot_from_csv(csv_path, output_dir='screenshots', width=1920, height=1080):
    # CSV с заголовком: building_id, photo_url
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    done = sum((output_dir / f'{row.building_id}.jpg').exists() for _, row in df.iterrows())
    print(f"Всего: {len(df)}, скачано: {done}, осталось: {len(df) - done}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, channel='chrome')
        page = browser.new_page(viewport={'width': width, 'height': height + 300})

        for _, row in df.iterrows():
            path = output_dir / f'{row.building_id}.jpg'
            if path.exists():
                continue

            page.goto(row.photo_url)
            page.wait_for_timeout(5000)

            # скрываем мини-карту и лишние элементы
            page.evaluate("""
                ['.panorama-mini-map', '.panorama-controls',
                 '.panorama-footer', '.panorama-overlay'].forEach(sel => {
                    const el = document.querySelector(sel);
                    if (el) el.style.display = 'none';
                });
            """)

            page.wait_for_timeout(500)
            page.screenshot(path=str(path), clip={'x': 0, 'y': 0, 'width': width, 'height': height})
            print(f'  ok {row.building_id}')

        browser.close()

    done = sum((output_dir / f'{row.building_id}.jpg').exists() for _, row in df.iterrows())
    print(f"Готово: {done} / {len(df)}")


if __name__ == '__main__':
    screenshot_from_csv(
        csv_path='data/processed/prospective_validation_dataset.csv',
        output_dir='screenshots',
    )

