import json
import random
from pathlib import Path

DB_PATH = Path("./data/database.json")
CRAWLED_FOLDER = Path("crawled")

# Đọc database gốc
with open(DB_PATH, encoding="utf-8") as f:
    db = json.load(f)

# Lấy danh sách key bài báo
keys = list(db.keys())

# Lấy 10 bài báo ngẫu nhiên
sample_keys = random.sample(keys, min(10, len(keys)))

for key in sample_keys:
    article = db[key]
    images = article.get("images", [])
    if not images:
        print(f"[LOG] Bài báo {key} không có ảnh.")
        continue
    first_img_id = images[0]
    # Tìm trong tất cả file json crawl xem có xuất hiện id này không
    found = False
    for crawl_file in CRAWLED_FOLDER.glob("*.json"):
        with open(crawl_file, encoding="utf-8") as f:
            crawl_data = json.load(f)
        for img in crawl_data.get("images", []):
            # So sánh id gốc với id crawl (có thể cần chuẩn hóa extension)
            if first_img_id in img["id"]:
                found = True
                break
        if found:
            break
    print(f"[LOG] Bài báo {key} - Ảnh đầu tiên: {first_img_id} - {'Có' if found else 'Không'} xuất hiện trong crawl.")
