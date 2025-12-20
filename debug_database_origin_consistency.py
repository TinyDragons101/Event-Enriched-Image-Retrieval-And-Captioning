import os
import json
from pathlib import Path

# Paths (update if needed)
DB_JSON = Path("data/database.json")
ORIGIN_IMG_DIR = Path("data/database_images/database_images_compressed90")

# 1. Read database.json
with open(DB_JSON, encoding="utf-8") as f:
    db = json.load(f)

# 2. Collect all image IDs from database.json
db_image_ids = set()
for article in db.values():
    db_image_ids.update(article.get("images", []))

print(f"Total image IDs in database.json: {len(db_image_ids)}")

# 3. List all image files in origin folder
origin_files = [f for f in os.listdir(ORIGIN_IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
origin_image_ids = set(os.path.splitext(f)[0] for f in origin_files)
print(f"Total image files in origin folder: {len(origin_image_ids)}")

# 4. Compare
missing_in_origin = db_image_ids - origin_image_ids
extra_in_origin = origin_image_ids - db_image_ids

print(f"Images in database.json but missing in origin folder: {len(missing_in_origin)}")
if missing_in_origin:
    print("Examples:", list(missing_in_origin)[:10])

print(f"Images in origin folder but not in database.json: {len(extra_in_origin)}")
if extra_in_origin:
    print("Examples:", list(extra_in_origin)[:10])
