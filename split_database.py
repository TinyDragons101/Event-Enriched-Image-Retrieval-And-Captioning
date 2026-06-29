
import os
import random
import json
import shutil

image_dir = "webCrawl/src/database_image"
query_dir = "data/vietnamNews/query.csv"
query_image_dir = "data/vietnamNews/query_image/"
number_query = 5000

random.seed(42)

list_image = os.listdir(image_dir)

list_choice = random.sample(list_image, number_query)

# remove dot from file name
list_choice = [img.split('.')[0] for img in list_choice]

# save to csv file with query_index format
os.makedirs(os.path.dirname(query_dir), exist_ok=True)
with open(query_dir, 'w', encoding='utf-8') as f:
    f.write("query_index\n")
    for img in list_choice:
        f.write(f"{img}\n")

# copy images from list_choice to query_image_dir (JPEG, png, Webp, etc.)
os.makedirs(query_image_dir, exist_ok=True)
for img in list_choice:
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        img_path = os.path.join(image_dir, img + ext)
        if os.path.exists(img_path):
            shutil.copy(img_path, query_image_dir)
            break
