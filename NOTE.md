pip3 install --cache-dir=/raid/ltnghia01/phucpv/Eventa/tmp -r requirements.txt

pip3 install --cache-dir=/raid/ltnghia01/phucpv/Eventa/tmp https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip3 install --upgrade pip wheel setuptools

### coi gpu:
nvidia-smi

### coi ổ đĩa:
du -sh *
dh -h

## Trước khi chạy
cd phucpv/Eventa
conda deactivate
source ./.venv/bin/activate

## Vô nvidia-smi coi xem gpu nào trống: ví dụ cpu 5 thì vô set device cuda:5 (không được lấy con 1)
## Vừa chạy vừa bật coi xem có bị tràn vram không

# Instructions
### 🔹 Phase 1: Retrieval & Reranking

1. **Create Embeddings**

```bash
python step_1_create_embeddings.py --input_folder data/database_images/database_images_compressed90 --output_folder embeddings/database
python step_1_create_embeddings.py --input_folder data/track1_private/query --output_folder embeddings/query
python step_1_create_embeddings.py --input_folder data/track1_public/query --output_folder embeddings/query
```

2. **Initial Retrieval**
```bash
python step_1_retrieval.py --database_folder embeddings/database --query_folder embeddings/query
```

3. **Reranking**

```bash
python step_1_rerank.py
```

---

### 🔹 Phase 2: Captioning & Semantic Reasoning

4. **Create the crawling database**

**Crawl articles and images from URLs in the origin database:**

```bash
python step_2_0_crawling.py
```
- Get article url from ./data/database.json and crawl
- Save content and metadata into ./crawled/*.json
- Save crawled images into ./imgs/*.jpg

---

**Create Embeddings for crawled images:**

```bash
python step_1_create_embeddings.py --input_folder imgs --output_folder embeddings/maching_new_database_internvlg
```
- Create embeddings for crawled images in ./imgs/*.jpg
- Save embeddings into ./embeddings/maching_new_database_internvlg/*.pt

---

**For each article, create a json file mapping the origin image and the crawled image**

```bash
python step_2_0_matching_image.py
```
- For each article in ./data/database.json, find the corresponding article in ./crawled/*.json
- Then compare the images origin in ./data/database_images/database_images_compressed90/*.jpg with crawled images in ./imgs/*.jpg
- Output the result in matching-01-no-threshold/*.json, with each json corresponds to an article like the following
```
{
  "origin_image_id_1": {
    "filename": "article_001_0.jpg",
    "score": 1.0
  },
  "origin_image_id_2": {
    "filename": "article_001_1.jpg",
    "score": 0.8732
  }
}
```

---

**Create database_new.json**

```bash
python step_2_0_new_database.py
```
- For each article in ./data/database.json, load the original article content and its list of original image IDs.
- Load the corresponding crawled article from ./crawled/*.json (if it exists).
- Load image matching results from matching-01-no-threshold/*.json, which map original image IDs to crawled image filenames and similarity scores.
- For each original image:
  - Find the matched crawled image (if any) using the matching file.
  - Locate the crawled image’s position in the crawled content using image markers.
  - Use surrounding text (before/after the image marker) to find the best matching position in the original article content:
    - Exact substring match first.
    - Fuzzy text match if exact match fails.
- Reconstruct a new article object by:
  - Keeping the original article’s URL, date, title, and full content.
  - Taking category, author, metadata, and statistics from the crawled article.
  - Replacing original images with crawled image URLs and assigning their estimated positions in the original content.
- Output the merged result into a single file database_new.json, where each key corresponds to one article and contains:
  - Original text content.
  - A list of images with { origin_image_id, new_image_url, position, score }.

---

**Create output corresponding to final_json_result/context_extraction_image_article.json**

```bash
python step_2_0_create_result.py
```
- Load the rebuilt article database from ./database_new.json, which contains original article content and mapped images with positions.
- Build a lookup table images_info mapping each image_id to its image metadata (url, position, score).
- Load all crawled articles from ./crawled/*.json to collect image captions and alt texts, indexed by image URL.
- Load image–article query from ./final_json_result/context_extraction_image_article.json.
- For each query, retrieve the corresponding article and image metadata.
- Extract and normalize image text (alt and caption) from crawled data:
  - Strip HTML tags.
  - Remove all urls
- Retrieve the full article content and the estimated image position within the article.
- Construct a unified record for each query containing
  - query_id
  - crawl_alt
  - crawl_caption
  - article (full article text)
  - article_position (image position in article)
  - image_url
  - article_url
  - image_id
  - article_id
- Output all records into result-tanphuc.json.

---

5. **Generate Query Captions**

```bash
python step_2_create_caption_query.py --device cuda:6
```
- Load existing caption results from the output JSON file ./public_test_final_elements_json/final_rerank_public_test_detail_top1_caption.json if it already exists.
- Read query definitions from final_csv_result/temp_final_rerank.csv.
- For each query in the CSV file:
  - Skip the query if captions already exist and are complete.
  - Load the corresponding image from the original image database folder
  - Generate an image caption using the InternVL caption model.
  - Store the generated caption indexed by query_id.
  - Incrementally write caption results to the output JSON file.
- Output the final caption dictionary to final_rerank_public_test_detail_top1_caption.json like:
```
{
  "query_id_001": "...generated caption...",
  "query_id_002": "...generated caption..."
}
```

File này dùng để sinh caption cho ảnh gốc bằng InternVL, theo từng query trong CSV, và lưu kết quả để phục vụ bước rerank / evaluation multimodal.


---

6. **Retrieve First Article Summary**
```bash
python step_2_first_article_summary.py
```
- Read retrieval results from
./final_csv_result/public_final_retrieval_merging_final_results.csv
- For each row in the CSV:
  - Extract query_id
  - Extract the top-1 retrieved article: article_id_1
- Load the article database from ./data/database.json
- For each (query_id, article_id_1) pair:
  - Skip if this query_id has already been processed
  - Retrieve the full article content from the database
  - Use Llama LLM to summarize the article content
- Store the summarized result in a JSON file: ./public_test_final_elements_json/reranking_query_first_article_question_answer.json

```
{
  "query_0001": {
    "article_id": "article_123",
    "summary": "The article discusses recent developments in the global economy, focusing on inflation trends and policy responses."
  },
  "query_0002": {
    "article_id": "article_045",
    "summary": "This news article reports on a major technology conference where new AI products were announced."
  }
}
```

---

7. **Caption Enhancement via Strategies**

- Read and merge intermediate results from multiple input files:
  - Generated image captions from
    ./public_test_final_elements_json/final_rerank_public_test_detail_top1_caption.json
  - Query → top-1 article summaries from
    ./public_test_final_elements_json/reranking_query_first_article_question_answer.json
  - Named entity extraction results from
    ./assemble_result/name_entity_llama.json
  - Question–answer results from
    ./assemble_result/questions_answers_llama.json
  - Enriched article + image context database from
    ./result-tanphuc.json

- Merge all above information into a unified input JSON:
  - Output merged file:
    ./public_test_final_elements_json/final_merge_result.json

- Load prompt templates (Jinja2) from:
  ./assemble_caption_prompt_template/*.j2

- Load few-shot examples from:
  ./assemble_caption_prompt_template/test.json

- For each query entry in ./public_test_final_elements_json/final_merge_result.json:
  - Extract:
    - question–answer context
    - article summary (raw / restruct / fact)
    
    - article content
    - generated image caption
    - crawled image caption

    - named entities
    - image context
    
  - Combine these fields into a single structured prompt using the selected template strategy
  - Send the rendered prompt to a LLaMA-based LLM (via LLMAssembler) to:
    - Generate an enhanced caption
    - OR generate question–answer text (if --qa is enabled)
    - OR extract named entities (if --name_entity is enabled)

- Save LLM-generated results incrementally to:
  ./assemble_result/{strategy}_{model_type}.json

- After processing all queries:
  - Create the final submission file using create_submission()

**Using Question Answering:**

```bash
python step_2_caption_process.py --qa --strategy questions_answers
```

---

**Using Named Entity Extraction:**

```bash
python step_2_caption_process.py --name_entity --strategy name_entity
```

---

**Using Chain-of-Thought Reasoning:**

```bash
python step_2_caption_process.py --strategy cot_5_things_fact_more_event
```

---

8. **Merge all elements**

```bash
python step_2_merge_all_elements.py
```

---

## Note:

After generating captions using `cot_5_things_fact_more_event`, ensure your submission captions are clean:

* 🔹 Remove any unwanted newlines like `\n\n` or stray `\n`.
* 🔹 Convert captions into proper single-line strings.

You can use the provided [`post_processing.py`](post_processing.py) script to automatically clean the final CSV before generating the submission file.


# Contact
- Nam-Quan Nguyen (nnquan23@apcs.fitus.edu.vn)
- Minh-Hoang Le (lmhoang22@apcs.fitus.edu.vn)
- Vinh-Toan Vong (vvtoan22@apcs.fitus.edu.vn)


  "f8097c7d27a8aac6": 

"ce24fe0263ef141f",
      "3eef9607326b2cba",
      "739f701a93b7ca13",
      "be1ea34f22dbd811",
      "228d72213d7a1651",
      "38f7b2209694b55e",
      "0f30a45329e423b2",
      "81d0b968ee9988cd"

find database_images -name 3eef9607326b2cba*