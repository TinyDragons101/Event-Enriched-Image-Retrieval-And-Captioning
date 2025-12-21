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

* Crawl articles and images from URLs in the origin database:

```bash
python step_2_0_crawling.py
```

* Create Embeddings for crawled images:

```bash
python step_1_create_embeddings.py --input_folder imgs --output_folder embeddings/maching_new_database_internvlg
```

* For each article, create a json file mapping the origin image and the crawled image

```bash
python step_2_0_matching_image.py
```

* Create database_new.json

```bash
python step_2_0_new_database.py
```

* Create output corresponding to final_json_result/context_extraction_image_article.json

```bash
python step_2_0_create_result.py
```

5. **Generate Query Captions**

```bash
python step_2_create_caption_query.py --device cuda:6
```

6. **Retrieve First Article Summary**
Llama đang dùng cuda:4 -> check trong code
```bash
python step_2_first_article_summary.py
```

7. **Caption Enhancement via Strategies**

* Using Question Answering:

```bash
python step_2_caption_process.py --qa --strategy questions_answers
```

* Using Named Entity Extraction:

```bash
python step_2_caption_process.py --name_entity --strategy name_entity
```

* Using Chain-of-Thought Reasoning:

```bash
python step_2_caption_process.py --strategy cot_5_things_fact_more_event
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