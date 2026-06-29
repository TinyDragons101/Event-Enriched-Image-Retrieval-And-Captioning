import json
import spacy
import string
from statistics import mean
from tqdm import tqdm
from pathlib import Path
from step_2_entity_scorer import EntityHeuristicScorer, save_scoring_results

# Giả sử bạn có các hàm utility này
# Nếu chưa có, tôi sẽ định nghĩa luôn
def word_length(text):
    """Đếm số từ trong text"""
    return len(text.split())

def remove_punctuation(text):
    """Loại bỏ dấu câu"""
    import string
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_punctuation_loose(text):
    """Loại bỏ dấu câu nhẹ hơn, giữ lại dấu gạch ngang"""
    punctuation = string.punctuation.replace('-', '')
    return text.translate(str.maketrans('', '', punctuation))

def normalize_spaces(text):
    """Chuẩn hóa khoảng trắng"""
    return ' '.join(text.split())

def chunking_words(text, max_words):
    """Cắt text về max_words"""
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words])

def rebuild_caption(tokens):
    """Ghép lại caption từ danh sách tokens"""
    caption = ""
    for i, token in enumerate(tokens):
        if i > 0:
            caption += " "
        caption += token
    return caption

def is_name_entity(word, entity_names):
    """Kiểm tra xem word có phải là named entity không"""
    word = remove_punctuation_loose(word)
    if word in entity_names:
        return True
    return False

def read_json(path):
    """Đọc file JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    """Lưu file JSON"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class CaptionNormalizer:
    """
    Normalizer để xử lý caption sau khi assemble từ LLM
    Tương tự như sematicNorm và entityEnricker
    """

    def __init__(self, max_words=105, expected_length=104, use_entity_scoring=False, entity_config_path=None):
        self.max_words = max_words
        self.expected_length = expected_length
        self.nlp = spacy.load("en_core_web_sm")

        # NEW: Entity scorer
        self.use_entity_scoring = use_entity_scoring
        if use_entity_scoring:
            self.entity_scorer = EntityHeuristicScorer(config_path=entity_config_path)
            print("✅ EntityHeuristicScorer initialized")
        else:
            self.entity_scorer = None

    def truncate_caption_semantic(self, text, max_word, expected_length):
        """
        Cắt ngắn caption dựa trên ngữ nghĩa, giữ lại named entities
        Tương tự hàm sematicNorm
        """
        text = normalize_spaces(text.replace("\n", ""))

        # Nếu đã ngắn hơn max_word thì return
        if word_length(remove_punctuation(text)) <= max_word:
            return text

        word_count = word_length(text)
        doc = self.nlp(text)

        # Trích xuất named entities
        entity_names = set(word for ent in doc.ents for word in ent.text.split())

        ls_word = text.split(" ")
        entity_mask = [False] * word_count

        # Đánh dấu các từ là entity
        for idx, wordi in enumerate(ls_word):
            entity_mask[idx] = is_name_entity(wordi, entity_names)

        word_entity_pairs = list(zip(ls_word, entity_mask))

        # Cắt bớt các từ không phải entity từ cuối câu
        truncated = []
        count = 0
        ngram_count = word_length(remove_punctuation(text))

        for word, is_entity in reversed(word_entity_pairs):
            if not is_entity:  # Không phải entity
                if ngram_count - count > expected_length:
                    count += 1
                    continue
            truncated.append((word, is_entity))

        truncated = list(reversed(truncated))
        final_tokens = [tok for tok, _ in truncated]

        caption = rebuild_caption(final_tokens).replace(",.", ".").replace(".,", ".")

        # Đảm bảo kết thúc bằng dấu chấm
        if caption and caption[-1].isalpha():
            caption += '.'
        elif caption and caption[-1] not in '.!?':
            caption = caption[:-1] + "."

        return caption

    def extract_entities_and_phrases(self, query_data):
        """
        MODIFIED: Use EntityHeuristicScorer instead of simple extraction
        Trích xuất named entities và noun phrases từ article
        """
        if self.use_entity_scoring and self.entity_scorer:
            # NEW APPROACH: Heuristic scoring
            selected_entities, _ = self.entity_scorer.score_query(query_data)

            # Extract noun phrases separately (keep this for backward compatibility)
            text = query_data.get("article", "")
            doc = self.nlp(text)
            noun_phrases = [chunk.text.strip() for chunk in doc.noun_chunks]

            return selected_entities, noun_phrases

        else:
            # OLD APPROACH: Extract all entities without filtering
            text = query_data.get("article", "")
            doc = self.nlp(text)

            entities = []
            noun_phrases = []

            # Named entities
            for ent in doc.ents:
                entities.append(ent.text.strip())

            # Noun phrases
            for chunk in doc.noun_chunks:
                noun_phrases.append(chunk.text.strip())

            return entities, noun_phrases

    def enrich_caption(self, caption, entities, noun_phrases, cutting_idx):
        """
        Làm giàu caption bằng cách thêm entities và noun phrases
        Tương tự entityEnricker
        """
        if word_length(remove_punctuation(caption)) >= cutting_idx:
            return caption

        caption_clean = remove_punctuation(caption.lower())

        # Thêm entities
        for entity in entities:
            if word_length(remove_punctuation(caption)) >= cutting_idx:
                break
            entity_clean = entity.lower().strip()
            if entity_clean and entity_clean not in caption_clean:
                caption += " " + entity
                caption_clean = remove_punctuation(caption.lower())

        # Thêm noun phrases
        for phrase in noun_phrases:
            if word_length(remove_punctuation(caption)) >= cutting_idx:
                break
            phrase_clean = phrase.lower().strip()
            if phrase_clean and phrase_clean not in caption_clean:
                caption += " " + phrase
                caption_clean = remove_punctuation(caption.lower())

        return caption

    def normalize_llm_output(self, llm_caption):
        """
        Làm sạch output từ LLM (loại bỏ preamble, "Here is the caption:", v.v.)
        """
        # Loại bỏ các câu giới thiệu thường gặp
        preambles = [
            "Here is the caption:",
            "Here is the rewritten caption:",
            "Caption:",
            "The caption is:",
            "Here's the caption:",
        ]

        caption = llm_caption.strip()
        for preamble in preambles:
            if caption.startswith(preamble):
                caption = caption[len(preamble):].strip()

        # Loại bỏ quotes nếu có
        if caption.startswith('"') and caption.endswith('"'):
            caption = caption[1:-1]

        return caption

    def process_single_caption(self, query_data, llm_caption):
        """
        Xử lý một caption đơn lẻ
        """
        # Bước 1: Làm sạch output từ LLM
        caption = self.normalize_llm_output(llm_caption)

        # Bước 2: Cắt ngắn dựa trên ngữ nghĩa (giữ entities)
        caption = self.truncate_caption_semantic(
            caption,
            self.max_words,
            self.expected_length
        )

        # Bước 3: Trích xuất entities và phrases từ article
        entities, noun_phrases = self.extract_entities_and_phrases(query_data)

        # Bước 4: Làm giàu caption nếu còn ngắn
        caption = self.enrich_caption(
            caption,
            entities,
            noun_phrases,
            self.expected_length
        )

        # Bước 5: Cắt lại nếu vượt quá
        if word_length(remove_punctuation(caption)) > self.expected_length:
            caption = chunking_words(caption, self.expected_length)

        # Đảm bảo kết thúc đúng
        caption = normalize_spaces(caption)
        if caption and caption[-1] not in '.!?':
            caption += '.'

        return caption

    def normalize_batch(
        self,
        llm_output_path,
        merge_data_path,
        output_path,
        verbose=True,
        save_entity_scores=True,
        entity_scores_path="assemble_result/entity_heuristic_scores.json"
    ):
        """
        MODIFIED: Save entity scoring results for debugging
        Xử lý toàn bộ batch caption

        Args:
            llm_output_path: Path đến file JSON output từ LLM (dict: query_id -> caption)
            merge_data_path: Path đến file merge data (list of dicts với article info)
            output_path: Path để lưu kết quả
            verbose: In thống kê
            save_entity_scores: Lưu entity scoring debug info (NEW)
            entity_scores_path: Path để lưu entity scores (NEW)
        """
        # Đọc dữ liệu
        llm_outputs = read_json(llm_output_path)
        merge_data = read_json(merge_data_path)

        # Tạo dict để lookup nhanh
        query_dict = {item['query_id']: item for item in merge_data}

        # NEW: Storage for entity scoring results
        entity_scoring_results = {}

        # Xử lý từng caption
        results = []
        lengths = []

        for query_id, llm_caption in tqdm(llm_outputs.items(), desc="Normalizing captions"):
            query_data = query_dict.get(query_id, {})

            # NEW: Capture entity scoring debug info
            if self.use_entity_scoring and save_entity_scores and self.entity_scorer:
                # Re-run scoring to get debug info
                _, debug_info = self.entity_scorer.score_query(query_data)
                entity_scoring_results[query_id] = debug_info

            # Xử lý caption
            normalized_caption = self.process_single_caption(query_data, llm_caption)

            # Tạo kết quả
            result = {
                "query_id": query_id,
                "original_llm_caption": llm_caption,
                "normalized_caption": normalized_caption,
                "caption_length": word_length(remove_punctuation(normalized_caption))
            }
            results.append(result)
            lengths.append(result["caption_length"])

        # Lưu kết quả
        save_json(results, output_path)

        # NEW: Save entity scoring results
        if save_entity_scores and entity_scoring_results:
            save_scoring_results(entity_scoring_results, entity_scores_path)

        # In thống kê
        if verbose:
            print("\n" + "="*50)
            print("Caption Normalization Statistics:")
            print("="*50)
            print(f"Total captions processed: {len(results)}")
            print(f"Mean length: {mean(lengths):.2f} words")
            print(f"Min length: {min(lengths)} words")
            print(f"Max length: {max(lengths)} words")
            print("="*50)

            # NEW: Entity scoring statistics
            if entity_scoring_results:
                total_extracted = sum(item['total_entities_extracted'] for item in entity_scoring_results.values())
                total_selected = sum(item['entities_selected'] for item in entity_scoring_results.values())
                avg_extracted = total_extracted / len(entity_scoring_results)
                avg_selected = total_selected / len(entity_scoring_results)

                print("\n" + "="*50)
                print("Entity Scoring Statistics:")
                print("="*50)
                print(f"Avg entities extracted per query: {avg_extracted:.1f}")
                print(f"Avg entities selected per query: {avg_selected:.1f}")
                print(f"Selection rate: {(total_selected/total_extracted*100) if total_extracted > 0 else 0:.1f}%")
                print("="*50)

        return results


def main():
    """
    Hàm main để chạy normalizer
    """
    # Cấu hình
    LLM_OUTPUT_PATH = "./assemble_result/cot_5_things_fact_more_event_llama.json"
    MERGE_DATA_PATH = "./public_test_final_elements_json/final_merge_result.json"
    OUTPUT_PATH = "./assemble_result/normalized_captions.json"

    MAX_WORDS = 105
    EXPECTED_LENGTH = 104

    # Khởi tạo normalizer
    normalizer = CaptionNormalizer(
        max_words=MAX_WORDS,
        expected_length=EXPECTED_LENGTH
    )

    # Xử lý
    print("Starting caption normalization...")
    results = normalizer.normalize_batch(
        llm_output_path=LLM_OUTPUT_PATH,
        merge_data_path=MERGE_DATA_PATH,
        output_path=OUTPUT_PATH,
        verbose=True
    )

    print(f"\nNormalized captions saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
