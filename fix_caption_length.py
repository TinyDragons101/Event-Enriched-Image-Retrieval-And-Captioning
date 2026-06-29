import csv

MAX_WORDS = 110
PAD_TEXT = "More details available"

input_csv = "./final_csv_result/temp.csv"
output_csv = "./final_csv_result/temp_fix.csv"


def fix_caption_length(caption):
    caption = caption.strip()
    words = caption.split()

    # Case 1: Dư word → cắt
    if len(words) > MAX_WORDS:
        return " ".join(words[:MAX_WORDS])

    # Case 2: Thiếu word → pad
    pad_words = PAD_TEXT.split()
    while len(words) < MAX_WORDS:
        for w in pad_words:
            if len(words) < MAX_WORDS:
                words.append(w)
            else:
                break

    return " ".join(words)


with open(input_csv, "r", encoding="utf-8") as fin, \
     open(output_csv, "w", encoding="utf-8", newline="") as fout:

    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)

    writer.writeheader()

    for row in reader:
        row["generated_caption"] = fix_caption_length(row["generated_caption"])
        writer.writerow(row)

print("Done! Saved to temp_fix.csv")