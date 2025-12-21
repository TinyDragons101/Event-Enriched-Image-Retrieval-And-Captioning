import pandas as pd
import argparse
import re

PREFIX_PATTERNS = [
    r'^Here is the caption:\s*',
    r'^Here is the rewritten caption:\s*',
    r'^Here is a rewritten caption:\s*',
    r'^Here is an image caption:\s*',
    r'^e is the caption:\s*',   # case bị cắt chữ như ví dụ của bạn
]

def clean_caption(text):
    if pd.isna(text):
        return ""

    text = text.strip()

    # Remove unwanted prefixes
    for pattern in PREFIX_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Replace newlines and normalize spaces
    text = text.replace('\n', ' ')
    text = ' '.join(text.split())

    return text


def post_process_csv(input_csv_path, output_csv_path=None):
    df = pd.read_csv(input_csv_path)

    if 'generated_caption' not in df.columns:
        raise ValueError("CSV must contain a 'generate_caption' column.")

    df['generated_caption'] = df['generated_caption'].apply(clean_caption)

    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
        print(f"Post-processed CSV saved to {output_csv_path}")
    else:
        print(df.to_csv(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, help="Path to the input CSV file.", default="./final_csv_result/submission_final.csv")
    parser.add_argument("--output_csv", type=str, help="Path to save the cleaned CSV file.", default="./final_csv_result/submission_final.csv")
    args = parser.parse_args()

    post_process_csv(args.input_csv, args.output_csv)