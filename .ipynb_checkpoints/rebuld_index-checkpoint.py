import pandas as pd
from rag import extract_text_rows, build_index_from_text_rows

CSV_PATH = "data/Sample_Superstore.csv"   # Your dataset path

def main():
    print("📌 Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    print("📌 Cleaning data...")
    df = df.fillna("")

    print("📌 Extracting text rows...")
    text_rows = extract_text_rows(df)

    print(f"📌 Total rows to embed: {len(text_rows)}")

    print("📌 Building FAISS index...")
    build_index_from_text_rows(text_rows)

    print("🎉 DONE! FAISS index built successfully.")

if __name__ == "__main__":
    main()
