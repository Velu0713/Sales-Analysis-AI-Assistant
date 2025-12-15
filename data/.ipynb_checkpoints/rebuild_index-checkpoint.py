import pandas as pd
from rag import build_index_from_text_rows
import os

CSV_FILE = "data/Sales.csv"

# Columns to combine
TEXT_COLUMNS = [
    "Customer Name", "City", "State", "Product Name",
    "Category", "Sub-Category", "Segment", "Region"
]

def clean_df(df):
    df = df.fillna("")
    df["__combined__"] = df[TEXT_COLUMNS].astype(str).agg(" | ".join, axis=1)
    return df

def main():
    print("📌 Loading CSV...")
    df = pd.read_csv(CSV_FILE)

    print("📌 Cleaning data...")
    df = clean_df(df)

    print("📌 Extracting text rows...")

    # CREATE (id, text) tuples
    text_rows = []
    for idx, text in enumerate(df["__combined__"].tolist()):
        text_rows.append((str(idx), text))

    print(f"📌 Total rows to embed: {len(text_rows)}")

    print("📌 Building FAISS index...")
    build_index_from_text_rows(text_rows)

    print("\n🎉 DONE! FAISS index stored in vectorstore/ folder.")

if __name__ == "__main__":
    main()

