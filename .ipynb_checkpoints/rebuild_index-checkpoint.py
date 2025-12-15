import pandas as pd
from rag import build_index_from_text_rows

# Path to your CSV file
CSV_FILE = "data/Sales.csv"   # Change to your actual file name

# Columns in your dataset
COLUMNS = [
    "Order ID", "Order Date", "Ship Date",
    "Customer Name", "Segment", "Region",
    "Product ID", "Category", "Sub-Category", "Product Name",
    "Sales", "Quantity", "Discount", "Profit"
]

def prepare_text_rows(df):
    """Combine selected columns into a single text field safely."""
    
    # Use only columns that exist (avoids KeyError)
    available = [c for c in COLUMNS if c in df.columns]
    print("Columns used for embedding:", available)

    df = df.fillna("")

    # Combine all selected columns into one text string
    df["combined"] = df[available].astype(str).agg(" | ".join, axis=1)

    # Prepare in (id, text) format
    text_rows = [(str(i), row) for i, row in enumerate(df["combined"].tolist())]
    return text_rows


def main():
    print("📌 Loading CSV...")
    df = pd.read_csv(CSV_FILE)

    print("📌 Preparing combined text rows...")
    text_rows = prepare_text_rows(df)

    print(f"📌 Total rows ready for embedding: {len(text_rows)}")

    print("📌 Building FAISS vector index...")
    build_index_from_text_rows(text_rows)

    print("\n🎉 DONE! Index saved in vectorstore/ folder.")


if __name__ == "__main__":
    main()