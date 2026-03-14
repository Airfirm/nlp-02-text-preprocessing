"""
text_preprocessing_modified_femi.py

Purpose
  Read text data, preprocess the text, compare raw and cleaned tokens,
  summarize the results, and answer additional business questions.

New Analytical Questions
- What topics or themes appear most often?
- What common 2-word phrases appear in the text?
- How repetitive or diverse is the vocabulary?
- Which records are unusually short or long?
- Which business-related categories are mentioned most?
"""

# ============================================================
# Section 1. Setup and Imports
# ============================================================
from collections import Counter
import logging
from pathlib import Path
import re

from datafun_toolkit.logger import get_logger, log_header, log_path
import matplotlib.pyplot as plt
import polars as pl

print("Imports complete.")

# ============================================================
# Configure Logging
# ============================================================
LOG: logging.Logger = get_logger("CI", level="DEBUG")

ROOT_PATH: Path = Path.cwd()
DATA_PATH: Path = ROOT_PATH / "data"
NOTEBOOKS_PATH: Path = ROOT_PATH / "notebooks"
SCRIPTS_PATH: Path = ROOT_PATH / "scripts"

log_header(LOG, "NLP")
LOG.info("START script.....")

log_path(LOG, "ROOT_PATH", ROOT_PATH)
log_path(LOG, "DATA_PATH", DATA_PATH)
log_path(LOG, "NOTEBOOKS_PATH", NOTEBOOKS_PATH)
log_path(LOG, "SCRIPTS_PATH", SCRIPTS_PATH)

# ============================================================
# Section 2. Read the Text Data
# ============================================================

input_path: Path = DATA_PATH / "text_data_femi.txt"

text_list: list[str] = input_path.read_text(encoding="utf-8").splitlines()
text_list = [line.strip() for line in text_list if line.strip()]

print("Data loaded successfully.")
print(f"Loaded {len(text_list):,} text records.")

raw_text: str = " ".join(text_list)

print(f"Raw text length: {len(raw_text):,} characters")
print("First 500 characters of raw text:")
print(raw_text[:500])

# ============================================================
# Section 3. Inspect the Raw Text
# ============================================================

print("First 5 text records:")
for line in text_list[:5]:
    print("-", line)

print(f"\nLoaded {len(text_list):,} text records.")
print(f"Raw text length: {len(raw_text):,} characters")

print("\nFirst 500 characters of combined text:")
print(raw_text[:500])

# ============================================================
# Section 4. Tokenize the Raw Text
# ============================================================

raw_tokens: list[str] = raw_text.split()
count_of_raw_tokens: int = len(raw_tokens)

print("First 20 raw tokens:")
print(raw_tokens[:20])
print(f"Total raw tokens: {count_of_raw_tokens:,}")

# ============================================================
# Section 5. Normalize the Text
# ============================================================

lower_text: str = raw_text.lower()

print("First 500 characters of lowercase text:")
print(lower_text[:500])

# ============================================================
# Section 6. Remove Punctuation and Tokenize Again
# ============================================================

no_punct_text: str = re.sub(r"[^a-z0-9\s]", " ", lower_text)

tokens_no_punct: list[str] = no_punct_text.split()
count_of_tokens_no_punct: int = len(tokens_no_punct)

print("First 20 tokens after lowercasing and punctuation removal:")
print(tokens_no_punct[:20])
print(f"Total tokens after punctuation removal: {count_of_tokens_no_punct:,}")

# ============================================================
# Section 7. Remove Stop Words
# ============================================================

STOP_WORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}

clean_tokens: list[str] = [
    token for token in tokens_no_punct if len(token) > 2 and token not in STOP_WORDS
]

count_of_clean_tokens: int = len(clean_tokens)

print("First 20 cleaned tokens:")
print(clean_tokens[:20])
print(f"Total cleaned tokens: {count_of_clean_tokens:,}")

# ============================================================
# Section 8. Build a Before/After Summary Table
# ============================================================

summary_df: pl.DataFrame = pl.DataFrame(
    {
        "stage": [
            "raw tokens",
            "after punctuation removal",
            "after stop word removal",
        ],
        "count": [
            count_of_raw_tokens,
            count_of_tokens_no_punct,
            count_of_clean_tokens,
        ],
    }
)

print("Preprocessing summary:")
print(summary_df)

# ============================================================
# Section 9. Build a Frequency Table with Polars
# ============================================================

token_df: pl.DataFrame = pl.DataFrame({"token": clean_tokens})
freq_df: pl.DataFrame = token_df.group_by("token").len().sort("len", descending=True)

print("Top 20 most frequent cleaned tokens:")
print(freq_df.head(20))

# ============================================================
# Section 10. Business Question:
# What topics/themes appear most often?
# ============================================================

print("\nTop 10 themes / keywords:")
print(freq_df.head(10))

# ============================================================
# Section 11. Business Question:
# What common 2-word phrases appear most often?
# ============================================================

bigrams: list[str] = [
    f"{clean_tokens[i]} {clean_tokens[i + 1]}" for i in range(len(clean_tokens) - 1)
]

bigram_counts = Counter(bigrams)

bigram_df: pl.DataFrame = pl.DataFrame(
    {
        "bigram": list(bigram_counts.keys()),
        "count": list(bigram_counts.values()),
    }
).sort("count", descending=True)

print("\nTop 10 bigrams:")
print(bigram_df.head(10))

# ============================================================
# Section 12. Business Question:
# How repetitive or diverse is the vocabulary?
# ============================================================

unique_tokens: int = len(set(clean_tokens))
type_token_ratio: float = (
    unique_tokens / count_of_clean_tokens if count_of_clean_tokens > 0 else 0.0
)

vocab_summary_df: pl.DataFrame = pl.DataFrame(
    {
        "total_clean_tokens": [count_of_clean_tokens],
        "unique_tokens": [unique_tokens],
        "type_token_ratio": [round(type_token_ratio, 4)],
    }
)

print("\nVocabulary summary:")
print(vocab_summary_df)

# ============================================================
# Section 13. Business Question:
# Which records are unusually short or long?
# ============================================================

record_lengths = [len(line.split()) for line in text_list]

records_df: pl.DataFrame = pl.DataFrame(
    {
        "record_text": text_list,
        "word_count": record_lengths,
    }
)

shortest_records_df = records_df.sort("word_count").head(5)
longest_records_df = records_df.sort("word_count", descending=True).head(5)

print("\n5 shortest records:")
print(shortest_records_df)

print("\n5 longest records:")
print(longest_records_df)

# ============================================================
# Section 14. Business Question:
# Which business categories are mentioned most?
# ============================================================
# These are example categories. Update them based on your project.

BUSINESS_CATEGORIES: dict[str, set[str]] = {
    "price": {"price", "cost", "cheap", "expensive", "value", "pricing"},
    "service": {"service", "support", "staff", "help", "agent", "team"},
    "quality": {"quality", "bad", "good", "excellent", "poor", "defective"},
    "delivery": {"delivery", "shipping", "late", "delay", "arrived", "package"},
}

category_counts: dict[str, int] = {}
clean_token_counter = Counter(clean_tokens)

for category, keywords in BUSINESS_CATEGORIES.items():
    category_counts[category] = sum(clean_token_counter[word] for word in keywords)

category_df: pl.DataFrame = pl.DataFrame(
    {
        "category": list(category_counts.keys()),
        "count": list(category_counts.values()),
    }
).sort("count", descending=True)

print("\nBusiness category mentions:")
print(category_df)

# ============================================================
# Section 15. Plot Top Cleaned Tokens
# ============================================================

top_df: pl.DataFrame = freq_df.head(10)

plt.figure(figsize=(10, 5))
plt.bar(top_df["token"], top_df["len"])

ax = plt.gca()
ax.tick_params(axis="x", labelrotation=45)

plt.title("Most Frequent Cleaned Tokens")
plt.xlabel("Token")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ============================================================
# Section 16. Plot Token Counts Across Stages
# ============================================================

plt.figure(figsize=(8, 5))
plt.bar(summary_df["stage"], summary_df["count"])

ax = plt.gca()
ax.tick_params(axis="x", labelrotation=20)

plt.title("Token Counts Across Preprocessing Stages")
plt.xlabel("Stage")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ============================================================
# Section 17. Plot Top Bigrams
# ============================================================

top_bigram_df = bigram_df.head(10)

plt.figure(figsize=(10, 5))
plt.bar(top_bigram_df["bigram"], top_bigram_df["count"])

ax = plt.gca()
ax.tick_params(axis="x", labelrotation=45)

plt.title("Most Frequent Bigrams")
plt.xlabel("Bigram")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ============================================================
# Section 18. Plot Business Categories
# ============================================================

plt.figure(figsize=(8, 5))
plt.bar(category_df["category"], category_df["count"])

ax = plt.gca()
ax.tick_params(axis="x", labelrotation=20)

plt.title("Business Category Mentions")
plt.xlabel("Category")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ============================================================
# LOG THE END
# ============================================================

LOG.info("========================")
LOG.info("Pipeline executed successfully!")
LOG.info("========================")
LOG.info("END main()")
