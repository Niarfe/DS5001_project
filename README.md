# Lexical Geometry Project — Overview and Next Steps

## What This Is

This project builds an **interpretable lexical geometry system** for comparing two corpora.

At a high level:

- Take two corpora
- Compute normalized word frequencies
- Represent each word as a point `(x, y)`
  - `x` = frequency in corpus A
  - `y` = frequency in corpus B
- Partition the 2D space into meaningful regions

---

## Geometric Regions and Meaning

### 1. Black Hole
- Very low-frequency words
- Weak signal, unreliable
- Typically ignored

### 2. Diagonal Band
- Words with similar frequency in both corpora
- Behave like **stop words / shared background words**

### 3. X-Axis Region
- High frequency in corpus A, low in B
- **Corpus A keywords (topic-specific)**

### 4. Y-Axis Region
- High frequency in corpus B, low in A
- **Corpus B keywords**

### 5. X-Wedge
- Moderately A-biased words
- **Style / fingerprint words for corpus A**

### 6. Y-Wedge
- Moderately B-biased words
- **Style / fingerprint words for corpus B**

---

## Core Insight

There are **three meaningful lexical layers**:

- **Stop words** → diagonal  
- **Topic words** → axes  
- **Style/fingerprint words** → wedges  

The wedge regions capture **generalizable stylistic signal**, not just topic.

---

## What the System Currently Does

- Builds a **symmetric vocabulary** across two corpora
- Computes normalized frequencies
- Visualizes word geometry
- Classifies words into six regions
- Extracts top words per region
- Produces interpretable plots and ranked outputs

---

## Why This Is Interesting

- Not a black-box model
- Fully **interpretable**
- Visual explanation of behavior
- Connects frequency statistics to linguistic structure

---

## Next Improvements

### 1. Classification on Unseen Text
Use extracted region words as a signature to classify new documents.

Example:
- Compare overlap with corpus A vs corpus B signature
- Assign based on strongest match

---

### 2. SDR-Style Matching
Use sparse distributed representation ideas:

- Select top N words per region
- Define match threshold (e.g. M out of N)
- Weighted matching based on region importance

Useful split:
- Axis words → strong topic signals
- Wedge words → strong style signals

---

### 3. N-Grams (Bigrams / Trigrams)
Extend vocabulary beyond unigrams.

Benefits:
- Captures phrase patterns
- Improves style detection

Examples:
- "could not"
- "of the"
- "it was"

Tradeoffs:
- Increased sparsity
- Larger vocabulary
- More complexity

---

### 4. Threshold Tuning
Current boundaries are heuristic.

Future work:
- Tune diagonal band width
- Tune wedge/axis thresholds
- Tune black-hole radius empirically

---

### 5. Stability Analysis
Measure how results change with corpus size.

Goal:
- Identify minimum text size for reliable signal

---

### 6. Multi-Corpus Extensions
Generalize beyond A vs B:

- Author vs author
- Language detection
- Domain classification
- One-vs-rest comparisons

---

### 7. Improved Ranking Functions
Refine scoring per region:

- Stop words → rank by frequency
- Axis words → rank by asymmetry
- Wedge words → rank by combined strength and bias

---

## Summary Statement

This project constructs a geometric representation of lexical frequency across two corpora, revealing six interpretable regions: low-signal words, shared stop words, corpus-specific keywords, and intermediate style/fingerprint zones. These regions can be used to extract lexical signatures and build a lightweight, interpretable classifier. Future extensions include SDR-style matching, n-gram features, and empirical threshold tuning.
```
