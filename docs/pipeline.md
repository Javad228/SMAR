# The SMAR Pipeline: From Paper to Presentation (A → J)

---

## A) Parse & Clean the Paper

First, the PDF is converted into a clean JSON file, preserving its structure:

- **Title & Abstract**: The core summary.
- **Sections & Paragraphs**: The main body of content.
- **Figures & Captions**: Visual aids and their descriptions.

*Example: The system correctly identifies section headings, the text within them, and the captions associated with figures.*

---

## B) Build Retrieval Indexes

To find the right sentences later, the system builds two powerful retrieval indexes:

1.  **BM25 (Keyword-Based)**: This index is great for finding exact words and phrases, like "pointer network." It’s precise and fast for keyword matches.
2.  **FAISS (Embedding-Based)**: This index understands meaning. It can find related phrases even if they use different words, ensuring conceptual relevance.

Together, they help grab both exact quotes and paraphrased evidence from the paper.

---

## C) Topic Generation (TG)

Using the paper’s title, abstract, and section headings, the system proposes a detailed outline of topics for the slides.

*Example: Instead of a generic heading like “Method,” the model generates more descriptive and expert-friendly titles like “Pointer-Network Transition System” or “Ablation: Label Attention & Decoder Choices.”*

This step uses a model fine-tuned on expert-created presentations to produce long, high-quality outlines.

---

## D) Content Extraction (CE)

For each generated topic, the system performs two key actions:

1.  **Gather Candidates**: It pulls relevant sentences from the paper using the BM25 and FAISS indexes.
2.  **Filter & Rewrite**: It then filters this content and rewrites it into concise slide bullets.

*Example: A raw sentence from the paper becomes a clean bullet point: “Tokens are encoded with a BiLSTM encoder.”*

---

## E) Preference & Reward Data

The system is trained to understand what makes a "good" slide. It learns from pairs of outputs:

-   **“Chosen”**: Outputs that are more detailed, clear, and written in an expert style.
-   **“Rejected”**: Outputs that are too short, vague, or poorly phrased.

This process creates the training signals (preferences and rewards) needed to refine the models.

---

## F) Alignment Pass

All generated bullets undergo an alignment pass to ensure they match a professional, slide-friendly style. This involves rewriting them to be:

-   **Concise**: Short and to the point.
-   **Parallel**: Grammatically consistent.
-   **Clear**: Easy to understand.

> **Before**: “We encode tokens with a BiLSTM.”
> **After**: “BiLSTM encodes token representations.”
>
> The result is more professional and impactful.

---

## G) Factuality / NLI Gating

To ensure accuracy, the system checks if every bullet point is factually correct based on the original paper. This is a Natural Language Inference (NLI) task.

*Example: If a bullet claims “SOTA on all datasets,” but the paper does not support this claim, the system will flag it as a contradiction.*

---

## H) Visual Selection & Layout

The system intelligently selects relevant visuals from the paper and chooses an appropriate layout for each slide.

-   **Figure Selection**: It picks figures that best match the slide's topic.
-   **Layout Choice**: It determines the best structure (e.g., Title + Image + Bullets).

---

## I) Final Export

Finally, the system assembles everything into a structured presentation deck. It organizes titles, bullets, images, and layouts into a coherent flow.

*Example Deck Structure:*

-   **Slide 1**: Motivation
-   **Slide 2**: Model Overview (with architecture figure)
-   **Slide 3**: Main Results (with ablation figure)