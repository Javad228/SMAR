# **Training**

### **LoRA**

I have slides with corresponding research papers (pdf).
I will parse both to align slide bullets and titles back to paper sentences.
My training will be:

* **Compression task**:
  `"paper sentence (or short span)" | "section name" → "slide bullet (≤15 words)"`

* **Title task**:
  `"3–5 final bullets + section name" → "action-style slide title (≤12 words, verb required)"`

This will be used as training for the **LoRA adapter**.

---

### **BERT reranker**

**Pass 1 (human anchors):**
`"all sentences in a section" | "section name" → labels: which sentences were actually used in the slides (positives vs hard negatives)`

**Pass 2 (LLM distillation):**
`"remaining sentences in a section" | "section name" → sentences ranked best to worst by GPT-5 (listwise soft labels)`

This will be used as training for the **cross-encoder BERT reranker**.

---

# **Running**

### **1st pass**

1. Parse PDF → sections/sentences.
2. **Retrieve** candidates per section (BM25 + vectors) (remove a few bad ones, keep \~100).
3. **Rerank** with the **zone/section query** (not a title) using my finetuned BERT reranker.
4. Select top \~10–12 sentences → **compress into 3–5 bullets** with LoRA (GPT-OSS-20B).
5. **Generate the title** from those bullets with LoRA (GPT-OSS-20B).

---

### **2nd pass**

1. Parse PDF → sections/sentences.
2. **Retrieve** candidates per section (BM25 + vectors).
3. **Rerank** with the **title from previous step 5** using my finetuned BERT reranker.
4. Select top \~10–12 sentences → **compress into 3–5 bullets** with LoRA (GPT-OSS-20B).
5. **Generate the final title** from those bullets with LoRA (GPT-OSS-20B).

