BM25 baseline (SciFact, K=10, ~150 queries)

nDCG@K ≈ 0.518

MRR ≈ 0.471

Hit@K ≈ 0.667

FAISS + Expansion (OpenAI embeddings, τ, top_n vary)

top_n=5, τ=0.5–0.6

nDCG ≈ 0.702–0.727

MRR ≈ 0.677–0.704

Hit ≈ 0.773–0.800

top_n=8, τ=0.5–0.6

nDCG ≈ 0.716–0.745

MRR ≈ 0.685–0.715

Hit ≈ 0.813–0.840

top_n=12, τ=0.5–0.6

nDCG ≈ 0.721–0.758

MRR ≈ 0.689–0.722

Hit ≈ 0.833–0.873

Fusion (Weighted)

Weights (BM25=0.35, Sem=0.65), top_n=12, τ=0.6

nDCG = 0.746

MRR = 0.714

Hit = 0.860

Weights (0.3 / 0.7), BM25 agg=softmax (temp=12), top_n=12, τ=0.55–0.6

nDCG = 0.700

MRR = 0.675

Hit = 0.780

Fusion (RRF)

rrf_k=60, top_n=8–12, τ=0.55–0.6

nDCG ≈ 0.704–0.708

MRR ≈ 0.649–0.656

Hit ≈ 0.853–0.860

Fusion (RRF → Weighted post-fusion)

rrf_k=30, depth=200, weights (0.3/0.7), top_n=12, τ=0.6

nDCG = 0.713

MRR = 0.669

Hit = 0.853