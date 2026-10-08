# Retrieval robustness (dev)

k = 10; case bank of 13334 posts (dev conversations excluded); mean ± half-width of the 95% cluster-bootstrap CI (conversations resampled). P@10 / Overlap in %, strategy JSD (base 2, pooled over the queries) ×10⁻². Overlap@10 self: Hinglish post vs its English original, same encoder. Overlap@10 vs. RoBERTa-EN: the post vs the base paper's retriever on the English original. Bold = best encoder per column; † = differs from RoBERTa-L (paired bootstrap CI of the difference excludes 0). RoBERTa-L = all-roberta-large-v1, the English retriever of MultiAgentESC; mMPNet = paraphrase-multilingual-mpnet-base-v2, FT = our contrastive (MNRL) fine-tuning on Hinglish-English pairs.

## All dev turns (EN n=151, Light n=151, Heavy n=151)

| Encoder | P@10 EN | P@10 Light | P@10 Heavy | Overlap@10 self Light | Overlap@10 self Heavy | Overlap@10 vs. RoBERTa-EN EN | Overlap@10 vs. RoBERTa-EN Light | Overlap@10 vs. RoBERTa-EN Heavy | Strategy JSD Light | Strategy JSD Heavy |
|---|---|---|---|---|---|---|---|---|---|---|
| RoBERTa-L (base) | **31.3** ± 8.4 | **30.7** ± 8.6 | **29.6** ± 7.6 | 56.0 ± 7.2 | 22.1 ± 7.3 | 100.0 | **56.0** ± 7.2 | **22.1** ± 7.3 | 0.14 ± 0.22 | 0.37 ± 0.38 |
| LaBSE | 24.3 ± 5.0 † | 25.6 ± 5.7 † | 24.9 ± 5.4 † | 59.7 ± 6.7 | 32.4 ± 6.8 † | 16.1 ± 2.7 | 15.0 ± 2.2 † | 10.9 ± 2.7 † | **0.02** ± 0.10 | 0.14 ± 0.22 |
| mMPNet | 30.3 ± 8.2 | 29.1 ± 7.7 | 26.7 ± 6.9 | 69.9 ± 7.6 † | 29.6 ± 7.6 † | **26.0** ± 3.0 | 22.5 ± 2.9 † | 14.4 ± 4.2 † | 0.06 ± 0.14 | 1.25 ± 0.62 † |
| mMPNet-FT (ours) | 27.6 ± 7.0 † | 26.7 ± 6.6 † | 28.5 ± 6.6 | **72.9** ± 5.0 † | **54.0** ± 4.4 † | 21.9 ± 4.1 | 20.9 ± 3.5 † | 17.7 ± 4.0 | 0.04 ± 0.08 | **0.03** ± 0.17 † |
| Random (chance) | 18.8 ± 3.6 | 18.8 ± 3.6 | 18.8 ± 3.6 | -- | -- | -- | -- | -- | -- | -- |

Difference to RoBERTa-L, the base paper's retriever (paired; [95% CI]):

| Encoder | P@10 EN | P@10 Light | P@10 Heavy | Overlap@10 self Light | Overlap@10 self Heavy | Overlap@10 vs. RoBERTa-EN EN | Overlap@10 vs. RoBERTa-EN Light | Overlap@10 vs. RoBERTa-EN Heavy | Strategy JSD Light | Strategy JSD Heavy |
|---|---|---|---|---|---|---|---|---|---|---|
| LaBSE | -7.0 [-12.9, -3.0] | -5.1 [-9.9, -1.6] | -4.7 [-8.9, -0.6] | +3.8 [-3.5, +10.3] | +10.3 [+6.3, +14.4] | -83.9 [-86.3, -80.8] | -40.9 [-46.3, -34.7] | -11.1 [-17.0, -6.4] | -0.11 [-0.42, +0.07] | -0.23 [-0.73, +0.04] |
| mMPNet | -1.0 [-4.0, +1.7] | -1.6 [-4.7, +0.9] | -2.9 [-6.7, +0.6] | +13.9 [+10.0, +17.7] | +7.5 [+4.6, +10.3] | -74.0 [-76.9, -70.9] | -33.5 [-38.6, -27.6] | -7.6 [-12.0, -3.8] | -0.08 [-0.36, +0.09] | +0.88 [+0.01, +1.59] |
| mMPNet-FT (ours) | -3.7 [-7.3, -0.7] | -4.0 [-7.8, -0.7] | -1.1 [-3.4, +1.2] | +17.0 [+13.7, +20.3] | +31.9 [+26.0, +37.4] | -78.1 [-82.1, -73.8] | -35.0 [-40.8, -28.4] | -4.3 [-9.1, +0.2] | -0.10 [-0.38, +0.01] | -0.34 [-0.78, -0.13] |
