# Retrieval robustness (test)

k = 10; case bank of 13484 posts; mean ± half-width of the 95% cluster-bootstrap CI (conversations resampled). P@10 / Overlap in %, strategy JSD (base 2, pooled over the queries) ×10⁻². Overlap@10 self: Hinglish post vs its English original, same encoder. Overlap@10 vs. RoBERTa-EN: the post vs the base paper's retriever on the English original. Bold = best encoder per column; † = differs from RoBERTa-L (paired bootstrap CI of the difference excludes 0). RoBERTa-L = all-roberta-large-v1, the English retriever of MultiAgentESC; mMPNet = paraphrase-multilingual-mpnet-base-v2, FT = our contrastive (MNRL) fine-tuning on Hinglish-English pairs.

## All test turns (EN n=1210, Light n=1210, Heavy n=1210)

| Encoder | P@10 EN | P@10 Light | P@10 Heavy | Overlap@10 self Light | Overlap@10 self Heavy | Overlap@10 vs. RoBERTa-EN EN | Overlap@10 vs. RoBERTa-EN Light | Overlap@10 vs. RoBERTa-EN Heavy | Strategy JSD Light | Strategy JSD Heavy |
|---|---|---|---|---|---|---|---|---|---|---|
| RoBERTa-L (base) | **36.0** ± 2.4 | **35.8** ± 2.3 | **33.8** ± 2.4 | 56.9 ± 1.9 | 27.0 ± 2.3 | 100.0 | **56.9** ± 1.9 | **27.0** ± 2.3 | 0.04 ± 0.03 | 0.15 ± 0.07 |
| LaBSE | 29.2 ± 1.5 † | 28.9 ± 1.5 † | 28.5 ± 1.3 † | 61.4 ± 2.0 † | 33.2 ± 2.2 † | 18.8 ± 1.3 | 16.9 ± 1.2 † | 12.7 ± 1.2 † | 0.02 ± 0.02 | 0.07 ± 0.05 |
| mMPNet | 32.8 ± 2.1 † | 31.6 ± 2.0 † | 28.4 ± 1.6 † | 69.9 ± 1.6 † | 31.3 ± 2.3 † | **27.5** ± 1.3 | 25.3 ± 1.3 † | 15.6 ± 1.4 † | 0.05 ± 0.03 | 0.86 ± 0.21 † |
| mMPNet-FT (ours) | 32.1 ± 2.1 † | 32.0 ± 2.0 † | 31.5 ± 1.9 † | **73.2** ± 1.3 † | **53.3** ± 1.8 † | 23.9 ± 1.4 | 23.1 ± 1.4 † | 19.4 ± 1.3 † | **0.01** ± 0.01 † | **0.01** ± 0.02 † |
| Random (chance) | 20.2 ± 0.9 | 20.2 ± 0.9 | 20.2 ± 0.9 | -- | -- | -- | -- | -- | -- | -- |

Difference to RoBERTa-L, the base paper's retriever (paired; [95% CI]):

| Encoder | P@10 EN | P@10 Light | P@10 Heavy | Overlap@10 self Light | Overlap@10 self Heavy | Overlap@10 vs. RoBERTa-EN EN | Overlap@10 vs. RoBERTa-EN Light | Overlap@10 vs. RoBERTa-EN Heavy | Strategy JSD Light | Strategy JSD Heavy |
|---|---|---|---|---|---|---|---|---|---|---|
| LaBSE | -6.8 [-8.5, -5.2] | -6.9 [-8.4, -5.4] | -5.3 [-7.0, -3.7] | +4.5 [+2.5, +6.5] | +6.3 [+4.5, +8.0] | -81.2 [-82.5, -79.9] | -40.1 [-41.8, -38.2] | -14.3 [-16.2, -12.5] | -0.01 [-0.05, +0.02] | -0.08 [-0.18, +0.00] |
| mMPNet | -3.2 [-4.5, -2.1] | -4.1 [-5.4, -2.9] | -5.3 [-7.0, -3.7] | +13.0 [+11.6, +14.5] | +4.3 [+3.0, +5.7] | -72.5 [-73.8, -71.2] | -31.7 [-33.3, -29.9] | -11.4 [-12.8, -9.8] | +0.01 [-0.03, +0.05] | +0.71 [+0.50, +0.95] |
| mMPNet-FT (ours) | -3.9 [-5.2, -2.6] | -3.7 [-5.0, -2.5] | -2.3 [-3.8, -0.9] | +16.3 [+14.8, +18.0] | +26.3 [+24.6, +28.2] | -76.1 [-77.5, -74.7] | -33.9 [-35.6, -32.0] | -7.6 [-9.3, -5.8] | -0.03 [-0.07, -0.01] | -0.13 [-0.22, -0.07] |

## sampled_uids(200) (EN n=200, Light n=200, Heavy n=200)

| Encoder | P@10 EN | P@10 Light | P@10 Heavy | Overlap@10 self Light | Overlap@10 self Heavy | Overlap@10 vs. RoBERTa-EN EN | Overlap@10 vs. RoBERTa-EN Light | Overlap@10 vs. RoBERTa-EN Heavy | Strategy JSD Light | Strategy JSD Heavy |
|---|---|---|---|---|---|---|---|---|---|---|
| RoBERTa-L (base) | **31.4** ± 3.7 | **32.5** ± 3.8 | **31.8** ± 3.8 | 52.6 ± 4.0 | 28.9 ± 5.1 | 100.0 | **52.6** ± 4.0 | **28.9** ± 5.1 | 0.12 ± 0.17 | 0.14 ± 0.19 |
| LaBSE | 28.2 ± 3.2 † | 27.3 ± 3.1 † | 29.2 ± 2.9 | 58.2 ± 3.9 † | 32.8 ± 4.7 † | 17.2 ± 3.1 | 14.4 ± 2.7 † | 11.2 ± 3.0 † | 0.16 ± 0.15 | 0.27 ± 0.27 |
| mMPNet | 30.6 ± 3.4 | 30.0 ± 3.2 | 27.6 ± 3.0 † | 68.2 ± 3.7 † | 31.4 ± 5.7 | **27.2** ± 3.2 | 24.4 ± 3.2 † | 15.3 ± 3.4 † | 0.12 ± 0.12 | 1.16 ± 0.56 † |
| mMPNet-FT (ours) | 31.1 ± 3.6 | 30.9 ± 3.4 | 31.1 ± 3.5 | **73.1** ± 3.1 † | **53.3** ± 4.0 † | 23.9 ± 3.4 | 22.7 ± 3.3 † | 19.2 ± 3.4 † | **0.04** ± 0.06 | **0.03** ± 0.10 |
| Random (chance) | 20.0 ± 1.1 | 20.0 ± 1.1 | 20.0 ± 1.1 | -- | -- | -- | -- | -- | -- | -- |

Difference to RoBERTa-L, the base paper's retriever (paired; [95% CI]):

| Encoder | P@10 EN | P@10 Light | P@10 Heavy | Overlap@10 self Light | Overlap@10 self Heavy | Overlap@10 vs. RoBERTa-EN EN | Overlap@10 vs. RoBERTa-EN Light | Overlap@10 vs. RoBERTa-EN Heavy | Strategy JSD Light | Strategy JSD Heavy |
|---|---|---|---|---|---|---|---|---|---|---|
| LaBSE | -3.2 [-6.7, -0.1] | -5.2 [-8.5, -1.9] | -2.6 [-5.9, +0.8] | +5.6 [+1.8, +9.4] | +3.9 [+0.9, +6.9] | -82.9 [-85.8, -79.6] | -38.3 [-41.9, -34.5] | -17.7 [-21.2, -14.5] | +0.04 [-0.17, +0.23] | +0.12 [-0.19, +0.47] |
| mMPNet | -0.8 [-3.7, +1.9] | -2.5 [-5.7, +0.9] | -4.2 [-7.9, -0.4] | +15.5 [+11.4, +19.8] | +2.5 [-0.7, +5.8] | -72.8 [-76.1, -69.6] | -28.2 [-32.0, -24.3] | -13.6 [-16.9, -10.5] | -0.00 [-0.27, +0.19] | +1.02 [+0.48, +1.70] |
| mMPNet-FT (ours) | -0.3 [-3.8, +3.2] | -1.7 [-5.0, +1.9] | -0.6 [-4.5, +3.2] | +20.4 [+16.4, +24.8] | +24.3 [+20.6, +27.9] | -76.1 [-79.3, -72.5] | -30.0 [-34.1, -25.9] | -9.7 [-13.4, -6.0] | -0.09 [-0.33, +0.03] | -0.11 [-0.37, +0.06] |
