Light runs on an earlier build of ESConv-HiEn Light (98 conversations; a JSON-parsing bug in
`scripts/build_hien.py` silently dropped the CMI fix rounds of many conversations, 80% of the
checked utterances in band). The final test set (`data/esconv_hien/test_light.json`, 100
conversations, 94.5% in band) changed 23% of the utterances, so every Light run was repeated on it
in `results/runs/`. Kept for transparency only; not used in any table.
