# Dataset Manifest & Provenance

| Dataset Name | Origin / URL | License | Intended Stage | Tokens / Size | Processing & Deduplication |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tiny_smoke_corpus` | Seed algorithmic implementations & technical prose | MIT / CC0 | Stage A & B Verification | 82,640 tokens (~325 KB) | Exact SHA-256 deduplicated, normalized |
| `sft_data` | Curated instruction-response QA pairs | MIT / Apache-2.0 | Stage C Instruction Tuning | 125 pairs (~28 KB) | Filtered, prompt-masked |

## Policy on Data Ingestion
1. Pretraining data is cleanly separated from validation sets.
2. SFT dataset applies loss-masking (`-100`) on user prompts so that gradients are computed exclusively on target response tokens.
3. No unlicensed or scraped web data is included.
