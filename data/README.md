# Data

## `vetai_dataset_100.jsonl` (not tracked)

The domain-adaptation stage uses a small curated set of ~100 veterinary
dialogues. The file itself is not committed; only the schema and a three-line
sample (`vetai_dataset.sample.jsonl`) are, so the format is reproducible without
redistributing the dataset.

**Format:** JSON Lines — one conversation per line, in the Hugging Face chat
schema that `tokenizer.apply_chat_template` consumes directly.

```json
{"messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

| Field | Type | Notes |
|---|---|---|
| `messages` | list of objects | One full conversation |
| `messages[].role` | `system` \| `user` \| `assistant` | `system` is optional |
| `messages[].content` | string | Plain text |

**Content guidelines used when writing the set:**

- Topics: routine care, nutrition, behaviour, when to see a vet.
- Assistant answers stay supportive and non-diagnostic, and defer to a
  veterinarian for anything that could be a medical decision.
- Answers are short (2–5 sentences) so the 44-step refinement pass shapes tone
  rather than length.

Place your own file at the path configured by `VETAI_DATASET_PATH` in the
notebook (by default `<OUTPUT_DIR>/vetai_dataset_100.jsonl`).

## `mlabonne/FineTome-100k`

The instruction-tuning stage streams
[FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) straight
from the Hub (`split="train[:10000]"`); nothing is stored in the repository.
