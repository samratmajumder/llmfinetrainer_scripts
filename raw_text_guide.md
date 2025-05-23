# Raw Text Training Example

This is an example of how to use the new raw text mode for training with Unsloth.

## Process your stories into raw text format

```bash
python story_processor.py --folder path/to/your/stories --output raw_text_dataset.jsonl --format raw_text
```

This will split your stories into paragraphs or smaller text blocks and create a JSONL file with entries like:

```json
{"text": "Paragraph 1 content goes here. The text can be longer but is typically sized to stay within token limits."}
{"text": "Paragraph 2 content goes here. Each paragraph becomes its own training example."}
```

## Fine-tune using raw text format

```bash
python unsloth_tuner.py --dataset raw_text_dataset.jsonl --format raw_text
```

This bypasses the prompt/completion formatting and uses the raw text directly for continued pre-training.
