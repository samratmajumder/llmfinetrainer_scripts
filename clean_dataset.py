#!/usr/bin/env python3
# Script to validate and clean a raw text dataset for Unsloth fine-tuning
import json
import argparse
import os
from tqdm import tqdm

def clean_text(text):
    """Clean text by removing problematic characters and ensuring proper string format."""
    if text is None:
        return ""
        
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)
        
    # Basic cleaning
    text = text.strip()
    
    # Remove any control characters
    text = ''.join(ch for ch in text if ord(ch) >= 32 or ch == '\n')
    
    return text

def process_jsonl_file(input_file, output_file, dry_run=False):
    """Process a JSONL file to clean and validate each entry."""
    print(f"Processing {input_file}...")
    
    valid_entries = 0
    invalid_entries = 0
    processed_entries = []
    
    # Read and process each line
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        for line in tqdm(lines, desc="Processing entries"):
            try:
                # Parse the JSON entry
                entry = json.loads(line.strip())
                
                # For raw text format, we expect a "text" field
                if "text" in entry:
                    # Clean the text
                    cleaned_text = clean_text(entry["text"])
                    
                    # If text is too short after cleaning, skip it
                    if len(cleaned_text) < 10:  # Arbitrary minimum length
                        print(f"Warning: Entry too short after cleaning: '{cleaned_text}'")
                        invalid_entries += 1
                        continue
                        
                    # Create a new clean entry
                    clean_entry = {"text": cleaned_text}
                    processed_entries.append(clean_entry)
                    valid_entries += 1
                else:
                    print(f"Warning: Entry missing 'text' field: {entry}")
                    invalid_entries += 1
            
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON line: {line.strip()[:80]}...")
                invalid_entries += 1
                continue
    
    # Write the processed entries to the output file if not a dry run
    if not dry_run and processed_entries:
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in processed_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"Cleaned dataset saved to {output_file}")
    
    print(f"Processing complete: {valid_entries} valid entries, {invalid_entries} invalid entries")
    return valid_entries, invalid_entries

def main():
    parser = argparse.ArgumentParser(description="Clean and validate a raw text JSONL dataset")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=str, help="Output cleaned JSONL file path (default: input_cleaned.jsonl)")
    parser.add_argument("--dry-run", action="store_true", help="Only validate without creating output file")
    
    args = parser.parse_args()
    
    input_file = args.input
    
    if not os.path.isfile(input_file):
        print(f"Error: Input file {input_file} does not exist")
        return 1
    
    if args.output:
        output_file = args.output
    else:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_cleaned{ext}"
    
    process_jsonl_file(input_file, output_file, args.dry_run)
    return 0

if __name__ == "__main__":
    main()
