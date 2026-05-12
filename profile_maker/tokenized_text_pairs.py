import json
import copy
from transformers import AutoTokenizer, pipeline
from datasets import Dataset
import os
import torch
from pathlib import Path

MODEL_PATH = Path("")
JSONL_FILE = Path("")
SAVE_PATH = Path("")


_translator_instance = None

def get_translator():
    global _translator_instance
    
    if not ENABLE_TRANSLATION:
        return None
    
    if _translator_instance is not None:
        return _translator_instance

    try:
        device_id = 1
        
        _translator_instance = pipeline(
            "translation",
            model=TRANSLATION_MODEL,
            device=device_id,
            max_length=1024
        )
        return _translator_instance
    except Exception as e:
        print(f"⚠️ translation model loading failed: {e}")
        _translator_instance = None
        return None

def process_pipeline():
    if ENABLE_TRANSLATION:
        print(f"Testing translation model: {TRANSLATION_MODEL}...")
        test_translator = get_translator()
        if test_translator is not None:
            print("✅ Translation model test loaded successfully (will be loaded on demand in subprocesses).")
        else:
            print("⚠️ Failed to load translation model, skipping translation step.")
    print(f"Loading Tokenizer: {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id
    model_max_length = getattr(tokenizer, 'model_max_length', 1024)
    max_token_length = model_max_length - 1
    print(f"Model max sequence length: {model_max_length}, tokenization max length: {max_token_length} (reserve space for EOS token).")
    
    def translate_text(text, translator_model):
        if translator_model is None:
            return text
        
        if len(text) <= 500:
            try:
                result = translator_model(text, max_length=512)
                return result[0]['translation_text']
            except Exception as e:
                print(f"Translation failed: {e}, returning original text.")
                return text
        
        paragraphs = text.split('\n\n')
        translated_paragraphs = []
        
        for para in paragraphs:
            if not para.strip():
                translated_paragraphs.append(para)
                continue
            if len(para) > 500:
                sentences = para.split('\n')
                translated_sentences = []
                for sent in sentences:
                    if sent.strip():
                        try:
                            result = translator_model(sent, max_length=512)
                            translated_sentences.append(result[0]['translation_text'])
                        except Exception as e:
                            print(f"Sentence translation failed: {e}, using original sentence.")
                            translated_sentences.append(sent)
                    else:
                        translated_sentences.append(sent)
                translated_paragraphs.append('\n'.join(translated_sentences))
            else:
                try:
                    result = translator_model(para, max_length=512)
                    translated_paragraphs.append(result[0]['translation_text'])
                except Exception as e:
                    print(f"Paragraph translation failed: {e}, using original paragraph.")
                    translated_paragraphs.append(para)
        
        return '\n\n'.join(translated_paragraphs)
    
    def format_report_to_text(report_dict):
        text_parts = []
        
        for key, value in report_dict.items():
            if isinstance(value, list):
                value_str = "\n".join([f"- {item}" for item in value])
                text_parts.append(f"[{key}]:\n{value_str}")
            else:
                text_parts.append(f"[{key}]: {str(value)}")
        
        full_text = "\n\n".join(text_parts)
        
        return full_text

    print("Loading JSONL dataset...")
    data_list = []
    skipped_count = 0
    data_norm = {
        "processed": True,
        "dataset_name": "",
        "sample_name": "",
        "report": "",
        "group_path": "",
        "slice_files": []
    }
    
    with open(JSONL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data.get("processed", True) is False or data.get("report") is None:
                    skipped_count += 1
                    continue
                
                normalized_data = copy.deepcopy(data_norm)
                for key in data_norm.keys():
                    if key in data:
                        value = data[key]
                        if key == "report" and isinstance(value, dict):
                            normalized_data[key] = format_report_to_text(value)
                        else:
                            normalized_data[key] = value
                
                data_list.append(normalized_data)

    dataset = Dataset.from_list(data_list)
    print(f"Loaded {len(dataset)} records, skipped {skipped_count} records with processed=False.")

    def process_function(example):
        formatted_text = example["report"]
        
        if ENABLE_TRANSLATION:
            translator = get_translator()
            if translator is not None:
                formatted_text = translate_text(formatted_text, translator)
        
        input_ids = tokenizer(
            formatted_text, 
            add_special_tokens=False,
            max_length=max_token_length,
            truncation=True
        )["input_ids"]
        input_ids.append(eos_token_id)
        
        return {
            "text_input_ids": input_ids,
            "text_length": len(input_ids),
            "eeg_files": example["slice_files"],
            "dataset_name": example["dataset_name"],
            "sample_name": example["sample_name"]
        }

    print("Start transforming and encoding...")
    if ENABLE_TRANSLATION:
        print("⚠️ Note: translation is enabled, processing will be slower.")
        processed_dataset = dataset.map(
            process_function,
            batched=False,
            num_proc=None,
            remove_columns=dataset.column_names,
        )
    else:
        processed_dataset = dataset.map(
            process_function,
            batched=False,
            num_proc=8,
            remove_columns=dataset.column_names,
        )

    print(f"Saving processed dataset to: {SAVE_PATH}")
    processed_dataset.save_to_disk(SAVE_PATH)
    
    print("\n=== Data inspection ===")
    print(f"Total samples: {len(processed_dataset)}")
    
    truncated_count = sum(1 for item in processed_dataset if item['text_length'] >= model_max_length)
    if truncated_count > 0:
        print(f"⚠️ Warning: {truncated_count} samples were truncated (reached max length {model_max_length}).")
    else:
        print(f"✅ All samples are within the max length limit ({model_max_length}).")
    
    lengths = [item['text_length'] for item in processed_dataset]
    print(f"Text length stats: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}")
    
    sample = processed_dataset[0]
    print(f"\nEEG files of first sample: {sample['eeg_files']}")
    print(f"Text length of first sample: {sample['text_length']}")
    print(f"Dataset name of first sample: {sample['dataset_name']}")
    print(f"Sample name of first sample: {sample['sample_name']}")
    print(f"Decode first sample text:\n{tokenizer.decode(sample['text_input_ids'][:])} ...")

if __name__ == "__main__":
    process_pipeline()