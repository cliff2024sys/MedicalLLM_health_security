from datasets import load_dataset, Dataset, DatasetDict
from collections import Counter
import statistics

def load_cupcase_dataset(source: str = "huggingface", file_path: str = None):
    """
    Load the CUPCase dataset from the Hugging Face Hub or a local file.
    """
    if source == "huggingface":
        dataset = load_dataset("ofir408/CupCase")
    elif source == "local":
        if file_path is None:
            raise ValueError("file_path must be provided when source='local'.")
        file_path_lower = file_path.lower()
        if file_path_lower.endswith(".json") or file_path_lower.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=file_path)
        elif file_path_lower.endswith(".csv"):
            dataset = load_dataset("csv", data_files=file_path)
        elif file_path_lower.endswith(".parquet"):
            dataset = load_dataset("parquet", data_files=file_path)
        else:
            raise ValueError("Unsupported file format. Please use JSON, CSV, or Parquet.")
    else:
        raise ValueError("Source must be 'huggingface' or 'local'.")
    if isinstance(dataset, DatasetDict):
        # Use the first split (CUPCase has a single split, e.g., 'test')
        dataset = dataset[next(iter(dataset.keys()))]
    return dataset

def preprocess_dataset(dataset: Dataset):
    """
    Ensure the dataset contains required fields and format it for model input.
    """
    required_fields = ["clean_case_presentation", "correct_diagnosis", "distractor1", "distractor2", "distractor3"]
    for field in required_fields:
        if field not in dataset.column_names:
            raise ValueError(f"Required field '{field}' is missing from the dataset.")
    # Ensure all values are strings 
    for field in required_fields:
        dataset = dataset.map(lambda x: {field: str(x[field]) if x[field] is not None else ""})
    return dataset

def analyze_dataset(dataset: Dataset):
    """
    Compute and print summary statistics for the dataset.
    """
    num_cases = len(dataset)
    diagnoses = dataset["correct_diagnosis"]
    num_unique_diagnoses = len(set(diagnoses))
    diagnosis_counts = Counter(diagnoses)
    top_diagnoses = diagnosis_counts.most_common(5)
    case_texts = dataset["clean_case_presentation"]
    lengths = [len(text.split()) for text in case_texts]
    min_len = min(lengths)
    max_len = max(lengths)
    mean_len = statistics.mean(lengths)
    # Print statistics
    print(f"Total number of cases: {num_cases}")
    print(f"Number of unique diagnoses: {num_unique_diagnoses}")
    print("Most common diagnoses (top 5):")
    for diag, count in top_diagnoses:
        print(f"  - {diag} ({count} cases)")
    print(f"Case presentation length (words) - min: {min_len}, max: {max_len}, mean: {mean_len:.1f}")

def validate_dataset_for_models(dataset: Dataset):
    """
    Validate format for BioMistral-7B and Meditron-7B and print a sample entry.
    """
    required_fields = ["clean_case_presentation", "correct_diagnosis", "distractor1", "distractor2", "distractor3"]
    # Check type of first sample's fields 
    sample = dataset[0]
    for field in required_fields:
        if not isinstance(sample[field], str):
            raise TypeError(f"Field '{field}' should be a string, but got {type(sample[field])}.")
    # Print a sample case (truncated for readability)
    print("\nSample case loaded:")
    print("Case presentation (truncated):", sample["clean_case_presentation"][:100] + "...")
    print("Correct diagnosis:", sample["correct_diagnosis"])
    print("Distractors:", sample["distractor1"], "|", sample["distractor2"], "|", sample["distractor3"])

def save_dataset(dataset: Dataset, output_path: str):
    """
    Save the processed dataset to disk for later use.
    """
    dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")

# Execute the data loading, processing, analysis, and saving if run as a script
if __name__ == "__main__":
    # 1. Load the dataset
    dataset = load_cupcase_dataset(source="huggingface")
    # 2. Preprocess the dataset
    dataset = preprocess_dataset(dataset)
    # 3. Analyze the dataset
    analyze_dataset(dataset)
    # 4. Validate compatibility with BioMistral-7B and Meditron-7B
    validate_dataset_for_models(dataset)
    # 5. Save the processed dataset for future use
    save_dataset(dataset, "cupcase_processed")
    
