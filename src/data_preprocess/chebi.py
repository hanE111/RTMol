import argparse
import os
import re
from verl.utils.hdfs_io import copy, makedirs
from rdkit import Chem
from datasets import load_dataset, Dataset, DatasetDict # <-- Import DatasetDict
import pandas as pd

SYSTEM_PROMPT = """Given the description of the molecule above, infer the SMILES of this molecule. Put all your thinking between <reasoning> and </reasoning>. Put only the SMILES between <answer> and </answer>.
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

INSTRUCTION = """You are an expert chemist. Given the molecular requirement description, your task is to design a new molecule using your experienced chemical knowledge.\nPlease strictly follow the format, no other information can be provided. You should only reply with SMILES string notations to represent the designed molecule. The SMILES must be valid and chemically reasonable.
Molecular requirement description: {}
Molecule SMILES:"""

def load_and_format_smiles_data(file_path: str) -> Dataset:
    """Loads data from a TSV file and formats it for GRPOTrainer."""
    try:
        # Load data using pandas, specifying tab separator
        df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip') # Skip bad lines if any
        # Ensure required columns exist
        if not {'SMILES', 'description'}.issubset(df.columns):
             raise ValueError(f"File {file_path} must contain 'SMILES' and 'description' columns.")

        # Convert to Hugging Face Dataset
        # df =df[df['SMILES'].str.len()<15]
        hf_dataset = Dataset.from_pandas(df)

        # Map to the required format
        def format_example(example):
            # Ensure description and SMILES are strings
            description = str(example['description']) if pd.notna(example['description']) else ""
            smiles = str(example['SMILES']) if pd.notna(example['SMILES']) else ""

            return {
                'question': description,
                'answer': smiles # Ground truth SMILES
            }

        formatted_dataset = hf_dataset.map(
            format_example,
            remove_columns=list(df.columns) # Remove original columns
        )
        # Filter out examples where the ground truth SMILES is invalid or empty, as they can't be used for training/evaluation
        formatted_dataset = formatted_dataset.filter(lambda x: x['answer'] and Chem.MolFromSmiles(x['answer']) is not None)
        return formatted_dataset

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error loading or formatting data from {file_path}: {e}")
        raise
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/chebi")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chenletian-240108120062/rl_project/"
    train_dataset = load_and_format_smiles_data(os.path.join(data_source_path,'train.txt'))
    valid_dataset = load_and_format_smiles_data(os.path.join(data_source_path,'validation.txt'))
    test_dataset = load_and_format_smiles_data(os.path.join(data_source_path,'test.txt')) # Load test set if needed later

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + SYSTEM_PROMPT
            question = INSTRUCTION.format(question_raw)

            answer_raw = example.pop("answer")
            solution = answer_raw
            data = {
                "data_source": 'chebi',
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "chem",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)