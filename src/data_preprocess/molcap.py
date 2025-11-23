import argparse
import os
import re
from verl.utils.hdfs_io import copy, makedirs
from rdkit import Chem
from datasets import load_dataset, Dataset, DatasetDict # <-- Import DatasetDict
import pandas as pd

SYSTEM_PROMPT = """Given the SMILES above. Generate a short and accurate description in natural language. 
The description should be enough for other chemists to restore the molecule without seeing the SMILES code.
Please make sure to exclude anything related to SMILES in the description before you give your final answer.
Put the description between <desc> and </desc>.
"""

INSTRUCTION="""You are an expert chemist. Given the molecule SMILES, your task is to provide the detailed description of the molecule using your experienced chemical knowledge.
Please strictly follow the format, no other information can be provided.
Molecule SMILES: {}
Description:
"""




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
            # description = str(example['description']) if pd.notna(example['description']) else ""
            smiles = str(example['SMILES']) if pd.notna(example['SMILES']) else ""

            return {
                'question': smiles,
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
    parser.add_argument("--local_dir", default="./data/molcap")
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

            # question = question_raw + " " + SYSTEM_PROMPT
            question = INSTRUCTION.format(question_raw)

            answer_raw = example.pop("answer")
            solution = answer_raw
            data = {
                "data_source": 'molcap',
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