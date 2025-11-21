import argparse
import pandas as pd
import os
from vllm import LLM, SamplingParams
from tqdm import tqdm

# prompt
instruction_m2t = """You are an expert chemist. Given the molecule SMILES, your task is to provide the detailed description of the molecule using your experienced chemical knowledge.
Please strictly follow the format, no other information can be provided.
Molecule SMILES: {}
Description:"""

instruction_t2m = """You are an expert chemist. Given the molecular requirement description, your task is to design a new molecule using your experienced chemical knowledge.
Please strictly follow the format, no other information can be provided. You should only reply with SMILES string notations to represent the designed molecule. The SMILES must be valid and chemically reasonable.
Molecular requirement description: {}
Molecule SMILES:"""

SYSTEM_PROMPT = ""

def extract_desc(text):
    return text.strip()

def process_fn(question_raw, answer_raw, split, idx):
    question = instruction_t2m.format(question_raw)
    return {
        "data_source": 't2m',
        "prompt": [
            {
                "role": "user",
                "content": question,
            }
        ],
        "ability": "chem",
        "reward_model": {"style": "rule", "ground_truth": answer_raw},
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": answer_raw,
            "question": question_raw,
        },
    }

def generate_with_vllm_chat(llm, smiles_batch, max_tokens=512):
    # chat prompts
    prompts = [
        [{"role": "user", "content": instruction_m2t.format(smi)}] for smi in smiles_batch
    ]
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        # top_p=1.0, temperature=1.0, do_sample=True
    )
    outputs = llm.chat(prompts, sampling_params)
    return [extract_desc(o.outputs[0].text) for o in outputs]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="vLLM-compatible model path (HF or local)")
    parser.add_argument("--input", required=True, help="Input .parquet with column 'SMILES' or 'extra_info'")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--split", default="train")
    # parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size for vLLM")
    args = parser.parse_args()

    # 加载数据
    df = pd.read_parquet(args.input)
    if "extra_info" in df.columns:
        smiles_list = [entry["answer"] for entry in df["extra_info"]]
    elif "SMILES" in df.columns:
        smiles_list = df["SMILES"].tolist()
    else:
        raise ValueError("Input must have either 'extra_info' or 'SMILES' column")

    llm = LLM(
        model=args.model_dir,
        tokenizer=args.model_dir,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.75,
    )

    results = []
    try:
        batch_descs = generate_with_vllm_chat(llm, smiles_list)
        for i, (smi, desc) in enumerate(zip(smiles_list, batch_descs)):
            result = process_fn(desc, smi, args.split, i)
            results.append(result)
    except Exception as e:
        print(f"Error in: {e}")

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    print(f"✅ Saved {len(out_df)} entries to {args.output}")

if __name__ == "__main__":
    main()
