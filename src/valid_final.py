from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import Levenshtein
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
try:
    import nltk
    nltk.data.find("corpora/wordnet")
except LookupError:
    print("Downloading NLTK data...")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

def standardize_smiles(smiles):
    """Standardizes SMILES strings."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ''
        standardized_smiles = Chem.MolToSmiles(mol, canonical=True)
        return standardized_smiles
    except Exception as e:
        print(f"Error standardizing SMILES: {e}")
        return ''
    
def evaluate_text2mol(label: str, pred: str):
    # Exact Match
    exact = int(label.strip() == pred.strip())

    # BLEU (Default: BLEU-4 with smoothing)
    smoothie = SmoothingFunction().method4
    reference = [list(label.strip())]
    hypothesis = list(pred.strip())
    bleu = sentence_bleu(reference, hypothesis, smoothing_function=smoothie)

    # Levenshtein Distance
    dis = Levenshtein.distance(label, pred)

    # METEOR
    meteor = meteor_score(reference, hypothesis)

    valid = int(standardize_smiles(pred) != '')
    if not valid:
        return {
        'original_smiles': label,
        'restored_smiles': pred,
        'Exact': exact,
        'BLEU': bleu,
        'Dis': dis,
        'valid': False,
        'exact_match': False,
        'MACCS': 0,
        'RDK': 0,
        'Morgan': 0,
        'METEOR': 0,
        }
    else:
        exact_match = (standardize_smiles(label) == standardize_smiles(pred))
        
        mol1 = Chem.MolFromSmiles(label)
        mol2 = Chem.MolFromSmiles(pred)
        
        if mol1 is None or mol2 is None:
            maccs, rdk, morgan = 0, 0, 0
        else:
            fp1 = AllChem.GetMACCSKeysFingerprint(mol1)
            fp2 = AllChem.GetMACCSKeysFingerprint(mol2)
            maccs = DataStructs.TanimotoSimilarity(fp1, fp2)

            fp1 = AllChem.RDKFingerprint(mol1)
            fp2 = AllChem.RDKFingerprint(mol2)
            rdk = DataStructs.TanimotoSimilarity(fp1, fp2)

            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 3, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)
            morgan = DataStructs.TanimotoSimilarity(fp1, fp2)
            
        
        return {
        'original_smiles': label,
        'restored_smiles': pred,
        'Exact': exact,
        'BLEU': bleu,
        'Dis': dis,
        'valid': True,
        'exact_match': exact_match,
        'MACCS': maccs,
        'RDK': rdk,
        'Morgan': morgan,
        'METEOR': meteor,
        }

def evaluate_mol2text(label: str, pred: str):
    smoothie = SmoothingFunction().method4
    reference = [label.split()]
    hypothesis = pred.split()

    # BLEU-2 and BLEU-4
    bleu2 = sentence_bleu(reference, hypothesis, weights=(0.5, 0.5), smoothing_function=smoothie)
    bleu4 = sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    # ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(label, pred)
    rouge_1 = rouge_scores['rouge1'].fmeasure
    rouge_2 = rouge_scores['rouge2'].fmeasure
    rouge_L = rouge_scores['rougeL'].fmeasure

    # METEOR
    meteor = meteor_score(reference, hypothesis)

    return {
        'BLEU-2': bleu2,
        'BLEU-4': bleu4,
        'ROUGE-1': rouge_1,
        'ROUGE-2': rouge_2,
        'ROUGE-L': rouge_L,
        'METEOR': meteor,
    }


import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
from vllm import LLM, SamplingParams
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output_file", type=str, required=True, help="Path to save Excel file")
args = parser.parse_args()

PROMPT = """You are an expert chemist. Given the molecular requirement description, your task is to design a new molecule using your experienced chemical knowledge.
Please strictly follow the format, no other information can be provided. You should only reply with SMILES string notations to represent the designed molecule. The SMILES must be valid and chemically reasonable.
Molecular requirement description: {}
Molecule SMILES:"""

# === 加载数据 ===
df = pd.read_parquet("/HOME/paratera_xy/pxy547/HDD_POOL/for_verl/verl/data/molcap_chemdfm/test_100.parquet")
df_data = pd.read_parquet("/HOME/paratera_xy/pxy547/HDD_POOL/for_verl/verl/data/chebi_chemdfm/test_100.parquet")
prompts_data = [one[0]["content"] for one in df_data["prompt"]]

questions = [one[0]["content"] for one in df["prompt"]]
ground_truths = [one["ground_truth"] for one in df["reward_model"]]

# === 初始化 vLLM 模型 ===
llm = LLM(model="/XYFS01/HDD_POOL/paratera_xy/pxy547/checkpoints/verl_grpo_chemdfm/chemdfm_hf", 
          trust_remote_code=True,tensor_parallel_size=8,gpu_memory_utilization=0.5) 
sampling_params = SamplingParams(max_tokens=512, temperature=0.)

# === m2t ===
print(">>> mol → text")
outputs1 = llm.generate(questions, sampling_params)
generated_descriptions = [request_output.outputs[0].text for request_output in outputs1]

# === t2m ===
prompts = [PROMPT.format(d) for d in generated_descriptions]
print(">>> text → mol")
outputs2 = llm.generate(prompts, sampling_params)
generated_smiles = [request_output.outputs[0].text for request_output in outputs2]
# using dataset descriptions
outputs3 = llm.generate(prompts_data, sampling_params)
generated_smiles_data = [request_output.outputs[0].text for request_output in outputs3]

# === reward ===
print(">>> Computing rewards...")

# Generate descriptions & evaluate
descriptions_label = [one["ground_truth"] for one in df_data["reward_model"]]
mol2text_metrics = []
for label, pred in zip(descriptions_label, generated_descriptions):
    mol2text_metrics.append(evaluate_mol2text(label, pred))

# Text2mol using ground truth & evaluate
text2mol_metrics = []
for label, pred in zip(ground_truths, generated_smiles_data):
    text2mol_metrics.append(evaluate_text2mol(label, pred))

# Restore SMILES from descriptions & evaluate
round_trip_metrics = []
for label, pred in zip(ground_truths, generated_smiles):
    round_trip_metrics.append(evaluate_text2mol(label, pred))

# Save results to xlsx
df1 = pd.DataFrame(mol2text_metrics)
df1['description_label'] = descriptions_label
df1['description_pred'] = generated_descriptions
df2 = pd.DataFrame(text2mol_metrics)
df3 = pd.DataFrame(round_trip_metrics)

with pd.ExcelWriter(args.output_file) as writer:
    df1.to_excel(writer, sheet_name="mol2text", index=False)
    df2.to_excel(writer, sheet_name="text2mol", index=False)
    df3.to_excel(writer, sheet_name="round-trip-processed", index=False)
print("\nEvaluation results saved to xlsx")
