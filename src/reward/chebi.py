import re
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def extract_xml_answer(text: str) -> str:
    try:
        # Handle potential variations in spacing or newlines
        match = re.findall(r"^\s*<reasoning>\s*.*\s*</reasoning>\s*<answer>(.*?)</answer>\s*$", text, re.DOTALL | re.IGNORECASE)
        if len(match)==1:
                return match[0].strip()
        else:
            return ""
    except Exception:
        return ""

def strict_format_reward_func(solution_str):
    pattern = r"^\s*<reasoning>\s*.*\s*</reasoning>\s*<answer>\s*.*\s*</answer>\s*$"

    match = re.match(pattern, solution_str, re.DOTALL)
    # Reward: 0.15 (Scaled down to fit total format reward of 0.5)
    return 0.15 if match else 0.0

def soft_format_reward_func(solution_str):
    pattern = r"<reasoning>.*?</reasoning>.*?<answer>.*?</answer>"
    match = re.match(pattern, solution_str, re.DOTALL)
    # Reward: 0.15 (Scaled down to fit total format reward of 0.5)
    return 0.15 if match else 0.0

def count_xml(solution_str) -> float:
    count = 0.0
    reasoning_start_count = solution_str.count("<reasoning>")
    reasoning_end_count = solution_str.count("</reasoning>")
    answer_start_count = solution_str.count("<answer>")
    answer_end_count = solution_str.count("</answer>")
    # Reward presence of exactly one pair of tags
    if reasoning_start_count == 1: count += 0.05 # Scaled down
    if reasoning_end_count == 1: count += 0.05 # Scaled down
    if answer_start_count == 1: count += 0.05 # Scaled down
    if answer_end_count == 1: count += 0.05 # Scaled down
    # Penalize multiple tags heavily
    if reasoning_start_count > 1: count -= 0.5
    if reasoning_end_count > 1: count -= 0.5
    if answer_start_count > 1: count -= 0.5
    if answer_end_count > 1: count -= 0.5
    # Penalize trailing text after </answer>
    parts = re.split(r"</answer>", solution_str, flags=re.IGNORECASE) # Use regex split for case-insensitivity
    if len(parts) > 1:
        trailing_text = parts[-1].strip()
        count -= len(trailing_text) * 0.01 # Small penalty for trailing characters
    return max(-1.0, count) # Allow small negative penalty but cap it


# --- 3.b. New SMILES Rewards ---

# Reward 1: SMILES Validity (1 point if valid, 0 otherwise)
def smiles_validity_reward_func(solution_str):
    # rewards = []
    smi = extract_xml_answer(solution_str)
    if not smi: # Empty SMILES is invalid
        return 0.0
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return 0.5
        else:
            return 0.0
    except Exception:
        return 0.0

# Reward 2: Fingerprint Similarity (Similarity * 2, 0 if invalid)
def fingerprint_similarity_reward_func(solution_str, true_smi):
    # pred_smi = extract_xml_answer(solution_str)
    pred_smi = solution_str
    pred_mol = Chem.MolFromSmiles(pred_smi) if pred_smi else None
    true_mol = Chem.MolFromSmiles(true_smi) # Assume true_smi is valid due to pre-filtering

    if pred_mol is None or true_mol is None: # If prediction is invalid OR true_mol invalid (shouldn't happen)
        return 0.0

    try:
        # e.g., Using Morgan fingerprints (ECFP4 equivalent) alone
        pred_fp = AllChem.GetMorganFingerprintAsBitVect(pred_mol, 2, nBits=2048)
        true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2, nBits=2048)
        similarity = DataStructs.TanimotoSimilarity(pred_fp, true_fp)
        return similarity
    
    except Exception:
        return 0.0

# Reward 3: Exact Match (Canonical SMILES) (3 points if exact match, 0 otherwise)
def exact_smiles_match_reward_func(solution_str, true_smi):
    # pred_smi = extract_xml_answer(solution_str)
    pred_smi = solution_str
    try:
        pred_mol = Chem.MolFromSmiles(pred_smi) if pred_smi else None
        true_mol = Chem.MolFromSmiles(true_smi) # Assumed valid

        if pred_mol is None or true_mol is None:
            return 0.0

        # Canonicalize both SMILES for fair comparison
        canonical_pred_smi = Chem.MolToSmiles(pred_mol, canonical=True)
        canonical_true_smi = Chem.MolToSmiles(true_mol, canonical=True)

        if canonical_pred_smi == canonical_true_smi:
            return 1.0
        else:
            return 0.0
    except:
        return 0.0
    
def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    # return (strict_format_reward_func(solution_str)+
    #         soft_format_reward_func(solution_str)+
    #         count_xml(solution_str)+
    #         smiles_validity_reward_func(solution_str)+
    #         fingerprint_similarity_reward_func(solution_str,ground_truth)+
    #         exact_smiles_match_reward_func(solution_str, ground_truth))
    # return exact_smiles_match_reward_func(solution_str, ground_truth)
    return fingerprint_similarity_reward_func(solution_str,ground_truth) + exact_smiles_match_reward_func(solution_str, ground_truth)