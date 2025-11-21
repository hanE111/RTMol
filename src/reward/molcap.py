import re
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
from vllm import LLM, SamplingParams
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# --- Global vLLM Initialization ---
LLM_MODEL_PATH = '/XYFS01/HDD_POOL/paratera_xy/pxy547/checkpoints/verl_grpo_chemdfm/chemdfm_hf'
MAX_TOKENS_LLM = 512  # Defined output length limit for LLM
PROMPT = """You are an expert chemist. Given the molecular requirement description, your task is to design a new molecule using your experienced chemical knowledge.
Please strictly follow the format, no other information can be provided. You should only reply with SMILES string notations to represent the designed molecule. The SMILES must be valid and chemically reasonable.
Molecular requirement description: {}
Molecule SMILES:"""


# --- End Global vLLM Initialization ---

def extract_desc(text):
    # Find all non-overlapping matches of <desc>...</desc>
    # We use a non-capturing group for the tags themselves to simplify counting.
    # The (.*?) is the part we are interested in if there's exactly one match.
    matches = re.findall(r"<desc>(.*?)</desc>", text, re.DOTALL | re.IGNORECASE)

    # Check if there is exactly one match
    if len(matches) == 1:
        # If so, return the content of the first (and only) group, stripped.
        # matches[0] will contain the content of the (.*?) group from the single match.
        return matches[0].strip()
    else:
        # If there are zero or more than one matches, return an empty string
        return ""
def extract_xml_answer(text: str) -> str:
    try:
        # Handle potential variations in spacing or newlines
        match = re.search(r"^\s*<reasoning>\s*.*\s*</reasoning>\s*<answer>(.*?)</answer>\s*$", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        else:
            # Attempt to find answer tag even if reasoning is missing or malformed
            match_answer_only = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
            if match_answer_only:
                return match_answer_only.group(1).strip()
            return "" # Return empty if no answer tag found
    except Exception:
        return ""

# Reward 2: Fingerprint Similarity
def fingerprint_similarity_reward_func(solution_str_xml, true_smi): # solution_str_xml is the XML output from LLM
    # pred_smi = extract_xml_answer(solution_str_xml)
    pred_smi = solution_str_xml
    pred_mol = Chem.MolFromSmiles(pred_smi) if pred_smi else None
    true_mol = Chem.MolFromSmiles(true_smi)  # Assume true_smi is valid due to pre-filtering

    if pred_mol is None or true_mol is None:  # If prediction is invalid OR true_mol invalid
        # print(f"Debug: Invalid SMILES. Pred: '{pred_smi}', True: '{true_smi}'")
        return 0.0

    try:
        # e.g., Using Morgan fingerprints (ECFP4 equivalent) alone
        pred_fp = AllChem.GetMorganFingerprintAsBitVect(pred_mol, 2, nBits=2048)
        true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2, nBits=2048)
        similarity = DataStructs.TanimotoSimilarity(pred_fp, true_fp)
        return similarity
    except Exception as e:
        # print(f"Debug: Error in fingerprint calculation for Pred: '{pred_smi}', True: '{true_smi}': {e}")
        return 0.0

# Reward 3: Exact Match (Canonical SMILES)
def exact_smiles_match_reward_func(solution_str_xml, true_smi): # solution_str_xml is the XML output from LLM
    # pred_smi = extract_xml_answer(solution_str_xml)
    pred_smi = solution_str_xml
    try:
        pred_mol = Chem.MolFromSmiles(pred_smi) if pred_smi else None
        true_mol = Chem.MolFromSmiles(true_smi)  # Assumed valid

        if pred_mol is None or true_mol is None:
            return 0.0

        # Canonicalize both SMILES for fair comparison
        canonical_pred_smi = Chem.MolToSmiles(pred_mol, canonical=True)
        canonical_true_smi = Chem.MolToSmiles(true_mol, canonical=True)

        if canonical_pred_smi == canonical_true_smi:
            return 1.0
        else:
            return 0.0
    except Exception:
        return 0.0

def compute_scores(solution_str_prompts: str, ground_truths: str):
    """
    Computes the score by first passing solution_str_prompt to the LLM,
    then using the LLM's output to calculate fingerprint similarity and exact match scores.
    """
    # Initialize vLLM model
    # This assumes the machine has the necessary GPUs and vLLM is installed correctly.
    try:
        # Added trust_remote_code=True as it's often needed for custom models on Hugging Face.
        # Remove or set to False if your model doesn't require it or for security reasons.
        llm = LLM(model=LLM_MODEL_PATH, tokenizer=LLM_MODEL_PATH, trust_remote_code=True,tensor_parallel_size=8,gpu_memory_utilization=0.5) 
        sampling_params = SamplingParams(max_tokens=MAX_TOKENS_LLM) # Using deterministic sampling
        # print(f"vLLM model loaded successfully from {LLM_MODEL_PATH}")
    except Exception as e:
        # print(f"Error loading vLLM model from {LLM_MODEL_PATH}: {e}")
        # print("LLM features will be disabled.")
        llm = None
        sampling_params = None
    if llm is None or sampling_params is None:
        print("LLM not initialized. Cannot generate new solution. Returning 0.0 score.")
        return 0.0

    # llm_generated_solution_xml = "" # Initialize to empty string
    try:
        # 1. solution_str_prompt is the prompt for the LLM
        # just for consistent input, verl apply chat by default
        # prompts = [
        #     [{"role": "user", "content": PROMPT.format(solution_str_prompt)}] for solution_str_prompt in solution_str_prompts
        #     ]
        prompts = [PROMPT.format(solution_str_prompt) for solution_str_prompt in solution_str_prompts]

        # 2. Get LLM's output
        # vLLM generate returns a list of RequestOutput objects
        # request_outputs = llm.chat(prompts, sampling_params)
        request_outputs = llm.generate(prompts, sampling_params)
        llm_generated_solutions = [request_output.outputs[0].text for request_output in request_outputs]

    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return [0.0 for _ in range(len(prompts))]

    print(f"example [0], text: {prompts[0]}, output: {llm_generated_solutions[0]}, ground truth: {ground_truths[0]}")
    scores = [(exact_smiles_match_reward_func(llm_generated_solution, ground_truth), fingerprint_similarity_reward_func(llm_generated_solution, ground_truth)) for llm_generated_solution, ground_truth in zip(llm_generated_solutions,ground_truths)]
    
    scores_filtered = [-5.0 if len(solution_str_prompt) > 1000 else score[0]+score[1] for solution_str_prompt, score in zip(solution_str_prompts, scores)]
    print(f'exact_match:{sum([score[0] for score in scores])/len(scores)}\n similarity:{sum([score[1] for score in scores])/len(scores)}')
    return scores_filtered
