# Usage

## Environment Setup
We implement our model on `Python 3.10`. These packages are mainly used:

```
rdkit                2025.3.3
torch                2.6.0+cu118
vllm				 0.8.4
numpy                2.1.2
verl                 0.4.0.dev0
nltk
rouge_score
Levenshtein
```

## Datasets

We use three datasets for train/val: [ChEBI-20](https://github.com/cnedwards/text2mol), [L+M](https://huggingface.co/datasets/language-plus-molecules/LPM-24_train), and [Mol-Instruction](https://huggingface.co/datasets/zjunlp/Mol-Instructions). We remove duplicate molecules that exist in ChEBI-20 from L+M and Mol-Instruction.

We use [ChEBI-100](https://github.com/OpenDFM/ChemDFM/blob/main/ChemLLMBench_eval_data/text_based_molecule_design.jsonl) for evaluation.

See `src/data_preprocess/` for initial data processing (e.g. `chebi.py`).

See `process_data.py` for data batch split.

## Reward Model

See `src/reward/` for reward calculation.

To Use reward model, please change several files in `verl` repo:
1. `cp ./src/reward/*.py verl/verl/utils/reward_score`

2. Modify `verl/verl/utils/reward_score/__init__.py`:
    ```python
    # Copyright 2024 Bytedance Ltd. and/or its affiliates
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # from . import gsm8k, math, prime_math, prime_code


    def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
        if data_source == "openai/gsm8k" or 'gsm8k' in data_source:
            from . import gsm8k

            res = gsm8k.compute_score(solution_str, ground_truth)
        elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
            from . import math

            res = math.compute_score(solution_str, ground_truth)
            # [Optional] Math-Verify Integration
            # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
            # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
            # To use it, override the `compute_score` function with the following implementation:

            # from . import math_verify
            # res = math_verify.compute_score(solution_str, ground_truth)
        elif data_source == "math_dapo" or data_source.startswith("aime"):
            from . import math_dapo

            res = math_dapo.compute_score(solution_str, ground_truth)
        elif data_source in [
            "numina_aops_forum",
            "numina_synthetic_math",
            "numina_amc_aime",
            "numina_synthetic_amc",
            "numina_cn_k12",
            "numina_olympiads",
        ]:
            from . import prime_math

            res = prime_math.compute_score(solution_str, ground_truth)
        elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
            from . import prime_code

            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
        elif data_source in ["hiyouga/geometry3k"]:
            from . import geo3k

            res = geo3k.compute_score(solution_str, ground_truth)
        elif data_source in ['chebi']:
            from . import chebi
            res = chebi.compute_score(solution_str,ground_truth)
        elif data_source in ['molcap']:
            from . import molcap
            res = molcap.compute_score(solution_str,ground_truth)
        else:
            raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

        if isinstance(res, dict):
            return res
        elif isinstance(res, (int, float, bool)):
            return float(res)
        else:
            return float(res[0])

    def _default_compute_scores(data_sources, solution_strs, ground_truths, extra_infos=None):
        if data_sources[0] in ['molcap']:
            from . import molcap
            res = molcap.compute_scores(solution_strs,ground_truths)
            return res
        else:
            raise NotImplementedError(f"Reward function is not implemented for {data_sources[0]}")

    ```

3. Modify `verl/verl/workers/reward_manager/batch.py`:
    ```python
    # Copyright 2025 Individual Contributor: Mert Unsal
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    from collections import defaultdict

    import torch

    from verl import DataProto
    from verl.utils.reward_score import _default_compute_scores


    class BatchRewardManager:
        def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **reward_kwargs):
            self.tokenizer = tokenizer
            self.num_examine = num_examine
            self.compute_score = compute_score or _default_compute_scores
            self.reward_fn_key = reward_fn_key
            self.reward_kwargs = reward_kwargs

        def verify(self, data):
            prompt_ids = data.batch["prompts"]
            response_ids = data.batch["responses"]
            attention_mask = data.batch["attention_mask"]

            prompt_len = prompt_ids.shape[-1]
            valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

            responses_str = []
            for i in range(len(data)):
                valid_len = valid_response_lengths[i]
                valid_response_ids = response_ids[i][:valid_len]
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                responses_str.append(response_str)

            ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
            data_sources = data.non_tensor_batch[self.reward_fn_key]
            extras = data.non_tensor_batch.get("extra_info", [None] * len(data))

            scores = self.compute_score(
                data_sources=data_sources,
                solution_strs=responses_str,
                ground_truths=ground_truths,
                extra_infos=extras,
                **self.reward_kwargs,
            )

            return scores

        def __call__(self, data: DataProto, return_dict=False):
            # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
            if "rm_scores" in data.batch.keys():
                if return_dict:
                    return {"reward_tensor": data.batch["rm_scores"]}
                else:
                    return data.batch["rm_scores"]

            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_extra_info = defaultdict(list)
            prompt_ids = data.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            attention_mask = data.batch["attention_mask"]
            valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
            data_sources = data.non_tensor_batch[self.reward_fn_key]

            scores = self.verify(data)
            rewards = []
            already_printed = {}

            for i in range(len(data)):
                length = valid_response_lengths[i].item()
                score = scores[i]

                if isinstance(score, dict):
                    reward = score["score"]
                    for key, value in score.items():
                        reward_extra_info[key].append(value)
                else:
                    reward = score

                rewards.append(reward)
                reward_tensor[i, length - 1] = reward

                data_source = data_sources[i]
                if already_printed.get(data_source, 0) < self.num_examine:
                    response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                    prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                    ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    print("[ground_truth]", ground_truth)
                    print("[score]", scores[i])
                    already_printed[data_source] = already_printed.get(data_source, 0) + 1

            # data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

            if return_dict:
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                return reward_tensor

    ```

## Experiments

Run `phase_1.sh` and `phase_2.sh` cyclically to train ChemDFM. Other models can be trained in a similar way by changing base model.

## Evaluation

Run `valid_final.py` to evaluate the model.

## Note

Please change the path accordingly.
