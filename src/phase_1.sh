#!/bin/bash
#SBATCH -p pxy547
module load anaconda3/2023.09 CUDA/12.3
module load nccl/2.19.3-cuda-12.3
source activate verl
unset ROCR_VISIBLE_DEVICES

export PYTHONUNBUFFERED=1

set -x

# export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_RMSNORM_KERNEL=0

# === 用户配置 ===
START=1
BASE_DIR="/XYFS01/HDD_POOL/paratera_xy/pxy547/roundtrip/checkpoints/verl_grpo_chemdfm/chemdfm"
TARGET_DIR="/XYFS01/HDD_POOL/paratera_xy/pxy547/checkpoints/verl_grpo_chemdfm/chemdfm_hf"
DATA_DIR="/XYFS01/HDD_POOL/paratera_xy/pxy547/for_verl/verl/data"
MODEL_MERGER_SCRIPT="/XYFS01/HDD_POOL/paratera_xy/pxy547/for_verl/verl/scripts/model_merger.py"
yEXPERIMENT_NAME="chemdfm"
i=$1

# === 开始训练轮次 ===

a=$((START + i * 2))
b=$((a + 1))
echo -e "\n===================== ROUND $i =====================\n"

# === Phase 1: text → mol ===
echo ">>> [Phase 1] text2mol Training..."
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/chebi_chemdfm/train_${i}.parquet \
    data.val_files=${DATA_DIR}/chebi_chemdfm/test_100.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=${TARGET_DIR} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.model_type='llama' \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=32 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.model_type='llama' \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    +actor_rollout_ref.rollout.max_new_tokens=256 \
    +actor_rollout_ref.rollout.max_tokens=256 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=[console,tensorboard] \
    trainer.project_name=verl_grpo_chemdfm \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    +trainer.remove_previous_ckpt_in_save=True \
    +trainer.max_ckpt_to_keep=2 \
    actor_rollout_ref.actor.checkpoint.save_contents='[model,optimizer,extra]' \
    trainer.test_freq=5 \
    trainer.total_epochs=${a}

# === 合并 FSDP 模型为 HF ===
echo ">>> Convert latest FSDP to HuggingFace..."

# 查找最大 step
LATEST_STEP=$(ls -d ${BASE_DIR}/global_step_*/actor 2>/dev/null | \
    sed -E 's|.*/global_step_([0-9]+)/actor|\1|' | sort -nr | head -n1)

if [ -z "$LATEST_STEP" ]; then
    echo "❌ 找不到 global_step_* 目录"
    exit 1
fi

LOCAL_DIR="${BASE_DIR}/global_step_${LATEST_STEP}/actor"
echo "✅ 使用最新模型目录: $LOCAL_DIR"

python3 ${MODEL_MERGER_SCRIPT} merge \
    --backend fsdp \
    --local_dir "$LOCAL_DIR" \
    --target_dir "$TARGET_DIR"
cp /XYFS01/HDD_POOL/paratera_xy/pxy547/models/ChemDFM-v1.5-8B/tokenizer_config.json /XYFS01/HDD_POOL/paratera_xy/pxy547/checkpoints/verl_grpo_chemdfm/chemdfm_hf/

echo ">>> Eval..."
python /HOME/paratera_xy/pxy547/HDD_POOL/roundtrip/valid_final.py \
  --output_file /XYFS01/HDD_POOL/paratera_xy/pxy547/roundtrip/res/${EXPERIMENT_NAME}_roundtrip_phase1_${i}.xlsx
  