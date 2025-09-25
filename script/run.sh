export REINFLOW_DIR=/home/hiratsuka/SourceCode/ReinFlow
export REINFLOW_DATA_DIR="${REINFLOW_DIR}/data"
export REINFLOW_LOG_DIR="${REINFLOW_DIR}/log"
export HYDRA_FULL_ERROR=1
export REINFLOW_WANDB_ENTITY=hirekatsu0523

uv run script/run.py --config-name=ft_ppo_pi0

# ./script/run.sh