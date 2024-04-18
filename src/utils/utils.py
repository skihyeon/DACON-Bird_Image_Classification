import os
import random
import torch
import numpy as np
import wandb

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def wandb_login():
    wandb_api_key = os.environ.get('WANDB_API_KEY')
    wandb.login(key=wandb_api_key)
    return wandb


def update_wandb_config(wandb, config):
    for key, value in config.settings.items():
        wandb.config.update({key: value}, allow_val_change=True)

def resume_latest_run_log(wandb, entity, project_name, target_run_name):
    # 특정 프로젝트에서 모든 실행 가져오기
    runs = wandb.Api().runs(f"{entity}/{project_name}")

    # 실행 이름으로 필터링하고 최신 실행 찾기
    latest_run = None
    for run in runs:
        if run.name == target_run_name:
            if latest_run is None or run.created_at > latest_run.created_at:
                latest_run = run

    # 최신 실행 정보가 있는 경우 실행을 재개
    if latest_run:
        print(f"Resuming run: {latest_run.name} with ID {latest_run.id}")
        wandb.init(
            project=project_name,
            entity=entity,
            name=latest_run.name,
            resume="must",
            id=latest_run.id  # 이 `id`로 실행을 명시적으로 재개
        )
    else:
        print("No matching runs found. Starting a new run.")
        wandb.init(project=project_name, entity=entity, name=target_run_name)