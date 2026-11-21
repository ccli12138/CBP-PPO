import torch
from PPO import SPGG
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import asyncio
import argparse
import os
import shutil
import re

import hashlib
import platform
import random
import time
import sys
def save_params_to_json(params, filename_prefix="params",output_path='data'):
    # 创建参数保存目录
    param_dir = Path(output_path)
    os.makedirs(output_path, exist_ok=True)
    # param_dir.mkdir(exist_ok=True)
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"
    filepath = param_dir / filename
    
    # 转换特殊数据类型
    serializable_params = {
        k: str(v) if isinstance(v, torch.device) else v  # 处理device类型
        for k, v in params.items()
    }
    
    # 保存到JSON
    with open(filepath, 'w') as f:
        json.dump(serializable_params, f, indent=4)
    
    src_file=str(Path(__file__).resolve())
    dst_file=str(Path(output_path) / Path(src_file).name)
    shutil.copy2(src_file, dst_file)
    src_file=str(Path(__file__).with_name('PPO.py').resolve())
    dst_file=str(Path(output_path) / Path(src_file).name)
    shutil.copy2(src_file, dst_file)

    print(f"参数已保存至: {filepath}")

# 主实验程序: original r/run loops and model construction retained.
async def main(args):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H%M%S")
    r_values = args.r
    if not r_values:
        raise ValueError('Explicit -r values are required; no implicit training grid')
    if args.device != 'cpu':
        raise ValueError('This study is frozen to CPU')
    device = torch.device('cpu')
    torch.set_num_threads(args.threads)
    outputs = []
    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ('PPO.py', 'main_PPO.py')}
    for r in r_values:
        for eta in args.eta:
            for beta in args.beta:
                for variant in args.variant:
                    for num in range(args.runs):
                        seed = args.seed + num
                        experiment_params = {
                            'r': r, 'epochs': args.epochs, 'L_num': args.L_num,
                            'alpha': args.alpha, 'gamma': args.gamma,
                            'clip_epsilon': args.clip_epsilon, 'question': args.question,
                            'ppo_epochs': args.ppo_epochs, 'batch_size': args.batch_size,
                            'gae_lambda': args.gae_lambda, 'device': str(device),
                            'seed': seed, 'delta': args.delta, 'rho': args.rho,
                            'eta': eta, 'beta': beta, 'variant': variant,
                            'threads': args.threads,
                        }
                        identity = {'config': experiment_params, 'sources': source_hashes}
                        run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
                        output_path = Path(args.output_path).resolve() / 'runs' / run_id
                        output_path.mkdir(parents=True, exist_ok=False)
                        np.random.seed(seed)
                        random.seed(seed)
                        torch.manual_seed(seed)
                        save_params_to_json(experiment_params, output_path=str(output_path))
                        provenance = {'status': 'running', 'model': 'PPO',
                                      'experiment': args.experiment, 'run_id': run_id,
                                      'config': experiment_params, 'source_sha256': source_hashes,
                                      'python': platform.python_version(), 'torch': torch.__version__,
                                      'numpy': np.__version__, 'platform': platform.platform(),
                                      'started_utc': datetime.now().astimezone().isoformat(),
                                      'reference_data': False}
                        provenance_path = output_path / 'provenance.json'
                        provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
                        model = SPGG(
                            L_num=args.L_num, device=device, alpha=args.alpha,
                            gamma=args.gamma, clip_epsilon=args.clip_epsilon,
                            r=r, epochs=args.epochs, now_time=formatted_time,
                            question=args.question, ppo_epochs=args.ppo_epochs,
                            batch_size=args.batch_size, gae_lambda=args.gae_lambda,
                            output_path=str(output_path), delta=args.delta, rho=args.rho,
                            eta=eta, beta=beta, variant=variant,
                        )
                        model.count = num
                        model.run_deadline = args.deadline
                        started = time.monotonic()
                        try:
                            D_Y, C_Y, D_Value, C_Value, all_value = model.run()
                            snapshot_epochs = sorted(model.snapshots)
                            metrics = model.settlement_metrics
                            arrays = {
                                'cooperation': np.asarray(C_Y), 'defection': np.asarray(D_Y),
                                'total_payoff': np.asarray(all_value),
                                'snapshot_epochs': np.asarray(snapshot_epochs, dtype=np.int64),
                                'strategy_snapshots': np.stack([model.snapshots[t] for t in snapshot_epochs]),
                                'reward_snapshots': np.stack([model.snapshot_rewards[t] for t in snapshot_epochs]),
                            }
                            for name in metrics[0]:
                                arrays[name] = np.asarray([m[name] for m in metrics])
                            error = arrays['adjusted_total'] - arrays['base_total'] + arrays['payment_total'] + arrays['fine_total']
                            scale = np.maximum(1.0, np.abs(arrays['base_total']) + arrays['payment_total'] + arrays['fine_total'])
                            if np.max(np.abs(error) / scale) > 2e-6:
                                raise ArithmeticError('Aggregate accounting check failed')
                            for name, arr in arrays.items():
                                if not np.isfinite(arr).all():
                                    raise FloatingPointError(name)
                            data_path = output_path / 'trajectory.npz'
                            np.savez_compressed(data_path, **arrays)
                            provenance.update(status='complete', elapsed_seconds=time.monotonic()-started,
                                              data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                                              array_sha256={name: hashlib.sha256(arr.tobytes()).hexdigest() for name, arr in arrays.items()},
                                              accounting_relative_error=float(np.max(np.abs(error)/scale)),
                                              completed_utc=datetime.now().astimezone().isoformat())
                        except BaseException as exc:
                            provenance.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                                              elapsed_seconds=time.monotonic()-started)
                            if hasattr(model, 'partial_trace'):
                                np.savez_compressed(output_path / 'partial_trajectory.npz',
                                                    cooperation=np.asarray(model.partial_trace[0]),
                                                    defection=np.asarray(model.partial_trace[1]),
                                                    total_payoff=np.asarray(model.partial_trace[2]))
                            provenance_path.write_text(json.dumps(provenance, indent=2)+'\n')
                            raise
                        provenance_path.write_text(json.dumps(provenance, indent=2)+'\n')
                        outputs.append(str(output_path))
    return outputs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Unintervened PPO baseline for this paper')
    parser.add_argument('-r', type=float, nargs='+', default=None)
    parser.add_argument('-epochs', type=int, default=1000)
    parser.add_argument('-runs', type=int, default=1)
    parser.add_argument('-L_num', type=int, default=200)
    parser.add_argument('-alpha', type=float, default=1e-3)
    parser.add_argument('-gamma', type=float, default=0.99)
    parser.add_argument('-clip_epsilon', type=float, default=0.2)
    parser.add_argument('-question', type=int, choices=[1,2,3], default=2)
    parser.add_argument('-ppo_epochs', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-gae_lambda', type=float, default=0.95)
    parser.add_argument('-device', type=str, choices=['cpu'], default='cpu')
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-output_path', '--output-dir', dest='output_path', default='results_new')
    parser.add_argument('-delta', type=float, default=0.5)
    parser.add_argument('-rho', type=float, default=0.01)
    parser.add_argument('--eta', type=float, nargs='+', default=[0.0])
    parser.add_argument('--beta', type=float, nargs='+', default=[0.0])
    parser.add_argument('--variant', nargs='+', choices=['original'], default=['original'])
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--experiment', default='explicit_cli')
    parser.add_argument('--deadline', type=float, default=None, help='Absolute wall-clock Unix deadline, execution only')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    asyncio.run(main(args))
