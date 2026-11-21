# CBP-PPO

Cross-group budget pooling and cooperation among PPO learners in spatial public goods games.

The package contains four sibling directories: `code/` with the reproduction programs, `output/` with archived data and run records, `paper/` with the marked manuscript and response letter, and `paper_clear/` with the clean manuscript. Earlier run records retain their original path strings. A historical `code/output/` path points to the corresponding file under `output/` in this package.

## Separate model files

| Model | Model implementation | Independent program | Internal identifier |
|---|---|---|---|
| PPO baseline used in the paper | `PPO.py` | `main_PPO.py` | `original`, eta=0, beta=0 |
| CBP-PPO | `CBP_PPO.py` | `main_CBP_PPO.py` | `UA`, eta=0.20, beta=3 |
| Groupwise-financing control | `CBP_PPO.py` | `main_CBP_PPO.py` | `GF` |

Each model file contains its own Actor–Critic and SPGG classes. CBP-PPO does not import the PPO model file. The shared network, column-reference advantage construction, normalized advantage plus focal-value critic target, losses and training loop are preserved from the executed paper implementation. The recorded GAE coefficient is inactive in the one-transition update.

## Run the six experiments

Use Python 3.12 and install the pinned dependencies in `requirements.txt`. Run these commands from `code/`:

```bash
python -m pip install -r requirements.txt
python -B scripts/run_all_formal.py --dry-run
python -B scripts/run_03_statistics.py --dry-run
python -B scripts/run_03_statistics.py --run --verify-archived --batch stats_replay --workers 4 --max-hours 24
```

The six `scripts/run_01_*.py` through `run_06_*.py` files list the scientific settings explicitly. The runner selects `main_PPO.py` for PPO and `main_CBP_PPO.py` for CBP-PPO/GF. Use `run_guides/` and `RUNNING.md` for each experiment.

Training requires explicit `--run`. Original results are not overwritten. New numeric data go to `../output/reruns/<batch>/`. Logs and checkpoints go to `../output/provenance/reruns/<batch>/`, and executed-source snapshots go to `provenance/reruns/<batch>/executed_sources/`. Failures are retained and stop new work. `--verify-archived` compares all saved arrays, rather than only endpoint cooperation. Different software environments may produce differences, which are reported as failures.

## Existing results and reproducibility

```bash
python -B scripts/verify_package.py
python -B scripts/analyze_results.py
python -B scripts/analyze_incentives.py
python -B scripts/redraw_figures.py
python -B scripts/validate_split_models.py --dry-run
```

The 2513 archived configurations and original trajectories remain unchanged. Their original execution hashes refer to the frozen joint source in `reference/submitted_execution_20261116/`. `config/experiments.json` records both original execution IDs and distinct IDs/source hashes for the separated programs. Historical provenance is not rewritten.

The source separation is checked by unchanged common-method ASTs, payoff/accounting comparisons for every configured payoff condition, and a fixed 56-run set of full 1000-epoch replays. The exact results and validation limits are recorded in `../output/provenance/reproduction_validation/` and `SOURCE_CHANGES.md`. This validation does not retrain all 2513 configurations.

Analysis and plotting reuse the archived data format and original drawing blocks. Figures are written only under `../paper/`; `../output/` contains no figures. Ordinary seed 9202 and statistical paired seeds 9201–9220 are unchanged. The statistical thresholds remain defined on the tested r grid and final 101-state window.

Validation summary: `../output/provenance/source_separation_20261116/validation_summary.json`.
