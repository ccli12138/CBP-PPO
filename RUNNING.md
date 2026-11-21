# 当前论文运行与复现

## 代码分离

- `PPO.py` 与 `main_PPO.py`：本文无干预PPO基线，只允许 `original`、eta=0、beta=0。
- `CBP_PPO.py` 与 `main_CBP_PPO.py`：本文CBP-PPO，默认 `UA`、eta=0.20、beta=3；`GF`为出资机制比较中的对照。
- 两份模型文件分别保留原ActorCritic和SPGG类，不通过相互导入或继承包装实现分离。网络、优势计算、标准化、critic目标、损失和训练循环不变。GAE系数0.95保留在实际配置中，单步更新下不起作用。
- `reference/submitted_execution_20261116/`保存产生原始结果的联合代码。数据的原执行ID及源码哈希不修改；分离后新运行有自己的源码哈希和ID。

## 环境

训练使用Python3.12、PyTorch2.5.1、NumPy1.26.4，CPU，每个训练进程1个Torch线程。依赖见`requirements.txt`。归档来源记录保留实际训练环境；验证报告另记本轮环境，不冒称版本完全相同。仅分析和绘图不需要PyTorch。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

以下命令均从 `code/` 目录执行。数据目录 `../output/` 与 `code/` 同层。历史来源记录保留当时的路径，原 `code/output/` 对应现包的 `output/`。六类实验脚本明确列出科学参数，运行器只按模型选取对应入口，不修改参数。

|脚本|实验|范围|唯一配置|
|---|---|---|---:|
|run_01_parameters.py|双参数|r=2.0:0.1:5.5；eta={0,.05,.10,.20,.30,.60}；beta={0,1,2,3,4,8}|1116|
|run_02_mechanisms.py|CBP-PPO与GF|r=3.0:0.1:5.0；eta=.20、beta=3|42|
|run_03_statistics.py|PPO与CBP-PPO配对统计|36个r，每r每模型20次，种子9201–9220|1440|
|run_04_across_r.py|不同r下合作与背叛|36个r，两模型，seed9202|72|
|run_05_spatial.py|空间演化与状态激励|r=3.5、4.1，两模型|4|
|run_06_initialization.py|初始化敏感性|r=3.5、4.1，两模型，随机一半/全背叛|8|

六类复用相同配置，合并为2513项，不能直接相加。question=2为上半背叛、下半合作；1为独立Bernoulli0.5；3为全背叛。共同设置：L=200、T=1000、Adam学习率0.001、gamma=.99、clip=.2、lambda=.95、critic权重.5、entropy权重.01、ppo_epochs=1、batch_size=1、两层32单元ReLU、梯度裁剪.5。具体每项配置见`config/experiments.json`。

## 预览与训练

```bash
python -B scripts/run_all_formal.py --dry-run
python -B scripts/run_03_statistics.py --dry-run
python -B scripts/run_03_statistics.py --run --verify-archived --batch stats_replay --workers 4 --max-hours 24
```

- 默认预览，不启动训练；必须显式提供`--run`。
- 推荐通过六类脚本启动，运行器自动选择独立模型入口，并将执行源码副本存入 `provenance/reruns/<batch>/executed_sources/`。运行日志与检查点写入 `../output/provenance/reruns/<batch>/`。
- 新数值结果写入`../output/reruns/<batch>/`，原始`../output/results/`只读。已有batch不覆盖。`--resume`仅接续未提交的配置；失败或未完成尝试必须先检查，无自动重试。
- `--verify-archived`逐项比较新旧NPZ中所有数组的形状、类型和字节。NPZ压缩容器可能因写入时间不同而有不同文件哈希，因此比较数组内容。
- 新记录的`source_sha256`对应实际分离源码，`archived_run_id`指向原始执行。不能把分离后文件哈希写回旧数据。
- 并行调度仅启动独立Python训练进程，不在共享线程中训练模型。Mac沙箱验证设置`KMP_USE_SHM=0`，收益核验与调度进程隔离；这是执行环境处理，不改变学习算法。

## 数据核验、分析与绘图

```bash
python -B scripts/verify_package.py
python -B scripts/analyze_results.py
python -B scripts/analyze_incentives.py
python -B scripts/redraw_figures.py
```

数据核验检查2513份原始轨迹、旧源码绑定、新入口/新ID映射、共同学习函数AST及当前交付清单。统计重算保留20次配对队列、最后101状态窗口、0.999全窗判据及20000次BCa bootstrap；新结果写入 `../output/` 下独立的 recomputed 目录。重绘复用原绘图块，仅写入paper下独立redrawn目录，正式图片不覆盖。

`analyze_incentives.py`从四项空间实验的20个快照重算固定状态激励、净收益调整及16个取样位置的收益差，并与归档诊断数组核对。`uniform_incentives.py`提供独立逐群组枚举核验。绘图清单位于`config/plot_manifest_20261015.json`，运行入口不依赖workflow中的工作记录。

## 分离代码的有限重跑验证

```bash
python -B scripts/validate_split_models.py --dry-run
python -B scripts/validate_split_models.py --run --workers 4 --max-hours 2 --batch my_split_check
```

验证清单在修改源码前冻结于`config/split_validation_selection.json`：56个完整L=200、T=1000配置，覆盖PPO、CBP-PPO、GF、参数边界、转变附近与初始化。对所有配置先检查入口及来源映射，对全部不同收益参数条件比较结算，再对56项重跑的12个数组逐字节比较。失败保留并停止，不更换有利种子。这不是重训全部2513项，也不是新的论文统计队列。

首次调度失败记录和后续实际通过范围均保留在`../output/provenance/reproduction_validation/`。完整验证结论见`../output/provenance/source_separation_20261116/validation_summary.json`，逐项轨迹比较见对应batch的`report.json`。
