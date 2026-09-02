# TA-TFN: Task-Adaptive Frequency Decoupling for Generalizable Battery State Estimation across Chemistries

Official PyTorch implementation of **TA-TFN**, accepted at **KDD 2026**.

> Jing Wang, Shiyuan Piao, and Fugee Tsung. “TA-TFN: Task-Adaptive Frequency Decoupling for Generalizable Battery State Estimation across Chemistries.” *Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '26)*, 2026. [DOI: 10.1145/3770855.3818162](https://doi.org/10.1145/3770855.3818162)

TA-TFN is a lightweight multi-task framework for joint battery state-of-charge (SOC) and state-of-health (SOH) estimation across heterogeneous battery chemistries. It explicitly addresses **spectral conflict**: SOC relies mainly on short-term electrochemical transients, whereas SOH depends more strongly on slow degradation trends.

This repository is developed on top of the [BatteryLife](https://github.com/Ruifeng-Tan/BatteryLife) data pipeline and benchmark. Datasets are **not redistributed here**; download them from their public sources and follow their licenses and citation requirements.

## Model overview

TA-TFN is based on the spectral separation hypothesis: information useful for SOC and SOH occupies different spectral subspaces. It consists of:

1. **Parameterized spectral front-end (TFconv)** — continuous complex Gabor kernels are generated from learnable center frequencies and bandwidths. Cosine/sine convolution followed by envelope extraction provides phase-invariant local spectral energy.
2. **Dual-domain feature manifolds** — TFconv produces the spectral stream while a parallel convolution preserves the temporal waveform stream.
3. **Task-Adaptive Frequency Decoupling (TAFD)** — separate learnable global gates allocate spectral and temporal information to SOC and SOH. A soft low-pass regularizer biases the SOH path toward slow degradation patterns.
4. **Hybrid hierarchical backbone** — a weight-shared Siamese-style CNN–Transformer processes the two task pathways without collapsing them into one representation.
5. **Task-specific heads** — global pooling and projection MLPs produce SOC and SOH estimates.

```text
E_task = alpha_task * H_freq + (1 - alpha_task) * Phi_task(H_time)
```

Here, `Phi_SOC` is the identity mapping and `Phi_SOH` is a smoothing operator. This representation-level decoupling is complementary to optimization methods such as FAMO, PCGrad and LibMTL-EW.

The main implementation is in [`models/TFN.py`](models/TFN.py), and the training/evaluation entry point is [`run_main.py`](run_main.py).

## Highlights

- Unified SOC/SOH estimation with explicit task-specific spectral allocation.
- End-to-end learnable Gabor filter bank without handcrafted frequency features.
- Evaluation on **1,102 cells from 16 sub-datasets**, spanning Li-ion, Na-ion and Zn-ion chemistries.
- Cell-level train/validation/test splitting to prevent leakage between cycles of the same battery.
- Few-shot cross-chemistry SOC transfer using only 5% target-domain adaptation data.
- **0.9735M parameters and 0.0252 GFLOPs**, providing a favorable accuracy-efficiency trade-off.

## Main results

Results are averaged over five seeds (`2024-2028`). Lower is better.

| Benchmark | SOC MAE | SOC RMSE | SOC MAPE | SOH MAE | SOH RMSE | SOH MAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MIX-ALL | 0.0106 ± 0.0010 | 0.0159 ± 0.0012 | 1.37 ± 0.15% | 0.0180 ± 0.0047 | 0.0345 ± 0.0118 | 2.13 ± 0.84% |
| MIX-Li | 0.0092 ± 0.0039 | 0.0137 ± 0.0063 | 1.20 ± 0.53% | 0.0089 ± 0.0014 | 0.0159 ± 0.0030 | 0.94 ± 0.15% |
| CALB | 0.0060 ± 0.0014 | 0.0071 ± 0.0011 | 0.61 ± 0.15% | 0.0033 ± 0.0009 | 0.0047 ± 0.0014 | 0.35 ± 0.10% |
| Na-ion | 0.0086 ± 0.0141 | 0.0093 ± 0.0147 | 1.00 ± 1.64% | 0.0091 ± 0.0109 | 0.0116 ± 0.0147 | 0.92 ± 1.09% |
| Zn-ion | 0.0172 ± 0.0041 | 0.0286 ± 0.0036 | 1.70 ± 0.39% | 0.0393 ± 0.0137 | 0.0508 ± 0.0170 | 4.84 ± 1.60% |

In the 5% few-shot cross-chemistry SOC setting, TA-TFN obtains MAEs of **0.0340**, **0.0669**, and **0.0832** for Target-Na, Target-Li, and Target-Zn, respectively.

## Repository structure

```text
battery-TA-TFN/
├── models/                 # TFN and comparison models
│   ├── TFN.py              # main TA-TFN implementation
│   └── TFN_ablation.py     # ablation variants
├── layers/                 # reusable neural-network layers
├── data_provider/          # BatteryLife loading and split logic
├── losses/                 # training objectives
├── utils/                  # metrics and training helpers
├── scripts/                # experiment and plotting scripts
├── visualization/          # explainability/figure scripts
├── run_main.py             # train and evaluate models
├── DATA_AVAILABILITY.md    # dataset sources and preprocessing notes
└── requirements.txt
```

Generated datasets, checkpoints, logs, pretrained models and complete experiment outputs are intentionally excluded from version control.

## Installation

Python 3.10 or 3.11 and a CUDA-capable GPU are recommended.

```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
cd <YOUR_REPOSITORY>

python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA version if the pinned build is not suitable. DeepSpeed is optional for ordinary single-GPU runs and may require Linux/CUDA.

## Data preparation

Public data links and usage notes are listed in [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md). The paper uses 16 BatteryLife sub-datasets and constructs **MIX-Li** (Li-ion only) and **MIX-ALL** (Li-ion, Na-ion and Zn-ion). Processed data is available from:

- [Hugging Face: BatteryLife_processed](https://huggingface.co/datasets/Hongwxx/BatteryLife_processed)
- [Zenodo: BatteryLife processed release](https://zenodo.org/records/15531867)

Place downloaded data outside Git tracking, for example:

```text
dataset/
├── CALB2024_dataset/
├── NAion2024_dataset/
├── ZN-coin_dataset/
└── ...
```

The exact layout follows the BatteryLife processed-data format. Always pass the dataset root with `--root_path`; the default in `run_main.py` is a developer-local path.

## Training

Train the full TA-TFN for joint SOC/SOH estimation:

```bash
python run_main.py \
  --is_training 1 \
  --model TFN \
  --prediction_task multi \
  --dataset CALB2024 \
  --root_path /path/to/dataset \
  --checkpoints ./checkpoints \
  --seq_len 96 \
  --charge_discharge_length 100 \
  --mid_channel 16 \
  --batch_size 2 \
  --learning_rate 5e-5 \
  --train_epochs 160 \
  --patience 30 \
  --loss MSE \
  --seed 2024
```

In Windows PowerShell, replace each trailing `\` with a backtick or place the command on one line.

Single-task examples:

```bash
# SOH
python run_main.py --is_training 1 --model TFN --prediction_task SOH --c_out 1 --dataset CALB2024 --root_path /path/to/dataset

# SOC
python run_main.py --is_training 1 --model TFN --prediction_task SOC --c_out 1 --dataset CALB2024 --root_path /path/to/dataset
```

| Option | Meaning | Default |
| --- | --- | ---: |
| `--prediction_task` | `SOC`, `SOH`, or `multi` | `multi` |
| `--dataset` | dataset/split identifier | `CALB2024` |
| `--seq_len` | number of observed cycles | `96` |
| `--charge_discharge_length` | resampled points per curve | `100` |
| `--mid_channel` | TFN hidden channels | `16` |
| `--init_alpha_soc` | optional initial SOC fusion coefficient | model default |
| `--init_alpha_soh` | optional initial SOH fusion coefficient | model default |
| `--multi_seed` | comma-separated seeds, e.g. `2024,2025,2026` | empty |

Run `python run_main.py --help` for all options.

## Evaluation

Set `--is_training 0` and reuse the trained run's model/data configuration. Keep `args.json` beside a released checkpoint so the experiment can be reconstructed.

```bash
python run_main.py \
  --is_training 0 \
  --model TFN \
  --prediction_task multi \
  --dataset CALB2024 \
  --root_path /path/to/dataset \
  --checkpoints ./checkpoints
```

Reports include MAE, RMSE, MAPE and R². Depending on enabled options, the pipeline can also export prediction curves, warning reports and explainability figures.

## Reproducibility

- Split at battery-cell level to avoid leakage between cycles of the same cell.
- Report the paper's five random seeds (`2024,2025,2026,2027,2028`) as mean ± standard deviation.
- Record the dataset version, split, command, environment and Git commit.
- Release `args.json`, metric summaries and representative figures with each checkpoint.
- Never commit API keys, local paths, private data or Weights & Biases credentials.

## Pretrained weights and results

Publish large checkpoints through **GitHub Releases**, Hugging Face or Zenodo rather than Git. A useful release bundle is:

```text
TA-TFN-CALB2024/
├── model.safetensors
├── args.json
├── label_scalers
├── run_metrics_summary.json
└── SHA256SUMS
```

If weights are not public yet, provide the exact reproduction command instead.

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{wang2026tatfn,
  author    = {Wang, Jing and Piao, Shiyuan and Tsung, Fugee},
  title     = {TA-TFN: Task-Adaptive Frequency Decoupling for Generalizable Battery State Estimation across Chemistries},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3770855.3818162}
}
```

Please also cite BatteryLife and the original dataset papers listed in [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).

## Acknowledgements

This project builds upon BatteryLife, BatteryML and related open-source time-series implementations. We thank their authors and the maintainers of the public battery datasets.

## License

Code is released under [`LICENSE`](LICENSE). Dataset files and third-party pretrained models remain subject to their own licenses.
