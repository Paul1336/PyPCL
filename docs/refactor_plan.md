# PyPCL Refactor 計畫（2026-11-22，`analysis` 分支）

依據：完整讀過 `src/`、`scripts/`、`logs/*/launch*.py`、`verify_scripts/`（4 路平行 review + 主幹 pipeline 親讀）。
行號皆指 `analysis` 分支 HEAD（49c0492）。

## 0. 總覽

| 指標 | 現況 |
|---|---|
| registry 演算法名 | 507（435 個是 factory closure；hparams 508 個 key，多 1 個殭屍） |
| `runners.py` | 2118 行，72 個函式，同一個訓練骨架複製 ~20 次 |
| 訓練 epoch 函式 | PiCO 族 17 個、ComCo 族 6 個、其他 6 個，分散在 7 個檔 |
| 同一 loss 的重複實作 | MCL-LOG ×6、SCL-NL ×5、confidence-weighted CE ×10、padded→dense mask ×7 |
| launcher | 4 代、~35 個，`/proc` parser 6 種寫法，hard-code 已不存在的 worktree 路徑 |
| 預估可刪減 | runners −900、訓練迴圈 −1100~1300、loss −400、launcher 整套換成 1 個 |

**最重要的結論：先修 bug、先建 golden test，再動結構。** 有 3 個 bug 已經影響（或會影響）實驗結論。

---

## 1. 先修的 bug（Phase 0，獨立於 refactor）

| # | 嚴重度 | 問題 | 位置 | 影響 |
|---|---|---|---|---|
| B1 | **高（影響已發表的結論）** | `ProdenLoss` 每步用 `candidate_mask = (conf > 0)`（`src/proden_loss.py:125`）推 candidate set，而不是真正的 Y。init 的支撐集因此被永久保留：`biased_oracle` → 整個訓練 candidate 只剩 {真實, 1 錯誤}（oracle 洩漏）；`biased_partial_random` → {真實 + wf}；`uniform_all`/`biased_all`/`biased_random_all` → candidate 限制消失 | proden_loss.py:125 | experiments_summary **C2**（BiasedInit +15.65）、**C4**（集中度方向相反）、biased_weight_sweep（BiasedAll→chance）、**D2**（BiasedRandAll N 越大越差）、**E2** 的 PRODEN-Biased* 結論都需要重新解讀/重跑。PiCO 的 PartialLoss 用真正的 `batchY`，不受影響 |
| B2 | 中（目前無資料受影響） | `--holdout_frac>0` 時 PiCO-CLS（runners.py:1765）和 SoLar（runners.py:1918）沒用 `train_idx`，會在 held-out 上訓練 | runners.py | 已查：所有 holdout run 都沒有這兩個演算法 → 尚無污染 |
| B3 | 中 | `IMAGE_ONLY_ALGORITHMS`（runner.py:23-41）只列舊名字，漏掉 PiCO-MCL-Fixed、*-Weighted-*、*-Uni/Mix-*、Hybrid、Swap*、Prune、PCD-*、PiCOPlus → 在非影像 dataset 上不會被跳過，直接 KeyError | runner.py | 只影響非 CIFAR dataset |
| B4 | 中 | cub200 / sun397 / clpl-lost 宣告 `supports_pico_family=True`，但它們的 loader builder 從不建 `pico`/`comco` loader；clpl-lost 還回傳 `orig_targets=None` | datasets/cub200.py:111, sun397.py:107, clpl_tv.py:164 | 同上 |
| B5 | 低 | trace JSON 檔名 `{alg}_k{k}_seed{s}.json` 不含 C（runners.py:942/1044/1351/2095；docstring 說有）→ 同 run dir 不同 C 會互蓋 | runners.py | 已查：目前沒有 run 混 C |
| B6 | 低 | `FixedMCL*` 不 clamp m（fixed_mcl_losses.py:56/73/90）、`wu_loss.py:52` 除以 (C−k) → `--only_q` 下 \|Ȳ\|=0 或 k=C 會 inf/NaN | | varq 在 k=C−1 時 ~35% 樣本 \|Ȳ\|=0 |
| B7 | 低 | 非 CIFAR dataset 套用 CIFAR normalization（cub200:113, sun397:109, clpl_tv:164）；SoLar/biasvariance/switch_diag/clsloss_solo 完全忽略 spec 的 norm | | |
| B8 | 低 | biasvariance 用 Python `hash()` 當 resample seed（biasvariance.py ~262），每個 process 不同 → 不可重現 | | `src/analysis/gradients.resampled_bias_variance` 已改成 deterministic |
| B9 | 低 | `training_pipelines.run_comco_training`、`model_setup.setup_comco` 各被定義兩次（後者蓋前者） | | 舊 workflow，刪掉即解 |
| B10 | 文件 | PiCO-SC 在 optimizer.step **前** 更新 confidence（engine.py:200），但 `pico_cls_loss.py:74` docstring 說要 step 後；run_pico_cls 是 step 後 → 兩個都叫 PiCO-CLS 但順序不同 | | 保留行為，改文件 |

B1 的修法：`ProdenLoss` 另存 `cand` mask（同 `PiCOCLSLoss` 的 `pl_mask`），更新時用 `cand` 而非 `conf>0`。**修完後 PRODEN-*Init 系列的數字會變**，舊結果保留但在 experiments_summary 標註。

---

## 2. Refactor 的安全網（Phase 1，動結構前必須完成）

1. **Golden snapshot test**：每個 family 挑一個代表（PRODEN、MCL-LOG、SCL-NL、PiCO、PiCO-Fixed、PiCO-Oracle、ComCo-Fixed、PCD-PiCO、PCD-ComCoMCLNL、PCD-PiCOPlus、一個 bandit `--switch_diag`、SwapComCo、Prune），固定 seed、`torch.backends.cudnn.deterministic=True`，跑 N 個 step，存：參數 hash、confidence、prototypes、queue、前幾個 batch 的 loss、diag CSV 第一列。新舊比對容忍度 1e-6（現行 check_pcdiag 也只到 1e-4，GPU 原本就不是 bit-for-bit）。
2. **Registry 不變式**：`len(ALGORITHMS)==507` 且名字集合不變（results.csv 是用名字做 resume key）。
3. **Launcher dry-run golden**：每個 `logs/*/launch.py` 的 `cmd_for()` argv 清單存檔，新 sweep 工具展開後要一致。
4. **verify_scripts → pytest**：`check_analysis / check_clsloss_swap / check_clsloss_solo_grads`（CPU，秒級）+ `check_pcdiag / check_pcdiag_plema`（`@pytest.mark.gpu`）。加 `conftest.py` 取代 `sys.path.insert(0,'.')`。
5. **保留 RNG 消耗順序**：model `__init__`（encoder_q → encoder_k → queue randn → queue_pseudo randn）、Hybrid 建了又丟的兩個 encoder、init_conf 與 model 建立的先後（`_run_pico_weighted_variant` 先 init 再 model，`_run_swap_variant` 相反）、oracle `randperm` 位置、MixA 的 `np.random.beta`。

---

## 3. 目標架構

```
src/
├── core/
│   ├── context.py        CellContext(run_dir, C, k_label, k_mean, seed, dataset_spec, diag: DiagConfig)
│   │                     取代 raw_cfg['_detail'/'_current_k'/'_current_seed'/'_results_dir'/'_switch_diag'/'_dataset_spec']
│   ├── config.py         PipelineConfig 由單一欄位表生成（argparse default、getattr、run_config.json 三處合一）
│   ├── io.py             TraceWriter(subdir)（檔名含 C）、RewritingCsvLog、seed-scoped detail 路徑
│   └── seed.py           set_seed + 具名 rng(seed, purpose)
├── data/
│   ├── registry.py       DatasetSpec（cifar100-subset 變成一般條目；norm stats；views_supported 由 loader 能力推導）
│   ├── sources/          cifar100 / cifar10 / mnist_family / uci / text20news / hf_images(cub+sun 共用) / pll_mat / clpl_tv / split
│   ├── labels/           純函式 (true_y, C, rng) -> CandidateSet：uniform(fixed_k, variable_q, bimodal) / hierarchical / biased / noise / complement / provenance
│   ├── transforms.py     NORMS 表；standard / weak / strong / solar_pair
│   ├── datasets.py       PartialLabelDataset(views=..., return_index=True, reader=array|path|feature) + EvalDataset
│   └── loaders.py        build_loaders -> {'pl','cl','views','test','holdout','train_idx'}；單一 collate（dense mask）
├── losses/
│   ├── batch.py          LabelBatch(logits, index, cand[B,C], true=None)；.comp/.k/.m；masks.py 唯一的 padded↔dense
│   ├── base.py           ClsLoss: per_sample(b) / forward / on_epoch(ep, epochs) / pre_step(b, signal, gate) / post_step(b)
│   ├── pll/              uce, clpl, wu
│   ├── cll/              MCL(base=log|mae|exp, scale=none|feng_eq12|c1_over_cm) 一個類別涵蓋 MCL-LOG/-Fixed/MCL-NL/ComCo-orig；SCLNL；OP/OPW；CPE
│   ├── confidence/       ConfidenceStore(N,C)（registered buffer + 獨立 cand mask，修 B1）；RenormUpdate / EMAOneHot / EMASoft；
│   │                     WeightedCE(store, rule, source=self|prototype|classifier, gate_epoch) -> PartialLoss/ProdenLoss/PiCOCLS/PRODENLoss/PiCOEMA 變成 config
│   ├── composite.py      Mix(a, b, alpha|tensor) / Switch / Unified(rho) / CandidatePrune
│   ├── solar.py
│   └── contrastive/      masked_infonce(anchor, pool, pos, denom, tau, weights) + moco_infonce（保留 CE 寫法）
│                         mask builders: label_eq / comco_masks / picoplus；transforms: oracle_remove_fp / oracle_add_tp / oracle_pos / oracle_neg / cand_filter
├── models/
│   ├── backbones.py      SupConResNet 等；encoder 統一回傳 EncOut(logits, feat, h?)
│   ├── pico.py           PiCOModel(track_true, track_cand, oracle_protos, proto_update='loop'|'vectorized', return_h)
│   │                     取代 PiCOOracleModel / PiCODiagModel；forward -> FwdOut dataclass；model.predict(x)
│   ├── comco.py          ComCoModel(prototypes, track_true, pl_ema_m) 取代 ComCoWeighted / ComCoDiag
│   └── queue.py          共用 momentum update / enqueue / prototype EMA
├── train/
│   ├── steps.py          pico_step / comco_step / simple_step / solo_step（顯式 hook 點）
│   ├── hooks.py          SelectionStats / OracleCorrection / PairStats / GradStats / Mixup / AlphaTrace / Detail(ckpt,tsne,conc,bv,knn)
│   ├── loop.py           run_epochs(ctx, step, hooks)：計時 / ETA / gc / 定期 eval / 最終 eval 只有一份
│   └── bandit/           policies(Fixed, Sched, SchedFrozen, Paired, Cont, PairedLong, TwoArm, Plateau, SFlip) + rewards + lookahead
├── algorithms/
│   ├── spec.py           AlgorithmSpec(name, paradigm, build, optim, needs_views, needs_holdout, aliases_of)
│   ├── families/         simple / proden / pico / comco / solar / frameworks(build_framework) / switch / swap
│   ├── sweeps.py         唯一產生名字的地方（alpha / EMA / AB / W / Wf / N / bandit specs）
│   └── __init__.py       ALGORITHMS = build(...)；assert 名字唯一、與 legacy 名單一致
├── analysis/             （已完成）診斷純函式 + 事後分析；hooks 直接呼叫它
├── pipeline/             runner.py（只剩 C×k×seed×alg 迴圈）、results.py（加 config_hash 欄）
└── sweep/                spec.py / scheduler.py / procscan.py / results_view.py   （取代所有 launcher）
scripts/
├── run_pipeline.py       subcommand 拆到 src/pipeline/cli/{run,report,plot,detail_plots}.py
├── sweep.py              run / status / dry-run / report / adopt / pause / resume
└── download_data.py
experiments/<id>.yaml     每個實驗一份宣告式 spec
tests/                    pytest（golden + 等價檢查）
```

---

## 4. 逐元件：問題 → 目標

### 4.1 Config 與 per-cell context（最先做，其他元件都依賴它）
- **問題**：每個 runner 共用同一個 11 參數位置簽名；每 cell 資訊塞進共享的 yaml dict `raw_cfg['_…']`（6 個 key，在 runner.py:122-246 設定，在 runners/detail/pcdiag 讀 ~30 處）。`_current_k` 在 fixk / varq / bimodal 下意義不同；`_results_dir` 兩處預設值不一致（`'results'` vs `'results/_pcdiag'`）。PipelineConfig 預設值寫三遍（argparse、`config.py:67-103` 的 getattr、`runner.py:104-117` 的 run_config.json 清單）。
- **目標**：`CellContext` 顯式傳入；`RunContext` 取代 11 參數（先加 adapter，舊 runner 不用一次改完）；`PipelineConfig` 由欄位表生成。
- **附帶修**：`run_config.json` 每個 worker 覆寫且缺 `only_k/only_c/algo/config/hparams_override/eval_split` → 改成每 cell 記錄完整參數。

### 4.2 Results schema
- **問題**：dedup key `(dataset, C, k, algorithm, seed)` 不含 `cl_bias / noise_eta / holdout_frac / batching / config / hparams_override`，隔離完全靠「每個設定一個 run_name」的慣例；k 用負數 sentinel 編 q 與 bimodal（三種編碼散在 8 個 launcher + 3 個 summarize）。`merge_shards` 每個 cell 後非原子地重寫 `results.csv`。
- **目標**：shard 列加 `config_hash` + `k_mean`/`q` 欄（向後相容：舊列 hash 為空）；sentinel 編解碼只在 `results.py` 一處；`results.csv` 用 tmp+rename 原子寫；事後分析一律讀 shards（`src/analysis/io.load_results` 已這樣做）。

### 4.3 Losses（`src/losses/`）
- **問題**：MCL-LOG 6 份、SCL-NL 5 份、confidence-weighted CE ~10 份；三種輸入格式（padded −1 list / dense Y / dense Ȳ / 只吃 index）；4 種 epoch schedule API、4 種 confidence 更新掛點；buffer 名稱與是否 register 不一致（`conf` / `confidence` / `partial.confidence` / `partial_loss.confidence`，影響 `.to()`、`state_dict`、deepcopy）；`clsloss_swap.OPLoss` 與 `op_loss.OPLoss` 同名。
- **目標**：見架構 `losses/`。統一 `LabelBatch` 輸入與 `ClsLoss` 協定（沿用 clsloss_swap 已有的 `set_epoch/pre_update/forward(logits, idx, Y)`，最成熟）。舊類別名保留為 alias。
- **刪除**：`proden`（legacy 無 lag 版）、`MCL_MAE/EXP`、`FixedMCLMae/Exp`（無使用者）、`cour_loss.py`（被 UCELoss 取代）。
- **刪減**：~400 行。

### 4.4 Models 與 contrastive
- **問題**：`PiCOOracleModel` 是 `PiCOModel` 的複製貼上而非子類別（pico/model.py:74-169），所以 `evaluate_model` 和 detail 要 isinstance 白名單；prototype EMA 迴圈 3 份；`_momentum_update_key_encoder` 2 種簽名、`_dequeue_and_enqueue` 3 種；forward 位置參數互不相容（`ComCoWeightedModel` 用 ComCo 的呼叫法會把 args 塞進 partial_Y）；回傳 tuple 長度與順序不同（`PiCOOracleModel` 第 3 個位置是 **true** pool，誤拆會靜默用 oracle mask 訓練）；`eval_only` 預設不一致（Hybrid/Solo 忽略）。PiCO 與 ComCo 的 queue 初始化不同（normalize dim 0 vs 1；queue_pseudo randn vs zeros）——**保留**。
- **目標**：見架構 `models/`。`FwdOut` dataclass + 一個版本的 tuple shim；`model.predict(x)` 取代白名單。contrastive 全部收斂到 `masked_infonce` + mask builders/transforms（check_pcdiag 已證明 comco_masks 與 ComCoContrastiveLoss 的 mask 完全一致）；**MoCo warm-up 保留 `F.cross_entropy` 寫法**，否則 plain PiCO/MOCO/MCL/SC 會有 1e-7 級差異。
- **刪減**：~250 行。

### 4.5 訓練迴圈（最大塊）
- **問題**：~20 個 epoch 函式共用同一骨架：`forward → (gated) conf update → L_cls → masks → L_cont → step`。各變體只在 6 個軸之一不同：cls 呼叫簽名、warm-up 時 L_cont 處理（omit vs MoCo）、conf 更新來源、mask transform（oracle）、統計收集、per-batch alpha。`detail.py` 的 4 個 `*_with_*stats` 是訓練迴圈的「加料複本」（oracle 修正數學逐字複製自 oracle_pico_engine）；`detail` 版 PiCO-Fixed 不支援 conf_source/conf_hard → runners.py:575 直接 `NotImplementedError`。
- **目標**：`pico_step` / `comco_step` + hook（順序必須與現行一致：forward → `cls_loss.pre_update(gate)` → `alpha_fn(Y)` → L_cls → masks → `mask_transform` → `before_loss`(stats) → L_cont → `extra_loss`(mixup) → step → `after_step`(stats)）。對應：

  | 現有 | 變成 |
  |---|---|
  | PiCO / MOCO | `warmup_cont='moco'` / `cont_mask='none'` |
  | PiCO-Fixed(+所有 sweep)、MCL(-Fixed)、SC、Weighted、Switch、SwapPiCO | 同一 step，只換 loss 物件 |
  | Oracle / Oracle-Add | `mask_transform=OracleRemoveFP/OracleAddTP` + `OracleCorrectionStatsHook` |
  | detail 版各迴圈 | 原迴圈 + `SelectionStatsHook`（CSV 格式不變） |
  | PCD-PiCO / PCD-ComCo | flags + `PairStatsHook` + `GradStatsHook`（需 `h`）+ `MixupHook`（統計直接呼叫 `src/analysis`） |
  | ComCo / ComCo-Fixed / Weighted / Switch / SwapComCo | `comco_step`，只換 loss 物件；alpha_trace → `AlphaTraceHook` |
  | PiCO+ | 保持獨立演算法，只共用 masked_infonce / moco_infonce / prototype EMA / conf-EMA schedule |
  | Solo | `cont=None` 路徑 |

- **注意**：prototype 更新 `proto_update='loop'` 為預設（vectorized 版與迴圈差 ~1e-6，只給 PCD/PiCO+ 用）；oracle precision 現行**含 self-pair**，保留為 `count_self=True` 旗標以重現舊結果。
- **刪減**：淨 −1100 ~ −1300 行（engine.py 的 PiCO/ComCo 部分、fixed_pico_engine、oracle_pico_engine、fixed_comco_engine、detail.py 352-900、clsloss_swap 425-486、solo_switch 32-52）。

### 4.6 Runners / registry（`runners.py` 2118 → ~1100）
- **問題**：
  - PiCO 族 9 個 runner 近乎複本（差異只在 prot_start、init、model、loss、epoch fn），hook 組合不一致（看起來是意外：pico_mcl 沒呼叫 `set_conf_ema_m`、PiCO 和 PiCO-Oracle 沒有 concentration/kNN）。
  - `comco_args` dict 手寫 10 份（798, 887, 978, 1086, 1113, 1149, 1806, 1860, 1962, 2050）；`_pico_args + prot_start_fixed` 11 份。
  - 舊 `_run_comco_weighted_bandit` ≡ generic `ComCo-Mix-BanditPaired`（只有 TwoArm 模式獨有）。
  - sweep factory ~300 行全是「綁 kwargs + 算名字」。
  - 名字在 4 處各自生成（runners、hparams 手打清單、pll_init、run_*_sweep.py），已漂移（`PCD-ComCo-OracleProto` 只在 hparams）。
  - hparams 508 個 entry 只有 `_ADAM`/`_SGD` 兩種值，卻在 import 時拉進 pcdiag、clsloss_swap → 與 pcdiag/picoplus 形成循環 import。
  - 28 個 alias（`-EMA100` ≡ 無後綴；`PRODEN-EMA000` ≡ 基底）。
  - paradigm 由名字前綴猜（Hybrid/Solo 被誤標 CLL）。
- **目標**：見架構 `algorithms/`。`AlgorithmSpec` 帶 `optim / needs_views / needs_holdout / aliases_of` → 取代 hparams 名單、`IMAGE_ONLY_ALGORITHMS`（修 B3）、前綴猜 paradigm。共用 `pico_args(fixed=)`、`comco_args(borrow_pico=)`、`build_framework(name, cls_loss)`（給 switch/swap/prune 共用）、`indexed_loader(ctx)` 一律尊重 `train_idx`（修 B2）。`_run_generic_bandit` 161 行狀態機拆成 Policy + Reward 物件。
- **新增演算法只動 1 個檔**（目前 3-5 個）。

### 4.7 Data（`src/data/`）
- **問題**：`prepare_cifar100_subset`（cifar100_subset.py:184-396）一個函式做 8 件事（驗證、讀檔、選類、7 路 generator 分派、noise、補集、JSON log、console）；`ComparisonDataGenerator` 8 個 generator 重複同一個「逐樣本 np.delete+choice」迴圈；`PicoDataset` 與 `ComCoDataset` augmentation 逐位元組相同；one-hot candidate 建 7 次；CIFAR normalization hard-code 在 10+ 處；label 生成與 holdout split 都用 global `np.random`（holdout 集合取決於前面 label 生成吃了多少 RNG）；`num_classes = len(set(labels))`（類別缺席就錯）；runners 自己再建 Dataset（`_IndexedDataset`、`SoLarDataset`、`WeakOnlyDataset`、`_DeterministicEvalDataset`、3 個 tensor builder）。
- **目標**：見架構 `data/`。label 生成改純函式 + 具名 rng（**會改變抽樣結果** → 以新 `data_version` 欄位區分，舊 cell 不重算；或保留 `legacy_global_rng=True` 預設直到你決定）。`pico`/`comco` 合併成一個 `views` loader，ComCo 自己取 `1−cand`。

### 4.8 診斷（`detail.py` 1735 行 + pcdiag/picoplus 的統計）
- **問題**：detail.py 混了 instrumented 訓練迴圈、CSV writer、12 個繪圖函式；CSV 全 append（重試會重複列）、路徑不分 seed（只有第一個 seed 做，per-cell launcher 傳 `--detail` 給 seed 43/44 會覆寫）；三種 pair precision 用不同 pair 集合、名字卻相近。
- **目標**：迴圈部分移到 hooks（4.5）；統計計算呼叫 `src/analysis`（已完成並驗證等價）；writer 改 `RewritingCsvLog` + seed-scoped 路徑；繪圖移到 `src/pipeline/plots/`。CSV 欄位名不變，舊 plot 指令照用。

### 4.9 Launcher → `src/sweep` + `experiments/*.yaml`
- **問題**：4 代 ~35 個 launcher。G2（`logs/*/launch.py`，16 個）共用骨架但全複製：bootstrap、q 編碼、worker-id 表、`done_set`（9 份逐字相同）、`/proc` 掃描（17 份 6 種 parser，**3 份仍有 11-06 那個會 crash 的寫法**：pcdiag_1103/launch_mcl.py:97-99、swapbandit_1027:113-117、banditrho_1028:79-81；6 份 `int(CUDA_VISIBLE_DEVICES)` 未保護）、log 格式（已被 smallk 當 API 解析）、重試只看 rc（沒有「rc=0 但沒寫結果」檢查）、重試時 log 用 `'w'` 開把錯誤訊息洗掉、子程序 file handle 不關、SIGSTOP/PID hard-code/阻塞式 chaining。全部 hard-code 已不存在的 worktree 路徑。
- **目標**：`scripts/sweep.py` + 宣告式 spec（grid、k_mode、條件 k grid、include/exclude、tier 排序、resources(mem 表 glob)、retry+backoff、`success: row_written`、`start_new_session` 永遠開、`wait_for`/`then` 非阻塞 chaining、`shard i/N`、外部結果排除清單、worker_id 持久化）；`procscan` 唯一、容錯、用 `SWEEP_CELL` env 而非 argv 辨識；pause/resume/drain 用 control file 取代 SIGSTOP。G1.5 的 `run_ema_sweep.py` 是最成熟的設計，以它為底。
- **驗收**：每個舊 launcher 的 cell 集合、worker id、順序與 `dry-run --emit-argv` 完全一致後才刪舊檔。

### 4.10 CLI（`run_pipeline.py`）
- **問題**：14 個 subcommand（docstring 寫 3 個）、`run` 有 37 個 flag、`main()` 230 行 if/elif、`report` 115 行 inline；`--gpu_id/--num_gpus` 被拿來當 shard id；`--algorithms` + `--algo` 冗餘；`--concentration`、`--knn_eval` 等 8 個 flag 沒有任何 launcher 用。
- **目標**：subcommand 拆模組；`--shard_id`；`--diag detail,tsne,concentration,knn,biasvariance,switch` + yaml `diag:` 區塊取代 12 個診斷 flag；`report` 改用 `src/analysis/stats`。

### 4.11 刪除 / 封存
| 對象 | 處理 |
|---|---|
| 舊 workflow：`scripts/run_experiment.py`、`src/{args,data_setup,training_pipelines,saving,plotting,clcifar}.py` | 新 pipeline 都沒 import → 刪（`model_setup.setup_solar` 先搬進 SoLar family；`engine.py` 是新 pipeline 核心，拆到 train/） |
| `scripts/launch_*.py/.sh`、`run_*_batch.sh`、`wait_and_launch_*.sh`、`organize_*.sh`、`logs/*/launch*.py`、`handoff.sh`、`c10_chain.sh` | port 成 spec 並通過 dry-run 比對後刪；`launch_clsloss_swap.build_plan` 保留（Nano4 shard 對應） |
| `logs/*/*_snapshot.py`、code_diff patch | 保留（provenance），之後由 sweep 自動 snapshot |
| `scripts/legacy/`（58 檔） | 移出 repo 或打 tag 後刪 |
| `verify_scripts` paper 重現類 | 保留為 CLI；共用的 `ResNet34CIFAR`、dataset shim、`set_seed`×5、CSV helper×6 抽成 `verify_scripts/common.py` |

---

## 5. 執行順序

| Phase | 內容 | 依賴 | 驗收 | 規模 |
|---|---|---|---|---|
| **0** | 修 B1–B10；experiments_summary 標註受 B1 影響的結論 | — | 新增單元測試；B1 修後重跑 PRODEN-BiasedInit 一個 cell 確認行為 | 小 |
| **1** | 安全網：golden snapshot、registry 不變式、launcher dry-run golden、pytest 化 | — | 全綠 | 中 |
| **2** | `CellContext` / `RunContext` adapter、config 欄位表、results schema（config_hash、原子寫）、TraceWriter | 1 | golden 不變 | 中 |
| **3** | `src/losses/`（含 ConfidenceStore）+ alias | 1 | check_clsloss_swap + golden | 中 |
| **4** | `models/` + contrastive 收斂 | 3 | check_pcdiag + golden | 中 |
| **5** | `train/steps + hooks + loop`；detail/pcdiag 統計改呼叫 `src/analysis` | 3, 4 | golden（1e-6）、diag CSV 第一列一致 | **大** |
| **6** | `algorithms/` 拆 families + sweeps + spec；刪舊 runners | 2, 5 | 507 名字不變 + golden | 大 |
| **7** | `src/data/` | 2 | 預設 `legacy_global_rng` 下 golden 不變 | 中 |
| **8** | `src/sweep/` + 把 16 個 G2 launcher port 成 spec | 2 | dry-run argv 完全一致 | 中 |
| **9** | CLI 拆分、刪舊 workflow/legacy、更新 CLAUDE.md | 全部 | — | 小 |

Phase 3/4、7、8 互相獨立，可以平行（各開 worktree）。

---

## 6. 需要你決定的事

1. **B1 修正後**：PRODEN-*Init 系列要重跑嗎？還是只在 experiments_summary 標註「該結果其實是 candidate 縮減 / 消失的效果」？
2. **舊 workflow**（`run_experiment.py`、完整 CIFAR-10/20/CLCIFAR）還要保留嗎？CLAUDE.md 還把它列為第二個 workflow。
3. **可重現性門檻**：接受新舊在 deterministic 模式下差 ≤1e-6（現行 check_pcdiag 是 1e-4），還是要求 bit-for-bit（會讓 prototype EMA、MoCo loss 等必須保留舊寫法）？
4. **資料 RNG**：label 生成改具名 rng 會讓之後的抽樣跟舊 run 不同，要切換（新 run 用新 `data_version`）還是永遠保留 global RNG？
5. **28 個 alias 名字**與舊 bandit（`ComCo-Weighted-BanditPaired`）：保留為 alias 或直接停用？
6. **兩套 results**（主 checkout 9 月以前 + analysis worktree 10 月以後）要合併嗎？
