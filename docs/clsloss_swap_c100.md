# clsloss_swap_c100：PiCO / ComCo × 全部 cls loss，C=100

> 建立於 2026-10-19。在**另一台 server** 上跑（那台不能跑 Claude）。跑完把結果帶回來，
> 開新的 Claude session 貼上，照本檔最後一節記錄與解析。

## 1. 目的

把 cls-loss swap 實驗（`clsloss_swap_1006`，C=20）延伸到 C=100：PiCO-Fixed 跟 ComCo-Fixed 兩個框架，
各接上全部 13 種分類損失，看每種 loss 在類別數變大時的表現。背景：

- `hard_c100_1017` 發現 `SwapComCo-SCLNL`（多互補標籤平均版 SCL-NL）在 C=100 完全學不起來
  （k=1 也只有約 8%），原因是梯度只有 CE 的約 1e-4。
- 對照 ComCo 論文後確認，多互補標籤時 ComCo 用的是 **MCL-NL**，所以新增了不縮放的 `MCLNL`
  （見 `docs/comco_explanation.md` 檔首 2026-10-19 更正）。
- 這次要回答：在 C=100 哪些 loss 還能學、`MCLNL` 是否修好 ComCo，以及 MCL 三種 scaling
  （`MCLLOG` / `MCLLOGFixed` / `MCLNL`）的差別。

## 2. 實驗設定

| 項目 | 值 |
|---|---|
| Run name | `clsloss_swap_c100` |
| 資料 | CIFAR-100 全部 100 類（`cifar100-subset`, C=100），uniform constant-k 候選集，clean |
| k | 1, 5, 10, 25, 50, 75, 99 |
| Seeds | 42, 43, 44 |
| 演算法 | `SwapPiCO-<loss>`、`SwapComCo-<loss>`，loss ∈ 13 種（見下表），共 26 個 |
| 訓練 | 200 epochs，batch 512，Adam 3e-4，wd 1e-4，**不切 held-out**（跟 `clsloss_swap_1006` 同一套 protocol） |
| 總 cell 數 | 26 × 7 × 3 = **546** |

### 跑的順序（launcher 自動照這個順序排）

| Phase | 內容 | Cells |
|---|---|---|
| 1 | 優先 loss × 兩個框架 × 7 個 k，**seed 42** | 112 |
| 2 | 優先 loss，seed 43、44 | 224 |
| 3 | 其餘 loss（Wu2022、SoLar、MCLLOG、OP、OPW），seed 42–44 | 210 |

優先 loss：UCE、CLPL、PRODEN、PiCOPartial、PiCOCLS、MCLLOGFixed、MCLNL、SCLNL。

### Loss 一覽

| # | Loss | 公式 / 意義 | 出處（/home/Paul/papers） |
|---|---|---|---|
| 1 | UCE | −(1/k)Σ_Y log p | PRODEN-naive |
| 2 | CLPL | squared-hinge CLPL | Cour 2011 |
| 3 | Wu2022 | proper PLL estimator | 無 PDF |
| 4 | PRODEN | 每個 batch 更新權重 | PRODEN |
| 5 | PiCOPartial | prototype 的 EMA one-hot（PiCO 原生） | PiCO |
| 6 | PiCOCLS | classifier 的 EMA one-hot | repo 自創 |
| 7 | SoLar | 簡化版 SoLar | 無 PDF |
| 8 | MCLLOG | L_LOG × (C−1)/k | 係數無出處 |
| 9 | MCLLOGFixed | L_LOG × 2(C−1)/m | Feng Eq. 12 |
| 10 | MCLNL | L_LOG 不縮放（ComCo 多互補標籤時的 loss） | ComCo §5.4.2 |
| 11 | SCLNL | SCL-NL 對 m 取平均 | 只在 m=1 時忠實 |
| 12 | OP | order-preserving | 無 PDF |
| 13 | OPW | 加權 OP | 無 PDF |

## 3. 在新 server 上怎麼跑

### 3.1 取得程式碼

程式碼在 `switch-analysis` 分支（含 `scripts/launch_clsloss_swap.py`、`scripts/summarize_clsloss_swap.py`、
`src/clsloss_swap.py` 的 `MCLNL`）。兩種方式擇一：

```bash
# A. 從 GitHub（需先在原 server 上 git push origin switch-analysis）
git clone <repo-url> PyPCL && cd PyPCL && git checkout switch-analysis

# B. 直接從原 server 複製（不含 results/logs/data）
rsync -a --exclude results --exclude logs --exclude data --exclude plots \
    <原server>:/nfs/home/Paul/PyPCL/.claude/worktrees/switch-analysis/ PyPCL/
```

### 3.2 環境

```bash
conda create -n PyPCL python=3.10 -y && conda activate PyPCL
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`requirements.txt` 釘的是 torch 1.13.1+cu117。A100 以前的卡沒問題；**H100（sm_90）需要 cu118 以上**，
那就改裝 cu118 或更新版的 torch。CIFAR-100 第一次跑會自動下載到 `--data_dir`（預設 `./data`）。

### 3.3 先跑一個 smoke test（約 5 分鐘）

```bash
python scripts/run_pipeline.py run --run_name smoke_c100_tmp \
    --algorithms SwapComCo-MCLNL SwapPiCO-PiCOPartial --c_values 100 --only_k 1 \
    --seeds 42 --epochs 2
# 看到兩行 "DONE ... acc=..." 就 OK
rm -rf results/smoke_c100_tmp
```

### 3.4 正式啟動

`--gpus` 填這台要用的 GPU 編號；`--slots` 是每張卡同時跑幾個。每個 run 約佔 **4.3 GB** 顯存
（RTX A5000 24 GB 上跑 4 個約 17 GB），依顯存決定 slots。

```bash
# 背景執行，登出也不會停
nohup python scripts/launch_clsloss_swap.py --plan c100 --gpus 0 1 2 3 4 5 6 7 --slots 4 \
    > logs/clsloss_swap_c100_launcher.out 2>&1 &
```

- **可以中斷後續跑**：重跑同一行指令，已完成的 cell 會自動跳過（看 `results/clsloss_swap_c100/shards/`）。
  中斷時正在跑的 cell 會從頭重跑。
- 想先看排程不啟動：`python scripts/launch_clsloss_swap.py --plan c100 --dry_run`

### 3.5 監看進度

```bash
python scripts/launch_clsloss_swap.py --plan c100 --status        # 各 phase 完成數
tail -f logs/clsloss_swap_c100/launcher.log                         # START / END 紀錄
grep "END rc=[^0]" logs/clsloss_swap_c100/launcher.log              # 失敗的 cell（rc≠0）
python scripts/summarize_clsloss_swap.py results/clsloss_swap_c100 --plan c100   # 目前結果表
```

### 3.6 預估時間

在原 server（RTX A5000，每卡 4 個並行）上，C=100 每個 run 約 **5–5.5 小時**。
總量約 546 × 5.2 ≈ 2,840 slot-小時：

| 並行 slot 數 | Phase 1（seed 42） | 全部 |
|---|---|---|
| 16（4 卡 × 4） | ~36 h | ~7.4 天 |
| 32（8 卡 × 4） | ~18 h | ~3.7 天 |

較快的 GPU 會更短。

## 4. 結果會產生在哪裡（相對於 repo 根目錄）

| 路徑 | 內容 |
|---|---|
| `results/clsloss_swap_c100/shards/worker<id>.csv` | 每個 cell 一個 shard（id = 排程中的編號 0–545） |
| `results/clsloss_swap_c100/results.csv` | 合併後的總表（每個 cell 跑完會自動合併；launcher 結束時再合併一次） |
| `results/clsloss_swap_c100/run_config.json` | 設定快照 |
| `logs/clsloss_swap_c100/launcher.log` | 排程紀錄（START / END rc= / ALL DONE） |
| `logs/clsloss_swap_c100/<algorithm>_k<k>_s<seed>.log` | 每個 cell 的訓練輸出 |
| `logs/cifar100_subset/*.json` | 每個 cell 的資料生成紀錄 |

CSV 欄位：`dataset,total_classes,k,algorithm,seed,final_accuracy,epochs,training_time_s,timestamp,git_commit`。
`final_accuracy` 是**最後一個 epoch** 的 test accuracy（%）。

## 5. 跑完（或跑到一半）要帶回來什麼

**方式 A（建議）：把結果目錄複製回原 server**，新 session 就能直接讀檔：

```bash
rsync -a results/clsloss_swap_c100 logs/clsloss_swap_c100 \
    <原server>:/nfs/home/Paul/PyPCL/.claude/worktrees/switch-analysis/import_c100/
```

然後在新 session 說：「C100 結果在 `/nfs/home/Paul/PyPCL/.claude/worktrees/switch-analysis/import_c100/`」。

**方式 B：直接貼文字**。在新 server 跑下面兩個指令，把輸出整段貼給 Claude：

```bash
python scripts/summarize_clsloss_swap.py results/clsloss_swap_c100 --plan c100 --raw
grep -c "END rc=0" logs/clsloss_swap_c100/launcher.log; grep "END rc=[^0]" logs/clsloss_swap_c100/launcher.log
```

`--raw` 會在最後附上每一列原始資料（546 列約 60 KB），Claude 可以用它重算任何統計。

## 6. 給 Claude（新 session）的處理說明

收到結果後依序做：

1. **檢查來源**：row 數、seeds、epochs 都是 200、`git_commit` 是 `switch-analysis` 分支上含
   `MCLNL` 的 commit；用 `--plan c100` 的 COMPLETION 段確認各 phase 完成數，列出缺的 cell 與失敗的 rc。
2. **存檔**：方式 B 的話，把 RAW CSV 存成
   `/nfs/home/Paul/PyPCL/.claude/worktrees/switch-analysis/results/clsloss_swap_c100/results.csv`
   （方式 A 則放在 `import_c100/`），之後用 `scripts/summarize_clsloss_swap.py` 重算。
3. **記錄**：依 `/log-start`、`/log-finish` 的流程，用 `scripts/log_experiment.py`
   登記 `clsloss_swap_c100`（date_start 用 CSV 最早的 timestamp；全部 546 cell 完成才算 completed，
   否則 status 用 running 或 stopped）。
4. **解析**（final_accuracy，3 seeds 的 mean±std；C=100 的 chance = 1%）：
   - 兩個框架各一張 loss × k 表，以及每個 k 的最佳 loss。
   - **`MCLNL` 是否修好 ComCo**：`SwapComCo-MCLNL` vs `SwapComCo-SCLNL`（`hard_c100_1017` 中 k=1 約 8%）。
   - **MCL 三種 scaling**：`MCLLOG`（(C−1)/k）vs `MCLLOGFixed`（2(C−1)/m）vs `MCLNL`（不縮放），
     注意 k 小時 MCLLOG 係數大、k 大時 MCLLOGFixed 係數大。
   - **框架效應**：同一個 loss 在 PiCO vs ComCo 的差距。
   - **跟 C=20 對照**：C=20 的結果在
     `/nfs/home/Paul/PyPCL/.claude/worktrees/pico-weighted-cls-loss/results/clsloss_swap_1006/`
     （24 個舊 loss）與 `switch-analysis/results/clsloss_swap_c20_mclnl/`（MCLNL），protocol 相同。
   - 參考：`hard_c100_1017` 用了 10% held-out，跟本實驗**不完全可比**，只能當旁證。
5. **回報**：表格 + 重點結論，標明 seed 不足 3 的格子。
