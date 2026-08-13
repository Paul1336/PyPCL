# SoLar：Sinkhorn Label Refinery for Imbalanced Partial-Label Learning

> **狀態**：骨架文件，Step 1（程式碼對應）已完成，Step 2（公式逐項比對）與 Step 3（benchmark 查證）
> 標記為 TODO，交給下一個 session 對照論文原文填寫。

**論文來源：** Wang, H., Xia, M., Li, Y., Mao, Y., Feng, L., Chen, G., & Zhao, J. (2022).
*SoLar: Sinkhorn Label Refinery for Imbalanced Partial-Label Learning.* NeurIPS 2022, 35:8104–8117.
[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/357a0a771bf65ee07926d6af41b75030-Abstract-Conference.html) ·
[arXiv](https://arxiv.org/abs/2209.10365)

**Algorithm ID（pipeline 內部字串）：** `SoLar`

---

## 對應程式碼

### Loss：`src/solar/utils_loss.py` (`partial_loss`)

- `__init__`（5-13 行）：`confidence = train_givenY / train_givenY.sum(dim=1)`，即候選集內
  均等分佈當作初始信心（跟 Proden/PiCO 一樣的均等初始化）。
- `forward(outputs, index, targets=None)`（15-25 行）：
  - 若 `targets=None`：用內部存的 `self.confidence[index]` 當 soft label
  - 若外部傳入 `targets`：直接用外部給的 target（給 Sinkhorn 精煉後的分佈用）
  - `loss = -(log_softmax(outputs) * confidence/targets).sum(1)`，跟 PartialLoss/ProdenLoss
    同樣是加權 CE 形式，回傳 `(average_loss, loss_vec)` —— **注意這裡多回傳一個
    per-sample 的 `loss_vec`**，供外部做 hard/soft selection 用（PiCO/Proden 沒有這個）。
  - `confidence_update(temp_un_conf, batch_index)`（27-30 行）：**直接覆蓋**
    `confidence[batch_index] = temp_un_conf`，**沒有 EMA**（跟 PiCO 的 EMA 更新機制不同，
    是本文件已知的一個關鍵差異點，需要在 Step 2 確認論文本身是否也是直接覆蓋、還是有額外的
    平滑機制在別處，例如 empirical distribution 的 EMA 是用 `gamma1`/`gamma2`，跟這裡的
    confidence 更新是兩件事，不要混淆）。

### 演算法：`src/solar/utils_algo.py`

- **`sinkhorn(pred, eta, r_in, rec=False)`**（8-45 行）：Sinkhorn-Knopp 矩陣縮放。
  - `PS = pred^eta`（18 行，把預測機率取 `eta` 次方後轉置成 `[K, N]`）
  - 迭代（最多 50 次，24-40 行）：`r = r_init / (PS @ c)`、`c_new = (1/N) / (r^T @ PS)^T`，
    每 10 次迭代檢查一次誤差（`err = sum(c_new) + sum(r)`），並有 NaN fallback（遞迴呼叫
    自己一次，加上微小擾動 `pred + 1e-5*(pred==0)`，31-38 行）。
  - 最終回傳 `PS` 縮放到列邊際 `r_in`、欄邊際 `1/N`（41-45 行）。
- **`linear_rampup(current, rampup_length)`**（47-53 行）：標準線性 ramp-up，`current/length`
  clip 到 `[0,1]`。

### 兩階段訓練

CLAUDE.md 既有描述（**待 Step 2 驗證是否仍準確**，因為 CLAUDE.md 本身已被判定過時，只有這部分是否
仍對得上 `train_solar`/`train_solar_epoch` 的實際程式碼需要確認）：

- **Stage 1（Pre-estimation，`est_epochs`）**：估計 empirical class distribution
- **Stage 2（Final Training）**：用精煉後的分佈 + Sinkhorn 做 pseudo-label 選擇，搭配 mixup、
  hard/soft selection、`gamma1`（stage 1 EMA rate）/`gamma2`（stage 2 EMA rate）更新
  empirical distribution

Entry point：

- 新版 pipeline：`run_solar`（`src/pipeline/algorithms/runners.py:387-409`，兩階段：
  `est_epochs` 預估計 + `epochs` 主訓練，實際訓練迴圈在 `src/engine.py` 的 `train_solar`），
  `AlgorithmSpec('SoLar', 'PLL', r.run_solar)`（`src/pipeline/algorithms/__init__.py:36`）
- 舊版 pipeline：`setup_solar`（`src/model_setup.py:149-168`）/
  `run_solar_training`（`src/training_pipelines.py:155`），兩套 pipeline 都是透過同一個
  `setup_solar` 建構模型/loss/optimizer

---

## 演算法保真度比對 — TODO（Step 2）

- [ ] `sinkhorn` 函式裡 `PS = pred^eta` 的 `eta` 參數，跟論文的 Sinkhorn 溫度/正則化參數是否對應
      一致？論文的 optimal transport 公式（cost matrix = 模型 softmax 預測 × 候選標籤 indicator）
      跟這裡的 `PS = pred.T^eta` 實作方式是否等價？
- [ ] Two-stage 的切分（`est_epochs` 預估計 → 正式訓練）是否跟論文演算法描述的流程一致？
      特別是 stage 1 的 EMA rate `gamma1=0.1` 與 stage 2 的 `gamma2=0.01` 是否為論文預設值。
- [ ] `confidence_update` 直接覆蓋（無 EMA）是否符合論文？（對照上面「對應程式碼」段落的提醒）
- [ ] hard/soft label selection 的 threshold `tau`、`rho_range` 是否跟論文一致？
- [ ] mixup 的使用方式（哪些樣本參與、係數怎麼抽樣）是否跟論文一致？

---

## 原論文使用的 Benchmark — TODO（Step 3）

SoLar 論文標題強調 "Imbalanced"，需要查證：論文用的是否為 long-tailed CIFAR-10/CIFAR-100 partial
label 版本（例如不同 imbalance ratio），對照本專案目前的 CIFAR-10/20/100-subset 生成邏輯
（`src/data_setup.py`、`src/cifar100_subset.py`）**目前看起來沒有 long-tailed/imbalance 采樣機制
（待確認）**——如果論文的核心 benchmark 就是不平衡資料集，而目前 pipeline 只有平衡資料集，
Step 5 設計 config 時必須先在這裡把這個落差寫清楚，不要假裝用平衡資料集就等於複現論文。

---

## Fixed 版本 — 尚未開始（Step 4）

放在 `src/solar/fixed_utils_loss.py` / `src/solar/fixed_utils_algo.py`，class/function 前綴
`Fixed`/`fixed_`。

---

## 實驗 Config — 尚未開始（Step 5）

見 [主引導文件](00_paper_alignment_guide.md) 的前置條件（pipeline 尚未驗證可跑）。若論文 benchmark
確實需要 imbalance 采樣而目前 pipeline 沒有，這部分要先跟使用者確認是否要擴充 pipeline。
