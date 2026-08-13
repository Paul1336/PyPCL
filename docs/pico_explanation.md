# PiCO：Contrastive Label Disambiguation for Partial Label Learning

> **狀態**：骨架文件，Step 1（程式碼對應）已完成，Step 2（公式逐項比對）與 Step 3（benchmark 查證）
> 標記為 TODO，交給下一個 session 對照論文原文填寫。

**論文來源：** Wang, H., Xiao, R., Li, Y., Feng, L., Niu, G., Chen, G., & Zhao, J. (2022).
*PiCO: Contrastive Label Disambiguation for Partial Label Learning.* ICLR 2022.
[OpenReview](https://openreview.net/forum?id=EhYjZy6e1gJ)

**Algorithm ID（pipeline 內部字串）：** `PiCO`
（repo 裡還有三個 PiCO 的變體 `PiCO-MCL`、`PiCO-SC`、`PiCO-CLS`，**不在使用者原始 8 篇論文範圍內，
本文件不處理**，只記錄這裡以免未來搞混。）

---

## 對應程式碼

### 模型架構：`src/pico/model.py` (`PiCOModel`)

- **雙 encoder（MoCo 風格 momentum contrast）**：`encoder_q`（會被梯度更新）與 `encoder_k`
  （用 momentum 更新，`_momentum_update_key_encoder`，24-27 行：
  `param_k = param_k * moco_m + param_q * (1 - moco_m)`）。
- **Prototype 記憶體**：`prototypes[C, low_dim]` buffer（20 行），在 `forward`（52-54 行）用
  EMA 依 pseudo label 更新：`prototypes[label] = prototypes[label]*proto_m + (1-proto_m)*feat`，
  之後 L2 normalize。
- **Pseudo label 產生**（45-46 行）：`predicted_scores = softmax(output) * partial_Y`（候選集
  mask），取 `argmax` 當作該 batch 的 pseudo label —— **這是目前模型自己的分類頭 softmax**，
  不是 prototype 相似度（prototype score `score_prot` 只回傳出去給外部的信心更新機制用，
  詳見 `PartialLoss.confidence_update`）。
- **Feature queue**：`queue`/`queue_pseudo`（17-19 行），`_dequeue_and_enqueue`（30-38 行）
  維護 FIFO 佇列，`assert moco_queue % batch_size == 0`（34 行，已知隱患，見主引導文件）。
- `forward` 回傳 `(output, features, pseudo_labels, score_prot)`，`features` 是
  `q, k, queue` 串接（60 行），`score_prot = softmax(q · prototypes^T)`（49-50 行）。

### Loss：`src/pico/utils_loss.py`

- **`PartialLoss`**（分類損失，5-29 行）：
  - `forward(outputs, index)`（17-21 行）：`-Σ log_softmax(outputs) * confidence[index]`，
    跟 Proden 的加權 CE 形式相同。
  - `confidence_update(temp_un_conf, batch_index, batchY)`（23-29 行）：EMA 更新，
    `pseudo_label = one_hot(argmax(temp_un_conf * batchY))`（用傳入的 `temp_un_conf`，**通常是
    prototype 相似度 `score_prot`，由呼叫端決定**，不是模型 forward 裡的候選 mask 版本），
    `confidence = conf_ema_m * confidence + (1-conf_ema_m) * pseudo_label`。
  - `conf_ema_m` 由 `set_conf_ema_m`（11-15 行）依 epoch 在 `conf_ema_range` 兩端線性內插。
- **`SupConLoss`**（對比損失，31-75 行）：兩種模式，由是否傳入 `mask` 決定：
  - **Partial Label Mode**（`mask` 給定，41-59 行）：標準 SupCon InfoNCE，用 label 相似度 mask
    決定正樣本。
  - **MoCo Loss 模式**（`mask=None`，60-74 行）：標準 InfoNCE，正樣本是 `q·k`，負樣本是
    `q·queue`。

### Entry point

- 新版 pipeline：`run_pico`（`src/pipeline/algorithms/runners.py:228-252`），
  `AlgorithmSpec('PiCO', 'PLL', r.run_pico)`（`src/pipeline/algorithms/__init__.py:32`）
- 舊版 pipeline：`setup_pico`（`src/model_setup.py:78-94`）/
  `run_pico_training`（`src/training_pipelines.py:69`）
- Dataloader：`pico_loader`（`PicoDataset`，弱/強 augmentation pair + one-hot 候選向量）
- 總損失（依 CLAUDE.md 既有描述，待 Step 2 驗證是否仍準確）：
  `cls_loss + loss_weight * cont_loss`

---

## 演算法保真度比對 — TODO（Step 2）

需要對照論文確認的重點（PiCO 是這 8 篇裡機制最複雜的一個，逐項核對很重要）：

- [ ] **Pseudo label 來源**：`PiCOModel.forward` 用模型自己的 softmax（`predicted_scores =
      softmax(output)*partial_Y`）產生 pseudo label 去更新 prototype；但 `confidence_update`
      是用外部傳入的 `temp_un_conf`（推測是 `score_prot`，prototype 相似度）。論文裡「prototype
      驅動信心更新」的機制，是否真的對應到 `score_prot` 被傳給 `confidence_update` 這條路徑？
      需要在呼叫端（`runners.py`/`engine.py` 的 `train_pico_epoch`）確認實際傳的是哪個張量。
- [ ] **Warm-up 機制**：CLAUDE.md 提到「confidence updated via EMA (after warmup epoch)」，
      對照 `prot_start` 參數，需要在程式碼裡找到對應的 if-else 分支確認 warm-up 前後行為差異
      是否符合論文描述。
- [ ] `conf_ema_range` 的線性內插排程是否跟論文一致（論文是否用同樣的 momentum schedule）？
- [ ] `SupConLoss` 的兩種模式（Partial Label Mode / MoCo Loss）如何組合使用，是否跟論文 Figure/
      演算法描述的訓練流程一致？
- [ ] `loss_weight` 加權方式（`cls_loss + loss_weight*cont_loss`）是否跟論文一致？

---

## 原論文使用的 Benchmark — TODO（Step 3）

需要查證：PiCO 論文的實驗資料集（CIFAR-10/CIFAR-100 partial-label 化、CUB-200 等）、partial label
生成參數（uniform partial ratio q），並對照本專案 CIFAR-10/20/100-subset 的候選集生成邏輯
是否一致。

---

## Fixed 版本 — 尚未開始（Step 4）

放在 `src/pico/fixed_utils_loss.py`（以及必要時 `src/pico/fixed_model.py`），class 前綴 `Fixed`。

---

## 實驗 Config — 尚未開始（Step 5）

見 [主引導文件](00_paper_alignment_guide.md) 的前置條件（pipeline 尚未驗證可跑，且 PiCO 對
`moco_queue % batch_size` 的 assert 特別敏感，設計 config 時要留意）。
