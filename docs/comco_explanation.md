# ComCo：Complementary Supervised Contrastive Learning for Complementary Label Learning

> **狀態**：骨架文件，Step 1（程式碼對應）已完成，Step 2（公式逐項比對）與 Step 3（benchmark 查證）
> 標記為 TODO，交給下一個 session 對照論文原文填寫。

**論文來源：** Jiang, H., Sun, Z., & Tian, Y. (2024).
*ComCo: Complementary Supervised Contrastive Learning for Complementary Label Learning.*
Neural Networks, Vol. 169, pp. 44–56. DOI: 10.1016/j.neunet.2023.10.013
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608023005683) ·
[PubMed](https://pubmed.ncbi.nlm.nih.gov/37857172/)

**Algorithm ID（pipeline 內部字串）：** `ComCo`

---

## 對應程式碼

### 模型架構：`src/comco/model.py` (`ComCoModel`)

- 結構跟 `PiCOModel` 高度相似（雙 momentum encoder + queue），但關鍵差異：
  **pseudo label 是 `cls_out.argmax(dim=1)`（46 行，未經任何 mask）**，不像 PiCO 是
  `softmax(output) * partial_Y` 後取 argmax —— 因為 CLL 設定下沒有候選集，只有互補集，
  沒辦法用同樣的 masking 邏輯。
- Queue 額外多存 `queue_comp`（22-23 行，`[Q, C]`），即每個 queue 元素的互補標籤 mask，
  供 `ComCoContrastiveLoss` 的負樣本選取策略使用。
- `forward` 回傳 `(cls_out, q, all_feats, all_pseudo, all_comp)`（59 行），
  `all_feats/all_pseudo/all_comp` 都是 `[query, key, queue]` 三段串接（52-55 行）。
- `_dequeue_and_enqueue`（31-39 行）一樣有 `assert queue_size % B == 0`（35 行，跟 PiCO
  同樣的隱患，見主引導文件）。

### Loss：`src/comco/utils_loss.py`

#### `ComCoCLSLoss`（分類損失，6-25 行，docstring 標註 "Eq. 3 in paper"）

$$L = -\log\Big(\sum_{c \notin \bar Y} \text{softmax}(logits)_c\Big) \cdot \frac{C-1}{C-m}$$

跟 `MCL_LOG`（`src/mcl_losses.py`）的數學形式**幾乎一模一樣**（非互補機率總和取負對數，乘上同樣的
`(C-1)/(C-m)` unbiased risk estimator 縮放）。docstring 自己也承認（14 行）：
「Reduces to SCL-NL (-log(1 - g_ybar)) for single complementary label」。
**Step 2 需要驗證**：ComCo 論文是否真的宣稱這個分類損失是 MCL-LOG 的直接復用，還是論文有自己
獨立推導、只是恰好在數學上等價；如果程式碼是直接搬 `MCL_LOG` 的公式，要在這裡明確記錄「兩個實作
共享同一個數學式，來源分別是哪篇論文」。

#### `ComCoContrastiveLoss`（對比損失，28-127 行）

- **正樣本集合 $P(x_i)$**（69-87 行）：
  - 一定包含 key-view embedding $k_i$（key_indices，73-74 行）
  - `warmup_pos` 之後（76-87 行）：additionally 選取 pool 中「pseudo label 相同」且
    integrated similarity 最高的 top-K 鄰居。Integrated similarity 公式
    （docstring 41 行）：$\text{Sim}(x_i,x_j) = [\tilde y_i = \tilde y_j] \cdot 0.5(1+\cos\_sim)$，
    docstring 標註「Strategy B from paper」。
- **負樣本集合 $N(x_i)$**（89-112 行，`warmup_neg` 之後才啟用）：
  - Pool 依互補標籤切成 $C$ 個子集 $S_c$（`comp_mask[:,c]==1` 的元素）
  - 選最遠離 anchor 的子集：$N(x_i) = S_{\arg\max_c \text{Dist\_min}(z_i, S_c)}$，
    $\text{Dist\_min}(z, S_c) = 0.5(1 - \max_{z_j \in S_c} z\cdot z_j)$（docstring 41 行，
    程式碼 94-104 行，用 for-loop 跑過 $C$ 個類別避免 $[B,A,C]$ 記憶體爆炸）
  - `warmup_neg` 之前：分母包含全部 pool 元素（標準 InfoNCE，110-112 行）
- 分母 = $P(x_i) \cup N(x_i)$（109 行 `denom_mask`），標準溫度縮放 InfoNCE softmax CE
  收尾（118-127 行）。

### Entry point

- 新版 pipeline：`run_comco`（`src/pipeline/algorithms/runners.py:348-381`），
  `AlgorithmSpec('ComCo', 'CLL', r.run_comco)`（`src/pipeline/algorithms/__init__.py:42`）
- 舊版 pipeline：`setup_comco`（`src/model_setup.py`，**重複定義兩次**，約第 52 行與第 96 行，
  後者覆蓋前者）/ `run_comco_training`（`src/training_pipelines.py`，同樣**重複定義兩次**，
  約 47-66 行與 94-113 行）——這是主引導文件列出的已知缺陷之一，Step 2 順便處理時建議先確認
  兩份定義是否真的完全等價（如果不等價，"被覆蓋" 的那份可能藏著沒被使用到的邏輯差異），再決定
  要不要清掉重複的那份。

---

## 演算法保真度比對 — TODO（Step 2）

- [ ] `ComCoCLSLoss` 是否真的等於論文 Eq. 3？（上面已經標記出它跟 `MCL_LOG` 數學等價，需要確認
      論文是否明確承認這個關聯，或是 ComCo 論文有自己獨立的推導脈絡）
  - [ ] "Strategy B" 的正樣本選取（top-K 同 pseudo-label 鄰居）是否為論文唯一策略，還是論文有
      提出 Strategy A/C 等其他選項，本專案只實作了其中一種？
  - [ ] 負樣本選取的 $\text{Dist\_min}$ 策略，是否跟論文 Appendix 的公式完全一致（包含
      $S_c$ 為空集合時的 fallback 行為，程式碼用 `-1e9` 讓該類別不會被選中，96-102 行）。
  - [ ] `warmup_pos`/`warmup_neg` 的切換時機（哪個 epoch 開始）是否對應論文描述的訓練排程。
  - [ ] `temperature=0.17`、`top_k=1`（`__init__` 47-50 行）是否為論文報告的預設超參數。

---

## 原論文使用的 Benchmark — TODO（Step 3）

需要查證 Neural Networks 2024 這篇論文的實驗資料集與互補標籤生成設定，對照本專案的
CIFAR-10/20/100-subset CLL 生成邏輯是否一致。

---

## Fixed 版本 — 尚未開始（Step 4）

放在 `src/comco/fixed_utils_loss.py`（以及必要時 `src/comco/fixed_model.py`），class 前綴
`Fixed`。修正前建議先處理「已知缺陷」段落提到的 `setup_comco`/`run_comco_training` 重複定義問題，
避免修正版接到錯誤的（被覆蓋的）舊邏輯。

---

## 實驗 Config — 尚未開始（Step 5）

見 [主引導文件](00_paper_alignment_guide.md) 的前置條件（pipeline 尚未驗證可跑，且 ComCo 跟 PiCO
一樣受 `moco_queue % batch_size` assert 影響）。
