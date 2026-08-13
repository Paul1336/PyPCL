# Cour 2011 / CLPL：Partial Label Learning 的 Squared-Hinge 損失

> **⚠️ 2026-08-13 修正說明**：這份文件原本把「本專案的 Cour 2011 實作」指向 `src/cour_loss.py`
> (`UniformCandidateCrossEntropyLoss` / `CourLoss`)，並以「均等平均候選集 log-likelihood」的公式
> 介紹核心思想。**這是錯誤的對應**——`src/cour_loss.py` 從未被 `model_setup.py`、
> `training_pipelines.py`、`src/pipeline/algorithms/runners.py` 任何一處匯入，是一個孤兒檔案，
> 而且它自己的 docstring 明白寫著：「This is NOT the CLPL loss from Cour, Sapp & Taskar (JMLR 2011)...
> kept for reference and backward-compatibility with earlier experiment runs」。
>
> 真正被 pipeline 呼叫、對應 `CLPL` 這個 algorithm ID 的實作是 **`src/clpl_loss.py` 的
> `CLPLSquaredHingeLoss`**，用的是 squared-hinge surrogate loss，不是 cross-entropy 均等平均。
> 下面已重寫「核心思想」與「在本專案中的實作」章節以反映這個事實；原本的均等平均版本移到
> [附錄：`cour_loss.py`（孤兒檔案）](#附錄cour_losspy孤兒檔案) 保留參考。
>
> **這份文件目前只完成 Step 1（程式碼對應）與 Step 3 起手式，Step 2 的公式逐項比對（squared-hinge
> 是否真的忠實對應論文原始寫法）尚未對照論文原文逐條驗證，標記為 TODO，交給下一個 session 完成。**

**論文來源：** Cour, T., Sapp, B., & Taskar, B. (2011).
*Learning from Partial Labels.* JMLR 12, 1501–1536.
[JMLR 連結](https://jmlr.org/papers/v12/cour11a.html)

**Algorithm ID（pipeline 內部字串）：** `CLPL`

---

## 問題設定

**Partial Label Learning (PLL)** 的假設：

- 每個訓練樣本 $x_i$ 有一個**候選標籤集** $Y_i \subseteq \{1, \ldots, C\}$
- 真實標籤 $y_i^* \in Y_i$ 一定在候選集裡，但我們不知道是哪一個
- 候選集以外的標籤絕對不是真實答案

例如：C=10，$Y_i = \{2, 5, 7\}$ → 真實 label 是 2、5 或 7 其中之一。

---

## 核心思想（依目前實際被使用的 `CLPLSquaredHingeLoss`）

Cour 2011 的 CLPL 把 PLL 轉成一個 **squared-hinge 的 margin-based 目標**（不是 cross-entropy）：

$$\mathcal{L}_i = \psi\!\Big(\tfrac{1}{|Y_i|}\sum_{a \in Y_i} g_a(x_i)\Big) \;+\; \sum_{a \notin Y_i} \psi\big(-g_a(x_i)\big)$$

其中 $\psi(u) = \max(0, 1-u)^2$（squared hinge），$g_a(x_i)$ 是模型對類別 $a$ 的原始 logit
（**未經過 softmax**）。展開後對應到 `src/clpl_loss.py` 的兩項：

- **positive term**：$\max(0, 1 - \overline{g}_{Y_i})^2$，其中 $\overline{g}_{Y_i}$ 是候選集內
  logits 的平均值 —— 希望候選集的平均分數被推高到 margin 1 以上
- **negative term**：$\sum_{a \notin Y_i} \max(0, 1 + g_a(x_i))^2$ —— 希望每個非候選類別的分數
  被壓低到 margin -1 以下

直覺：因為不知道候選集裡哪個是真的，所以只約束「候選集的平均分數要高」，同時「每個非候選類別的分數
都要低」，而不是像 cross-entropy 那樣對個別候選 label 給機率權重。

### 與 Proden（ICML 2020）的差別

| 方法 | 損失型態 | 對候選集的處理方式 |
|------|---------|-----------------|
| **CLPL (Cour 2011)** | squared-hinge margin loss，作用在原始 logits | 用候選集的**平均分數**當作一個整體 margin 目標，非候選類別各自被壓低 |
| **Proden** | cross-entropy | 每個候選 label 各自依模型信心 $\frac{p_\theta(j\mid x_i)}{\sum_{j'\in Y_i} p_\theta(j'\mid x_i)}$ 加權，訓練過程中動態更新信心（progressive identification） |

CLPL 不依賴模型自身對候選標籤的相對信心排名（沒有 Proden 那種 self-training/EMA 更新機制），是一個
凸的 surrogate loss；Proden 則是非凸、但透過信心動態調整逐步逼近真實標籤。

---

## 為什麼有效（理論保證）— TODO

> **Step 2 待辦**：論文的核心貢獻之一是一個 consistency/generalization 定理。下面這段是舊版文件對
> **均等平均 cross-entropy 版本**（非目前實作）寫的理論直覺，尚未針對 squared-hinge 版本重新對照
> JMLR 論文原文驗證，也還沒確認論文本身是否包含這裡描述的 consistency 論述。下一個 session 做 Step 2
> 時請直接讀論文原文重寫這節，不要假設下面的敘述仍然適用於 squared-hinge 版本。

（暫存舊敘述，待驗證後改寫或刪除）：當候選集生成滿足某些條件（候選集足夠稀疏、隨機生成），最小化
候選集相關的損失在樣本數趨近無窮大時，其最小化解會等價於監督式學習的解。

---

## 在本專案中的實作

```
src/clpl_loss.py → CLPLSquaredHingeLoss.forward(outputs, partial_labels)
（別名 CourCLPLSquaredHingeLoss，同一個 class，src/clpl_loss.py:55）
```

### 輸入

| 變數 | Shape | 說明 |
|------|-------|------|
| `outputs` | `[B, C]` | 模型輸出 **logits**（未 softmax，這點對 squared-hinge 很重要） |
| `partial_labels` | `[B, L]` | 候選 label indices，-1 為 padding |

### 計算步驟（`src/clpl_loss.py:28-51`）

```python
# Step 1: 建立 binary candidate mask [B, C]（3D scatter 向量化，避免 for-loop）
candidate_mask = ...   # 1 = 候選, 0 = 非候選
negative_mask = 1.0 - candidate_mask

# Step 2: positive term = psi(mean_{a in Y_i} g_a)
avg_score = (outputs * candidate_mask).sum(dim=1) / count
positive_loss = F.relu(1.0 - avg_score).pow(2)

# Step 3: negative term = sum_{a not in Y_i} psi(-g_a) = sum max(0, 1+g_a)^2
negative_loss = (negative_mask * F.relu(1.0 + outputs).pow(2)).sum(dim=1)

return (positive_loss + negative_loss).mean()
```

### 與現有程式的接口

- **新版 pipeline**：`run_clpl`（`src/pipeline/algorithms/runners.py:126`），
  註冊為 `AlgorithmSpec('CLPL', 'PLL', r.run_clpl)`（`src/pipeline/algorithms/__init__.py:29`）
- **舊版 pipeline**：`setup_cour`（`src/model_setup.py:14`，標註為 "Cour 2011 CLPL (squared-hinge)"）
  / `run_cour_training`（`src/training_pipelines.py:5`）—— 注意函式名稱仍叫 `cour`，但實際建構的是
  `CLPLSquaredHingeLoss`，不是 `cour_loss.py` 裡的類別，命名容易造成混淆
- 使用 PL dataloader（`loaders['pl']`，標準的 `WeaklySupervisedDataset`）
- 與 `train_algorithm()` in `src/engine.py` 完全相容（和 Proden 使用同一套訓練 loop）
- 不需要特殊的 augmentation 或 queue，架構最單純

---

## 原論文使用的 Benchmark — TODO（Step 3）

> **尚未查證。** 根據論文標題與年代（2011，pre-deep-learning 影像分類），Cour 2011 的原始實驗**很可能
> 不是用 CIFAR**，而是傳統的人臉辨識/物件辨識 partial-label 資料集（例如 Yahoo! News、MSRCv2、Lost
> 等在 PLL 文獻中常見的 benchmark），需要下一個 session 實際查閱論文確認資料集清單、規模、
> partial label 生成方式。
>
> 這意味著 Step 5（用現有 CIFAR pipeline 生成實驗 config）**不能宣稱是「複現原論文實驗」**，只能算是
> 「用同一個 squared-hinge loss 在 CIFAR 衍生資料集上驗證」。如果要真正複現原論文 benchmark，需要
> 新增對應的 dataset loader，這超出目前 pipeline 範圍，需要另外跟使用者確認是否要做。

---

## Fixed 版本 — 尚未開始（Step 4）

Step 2 的公式逐項比對還沒完成，因此還不知道 `CLPLSquaredHingeLoss` 跟論文原始寫法是否有需要修正的
落差。等 Step 2 完成後，若有落差，修正版會放在 `src/fixed_clpl_loss.py`
（class `FixedCLPLSquaredHingeLoss`），並在這裡補上連結與修改說明。

---

## 實驗 Config — 尚未開始（Step 5）

等 Step 3（benchmark 查證）與 Step 4（fixed 實作，若需要）完成、且[主引導文件](00_paper_alignment_guide.md)
列出的 pipeline 前置條件（重新下載 CIFAR-100、跑通至少一次 smoke test）確認可行後才開始。

---

## 在 Sweep 實驗裡的角色

本專案以 CLPL (Cour 2011) 作為 **PLL 的代表**，與 **MCL-LOG（CLL 的代表）** 進行比較：

| | CLPL / Cour 2011 (PLL) | MCL-LOG (CLL) |
|--|--|--|
| 監督信號 | 候選集 $Y_i$（包含真實 label） | 互補集 $\bar{Y}_i = \{1..C\} \setminus Y_i$ |
| 學習信號方向 | 推高候選集平均分數、壓低非候選分數 | 最大化非互補集機率 |
| k=1 時 | = 完全監督（只有真實 label） | = 幾乎無信號（C-1 個互補 label） |
| k=C-1 時 | = 幾乎無信號（C-1 個候選） | = 完全監督（只有 1 個互補 label） |

兩者的關係：**相同的 (C, k) 設定下，PL 和 CL 的資訊是互補的（資訊量之和 = 完全監督的資訊量）**，所以在
不同 k 下的表現交叉點是這類 sweep 實驗最感興趣的分析對象。

---

## 附錄：`cour_loss.py`（孤兒檔案）

`src/cour_loss.py` 定義 `UniformCandidateCrossEntropyLoss`（別名 `CourLoss`），公式是：

$$\mathcal{L}_{\text{uniform-CE}} = -\frac{1}{N}\sum_{i=1}^N \frac{1}{|Y_i|}\sum_{j \in Y_i} \log p_\theta(j\mid x_i)$$

即「均等平均候選集內每個 label 的 log-likelihood」，是一個 cross-entropy 型態的簡化 loss，**不是**
Cour 2011 論文的 squared-hinge CLPL loss（該檔案自己的 docstring 也這樣聲明）。目前完全沒有任何
pipeline 程式碼匯入這個類別，推測是早期實驗遺留下來、之後被 `clpl_loss.py` 的正確實作取代，但沒有
被清除。

**待決定事項（Step 1/2 處理 CLPL 論文文件時一併處理）：**
- 是否有舊實驗結果（`results/`、`plots/` 底下）依賴這個版本，需要確認後才能決定刪除或保留
- 如果保留，建議把類別/別名改成不含 "Cour" 字樣的名字（例如 `UniformCandidateCELoss`），避免
  未來又被誤認為是論文對應的實作
