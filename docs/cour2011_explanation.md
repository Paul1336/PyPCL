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
> **2026-08-14 更新**：Step 2（公式逐項比對）與 Step 3（benchmark 查證）已完成，直接對照論文 PDF
> （`C:\Users\User\Desktop\papers\CLPL.pdf`）逐條驗證。結論：**`CLPLSquaredHingeLoss` 與論文 Eq. 2
> 完全一致（squared-hinge 版本），不需要 fixed 版本**。詳見下方「演算法保真度比對」與「原論文使用的
> Benchmark」兩節。
>
> **2026-08-14 二次覆核**：改用 `pypdf` 直接擷取 PDF 純文字（不需要 poppler/pdftoppm）重新逐字比對
> 論文原文：Eq. 2 原文 `Lψ(g(x),y) = ψ(1/|y| Σ_{a∈y} ga(x)) + Σ_{a∉y} ψ(−ga(x))` 跟下方公式、
> 跟程式碼逐項一致；Table 2 的 benchmark 維度（dermatology 366/34/6、ecoli 336/8/8、
> abalone 4177/8/29、FIW 系列、Lost audio 522/50/19）也逐一核對過，跟下方表格完全吻合。這次沒有
> 發現需要修正的地方。

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

## 演算法保真度比對（Step 2，已完成）

### 論文原始公式

論文 Section 4.1 的 **Eq. 2**（Convex Loss for Partial Labels）：

$$\mathcal{L}_\psi(g(x), Y) = \psi\!\Big(\tfrac{1}{|Y|}\sum_{a\in Y} g_a(x)\Big) + \sum_{a\notin Y}\psi(-g_a(x))$$

論文對 $\psi$ 只要求「凸、遞減、下有界」的代理函數（hinge、squared hinge、exponential、logistic 都可以），
在 SVM 實驗中主要用的就是 squared hinge：$\psi(u) = \max(0,1-u)^2$（論文 p.1510、Section 4.4/5.1）。

論文特別強調 Eq. 2 是「**候選集內取平均**」（mean），並明確跟另外兩個變體區分開：
- **"naive" loss**（論文 Eq. 3，引用 Jin & Ghahramani 2002）：對每個候選 label 個別套用 $\psi$
  再平均，`(1/|Y|)Σ_{a∈Y}ψ(g_a) + Σ_{a∉Y}ψ(-g_a)`
- **"sum" 變體**（論文 Eq. 6）：候選集內**取和**而非取平均，`ψ(Σ_{a∈Y}g_a) + Σ_{a∉Y}ψ(-g_a)`

CLPL（Eq. 2，"mean" 模型）在論文 **Proposition 6** 被證明是比 naive loss 更緊的凸上界
（`2·L_ambiguous ≤ L_max_ψ ≤ L_ψ(CLPL) ≤ L_naive_ψ`）。

### 逐項比對結果：**完全一致**

| 論文 Eq. 2 | `src/clpl_loss.py` | 是否一致 |
|---|---|---|
| $\psi\big((1/\|Y\|)\sum_{a\in Y}g_a(x)\big)$（候選集**平均**分數） | `avg_score = (outputs*candidate_mask).sum(1)/count`，`positive_loss = relu(1-avg_score)^2` | ✅ 完全一致，正確用的是論文的「mean」模型，不是 naive/sum 變體 |
| $\sum_{a\notin Y}\psi(-g_a(x))$ | `negative_loss = (negative_mask*relu(1+outputs)^2).sum(1)`（因 $1+g_a = 1-(-g_a)$） | ✅ 完全一致 |
| $\psi(u)=\max(0,1-u)^2$ | `F.relu(...).pow(2)` | ✅ 完全一致 |
| $g_a(x)$ 為**未經過 softmax 的原始判別分數** | 程式碼明確要求 `outputs` 是 logits，docstring 也強調 NOT softmax | ✅ 一致，且這點很關鍵：squared hinge 的 margin=1 是針對無界實數分數設計的，如果誤用 softmax 機率（有界在 [0,1]）會讓 margin 幾乎打不到，這是深度學習重現這篇論文時最容易犯的錯，這份程式碼沒有犯 |
| 無額外正則化/kernel 項混在 loss 裡 | 沒有 | ✅ 一致（論文的 $\frac{1}{2}\|w\|^2$ 正則化是在 optimizer 層級加的，不在 loss 定義裡；程式碼對應到 optimizer 的 `weight_decay`） |

**結論：`CLPLSquaredHingeLoss` 是論文 Eq. 2（squared-hinge 版本）逐項精確的實作，沒有發現任何數學
上的落差。**

### 論文的模型假設 vs 本專案的深度網路

論文本身只用**線性模型**（$g_a(x)=w_a\cdot f(x)$，Section 4）和**kernel SVM / boosting**
（Section 6、Appendix A）做實驗，2011 年的論文沒有深度網路。這點對移植到深度學習有兩個含義：

1. **Consistency 定理（Proposition 5）是模型無關的**：證明是在「假設空間 $\mathcal{G}\to\mathbb{R}^L$
   任意豐富」的極限下做的，原則上可以套用到深度網路，但需要滿足：(a) $\psi$ 可微、凸、下界、遞減，
   $\psi'(0)<0$；(b) 「最可能的真實標籤」也必須是「最可能出現在候選集裡的標籤」；(c) 一個
   dominance condition（論文 p.1511 給了一個具體反例說明這個條件不成立時 CLPL 會選錯標籤）。
   **本專案目前沒有驗證 CIFAR 衍生的候選集生成方式是否滿足這個 dominance condition**——
   這是 Step 2 額外發現的一個新 TODO，建議在設計 Step 5 實驗時一併記錄。
2. **有限樣本 generalization bound（Propositions 7–8）是針對線性/kernel 模型证明的**，
   不會自動套用到深度網路——換句話說，用這個 loss 訓練 CNN 在數學上仍然是在做同一件事
   （最小化 Eq. 2），但論文對「這個 loss 訓練出來的分類器有多好」的**理論保證不會自動移植過來**，
   這是一個誠實的落差，不是程式碼錯誤，只是理論適用範圍的邊界。

### 其他理論結果（供參考，不影響實作忠實度）

- **Proposition 1**：模型無關的界，把「候選集 0/1 loss」跟「真實 0/1 loss」用 *ambiguity degree*
  $\epsilon$ 連起來：$E[L_a] \le E[L] \le \frac{1}{1-\epsilon}E[L_a]$
- **Proposition 5**：CLPL 的一致性（consistency）證明，條件見上
- **Proposition 6**：CLPL 比 naive loss 更緊的凸上界（見上）

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

## 原論文使用的 Benchmark（Step 3，已完成）

**確認：論文完全沒有用 CIFAR，用的是 2011 年代的 UCI 表格資料、人臉資料、真實電視劇弱標註資料。**
（論文 Table 2 / Section 7-8）

| 資料集 | 樣本數 | 特徵維度 | 類別數 | 任務 | 候選集來源 |
|---|---|---|---|---|---|
| UCI: dermatology | 366 | 34 | 6 | 疾病診斷 | 合成（人工加噪聲） |
| UCI: ecoli | 336 | 8 | 8 | 蛋白質位置預測 | 合成 |
| UCI: abalone | 4177 | 8 | 29 | 年齡估計 | 合成 |
| FIW(10b)（Faces in the Wild） | 500 | 50（PCA） | 10（平衡） | 人臉辨識 | 合成 |
| FIW(10) | 1456 | 50（PCA） | 10（不平衡） | 人臉辨識 | 合成 |
| FIW(100) | 3011 | 50（PCA） | 100（不平衡） | 人臉辨識 | 合成 |
| *Lost* 影集語音 | 522 | 50（PCA，MFCC+pitch+LPC） | 19 | 語者辨識 | **真實**（劇本字幕與語音強制對齊） |
| TV+movies（*Lost*/*C.S.I.* 人臉） | 10,000 | 50（PCA） | 100 | 人臉命名 | **真實**（劇本場景提及的角色當候選集） |

- **UCI 三個資料集**：標準 UCI repository 表格資料，候選集用論文自己的 p/q/ε 噪聲模型合成生成
  （p=有候選集歧義的樣本比例，q=額外加入的假標籤數，ε=ambiguity degree，控制假標籤集中程度）。
- **FIW（LFW 人臉）**：LFW 人臉照片，裁切成 60×90 灰階，PCA 降到 50 維，候選集同樣是合成的。
- ***Lost* 語音語者辨識**：真實弱監督——522 段語音片段，用 HTK 把劇本字幕跟語音做強制對齊，
  候選集是「這段音軌對應到的那幾句台詞裡出現的角色」，19 個語者。
- **TV+movies（人臉命名，Section 8 的旗艦實驗）**：100 小時的 *Lost* + *C.S.I.* 影片，人臉軌跡用
  OpenCV+part-detector 抽出，候選集是「該場景劇本提到/出現的角色名單」（*Lost* 平均候選集大小
  2.13，*C.S.I.* 2.17），對照約 3,000 張人工標註的 ground truth 人臉（論文貢獻的 "Annotated Faces
  on TV" 資料集，40 個角色，8 集）。旗艦結果：**8-way 角色命名 6% 錯誤率**（32-way 為 13%）。
- **Train/test split**：inductive 實驗 50/50 切分，20 次隨機重跑取平均±標準差；transductive
  實驗訓練/評估用同一批（全部有候選集標註的）樣本，同樣 20 次重跑。
- **評估指標**：平均 0/1 分類錯誤率；人臉命名任務另外用 precision-recall 曲線（含拒絕預測機制）。

### 對 Step 5 的影響

**這意味著 Step 5「用現有 CIFAR pipeline 生成實驗 config」不能宣稱是「複現原論文實驗」**——資料模態
完全不同（表格資料/人臉/語音 vs. 自然影像分類），候選集生成方式也不同（真實弱標註 vs. 本專案的合成
partial label 生成）。Step 5 若要用本專案的 CIFAR pipeline 驗證 `CLPL`，只能定位成「用同一個
squared-hinge loss 公式在 CIFAR 衍生的合成 partial label 資料集上做架構驗證」，不是論文結果的複現。
若要真正複現，需要另外實作 UCI/FIW/Lost 對應的 dataset loader，這超出目前 pipeline 範圍，留給使用者
決定是否要做。

---

## Fixed 版本 — 不需要

Step 2 逐項比對後**沒有發現任何需要修正的數學落差**（見上方比對表，逐項全部 ✅）。
`CLPLSquaredHingeLoss` 已經是論文 Eq. 2（squared-hinge 版本）的忠實實作，**不會產出
`fixed_clpl_loss.py`**。

## 實驗 Config（Step 5）

> **2026-08-14 更新**：上面「對 Step 5 的影響」一節說「本專案沒有實作原論文的 UCI/人臉/語音資料集」
> ——這件事後來做了。`--dataset clpl-lost`（CLPL 論文自己的 1122 張 90×90 *Lost* 人臉截圖，
> **真實**劇本推導候選集）跟 `--dataset lost`（同一組底層資料的 108 維手工特徵版本，可與
> `PRODEN`/`MCL-LOG` 常用的 "Lost" 表格 benchmark 直接比較）都已經在本機驗證過可以端到端訓練。
> 詳見 [00_paper_alignment_guide.md](00_paper_alignment_guide.md) 的「資料集支援」一節。
>
> ```bash
> python scripts/run_pipeline.py run --run_name clpl_original_benchmark \
>     --algorithms CLPL --dataset clpl-lost --epochs 100
> ```
>
> 下方原本的 CIFAR sweep 建議仍然有效，適合跟其他方法做架構層級的橫向比較。

依[主引導文件](00_paper_alignment_guide.md)，pipeline 已在使用者的 server 上驗證可跑。由於 CLPL
不需要 fixed 版本，可以直接用現有的 `CLPLSquaredHingeLoss`（algorithm ID `CLPL`）產生實驗 config。

**重要提醒（承接上方 Step 3 的結論）**：這裡的實驗**不是複現論文結果**，只是在 CIFAR 衍生資料集上
驗證這個 squared-hinge loss 的訓練行為，跟 CLPL/PRODEN/MCL-LOG 等其他方法做橫向比較。

建議掃描設定（沿用本專案既有的 C×k sweep 慣例）：

```bash
python scripts/run_pipeline.py run --run_name clpl_fidelity_check \
    --algorithms CLPL --c_values 5 20 --epochs 200
```

- `c_values 5 20`：對照主引導文件既有的 C×k sweep 慣例（`src/pipeline/data.py` 的 k-schedule）
- 之後可以跟 `PRODEN`（同為 PLL、cross-entropy 型）、`MCL-LOG`（CLL 對照組）的結果並排比較，
  驗證 squared-hinge margin loss 跟 cross-entropy 系方法在候選集大小變化下的行為差異
- 若要更貼近論文本身的實驗設計（Section 7 的 p/q/ε 噪聲模型），需要另外在 `src/data_setup.py`
  加一個對應的候選集生成模式，這超出本次範圍，留待使用者決定是否要做

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
