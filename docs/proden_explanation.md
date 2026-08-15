# PRODEN：Progressive Identification of True Labels for Partial-Label Learning

> **2026-08-14 更新**：Step 2（公式逐項比對）與 Step 3（benchmark 查證）已完成，直接對照論文 PDF
> （`C:\Users\User\Desktop\papers\PRODEN.pdf`）逐條驗證。結論：**目前主 pipeline 實際使用的
> `ProdenLoss` 忠實對應論文 Algorithm 1，不需要 fixed 版本**；但同檔案裡的第二個 class `proden`
> （只有舊版 pipeline 會用到）有**實質性的演算法錯誤**，詳見下方。
>
> **2026-08-14 二次覆核**：改用 `pypdf` 直接擷取 PDF 純文字重新逐字比對。Algorithm 1 的步驟順序
> （第 6 行用「目前存好的」w 算 loss、第 8 行才用這一步的 forward 結果更新 w、第 9 行才更新 Θ）、
> Eq. 9（$w^0_{ij}=1/|s_i|$）、Eq. 10（$w_{ij}=g_j(x_i)/\sum_{k\in s_i}g_k(x_i)$）全部逐字確認
> 跟下方比對表一致。沒有發現需要修正的地方。

**論文來源：** Lv, J., Xu, M., Feng, L., Niu, G., Geng, X., & Sugiyama, M. (2020).
*Progressive Identification of True Labels for Partial-Label Learning.* ICML 2020, PMLR 119:6500–6510.
[PMLR](https://proceedings.mlr.press/v119/lv20a.html) ·
[arXiv](https://arxiv.org/abs/2002.08053)

**Algorithm ID（pipeline 內部字串）：** `PRODEN`

---

## 對應程式碼

`src/proden_loss.py` 裡有 **兩個** class：

### 1. `ProdenLoss`（`src/proden_loss.py:6-49`）—— **新版 pipeline 實際使用的版本**

- 跨 epoch 信心累積（persistent buffer）：`__init__`（21-29 行）為每個訓練樣本建立 `conf[N, C]`
  buffer，初始值為候選集內均等分佈（`1/k`）。
- `forward(outputs, indices)`（31-49 行）：
  - Loss：用**上一步存下來的** `conf[indices]` 當 soft label 權重算加權 CE
  - Update（`torch.no_grad()`，43-47 行）：用**這一步**模型的 softmax，限制在候選集內重新正規化，
    寫回 `self.conf[indices]`，供下次呼叫使用
- Entry point：新版 pipeline `run_proden`（`src/pipeline/algorithms/runners.py:194-222`），
  `AlgorithmSpec('PRODEN', 'PLL', r.run_proden)`（`src/pipeline/algorithms/__init__.py:31`）

### 2. `proden`（`src/proden_loss.py:52-75`）—— **只有舊版 pipeline 會用到**

- 每次 `forward` 都重新計算權重，沒有 persistent buffer，且權重計算**沒有 detach**（見下方 Step 2
  比對，這是問題所在）。
- Entry point：舊版 pipeline `setup_proden`（`src/model_setup.py:21-26`）/
  `run_proden_training`（`src/training_pipelines.py:15`）

**結論（原本 TODO 已確認）**：Step 5 實驗用的是新版 pipeline（`scripts/run_pipeline.py`），走的是
`run_proden` → `ProdenLoss`，也就是忠實版本。`proden` 只存在於舊版 `scripts/run_experiment.py` 路徑，
不影響 Step 5 的實驗結果。

---

## 演算法保真度比對（Step 2，已完成）

### 論文原始演算法（Algorithm 1，論文 p.4）

論文的經驗風險（Eq. 8）：
$$\widehat{\mathcal{R}}_{\mathrm{PLL}} = \frac1n\sum_{i=1}^n\sum_{j=1}^c \mathrm{w}_{ij}\,\ell(g_j(x_i), e_j^{s_i})$$

用 CE 展開就是 $-\sum_{j\in s_i}\mathrm{w}_{ij}\log g_j(x_i)$。權重更新規則（Eq. 10，"identification" 步驟）：
$$\mathrm{w}_{ij} = \begin{cases} g_j(x_i)\big/\sum_{k\in s_i}g_k(x_i) & j\in s_i \\ 0 & \text{否則}\end{cases}$$

**關鍵在 Algorithm 1 的步驟順序**（論文 p.4，逐行）：
```
6:  用「目前存好的」w 計算 loss L（Eq. 8）
7:  計算梯度 −∇_Θ L
8:  用「這一步」模型的 forward 結果，依 Eq. 10 更新 w（在 Θ 更新之前）
9:  用梯度更新 Θ
```
也就是：算 loss 用的是**上一次**存的舊權重，算完 loss 之後才用**這一次**的 forward 結果刷新權重
（供下次使用）——兩者之間有一步的 lag。而且論文明確把 $\mathrm{w}$ 當成 EM 演算法裡的**潛變數**
（"Because the weights are latent, the minimizer of Eq. (8) cannot be solved directly"），
不應該讓梯度穿過權重計算本身。

初始化（Eq. 9）：候選集內均等分佈 $1/|s_i|$，其餘為 0——與程式碼一致。

### 逐項比對結果

| 論文 Algorithm 1 | `ProdenLoss` | `proden` |
|---|---|---|
| 初始化 $\mathrm{w}^0_{ij}=1/\|s_i\|$（Eq. 9） | ✅ `conf[i,j]=1.0/k` | ✅ 每次重算，數學形式相同 |
| Loss 用**舊**權重（Algorithm 1 第 6 行） | ✅ 用 `self.conf[indices]`（更新前的值） | ❌ 用**這一步自己**的 forward 結果，沒有 lag |
| 權重更新在 loss 算完之後才刷新（第 8 行） | ✅ 43-47 行在 `torch.no_grad()` 內，loss 算完後才更新 | ❌ 不適用，因為根本沒有分開兩步 |
| $\mathrm{w}$ 是潛變數，梯度不應穿過權重計算 | ✅ `torch.no_grad()` 正確 detach | ❌ **`weights` 完全沒有 detach**，梯度會同時流過 log-prob 項和權重項 |

**`ProdenLoss` 判定：忠實對應 Algorithm 1**，包括初始化、權重更新公式、loss 形式，以及最關鍵的
「loss 用舊權重、算完才刷新」這個時序設計。

**`proden` 判定：有實質性錯誤，不是論文演算法的有效簡化**。論文附錄有三個消融基準
（PRODEN-itera：每 100 epoch 才更新一次權重；PRODEN-sudden：權重改成 one-hot 硬性 argmax，
每步更新；PRODEN-naive：權重永遠不更新），`proden` 都對不上——它既不是這三者之一，也偏離了
Algorithm 1 的「潛變數 + 有 lag」設計。論文的消融結果顯示 PRODEN-sudden（三者中最接近「積極、
無延遲識別」的版本）系統性地比 PRODEN 差（"the sudden identification concentrates all the weights
to the winners, resulting in their poorer performance"）——`proden` 因為權重沒有 detach、又沒有 lag，
比 PRODEN-sudden 更激進，有自我強化（"贏者全拿"）collapse 的風險，理論性質比論文任何一個消融基準
都弱。

**由於 `proden` 不在新版 pipeline（Step 5 實驗的執行路徑）裡，這個錯誤不影響本輪實驗結果**，
只記錄為已知缺陷（見下方「已知缺陷」與 [主引導文件](00_paper_alignment_guide.md)）。

### 其他論文內容

- 論文**沒有**額外提供 Algorithm 1 的收斂/單調遞減證明；理論結果集中在 **Theorem 1**
  （classifier-consistency，population risk 版本，需要 ambiguity degree $\gamma<1$）與
  **Theorem 2**（Rademacher complexity generalization bound），兩者都不是針對 Algorithm 1 這個
  「加權訓練動態」本身的收斂證明，而是針對風險估計量的統計性質。
- 論文提出 PRODEN 的動機論述（非定理）：傳統 EM 方法要求 M-step 內完全收斂才能進下一個 E-step，
  容易對雜訊初始先驗過擬合；PRODEN 把 E-step（權重更新）跟 M-step（梯度更新）合併，讓權重可以
  每個 epoch 甚至每個 batch 更新，不需要 epoch 內先收斂——這是「每個 batch 更新一次權重」設計的
  理由，但沒有要求「同一步內零延遲」。

---

## 原論文使用的 Benchmark（Step 3，已完成）

- **資料集**：MNIST、Fashion-MNIST、Kuzushiji-MNIST、CIFAR-10（binomial flipping + pair flipping
  兩種候選集生成），加上 5 個 UCI 資料集（Yeast、Texture、Dermatology、Synthetic Control、
  20Newsgroups）與 5 個真實世界 PLL 資料集（Lost、Birdsong、MSRCv2、Soccer Player、Yahoo! News）。
  **沒有 CIFAR-20，也沒有 CLCIFAR**（這兩個是本專案自己擴充的資料集，論文沒有評測過）。
- **候選集生成**：binomial flipping，機率 $q=\Pr(\tilde y=1\mid y=0)$，對每個負類獨立做
  Bernoulli($q$) 試驗；若全部沒翻轉，強制翻轉一個負類確保每個樣本都有候選集。實驗用
  $q=0.1$（"less-partial"）與 $q=0.7$（"strong-partial"）。另有 pair-flipping（只在語意相近的
  類別間翻轉）用於控制 ambiguity degree $\gamma$。
- **模型架構**：MNIST 系列用線性模型 + 5 層 MLP；**CIFAR-10 用 12 層 ConvNet（Laine & Aila 2017）
  和 32 層 ResNet**（本專案目前用的是 ResNet-18，深度不同）；UCI/真實世界資料集只用線性模型。
- **訓練設定**：SGD + momentum 0.9，batch size **256**，**500** epochs；UCI/真實世界資料集用
  5-fold cross-validation + paired t-test；PRODEN 的準確率取**最後 10 個 epoch** 平均（用來展示
  對過擬合的穩定性）。
- **對照本專案**：CIFAR-10/20/100-subset 用 ResNet-18、batch size 512（預設）、最多 1000 epoch，
  跟論文的 32 層 ResNet / batch 256 / 500 epoch 不同——這是專案級的統一設定（所有 6 個方法共用），
  不是 PRODEN 專屬的調整，Step 5 設計 config 時可以視需要另外開一組貼近論文設定的版本做對照。

---

## Fixed 版本

**`ProdenLoss`（新版 pipeline 使用的版本）不需要修正**——已確認忠實對應論文 Algorithm 1。

**`proden`（舊版 pipeline 專用）有實質錯誤，但因為不在 Step 5 的實驗路徑上，這次不產出
`fixed_proden_loss.py`**。原因：
1. 新版 pipeline（`run_proden`）本來就用的是正確的 `ProdenLoss`，Step 5 實驗不會受影響
2. 修正 `proden` 只會影響舊版 `scripts/run_experiment.py` 這條已經逐步被取代的路徑

若未來要清理舊版 pipeline 或有人手動呼叫 `setup_proden`，建議：在 `proden.forward` 裡把
`predictions`/`weights` 包進 `with torch.no_grad(): ... .detach()`，並改成使用上一步存的權重
（等同於把 `proden` 改造成 `ProdenLoss` 的無狀態近似），而不是簡單修修補補——這已經記錄在
[主引導文件](00_paper_alignment_guide.md) 的已知缺陷清單，交由需要用到舊版 pipeline 的 session
決定是否處理。

---

## 實驗 Config（Step 5）

> **2026-08-14 更新**：論文用的 MNIST/Fashion-MNIST/Kuzushiji-MNIST、UCI 表格資料
> （**yeast/texture/dermatology/synthetic-control**——注意不是 ecoli/abalone，那兩個是 CLPL
> 論文用的另一組 UCI 資料集，只有 dermatology 重疊）、以及 5 個真實世界 PLL 資料集
> （Lost/MSRCv2/BirdSong/Soccer Player/Yahoo!News）現在都可以透過 `--dataset` 直接跑，全部都在
> 本機驗證過端到端訓練。詳見 [00_paper_alignment_guide.md](00_paper_alignment_guide.md) 的
> 「資料集支援」一節。
>
> ```bash
> python scripts/run_pipeline.py run --run_name proden_original_benchmark \
>     --algorithms PRODEN --dataset mnist --epochs 200
> for ds in yeast texture dermatology synthetic-control; do
>     python scripts/run_pipeline.py run --run_name proden_original_benchmark \
>         --algorithms PRODEN --dataset "$ds" --epochs 200
> done
> for ds in lost msrcv2 birdsong soccer-player yahoo-news; do
>     python scripts/run_pipeline.py run --run_name proden_original_benchmark \
>         --algorithms PRODEN --dataset "$ds" --epochs 200
> done
> ```

Pipeline 已在使用者的 server 上驗證可跑。`PRODEN` 走的是忠實的 `ProdenLoss`，可以直接用：

```bash
python scripts/run_pipeline.py run --run_name proden_check \
    --algorithms PRODEN --c_values 5 20 --epochs 200
```

- 若要更貼近論文本身的訓練規模（batch 256、500 epoch、32 層 ResNet），可以另開一組
  `--batch_size 256 --epochs 500` 的對照 config，但 backbone 深度（ResNet-18 vs 論文的 32 層
  ResNet）目前無法透過 CLI 調整，需要改 `src/models.py`，超出本次範圍。
- 建議跟 `CLPL`、`MCL-LOG` 並排比較（PLL vs CLL 的 k 值交叉點分析，沿用主引導文件的 sweep 慣例）。
