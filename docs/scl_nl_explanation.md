# SCL-NL：Unbiased Risk Estimators Can Mislead

> **2026-08-14 更新**：Step 2、Step 3 已完成，對照論文 PDF（`C:\Users\User\Desktop\papers\SCL-NL.pdf`）
> 逐條驗證。結論：**單一互補標籤的核心公式（Eq. 11）完全忠實**，**不需要 fixed 版本**。但發現一個
> 值得記錄的事實：本專案的「multi-CL 平均 wrapper」**論文完全沒有討論**——論文的理論（consistency
> 相關論述、Proposition 1-3）全部只針對「每個樣本剛好一個互補標籤」推導，multi-CL 是另一篇論文
> （Feng et al. 2020，也就是 MCL-LOG 那篇）的範圍。這不是「錯誤」（因為沒有論文版本可以拿來對照
> 修正），但文件必須誠實標註這是本專案的外推，不是論文驗證過的用法。
>
> **2026-08-14 二次覆核**：改用 `pypdf` 直接擷取 PDF 純文字重新逐字比對。Eq. (11)
> `φNL(y,g(x)) = −log(1−py)`（Kim et al. 2019 歸屬）逐字確認無誤；"Uniform Assumption: In the
> rest of this paper, we assume CLs are sampled uniformly from [K]\{y}" 跟 "Several studies have
> also extended to learning with multiple complementary labels (Feng et al., 2020)" 這兩句原文
> 也都逐字讀到——**再次確認 multi-CL wrapper 沒有論文依據的判定是正確的**。

**論文來源：** Chou, Y.-T., Niu, G., Lin, H.-T., & Sugiyama, M. (2020).
*Unbiased Risk Estimators Can Mislead: A Case Study of Learning with Complementary Labels.*
ICML 2020, PMLR 119:1929–1938.
[PMLR](https://proceedings.mlr.press/v119/chou20a.html) ·
[arXiv](https://arxiv.org/abs/2007.02235)

**Algorithm ID（pipeline 內部字串）：** `SCL-NL`

---

## 對應程式碼：`src/scl_loss.py` (`SCL_NL`)

- **Single-CL 形式**（46-60 行）：$\phi_{NL}(\bar y,g(x))=-\log(1-p_{\bar y})$，用
  `log1p` 做數值穩定實作，等價於原始公式
- **Multi-CL 平均 wrapper**（62-64 行）：$L(x,\bar Y)=\frac1{|\bar Y|}\sum_{\bar y\in\bar
  Y}\phi_{NL}(\bar y,g(x))$

Entry point：新版 pipeline `run_scl_nl`（`src/pipeline/algorithms/runners.py:138-139`），
`AlgorithmSpec('SCL-NL', 'CLL', r.run_scl_nl)`；舊版 pipeline `setup_scl`（`src/model_setup.py:42-49`）
/ `run_scl_training`（`src/training_pipelines.py:35`）。

---

## 演算法保真度比對（Step 2，已完成）

### 1. 核心公式：完全一致

論文 **Eq. 11**（Section 3.2）："Negative learning loss (SCL-NL)"：
$$\phi_{NL}(\bar y,g(x)) = -\log(1-p_{\bar y})$$
（"NL" 是 "Negative Learning" 的縮寫，公式本身最早來自 Kim et al. 2019 的 NLNL，Chou et al. 把它
納入自己的 SCL 框架當三個 baseline 之一）。程式碼的 `-log1p(-p.clamp(...))` 在數學上完全等價，
只是數值穩定寫法，**逐項比對結果：完全一致，沒有任何落差**。

### 2. 論文的核心論點：URE 為什麼會誤導（跟 MCL-LOG 系方法的哲學差異）

論文標題「Unbiased Risk Estimators Can Mislead」的論證邏輯：
- URE 類方法（例如 MCL-LOG）透過「先用普通標籤設計 loss，再用代數方式把 risk 重寫成互補標籤的
  期望式」得到無偏估計量，但這個重寫過程會引入一個**負的修正項**（Eq. 5 的 `-(K-1)ℓ(ȳ,g(x))`），
  這一項只有在對「全部可能的互補標籤取期望」時才會維持非負，但實際訓練每個樣本只有**一個固定的**
  互補標籤，經驗風險會被訓成負的——這是過擬合的明確訊號（論文 Figure 1）
- SCL 系方法反過來：**先把 0-1 分類誤差用代數方式精確改寫成「互補 0-1 誤差」**（Proposition 2，
  精確等式，不是有雜訊的重寫），**再**對這個互補 0-1 誤差套用平滑代理函數 $\phi$——因為重寫在套
  代理函數之前就做完了，不會引入負修正項，也沒有 URE 的高變異數問題（論文 Table 2-3、Figure 4：
  URE 幾乎零偏差但變異數大好幾個數量級，SCL 犧牲一點偏差換取變異數大幅下降）
- 論文自己的說法："SCL introduces inductive bias towards minimizing the CL likelihood, trading
  zero bias with reduced variance"

**這跟 MCL-LOG 是根本不同的設計哲學**——程式碼 docstring 已經正確點出「這不是 MCL 的 unbiased
risk estimator」，這個定性描述是對的。

### 3. Consistency 保證：比預期弱，值得記錄

**論文並沒有針對 SCL-NL 給出一個編號的 consistency/classification-calibration 定理**。實際證明的是：
- **Proposition 1**：反向重寫得到的 risk 是原始風險的精確無偏估計量（在 transition matrix 可逆
  的假設下）——這是關於 URE 的性質，不是 SCL 的
- **Proposition 2**：互補 0-1 誤差跟原始 0-1 誤差的無偏估計量之間差一個常數倍：
  $R(g;\ell_{01})=(K-1)\cdot\bar R(g;\bar\ell_{01})$——這是一個**精確代數等式**，是「最小化互補
  0-1 誤差」跟「最小化原始分類誤差」有同一個最小值點的依據，但這是針對**不可微的 0-1 loss**，
  不是針對 $\phi_{NL}$ 這個平滑代理函數的 calibration 證明
- **Proposition 3**：URE 的梯度也是無偏的（UGE）——一樣是關於 URE，用來論證「無偏不代表訓練穩定」
- 論文明確把「consistency 定理」這頂帽子讓給其他論文（"theoretical results from Ishida et al.
  (2017) proved the consistency of the risk estimator..."），不是自己宣稱證明過

**結論**：如果之前有任何文件（包括本專案 README/CLAUDE.md）宣稱「論文證明了 SCL-NL 是一致估計量」，
這是過度陳述。論文的支持證據主要是實證的（準確率、梯度方向 cosine similarity、bias-variance
分解），加上一個關於不可微 0-1 loss 的精確等式，而不是對 $\phi_{NL}$ 本身的 calibration 定理。

### 4. Single vs Multi-CL：關鍵發現

**論文從頭到尾只處理「每個樣本剛好一個互補標籤」的情況**——形式定義（"the complementary label
$\bar y_i$ is **a class**..."，單數不是集合）、transition matrix、Eq. 4/5 的重寫、Proposition
2/3、Eq. 11 本身的記號 $\phi_{NL}(\bar y,\cdot)$（單一 $\bar y$，不是集合 $\bar Y$）、以及**所有**
實驗都是單一互補標籤設定。

Multiple complementary labels 在論文裡只出現一次，而且是當成別人的相關工作提到：
> "Several studies have also extended to learning with multiple complementary labels
> (Feng et al., 2020)..."

Feng et al. 2020 就是本專案 `MCL-LOG` 對應的那篇論文——是一篇**獨立**的論文，Chou et al. 2020
只是引用，沒有採用、沒有延伸、也沒有分析 multi-CL 設定。

**結論：本專案 `SCL_NL` 的 multi-CL 平均 wrapper 在 Chou et al. (2020) 裡沒有任何理論或實證依據**。
這是一個合理的工程延伸（把單一標籤公式逐元素套用到集合上再平均），程式碼自己的 docstring 也誠實
承認這是一個 "wrapper"，但論文的 uniform-CL 假設（$T=\frac1{K-1}(\mathbf{1}_K-I_K)$）跟本專案
「多個互補標籤」的資料生成方式（PL 生成的補集）本來就不是同一回事——這不是程式碼寫錯，而是
「拿一個只驗證過單一標籤設定的 loss，套用到本專案的多標籤資料上」，沒有論文驗證過的擔保。

### 5. 論文裡的其他 SCL 變體（脈絡參考）

論文 Eq. 10-12 統一了三個 baseline：**SCL-FWD**（forward correction，Yu et al. 2018 提出）、
**SCL-NL**（本文件對應的，Kim et al. 2019 提出）、**SCL-EXP**（$\phi_{EXP}=\exp(p_{\bar y})$，
Chou et al. 自己提出）。本專案目前只有 `SCL_NL` 這一個。

---

## 原論文使用的 Benchmark（Step 3，已完成）

- **資料集**：MNIST、Kuzushiji-MNIST、Fashion-MNIST、CIFAR-10。**沒有 CIFAR-20，沒有 CLCIFAR**。
- **模型架構**：MNIST 系列用線性模型 + 單隱藏層 MLP（`d-500-10`）；CIFAR-10 用 **ResNet-34**
  跟 **DenseNet**（本專案目前是 ResNet-18）。
- **互補標籤生成**：uniform 假設——從 $[K]\setminus\{y\}$ 均勻隨機抽**一個**互補標籤（Sec 2.2）。
- **訓練設定**：
  - 準確率比較（Table 1）：Adam，learning rate 網格搜尋 $\{10^{-1},...,10^{-5}\}$，300 epoch
  - 梯度分析（Section 4）：SGD，固定 learning rate $10^{-2}$，300 epoch，只用普通標籤梯度更新
    模型（互補梯度只算出來比較方向，不拿去更新參數，確保比較公平）
- **評估**：test accuracy（Table 1）；互補梯度跟普通梯度的 cosine similarity（Figure 3，
  Expected/Fixed 兩種設定）；梯度的 MSE/Bias²/Variance 分解（Figure 4、Table 2-3）。

---

## Fixed 版本 — 不需要

單一互補標籤的核心公式（Eq. 11）逐項比對完全一致，**不會產出 `fixed_scl_loss.py`**。

Multi-CL 平均 wrapper 雖然沒有論文依據，但也**沒有論文提供的「正確」版本可以拿來修正**——論文
根本沒討論這個設定，所以無法定義什麼是「修正後跟論文一致」的版本。這點只記錄為文件層級的誠實
揭露（見上方 Step 2 第 4 項），不產出 fixed 程式碼。

---

## 實驗 Config（Step 5）

> **2026-08-14 更新**：論文用的 MNIST/Kuzushiji-MNIST/Fashion-MNIST 現在可以透過 `--dataset` 直接
> 跑，已在本機驗證過端到端訓練。詳見 [00_paper_alignment_guide.md](00_paper_alignment_guide.md)
> 的「資料集支援」一節。
>
> ```bash
> python scripts/run_pipeline.py run --run_name scl_nl_original_benchmark \
>     --algorithms SCL-NL --dataset mnist --epochs 200
> ```

```bash
python scripts/run_pipeline.py run --run_name scl_nl_check \
    --algorithms SCL-NL --c_values 5 20 --epochs 200
```

- 因為 multi-CL 平均沒有論文依據，**建議特別在 `--c_values` 用較小的 C（例如固定用單一互補標籤的
  k schedule）額外跑一組**，讓 SCL-NL 在「單一互補標籤」這個論文真正驗證過的設定下也有一組結果
  可以對照，跟 multi-CL 設定下的結果分開看
- 可以跟 `MCL-LOG`（URE 家族代表）並排比較，驗證論文的核心論點（URE 變異數大 vs SCL 偏差小）
  在深度學習 + CIFAR 設定下是否還成立
