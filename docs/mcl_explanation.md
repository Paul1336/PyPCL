# MCL-LOG（及 MAE / EXP 變體）：Learning with Multiple Complementary Labels

> **2026-08-14 更新**：Step 2、Step 3 已完成，對照論文 PDF（`C:\Users\User\Desktop\papers\MCL-LOG.pdf`）
> 逐條驗證。**發現一個實質性數學錯誤**：unbiased risk estimator 的 scaling factor 程式碼寫成
> `(C-1)/(C-m)`，論文推導出來的其實是 `2(C-1)/m`——兩者隨 `m`（互補標籤數）變化的方向是**相反的**。
> 已產出 `src/fixed_mcl_losses.py` 修正版，並註冊進新版 pipeline（algorithm ID `MCL-LOG-Fixed`）。
>
> **2026-08-14 二次覆核**：改用 `pypdf` 直接擷取 PDF 純文字重新逐字比對，這次直接讀到 Eq. (12)
> 原文：`R̂(f) = (k-1)/|Ȳ_i| Σ_{y∉Ȳ_i} L_MAE(f(x_i),y) = (2k-2)/|Ȳ_i| L'_MAE(f(x_i),Ȳ_i) + Z_i`
> ——scaling factor 是 `(2k-2)/|Ȳ_i|` = `2(k-1)/m`，跟先前的結論完全一致，**bug 判定再次確認為
> 正確**。Eq. (11) 的一般化 estimator、"By replacing L'_MAE by L_EXP and L_LOG in Eq. (12)" 這句
> 原文也都逐字核對過。

**論文來源：** Feng, L., Kaneko, T., Han, B., Niu, G., An, B., & Sugiyama, M. (2020).
*Learning with Multiple Complementary Labels.* ICML 2020, PMLR 119:3072–3081.
[PMLR](https://proceedings.mlr.press/v119/feng20a.html)

**Algorithm ID（pipeline 內部字串）：** `MCL-LOG`（修正版：`MCL-LOG-Fixed`）

---

## 對應程式碼：`src/mcl_losses.py`

三個 class（`MCL_LOG`、`MCL_MAE`、`MCL_EXP`）結構相同，只有最後的 loss 函數形式不同：

| Class | 檔案行號 | Loss 公式（scaling 前） |
|---|---|---|
| `MCL_LOG` | `mcl_losses.py:5-34` | `loss = -log(sum_probs_not_in_complementary + 1e-7)` |
| `MCL_MAE` | `mcl_losses.py:36-65` | `loss = 1 - sum_probs_not_in_complementary` |
| `MCL_EXP` | `mcl_losses.py:67-96` | `loss = exp(-sum_probs_not_in_complementary)` |

三者算完 base loss 後都乘上同一個 `scaling_factor = (C-1)/(C-num_complementary)`。

Entry point：新版 pipeline只有 `MCL_LOG` 註冊——`run_mcl_log`（`src/pipeline/algorithms/runners.py:134-135`），
`AlgorithmSpec('MCL-LOG', 'CLL', r.run_mcl_log)`（`src/pipeline/algorithms/__init__.py:37`）；
舊版 pipeline `setup_mcl(loss_type='log'|'mae'|'exp')`（`src/model_setup.py:28-40`）三種都支援。

---

## 演算法保真度比對（Step 2，已完成）

### 論文原始推導

論文 Section 4.3（"Practical Implementation"）先定義三個 loss 的**未縮放**形式（$m=|\bar Y|$，
$C$ 為總類別數）：

- $L'_{\text{MAE}} = 1 - \sum_{j\notin\bar Y} p_\theta(j|x)$
- $L_{\text{EXP}} = \exp\!\big(-\sum_{j\notin\bar Y}p_\theta(j|x)\big)$
- $L_{\text{LOG}} = -\log\!\big(\sum_{j\notin\bar Y}p_\theta(j|x)\big)$

**這三個未縮放公式跟程式碼逐項核對，完全一致**——函數形式沒有問題。

問題出在 **unbiased risk estimator 的 scaling factor**。論文推導分兩階段：

**階段 A（Theorem 3，Eq. 8-10）**：對任意 loss $L$ 的一般化無偏估計量
$$\bar L_j(f(x),\bar Y) = \sum_{y\notin\bar Y}L(f(x),y) - \frac{k-1-j}{j}\sum_{y'\in\bar Y}L(f(x),y')$$
（$k$=類別數，$j=|\bar Y|$；當 $j=1$ 時退化成 Ishida et al. 2019 的單一互補標籤 FREE 估計量）

**階段 B（Eq. 12，MAE 專屬的代數化簡）**：因為 MAE 是對稱的
（$\sum_{y=1}^k L_{\text{MAE}}(f(x),y)=2k-2$，是常數），把 $L_{\text{MAE}}$ 代入 Eq. 11 後可以化簡成：
$$\hat R(f) = \frac{2k-2}{|\bar Y_i|}\cdot L'_{\text{MAE}}(f(x_i),\bar Y_i) + Z_i$$
（$Z_i$ 是跟 $f$ 無關的常數）。**論文接著把 $L_{\text{EXP}}$、$L_{\text{LOG}}$ 直接代入這個
「MAE 化簡後」的 Eq. 12 框架**（原文："By replacing $L'_{\text{MAE}}$ by $L_{\text{EXP}}$ and
$L_{\text{LOG}}$ in Eq. (12), we obtain two new methods"），所以論文實際使用的最終訓練目標是：

$$\text{Loss}_i = \frac{2(C-1)}{m_i}\cdot L_{\text{LOG or EXP}}(f(x_i),\bar Y_i)$$

即 scaling factor 是 **$2(C-1)/m$**，不是程式碼裡的 $(C-1)/(C-m)$。

### 兩個公式方向相反

| $m$（互補標籤數） | 論文 $2(C-1)/m$ | 程式碼 $(C-1)/(C-m)$ |
|---|---|---|
| $m=1$（單一互補標籤） | $2(C-1)$ —— **最大** | $1$ —— 最小 |
| $m\to C-1$（接近類別數） | $2$ —— 最小 | $(C-1)$ —— **最大**，且 $m\to C$ 時發散 |

具體數字（以 CIFAR-10，$C=10$ 為例）：$m=1$ 時論文係數是 **18**，程式碼算出來是 **1**（差 18 倍）；
$m=8$ 時論文係數是 **2.25**，程式碼算出來是 **9**（差 4 倍，而且方向相反）。這不是縮放常數的
cosmetic 差異，而是**改變了每個樣本依互補標籤數量該被加多少權重**，直接影響梯度大小與樣本間的
相對權重，尤其在本專案的 "variable" 生成模式（每個樣本的 $m$ 不固定）下影響更明顯。

（可能的錯誤來源，僅供參考：論文 Table 1 定義了一個叫 "Supervision Purity" 的診斷量
$1/(k-s)$，分母形狀跟程式碼的 $(C-m)$ 一樣，但那只是一個說明「監督訊號被稀釋程度」的示意指標，
不是 Eq. 12 的 risk estimator 係數，分子也是 1 不是 $C-1$，推測程式碼是把這兩個東西搞混了。）

### Multi-CL 處理方式：一致

論文的核心貢獻是**把整個互補集 $\bar Y$ 當一個整體處理**（Section 4.2："processes each set of MCLs
as a whole"），跟論文自己討論的另一種「拆解法」（decomposition wrapper，把多標籤拆成多個單標籤
分別訓練）相對，論文 Table 1 證明拆解法會稀釋監督訊號純度（從 $1/(k-s)$ 稀釋到 $1/(k-1)$），
Table 2-4 顯示整體法在幾乎所有資料集上都贏過拆解法。程式碼用一個 `mask_complementary` mask
一次算 $\sum_{c\notin\bar Y}p_c$，正是論文的「整體法」，這部分**完全一致**。

### 候選集/互補集生成假設

論文 Section 3.1/5：先按 $p(s)=\binom{k}{s}/(2^k-2)$ 抽出互補集大小 $s$，再從所有不含真實標籤、
大小為 $s$ 的子集中**均勻隨機**抽一個。本專案是把 PL（partial label）生成的補集當 CL
（constant-k 或 variable-q 兩種模式，外加可選的 noise η），生成機制跟論文不同——這點需要在
`src/data_setup.py` 進一步確認，超出這次 agent 讀論文的範圍，記錄為待查項目。

---

## 原論文使用的 Benchmark（Step 3，已完成）

- **資料集**：MNIST、Kuzushiji-MNIST、Fashion-MNIST、20Newsgroups、CIFAR-10，加上 4 個 UCI 資料集
  （Yeast、Texture、Dermatology、Synthetic Control）。**沒有 CIFAR-20，沒有 CLCIFAR**。
- **互補標籤生成**：$p(s)=\binom{k}{s}/(2^k-2)$ 抽大小、再均勻抽子集，沒有另外的 noise 參數
  （論文沒有本專案 `η` 這種噪聲旋鈕）。
- **模型架構**：線性模型、MLP（`d-500-k`）、**ResNet-34**、**DenseNet-22**（CIFAR-10 用這兩個，
  本專案目前是 ResNet-18，深度不同）。
- **訓練設定**：**Adam**，batch size **256**，**250** epoch，learning rate 跟 weight decay 都在
  $\{10^{-6},...,10^{-1}\}$ 網格搜尋，用 10% held-out validation 挑最佳超參數，V100 GPU。
  本專案目前預設是 SGD 風格（`--momentum`/`--weight_decay` CLI 參數），batch size 512，
  最多 1000 epoch——這是全專案共用設定，不是 MCL-LOG 專屬調整。
- **評估**：5 次重跑的平均±標準差 test accuracy，paired t-test 顯著性標記。

---

## Fixed 版本（Step 4，已完成）

`src/fixed_mcl_losses.py`，新增 `FixedMCLLog`、`FixedMCLMae`、`FixedMCLExp` 三個 class
（三者都受同一個 scaling factor 錯誤影響，一併修正）：

- 唯一的修改：`scaling_factor = (self.num_classes - 1) / (self.num_classes - num_complementary)`
  → `scaling_factor = 2.0 * (self.num_classes - 1) / num_complementary`（對照論文 Eq. 12）
- Loss 函數本體（LOG/MAE/EXP 三個公式）不變，因為那部分已確認跟論文一致
- 已在新版 pipeline 註冊 `MCL-LOG-Fixed`（`AlgorithmSpec('MCL-LOG-Fixed', 'CLL', r.run_mcl_log_fixed)`）

**已知落差（沿用主引導文件）**：`MCL_MAE`/`MCL_EXP`（以及對應的 `FixedMCLMae`/`FixedMCLExp`）目前
只有 `MCL-LOG`/`MCL-LOG-Fixed` 真的註冊進新版 pipeline 可以直接跑；MAE/EXP 變體如果要透過新版
pipeline 跑，需要額外新增 `AlgorithmSpec` 條目，這次先把 class 寫好，註冊留給下一輪視需要決定。

---

## 實驗 Config（Step 5）

> **2026-08-14 更新**：論文用的 MNIST/Fashion-MNIST/Kuzushiji-MNIST/20Newsgroups + UCI 表格資料
> （**yeast/texture/dermatology/synthetic-control**，`texture` 不在 `ucimlrepo` 裡，改抓 OpenML
> id=40499）現在都可以透過 `--dataset` 直接跑，且都在本機驗證過端到端訓練（真實 loss 下降、
> 準確率上升）。詳見 [00_paper_alignment_guide.md](00_paper_alignment_guide.md) 的「資料集支援」
> 一節。**這也是驗證 scaling factor bug 影響的好機會**——這些真實資料集裡每個樣本的互補標籤數
> 不一定是固定值（尤其 20Newsgroups/UCI 資料，取決於 `--dataset` 底層的候選集生成方式），原版跟
> Fixed 版在這些資料上的差異可能比 CIFAR 衍生資料更明顯。
>
> ```bash
> python scripts/run_pipeline.py run --run_name mcl_log_original_benchmark \
>     --algorithms MCL-LOG MCL-LOG-Fixed --dataset mnist --epochs 200
> python scripts/run_pipeline.py run --run_name mcl_log_original_benchmark \
>     --algorithms MCL-LOG MCL-LOG-Fixed --dataset 20newsgroups --epochs 100
> for ds in yeast texture dermatology synthetic-control; do
>     python scripts/run_pipeline.py run --run_name mcl_log_original_benchmark \
>         --algorithms MCL-LOG MCL-LOG-Fixed --dataset "$ds" --epochs 200
> done
> ```

```bash
python scripts/run_pipeline.py run --run_name mcl_log_bugfix_check \
    --algorithms MCL-LOG MCL-LOG-Fixed --c_values 5 20 --epochs 200
```

- **建議把 `MCL-LOG`（有 bug 的原版）跟 `MCL-LOG-Fixed`（修正版）並排跑**，這是驗證這個 scaling
  factor 錯誤是否真的影響最終準確率的直接方法。**修正**：新版 pipeline（`scripts/run_pipeline.py`）
  只支援 constant-k 生成（每個 (C,k) cell 裡所有樣本的互補標籤數 $m=C-k$ 是固定的，沒有
  per-sample 變動的 "variable" 模式可以透過 CLI 選——那是舊版 `scripts/run_experiment.py` 才有的
  `--type variable` 選項，新版 pipeline 沒有對應旗標）。既然 $m$ 在單次 (C,k) run 裡是常數，
  這個 bug 的效果會展現在**跨 k 值的 sweep**上：`--c_values 5 20` 搭配預設 k-schedule
  （`src/pipeline/data.py::get_k_values`，$k$ 從 1 掃到 $C-1$，對應 $m$ 從 $C-1$ 掃到 1）會讓
  兩個係數公式在不同 k 值下產生不同方向的偏移，畫出 accuracy-vs-k 曲線後兩個版本的形狀差異應該
  在 k 極端值（$m$ 很大或很小）處最明顯。
- 若要更貼近論文設定（Adam、batch 256、250 epoch、ResNet-34/DenseNet-22），可以另開一組
  `--batch_size 256 --epochs 250` 的對照；optimizer 這項不需要改，`src/pipeline/algorithms/hparams.py`
  的 `MCL-LOG`/`MCL-LOG-Fixed` 已經是 Adam，跟論文一致。
