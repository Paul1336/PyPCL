# ComCo：Complementary Supervised Contrastive Learning for Complementary Label Learning

> **2026-08-14 更新**：Step 2、Step 3 已完成，對照論文 PDF（`C:\Users\User\Desktop\papers\ComCo.pdf`）
> 逐條驗證。結論：**對比損失（正/負樣本選取策略、warm-up 排程、超參數預設）全部精確對應論文**，
> 這部分做得非常好，連論文自創的 "Strategy B"、"Eq. 9 (strategy 1)" 這種命名都對得上。**分類損失
> 有實質性落差**：論文 Section 5.1 明講 ComCo 用的是**不加權的 SCL-NL**，程式碼卻套用了跟
> MCL-NL 一樣的 `(C-1)/(C-m)` scaling（借用自另一篇論文），只有在單一互補標籤（m=1）時剛好退化
> 成跟論文一致，多標籤時就不對了。已產出 `src/comco/fixed_utils_loss.py` 修正。
>
> **2026-08-14 二次覆核**：改用 `pypdf` 直接擷取 PDF 純文字重新逐字比對。Eq. (3) 原文
> `ℒ_cls(x_i,ȳ_i) = L̄(g(x_i),ȳ)，where L̄(g(x_i),ȳ) represents an arbitrary complementary loss`
> ——確認真的只是個通用 wrapper；"For ComCo, we choose SCL-NL as complementary loss ℒ_cls" 跟
> "MCL-NL (Feng et al., 2020) and MCL-EXP (Feng et al., 2020) are the extension of SCL-NL and
> SCL-EXP to multi-complementary label scenarios" 這兩句原文也逐字讀到——**再次確認分類損失
> scaling 的 bug 判定正確**。"we denote the strategy in PiCO as strategy A and the strategy of
> ComCo as strategy B"、K=1/warmup_pos=100/warmup_neg=1 的原文也都核對過，對比損失部分維持
> 「完全忠實」的結論不變。

**論文來源：** Jiang, H., Sun, Z., & Tian, Y. (2024).
*ComCo: Complementary Supervised Contrastive Learning for Complementary Label Learning.*
Neural Networks, Vol. 169, pp. 44–56. DOI: 10.1016/j.neunet.2023.10.013
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608023005683) ·
[PubMed](https://pubmed.ncbi.nlm.nih.gov/37857172/)

**Algorithm ID（pipeline 內部字串）：** `ComCo`（修正版：`ComCo-Fixed`）

---

## 對應程式碼

### 模型架構：`src/comco/model.py` (`ComCoModel`)

雙 momentum encoder + queue（跟 PiCO 架構相似），但 pseudo label 是**未加 mask 的**
`cls_out.argmax(dim=1)`（46 行）——因為 CLL 沒有候選集，這點跟 PiCO 用候選集 mask 不同。
Queue 額外存 `queue_comp`（互補標籤 mask），供負樣本選取用。

### Loss：`src/comco/utils_loss.py`

- **`ComCoCLSLoss`**（分類損失，docstring 標註 "Eq. 3 in paper"）：
  $L=-\log(\sum_{c\notin\bar Y}\text{softmax}(logits)_c)\cdot\frac{C-1}{C-m}$，跟 `MCL_LOG`
  數學形式幾乎一樣
- **`ComCoContrastiveLoss`**（對比損失）：正樣本集合固定含 key-view，`warmup_pos` 後加入
  top-K 同 pseudo-label 鄰居（"Strategy B from paper"）；負樣本集合 `warmup_neg` 後選
  $\text{Dist\_min}$ 最大的單一互補標籤子集

Entry point：`run_comco`（`src/pipeline/algorithms/runners.py:388-421`），
`AlgorithmSpec('ComCo', 'CLL', r.run_comco)`。

---

## 演算法保真度比對（Step 2，已完成）

### 1. 分類損失：**發現實質落差**

論文 **Eq. 3**（Section 4.1）其實只是一個通用 wrapper："$\mathcal L_{cls}(x_i,\bar y_i) =
\bar L(g(x_i),\bar y_i)$，其中 $\bar L$ 代表**任意**互補損失（在 Section 3.2 定義）"——Eq. 3
本身不是一個具體公式。Section 3.2 定義的兩個候選是：
- Eq. 1（SCL-NL）：$\bar L=-\log(1-g_{\bar y}(X))$（**不加權**）
- Eq. 2（SCL-EXP）：$\bar L=\exp(g_{\bar y}(X))$

**論文 Section 5.1「Implementation」明講**："For ComCo, we choose **SCL-NL** as complementary
loss $\mathcal L_{cls}$"。而且論文 Section 2（Related Works）把 SCL-NL 定調為
「**放棄** unbiased risk estimator」的方法（"abandons the unbiased risk estimator of Ishida et
al. (2017)'s PC loss"）——SCL-NL 存在的意義就是不要那個修正/縮放項。

程式碼的 `ComCoCLSLoss` 卻是 `-log(Σ_{c∉Ȳ}p_c) * (C-1)/(C-m)`——這是 **MCL-NL**（Feng et al.
2020，也就是本專案 `MCL-LOG` 對應的那篇論文）的公式形式，不是論文自己選的 SCL-NL。ComCo 論文
只把 MCL-NL 當成引用的 baseline 之一，不是自己方法的分類損失。

**這個 bug 何時會顯現**：當 $m=1$（單一互補標籤）時，$\frac{C-1}{C-1}=1$，公式剛好退化成
$-\log(1-p_{\bar y})$，跟論文的 SCL-NL 一致——這也是為什麼程式碼的 docstring 寫
「"Reduces to SCL-NL for single complementary label"」，這句話本身是對的。但只要樣本有
**超過一個**互補標籤（$m>1$，本專案的 CL 生成邏輯很容易產生這種樣本），程式碼算出來的權重
就跟論文的實際選擇不一樣了。

（論文本身也沒有給 ComCo 分類損失的 multi-CL 明確公式——Section 3.3 只說 multi-CL 延伸
「幾乎一樣」，沒有寫出公式。因此「修正」的做法是：既然論文明確選了 SCL-NL 當分類損失，multi-CL
延伸就採用本專案已經驗證過的 `SCL_NL`（`src/scl_loss.py`）的 multi-CL 平均 wrapper 形式——
即對每個互補標籤各自算 $-\log(1-p_c)$ 再平均，不加任何 `(C-1)/(C-m)` scaling。）

### 2. 對比損失：完全一致，包含論文自創的命名

| 論文 | 程式碼 | 結果 |
|---|---|---|
| 正樣本集合 $P(x_i)$：一定含 $k_i$，`warmup_pos`（100 epoch）後加入依 $\text{Sim}(x_i,x_j)=[\tilde y_i=\tilde y_j]\cdot 0.5(1+\cos\_sim)$ 排序的 top-K 鄰居（Eq. 5-7，論文自己稱為 **"Strategy B"**，並在 Appendix A.1 證明 $P_B\subseteq P_A$（$A$ 是 PiCO 的策略）） | `integrated_sim`、`topk`，docstring 標註 "Strategy B from paper" | ✅ 完全一致，連命名都對得上 |
| 負樣本集合：`warmup_neg`（1 epoch）後選 $\text{Dist\_min}$ 最大的單一互補標籤子集（Eq. 8-9，論文稱 **"strategy 1"**，論文自己承認也有 "strategy 2"／Dist_mean，但 Section 5.1 確認 ComCo 預設用 strategy 1） | `dist_min`/`argmax` 選單一子集 | ✅ 完全一致，程式碼採用的正是論文自己的預設選擇 |
| $K=1$，`warmup_pos=100`，`warmup_neg=1`（Section 5.1） | `config.yaml`: `top_k:1`, `warmup_pos:100`, `warmup_neg:1` | ✅ 完全一致 |
| $\tau=0.17$，$\lambda=0.3$，queue=8192（CIFAR 設定） | `config.yaml`: `temperature:0.17`, `loss_weight:0.3`, `moco_queue:8192` | ✅ 完全一致 |
| Pseudo label = 未加 mask 的分類器 argmax，不做平滑 | `cls_out.argmax(dim=1)` | ✅ 完全一致 |

### 3. Optimizer：論文用 Adam，新版 pipeline 已經是對的

論文 Section 5.1："For all models, we use **Adam** optimizer... weight-decay 固定 1e-4"。
**舊版 pipeline 的 `setup_comco`（`src/model_setup.py`）用的是 SGD**，跟論文不符；但
**Step 5 實驗實際會用的新版 pipeline，`src/pipeline/algorithms/hparams.py` 裡 `ComCo` 已經是
`_ADAM`**——這部分不需要修正，只在此記錄舊版 pipeline 的落差供未來清理舊版時參考。

---

## 原論文使用的 Benchmark（Step 3，已完成）

- **資料集**：MNIST、Fashion-MNIST、KMNIST、CIFAR-10（unbiased/biased 單一互補標籤設定）、
  KMNIST+CIFAR-10（固定數量 multi-CL）、CIFAR-10+CIFAR-100（變動數量 multi-CL），加上 CUB-200、
  SUN-397（fine-grained 延伸實驗）——**總共 6 個資料集、5 種評測協定**，比本專案目前只在
  CIFAR-10/20/CLCIFAR-10/20 上跑 ComCo 的範圍大很多。
- **互補標籤生成**：論文用三種合成方式——unbiased（均勻抽樣，Ishida et al. 2017）、biased
  （固定機率的偏態抽樣，Gao & Zhang 2021）、MCLL（Feng et al. 2020 的 $\binom{k}{s}/(2^k-2)$
  組合抽樣）。本專案是用 PL 生成的補集（constant-k / variable-q + 可選 noise η）——**跟論文三種
  方式都不一樣**，CLCIFAR-10/20（真人標註）則是論文完全沒用過的資料。
- **Backbone**：CIFAR-10/100 用 ResNet-18（跟本專案 `SupConResNet` 一致）；MNIST 系列用 5 層 FC。
- **評估**：top-1 accuracy，5 次重跑平均±標準差；CIFAR-10/100 訓練 1000 epoch，其餘 800 epoch。
- **其他超參數**：MLP head 兩層，512→128 維（對應本專案 `low_dim:128`）；query 增強用
  SimAugment，key 增強在 CIFAR-100/CUB/SUN 用 RandAugment（MNIST 系列/CIFAR-10 的 key 也用
  SimAugment）——這部分本專案的實際 augmentation pipeline 需要另外對照 `src/data_setup.py`，
  超出這次兩個檔案的檢視範圍。

---

## Fixed 版本（Step 4，已完成）

`src/comco/fixed_utils_loss.py`：新增 `FixedComCoCLSLoss`，實作**不加權的 SCL-NL 平均**
（跟本專案 `src/scl_loss.py` 的 `SCL_NL` multi-CL wrapper同一種形式），移除 `(C-1)/(C-m)` scaling。
對比損失（`ComCoContrastiveLoss`）跟模型架構（`ComCoModel`）**不變**——已確認忠實對應論文。

`run_comco_fixed`（`src/pipeline/algorithms/runners.py`）：沿用跟 `run_comco` 相同的模型/對比損失
建構，只把 `ComCoCLSLoss()` 換成 `FixedComCoCLSLoss()`。已註冊
`AlgorithmSpec('ComCo-Fixed', 'CLL', r.run_comco_fixed)`。

---

## 實驗 Config（Step 5）

> **2026-08-14 更新**：`--dataset cub200`（論文 Section 5.6 用過的真實照片資料，見
> [00_paper_alignment_guide.md](00_paper_alignment_guide.md) 的「資料集支援」一節）已在本機
> 端到端驗證過。論文另外用過的 SUN-397：`--dataset sun397` **程式碼已經寫好、註冊進 pipeline**
> （跟 `cub200` 同一套 lazy-path 載入機制），但因為資料集本身太大（HuggingFace 上找到的最小
> mirror 也要 ~17GB，官方版本更是 ~37GB），這次**沒有實際觸發下載驗證過**，第一次真的拿來跑
> 之前建議先用少量 epoch/小 batch 跑一次 smoke test 確認。MNIST/Fashion-MNIST/KMNIST 也可以跑
> （論文用過的另一批資料集），但都不支援 PiCO/ComCo 系列（灰階），只能用來測
> `ComCoCLSLoss`/`ComCoCLSLoss-Fixed` 以外的部分。
>
> ```bash
> python scripts/run_pipeline.py run --run_name comco_original_benchmark \
>     --algorithms ComCo ComCo-Fixed --dataset cub200 --epochs 100
> ```

```bash
python scripts/run_pipeline.py run --run_name comco_cls_bugfix_check \
    --algorithms ComCo ComCo-Fixed --c_values 5 20 --epochs 200
```

- **建議跟 MCL-LOG 那組實驗一樣，`ComCo` 跟 `ComCo-Fixed` 並排跑**。新版 pipeline 只支援
  constant-k 生成（同一個 (C,k) cell 裡每個樣本的互補標籤數 $m=C-k$ 固定，沒有 CLI 可選的
  "variable" 模式），這個 bug 只在 $m>1$ 時才會顯現差異，所以要選 $k$ 較小（$m=C-k$ 較大）的
  cell 才看得出差別——`--c_values 5 20` 搭配預設 k-schedule 裡 $k=1$（$m=C-1$，最大）的那幾個點
  應該最能突顯這個 bug，$k=C-1$（$m=1$）的點則理論上兩個版本會幾乎沒有差異（可以當作 sanity
  check：如果 $k=C-1$ 時兩者結果差很多，代表哪裡還有沒抓到的問題）
- 若要更貼近論文完整評測矩陣（unbiased/biased/MCLL 三種生成方式），需要在 `src/data_setup.py`
  額外實作對應的生成邏輯，超出本次範圍，留給使用者決定是否要做
- 可以跟 `PiCO`（PLL 對照組，同樣是 dual-encoder + queue 架構）並排比較訓練動態
