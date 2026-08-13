# OP-W：Consistent Complementary-Label Learning via Order-Preserving Losses

> **狀態**：骨架文件，Step 1（程式碼對應）已完成，Step 2（公式逐項比對）與 Step 3（benchmark 查證）
> 標記為 TODO，交給下一個 session 對照論文原文填寫。

**論文來源：** Liu, S., Cao, Y., Zhang, Q., Feng, L., & An, B. (2023).
*Consistent Complementary-Label Learning via Order-Preserving Losses.* AISTATS 2023, PMLR 206:8734–8748.
[PMLR PDF](https://proceedings.mlr.press/v206/liu23g/liu23g.pdf)

**Algorithm ID（pipeline 內部字串）：** `OP-W`
（`OP`（未加權的 Order-Preserving 基礎版本）也在同一個檔案裡，同一篇論文提出，本文件一併記錄。）

---

## 對應程式碼：`src/op_loss.py`

程式碼 docstring 已標明論文出處與 Definition 編號，是這 8 個實作裡文件化最完整的之一。

### 共用 helper：`_build_comp_mask`（6-16 行）

跟 `scl_loss.py` 相同風格，用 3D scatter 向量化建立互補標籤 binary mask，回傳 `(comp_mask, m)`。

### `OPLoss`（Order-Preserving 基礎版，19-50 行，Definition 3.1）

- 核心洞見（docstring 26-29 行）：因為 $P(\bar y=k\mid x) \propto 1 - P(y=k\mid x)$，互補標籤
  應該在 $g(x)$ 裡分數**最低**。把 logits 取負號（$-g(x)$）後排序反轉，這樣互補標籤在 $-g(x)$
  裡就變成分數最高 → 標準 CE 可以直接把它當作「target」訓練。
- Single-CL 公式：$L_{OP} = \ell(-g(x), \bar y) = -\log\text{softmax}(-g(x))_{\bar y}$
  （程式碼 47 行：`per_label_loss = -F.log_softmax(-outputs, dim=1)`）
- Multi-CL wrapper（36 行 docstring，48-49 行程式碼）：對候選集內每個互補標籤平均。
- Docstring 特別指出（38-39 行）：這跟 SCL-NL **不是同一個東西** —— SCL-NL 用
  $-\log(1-p_{\bar y})$，OP 用 $-\log\text{softmax}(-g)_{\bar y}$，函數形式完全不同。
- Docstring 提到論文 Theorem 3.1：risk estimator 天生非負（避免 URE 常見的「負風險」過擬合問題）
  且 classifier-consistent —— **這個理論性質本身需要 Step 2 對照論文原文驗證，程式碼註解只是
  轉述，不代表已經被驗證過**。

### `OPWLoss`（加權版，53-97 行，Definition 4.1）

- 在 `OPLoss` 的基礎上加權重：$L_{OP\text{-}W} = w(g(x),\bar y)\cdot \ell(-g(x),\bar y)$
- 權重公式（docstring 62-64 行，程式碼 85-92 行，來自論文 Appendix D）：
  $$w(g(x), y) = \text{softmax}(u(x)+1)_y \cdot \text{softmax}(g(x))_y + \epsilon,\quad
    u_j(x) = \frac{1}{\text{softmax}(-g(x))_j}$$
- 直覺（docstring 66-69 行）：當模型已經把 $\bar y$ 排得很低（$\text{softmax}(-g)_{\bar y}$ 小
  → $u_{\bar y}$ 大）時權重變大，避免演算法忽略排名墊底的難分互補標籤。

### Entry point

- **只有新版 pipeline**：`run_op`（`src/pipeline/algorithms/runners.py:142-143`）、
  `run_op_w`（`146-147` 行），`AlgorithmSpec('OP', 'CLL', r.run_op)` /
  `AlgorithmSpec('OP-W', 'CLL', r.run_op_w)`（`src/pipeline/algorithms/__init__.py:39-40`）。
  **舊版 pipeline（`model_setup.py`/`training_pipelines.py`）完全沒有 OP/OP-W 的 `setup_*`/
  `run_*_training`**——這是 8 篇論文裡唯一一個只存在於新版 pipeline 的演算法，意味著它從未在
  舊版 pipeline 跑過，也沒有 `scripts/run_experiment.py` 可以直接驗證。

---

## 演算法保真度比對 — TODO（Step 2）

- [ ] `OPLoss` 的 $-\log\text{softmax}(-g(x))_{\bar y}$ 是否精確對應論文 Definition 3.1 的原始
      定義（含是否有額外的正規化/縮放項）。
- [ ] `OPWLoss` 的權重公式（Appendix D）逐項核對：$u_j = 1/\text{softmax}(-g)_j$ 這個倒數形式，
      在 $\text{softmax}(-g)_j \to 0$ 時 $u_j \to \infty$，程式碼用 `p_neg.clamp(min=1e-7)`
      （88 行）避免除以 0——這個 clamp 的閾值是否跟論文的數值穩定處理方式一致，還是本專案自行
      加上去的，需要在論文裡確認有沒有對應描述。
- [ ] Theorem 3.1（risk 非負、consistency）是否真的在這個實作下成立，還是只是論文對「理想化」
      loss 的保證，程式碼裡的數值穩定近似（`log_softmax`、`clamp`）是否可能破壞這個保證。

---

## 原論文使用的 Benchmark — TODO（Step 3）

需要查證論文的實驗資料集設定（AISTATS 2023，CLL 常見組合可能包含 MNIST/CIFAR-10/CIFAR-100），
並確認論文是否也討論 multiple-complementary-label 設定（本專案的 wrapper 是 multi-CL 平均）。

---

## Fixed 版本 — 尚未開始（Step 4）

放在 `src/fixed_op_loss.py`，class `FixedOPLoss`/`FixedOPWLoss`。

---

## 實驗 Config — 尚未開始（Step 5）

見 [主引導文件](00_paper_alignment_guide.md) 的前置條件（pipeline 尚未驗證可跑）。由於 OP/OP-W
只存在於新版 pipeline，這也是驗證新版 pipeline 是否真的可用的一個好測試對象。
