# SCL-NL：Unbiased Risk Estimators Can Mislead

> **狀態**：骨架文件，Step 1（程式碼對應）已完成，Step 2（公式逐項比對）與 Step 3（benchmark 查證）
> 標記為 TODO，交給下一個 session 對照論文原文填寫。

**論文來源：** Chou, Y.-T., Niu, G., Lin, H.-T., & Sugiyama, M. (2020).
*Unbiased Risk Estimators Can Mislead: A Case Study of Learning with Complementary Labels.*
ICML 2020, PMLR 119:1929–1938.
[PMLR](https://proceedings.mlr.press/v119/chou20a.html) ·
[arXiv](https://arxiv.org/abs/2007.02235)

**Algorithm ID（pipeline 內部字串）：** `SCL-NL`

---

## 對應程式碼：`src/scl_loss.py` (`SCL_NL`)

程式碼本身已經有清楚的論文引用與公式 docstring（6-38 行），這是這 8 個實作裡文件化程度最好的一個：

- **Single-CL 形式**（Eq. 11，docstring 14-15 行）：
  $$\phi_{NL}(\bar y, g(x)) = -\log(1 - p_{\bar y})$$
  程式碼用數值穩定的 `log1p` 實作（60 行）：`-log1p(-p.clamp(max=1-eps))`，等價於
  `-log(1 - p_{\bar y})` 但避免 `p` 接近 1 時的數值問題。
- **Multi-CL 平均 wrapper**（docstring 22-27 行，程式碼 62-64 行）：
  $$L(x, \bar Y) = \frac{1}{|\bar Y|}\sum_{\bar y \in \bar Y} \phi_{NL}(\bar y, g(x))$$
  用平均（不是加總）讓不同 `|\bar Y|` 的樣本對 batch loss 的貢獻相等（docstring 29-31 行明講
  這個設計理由）。
- Docstring 特別強調（32-33 行）：**這不是 MCL 的 unbiased risk estimator（MCL-LOG）**，
  沒有對「非互補標籤」求和的項 —— 這是 SCL-NL 與 MCL-LOG 兩篇論文的核心區別，Step 2 比對時
  這條要重點驗證程式碼是否真的沒有混入 MCL 的機制。
- Binary mask 建構方式（46-56 行）跟 `clpl_loss.py`/`op_loss.py` 一樣用 3D scatter 向量化，
  沒有 for-loop（跟 `mcl_losses.py` 的寫法不同）。

### Entry point

- 新版 pipeline：`run_scl_nl`（`src/pipeline/algorithms/runners.py:138-139`），
  `AlgorithmSpec('SCL-NL', 'CLL', r.run_scl_nl)`（`src/pipeline/algorithms/__init__.py:38`）
- 舊版 pipeline：`setup_scl`（`src/model_setup.py:42-49`，標註 "SCL-NL (Chou et al. 2020)"）/
  `run_scl_training`（`src/training_pipelines.py:35`）

---

## 演算法保真度比對 — TODO（Step 2）

程式碼本身的 docstring 已經很接近論文語言，但還沒有逐條對照論文原文確認：

- [ ] 論文 Eq. 11 的 `phi_NL` 精確形式，跟程式碼的 `-log(1-p_ybar)` 是否完全一致（含是否有額外的
      係數或截斷方式）。
- [ ] 論文標題強調「Unbiased Risk Estimators Can Mislead」，SCL-NL 本身是作為 URE（如 MCL）的
      對照組被提出的 **surrogate loss**（不是 URE）—— 需要確認論文對 "consistency"/"convergence"
      的理論保證跟 URE 類方法有什麼不同，這部分程式碼註解沒有寫，需要讀論文補充。
- [ ] Multi-CL 平均 wrapper 是這個 repo 自己加的，還是論文本身就有討論 multi-complementary-label
      的擴展形式？如果論文只討論 single-CL，這個 wrapper 的正確性需要額外驗證。

---

## 原論文使用的 Benchmark — TODO（Step 3）

需要查證論文的實驗資料集與互補標籤生成設定（single CL per sample，均勻抽樣），對照本專案的
CLL 生成邏輯是否一致。

---

## Fixed 版本 — 尚未開始（Step 4）

放在 `src/fixed_scl_loss.py`，class `FixedSCLNL`。

---

## 實驗 Config — 尚未開始（Step 5）

見 [主引導文件](00_paper_alignment_guide.md) 的前置條件（pipeline 尚未驗證可跑）。
