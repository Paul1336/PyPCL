# MCL-LOG（及 MAE / EXP 變體）：Learning with Multiple Complementary Labels

> **狀態**：骨架文件，Step 1（程式碼對應）已完成，Step 2（公式逐項比對）與 Step 3（benchmark 查證）
> 標記為 TODO，交給下一個 session 對照論文原文填寫。

**論文來源：** Feng, L., Kaneko, T., Han, B., Niu, G., An, B., & Sugiyama, M. (2020).
*Learning with Multiple Complementary Labels.* ICML 2020, PMLR 119:3072–3081.
[PMLR](https://proceedings.mlr.press/v119/feng20a.html)

**Algorithm ID（pipeline 內部字串）：** `MCL-LOG`
（`MCL-MAE`、`MCL-EXP` 是同一篇論文提到的另外兩種 loss 變體，程式碼存在但**目前只有 MCL-LOG 註冊進
新版 pipeline**，見下方「已知落差」。）

---

## 對應程式碼：`src/mcl_losses.py`

三個 class 結構幾乎相同，差別只在 loss 的函數形式：

### 共用邏輯（三個 class 都一樣）

1. 建立互補標籤的 binary mask `mask_complementary`（`for i in range(batch_size)` 逐樣本
   `scatter_`，例如 `MCL_LOG` 的 19-22 行）—— **這裡是 Python for-loop，不是向量化實作**，
   跟其他 loss 檔案（`clpl_loss.py`、`scl_loss.py`、`op_loss.py`）用 3D scatter 向量化的寫法不同，
   Step 2 可以順便評估這是否值得優化（不影響數學正確性，但影響效能）。
2. `sum_probs_not_in_complementary_set = (probs_all * mask_non_complementary).sum(dim=1)`
   —— 非互補類別的 softmax 機率總和。
3. Unbiased risk estimator 縮放：`scaling_factor = (C-1) / (C - num_complementary)`
   （`MCL_LOG` 31-32 行、`MCL_MAE` 62-63 行、`MCL_EXP` 93-94 行）。

### 三種 loss 函數形式

| Class | 檔案行號 | Loss 公式 |
|---|---|---|
| `MCL_LOG` | `mcl_losses.py:5-34` | `loss = -log(sum_probs_not_in_complementary + 1e-7)` |
| `MCL_MAE` | `mcl_losses.py:36-65` | `loss = 1 - sum_probs_not_in_complementary` |
| `MCL_EXP` | `mcl_losses.py:67-96` | `loss = exp(-sum_probs_not_in_complementary)` |

三者都在算完 base loss 後乘上同一個 `scaling_factor`。

### Entry point

- 新版 pipeline：只有 `MCL_LOG` —— `run_mcl_log`（`src/pipeline/algorithms/runners.py:134-135`），
  `AlgorithmSpec('MCL-LOG', 'CLL', r.run_mcl_log)`（`src/pipeline/algorithms/__init__.py:37`）
- 舊版 pipeline：`setup_mcl(..., loss_type='log'|'mae'|'exp')`（`src/model_setup.py:28-40`）/
  `run_mcl_training`（`src/training_pipelines.py:25`，依 `loss_type` 參數化）支援全部三種

**已知落差**：`MCL_MAE`、`MCL_EXP` 目前沒有被註冊進 `pipeline/algorithms/__init__.py`，只能透過
舊版 pipeline 執行。使用者這次要求的是「MCL-LOG」這篇論文對應的文件（論文本身涵蓋三種 loss 變體），
Step 2/3 的比對建議三個變體一起做，但**是否要把 MAE/EXP 也註冊進新版 pipeline 屬於 Step 4/5 的實作
決定，不是 Step 1-3 文件範圍**，需要另外確認。

---

## 演算法保真度比對 — TODO（Step 2）

- [ ] 論文的 unbiased risk estimator 完整公式是什麼？確認 `(C-1)/(C-m)` 這個縮放係數是否跟論文
      一致（`m` = 互補標籤數）。
- [ ] 論文對 LOG / MAE / EXP 三種 loss 的定義，跟上表逐一核對是否一致（尤其注意 `MCL_LOG` 加了
      `epsilon=1e-7` 的數值穩定項，論文是否也有討論這個細節）。
- [ ] 論文的 "multiple complementary labels" 設定（每個樣本可以有 >1 個互補標籤）跟這裡
      `complementary_labels: [B, max_m]` 的多標籤 padding 表示法是否對應論文的資料假設。

---

## 原論文使用的 Benchmark — TODO（Step 3）

需要查證論文用的資料集（MNIST、Kuzushiji-MNIST、CIFAR-10 等 CLL 文獻常見組合）與互補標籤生成方式
（每個樣本隨機抽 m 個互補標籤，m 是固定還是變動），對照本專案的互補標籤生成邏輯（PL 的補集，
`src/data_setup.py`）是否一致。

---

## Fixed 版本 — 尚未開始（Step 4）

放在 `src/fixed_mcl_losses.py`，class 前綴 `Fixed`（`FixedMCLLog` 等）。

---

## 實驗 Config — 尚未開始（Step 5）

見 [主引導文件](00_paper_alignment_guide.md) 的前置條件（pipeline 尚未驗證可跑）。
