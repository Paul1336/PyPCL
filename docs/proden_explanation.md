# PRODEN：Progressive Identification of True Labels for Partial-Label Learning

> **狀態**：骨架文件，Step 1（程式碼對應）已完成，Step 2（公式逐項比對）與 Step 3（benchmark 查證）
> 標記為 TODO，交給下一個 session 對照論文原文填寫。

**論文來源：** Lv, J., Xu, M., Feng, L., Niu, G., Geng, X., & Sugiyama, M. (2020).
*Progressive Identification of True Labels for Partial-Label Learning.* ICML 2020, PMLR 119:6500–6510.
[PMLR](https://proceedings.mlr.press/v119/lv20a.html) ·
[arXiv](https://arxiv.org/abs/2002.08053)

**Algorithm ID（pipeline 內部字串）：** `PRODEN`

---

## 對應程式碼

`src/proden_loss.py` 裡實際上有 **兩個** class，代表這個 repo 裡 PRODEN 的兩種實作版本：

### 1. `ProdenLoss`（`src/proden_loss.py:6-49`）—— 目前被兩套 pipeline 使用的版本

- **跨 epoch 信心累積（persistent buffer）**：`__init__`（21-29 行）為每個訓練樣本建立一個
  `conf[N, C]` buffer，初始值為候選集內均等分佈（`1/k`，k=候選數）。
- `forward(outputs, indices)`（31-49 行）：
  - Loss：用**上一步存下來的** `conf[indices]` 當作 soft label 權重，算加權 cross-entropy：
    `loss = -(conf[indices] * log_softmax(outputs)).sum(1).mean()`
  - Update（`torch.no_grad()`，43-47 行）：用**這一步**模型的 softmax，限制在候選集 mask 內重新
    正規化，寫回 `self.conf[indices]`，供下一次呼叫使用 —— 這是 self-training / progressive
    identification 的核心機制。
- Docstring 自稱「original paper algorithm」（第 8 行）。

### 2. `proden`（`src/proden_loss.py:52-75`）—— 舊版/簡化版，無跨 epoch 記憶

- 每次 `forward` 都**重新計算**權重（沒有 persistent buffer）：對候選集內的 softmax 機率做
  正規化當作權重，再乘上 `-log_softmax` 取加權和。
- 沒有 `ProdenLoss` 的 EMA/self-training 機制，等同於每個 batch 獨立做一次 M-step。

**目前哪個版本被使用？**（TODO：下一個 session 需要在改 README/程式碼註解時，逐一確認新舊 pipeline
分別呼叫哪一個 class，並在下面補上結論）

- 新版 pipeline：`run_proden`（`src/pipeline/algorithms/runners.py:194-222`）
- 舊版 pipeline：`setup_proden`（`src/model_setup.py:21-26`）/
  `run_proden_training`（`src/training_pipelines.py:15`）

Entry point registration：`AlgorithmSpec('PRODEN', 'PLL', r.run_proden)`
（`src/pipeline/algorithms/__init__.py:31`）。

---

## 演算法保真度比對 — TODO（Step 2）

論文原始演算法是一個 EM 風格的迭代方法：E-step 用當前模型對候選標籤重新估計信心分數，M-step 用信心
分數加權訓練模型。需要對照論文確認：

- [ ] 論文的信心更新公式，是否等於 `ProdenLoss.forward` 裡 `new_conf = candidate_mask *
      softmax(outputs)` 再正規化的寫法？（尤其留意論文是否用 `softmax` 還是別的正規化方式）
- [ ] 論文的信心初始化是否也是均等分佈 `1/k`？
- [ ] `proden`（簡化版）跟論文演算法的差異是否只在於「有沒有跨 batch/epoch 記憶」？這個差異
      對收斂行為有多大影響，論文有沒有討論類似的 ablation？
- [ ] 論文是否有額外的 warm-up、退火、或其他訓練技巧（例如 loss 之外的正則化）沒有出現在
      `ProdenLoss` 裡？

---

## 原論文使用的 Benchmark — TODO（Step 3）

需要查證：論文 Table/實驗章節用了哪些資料集（MNIST、Kuzushiji-MNIST、Fashion-MNIST、CIFAR-10 等是
PLL 文獻常見組合，但需要逐一確認）、partial label 生成方式（uniform q、flip probability 等）、
以及與本專案目前 `--type constant|variable` 的候選集生成邏輯（`src/data_setup.py`）是否一致。

---

## Fixed 版本 — 尚未開始（Step 4）

等 Step 2 完成、確認 `ProdenLoss` 或 `proden` 與論文演算法有落差後才開始，會放在
`src/fixed_proden_loss.py`（class `FixedProdenLoss`）。

---

## 實驗 Config — 尚未開始（Step 5）

見 [主引導文件](00_paper_alignment_guide.md) 的前置條件（pipeline 尚未驗證可跑）。
