# PiCO：Contrastive Label Disambiguation for Partial Label Learning

> **2026-08-14 更新**：Step 2、Step 3 已完成，對照論文 PDF（`C:\Users\User\Desktop\papers\PiCO.pdf`）
> 逐條驗證。結論：**核心數學機制（prototype 更新、confidence 更新、分類損失、對比損失、總損失組合）
> 全部逐項精確對應論文**，包括先前文件標記的「兩個訊號來源」疑慮（見下方 Step 2 第一項，已確認正確）。
> **唯一實質落差在 warm-up 機制**——論文說 warm-up 期間直接把 $L_{cont}$ 從總損失拿掉，程式碼卻是
> 把 $L_{cont}$ 換成 unsupervised MoCo 版本繼續加進總損失，而且預設的 `prot_start=80` 跟論文任何
> 一個報告值都對不上。已產出 `src/fixed_pico_engine.py` + `run_pico_fixed`（algorithm ID
> `PiCO-Fixed`）修正這個落差。
>
> **2026-08-14 二次覆核**：改用 `pypdf` 直接擷取 PDF 純文字重新逐字比對，這次連 CIFAR-100-H 的
> q-based 生成方式都一併查證（見下方「原論文使用的 Benchmark」）。Eq. 6（confidence 更新用
> prototype 相似度 $q^\top\mu_j$）、Eq. 7（prototype 更新用分類器自己的 softmax
> $\arg\max_{j\in Y}f_j$）、Eq. 5（$L=L_{cls}+\lambda L_{cont}$，$\lambda=0.5$）、Eq. 1（分類
> 損失 $\sum_j -s_{i,j}\log f_j(x_i)$）、以及 warm-up 原文（"we disable contrastive learning in
> the first 100 epoch for CIFAR-100 with q=0.1 and 1 epoch for the remaining experiments"）都
> 逐字確認跟下方比對表、跟已知落差描述完全一致。沒有發現新的問題。
>
> **2026-08-16 三次覆核（發現第二個實質落差）**：透過 `pypdf` 查證論文 "Pseudo Target Updating"
> 段落原文——"we first initialize the pseudo targets with a uniform distribution,
> $s_j=\frac1{|Y|}\mathbb{I}(j\in Y)$"——確認 pseudo target 的**初始化**應該跟 PRODEN 同一個精神：
> 只在候選集 $Y$ 內均勻分佈，候選集外一律是 0。但 `run_pico`/`run_pico_fixed` 兩個都用
> `torch.ones(N, C) / C`，是**全部 C 類均勻**，候選集外的類別也拿到跟候選集內一樣的初始權重。
> 因為 `PartialLoss.forward` 沒有用候選集 mask 這個 confidence（`Eq. 6` 的更新又要等 warm-up
> 結束才開始跑），這代表整個 warm-up 期間（原版預設 80 epoch）分類損失都在把機率往「所有類別」
> 均攤，而不是論文設計的「只在候選集內給均等權重」——warm-up 期間的分類訊號被嚴重稀釋。
> **已修正 `PiCO-Fixed`**（見下方「Fixed 版本」第 3 項），原版 `PiCO` 維持不動（使用者要求
> 「原版不動」）。這是獨立於 warm-up $L_{cont}$ 那個已知落差的**第二個**問題。

**論文來源：** Wang, H., Xiao, R., Li, Y., Feng, L., Niu, G., Chen, G., & Zhao, J. (2022).
*PiCO: Contrastive Label Disambiguation for Partial Label Learning.* ICLR 2022.
[OpenReview](https://openreview.net/forum?id=EhYjZy6e1gJ)

**Algorithm ID（pipeline 內部字串）：** `PiCO`（修正版：`PiCO-Fixed`）
（repo 裡還有三個 PiCO 的變體 `PiCO-MCL`、`PiCO-SC`、`PiCO-CLS`，**不在使用者原始 8 篇論文範圍內，
本文件不處理**，只記錄這裡以免未來搞混。其中 `PiCO-SC` 的 ablation 設計——用分類器自己的 softmax
取代 prototype 相似度做 confidence 更新——剛好對應到下面 Step 2 第一項討論的「兩個訊號」問題，
是本專案自創的 ablation，論文本身沒有這個變體。）

---

## 對應程式碼

### 模型架構：`src/pico/model.py` (`PiCOModel`)

- **雙 encoder（MoCo 風格 momentum contrast）**：`encoder_q`（會被梯度更新）與 `encoder_k`
  （用 momentum 更新，`_momentum_update_key_encoder`，24-27 行）。
- **Prototype 記憶體**：`prototypes[C, low_dim]` buffer，在 `forward`（52-54 行）用 EMA 依 pseudo
  label 更新，之後 L2 normalize。
- **Pseudo label 產生**（45-46 行）：`predicted_scores = softmax(output) * partial_Y`，取 argmax
  當作**更新 prototype 用**的 pseudo label（這是分類頭自己的 softmax，候選集 mask 後取 argmax）。
- **Feature queue**：`queue`/`queue_pseudo`，`_dequeue_and_enqueue`，`assert moco_queue %
  batch_size == 0`（已知隱患，見主引導文件）。
- `forward` 回傳 `(output, features, pseudo_labels, score_prot)`，`score_prot = softmax(q ·
  prototypes^T)`（prototype 相似度分數，回傳給外部的 confidence 更新機制用）。

### Loss：`src/pico/utils_loss.py`

- **`PartialLoss`**（分類損失）：`forward` 是加權 CE；`confidence_update(temp_un_conf,
  batch_index, batchY)` 用外部傳入的 `temp_un_conf * batchY` 做 argmax 更新信心（EMA）。
- **`SupConLoss`**（對比損失）：`mask` 給定時是候選集相似度驅動的 SupCon InfoNCE；`mask=None`
  時退化成標準 MoCo InfoNCE（`q·k` 正樣本、`q·queue` 負樣本）。

### 訓練迴圈：`src/engine.py::train_pico_epoch`

```python
start_upd_prot = epoch >= pico_args['prot_start']
...
if start_upd_prot:
    loss_fn.confidence_update(temp_un_conf=score_prot.detach(), batch_index=index, batchY=partial_Y)
mask = torch.eq(...).float() if start_upd_prot else None
loss_cont = loss_cont_fn(features=features, mask=mask, batch_size=batch_size)
loss = loss_cls + pico_args['loss_weight'] * loss_cont   # loss_cont 永遠都加進去，不管 warm-up
```

### Entry point

- 新版 pipeline：`run_pico`（`src/pipeline/algorithms/runners.py`），
  `AlgorithmSpec('PiCO', 'PLL', r.run_pico)`
- 舊版 pipeline：`setup_pico`（`src/model_setup.py:78-94`）/
  `run_pico_training`（`src/training_pipelines.py:69`）

---

## 演算法保真度比對（Step 2，已完成）

### 1. Pseudo label 來源（先前文件標記的疑慮，已解決）

論文明確用**兩個不同訊號**做兩件不同的事：

- **Prototype 更新（Eq. 7）**：$\mu_c = \text{Normalize}(\gamma\mu_c + (1-\gamma)q)$，
  **選哪個 $c$ 更新用分類器自己的 softmax**：$c=\arg\max_{j\in Y}f^j(\text{Aug}_q(x))$
- **Confidence/pseudo-target 更新（Eq. 6）**：$s=\phi s+(1-\phi)z$，$z$ 用
  **prototype 相似度**決定：$z_c=1$ 若 $c=\arg\max_{j\in Y}q^\top\mu_j$

程式碼裡：`PiCOModel.forward` 的 `pseudo_labels_b`（分類器 softmax argmax）驅動 prototype 更新 ——
對應 Eq. 7；`PartialLoss.confidence_update` 收到的 `temp_un_conf` 在訓練迴圈裡被指定為
`score_prot.detach()`（prototype 相似度），乘上候選集 mask `batchY` 後取 argmax —— 對應 Eq. 6。
**兩者用的訊號完全跟論文對上，程式碼正確地把兩個不同用途的訊號分開處理，沒有混用。**

（`conf_ema_m`／$\phi$：論文從 0.95 線性遞減到 0.8，程式碼 `conf_ema_range: [0.95, 0.8]`，
`set_conf_ema_m` 的線性內插公式也對得上——這項完全一致。）

### 2. Warm-up 機制（**發現實質落差**）

論文原文（Appendix B.1）："we disable contrastive learning in the first 100 epoch for CIFAR-100
with q=0.1 and **1 epoch** for the remaining experiments"——**warm-up 期間是把整個 $L_{cont}$
從總損失拿掉**，Algorithm 1 的偽代碼本身完全沒有 gate（prototype 更新、pseudo-target 更新、
$L_{cls}$、$L_{cont}$ 都是每個 iteration 無條件執行），warm-up 只是文字說明裡的一個實務補充。

程式碼的 warm-up 做的是**不同的事**：
- $L_{cont}$ **並沒有被拿掉**，而是把 `SupConLoss` 從「候選集相似度驅動」切換成「純 unsupervised
  MoCo InfoNCE」（`mask=None` 分支），然後繼續加進總損失
- confidence/pseudo-target 更新（Eq. 6）在 warm-up 期間被關掉（`start_upd_prot` 判斷），這點雖然
  Algorithm 1 沒有明講要 gate，但跟論文的敘述動機一致（"fitting uniform pseudo targets results
  in a good initialization... contrastive embeddings are less distinguishable at the beginning"），
  可以視為對論文精神的合理延伸，不算錯誤
- **`prot_start` 預設值 80**（`config.yaml`）跟論文報告的任何一個值都對不上（論文預設 1 epoch，
  只有 CIFAR-100 q=0.1 這個最難的設定用 100 epoch）——80 是本專案自創的折衷值，沒有論文依據

### 3. Pseudo target 初始化（**發現第二個實質落差**，2026-08-16）

論文（"Pseudo Target Updating" 段落）："we first initialize the pseudo targets with a uniform
distribution, $s_j=\frac1{|Y|}\mathbb{I}(j\in Y)$"——只在候選集 $Y$ 內均勻分佈，候選集外是 0。

`run_pico`/`run_pico_fixed`（修正前）都用 `torch.ones(len(pl_ds), C) / C`：**全部 C 類均勻**，
不管是不是候選集內的類別。`PartialLoss.forward`（`src/pico/utils_loss.py:29-33`）沒有用候選集
mask 這個 confidence（`final_outputs = logsm_outputs * self.confidence[index, :]`，沒有再乘
`partial_Y`），而 confidence 只有 warm-up 結束後才會被 `confidence_update` 刷新（見上方第 2 項）——
兩者疊加的結果：整個 warm-up 期間，分類損失都是拿「全 C 類均勻」當目標分佈，等同於單純把模型的
輸出往均勻分佈推，而不是論文設計的「候選集內均勻、候選集外靠 softmax 正規化間接壓低、不直接給
它們訊號」。原版預設 `prot_start=80`，代表 80 epoch 的分類訊號被嚴重稀釋。

**已修正 `PiCO-Fixed`**（`_candidate_masked_init_conf`，`src/pipeline/algorithms/runners.py`），
跟 `ProdenLoss.__init__` 用同一種建構方式（候選集內 `1/|s_i|`，候選集外 0）。**`PiCO` 原版維持
不動**，只是本專案沿用至今的既有行為，不重現論文這個初始化細節。

### 4. 其餘逐項核對：全部一致

| 論文 | 程式碼 | 結果 |
|---|---|---|
| 分類損失 Eq. 1：$L_{cls}=\frac1{\|B\|}\sum_i\sum_j -s_{ij}\log f^j(\text{Aug}_q(x_i))$ | `PartialLoss.forward` | ✅ 完全一致 |
| 對比損失 Eq. 3/4：InfoNCE，anchor 只用 $B_q$，正樣本集合由候選集相似度 mask 決定，池 $A=B_q\cup B_k\cup\text{queue}$ | `SupConLoss.forward`（mask 分支）+ `train_pico_epoch` 的 mask 建構 | ✅ 完全一致，包含論文的「label queue 儲存過去預測」（`queue_pseudo`） |
| 總損失 Eq. 5：$L=L_{cls}+\lambda L_{cont}$，預設 $\lambda=0.5$ | `loss = loss_cls + loss_weight*loss_cont`，`config.yaml: loss_weight: 0.5` | ✅ 完全一致 |
| 超參數預設：`moco_queue=8192`、`moco_m=0.999`、`proto_m=0.99`、`τ=0.07`、`low_dim=128` | `config.yaml` / `SupConLoss.__init__` 預設值 | ✅ 全部一致 |

---

## 原論文使用的 Benchmark（Step 3，已完成）

- **資料集**：CIFAR-10、CIFAR-100、CUB-200、CIFAR-100-H（2026-08-14 更新：`cub200`、`cifar100-h`
  現在都可以透過 `--dataset` 跑，見下方「實驗 Config」）
- **候選集生成**：uniform flip 機率 $q$，CIFAR-10 用 $q\in\{0.1,0.3,0.5\}$，CIFAR-100 用
  $q\in\{0.01,0.05,0.1\}$。CIFAR-100-H（Section 4.4，論文原文逐字確認過）用同一套「每個候選標籤
  獨立以機率 $q$ 加入」機制，但只在**同一個 coarse superclass** 內操作，$q=0.5$（Table 6 另外
  測過 $q\in\{0.1,0.5,0.8\}$）；CUB-200 用 $q=0.05$。
- **Backbone**：18 層 ResNet + 2 層 MLP projection head → 128 維（本專案的 `SupConResNet` 一致）
- **訓練設定**：SGD momentum 0.9 + cosine LR，**batch size 256**，**800 epoch**（本專案專案級預設
  是 batch 512、最多 1000 epoch，是全部 6 個方法共用設定，不是 PiCO 專屬調整）
- **評估**：5 個 random seed 的平均±標準差，10% held-out clean validation（訓練時折回 PLL 訓練集）

---

## Fixed 版本（Step 4，已完成）

`src/fixed_pico_engine.py`：新增 `train_pico_epoch_fixed`，**唯一修改**是 warm-up 期間
（`epoch < prot_start`）直接用 `loss = loss_cls`（完全不計算、不加 $L_{cont}$），而不是像原版
繼續算一個 unsupervised MoCo 版本的 $L_{cont}$ 加進去。confidence/pseudo-target 更新的 gate
維持不變（原因見上方 Step 2 第 2 項的討論）。

`run_pico_fixed`（`src/pipeline/algorithms/runners.py`）：
- 沿用跟 `run_pico` 相同的模型/loss 建構，只是訓練迴圈換成 `train_pico_epoch_fixed`
- `prot_start` 預設改成 **1**（跟隨論文的一般預設；若要重現 CIFAR-100 q=0.1 的設定，需要在
  `config.yaml` 的 `pico` 區塊加一個 `prot_start_fixed: 100` 覆寫，程式碼已支援這個可選 key）
- **（2026-08-16 新增）pseudo target 初始化改用 `_candidate_masked_init_conf`**：候選集內
  `1/|s_i|`、候選集外 0（論文 Eq. 6：$s_j=\frac1{|Y|}\mathbb{I}(j\in Y)$），取代原本的
  `torch.ones(N,C)/C` 全類均勻。見上方 Step 2 第 3 項。單元測試（純 CPU，不需要資料集）驗證過
  `[0,2,5]` 候選集在 C=6 下輸出 `[1/3,0,1/3,0,0,1/3]`，總和為 1，數值正確。
- `PiCO`（原版）**維持不動**，這兩個落差（warm-up $L_{cont}$ 處理方式、pseudo target 初始化）
  都只反映在 `PiCO-Fixed`。
- 已註冊為 `AlgorithmSpec('PiCO-Fixed', 'PLL', r.run_pico_fixed)`

---

## 實驗 Config（Step 5）

> **2026-08-14 更新**：`--dataset cub200`（真實鳥類照片，抓自 HuggingFace mirror，官方 Caltech
> 連結已死）跟 `--dataset cifar100-h`（階層式候選集生成，細節與 caveat 見
> [00_paper_alignment_guide.md](00_paper_alignment_guide.md) 的「資料集支援」一節）都已經在本機
> 端到端驗證過。CUB-200 因為本專案的 CNN 架構是針對小圖（~32px）設計的，載入時會 resize 到
> 64×64，跟論文可能用的更大解析度/預訓練 backbone 有落差，細節見
> `src/pipeline/datasets/cub200.py` docstring。
>
> ```bash
> python scripts/run_pipeline.py run --run_name pico_original_benchmark \
>     --algorithms PiCO PiCO-Fixed --dataset cub200 --epochs 100
> python scripts/run_pipeline.py run --run_name pico_original_benchmark \
>     --algorithms PiCO PiCO-Fixed --dataset cifar100-h --c_values 100 --epochs 200
> ```

```bash
python scripts/run_pipeline.py run --run_name pico_warmup_check \
    --algorithms PiCO PiCO-Fixed --c_values 5 20 --epochs 200
```

- **建議 `PiCO` 跟 `PiCO-Fixed` 並排跑**，直接驗證 warm-up 機制的差異對最終準確率有多大影響——
  尤其 `PiCO-Fixed` 的 `prot_start=1` 比原版 `prot_start=80` 短很多，訓練初期的 loss 曲線應該會
  有明顯差異，值得同時記錄 per-epoch 準確率曲線（不只是最終準確率）來觀察。
- 若要更貼近論文的難設定（CIFAR-100, q=0.1, prot_start=100），需要另外準備一組
  `config.yaml` 覆寫 `pico.prot_start_fixed: 100`。
- PiCO 對 `moco_queue % batch_size` 的 assert 特別敏感，若要調整 `--batch_size`，記得同步確認
  `config.yaml` 的 `pico.moco_queue`（預設 8192）仍然整除。
