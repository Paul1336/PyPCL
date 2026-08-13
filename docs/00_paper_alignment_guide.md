# 論文 ↔ 實作對應：主引導文件

> **這份文件是給「未來 session」的入口點**，不是任務本身的完成報告。目前只完成了「文件骨架 +
> 程式碼對應」，論文公式逐項比對、`fixed_` 修正實作、實驗 config 都還沒開始（見下方各節的
> 前置條件與範圍）。如果你是被指派來接續這個任務的 session，請先讀完這份文件再開始動手。

## 任務由來

使用者提供了 8 篇論文（涵蓋 Partial Label Learning / Complementary Label Learning 六大方法家族），
要求對每一篇完成以下 5 件事：

1. 在 repo 中找出對應的實作程式碼，在 README 補上對應說明，並在程式碼裡加註解
2. 詳細比對「目前實作」是否符合「論文演算法」，若有差異要另外寫文件說明差異、以及原論文怎麼做
3. 在每篇論文對應的文件中，說明原論文使用的 benchmark 是什麼
4. 根據上述文件，產出該方法的「修正版」（`fixed` 前綴），確保跟原論文方法一致
5. 根據上述文件，用目前的主要 pipeline + 論文原始 benchmark（若目前沒實作）生成多組 config 跑實驗

這 5 步驟工作量很大、且部分依賴讀論文原文才能完成，因此**先產出這份引導文件**，讓工作可以被拆分到
多個 session 接力完成，而不是一次做完。

## 前置條件：主 Pipeline 尚未驗證可跑

repo 裡現在有兩套 pipeline：

| | 舊版 pipeline | 新版 pipeline（`src/pipeline/`，目前 untracked） |
|---|---|---|
| 進入點 | `scripts/run_experiment.py` | `scripts/run_pipeline.py`（`run`/`merge`/`plot` 三個子指令） |
| 資料集 | CIFAR-10 / CIFAR-20 / CLCIFAR-10 / CLCIFAR-20 | CIFAR-100 任意類別子集（`src/cifar100_subset.py`），依 C（類別數）× k（partial label 數）掃描 |
| 已串接演算法 | 7 個（PRODEN, MCL-LOG/MAE/EXP, ComCo, PiCO, SoLar），寫死在 `main()` 裡，不可用 CLI 選 | 14 個（見下表），透過 `src/pipeline/algorithms/__init__.py` 的 registry 統一呼叫 |
| Config 來源 | `config.yaml` 的 `training:`/`data_generation:` 區塊直接餵給 `argparse` 預設值 | `config.yaml` 只提供 `pico:`/`comco:`/`solar:` 演算法專屬超參數；一般訓練參數（epochs/batch_size/lr...)改成 `run_pipeline.py` 內建的 CLI 預設值，兩者**不是同一套** |

**新版 pipeline 的程式碼本身是完整、通過所有靜態一致性檢查的**（import 全部能解析、函式簽名對得上、
config key 對得上），但**在本機環境從來沒有成功跑完一次真實資料的訓練**。截至 2026-08-13 21:44
的狀態：

- `scripts/legacy/` 已建立，57 支舊 script 都搬過去了；`README.md` 也已補上新版 pipeline 的用法說明
  —— 這兩項先前的落差已經修好
- 曾經跑過一次**合成資料**的 smoke test（`logs/smoke_synth/*.json`，100 筆訓練/50 筆測試的假資料）——
  這只驗證了資料生成邏輯，**不是真實訓練**
- `results/smoke/results.csv`、`results/smoke/shards/` 都還不存在 —— 代表目前**沒有任何一個
  (C, k, algorithm) 組合用真實資料跑完過一次**
- `data/cifar-100-python.tar.gz` 在本機仍是壞檔（102,824,000 bytes，正常應該 ~161MB），
  `tarfile.open()` 會丟 `gzip.EOFError`

**如果是在已經有下載好的 CIFAR-100 的 server 上執行**（使用者已確認會這樣做）：

- 資料下載這步驟不適用，可以跳過「重新下載 tarball」
- 但**「從沒有用真實資料跑完一次」這件事本身還沒被驗證過**——目前為止只確認過程式碼靜態一致（import/
  簽名對得上）和合成假資料能跑過資料生成邏輯，兩者都不等於「真實訓練迴圈能跑完、loss 會下降、
  `results.csv` 能正確寫出」。**強烈建議在 server 上，動手做 Step 4/5 之前，先花幾分鐘跑一次最簡單的
  真實 smoke test**：
  ```bash
  python scripts/run_pipeline.py run --run_name smoke_real \
      --algorithms CLPL --c_values 5 --epochs 1
  ```
  確認 `results/smoke_real/results.csv` 有正確產生、數字合理（不是 NaN、不是隨機猜測的準確率），
  再開始大規模跑 Step 5 的實驗 config，避免所有論文的實驗都建立在一個沒驗證過的訓練迴圈上。
- 留意 `src/pico/model.py:34`、`src/comco/model.py:35` 的 `assert moco_queue % batch_size == 0` ——
  自訂 `--batch_size` 時容易在 PiCO / PiCO-MCL / PiCO-SC / ComCo 上炸掉，跑這幾個演算法的 smoke test
  時要特別確認

**這個前置條件只擋 Step 4 的「驗證」與 Step 5。Step 1–3（文件、公式比對、benchmark 標註）是純閱讀/
寫文件的工作，不需要跑程式，現在就可以做，不用等 pipeline 驗證完成。**

## 論文 ↔ 程式碼對應總表

| 論文 | Algorithm ID | 主要程式碼 | 新版 entry point | 舊版 entry point | 對應文件 |
|---|---|---|---|---|---|
| PRODEN — Lv et al., ICML 2020 | `PRODEN` | `src/proden_loss.py` (`ProdenLoss`) | `run_proden`（`pipeline/algorithms/runners.py:194`） | `setup_proden`（`model_setup.py:21`）/ `run_proden_training`（`training_pipelines.py:15`） | [proden_explanation.md](proden_explanation.md) |
| Cour 2011 / CLPL — Cour, Sapp & Taskar, JMLR 2011 | `CLPL` | `src/clpl_loss.py` (`CLPLSquaredHingeLoss`) | `run_clpl`（`runners.py:126`） | `setup_cour`（`model_setup.py:14`）/ `run_cour_training`（`training_pipelines.py:5`） | [cour2011_explanation.md](cour2011_explanation.md)（**已修正**，原本誤指到孤兒檔案，見下方已知缺陷） |
| PiCO — Wang et al., ICLR 2022 | `PiCO` | `src/pico/model.py` (`PiCOModel`), `src/pico/utils_loss.py` (`PartialLoss`, `SupConLoss`) | `run_pico`（`runners.py:228`） | `setup_pico`（`model_setup.py:78`）/ `run_pico_training`（`training_pipelines.py:69`） | [pico_explanation.md](pico_explanation.md) |
| SoLar — Wang et al., NeurIPS 2022 | `SoLar` | `src/solar/utils_loss.py` (`partial_loss`), `src/solar/utils_algo.py` (`sinkhorn`, `linear_rampup`) | `run_solar`（`runners.py:387`） | `setup_solar`（`model_setup.py:149`）/ `run_solar_training`（`training_pipelines.py:155`） | [solar_explanation.md](solar_explanation.md) |
| MCL-LOG — Feng et al., ICML 2020 | `MCL-LOG` | `src/mcl_losses.py` (`MCL_LOG`；`MCL_MAE`/`MCL_EXP` 同檔) | `run_mcl_log`（`runners.py:134`） | `setup_mcl`（`model_setup.py:28`，`loss_type` 參數化）/ `run_mcl_training`（`training_pipelines.py:25`） | [mcl_explanation.md](mcl_explanation.md) |
| SCL-NL — Chou et al., ICML 2020 | `SCL-NL` | `src/scl_loss.py` (`SCL_NL`) | `run_scl_nl`（`runners.py:138`） | `setup_scl`（`model_setup.py:42`）/ `run_scl_training`（`training_pipelines.py:35`） | [scl_nl_explanation.md](scl_nl_explanation.md) |
| OP-W — Liu et al., AISTATS 2023 | `OP-W`（`OP` 亦存在） | `src/op_loss.py` (`OPWLoss`；`OPLoss`同檔) | `run_op_w`（`runners.py:146`），`run_op`（`runners.py:142`） | 無（只在新版 pipeline） | [op_w_explanation.md](op_w_explanation.md) |
| ComCo — Jiang, Sun & Tian, Neural Networks 2024 | `ComCo` | `src/comco/model.py` (`ComCoModel`), `src/comco/utils_loss.py` (`ComCoCLSLoss`, `ComCoContrastiveLoss`) | `run_comco`（`runners.py:348`） | `setup_comco`（`model_setup.py`，**重複定義**見下）/ `run_comco_training`（`training_pipelines.py`，**重複定義**見下） | [comco_explanation.md](comco_explanation.md) |

> repo 內另外還有 6 個沒有對應到使用者這份論文清單的演算法（`Wu2022`、`PiCO-MCL`、`PiCO-SC`、
> `PiCO-CLS`、`MCL-MAE`、`MCL-EXP`、`CPE`）。依使用者指示，**這份引導文件不處理它們**，範圍只限定在
> 上表 8 篇論文。

## 完成度追蹤表

多個 session 接力執行時，用這張表記錄目前做到哪一步。**每完成一步，請直接編輯這張表**（不要另外開
進度文件）。

| 論文 | Step 1（README+註解） | Step 2（保真度比對） | Step 3（benchmark 標註） | Step 4（fixed 實作） | Step 5（實驗 config） |
|---|---|---|---|---|---|
| PRODEN | ☐ | ☐（骨架已建，公式比對待填） | ☐（待填） | ☐ | ☐ |
| Cour2011/CLPL | ☐ | ☐（骨架已建並修正程式碼指向，公式比對待填） | ☐（待填） | ☐ | ☐ |
| PiCO | ☐ | ☐（骨架已建，公式比對待填） | ☐（待填） | ☐ | ☐ |
| SoLar | ☐ | ☐（骨架已建，公式比對待填） | ☐（待填） | ☐ | ☐ |
| MCL-LOG | ☐ | ☐（骨架已建，公式比對待填） | ☐（待填） | ☐ | ☐ |
| SCL-NL | ☐ | ☐（骨架已建，公式比對待填） | ☐（待填） | ☐ | ☐ |
| OP-W | ☐ | ☐（骨架已建，公式比對待填） | ☐（待填） | ☐ | ☐ |
| ComCo | ☐ | ☐（骨架已建，公式比對待填） | ☐（待填） | ☐ | ☐ |

Step 4/5 在 pipeline 前置條件（見上）完成之前不要開始。

## `fixed_` 命名慣例（Step 4 用）

- 單檔案演算法（PRODEN、CLPL、MCL、SCL-NL、OP-W）：在同一層新增 `fixed_<原檔名>.py`
  （例如 `src/fixed_proden_loss.py`），類別名稱前綴 `Fixed`（例如 `FixedProdenLoss`）。
- 子套件演算法（PiCO、SoLar、ComCo）：在同一子套件目錄下新增 `fixed_<原檔名>.py`
  （例如 `src/pico/fixed_utils_loss.py`），不要新建整個平行子套件。
- **每個 `fixed_*` 檔案開頭要有一段註解，寫清楚**：改了哪裡、對應到該論文 Step 2 文件裡的哪一項
  差異、為什麼這樣改才符合原論文。不要沒有依據地「順手改」——所有修改都必須能回溯到 Step 2 的
  比對結論。
- `fixed_*` 完成後，要註冊進 `src/pipeline/algorithms/__init__.py`（新增一個 `AlgorithmSpec`，
  Algorithm ID 建議用 `<原ID>-Fixed`，例如 `PRODEN-Fixed`），並在該論文的 Step 4 章節補上連結。
- 寫完後至少跑一次 pipeline smoke test（1 epoch）確認能跑，再進入 Step 5。

## 已知程式碼缺陷（Step 2/4 時請一併留意）

這些是本次探索（讀程式碼，尚未讀論文）已經發現、但還沒有處理的問題：

1. **`src/cour_loss.py` 是孤兒檔案**：定義 `UniformCandidateCrossEntropyLoss`（別名 `CourLoss`），
   均等平均候選標籤的 CE loss，但完全沒有被 `model_setup.py`、`training_pipelines.py`、
   `pipeline/algorithms/runners.py` 任何一處匯入。它自己的 docstring 也承認「NOTE: This is NOT the
   CLPL loss from Cour, Sapp & Taskar (JMLR 2011)... kept for reference and backward-compatibility
   with earlier experiment runs」。真正被使用、對應論文的是 `src/clpl_loss.py` 的
   `CLPLSquaredHingeLoss`。**下一個 session 處理 CLPL 論文文件時，需要決定 `cour_loss.py` 到底要
   刪除、保留當歷史備查、還是重新命名成不會混淆的名字**——不要直接刪，先確認是否有舊實驗結果依賴它。
2. **`setup_comco` 重複定義**：`src/model_setup.py` 裡定義了兩次（約第 52 行、第 96 行），內容幾乎
   相同，後者會覆蓋前者。`src/training_pipelines.py` 的 `run_comco_training` 同樣重複定義。屬於
   copy-paste 遺留的死碼，處理 ComCo 論文文件時要清理。
3. **`MCL_MAE`/`MCL_EXP` 未註冊進新版 pipeline**：兩者都在 `src/mcl_losses.py` 裡定義好，且能透過
   舊版 pipeline 的 `setup_mcl(loss_type='mae'|'exp')` 執行，但 `pipeline/algorithms/__init__.py`
   只註冊了 `MCL-LOG`。使用者這次只要求 MCL-LOG 對應的論文（Feng et al. 2020 涵蓋三種變體），
   Step 2 文件應該提到 MAE/EXP 的存在，但註冊與否留給該 session 判斷是否在範圍內。
4. **`moco_queue % batch_size` 的 assert 隱患**：`src/pico/model.py:34`、`src/comco/model.py:35`
   都有 `assert args['moco_queue'] % batch_size == 0`。目前預設 `batch_size=512`、
   `moco_queue=8192`（能整除）所以沒事，但 Step 5 設計實驗 config 時，若要用非預設 batch size 掃
   PiCO / PiCO-MCL / PiCO-SC / ComCo，要先確認整除，否則會直接 crash。
5. **`scripts/run_pipeline.py` docstring 與現實不符**：宣稱舊 script 已搬到 `scripts/legacy/`，但
   該目錄不存在，57 支舊 script 仍全部留在 `scripts/` 底下。屬於遷移工作本身沒做完，不影響
   Step 1–5 的內容，但如果之後有 session 要動 `scripts/` 目錄結構，先知道這件事。

## Step 3 的一個重要提醒：原論文 benchmark 不一定是 CIFAR

目前 pipeline（新舊皆然）只支援 CIFAR-10 / CIFAR-20 / CIFAR-100 子集 / CLCIFAR-10 / CLCIFAR-20。
但論文原始實驗不一定用 CIFAR。已知的落差（尚未逐篇查證，僅先標記提醒）：

- **Cour 2011**：原論文用的是 Yahoo! News、MSRCv2、Lost 等人臉/物件辨識 partial-label 資料集，
  跟 CIFAR 完全是不同模態的資料。用 CIFAR 近似是「借用同一套損失函數在不同資料集上驗證」，並不是
  複現原論文的實驗設定，Step 3 文件要把這點講清楚，不要含糊帶過。
- 其餘 7 篇論文大多本來就有用 CIFAR-10/CIFAR-100（有些變體），需要在各自 Step 3 文件裡確認真正
  用的是哪個切分、有沒有用 PLL/CLL 特化的合成噪聲設定，再對照目前 pipeline 的 CIFAR-10/20/100
  子集生成方式（`src/data_setup.py`、`src/cifar100_subset.py`）是否一致。
- 如果原論文 benchmark 現有 pipeline 無法近似（例如需要全新的 dataset loader），**不要自行假設
  要不要新增**——先在該論文的 Step 3 文件寫清楚落差，交由使用者決定是否要擴充 pipeline 支援新資料集，
  這超出「用現有主 pipeline 生成 config」的範圍。

## 每篇論文文件的標準模板

以下是骨架文件共用的章節結構（已套用到 `docs/*_explanation.md`），未來 session 逐項填空即可：

1. **論文出處**：作者、會議/期刊、連結（直接沿用使用者提供的表格）
2. **對應程式碼**：檔案、class/function、行號、目前透過哪個 entry point 被呼叫（新舊 pipeline 都要列）
3. **演算法保真度比對**（Step 2，目前是 TODO）：論文公式 vs 程式碼實作逐項對照，列出「一致」與
   「不一致」，不一致要說明原論文怎麼做、程式碼怎麼做、可能造成的影響
4. **原論文 benchmark**（Step 3，目前是 TODO）：資料集、split、評估方式、跟目前 pipeline 的落差
5. **Fixed 版本**（Step 4 完成後回填）：連結到 `fixed_*` 實作，逐條列出改了什麼、依據 Step 2 的
   哪一項發現
6. **實驗 config**（Step 5 完成後回填）：對應 config 檔案連結、跑法、結果摘要位置
