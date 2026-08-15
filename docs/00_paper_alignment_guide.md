# 論文 ↔ 實作對應：主引導文件

> **這份文件是給「未來 session」的入口點**。截至 **2026-08-14**：使用者提供了 6 篇論文 PDF
> （`C:\Users\User\Desktop\papers\{CLPL,ComCo,MCL-LOG,PRODEN,PiCO,SCL-NL}.pdf`），這 6 篇已經
> **完整跑完 Step 1-5**（README/程式碼註解、公式逐項比對、benchmark 查證、fixed 版本、實驗
> config）。剩下 **SoLar、OP-W 兩篇還沒有 PDF**，仍停留在 Step 1 骨架階段，需要使用者提供 PDF
> 才能繼續。如果你是被指派來接續這個任務的 session，請先讀完下方「完成度追蹤表」確認目前狀態，
> 再決定要做哪一塊。
>
> **同一天（2026-08-14）稍後**：pipeline 進一步擴充，新增了 `--dataset` 旗標，讓 6 篇論文的
> **原始 benchmark**（不只是 CIFAR 衍生資料）現在真的可以透過主 pipeline 跑實驗。詳見下方
> 「資料集支援（`--dataset`）」一節，以及 `docs/dataset_availability_report.md`。
>
> **同一天再稍後**：裝了 `pypdf`（純文字擷取，不需要 poppler/pdftoppm），對 6 篇論文全部重新做
> 一次逐字覆核（不是重新讀論文，是拿已經寫好的文件對照論文原文抓漏）。**結果：6 篇文件的公式
> 轉錄跟保真度判定全部正確，沒有發現新的錯誤**，包括先前發現的兩個 bug（MCL-LOG 的 scaling
> factor、ComCo 的分類損失 scaling）都用論文原文的精確字句再次確認過。每篇文件的開頭都補上了
> 「二次覆核」段落，附上關鍵原文引用。這次覆核順便也用同樣的方法確認了 CIFAR-100-H 的真正生成
> 方式（見「資料集支援」一節），過程中發現並修正了一個原本沒抓到的落差（k-based vs 論文實際的
> q-based 生成）。

## 資料集支援（`--dataset`）

前面 Step 3 的比對發現：這 6 篇論文的原始 benchmark 大多不是 CIFAR（見「Step 3 的一個重要提醒」
一節）。這次新增了一個**資料集註冊架構**（仿照既有的演算法 registry pattern，見
`src/pipeline/datasets/`），讓 `scripts/run_pipeline.py run` 可以用 `--dataset <name>` 選擇
訓練資料集，不再只能用 CIFAR-100 子集。**每一個都在本機真的跑過一次端到端訓練驗證過**（不只是
語法檢查），細節見 `src/pipeline/datasets/` 底下每個模組的 docstring。

| `--dataset` 值 | 對應論文 | 型態 | 備註 |
|---|---|---|---|
| `cifar100-subset`（預設） | — | 影像 | 原本就有，行為完全不變（已用回歸測試確認） |
| `mnist` / `fashion-mnist` / `kmnist` | PRODEN, MCL-LOG, SCL-NL | 影像（灰階，28×28） | torchvision 內建；不支援 PiCO/ComCo/SoLar（灰階） |
| `dermatology` / `ecoli` / `abalone` | CLPL | 表格（UCI） | Cour et al. 2011 的 UCI benchmark；MLP backbone；只支援 simple-shape 演算法 |
| `dermatology` / `yeast` / `synthetic-control` / `texture` | PRODEN, MCL-LOG | 表格（UCI/OpenML） | **注意**：跟 CLPL 只有 `dermatology` 重疊，不是同一組 4 個資料集；`texture` 不在 `ucimlrepo` 裡，改抓 OpenML id=40499；MLP backbone；只支援 simple-shape 演算法 |
| `20newsgroups` | MCL-LOG | 文字（TF-IDF） | MLP backbone；只支援 simple-shape 演算法 |
| `cub200` | PiCO, ComCo | 影像（RGB，實際下載自 HuggingFace mirror，官方 Caltech 連結已死） | 64×64 resize（原始解析度跟本專案 CNN 架構不搭，細節見 `cub200.py` docstring） |
| `cifar100-h` | PiCO | 影像 | CIFAR-100 + 階層式（同 coarse superclass）候選集生成，q=0.5（**論文原文逐字確認過**，見下方） |
| `sun397` | ComCo | 影像（RGB） | **程式碼已寫好、已註冊，但因資料集過大（~17-37GB）沒有實際下載驗證過**，見下方 |
| `lost` / `msrcv2` / `birdsong` / `soccer-player` / `yahoo-news` | PRODEN, MCL-LOG | 表格（**真實候選集**，非合成） | 5 個經典 PLL 真實資料集；CL 為 PL 補集合成，訓練時會印出警告 |
| `clpl-lost` | CLPL | 影像（RGB，**真實候選集**） | CLPL 論文自己的原始資料，跟 `lost`（表格版）是同一組底層資料的兩種特徵表示 |
| `clpl-fiw` | CLPL | 影像（灰階，48×48） | CLPL 論文的 LFW 衍生資料，乾淨標籤，合成生成候選集 |

**已知限制/待辦**：
- ~~`soccer-player`、`yahoo-news` 未驗證~~ **已解決（2026-08-14 稍後補做）**：兩個都實際下載
  （36MB/28.7MB）+ 解析 + 訓練驗證過，維度跟論文完全吻合（Soccer Player：17472 樣本/279 特徵/
  171 類；Yahoo!News：22991 樣本/163 特徵/219 類）。5 個真實 PLL 資料集現在全部驗證完畢。
- ~~`cifar100-h` 定義未經論文原文確認~~ **已解決**：改用 `pypdf`（純文字擷取，不需要
  `pdftoppm`/poppler）直接讀出 PiCO 論文 Section 4.4 原文，逐字確認："CIFAR-100 with hierarchical
  labels (CIFAR-100-H), where we generate candidate labels that belong to the same superclass...
  We set q=0.5 for CIFAR-100 with hierarchical labels"——**是 q 機率式生成（每個同 superclass 的
  假標籤各自獨立以機率 q 被加入候選集），不是固定大小 k**，跟原本實作的 k-based 版本不同。已修正：
  `ComparisonDataGenerator` 新增 `generate_pl_dataset_hierarchical_variable(q, class_coarse)`
  （論文精確版），`--dataset cifar100-h` 現在預設用這個（q=0.5，論文 Table 3 的主要設定），原本的
  k-based 版本（`generate_pl_dataset_hierarchical`）保留但不再是 `--dataset cifar100-h` 的預設路徑
  ——因為本 pipeline 的 CLI 是圍繞 k-sweep 設計的，q-based 生成改成單一 cell（`DatasetSpec.
  sweeps_k=False`，跟 preambiguous 資料集共用同一套機制），不支援像 k 一樣 sweep 多個 q 值。用
  獨立單元測試驗證過生成邏輯正確（候選集絕不跨 superclass、平均候選集大小 ≈ 3.03，符合
  q=0.5 時 1+Binomial(4,0.5) 的理論期望值 3）。**用小 C 子集跑時，大部分樣本會因為同 superclass
  的其他類別沒被選進子集而 fallback 成均勻採樣**（C=20 時約 85% fallback）——這個模式在 C 接近
  100（用滿所有類別）時才最有意義，程式碼會印出 fallback 比例，不會靜默發生。
- `sun397`（ComCo Section 5.6 的補充實驗）**程式碼已經寫好、註冊進 registry**（跟 `cub200` 同一套
  lazy-path 載入機制，`src/pipeline/datasets/sun397.py`），但這次沒有實際觸發下載——HuggingFace
  上找到的最小 mirror（`tanganke/sun397`）也要 ~17GB，官方 torchvision 版本更是 ~37GB，評估後
  判斷這次不值得為了驗證下載這麼大的資料。**第一次真的用 `--dataset sun397` 之前，建議先用
  `--epochs 1 --batch_size 16` 之類的小規模設定跑一次 smoke test**，確認下載、解壓、訓練整條路徑
  沒問題，不要直接假設能跑。
- `--dataset` 目前只有 `run` 子指令支援；`plot` 子指令、`results.py::load_results` 還沒有
  按資料集分開畫圖的邏輯（多資料集混在同一個 run 裡畫圖時，同樣的 (C,k,algorithm) 在不同資料集
  下會互相覆蓋，只是畫圖層級的限制，`results.csv` 本身的資料是正確分開存的）。
- Phase 0 的探測腳本 `scripts/probe_dataset_availability.py` 值得定期重跑，因為這次用到的幾個
  外部資料來源都是個人學術網頁/非官方鏡像（`timotheecour.com`、`palm.seu.edu.cn`），沒有 SLA
  保證長期可用。

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

## Pipeline 狀態：已在 server 上驗證可跑

**2026-08-14 更新**：使用者已在自己的 server 上（已有下載好的 CIFAR-100）成功跑過新版 pipeline，
Step 4/5 的前置條件已解除。以下維持原本對兩套 pipeline 架構的記錄，供之後理解程式碼時參考；
「尚未驗證可跑」的內容已不適用，僅保留在文件下方當歷史記錄。

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

| 論文 | Algorithm ID | Fixed 版 Algorithm ID | 主要程式碼 | 新版 entry point | 對應文件 |
|---|---|---|---|---|---|
| PRODEN — Lv et al., ICML 2020 | `PRODEN` | 不需要（`ProdenLoss` 忠實） | `src/proden_loss.py` (`ProdenLoss`) | `run_proden`（`runners.py:194`） | [proden_explanation.md](proden_explanation.md) |
| Cour 2011 / CLPL — Cour, Sapp & Taskar, JMLR 2011 | `CLPL` | 不需要（忠實） | `src/clpl_loss.py` (`CLPLSquaredHingeLoss`) | `run_clpl`（`runners.py:126`） | [cour2011_explanation.md](cour2011_explanation.md)（**已修正**，原本誤指到孤兒檔案） |
| PiCO — Wang et al., ICLR 2022 | `PiCO` | `PiCO-Fixed`（warm-up 機制修正） | `src/pico/model.py`、`src/pico/utils_loss.py` | `run_pico`（`runners.py`） | [pico_explanation.md](pico_explanation.md) |
| SoLar — Wang et al., NeurIPS 2022 | `SoLar` | **尚無 PDF，未開始** | `src/solar/utils_loss.py`、`src/solar/utils_algo.py` | `run_solar`（`runners.py`） | [solar_explanation.md](solar_explanation.md) |
| MCL-LOG — Feng et al., ICML 2020 | `MCL-LOG` | `MCL-LOG-Fixed`（URE scaling 修正） | `src/mcl_losses.py` (`MCL_LOG`) | `run_mcl_log`（`runners.py`） | [mcl_explanation.md](mcl_explanation.md) |
| SCL-NL — Chou et al., ICML 2020 | `SCL-NL` | 不需要（單CL公式忠實） | `src/scl_loss.py` (`SCL_NL`) | `run_scl_nl`（`runners.py`） | [scl_nl_explanation.md](scl_nl_explanation.md) |
| OP-W — Liu et al., AISTATS 2023 | `OP-W`（`OP` 亦存在） | **尚無 PDF，未開始** | `src/op_loss.py` (`OPWLoss`) | `run_op_w`（`runners.py`） | [op_w_explanation.md](op_w_explanation.md) |
| ComCo — Jiang, Sun & Tian, Neural Networks 2024 | `ComCo` | `ComCo-Fixed`（分類損失 scaling 修正） | `src/comco/model.py`、`src/comco/utils_loss.py` | `run_comco`（`runners.py`） | [comco_explanation.md](comco_explanation.md) |

> repo 內另外還有 6 個沒有對應到使用者這份論文清單的演算法（`Wu2022`、`PiCO-MCL`、`PiCO-SC`、
> `PiCO-CLS`、`MCL-MAE`/`MCL-EXP`、`CPE`）。依使用者指示，**這份引導文件不處理它們**，範圍只限定在
> 上表 8 篇論文。（`MCL-MAE`/`MCL-EXP` 的 fixed 版本 `FixedMCLMae`/`FixedMCLExp` 已經在
> `src/fixed_mcl_losses.py` 裡順手寫好——因為跟 `MCL_LOG` 是同一個 scaling bug、同一個檔案，
> 修 `MCL_LOG` 時一起改了——但目前**沒有註冊進 pipeline registry**，因為 MAE/EXP 本身不在這次
> 8 篇論文範圍內，是否要註冊留給之後需要用到的 session 決定。）

## 完成度追蹤表

多個 session 接力執行時，用這張表記錄目前做到哪一步。**每完成一步，請直接編輯這張表**（不要另外開
進度文件）。

| 論文 | Step 1（README+註解） | Step 2（保真度比對） | Step 3（benchmark 標註） | Step 4（fixed 實作） | Step 5（實驗 config） |
|---|---|---|---|---|---|
| PRODEN | ✅ | ✅ 忠實（`ProdenLoss`）；`proden`（舊版 pipeline 專用）有錯但不影響 Step 5 | ✅ MNIST/F-MNIST/K-MNIST/CIFAR-10 + UCI，非本專案資料集 | ✅ 不需要 | ✅ `scripts/run_paper_alignment_experiments.sh` |
| Cour2011/CLPL | ✅ | ✅ 完全忠實（Eq. 2） | ✅ **完全不是 CIFAR**（UCI/人臉/真實電視劇弱標註） | ✅ 不需要 | ✅ 同上 |
| PiCO | ✅ | ✅ 核心機制忠實；⚠️ warm-up 機制與預設值不符論文 | ✅ CIFAR-10/100/CUB-200/CIFAR-100-H | ✅ `PiCO-Fixed` | ✅ 同上 |
| SoLar | ☐ | ☐（骨架已建，**待 PDF**） | ☐（待填） | ☐ | ☐ |
| MCL-LOG | ✅ | ✅ Loss 公式忠實；❌ **URE scaling factor 錯誤**（已修正） | ✅ MNIST/F-MNIST/K-MNIST/20News/CIFAR-10 + UCI | ✅ `MCL-LOG-Fixed` | ✅ 同上 |
| SCL-NL | ✅ | ✅ 單CL公式忠實；multi-CL wrapper 無論文依據（非錯誤，僅記錄） | ✅ MNIST/K-MNIST/F-MNIST/CIFAR-10 | ✅ 不需要 | ✅ 同上 |
| OP-W | ☐ | ☐（骨架已建，**待 PDF**） | ☐（待填） | ☐ | ☐ |
| ComCo | ✅ | ✅ 對比損失完全忠實；❌ **分類損失 scaling 錯誤**（已修正） | ✅ 6 資料集 5 協定，遠比本專案評測範圍廣 | ✅ `ComCo-Fixed` | ✅ 同上 |

**SoLar、OP-W 需要使用者提供論文 PDF 才能繼續**（放到 `C:\Users\User\Desktop\papers\SoLar.pdf`、
`C:\Users\User\Desktop\papers\OP-W.pdf`，流程比照這次 6 篇的做法：讀 PDF → 對照
`src/solar/utils_loss.py`/`utils_algo.py`（或 `src/op_loss.py`）逐項比對 → 更新對應骨架文件 →
視發現決定是否需要 fixed 版本）。

## Step 5 實驗腳本

`scripts/run_paper_alignment_experiments.sh` 已經把這 6 篇論文的實驗 config 整理成一支可執行腳本：
- Group A：3 組有 bug 的演算法（`MCL-LOG`/`ComCo`/`PiCO`）跟各自的 `-Fixed` 版本並排跑
- Group B：3 個已驗證忠實、不需要 fixed 版本的演算法（`CLPL`/`PRODEN`/`SCL-NL`）
- Group C（選用）：貼近各篇論文原始訓練規模的對照組（batch size / epoch 數不同，仍在本專案的
  CIFAR-100 子集資料上跑，不是複現論文原始 benchmark——後者需要新的 dataset loader，見上方
  「Step 3 的一個重要提醒」）

之後 SoLar、OP-W 做完 Step 4/5，應該把對應的 `run_pipeline.py` 指令一併加進這支腳本，保持
「一支腳本涵蓋所有已完成論文的實驗」。

## `fixed_` 命名慣例（Step 4 用）

- 單檔案演算法（PRODEN、CLPL、MCL、SCL-NL、OP-W）：在同一層新增 `fixed_<原檔名>.py`
  （例如 `src/fixed_mcl_losses.py`），類別名稱前綴 `Fixed`（例如 `FixedMCLLog`）。
- 子套件演算法（PiCO、SoLar、ComCo）：在同一子套件目錄下新增 `fixed_<原檔名>.py`
  （例如 `src/comco/fixed_utils_loss.py`），不要新建整個平行子套件。例外：PiCO 這次的落差在
  訓練迴圈（`src/engine.py::train_pico_epoch`）而不是 loss/model 檔案，因此放在
  `src/fixed_pico_engine.py`（跟 `src/pico/` 平行，不是子套件內），內含
  `train_pico_epoch_fixed`——如果未來又遇到「落差在訓練迴圈而非 loss class」的情況，可以參考這個
  先例，不用勉強塞進 loss 檔案的 `fixed_` 版本裡。
- **每個 `fixed_*` 檔案開頭要有一段註解，寫清楚**：改了哪裡、對應到該論文 Step 2 文件裡的哪一項
  差異、為什麼這樣改才符合原論文。不要沒有依據地「順手改」——所有修改都必須能回溯到 Step 2 的
  比對結論。
- `fixed_*` 完成後，要註冊進 `src/pipeline/algorithms/__init__.py`（新增一個 `AlgorithmSpec`）、
  `src/pipeline/algorithms/hparams.py`（`ALGO_HPARAMS` 加一行，通常跟原版用同一組超參數）、
  以及對應的 `run_*_fixed` 函式（`src/pipeline/algorithms/runners.py`），Algorithm ID 用
  `<原ID>-Fixed`。已完成的三個範例可以直接參考：`MCL-LOG-Fixed`、`PiCO-Fixed`、`ComCo-Fixed`。
- **不是每篇論文都需要 fixed 版本**——這次 6 篇裡有 3 篇（PRODEN 的 `ProdenLoss`、CLPL、SCL-NL
  的單CL公式）逐項比對後完全忠實，沒有產出 fixed 版本，這是正確且預期的結果，不要為了「湊數」
  硬找一個要修的地方。

## 已知程式碼缺陷

1. **`src/cour_loss.py` 是孤兒檔案**（**尚未處理**）：定義 `UniformCandidateCrossEntropyLoss`
   （別名 `CourLoss`），均等平均候選標籤的 CE loss，但完全沒有被 `model_setup.py`、
   `training_pipelines.py`、`pipeline/algorithms/runners.py` 任何一處匯入。它自己的 docstring
   也承認「NOTE: This is NOT the CLPL loss from Cour, Sapp & Taskar (JMLR 2011)...」。真正被使用、
   對應論文的是 `src/clpl_loss.py` 的 `CLPLSquaredHingeLoss`（已於 2026-08-14 驗證忠實）。
   **仍待決定**：`cour_loss.py` 到底要刪除、保留當歷史備查、還是重新命名——不要直接刪，先確認是否
   有舊實驗結果依賴它。
2. **`setup_comco` 重複定義**（**尚未處理**）：`src/model_setup.py` 裡定義了兩次（約第 52 行、
   第 96 行），內容幾乎相同，後者會覆蓋前者。`src/training_pipelines.py` 的 `run_comco_training`
   同樣重複定義。屬於 copy-paste 遺留的死碼，只影響舊版 pipeline，不影響 Step 5 用的新版 pipeline。
3. **`MCL_MAE`/`MCL_EXP` 未註冊進新版 pipeline**（**部分處理**）：兩者的 scaling bug 已經跟
   `MCL_LOG` 一起修正，`FixedMCLMae`/`FixedMCLExp` 已經寫在 `src/fixed_mcl_losses.py` 裡，但
   **原版跟 fixed 版都還沒註冊進 `pipeline/algorithms/__init__.py`**（不在這次 8 篇論文範圍內）。
4. **`moco_queue % batch_size` 的 assert 隱患**（**仍要留意**）：`src/pico/model.py:34`、
   `src/comco/model.py:35` 都有 `assert args['moco_queue'] % batch_size == 0`。目前預設
   `batch_size=512`、`moco_queue=8192`（能整除）所以沒事，但若要用非預設 batch size 掃
   PiCO / PiCO-Fixed / PiCO-MCL / PiCO-SC / ComCo / ComCo-Fixed，要先確認整除，否則會直接 crash。
5. ~~`scripts/run_pipeline.py` docstring 與現實不符~~ **已解決**：`scripts/legacy/` 已建立，
   57 支舊 script 都搬過去了（2026-08-14 之前完成，詳見「Pipeline 狀態」一節）。

## Step 3 的一個重要提醒：原論文 benchmark 不一定是 CIFAR

目前 pipeline（新舊皆然）只支援 CIFAR-10 / CIFAR-20 / CIFAR-100 子集 / CLCIFAR-10 / CLCIFAR-20。
但論文原始實驗不一定用 CIFAR。**6 篇已完成查證的論文，結論如下**：

- **CLPL (Cour 2011)**：原論文完全沒用 CIFAR，用的是 UCI 表格資料（dermatology/ecoli/abalone）、
  LFW 人臉、以及真實電視劇（*Lost*/*C.S.I.*）弱標註資料。用 CIFAR 近似是「借用同一套損失函數在
  不同資料集上驗證」，不是複現原論文實驗。
- **PRODEN**：用 MNIST/Fashion-MNIST/Kuzushiji-MNIST/CIFAR-10 + UCI + 5 個真實 PLL 資料集，
  **沒有 CIFAR-20/CLCIFAR**。CIFAR-10 用 12 層 ConvNet／32 層 ResNet（本專案是 ResNet-18）。
- **PiCO**：CIFAR-10/CIFAR-100/CUB-200/CIFAR-100-H，**沒有本專案的 CIFAR-20/CLCIFAR**。
- **MCL-LOG**：MNIST/Kuzushiji-MNIST/Fashion-MNIST/20Newsgroups/CIFAR-10 + UCI，**沒有 CIFAR-20**。
- **SCL-NL**：MNIST/Kuzushiji-MNIST/Fashion-MNIST/CIFAR-10，**沒有 CIFAR-20**。
- **ComCo**：MNIST/Fashion-MNIST/KMNIST/CIFAR-10/CIFAR-100/CUB-200/SUN-397，共 6 資料集 5 種
  互補標籤生成協定，遠比本專案評測範圍廣，且生成方式（unbiased/biased/MCLL）跟本專案的
  PL-補集生成方式都不一樣。

**共同結論：本專案在 CIFAR-100 子集上跑這些方法，都不是「複現論文結果」，是「用同一套已驗證忠實的
損失函數在架構上做驗證」。** 若要真正複現任何一篇論文的原始 benchmark，需要新增對應的 dataset
loader（UCI/人臉/語音、或補齊 CUB-200/SUN-397/20Newsgroups 等），這超出「用現有主 pipeline 生成
config」的範圍，不要自行假設要不要新增，留給使用者決定。

**SoLar、OP-W 尚未查證**，需要等 PDF。

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
