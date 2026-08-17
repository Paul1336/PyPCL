# Server 操作指令速查表

這份文件整理了在多台 GPU server(`cl10`、`cl777` 等)上執行/監控/收集 PyPCL 實驗結果時，
實際用過的指令，依用途分類、附說明。所有指令預設在 repo 根目錄(`~/PyPCL`)下執行，除非另外標註「本機」。

---

## 0. 開始前：確認自己在哪台機器、環境對不對

不同 server 是**完全獨立的機器**，各自有自己的硬碟/家目錄，檔案不互通(即使路徑看起來一樣，例如
兩邊都是 `~/PyPCL`，那也是各自獨立 `git clone` 出來的兩份資料)。GPU 硬體也可能不同，下指令前養成
先確認的習慣：

```bash
hostname                 # 確認自己在哪台機器
df -h ~                  # 確認掛載的檔案系統（跟別台比對用）
```

**啟用正確的 conda 環境**(這個專案的 torch/torchvision 需要特定環境，`(base)` 環境通常沒裝)：

```bash
conda env list                                   # 列出所有環境，找到有裝 torch 的那個
conda activate <環境名稱>
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 確認 import 成功、看得到 GPU
```

**GPU 編號校正**：部分 server 的 `nvidia-smi` 顯示編號跟 CUDA 實際使用的編號不一致（`nvidia-smi`
用 PCI bus 順序，CUDA 預設用 `FASTEST_FIRST`），若不校正，`CUDA_VISIBLE_DEVICES=6` 可能實際指向
`nvidia-smi` 顯示的別張卡（例如已經被別人佔滿的卡），造成 OOM 或誤用到不該用的 GPU：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

`scripts/run_main_pipeline_batch.sh` 跟 `scripts/run_threshold_oracle_batch.sh` 已經把這行寫進腳本
自動 export，不用手動加；直接用 `python ... CUDA_VISIBLE_DEVICES=X python foo.py` 這種手動指令時才需要
自己先 export。

---

## 1. 三個批次派工腳本 (dispatcher)

三個腳本共用同一套「GPU-parallel 派工 + 失敗不中斷 + 全部跑過一輪後自動重試一次」的邏輯，差別只在
跑什麼、怎麼命名結果資料夾。

### 1.1 `scripts/run_main_pipeline_batch.sh` —— 主 pipeline 9 個演算法批次

跑 CLPL、PRODEN、PiCO、PiCO-Fixed、PiCO-Oracle、PiCO-MOCO、MCL-LOG-Fixed、ComCo-Fixed、SCL-NL，
C=20 固定，k 從大到小掃過 {19,15,10,5,1}，`--detail --tsne` 全開。

```bash
# 預設：GPU 0-3，每張卡 2 個 job 同時跑，200 epoch
scripts/run_main_pipeline_batch.sh

# 指定 GPU 清單、每張卡幾個 job、epoch 數
scripts/run_main_pipeline_batch.sh --gpus 0,2,3,7 --jobs_per_gpu 1 --epochs 300

# 只看排程、不真的執行（確認 job 清單/命名對不對）
scripts/run_main_pipeline_batch.sh --dry_run
```

結果：`results/new_main_c20_k<k>_<alg>/results.csv`
Detail/tsne：`results/new_main_c20_k<k>_<alg>/detail/<algorithm>/C20_k<k>/`
Log：`logs/main_pipeline_batch/<run_name>.log`，失敗清單在 `logs/main_pipeline_batch/failures.log`

### 1.2 `verify_scripts/run_verify.sh` —— 六篇論文的 paper-exact 驗證腳本

每個方法各自用論文原始的 network/超參數/資料集（不是主 pipeline 共用的 ResNet-18），跑到收斂為止。

```bash
# 預設：六個全跑，GPU 0-3
verify_scripts/run_verify.sh

# 只跑指定幾個方法（可選：clpl proden pico mcl_log comco scl_nl）
verify_scripts/run_verify.sh --methods pico comco

# 指定 GPU 清單（CLPL 是 CPU-only 的 sklearn SVM，排隊還是會佔一個 slot，但不會真的用到 GPU）
verify_scripts/run_verify.sh --gpus 0,1

# 幫某個方法加額外參數，常用來做小規模 smoke test
verify_scripts/run_verify.sh --extra-args proden "--epochs 5 --batch_size 32"

# 預覽排程，不執行
verify_scripts/run_verify.sh --dry_run
```

結果：`verify_results/<method>.csv`（含 `paper_target_accuracy` 欄位可直接比對）
Log：`logs/verify_scripts/<method>.log`，失敗清單在 `logs/verify_scripts/failures.log`

### 1.3 `scripts/run_threshold_oracle_batch.sh` —— PiCO-Oracle 精度閾值掃描

固定 C=20，對 k∈{19,15,10} × threshold∈{0,0.05,0.07,0.09,0.11,0.15,0.2,0.25,0.5,0.75,1} 做
33 組交叉實驗，只跑 `PiCO-Oracle`，`--detail` 開啟。每個 threshold 會各自產生一份
`configs/thresholdoracle/threshold_t<tag>.yaml`（複製 `config.yaml` 再覆寫
`pico.oracle_precision_threshold`），避免多個併發 job 搶著改同一份共用設定檔。

```bash
# 預設：GPU 0-3，每張卡 1 個 job，200 epoch
scripts/run_threshold_oracle_batch.sh

# 指定 GPU、每張卡幾個 job
scripts/run_threshold_oracle_batch.sh --gpus 0,2,3,7 --jobs_per_gpu 1 --epochs 200

# 預覽
scripts/run_threshold_oracle_batch.sh --dry_run
```

結果：`results/thresholdoracle_c20_k<k>_t<threshold>/results.csv`
（threshold 的小數點在檔名/run_name 裡會被換成 `p`，例如 `0.05` → `t0p05`）
Detail：`results/thresholdoracle_c20_k<k>_t<threshold>/detail/PiCO-Oracle/C20_k<k>/`
Log：`logs/thresholdoracle_batch/<run_name>.log`，失敗清單在 `logs/thresholdoracle_batch/failures.log`

**重跑注意**：三個腳本都是「每組實驗有自己獨立的 `run_name`/`results/` 資料夾」，所以重新執行同一個
指令時，已經成功、寫出 `results.csv` 的組合會被自動偵測並跳過（印 `[skip]`），只有沒完成/失敗的會
真的重新訓練 —— 空間清完、環境修好之後直接重跑同一行指令就好，不用自己篩選要重跑哪些。

---

## 2. 不透過 `.sh`，直接執行單一驗證腳本

想要即時看到輸出（不被導向 log 檔），或只想跑一個方法時：

```bash
CUDA_VISIBLE_DEVICES=0 python verify_scripts/pico_verify.py
CUDA_VISIBLE_DEVICES=1 python verify_scripts/comco_verify.py

# 可覆寫的常見參數
python verify_scripts/proden_verify.py --epochs 5 --batch_size 32 --seed 1
```

主 pipeline 單一 cell 也可以這樣直接跑（不透過 batch 腳本）：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_pipeline.py run \
    --run_name my_test --algorithms PiCO-Fixed --c_values 20 --only_k 19 \
    --epochs 200 --detail --tsne
```

---

## 3. tmux：讓長時間任務在 SSH 斷線後繼續跑，還能隨時接回去看畫面

```bash
tmux new -s <session名稱>          # 開新 session，進去後直接在前景跑指令（不加 nohup、不加 &）
# Ctrl+b 放開後按 d                 # 離開但不中斷，process 繼續在背景跑

tmux ls                            # 列出所有 session
tmux attach -t <session名稱>       # 重新接回去看即時畫面
tmux kill-session -t <session名稱> # 整個 session 連同裡面的 process 一起砍掉
```

跟 `nohup ... &` 的差別：`nohup` 只是讓 process 不被 SSH 斷線的 hangup 訊號終止，你只能事後看 log
檔；tmux 是開一個獨立的虛擬終端機，SSH 斷線後畫面內容都還保留著，重新連上可以直接看到即時輸出。

---

## 4. 進度查詢

### 4.1 有哪些 process 在跑

```bash
# 看自己所有 process（不限定專案）
ps -u Paul -o pid,%cpu,%mem,etime,cmd | grep -v grep

# 只看 python
ps -u Paul -o pid,%cpu,%mem,etime,cmd | grep python | grep -v grep

# 只看這個專案相關的（run_pipeline.py / verify_scripts）
ps -u Paul -o pid,etime,cmd | grep -E "run_pipeline\.py|verify_scripts/.*\.py" | grep -v grep

# 對照 GPU 使用狀況，確認哪些 process 真的在吃 GPU
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv

# 完整看 GPU + process 對照表
nvidia-smi
```

`etime` 那欄可以看 process 已經跑多久，快速判斷是不是卡住。

### 4.2 個別 job 的訓練進度（log 檔最後幾行）

```bash
# 單一 log 持續盯著看（Ctrl+C 離開，不影響背景 process）
tail -f logs/verify_scripts/pico.log

# 一次看某個批次全部 job 各自的最後兩行
for f in logs/main_pipeline_batch/new_main_c20_*.log; do
  echo "== $(basename "$f") =="
  tail -n 2 "$f"
  echo
done

# 只看「真的還在跑」的 process 對應的 log（比翻全部 log 準，會自動抓出 run_name）
for pid in $(ps aux | grep run_pipeline.py | grep -v grep | awk '{print $2}'); do
  run_name=$(ps -p "$pid" -o cmd= | grep -oP '(?<=--run_name )\S+')
  echo "== $run_name (pid $pid) =="
  tail -n 2 "logs/main_pipeline_batch/${run_name}.log"
  echo
done
```

### 4.3 有沒有失敗

```bash
cat logs/main_pipeline_batch/failures.log
cat logs/verify_scripts/failures.log
cat logs/thresholdoracle_batch/failures.log
```

空的代表目前這一輪(含自動重試)全部成功，沒有需要人工介入的失敗。

---

## 5. 結果彙整（CSV）

CSV 欄位裡如果有引號包住、內含逗號的欄位（例如 `notes`），**不要用 `awk -F','` 手動切**，會被裡面的
逗號弄亂，一律用 Python 的 `csv.DictReader`。

### 5.1 主 pipeline 批次（`results/new_main_c20_k*_*/results.csv`）

```bash
python3 -c "
import csv, glob

rows = []
for path in glob.glob('results/new_main_c20_k*_*/results.csv'):
    with open(path, newline='') as f:
        rows.extend(csv.DictReader(f))

rows.sort(key=lambda r: (r['algorithm'], -int(r['k'])))
print(f'{\"algorithm\":<15}{\"k\":>4}{\"final_accuracy\":>16}')
for r in rows:
    print(f'{r[\"algorithm\"]:<15}{r[\"k\"]:>4}{float(r[\"final_accuracy\"]):>16.2f}')
print(f'\ntotal rows: {len(rows)} / 45 expected')
"
```

### 5.2 Threshold-Oracle 批次（`results/thresholdoracle_c20_k*_t*/results.csv`）

```bash
python3 -c "
import csv, glob, re

rows = []
for path in sorted(glob.glob('results/thresholdoracle_c20_k*_t*/results.csv')):
    m = re.search(r'k(\d+)_t([\dp]+)', path)
    k, thr_tag = int(m.group(1)), m.group(2).replace('p', '.')
    with open(path, newline='') as f:
        r = next(csv.DictReader(f))
        rows.append((k, float(thr_tag), float(r['final_accuracy'])))

rows.sort(key=lambda x: (-x[0], x[1]))
print(f'{\"k\":>4}{\"threshold\":>12}{\"final_accuracy\":>16}')
for k, thr, acc in rows:
    print(f'{k:>4}{thr:>12.2f}{acc:>16.2f}')
print(f'\ndone: {len(rows)} / 33')
"
```

### 5.3 有哪些 (演算法, k) 組合是「已完成 / 執行中 / 排隊中」

```bash
python3 -c "
import csv, glob, subprocess

ALGS = ['CLPL','PRODEN','PiCO','PiCO-Fixed','PiCO-Oracle','PiCO-MOCO','MCL-LOG-Fixed','ComCo-Fixed','SCL-NL']
KS = [19,15,10,5,1]

def run_name(alg, k):
    return f'new_main_c20_k{k}_' + alg.lower().replace('-', '_')

done = set()
for path in glob.glob('results/new_main_c20_k*_*/results.csv'):
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            done.add((r['algorithm'], int(r['k'])))

ps = subprocess.run(['ps','-eo','cmd'], capture_output=True, text=True).stdout
running = set()
for line in ps.splitlines():
    if '--run_name' in line and 'new_main_c20' in line:
        name = line.split('--run_name')[1].split()[0]
        running.add(name)

for k in KS:
    for alg in ALGS:
        rn = run_name(alg, k)
        if (alg, k) in done:
            status = 'DONE'
        elif rn in running:
            status = 'RUNNING'
        else:
            status = 'queued'
        print(f'{status:<8}{alg:<15}k={k}')
"
```

### 5.4 六篇論文驗證結果（`verify_results/*.csv`）

```bash
# 整份 CSV 直接看
for f in verify_results/*.csv; do
  echo "== $f =="
  cat "$f"
  echo
done

# 只看關鍵欄位（accuracy vs. 論文目標值）
python3 -c "
import csv, glob

for path in sorted(glob.glob('verify_results/*.csv')):
    print(f'== {path} ==')
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            acc = r.get('final_accuracy', r.get('mean_accuracy', '?'))
            target = r.get('paper_target_accuracy', '?')
            print(f'  acc={acc}  target={target}')
    print()
"
```

---

## 6. 畫圖

### 6.1 多方法 accuracy-vs-k 比較圖

```bash
mkdir -p plots/<日期或代號>

python scripts/run_pipeline.py plot \
    --runs $(for k in 19 15 10 5 1; do
        for alg in clpl proden pico_fixed mcl_log_fixed comco_fixed scl_nl; do
            echo "new_main_c20_k${k}_${alg}"
        done
    done) \
    --algorithms CLPL PRODEN PiCO-Fixed MCL-LOG-Fixed ComCo-Fixed SCL-NL \
    --c_values 20 \
    --group_by all-in-one \
    --out plots/<日期或代號>/six_method_comparison.png
```

`--group_by paradigm`（預設值）會用內建的 PLL/CLL 演算法白名單過濾，**Fixed 系列的名字不在那份
白名單裡，會被整個濾掉**——只要畫面裡有任何 `*-Fixed` / `PiCO-MOCO` 之類的名字，一定要用
`--group_by all-in-one` 或 `per-algorithm`，不能用預設的 `paradigm`。

還沒完成的 (algorithm, k) 組合會被自動跳過，不用等全部跑完才能畫。

### 6.2 單一演算法某個 (C,k) 的 per-class accuracy / CE loss 熱力圖（需要 `--detail`）

```bash
python scripts/run_pipeline.py detail-plot \
    --run new_main_c20_k19_pico_fixed \
    --alg_l PiCO-Fixed \
    --C 20 --k 19 \
    --out plots/<日期或代號>/detail/pico_fixed_c20_k19_heatmap.png

# 只要 accuracy、不要 CE loss 那一列
python scripts/run_pipeline.py detail-plot ... --acc_only

# 左右並排比較兩個演算法（要求兩個演算法的 detail/ 都在同一個 --run 底下，
# 這批用「每個演算法各自獨立 run_name」的命名方式辦不到，除非手動合併資料夾）
python scripts/run_pipeline.py detail-plot \
    --run <run_name> --alg_l PRODEN --alg_r PiCO-Fixed --C 20 --k 19 --out xxx.png
```

只有 PRODEN、PiCO 系列（PiCO/PiCO-Fixed/PiCO-Oracle/PiCO-MOCO）、ComCo-Fixed 有寫
`per_class_loss.csv`；CLPL/MCL-LOG-Fixed/SCL-NL 沒有中途 checkpoint 可評估，這個指令對它們沒有資料。

### 6.3 PiCO / PiCO-Fixed 的 contrastive pair-selection precision 折線圖（需要 `--detail`）

```bash
python scripts/run_pipeline.py detail-plot-pico \
    --run new_main_c20_k19_pico_fixed \
    --alg PiCO-Fixed \
    --C 20 --k 19 \
    --out plots/<日期或代號>/detail/pico_fixed_c20_k19_selection_precision.png
```

只有 `PiCO`、`PiCO-Fixed` 有寫 `pico_selection_stats.csv`（`PiCO-Oracle`/`PiCO-MOCO` 沒有接這個 log）。

### 6.4 PiCO-Oracle 的閾值修正前後 precision 對比圖（需要 `--detail`）

```bash
python scripts/run_pipeline.py detail-plot-pico-oracle \
    --run thresholdoracle_c20_k19_t0p5 \
    --C 20 --k 19 \
    --out plots/<日期或代號>/detail/pico_oracle_c20_k19_t0p5_correction.png
```

圖上三條線：修正前 precision（自然值，等同 PiCO-Fixed）、修正後 precision（實際拿去訓練用的）、
negative-pair precision。

### 6.5 t-SNE 圖

不用額外指令，`--tsne` 開啟時訓練過程中就自動存好了：

```bash
ls results/<run_name>/detail/<algorithm>/C<C>_k<k>/tsne/
# ep0010.png, ep0020.png, ... 每個 checkpoint 一張
```

### 6.6 批次幫「目前所有已完成的組合」自動產圖（迴圈版）

```bash
mkdir -p plots/<日期或代號>/detail

for run_dir in results/new_main_c20_k*_*/; do
    run_name=$(basename "$run_dir")
    csv="${run_dir}results.csv"
    [[ -f "$csv" ]] || continue

    read -r algorithm C k <<< "$(python3 -c "
import csv
with open('$csv', newline='') as f:
    r = next(csv.DictReader(f))
    print(r['algorithm'], r['total_classes'], r['k'])
")"

    heatmap_csv="${run_dir}detail/${algorithm}/C${C}_k${k}/per_class_loss.csv"
    if [[ -f "$heatmap_csv" ]]; then
        python scripts/run_pipeline.py detail-plot \
            --run "$run_name" --alg_l "$algorithm" --C "$C" --k "$k" \
            --out "plots/<日期或代號>/detail/${run_name}_heatmap.png"
    fi

    sel_csv="${run_dir}detail/${algorithm}/C${C}_k${k}/pico_selection_stats.csv"
    if [[ -f "$sel_csv" ]]; then
        python scripts/run_pipeline.py detail-plot-pico \
            --run "$run_name" --alg "$algorithm" --C "$C" --k "$k" \
            --out "plots/<日期或代號>/detail/${run_name}_selection_precision.png"
    fi
done
```

---

## 7. 把結果/圖同步回本機（Windows）

### 7.1 PowerShell（沒裝 rsync 時用這個：ssh + tar + scp）

```powershell
# 1. server 上把要的檔案打包成一個 tar.gz（-print0/--null 處理檔名含特殊字元）
ssh Paul@<host> "cd ~/PyPCL && find plots results -name '*.png' -print0 | tar --null -czf /tmp/pypcl_pngs.tar.gz -T -"

# 2. 抓回本機
scp Paul@<host>:/tmp/pypcl_pngs.tar.gz "c:\Users\User\Desktop\現在式\PyPCL\pypcl_pngs.tar.gz"

# 3. 本機解開（照 server 上的相對路徑還原到對應資料夾）
tar -xzf "c:\Users\User\Desktop\現在式\PyPCL\pypcl_pngs.tar.gz" -C "c:\Users\User\Desktop\現在式\PyPCL"

# 4. 清暫存
Remove-Item "c:\Users\User\Desktop\現在式\PyPCL\pypcl_pngs.tar.gz"
ssh Paul@<host> "rm /tmp/pypcl_pngs.tar.gz"
```

`tar`/`ssh`/`scp` 在 Windows 10 之後都內建，不用額外安裝。避免直接用 PowerShell 的 `|` 管線接
`ssh` 跟 `tar` 兩個 native 執行檔（PowerShell 5.1 對二進位資料的管線轉發不保證乾淨），所以拆成
「先在遠端打包成一個檔案 → 落地 → 本機解壓」三步，比較穩。

### 7.2 只抓 CSV、不要整個 `results/`（`--detail` 產物可能很大）

```bash
rsync -avz --progress --include='*/' --include='results.csv' --exclude='*' \
    cl10:~/PyPCL/results/ "/c/Users/User/Desktop/現在式/PyPCL/results/"
```
（Git Bash 有裝 rsync 才能用；PowerShell 沒有，要用 7.1 的 tar 版本）

### 7.3 整個資料夾同步（Git Bash + rsync，有裝的話）

```bash
rsync -avz --progress cl10:~/PyPCL/results/ "/c/Users/User/Desktop/現在式/PyPCL/results/"
rsync -avz --progress cl10:~/PyPCL/plots/   "/c/Users/User/Desktop/現在式/PyPCL/plots/"
```

---

## 8. Disk quota / 磁碟空間清理

`OSError: [Errno 122] Disk quota exceeded` 是**個人 NFS 配額用完**，不是硬碟真的滿了，每台 server
配額是分開算的，這台清完不代表另一台也沒事。

```bash
# 看配額用量（部分機器沒裝這個指令）
quota -s

# 找出家目錄裡誰占最多空間
du -sh ~/* 2>/dev/null | sort -rh | head -20

# 常見大宗：pip / huggingface 下載快取
du -sh ~/.cache/* 2>/dev/null | sort -rh | head -20

# pip 快取：100% 安全，清了只是下次 install 要重新下載
pip cache purge

# huggingface 快取：要先確認有沒有其他專案還在用，不確定就先問
du -sh ~/.cache/huggingface 2>/dev/null
rm -rf ~/.cache/huggingface

# repo 內部的候選清理對象（legacy/舊 log，通常比較安全，動手前先確認不需要了）
du -sh ~/PyPCL/results_legacy 2>/dev/null
du -sh ~/PyPCL/logs/* 2>/dev/null | sort -rh | head -10
```

---

## 9. 終止 process

```bash
# 溫和終止（給 process 機會自己清理再結束）
kill <PID>

# 過幾秒確認是否真的結束
ps -p <PID>

# 還在的話強制砍
kill -9 <PID>

# 針對特定專案的 process 批次砍——務必把 pattern 收窄到只匹配你要的東西，
# 不然容易誤殺別的還在跑的 job（曾經發生過 pattern 太寬，把不相干的 batch 一起殺掉的事故）
pkill -f "run_pipeline.py run --run_name new_main_c20"     # 只殺這個前綴的 run_name
pkill -f "run_main_pipeline_batch.sh"                       # 只殺 dispatcher 本身
```

**`pkill`/`kill` 只作用在你目前登入的這台機器**，不會影響到別台 server 上的 process，即使你在兩邊
跑了同名的腳本也一樣——process 是機器本地的，沒有任何機制可以跨機器誤殺。

---

## 10. 已知的坑（踩過一次，寫下來避免重踩）

- **`wait` 不能寫在 `$(...)` command substitution 子殼層裡**：子殼層不是背景 job 在 OS 層級的
  parent，`wait` 在那裡抓不到正確的結束碼。三個 dispatcher 腳本裡的 `_find_free_slot`/`_reap_slot`
  都刻意寫成在主 shell 裡直接跑，不要「優化」成看起來更簡潔的子殼層寫法。
- **CSV 裡有引號包住、含逗號的欄位時不要用 `awk -F','` 手動切**，一律用 Python `csv.DictReader`。
- **`pkill -f` 的 pattern 一定要收窄到唯一匹配你要的東西**，太寬的 pattern（例如只寫
  `verify_scripts/.*\.py`）可能連正在跑的其他批次一起殺掉。
- **Windows 上 `chmod +x` 本機做了沒用**（`core.fileMode` 通常關閉，不會被 git 記錄），新增的 `.sh`
  腳本要用 `git update-index --chmod=+x <file>` 明確標記可執行位元，否則 clone 到 Linux 上執行權限
  預設是關的。
- **多個 GPU-worker 併發跑同一批次時，每個 (演算法,k) 或 (k,threshold) 組合一定要有自己獨立的
  `run_name`**，不要共用同一個 shard 檔案，不然併發寫入會互相覆蓋/衝突。
- **`run_pipeline.py` 的 resume/dedup 是用 `(dataset, C, k, algorithm)` 當 key，沒有 per-epoch
  checkpoint**——中途砍掉 process 會遺失該 cell 的全部進度（不是只退回上一個 checkpoint），砍之前
  想清楚。
