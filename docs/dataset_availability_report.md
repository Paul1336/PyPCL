# Dataset Availability Report

Generated 2026-08-14T15:40:30 by `scripts/probe_dataset_availability.py`. Cheap HTTP HEAD/GET probes only (status code / content-length / content-type) -- no file parsing, no schema validation. A verdict of AVAILABLE means the URL responds, not that the file has been downloaded and parsed successfully; DEAD means the URL did not resolve or returned an HTTP error at probe time.

| Verdict | Name | URL | Status | Content-Length | Content-Type | Note |
|---|---|---|---|---|---|---|
| AVAILABLE (class exists) | torchvision.datasets.MNIST | (built into torchvision) | n/a | n/a | n/a | Phase 1 -- built-in loader class |
| AVAILABLE (class exists) | torchvision.datasets.FashionMNIST | (built into torchvision) | n/a | n/a | n/a | Phase 1 -- built-in loader class |
| AVAILABLE (class exists) | torchvision.datasets.KMNIST | (built into torchvision) | n/a | n/a | n/a | Phase 1 -- built-in loader class |
| AVAILABLE (class exists) | torchvision.datasets.SUN397 | (built into torchvision) | n/a | n/a | n/a | Phase 1 -- built-in loader class |
| DEAD (HTTP 404) | CUB-200-2011 (Caltech official, old path -- known dead) | https://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz | 404 | 1270 | text/html; charset=utf-8 | Phase 3 -- superseded, kept here to document the dead old URL |
| AVAILABLE | CUB-200-2011 (CaltechDATA record page, current official source) | https://data.caltech.edu/records/65de6-vp158 | 200 | unknown | text/html; charset=utf-8 | Phase 3 primary source -- redirects to a presigned download URL, ~1.1GB |
| AVAILABLE | CUB-200-2011 (HuggingFace mirror) | https://huggingface.co/datasets/bentrevett/caltech-ucsd-birds-200-2011 | 200 | 707005 | text/html; charset=utf-8 | Phase 3 fallback mirror |
| AVAILABLE | PLL: Lost (.rar) | http://palm.seu.edu.cn/zhangml/files/lost.rar | 200 | 936168 | unknown | Phase 4 -- PRODEN/MCL-LOG real-world benchmark; 936KB |
| AVAILABLE | PLL: MSRCv2 (.rar) | http://palm.seu.edu.cn/zhangml/files/MSRCv2.rar | 200 | 381162 | unknown | Phase 4 -- PRODEN/MCL-LOG real-world benchmark; 381KB |
| AVAILABLE | PLL: BirdSong (.rar) | http://palm.seu.edu.cn/zhangml/files/BirdSong.rar | 200 | 1048026 | unknown | Phase 4 -- PRODEN/MCL-LOG real-world benchmark; 1.0MB |
| AVAILABLE | PLL: Soccer Player (.rar) | http://palm.seu.edu.cn/zhangml/files/Soccer%20Player.rar | 200 | 36018748 | unknown | Phase 4 -- PRODEN/MCL-LOG real-world benchmark; 36MB |
| AVAILABLE | PLL: Yahoo!News (.rar) | http://palm.seu.edu.cn/zhangml/files/Yahoo!%20News.rar | 200 | 28710594 | unknown | Phase 4 -- PRODEN/MCL-LOG real-world benchmark; 28.7MB |
| AVAILABLE | DNPL-PyTorch GitHub (source of exact filenames) | https://github.com/mikigom/DNPL-PyTorch | 206 | unknown | text/html; charset=utf-8 | Confirms the 5 PLL dataset filenames above |
| AVAILABLE | CLPL: tv_data.html (author page) | http://www.timotheecour.com/tv_data/tv_data.html | 200 | 1858 | text/html | Phase 4 -- CLPL original benchmark landing page |
| AVAILABLE | CLPL: tv_data.tar.gz (full, 117MB) | http://www.timotheecour.com/tv_data/tv_data.tar.gz | 200 | 122692358 | application/x-gzip | Phase 4 -- full CLPL dataset: raw 90x90x3 uint8 cropped face images + real candidate labels |
| AVAILABLE | CLPL: tv_data_small.tar.gz (13MB) | http://www.timotheecour.com/tv_data/tv_data_small.tar.gz | 200 | 13657683 | application/x-gzip | Phase 4 -- 1-episode starter version, used for local verification |

## Summary: 15/16 available, 1 dead

**Dead / unreachable at probe time:**
- CUB-200-2011 (Caltech official, old path -- known dead): DEAD (HTTP 404) -- superseded by the
  CaltechDATA record page and the HuggingFace mirror, both alive, so this is not a blocker.

**Bottom line: every dataset needed for the 6 already-verified papers is obtainable.** This is a
much better result than the original risk assessment assumed (Phase 4's real-world PLL/CLPL data
was flagged as the highest-risk, most-likely-to-be-dead part of the whole effort). All 5 classic
real-world PLL datasets (Lost/MSRCv2/BirdSong/Soccer Player/Yahoo!News) and the CLPL original data
(`tv_data.tar.gz`) resolved on the first real check.

## Manual follow-up notes (not covered by the automated probe)

- **`.rar` extraction tooling**: the 5 PLL datasets are `.rar` archives. Python has no pure-Python
  RAR decoder capable of extracting compressed (non-store) entries; the `rarfile` package still
  needs an external `unrar`/`bsdtar` binary. This machine has **WinRAR installed** at
  `C:\Program Files\WinRAR\UnRAR.exe` (found via manual filesystem check, not in `PATH`) — usable
  via `subprocess` with an absolute path. If Phase 4 code runs on a different machine (e.g. the
  user's Linux training server), it will need `unrar` or `bsdtar` installed there instead —
  don't assume WinRAR's presence carries over.
- **CLPL's real data format**: `tv_data.html`'s own documentation (fetched directly, see Phase 4
  notes in the main plan) confirms the `.mat` files contain **raw cropped face images**
  (`image_registered: [90x90x3 uint8]`) plus real screenplay-derived candidate label sets — NOT
  pre-extracted PCA features as the original 2011 JMLR paper's Section 6-7 (linear/kernel SVM
  experiments) used for its own experiments. This means CLPL's real data can go through the
  existing **image** loading path (CNN backbone) rather than needing the tabular/MLP path — a
  correction to the original Phase 4 assumption that this data was feature-vector-only.
- **CUB-200's presigned URL**: `data.caltech.edu`'s download link is a short-TTL (60s) presigned
  S3 URL, so a plain `HEAD` probe 403s (signature/method mismatch) even though the underlying file
  is real and downloadable via a normal `GET` in a browser or `requests.get(..., stream=True)`.
  Treat the CaltechDATA record page returning 200 as the availability signal, not the presigned
  URL's HEAD status.
