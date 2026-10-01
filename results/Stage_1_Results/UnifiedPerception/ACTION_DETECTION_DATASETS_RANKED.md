# Public Datasets for Spatio-Temporal Action Detection & Localization

**Task:** figure out *what* action each person is doing and *where + when* it happens
in surveillance video — i.e. per-frame **(bounding box, action)** tracked over time.
This is **spatio-temporal action detection (AD-ST)**.

The single table below lists the most useful public datasets, **ordered by how
directly they support this task** (top = best fit). It merges both survey rounds.

**Type:** `AD-ST` = boxes in space **and** time (best fit) · `AD-T` = time only
(when, not where) · `AR` = trimmed-clip classification (pretraining / vocabulary).
**Use:** ✅ commercial-allowed · 🔒 research-only · ❓ unclear · ⚠️ labels are
permissive but the **source video is copyrighted** — clear video rights before any
commercial use. (Always check the licence.)

| # | Dataset | Type | Why it fits (key content) | Use | Link |
|---|---|---|---|---|---|
| 1 | **AVA / AVA-Kinetics** | AD-ST | 80 atomic actions with per-frame person boxes — the standard STAD training set (`open (car door)`, `carry/hold`, `crouch`, `fight/hit`) | ⚠️ annotations CC BY 4.0; **movie video © (YouTube-sourced)** | https://research.google.com/ava/ |
| 2 | **MEVA / ActEV-SDL** | AD-ST | static-camera person↔vehicle events (`opens vehicle door`, `abandons package`, `steals object`) | ✅ CC BY-4.0 | https://mevadata.org |
| 3 | **CHAD** | AD-ST +pose+ReID | 4-camera parking-lot (`Fighting`, `Theft`, `Chasing`, `Pushing`) — closest scene | ✅ Apache-2.0 | https://github.com/TeCSAR-UNCC/CHAD |
| 4 | **VIRAT (+DIVA)** | AD-ST | surveillance person-vehicle (`load/unload`, `open trunk`, `get in/out`) | ✅ commercial allowed (usage agreement; no redistribution) | https://viratdata.org/ |
| 5 | **ROAD** | AD-ST tubes | automotive agent+action+location tubes (`Push object`, `Move towards`) | 🔒 CC BY-NC-SA | https://github.com/gurkirt/road-dataset |
| 6 | **CAVIAR** | AD-ST | bbox + scenario (`loiter`, `LeftBag`/`LeftBox`, `Fight`) | ✅ CC BY-SA | http://homepages.inf.ed.ac.uk/rbf/CAVIAR/ |
| 7 | **UCF101-24** | AD-ST | classic 24-class spatio-temporal tube benchmark | ❓ unclear | https://github.com/gurkirt/corrected-UCF101-Annots |
| 8 | **J-HMDB-21** | AD-ST +pose | box+pose tubes (`throw`, `push`, `sit`, `walk`) | ❓ unclear | http://jhmdb.is.tue.mpg.de/ |
| 9 | **MultiSports** | AD-ST | densest per-frame multi-person STAD benchmark | 🔒 CC BY-NC 4.0 (YouTube sports video) | https://deeperaction.github.io/datasets/multisports.html |
| 10 | **Street Scene (MERL)** | AD-ST +track | surveillance anomaly, bbox+track (`Loitering`, `Illegal parking`, `opening trunk`) | ❓ unclear | http://www.merl.com/demos/video-anomaly-detection |
| 11 | **HiEve** | AD-ST +pose | crowd STAD with pose+tracking (`fighting`, `gathering`, `fall-over`, `queuing`) | 🔒 research | http://humaninevents.org/ |
| 12 | **MOMA-LRG** | AD-ST | multi-actor STAD + actor-object relation graphs | 🔒 research | https://github.com/StanfordVL/moma |
| 13 | **Okutama-Action** | AD-ST (aerial) | drone STAD (`Carrying`, `Push/Pull`, `Handshaking`, `Sitting`) | 🔒 CC BY-NC-SA | http://okutama-action.org |
| 14 | **JRDB-Act** | AD-ST +groups | atomic actions + social-group IDs (campus/robot view) → gatherings | 🔒 non-commercial | https://jrdb.erc.monash.edu |
| 15 | **ROAD-Waymo (ROAD++)** | AD-ST | larger automotive STAD, diverse weather (same taxonomy as ROAD) | 🔒 research | https://arxiv.org/abs/2411.01683 |
| 16 | **SVAG-Bench** | AD-ST (grounding) | multi-instance spatio-temporal action grounding by language query | 🔒 research — video reused from MOT17/MOT20/OVIS (non-commercial) | https://arxiv.org/abs/2510.13016 |
| 17 | **PKU-MMD** | AD-T (skeleton) | continuous, **untrimmed** skeleton action detection (`pushing`, `punching`, `handing over object`) | 🔒 research | https://struct002.github.io/PKUMMD/ |
| 18 | **NWPU Campus** | AD-T | temporal campus anomaly (`Loitering`, `Fighting`, `U-turn`, `Stealing`, `Snatching`) | 🔒 research | https://campusvad.github.io/ |
| 19 | **UCF-Crime** | AD-T (weak) | video-level temporal anomaly (`Vandalism`, `Burglary`, `Arson`, `Fighting`) | ❓ unclear | https://www.crcv.ucf.edu/projects/real-world/ |
| 20 | **ShanghaiTech / CUHK Avenue / IITB-Corridor** | AD-T | frame-level video-anomaly (`loitering`, `throwing`, `fighting`, `unattended baggage`) | 🔒/❓ research | https://svip-lab.github.io/dataset/campus_dataset.html |
| 21 | **MSAD** | AD-T | multi-scenario anomaly, day/night, outdoor/parking (`Assault`, `Vandalism`, `Fire`) | ❓ unclear | https://msad-dataset.github.io/ |
| 22 | **PETS2006 / PETS2007** | AD-T | abandoned-object + loitering (temporal) | 🔒 research | https://www.cvg.reading.ac.uk/PETS2007/data.html |
| 23 | **Kinetics-700-2020** | AR | pretraining backbone; rich vocabulary (`breaking glass`, `spray painting`, `lighting fire`, `sword fighting`) | ⚠️ annotations CC BY 4.0; **YouTube video ©** | https://github.com/cvdfoundation/kinetics-dataset |
| 24 | **NTU RGB+D 120** | AR (skeleton) | weapon/assault classes (`wield knife`, `shoot gun`, `hit with something`) | 🔒 research | https://rose1.ntu.edu.sg/dataset/actionRecognition/ |
| 25 | **Something-Something V2** | AR (hand-obj) | fine hand-object verbs (`Putting`, `Pulling`, `Attaching`, `Poking`) | 🔒 research | https://www.qualcomm.com/developer/software/something-something-v-2-dataset |
| 26 | **Moments in Time / HMDB51** | AR | broad verb vocabulary (`breaking`, `spraying`, `stealing`; `sword fight`, `shoot gun`, `punch`) | 🔒/❓ | http://moments.csail.mit.edu/ |

*Rows 1–16 directly do spatio-temporal detection (space + time); 17–22 localize in
time (pair with a person tracker for boxes); 23–26 are trimmed-clip sets — use them
only to pretrain the action head / borrow class names.*

> **⚠️ Licence caveat (important for commercial use):** AVA (#1) and Kinetics (#23)
> release their **annotations** under CC BY 4.0, but the **underlying videos are
> movies / YouTube clips owned by their original rights holders** — they are *not*
> free for commercial use. They're great for research and pretraining; if you ship a
> product, clear the raw-video rights separately or retrain the final model on
> genuinely open video (e.g. **MEVA** CC BY-4.0, **CHAD** Apache-2.0, **CAVIAR** CC BY-SA,
> **VIRAT** commercial-allowed) — these ship the video itself under a usable licence.
