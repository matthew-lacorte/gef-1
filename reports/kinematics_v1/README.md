# THIS IS WHAT GPT THINKS

reports/
└─ 2025_gef_paper1/
   ├─ README.md                   # how to reproduce; runtime; expected outputs
   ├─ CITATION.cff                # citation metadata
   ├─ LICENSE                     # license for artifacts (if different)
   ├─ CODE_REF                    # git SHA or tag (if pinned approach)
   ├─ environment/
   │  ├─ environment.yml          # conda spec
   │  └─ conda-lock.yml           # optional: locked explicit specs
   ├─ dvc.yaml                    # stages to regenerate figures/tables
   ├─ dvc.lock                    # pinned inputs/commands hashes
   ├─ params.yaml                 # figure switches, seeds, resolution
   ├─ configs/                    # paper-specific configs importing base defaults
   │  └─ stability_map.yml
   ├─ scripts/                    # tiny wrappers (or vendored copies)
   │  └─ make_fig2a.py
   ├─ notebooks/                  # optional: report notebooks (outputs → figures/)
   │  └─ fig2_supplement.ipynb
   ├─ data/                       # small text metadata only; big stuff via DVC
   ├─ figures/                    # DVC-tracked “golden” images
   │  ├─ fig2a_kflow_profile.png
   │  └─ fig2b_stability_map.png
   ├─ tables/                     # DVC-tracked CSV/JSON/TeX
   │  └─ table_s1_parameters.csv
   └─ checksums/                  # optional: SHA256 of each “golden” artifact
      └─ figures.sha256

## Key points

- DVC is the backbone: figures/tables are outs:; data dependencies (raw/processed) are deps:. dvc repro rebuilds everything deterministically.
- Environment is pinned: provide environment.yml + (ideally) a conda-lock or a Dockerfile. Locking wins reproducibility wars.
- No ad-hoc writes: notebooks and wrappers should put all artifacts only into figures/ and tables/.
- Small wrappers call your canonical scripts with paper configs; they should also:
    - check that git rev-parse HEAD equals CODE_REF (fail fast if not),
    - write a tiny provenance.json (commit SHA, command, start/end time) next to each artifact.