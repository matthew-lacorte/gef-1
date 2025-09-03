scripts/
├─ exp/                 # exploratory one-offs (kept tidy; can graduate later)
│  ├─ geometry/
│  │  └─ sweep_stability_window/
│  │     ├─ sweep_stability_window.py
│  │     └─ configs/
│  │        └─ default.yml
│  └─ numerics/
│     └─ profile_relaxer/
│        ├─ profile_relaxer.py
│        └─ configs/default.yml
│
├─ tasks/               # reusable “utility” commands (data prep, conversions)
│  ├─ data/
│  │  └─ ingest_fractal_seed/
│  │     ├─ ingest_fractal_seed.py
│  │     └─ configs/default.yml
│  └─ tools/
│     └─ make_api_signature/
│        ├─ make_api_signature.py
│        └─ configs/default.yml
│
├─ pipelines/           # DVC-described multi-step jobs (reproducible)
│  ├─ geometry/
│  │  └─ stability_map/
│  │     ├─ run_stability_map.py
│  │     ├─ dvc.yaml               # optional: stage wrapper here
│  │     └─ configs/default.yml
│  └─ spectra/
│     └─ run_mass_spectrum/
│        ├─ run_mass_spectrum.py
│        └─ configs/default.yml
│
├─ papers/              # exact figure/number reproduction for manuscripts
│  └─ 2025_geF_paper1/
│     ├─ fig_2a_kflow_profile/
│     │  ├─ fig_2a_kflow_profile.py
│     │  └─ configs/default.yml
│     └─ table_s1_parameter_scan/
│        ├─ table_s1_parameter_scan.py
│        └─ configs/default.yml
│
├─ demos/               # tiny, friendly, minimal deps (for talks/tutorials)
│  └─ hopfions/
│     └─ demo_generate_slice/
│        ├─ demo_generate_slice.py
│        └─ configs/default.yml
│
└─ maintenance/         # repo hygiene, CI helpers, migrations
   └─ bump_semver/
      ├─ bump_semver.py
      └─ configs/default.yml
