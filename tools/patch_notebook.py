"""
Patch notebooks/final_develop.ipynb:
  Fix the M3 training cell (id="_w8ulHWgSHlq") that references
  undefined variables OUTPUT, MODELS, CANDIDATES.
  Replace with correct variables + import statement.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NB_PATH = REPO / "notebooks" / "final_develop.ipynb"

NEW_SOURCE = [
    "# ============================================================\n",
    "# M3 FULL 10K TRAINING\n",
    "# Prerequisite: run the '10K FILTERED CACHE' cell above first.\n",
    "# Variables expected in scope:\n",
    "#   DATA_10K   — Path to data dir containing train_ground_truth.tsv\n",
    "#   CACHE_10K  — Path to filtered M2 Parquet cache\n",
    "#   UNION      — Path to candidate_pairs_10k.tsv\n",
    "#   REPO_ROOT  — Repo root (set in env/path cell)\n",
    "# ============================================================\n",
    "\n",
    "import sys\n",
    "from pathlib import Path\n",
    "\n",
    "sys.path.insert(0, str(REPO_ROOT / 'src'))\n",
    "from train import train  # noqa: E402  (M3)\n",
    "\n",
    "# Output directories (inside the 10K sandbox)\n",
    "BASE_10K   = Path('/content/m3_10k')\n",
    "OUTPUT_10K = BASE_10K / 'output'\n",
    "MODELS_10K = BASE_10K / 'models'\n",
    "OUTPUT_10K.mkdir(parents=True, exist_ok=True)\n",
    "MODELS_10K.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "# Candidate pairs produced by the 10K UNION cell above\n",
    "CANDIDATES_10K = UNION  # Path defined in the '10K UNION' cell\n",
    "\n",
    "print('=' * 60)\n",
    "print('RUNNING M3 FULL 10K TRAINING')\n",
    "print('=' * 60)\n",
    "print(f'  data_dir       : {DATA_10K}')\n",
    "print(f'  candidates_path: {CANDIDATES_10K}')\n",
    "print(f'  output_dir     : {OUTPUT_10K}')\n",
    "print(f'  models_dir     : {MODELS_10K}')\n",
    "print(f'  cache_dir      : {CACHE_10K}')\n",
    "print()\n",
    "\n",
    "train(\n",
    "    data_dir=DATA_10K,\n",
    "    candidates_path=CANDIDATES_10K,\n",
    "    output_dir=OUTPUT_10K,\n",
    "    models_dir=MODELS_10K,\n",
    "    cache_dir=CACHE_10K,\n",
    ")\n",
    "\n",
    "print('\\n' + '=' * 60)\n",
    "print('\\u2705 M3 10K TRAINING FINISHED')\n",
    "print('=' * 60)\n",
    "print('\\nArtifacts:')\n",
    "for p in [\n",
    "    MODELS_10K / 'matcher.pkl',\n",
    "    MODELS_10K / 'model_config.json',\n",
    "    OUTPUT_10K / 'baseline_training_results.json',\n",
    "    OUTPUT_10K / 'baseline_threshold_results.tsv',\n",
    "]:\n",
    "    print(('\\u2705' if p.exists() else '\\u274c'), p)\n",
]

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

patched = 0
for cell in nb["cells"]:
    if (
        cell.get("cell_type") == "code"
        and cell.get("metadata", {}).get("id") == "_w8ulHWgSHlq"
    ):
        old_src = "".join(cell["source"])
        # Guard: only patch if the broken references are present
        if "candidates_path=CANDIDATES," in old_src and "output_dir=OUTPUT," in old_src:
            cell["source"] = NEW_SOURCE
            cell["outputs"] = []
            cell["execution_count"] = None
            patched += 1
            print(f"Patched cell id=_w8ulHWgSHlq")
        else:
            print(f"Cell already patched or unexpected content — skipping")

if patched == 0:
    raise RuntimeError("Target cell not found or already patched")

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Written: {NB_PATH}")
