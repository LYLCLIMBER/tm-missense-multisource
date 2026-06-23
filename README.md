# tm-missense-multisource

Source code and necessary data files for a multi-source computational framework for transmembrane missense variant prioritization.

The study integrates mutation-level engineered features, ESM-3 wild-type residue representations, and UniProt-derived protein annotations for pathogenicity analysis on the BorodaTM transmembrane missense mutation benchmark.

## Repository Contents

- `data.xlsx`: BorodaTM benchmark table used by the experiments.
- `experiment/`: preprocessing, model training, validation, DeLong testing, SHAP analysis, and supplementary experiment scripts.
- `experiment/features/X_uniprot.npy`: processed UniProt-derived feature matrix.
- `experiment/features/esm3_79/X_esm3_79.npy`: processed ESM-3 layer-79 site-context feature matrix.
- `experiment/features/metadata.pkl` and `experiment/features/sequences.fasta`: metadata and sequences used by the feature-processing workflow.
- `experiment/features/uniprot_cache/`: cached UniProt JSON responses used to construct the released UniProt feature matrix.

Recorded result tables, generated figures, LaTeX files, and large intermediate model-cache files are not included in this source/data release. Running the scripts will create output directories locally as needed.

Large intermediate extraction caches, such as per-protein `.pt` files and per-mutation `.safetensors` ESM caches, are intentionally excluded because the released processed matrices are sufficient to rerun the tabular model experiments.

## Environment

Install the Python dependencies with:

```bash
pip install -r experiment/requirements.txt
```

The experiments were run with fixed random seeds defined in the scripts.

## Main Reproduction Commands

Run the main feature-fusion experiments:

```bash
python experiment/runner_final_no_wapssm.py
```

Run the supplementary analyses:

```bash
python experiment/runner_supplement.py
```

Run external predictor comparisons:

```bash
python experiment/runner_tool_comparison.py
```

Run DeLong tests:

```bash
python experiment/runner_delong_109d.py
```

## Data Notes

The benchmark data are derived from the BorodaTM transmembrane missense mutation dataset. Labels are mapped as pathogenic = 1 and benign/neutral = 0 in the preprocessing code.

The ESM-3 feature matrix used by the main paper is:

```text
experiment/features/esm3_79/X_esm3_79.npy
```

The UniProt feature matrix is:

```text
experiment/features/X_uniprot.npy
```

## Citation

If using this repository, please cite the associated manuscript and the original BorodaTM benchmark publication.
