# Biome-Specific Supplementary Analysis (Quarto)

This folder contains a fully scoped supplementary analysis for a scientific manuscript comparing:

- A globally calibrated S2BIOPHYS hybrid PROSAIL inversion model
- Vegetation-specific calibrations

All files are intentionally localized under this directory for reproducibility and manuscript packaging.

## Structure

- `_quarto.yml`: Quarto project configuration
- `supplementary.qmd`: publication-ready supplementary manuscript
- `references.bib`: bibliography used by Quarto
- `config/validation_groups.yml`: vegetation grouping and split settings
- `scripts/`: reproducible pipeline scripts
- `data/raw/`: raw inputs
- `data/processed/`: derived intermediate files
- `results/model_outputs/`: trained model artifacts
- `results/tables/`: manuscript tables
- `results/figures/`: manuscript figures

## Suggested execution order

Run from this directory:

```bash
python scripts/01_prepare_validation_groups.py --input data/raw/validation_points.csv --class-column land_cover
python scripts/02_create_cv_splits.py
python scripts/03_train_models.py --target target
python scripts/04_evaluate_models.py --target target
python scripts/05_make_figures_tables.py
quarto render supplementary.qmd
```

## Notes

- Script defaults are relative to this folder.
- Replace placeholder column names (for example `target`, `land_cover`) with your dataset schema.
- Update `references.bib` with exact citations used in the manuscript.
