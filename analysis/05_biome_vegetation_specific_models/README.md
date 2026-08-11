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



## Optuna MySQL setup

Optuna studies use a private MySQL instance running on port `3307`.

### Initial setup

Create and initialize a user-owned MySQL data directory:

```bash
mkdir -p ~/mysql-data

mysqld \
  --initialize-insecure \
  --datadir="$HOME/mysql-data"
```

This only needs to be done once.

### Start MySQL

```bash
mysqld \
  --datadir="$HOME/mysql-data" \
  --port=3307 \
  --socket="$HOME/mysql.sock" \
  --mysqlx=OFF \
  > "$HOME/mysql.log" 2>&1 &
```

Check that the server is running:

```bash
mysqladmin -u root --socket="$HOME/mysql.sock" ping
```

Expected output:

```text
mysqld is alive
```

### Create the Optuna database

Connect:

```bash
mysql -u root --socket="$HOME/mysql.sock"
```

Create the database:

```sql
CREATE DATABASE oemc_supp_05;
SHOW DATABASES;
exit;
```

This only needs to be done once.

### Optuna configuration

Use the private MySQL instance on port `3307`:

```yaml
optuna_storage: "mysql://root@127.0.0.1:3307/oemc_supp_05"
```

### Stop MySQL

```bash
mysqladmin -u root --socket="$HOME/mysql.sock" shutdown
```

### After a restart

The database persists in `~/mysql-data`. Only restart the server:

```bash
mysqld \
  --datadir="$HOME/mysql-data" \
  --port=3307 \
  --socket="$HOME/mysql.sock" \
  --mysqlx=OFF \
  > "$HOME/mysql.log" 2>&1 &
```

Do **not** rerun `--initialize-insecure` after the initial setup.
