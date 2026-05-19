# Measuring Directional Skill Portability Between Occupations

**Author:** Jacob Guzman

## Summary

This project constructs a measure of directional skill portability between occupations. Using O\*NET skill data (202 dimensions), CPS occupational switching flows, geographic concentration (Duncan overlap index), and job postings from Lightcast, we estimate a gravity model of occupational switching and derive occupation-level portability indices. We test whether portability predicts long-term unemployment in sectoral downturns. The pipeline implements Equations 1--7 from the project specification.

## Raw Data Acquisition

Place the following files in the `raw/` directory before running the pipeline.

| File | Place in `raw/` as | Source |
|------|-------------------|--------|
| O\*NET 30.1 database (text files) | `raw/onet/` (directory containing `Skills.txt`, `Abilities.txt`, `Knowledge.txt`, `Work Activities.txt`) | [O\*NET Resource Center](https://www.onetcenter.org/database.html) -- download "Database" (text format) |
| Census 2018 occupation crosswalk | `raw/census_2018_occ_crosswalk.csv` | [Census Bureau](https://www.census.gov/topics/employment/industry-occupation/guidance/code-lists.html) -- "2018 Census Occupation Code List and Crosswalk" |
| CPS ASEC extract | `raw/cps_asec.csv.gz` | [IPUMS CPS](https://cps.ipums.org/) -- Variables: `YEAR`, `MONTH`, `OCC`, `OCCLY`, `OCC2010`, `ASECFLAG`, `ASECWT`, `STATEFIP`, `EMPSTAT`, `LABFORCE`, `AGE`, `SEX`, `RACE`, `EDUC`, `DURUNEMP`, `CPSIDP`, `INCTOT`. Sample: ASEC supplements 2020--2025 |
| ACS 2021 1-year extract | `raw/acs_2021.csv.gz` | [IPUMS USA](https://usa.ipums.org/) -- Variables: `STATEFIP`, `PUMA`, `OCC`, `PERWT`, `EMPSTAT`, `AGE`. Must be ACS 2021 (2010-vintage PUMAs) |
| Lightcast job postings | `raw/lightcast_time_series.csv` | Lightcast (proprietary) -- columns: `year`, `month`, `soc_2021_5`, `soc_2021_5_name`, `total_postings`, `entry_level_postings`, `entry_level_pct` |
| Dorn PUMA-to-CZ crosswalk | `raw/cw_puma2010_czone.zip` | [David Dorn's website](https://www.ddorn.net/data.htm) -- "Geographical Crosswalks" section |
| AI exposure scores | `raw/ai_exposure.csv` | Eloundou et al. (2023), "GPTs are GPTs" -- columns include `OCC_CODE`, `gpt4_beta`, `automation`, `human_beta` |

## Dependencies

```bash
pip install -r requirements.txt
```

Required packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `scipy`, `statsmodels`.

## Pipeline Execution

Run scripts in order from the project root directory. All scripts use relative paths from the project root.

```bash
# Step 1: Process O*NET skill data -> skill matrix
python scripts/01_process_onet.py

# Step 2: Build Census 2018 -> O*NET crosswalk with skill vectors
python scripts/02_build_crosswalk.py

# Step 3: Process CPS switching data (pass CPS extract path as argument)
python scripts/03_process_cps.py raw/cps_asec.csv.gz

# Step 3a: Process Lightcast openings data
python scripts/03a_process_openings.py

# Step 3b: Build geographic distance (pass ACS extract path as argument)
python scripts/03b_build_geographic_distance.py raw/acs_2021.csv.gz

# Step 4: Build pairwise dataset with skill distances
python scripts/04_build_pairwise.py

# Step 5: Estimate switching models (slow -- uses ML cross-validation)
python -u scripts/05_estimate_models.py

# Step 5b: Build presentation outputs (portability by occupation)
python scripts/05b_build_presentation_outputs.py

# Step 5c: Fixed-delta1 portability index
python scripts/05c_fixed_delta1_portability_index.py

# Step 5d: Coefficient tables with two-way clustered SEs
python scripts/05d_coefficient_tables.py

# Step 6: Process AI exposure data
python scripts/06_process_ai_exposure.py

# Step 7: Sectoral downturn / long-term unemployment test
python scripts/07_sectoral_downturn.py raw/cps_asec.csv.gz
```

Intermediate outputs are written to `data/`, final results to `output/`, and figures to `figures/`.

## Figure Scripts

Figure scripts in `figure_scripts/` produce the plots used in the paper. Run them after the main pipeline completes:

```bash
python figure_scripts/fig_model_comparison.py
python figure_scripts/fig_portability_distribution.py
python figure_scripts/fig_lasso_coefficients.py
python figure_scripts/fig_geographic_concentration.py
python figure_scripts/fig_switching_distribution.py
python figure_scripts/fig_switching_heatmap.py
python figure_scripts/fig_skill_radar.py
python figure_scripts/fig_portability_by_industry.py
python figure_scripts/fig_portability_ltu.py
python figure_scripts/fig_rank_scatter.py
```

## Paper

The paper source is in `paper/paper_draft_copy.tex`. Compile with:

```bash
cd paper && pdflatex paper_draft_copy.tex && bibtex paper_draft_copy && pdflatex paper_draft_copy.tex && pdflatex paper_draft_copy.tex
```

A pre-compiled PDF is included at `paper/paper_draft_copy.pdf`.
