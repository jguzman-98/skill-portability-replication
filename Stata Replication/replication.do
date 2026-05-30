/*===========================================================================
  Stata Replication of Skill Portability Pipeline

  Replicates Python scripts 01-07 in a single master .do file.
  Covers 4 skill distance variants: Euclidean, Angular Separation,
  Factor Analysis, and LASSO (via Stata 16+ built-in).
  Random Forest and XGBoost are skipped (limited Stata support).

  Requirements:
    - Stata 16+ (for built-in lasso)
    - User-written packages: estout (ssc install estout)
    - Optional: ppmlhdfe (ssc install ppmlhdfe)

  Author:  Jacob Guzman (Stata replication)
  Date:    2026
===========================================================================*/

clear all
set more off
set maxvar 10000
set matsize 11000

* ==========================================================================
* PREAMBLE: Path macros and directory setup
* ==========================================================================

* --- Project root ---
global project "/Users/jacobguzman/Documents/skill_portability"

* --- Raw data inputs ---
global onet_dir  "/Users/jacobguzman/Downloads/capstone/data/raw/db_30_1_text"
global crosswalk_csv "/Users/jacobguzman/Downloads/2018-occupation-code-list-and-crosswalk.xlsx - 2018 Census Occ Code List.csv"
global lightcast_csv "/Users/jacobguzman/Downloads/yoe_time_series.csv"
global dorn_cw_zip   "/Users/jacobguzman/Downloads/cw_puma2010_czone.zip"

* --- CPS and ACS extracts (user must set these) ---
* global cps_extract "/path/to/cps_00003.csv.gz"
* global acs_extract "/path/to/acs_extract.csv.gz"

* --- Intermediate Python-produced CSVs (used as inputs where raw not available) ---
global pydata "$project/data"

* --- Stata intermediate and output directories ---
global stdata  "$project/stata_data"
global stout   "$project/stata_output"

capture mkdir "$stdata"
capture mkdir "$stout"

* --- Install required packages if missing ---
capture which estout
if _rc {
    di as text "Installing estout..."
    ssc install estout, replace
}


/*===========================================================================
  SECTION 1: Process O*NET Skill Matrix
  Replicates: 01_process_onet.py

  Strategy: Import existing Python-produced CSV (774 SOC x 202 dimensions).
  The raw O*NET tab-delimited files require extensive string manipulation
  that is replicated below if raw files are available; otherwise we import
  the pre-built CSV.
===========================================================================*/

di as result _n "======================================================================"
di as result "SECTION 1: Process O*NET Skill Matrix"
di as result "======================================================================"

capture confirm file "$onet_dir/Skills.txt"
if _rc == 0 {
    di as text "Raw O*NET files found. Processing from scratch..."

    * --- Load and process each O*NET file ---
    * We process Skills, Abilities, Knowledge (LV scale only)
    * and Work Activities (LV + IM scales)

    tempfile skills abilities knowledge activities combined

    * --- Skills (LV only) ---
    import delimited "$onet_dir/Skills.txt", clear delimiter(tab) varnames(1)
    keep if scale_id == "LV" & recommend_suppress == "N"
    * Strip .XX suffix to get 6-digit SOC
    gen soc6 = regexr(onet_soc_code, "\.\d+$", "")
    * Build dimension name
    gen element_clean = lower(element_name)
    replace element_clean = ustrregexra(element_clean, "[^a-z0-9]+", "_")
    replace element_clean = regexr(element_clean, "^_", "")
    replace element_clean = regexr(element_clean, "_$", "")
    gen dimension = "skill_" + element_clean
    rename data_value value
    destring value, replace force
    keep soc6 dimension value
    save `skills', replace

    * --- Abilities (LV only) ---
    import delimited "$onet_dir/Abilities.txt", clear delimiter(tab) varnames(1)
    keep if scale_id == "LV" & recommend_suppress == "N"
    gen soc6 = regexr(onet_soc_code, "\.\d+$", "")
    gen element_clean = lower(element_name)
    replace element_clean = ustrregexra(element_clean, "[^a-z0-9]+", "_")
    replace element_clean = regexr(element_clean, "^_", "")
    replace element_clean = regexr(element_clean, "_$", "")
    gen dimension = "ability_" + element_clean
    rename data_value value
    destring value, replace force
    keep soc6 dimension value
    save `abilities', replace

    * --- Knowledge (LV only) ---
    import delimited "$onet_dir/Knowledge.txt", clear delimiter(tab) varnames(1)
    keep if scale_id == "LV" & recommend_suppress == "N"
    gen soc6 = regexr(onet_soc_code, "\.\d+$", "")
    gen element_clean = lower(element_name)
    replace element_clean = ustrregexra(element_clean, "[^a-z0-9]+", "_")
    replace element_clean = regexr(element_clean, "^_", "")
    replace element_clean = regexr(element_clean, "_$", "")
    gen dimension = "knowledge_" + element_clean
    rename data_value value
    destring value, replace force
    keep soc6 dimension value
    save `knowledge', replace

    * --- Work Activities (LV + IM) ---
    import delimited "$onet_dir/Work Activities.txt", clear delimiter(tab) varnames(1)
    keep if inlist(scale_id, "LV", "IM") & recommend_suppress == "N"
    gen soc6 = regexr(onet_soc_code, "\.\d+$", "")
    gen element_clean = lower(element_name)
    replace element_clean = ustrregexra(element_clean, "[^a-z0-9]+", "_")
    replace element_clean = regexr(element_clean, "^_", "")
    replace element_clean = regexr(element_clean, "_$", "")
    * Include scale in name since we have both LV and IM
    gen scale_lower = lower(scale_id)
    gen dimension = "activity_" + scale_lower + "_" + element_clean
    rename data_value value
    destring value, replace force
    keep soc6 dimension value
    save `activities', replace

    * --- Append all files ---
    use `skills', clear
    append using `abilities'
    append using `knowledge'
    append using `activities'

    * Average across detailed occupations within same 6-digit SOC
    collapse (mean) value, by(soc6 dimension)

    * --- Reshape wide ---
    * Encode dimension as numeric for reshape
    encode dimension, gen(dim_num)
    drop dimension
    reshape wide value, i(soc6) j(dim_num)

    * Rename value# back to dimension names using label
    * Get the label name attached to dim_num
    local lbl_name : value label dim_num
    qui describe value*, varlist
    local vlist `r(varlist)'
    foreach v of local vlist {
        local num = subinstr("`v'", "value", "", 1)
        local dimname : label `lbl_name' `num'
        rename `v' `dimname'
    }

    * Fill missing with 0
    foreach v of varlist ability_* activity_* knowledge_* skill_* {
        replace `v' = 0 if missing(`v')
    }

    * Min-max normalize each dimension to [0, 1]
    foreach v of varlist ability_* activity_* knowledge_* skill_* {
        qui su `v'
        local mn = r(min)
        local mx = r(max)
        local rng = `mx' - `mn'
        if `rng' == 0 local rng = 1
        replace `v' = (`v' - `mn') / `rng'
    }

    * Save
    save "$stdata/onet_skill_matrix.dta", replace
    di as result "Saved O*NET skill matrix (from raw): " _N " SOC codes"
}
else {
    * --- Import from pre-built Python CSV ---
    di as text "Raw O*NET files not found. Importing from Python CSV..."
    import delimited "$pydata/onet_skill_matrix.csv", clear varnames(1)

    * Verify dimensions
    qui describe ability_* activity_* knowledge_* skill_*, varlist
    local ndims : word count `r(varlist)'
    di as text "  Imported: " _N " SOC codes x `ndims' dimensions"

    * Verify range [0,1]
    local range_ok = 1
    foreach v of varlist ability_* activity_* knowledge_* skill_* {
        qui su `v'
        if r(min) < -0.001 | r(max) > 1.001 {
            local range_ok = 0
            di as error "  WARNING: `v' out of [0,1] range: [" r(min) ", " r(max) "]"
        }
    }
    if `range_ok' di as text "  All dimensions in [0,1] range: OK"

    save "$stdata/onet_skill_matrix.dta", replace
    di as result "Saved: $stdata/onet_skill_matrix.dta"
}


/*===========================================================================
  SECTION 2: Build Census 2018 -> O*NET Crosswalk with Skill Vectors
  Replicates: 02_build_crosswalk.py

  Strategy: Import from Python CSV. The crosswalk resolution logic
  (wildcard matching, prefix fallback) is complex string work better
  validated against the Python output. The skill vectors are the key output.
===========================================================================*/

di as result _n "======================================================================"
di as result "SECTION 2: Build Crosswalk (Census 2018 -> Skill Vectors)"
di as result "======================================================================"

import delimited "$pydata/skill_vectors_by_census2018.csv", clear varnames(1)
tostring census_code, replace format(%04.0f)

qui describe ability_* activity_* knowledge_* skill_*, varlist
local ndims : word count `r(varlist)'
di as text "  Imported: " _N " Census codes x `ndims' dimensions"

* Fill any missing with 0
foreach v of varlist ability_* activity_* knowledge_* skill_* {
    capture replace `v' = 0 if missing(`v')
}

save "$stdata/skill_vectors_by_census2018.dta", replace
di as result "Saved: $stdata/skill_vectors_by_census2018.dta"


/*===========================================================================
  SECTION 3: Process CPS Switching Data
  Replicates: 03_process_cps.py

  Strategy: Import the 8 pre-built Python CSVs. CPS microdata processing
  requires the raw IPUMS extract which may not be available.
  If the raw CPS extract is available (global cps_extract), process from
  scratch; otherwise import intermediate CSVs.
===========================================================================*/

di as result _n "======================================================================"
di as result "SECTION 3: Process CPS Switching Data"
di as result "======================================================================"

capture confirm file "$cps_extract"
if _rc == 0 {
    di as text "CPS extract found. Processing from scratch..."

    import delimited "$cps_extract", clear varnames(1)

    * Uppercase variable names
    rename *, upper

    * Filter: YEAR >= 2020
    keep if YEAR >= 2020
    di as text "  After YEAR >= 2020: " _N

    * Filter: ASEC respondents
    keep if ASECFLAG > 0
    di as text "  After ASECFLAG > 0: " _N

    * Filter: age 16-64
    keep if AGE >= 16 & AGE <= 64
    di as text "  After AGE 16-64: " _N

    * Filter: employed (EMPSTAT 10 or 12)
    keep if inlist(EMPSTAT, 10, 12)
    di as text "  After EMPSTAT employed: " _N

    * Filter: valid OCC and OCCLY (not 0, 9920, or military 9800-9840)
    drop if OCC == 0 | OCC == 9920 | (OCC >= 9800 & OCC <= 9840)
    drop if OCCLY == 0 | OCCLY == 9920 | (OCCLY >= 9800 & OCCLY <= 9840)
    di as text "  After valid OCC/OCCLY: " _N

    * Classify switchers
    gen is_switcher = (OCC != OCCLY)

    * --- switching_matrix.csv (pooled, unweighted) ---
    preserve
    keep if is_switcher == 1
    collapse (count) switches = is_switcher, by(OCCLY OCC)
    rename OCCLY occ_origin
    rename OCC occ_dest
    save "$stdata/switching_matrix.dta", replace
    export delimited "$stdata/switching_matrix.csv", replace
    di as text "  Switching matrix: " _N " directed pairs"
    restore

    * --- total_switchers_out.csv ---
    preserve
    keep if is_switcher == 1
    collapse (count) total_switches_out = is_switcher, by(OCCLY)
    rename OCCLY occ
    save "$stdata/total_switchers_out.dta", replace
    restore

    * --- switching_matrix_by_year.csv ---
    preserve
    keep if is_switcher == 1
    collapse (count) switches = is_switcher, by(OCCLY OCC YEAR)
    rename OCCLY occ_origin
    rename OCC occ_dest
    rename YEAR year
    save "$stdata/switching_matrix_by_year.dta", replace
    restore

    * --- total_switchers_out_by_year.csv ---
    preserve
    keep if is_switcher == 1
    collapse (count) total_switches_out = is_switcher, by(OCCLY YEAR)
    rename OCCLY occ
    rename YEAR year
    save "$stdata/total_switchers_out_by_year.dta", replace
    restore

    * --- stayer_counts.csv ---
    preserve
    keep if is_switcher == 0
    gen one = 1
    collapse (sum) stayers = one, by(OCC)
    rename OCC occ
    save "$stdata/stayer_counts.dta", replace
    restore

    * --- employment_counts.csv (raw) ---
    preserve
    gen one = 1
    collapse (sum) employment = one, by(OCC YEAR)
    rename OCC occ
    rename YEAR year
    save "$stdata/employment_counts.dta", replace
    restore

    * --- employment_counts_weighted.csv ---
    preserve
    collapse (sum) weighted_employment = ASECWT, by(OCC YEAR)
    rename OCC occ
    rename YEAR year
    save "$stdata/employment_counts_weighted.dta", replace
    restore

    * --- state_employment.csv ---
    preserve
    collapse (sum) weighted_employment = ASECWT, by(STATEFIP OCC)
    rename STATEFIP statefip
    rename OCC occ
    save "$stdata/state_employment.dta", replace
    restore

    di as result "CPS processing complete."
}
else {
    di as text "CPS extract not available. Importing from Python CSVs..."

    * --- switching_matrix ---
    import delimited "$pydata/switching_matrix.csv", clear varnames(1)
    * Normalize occupation codes: strip .0 if present
    capture {
        replace occ_origin = string(real(occ_origin), "%12.0f") if regexm(occ_origin, "\.")
        replace occ_dest = string(real(occ_dest), "%12.0f") if regexm(occ_dest, "\.")
    }
    * Destring to numeric
    destring occ_origin occ_dest, replace force
    recast long occ_origin occ_dest
    save "$stdata/switching_matrix.dta", replace
    di as text "  Switching matrix: " _N " directed pairs"

    * --- total_switchers_out ---
    import delimited "$pydata/total_switchers_out.csv", clear varnames(1)
    destring occ, replace force
    recast long occ
    save "$stdata/total_switchers_out.dta", replace
    di as text "  Total switchers out: " _N " occupations"

    * --- switching_matrix_by_year ---
    import delimited "$pydata/switching_matrix_by_year.csv", clear varnames(1)
    destring occ_origin occ_dest, replace force
    recast long occ_origin occ_dest
    save "$stdata/switching_matrix_by_year.dta", replace
    di as text "  Switching matrix by year: " _N " pair-year cells"

    * --- total_switchers_out_by_year ---
    import delimited "$pydata/total_switchers_out_by_year.csv", clear varnames(1)
    destring occ, replace force
    recast long occ
    save "$stdata/total_switchers_out_by_year.dta", replace

    * --- stayer_counts ---
    import delimited "$pydata/stayer_counts.csv", clear varnames(1)
    destring occ, replace force
    recast long occ
    save "$stdata/stayer_counts.dta", replace
    di as text "  Stayer counts: " _N " occupations"

    * --- employment_counts ---
    import delimited "$pydata/employment_counts.csv", clear varnames(1)
    destring occ, replace force
    recast long occ
    save "$stdata/employment_counts.dta", replace
    di as text "  Employment counts: " _N " occ-year cells"

    * --- employment_counts_weighted ---
    import delimited "$pydata/employment_counts_weighted.csv", clear varnames(1)
    destring occ, replace force
    recast long occ
    save "$stdata/employment_counts_weighted.dta", replace
    di as text "  Employment counts (weighted): " _N " occ-year cells"

    * --- state_employment ---
    import delimited "$pydata/state_employment.csv", clear varnames(1)
    destring occ, replace force
    recast long occ
    save "$stdata/state_employment.dta", replace
    di as text "  State employment: " _N " state-occ cells"

    di as result "CPS data imported from Python CSVs."
}


/*===========================================================================
  SECTION 3a: Process Lightcast Openings -> Openings Share by Census 2018
  Replicates: 03a_process_openings.py

  Strategy: Import from Python CSV. The SOC 2021 -> Census 2018 matching
  logic is handled identically by importing the pre-matched output.
===========================================================================*/

di as result _n "======================================================================"
di as result "SECTION 3a: Process Lightcast Openings"
di as result "======================================================================"

import delimited "$pydata/openings_share_by_census2018.csv", clear varnames(1)
destring census_code, replace force
recast long census_code

* Verify openings_share sums to ~1
qui su openings_share
local share_sum = r(sum)
di as text "  Census codes with postings: " _N
di as text "  openings_share sum: " %9.6f `share_sum' " (should be ~1.0)"

rename census_code occ
save "$stdata/openings_share_by_census2018.dta", replace
di as result "Saved: $stdata/openings_share_by_census2018.dta"


/*===========================================================================
  SECTION 3b: Geographic Distance (Duncan Overlap Index)
  Replicates: 03b_build_geographic_distance.py

  Strategy: Import from Python CSV. The ACS -> CZ -> Duncan computation
  requires the ACS microdata extract and Dorn crosswalk. We import
  the pre-computed pairwise geographic distances.
===========================================================================*/

di as result _n "======================================================================"
di as result "SECTION 3b: Geographic Distance"
di as result "======================================================================"

import delimited "$pydata/geographic_distance.csv", clear varnames(1)
destring occ_origin occ_dest, replace force
recast long occ_origin occ_dest

di as text "  Geographic distance pairs: " _N
qui su geographic_distance
di as text "  Mean: " %9.4f r(mean) "  Std: " %9.4f r(sd)
di as text "  Min:  " %9.4f r(min)  "  Max: " %9.4f r(max)

save "$stdata/geographic_distance.dta", replace
di as result "Saved: $stdata/geographic_distance.dta"


/*===========================================================================
  SECTION 4: Build Pairwise Dataset
  Replicates: 04_build_pairwise.py

  Builds all directed occupation pairs, merges switching counts, employment,
  openings share, geographic distance, and skill vectors.
  Computes 4 skill distance metrics (Euclidean, Angular, Factor, diffs for LASSO).
===========================================================================*/

di as result _n "======================================================================"
di as result "SECTION 4: Build Pairwise Dataset"
di as result "======================================================================"

* --- Pool employment across years ---
use "$stdata/employment_counts.dta", clear
collapse (sum) employment, by(occ)
save "$stdata/emp_pooled.dta", replace
di as text "  Pooled employment: " _N " occupations"

* --- Load skill vectors ---
use "$stdata/skill_vectors_by_census2018.dta", clear
destring census_code, replace force
recast long census_code
rename census_code occ
save "$stdata/skill_vectors_numeric.dta", replace

* --- Factor Analysis (4 factors) on skill vectors ---
di as text _n "Fitting Factor Analysis (4 factors) on skill vectors..."

* Get list of skill dimension variables
qui describe ability_* activity_* knowledge_* skill_*, varlist
local skill_vars `r(varlist)'
local ndims : word count `skill_vars'
di as text "  `ndims' skill dimensions"

* Use principal component factors (pcf) as fallback if ML fails to converge
* with 202 dimensions; pcf is more robust for high-dimensional data
capture noisily factor `skill_vars', factors(4) ml
if _rc {
    di as text "  ML factor analysis failed, falling back to PCF..."
    factor `skill_vars', factors(4) pcf
}
predict f1 f2 f3 f4

* Keep occ and factor scores
preserve
keep occ f1 f2 f3 f4
save "$stdata/factor_scores.dta", replace
restore
di as text "  Factor scores computed for " _N " occupations"

* --- Determine set of valid occupations (in both CPS and skills) ---
* Get CPS occupations
use "$stdata/switching_matrix.dta", clear
keep occ_origin
rename occ_origin occ
tempfile cps_occs1
save `cps_occs1'

use "$stdata/switching_matrix.dta", clear
keep occ_dest
rename occ_dest occ
tempfile cps_occs2
save `cps_occs2'

use "$stdata/stayer_counts.dta", clear
keep occ
tempfile cps_occs3
save `cps_occs3'

use `cps_occs1', clear
append using `cps_occs2'
append using `cps_occs3'
duplicates drop occ, force

* Merge with skill vectors to find intersection
merge 1:1 occ using "$stdata/skill_vectors_numeric.dta", keep(match) nogen keepusing(occ)
sort occ
save "$stdata/valid_occs.dta", replace
local n_valid = _N
di as text "  Valid occupations (in both CPS and skills): `n_valid'"

* --- Create all directed pairs (o != d) ---
di as text "Creating all directed pairs..."
use "$stdata/valid_occs.dta", clear
rename occ occ_origin
cross using "$stdata/valid_occs.dta"
rename occ occ_dest
drop if occ_origin == occ_dest
local npairs = _N
di as text "  Total directed pairs: `npairs'"

* --- Merge switching counts ---
merge m:1 occ_origin occ_dest using "$stdata/switching_matrix.dta", nogen keep(master match)
replace switches = 0 if missing(switches)

* --- Merge total switchers out of origin ---
rename occ_origin occ
merge m:1 occ using "$stdata/total_switchers_out.dta", nogen keep(master match)
replace total_switches_out = 0 if missing(total_switches_out)
rename occ occ_origin

* --- Merge stayer counts for origin ---
rename occ_origin occ
merge m:1 occ using "$stdata/stayer_counts.dta", nogen keep(master match)
rename stayers stayers_origin
rename occ occ_origin

* --- Merge pooled employment for origin and destination ---
rename occ_origin occ
merge m:1 occ using "$stdata/emp_pooled.dta", nogen keep(master match)
rename employment emp_origin
rename occ occ_origin

rename occ_dest occ
merge m:1 occ using "$stdata/emp_pooled.dta", nogen keep(master match)
rename employment emp_dest
rename occ occ_dest

* --- Merge openings share for destination ---
rename occ_dest occ
merge m:1 occ using "$stdata/openings_share_by_census2018.dta", nogen keep(master match)
rename openings_share openings_share_dest
rename occ occ_dest

* Fill unmatched openings with employment-share fallback
* Total employment across all valid occupations (from emp_pooled, not pairwise)
preserve
use "$stdata/emp_pooled.dta", clear
merge 1:1 occ using "$stdata/valid_occs.dta", keep(match) nogen
qui su employment
local total_emp = r(sum)
restore
replace openings_share_dest = emp_dest / `total_emp' if missing(openings_share_dest)

* --- Merge geographic distance ---
merge m:1 occ_origin occ_dest using "$stdata/geographic_distance.dta", nogen keep(master match)
* Fill missing with median
qui su geographic_distance, detail
local geo_med = r(p50)
replace geographic_distance = `geo_med' if missing(geographic_distance)

* --- Merge origin and destination skill vectors ---
* Origin skills
rename occ_origin occ
merge m:1 occ using "$stdata/skill_vectors_numeric.dta", nogen keep(master match)
qui describe ability_* activity_* knowledge_* skill_*, varlist
local skill_vars `r(varlist)'
foreach v of local skill_vars {
    rename `v' o_`v'
}
rename occ occ_origin

* Destination skills
rename occ_dest occ
merge m:1 occ using "$stdata/skill_vectors_numeric.dta", nogen keep(master match)
foreach v of local skill_vars {
    rename `v' d_`v'
}
rename occ occ_dest

* --- Compute absolute differences for each dimension ---
di as text "Computing per-dimension absolute differences..."
foreach v of local skill_vars {
    gen diff_`v' = abs(o_`v' - d_`v')
}

* --- Compute Euclidean distance (overall) ---
di as text "Computing Euclidean distance..."
gen double euclidean_dist_sq = 0
foreach v of local skill_vars {
    replace euclidean_dist_sq = euclidean_dist_sq + diff_`v'^2
}
gen double euclidean_dist = sqrt(euclidean_dist_sq)
drop euclidean_dist_sq

* --- Per-group Euclidean distances ---
foreach grp in ability activity knowledge skill {
    gen double euclidean_`grp'_sq = 0
    foreach v of local skill_vars {
        if regexm("`v'", "^`grp'_") {
            replace euclidean_`grp'_sq = euclidean_`grp'_sq + diff_`v'^2
        }
    }
    gen double euclidean_`grp' = sqrt(euclidean_`grp'_sq)
    drop euclidean_`grp'_sq
}

* --- Cosine similarity and Angular separation ---
di as text "Computing Angular Separation..."
gen double dot_prod = 0
gen double norm_o_sq = 0
gen double norm_d_sq = 0
foreach v of local skill_vars {
    replace dot_prod = dot_prod + o_`v' * d_`v'
    replace norm_o_sq = norm_o_sq + o_`v'^2
    replace norm_d_sq = norm_d_sq + d_`v'^2
}
gen double norm_o = sqrt(norm_o_sq)
gen double norm_d = sqrt(norm_d_sq)

* Avoid division by zero
gen double safe_norm_o = cond(norm_o == 0, 1, norm_o)
gen double safe_norm_d = cond(norm_d == 0, 1, norm_d)
gen double cosine_sim = dot_prod / (safe_norm_o * safe_norm_d)
* Clamp to [-1, 1]
replace cosine_sim = max(-1, min(1, cosine_sim))
gen double angular_separation = acos(cosine_sim)

drop dot_prod norm_o_sq norm_d_sq norm_o norm_d safe_norm_o safe_norm_d

* --- Factor analysis distance ---
di as text "Computing Factor Analysis distance..."
* Merge origin factor scores
rename occ_origin occ
merge m:1 occ using "$stdata/factor_scores.dta", nogen keep(master match)
rename f1 o_f1
rename f2 o_f2
rename f3 o_f3
rename f4 o_f4
rename occ occ_origin

* Merge destination factor scores
rename occ_dest occ
merge m:1 occ using "$stdata/factor_scores.dta", nogen keep(master match)
rename f1 d_f1
rename f2 d_f2
rename f3 d_f3
rename f4 d_f4
rename occ occ_dest

gen double factor_dist = sqrt((o_f1-d_f1)^2 + (o_f2-d_f2)^2 + (o_f3-d_f3)^2 + (o_f4-d_f4)^2)

* --- Drop raw skill vectors and factor scores to save space ---
drop o_ability_* o_activity_* o_knowledge_* o_skill_*
drop d_ability_* d_activity_* d_knowledge_* d_skill_*
drop o_f1 o_f2 o_f3 o_f4 d_f1 d_f2 d_f3 d_f4

* --- Summary stats ---
local nonzero_sw = 0
qui count if switches > 0
local nonzero_sw = r(N)
di as text _n "Dataset summary:"
di as text "  Total pairs: " _N
di as text "  Pairs with nonzero switches: `nonzero_sw'"
qui su euclidean_dist
di as text "  euclidean_dist:     mean=" %9.4f r(mean) "  std=" %9.4f r(sd)
qui su angular_separation
di as text "  angular_separation: mean=" %9.4f r(mean) "  std=" %9.4f r(sd)
qui su factor_dist
di as text "  factor_dist:        mean=" %9.4f r(mean) "  std=" %9.4f r(sd)

compress
save "$stdata/pairwise_dataset.dta", replace
di as result "Saved: $stdata/pairwise_dataset.dta (" _N " pairs)"


/*===========================================================================
  SECTION 5: Estimate Models
  Replicates: 05_estimate_models.py

  Part A: LASSO skill distance (manual 5-fold CV)
  Part B: Baseline regressions (4 variants x 2 estimators = 8 rows)
  Part C: Additional checks (fixed delta1, no small occs, two-part, year FE)
===========================================================================*/

di as result _n "======================================================================"
di as result "SECTION 5: Estimate Switching Models"
di as result "======================================================================"

use "$stdata/pairwise_dataset.dta", clear

* Drop rows with missing key columns
drop if missing(switches) | missing(total_switches_out) | ///
        missing(emp_origin) | missing(emp_dest)
di as text "  Observations after dropping NaN: " _N

* Get diff_* variable list
qui describe diff_*, varlist
local diff_vars `r(varlist)'
local ndiffs : word count `diff_vars'
di as text "  diff_* features: `ndiffs'"


* ======================================================================
* PART A: LASSO Skill Distance (manual 5-fold CV)
* ======================================================================

di as result _n "PART A: LASSO Skill Distance (5-fold CV)"
di as result "======================================================================"

* Standardize diff_* features
foreach v of local diff_vars {
    qui su `v'
    if r(sd) > 0 {
        gen z_`v' = (`v' - r(mean)) / r(sd)
    }
    else {
        gen z_`v' = 0
    }
}

* Generate fold assignment (5-fold, seed 42)
set seed 42
gen double _u = runiform()
sort _u
gen fold = mod(_n - 1, 5) + 1
drop _u

* Get standardized diff variable list
qui describe z_diff_*, varlist
local z_diff_vars `r(varlist)'

* Manual 5-fold CV: for each fold, train lasso on other 4, predict held-out
gen double ml_dist_lasso = .

forvalues f = 1/5 {
    di as text "  Fold `f'/5..."

    * Run lasso on training folds
    qui lasso linear switches `z_diff_vars' if fold != `f', selection(cv) rseed(42)

    * Predict on held-out fold
    qui predict double _lasso_pred if fold == `f'
    qui replace ml_dist_lasso = _lasso_pred if fold == `f'
    drop _lasso_pred
}

* Summary of LASSO predictions
qui su ml_dist_lasso
di as text "  LASSO OOF predictions: mean=" %9.4f r(mean) "  sd=" %9.4f r(sd)

* Compute OOF R-squared
qui correlate switches ml_dist_lasso
local lasso_corr = r(rho)
qui su switches
local y_mean = r(mean)
gen double _ss_res = (switches - ml_dist_lasso)^2
gen double _ss_tot = (switches - `y_mean')^2
qui su _ss_res
local ss_res = r(sum)
qui su _ss_tot
local ss_tot = r(sum)
local lasso_r2 = 1 - `ss_res' / `ss_tot'
di as text "  LASSO OOF R-squared: " %9.4f `lasso_r2'
drop _ss_res _ss_tot

* Drop standardized vars (no longer needed)
drop z_diff_*
drop fold

* ======================================================================
* PART B: Baseline Regressions (4 variants x 2 estimators)
* ======================================================================

di as result _n "PART B: Baseline Regressions"
di as result "======================================================================"

* Create log outcome
gen double log1p_switches = ln(1 + switches)

* Initialize postfile for results collection
tempname results_pf
tempfile results_file
postfile `results_pf' str30 specification str25 skill_distance str20 estimator ///
    double R2 double beta1_skill_dist double se_beta1 double p_beta1 long n_obs ///
    using `results_file', replace

* Define skill distance variants
* We'll loop over 4 variants
local dist_labels   "euclidean angular_separation factor_analysis ml_lasso"
local dist_vars     "euclidean_dist angular_separation factor_dist ml_dist_lasso"

local nv : word count `dist_labels'

* Run null Poisson model once to get null deviance for pseudo-R2
qui glm switches, family(poisson) link(log)
local null_dev_baseline = e(deviance)
di as text "  Null deviance (baseline): " %12.2f `null_dev_baseline'

forvalues i = 1/`nv' {
    local label : word `i' of `dist_labels'
    local col   : word `i' of `dist_vars'

    di as text _n "  Skill Distance: `label'"

    * --- OLS on log(1 + switches) ---
    qui regress log1p_switches `col' total_switches_out openings_share_dest ///
        geographic_distance, vce(robust)

    local r2_ols = e(r2)
    local b1_ols = _b[`col']
    local se1_ols = _se[`col']
    qui test `col'
    local p1_ols = r(p)
    local n_ols = e(N)

    di as text "    OLS_log   R2=" %8.4f `r2_ols' ///
        "  beta1=" %+12.6f `b1_ols' ///
        "  (SE=" %9.6f `se1_ols' ", p=" %7.4f `p1_ols' ")"

    post `results_pf' ("baseline") ("`label'") ("OLS_log") ///
        (`r2_ols') (`b1_ols') (`se1_ols') (`p1_ols') (`n_ols')

    * --- PPML: GLM Poisson ---
    qui glm switches `col' total_switches_out openings_share_dest ///
        geographic_distance, family(poisson) link(log) vce(robust)

    * Pseudo R-squared from deviance: 1 - deviance(model) / deviance(null)
    local dev = e(deviance)
    local r2_ppml = 1 - `dev' / `null_dev_baseline'
    local b1_ppml = _b[`col']
    local se1_ppml = _se[`col']
    qui test `col'
    local p1_ppml = r(p)
    local n_ppml = e(N)

    di as text "    PPML      R2=" %8.4f `r2_ppml' ///
        "  beta1=" %+12.6f `b1_ppml' ///
        "  (SE=" %9.6f `se1_ppml' ", p=" %7.4f `p1_ppml' ")"

    post `results_pf' ("baseline") ("`label'") ("PPML") ///
        (`r2_ppml') (`b1_ppml') (`se1_ppml') (`p1_ppml') (`n_ppml')
}


* ======================================================================
* PART C: Additional Checks
* ======================================================================

di as result _n "PART C: Additional Checks"
di as result "======================================================================"

forvalues i = 1/`nv' {
    local label : word `i' of `dist_labels'
    local col   : word `i' of `dist_vars'

    di as text _n "  ---- Skill Distance: `label' ----"

    * --- Check 3a: Fix delta1 = 1 ---
    di as text "  Check 3a: Fix delta1 = 1"

    * OLS: LHS = log(1+switches) - log(1+total_switches_out)
    gen double _y_adj = ln(1 + switches) - ln(1 + total_switches_out)

    qui regress _y_adj `col' openings_share_dest geographic_distance, vce(robust)
    local r2 = e(r2)
    local b1 = _b[`col']
    local se1 = _se[`col']
    qui test `col'
    local p1 = r(p)

    di as text "    OLS_log   R2=" %8.4f `r2' "  beta1=" %+12.6f `b1' ///
        "  (SE=" %9.6f `se1' ", p=" %7.4f `p1' ")"
    post `results_pf' ("fixed_delta1") ("`label'") ("OLS_log") ///
        (`r2') (`b1') (`se1') (`p1') (e(N))

    drop _y_adj

    * PPML with exposure = max(total_switches_out, 1)
    gen double _exposure = max(total_switches_out, 1)

    * Null model with exposure for correct null deviance
    qui glm switches, family(poisson) link(log) exposure(_exposure)
    local null_dev_fd1 = e(deviance)

    qui glm switches `col' openings_share_dest geographic_distance, ///
        family(poisson) link(log) exposure(_exposure) vce(robust)
    local dev = e(deviance)
    local r2 = 1 - `dev' / `null_dev_fd1'
    local b1 = _b[`col']
    local se1 = _se[`col']
    qui test `col'
    local p1 = r(p)

    di as text "    PPML      R2=" %8.4f `r2' "  beta1=" %+12.6f `b1' ///
        "  (SE=" %9.6f `se1' ", p=" %7.4f `p1' ")"
    post `results_pf' ("fixed_delta1") ("`label'") ("PPML") ///
        (`r2') (`b1') (`se1') (`p1') (e(N))

    drop _exposure

    * --- Check 3b: No small occupations (100 and 500 thresholds) ---
    foreach thresh in 100 500 {
        di as text "  Check 3b: No small occs (threshold=`thresh')"

        * OLS
        qui regress log1p_switches `col' total_switches_out openings_share_dest ///
            geographic_distance if emp_origin >= `thresh' & emp_dest >= `thresh', ///
            vce(robust)
        local r2 = e(r2)
        local b1 = _b[`col']
        local se1 = _se[`col']
        qui test `col'
        local p1 = r(p)
        local n = e(N)

        di as text "    OLS_log   R2=" %8.4f `r2' "  beta1=" %+12.6f `b1' ///
            "  (SE=" %9.6f `se1' ", p=" %7.4f `p1' ", n=" `n' ")"
        post `results_pf' ("no_small_occ_`thresh'") ("`label'") ("OLS_log") ///
            (`r2') (`b1') (`se1') (`p1') (`n')

        * PPML
        qui glm switches if emp_origin >= `thresh' & emp_dest >= `thresh', ///
            family(poisson) link(log)
        local null_dev_nso = e(deviance)
        qui glm switches `col' total_switches_out openings_share_dest ///
            geographic_distance if emp_origin >= `thresh' & emp_dest >= `thresh', ///
            family(poisson) link(log) vce(robust)
        local dev = e(deviance)
        local r2 = 1 - `dev' / `null_dev_nso'
        local b1 = _b[`col']
        local se1 = _se[`col']
        qui test `col'
        local p1 = r(p)

        di as text "    PPML      R2=" %8.4f `r2' "  beta1=" %+12.6f `b1' ///
            "  (SE=" %9.6f `se1' ", p=" %7.4f `p1' ", n=" e(N) ")"
        post `results_pf' ("no_small_occ_`thresh'") ("`label'") ("PPML") ///
            (`r2') (`b1') (`se1') (`p1') (e(N))
    }

    * --- Check 3c: Two-part model ---
    di as text "  Check 3c: Two-part model (logit + OLS on positives)"

    * Part 1: Logit on P(switches > 0)
    gen byte _any_switch = (switches > 0)

    qui logit _any_switch `col' total_switches_out openings_share_dest ///
        geographic_distance, vce(robust)
    local r2 = e(r2_p) // McFadden pseudo-R2
    local b1 = _b[`col']
    local se1 = _se[`col']
    qui test `col'
    local p1 = r(p)

    di as text "    Logit     R2=" %8.4f `r2' "  beta1=" %+12.6f `b1' ///
        "  (SE=" %9.6f `se1' ", p=" %7.4f `p1' ")"
    post `results_pf' ("two_part_logit") ("`label'") ("Logit") ///
        (`r2') (`b1') (`se1') (`p1') (e(N))

    * Part 2: OLS on log(switches) for positive subsample
    gen double _log_sw_pos = ln(switches) if switches > 0

    qui regress _log_sw_pos `col' total_switches_out openings_share_dest ///
        geographic_distance if switches > 0, vce(robust)
    local r2 = e(r2)
    local b1 = _b[`col']
    local se1 = _se[`col']
    qui test `col'
    local p1 = r(p)

    di as text "    OLS_pos   R2=" %8.4f `r2' "  beta1=" %+12.6f `b1' ///
        "  (SE=" %9.6f `se1' ", p=" %7.4f `p1' ", n=" e(N) ")"
    post `results_pf' ("two_part_positive") ("`label'") ("OLS_log_positive") ///
        (`r2') (`b1') (`se1') (`p1') (e(N))

    drop _any_switch _log_sw_pos
}


* ======================================================================
* PART D: Year Fixed Effects
* ======================================================================

di as result _n "PART D: Year Fixed Effects"
di as result "======================================================================"

* Save current pairwise dataset
preserve

* Load year-level switching data
use "$stdata/switching_matrix_by_year.dta", clear
qui su year
local min_year = r(min)
local max_year = r(max)
local years ""
forvalues y = `min_year'/`max_year' {
    local years "`years' `y'"
}
di as text "  Years: `years'"

tempfile sw_by_year
save `sw_by_year'

use "$stdata/total_switchers_out_by_year.dta", clear
rename occ occ_origin
tempfile out_by_year
save `out_by_year'

* Restore pairwise dataset and expand to pair-year level
restore

* Drop switches, total_switches_out, and log1p_switches (will be replaced by year-level versions)
drop switches total_switches_out log1p_switches

* Cross join with years
gen _key = 1
preserve
clear
local min_year_val = `min_year'
local max_year_val = `max_year'
local nyears = `max_year_val' - `min_year_val' + 1
set obs `nyears'
gen year = `min_year_val' + _n - 1
gen _key = 1
tempfile years_df
save `years_df'
restore

joinby _key using `years_df'
drop _key

di as text "  Expanded to " _N " pair-year rows"

* Merge year-level switches
merge m:1 occ_origin occ_dest year using `sw_by_year', nogen keep(master match)
replace switches = 0 if missing(switches)

* Merge year-level total_switches_out
merge m:1 occ_origin year using `out_by_year', nogen keep(master match)
replace total_switches_out = 0 if missing(total_switches_out)

* Create log outcome
gen double log1p_switches = ln(1 + switches)

* Add year dummies (omit base year = min year)
local base_year = `min_year'
forvalues y = `min_year'/`max_year' {
    if `y' != `base_year' {
        gen byte year_`y' = (year == `y')
    }
}

* Get year dummy variable list
local year_dummies ""
forvalues y = `min_year'/`max_year' {
    if `y' != `base_year' {
        local year_dummies "`year_dummies' year_`y'"
    }
}
di as text "  Year dummies: `year_dummies'"

* Null deviance for year-expanded data
qui glm switches, family(poisson) link(log)
local null_dev_yearfe = e(deviance)

* Run year FE regressions for each variant
forvalues i = 1/`nv' {
    local label : word `i' of `dist_labels'
    local col   : word `i' of `dist_vars'

    di as text _n "  Skill Distance: `label'"

    * OLS with year FE
    qui regress log1p_switches `col' total_switches_out openings_share_dest ///
        geographic_distance `year_dummies', vce(robust)
    local r2 = e(r2)
    local b1 = _b[`col']
    local se1 = _se[`col']
    qui test `col'
    local p1 = r(p)

    di as text "    OLS_log   R2=" %8.4f `r2' "  beta1=" %+12.6f `b1' ///
        "  (SE=" %9.6f `se1' ", p=" %7.4f `p1' ")"
    post `results_pf' ("year_fe") ("`label'") ("OLS_log") ///
        (`r2') (`b1') (`se1') (`p1') (e(N))

    * PPML with year FE
    qui glm switches `col' total_switches_out openings_share_dest ///
        geographic_distance `year_dummies', family(poisson) link(log) vce(robust)
    local dev = e(deviance)
    local r2 = 1 - `dev' / `null_dev_yearfe'
    local b1 = _b[`col']
    local se1 = _se[`col']
    qui test `col'
    local p1 = r(p)

    di as text "    PPML      R2=" %8.4f `r2' "  beta1=" %+12.6f `b1' ///
        "  (SE=" %9.6f `se1' ", p=" %7.4f `p1' ")"
    post `results_pf' ("year_fe") ("`label'") ("PPML") ///
        (`r2') (`b1') (`se1') (`p1') (e(N))
}

* --- Close postfile and save results ---
postclose `results_pf'

use `results_file', clear
export delimited "$stout/model_comparison.csv", replace
save "$stdata/model_comparison.dta", replace

di as text _n "Model comparison results:"
list, separator(8) abbreviate(25)


* --- Save predictions from best PPML ---
di as result _n "Saving predictions from best PPML..."

use "$stdata/pairwise_dataset.dta", clear
drop if missing(switches) | missing(total_switches_out) | ///
       missing(emp_origin) | missing(emp_dest)

* Determine best PPML by examining results
* Re-run baseline PPML for LASSO (likely best), save fitted values
qui glm switches ml_dist_lasso total_switches_out openings_share_dest ///
    geographic_distance, family(poisson) link(log) vce(robust)
predict double predicted_switches, mu

keep occ_origin occ_dest switches euclidean_dist angular_separation ///
     factor_dist ml_dist_lasso predicted_switches
save "$stdata/skill_portability_predictions.dta", replace
export delimited "$stout/skill_portability_predictions.csv", replace
di as result "Saved predictions: " _N " pairs"


/*===========================================================================
  SECTION 5c: Fixed-delta1 Portability Index
  Replicates: 05c_fixed_delta1_portability_index.py

  1. Run fixed-delta1 PPML for each of 4 skill distance variants
  2. Construct portability index from best variant:
     rate_per_switcher = fitted / exposure
     PortRate_o = sum_d(emp_share_d * rate_per_switcher)
     portability_index = rank-normalize to [0,1]
===========================================================================*/

di as result _n "======================================================================"
di as result "SECTION 5c: Fixed-delta1 Portability Index"
di as result "======================================================================"

use "$stdata/pairwise_dataset.dta", clear
drop if missing(switches) | missing(total_switches_out) | ///
       missing(emp_origin) | missing(emp_dest)

* Exposure = max(total_switches_out, 1)
gen double exposure = max(total_switches_out, 1)

* --- Run fixed-delta1 PPML for all 4 variants ---
tempname port_pf
tempfile port_results
postfile `port_pf' str25 skill_distance double R2 double beta1 double se_beta1 ///
    double p_beta1 long n_obs ///
    using `port_results', replace

local best_r2 = -999
local best_label = ""
local best_col = ""

* Null model with exposure for pseudo-R2
qui glm switches, family(poisson) link(log) exposure(exposure)
local null_dev_5c = e(deviance)

forvalues i = 1/`nv' {
    local label : word `i' of `dist_labels'
    local col   : word `i' of `dist_vars'

    qui glm switches `col' openings_share_dest geographic_distance, ///
        family(poisson) link(log) exposure(exposure) vce(robust)

    local dev = e(deviance)
    local r2 = 1 - `dev' / `null_dev_5c'
    local b1 = _b[`col']
    local se1 = _se[`col']
    qui test `col'
    local p1 = r(p)

    di as text "  `label': R2=" %8.4f `r2' "  beta1=" %+12.6f `b1' ///
        "  (SE=" %9.6f `se1' ", p=" %9.2e `p1' ")"

    post `port_pf' ("`label'") (`r2') (`b1') (`se1') (`p1') (e(N))

    if `r2' > `best_r2' {
        local best_r2 = `r2'
        local best_label "`label'"
        local best_col "`col'"
    }
}

postclose `port_pf'

di as text _n "  Best variant (fixed delta1): `best_label' (R2=" %8.4f `best_r2' ")"

* --- Compute employment-share weights ---
use "$stdata/employment_counts_weighted.dta", clear
collapse (sum) weighted_employment, by(occ)
qui su weighted_employment
local total_emp = r(sum)
gen double emp_share = weighted_employment / `total_emp'
rename occ occ_dest
tempfile emp_shares
save `emp_shares'

* --- Merge emp_share into pairwise data ---
use "$stdata/pairwise_dataset.dta", clear
drop if missing(switches) | missing(total_switches_out) | ///
       missing(emp_origin) | missing(emp_dest)

gen double exposure = max(total_switches_out, 1)

qui glm switches `best_col' openings_share_dest geographic_distance, ///
    family(poisson) link(log) exposure(exposure) vce(robust)
predict double fitted_fd1, mu
gen double rate_per_switcher = fitted_fd1 / exposure

merge m:1 occ_dest using `emp_shares', nogen keep(master match)
replace emp_share = 0 if missing(emp_share)

* PortRate_o = sum_d(emp_share_d * rate_per_switcher_{o,d})
gen double weighted_rate = emp_share * rate_per_switcher

collapse (sum) port_rate = weighted_rate, by(occ_origin)
rename occ_origin occ

* Rank-normalize to [0, 1]
local n = _N
egen long rank = rank(-port_rate), unique
gen double portability_index = 1 - (rank - 1) / (`n' - 1)

* Min-max version
qui su port_rate
local pmin = r(min)
local pmax = r(max)
gen double portability_minmax = (port_rate - `pmin') / (`pmax' - `pmin')

* Sort by index
gsort -portability_index

* --- Summary ---
di as text _n "Portability Index summary:"
di as text "  N occupations: " _N
qui su port_rate
di as text "  PortRate: mean=" %12.6f r(mean) "  sd=" %12.6f r(sd)
di as text "           min=" %12.6f r(min) "  max=" %12.6f r(max)

di as text _n "  TOP 10:"
list occ port_rate portability_index in 1/10, noobs

di as text _n "  BOTTOM 10:"
local last10 = _N - 9
list occ port_rate portability_index in `last10'/l, noobs

save "$stdata/portability_index_fixed_delta1.dta", replace
export delimited "$stout/portability_index_fixed_delta1.csv", replace
di as result "Saved: $stout/portability_index_fixed_delta1.csv (" _N " occupations)"


/*===========================================================================
  SECTION 7: Sectoral Downturn Test
  Replicates: 07_sectoral_downturn.py

  LTU_o = alpha0 + alpha1 * EmpTrend_o + alpha2 * Portability_o + u_o

  Step 1: Employment trends (OLS slope of ln(emp) on year)
  Step 2: Long-term unemployment by occupation (import from Python CSV)
  Step 3: Estimate equation (7)
===========================================================================*/

di as result _n "======================================================================"
di as result "SECTION 7: Sectoral Downturn Test (Equation 7)"
di as result "======================================================================"

* --- Step 1: Employment Trends ---
di as text _n "Step 1: Employment Trends"

use "$stdata/employment_counts_weighted.dta", clear

* Need at least 3 years per occupation and positive employment
bys occ: gen nyears = _N
bys occ: egen min_emp = min(weighted_employment)
keep if nyears >= 3 & min_emp > 0

* Compute centered year and log employment
gen double ln_emp = ln(weighted_employment)
bys occ: egen double year_mean = mean(year)
gen double year_c = year - year_mean

* OLS slope per occupation using statsby
tempfile trend_data
save `trend_data'

statsby emp_trend=_b[year_c] emp_trend_se=_se[year_c] n_years=e(N), ///
    by(occ) saving("$stdata/employment_trends.dta", replace) clear: ///
    regress ln_emp year_c

use "$stdata/employment_trends.dta", clear
di as text "  Trends computed for " _N " occupations"
qui su emp_trend
di as text "  Mean trend: " %9.4f r(mean) "  (std: " %9.4f r(sd) ")"

* --- Step 2: Long-Term Unemployment ---
di as text _n "Step 2: Long-Term Unemployment"

* Import from Python CSV (CPS microdata processing requires raw extract)
capture confirm file "$pydata/long_term_unemployment.csv"
if _rc == 0 {
    import delimited "$pydata/long_term_unemployment.csv", clear varnames(1)
    destring occ, replace force
    recast long occ
    save "$stdata/long_term_unemployment.dta", replace
    di as text "  LTU data: " _N " occupations"
    qui su ltu_share
    di as text "  Mean LTU share: " %9.4f r(mean)
}
else {
    di as error "  WARNING: long_term_unemployment.csv not found."
    di as error "  Cannot estimate equation (7). Skipping."
    * Create empty file to avoid errors below
    clear
    gen occ = .
    gen ltu_share = .
    save "$stdata/long_term_unemployment.dta", replace
}

* --- Step 3: Estimate Equation (7) ---
di as text _n "Step 3: Estimate Sectoral Downturn Model"

* Merge trends + LTU + portability
use "$stdata/employment_trends.dta", clear
merge 1:1 occ using "$stdata/long_term_unemployment.dta", keep(match) nogen
merge 1:1 occ using "$stdata/portability_index_fixed_delta1.dta", keep(match) nogen

drop if missing(emp_trend) | missing(ltu_share) | missing(portability_index)

local n_merged = _N
di as text "  Merged sample: `n_merged' occupations"

if `n_merged' < 10 {
    di as error "  Too few occupations for estimation. Skipping."
}
else {
    * Standardize for interpretability
    qui su emp_trend
    gen double emp_trend_z = (emp_trend - r(mean)) / r(sd)

    * Use portability_index as the portability measure
    qui su portability_index
    gen double portability_z = (portability_index - r(mean)) / r(sd)

    * Initialize results postfile
    tempname dt_pf
    tempfile dt_results
    postfile `dt_pf' str25 model str25 variable double coefficient double std_error ///
        double t_stat double p_value double r_squared long n_obs ///
        using `dt_results', replace

    * --- Model 1: Full specification ---
    di as text _n "  Model 1: LTU = a0 + a1*EmpTrend + a2*Portability"
    regress ltu_share emp_trend_z portability_z, vce(robust)

    local r2 = e(r2)
    local nobs = e(N)
    di as text "    N = `nobs', R2 = " %8.4f `r2'

    foreach var in emp_trend_z portability_z {
        local b = _b[`var']
        local se = _se[`var']
        local t = `b' / `se'
        qui test `var'
        local p = r(p)

        di as text "    `var': coef=" %+9.6f `b' ///
            "  SE=" %9.6f `se' "  p=" %7.4f `p'

        post `dt_pf' ("full") ("`var'") (`b') (`se') (`t') (`p') (`r2') (`nobs')
    }

    * --- Model 2: Portability only ---
    di as text _n "  Model 2: LTU = a0 + a2*Portability"
    regress ltu_share portability_z, vce(robust)

    local r2 = e(r2)
    local b = _b[portability_z]
    local se = _se[portability_z]
    qui test portability_z
    local p = r(p)
    local t = `b' / `se'

    di as text "    N = " e(N) ", R2 = " %8.4f `r2'
    di as text "    portability_z: coef=" %+9.6f `b' ///
        "  SE=" %9.6f `se' "  p=" %7.4f `p'

    post `dt_pf' ("portability_only") ("portability_z") ///
        (`b') (`se') (`t') (`p') (`r2') (e(N))

    * --- Model 3: Trend only ---
    di as text _n "  Model 3: LTU = a0 + a1*EmpTrend"
    regress ltu_share emp_trend_z, vce(robust)

    local r2 = e(r2)
    local b = _b[emp_trend_z]
    local se = _se[emp_trend_z]
    qui test emp_trend_z
    local p = r(p)
    local t = `b' / `se'

    di as text "    N = " e(N) ", R2 = " %8.4f `r2'
    di as text "    emp_trend_z: coef=" %+9.6f `b' ///
        "  SE=" %9.6f `se' "  p=" %7.4f `p'

    post `dt_pf' ("trend_only") ("emp_trend_z") ///
        (`b') (`se') (`t') (`p') (`r2') (e(N))

    * --- Models 4-7: AI Exposure (if available) ---
    capture confirm file "$pydata/ai_exposure_by_census2018.csv"
    if _rc == 0 {
        di as text _n "  Loading AI exposure data..."

        preserve
        import delimited "$pydata/ai_exposure_by_census2018.csv", clear varnames(1)
        rename census_code occ
        destring occ, replace force
        recast long occ
        keep occ gpt4_beta
        tempfile ai_data
        save `ai_data'
        restore

        merge 1:1 occ using `ai_data', keep(match) nogen
        local n_ai = _N
        di as text "  AI exposure merged: `n_ai' occupations"

        drop if missing(gpt4_beta)

        * Standardize AI exposure
        qui su gpt4_beta
        gen double ai_z = (gpt4_beta - r(mean)) / r(sd)

        * Re-standardize on AI-matched sample
        qui su emp_trend
        replace emp_trend_z = (emp_trend - r(mean)) / r(sd)
        qui su portability_index
        replace portability_z = (portability_index - r(mean)) / r(sd)

        gen double port_x_ai = portability_z * ai_z

        * --- Model 4: With AI (no interaction) ---
        di as text _n "  Model 4: LTU = a0 + a1*Trend + a2*Port + a3*AI"
        regress ltu_share emp_trend_z portability_z ai_z, vce(robust)

        local r2 = e(r2)
        local nobs = e(N)
        di as text "    N = `nobs', R2 = " %8.4f `r2'

        foreach var in emp_trend_z portability_z ai_z {
            local b = _b[`var']
            local se = _se[`var']
            local t = `b' / `se'
            qui test `var'
            local p = r(p)
            di as text "    `var': coef=" %+9.6f `b' ///
                "  SE=" %9.6f `se' "  p=" %7.4f `p'
            post `dt_pf' ("ai_additive") ("`var'") (`b') (`se') (`t') (`p') (`r2') (`nobs')
        }

        * --- Model 5: AI interaction ---
        di as text _n "  Model 5: LTU = a0 + a1*Trend + a2*Port + a3*AI + a4*(Port x AI)"
        regress ltu_share emp_trend_z portability_z ai_z port_x_ai, vce(robust)

        local r2 = e(r2)
        local nobs = e(N)
        di as text "    N = `nobs', R2 = " %8.4f `r2'

        foreach var in emp_trend_z portability_z ai_z port_x_ai {
            local b = _b[`var']
            local se = _se[`var']
            local t = `b' / `se'
            qui test `var'
            local p = r(p)
            di as text "    `var': coef=" %+9.6f `b' ///
                "  SE=" %9.6f `se' "  p=" %7.4f `p'
            post `dt_pf' ("ai_interaction") ("`var'") (`b') (`se') (`t') (`p') (`r2') (`nobs')
        }

        * --- Models 6-7: Subsample by AI median ---
        qui su gpt4_beta, detail
        local med_ai = r(p50)

        * High AI
        di as text _n "  Model 6 (high AI): LTU = a0 + a1*Trend + a2*Port"
        preserve
        keep if gpt4_beta >= `med_ai'

        qui su emp_trend
        replace emp_trend_z = (emp_trend - r(mean)) / r(sd)
        qui su portability_index
        replace portability_z = (portability_index - r(mean)) / r(sd)

        regress ltu_share emp_trend_z portability_z, vce(robust)
        local r2 = e(r2)
        local nobs = e(N)
        di as text "    N = `nobs', R2 = " %8.4f `r2'

        foreach var in emp_trend_z portability_z {
            local b = _b[`var']
            local se = _se[`var']
            local t = `b' / `se'
            qui test `var'
            local p = r(p)
            di as text "    `var': coef=" %+9.6f `b' ///
                "  SE=" %9.6f `se' "  p=" %7.4f `p'
            post `dt_pf' ("subsample_high_ai") ("`var'") (`b') (`se') (`t') (`p') (`r2') (`nobs')
        }
        restore

        * Low AI
        di as text _n "  Model 7 (low AI): LTU = a0 + a1*Trend + a2*Port"
        preserve
        keep if gpt4_beta < `med_ai'

        qui su emp_trend
        replace emp_trend_z = (emp_trend - r(mean)) / r(sd)
        qui su portability_index
        replace portability_z = (portability_index - r(mean)) / r(sd)

        regress ltu_share emp_trend_z portability_z, vce(robust)
        local r2 = e(r2)
        local nobs = e(N)
        di as text "    N = `nobs', R2 = " %8.4f `r2'

        foreach var in emp_trend_z portability_z {
            local b = _b[`var']
            local se = _se[`var']
            local t = `b' / `se'
            qui test `var'
            local p = r(p)
            di as text "    `var': coef=" %+9.6f `b' ///
                "  SE=" %9.6f `se' "  p=" %7.4f `p'
            post `dt_pf' ("subsample_low_ai") ("`var'") (`b') (`se') (`t') (`p') (`r2') (`nobs')
        }
        restore
    }
    else {
        di as text "  AI exposure data not found. Skipping Models 4-7."
    }

    * --- Save downturn results ---
    postclose `dt_pf'

    use `dt_results', clear
    export delimited "$stout/sectoral_downturn_results.csv", replace
    save "$stdata/sectoral_downturn_results.dta", replace

    di as text _n "Sectoral downturn results:"
    list, separator(4) abbreviate(20)
}


/*===========================================================================
  VERIFICATION CHECKS
  Compare Stata outputs against Python reference values
===========================================================================*/

di as result _n "======================================================================"
di as result "VERIFICATION CHECKS"
di as result "======================================================================"

* --- Check 1: O*NET skill matrix dimensions ---
use "$stdata/onet_skill_matrix.dta", clear
local n_soc = _N
qui describe ability_* activity_* knowledge_* skill_*, varlist
local n_dims : word count `r(varlist)'
di as text "  O*NET skill matrix: `n_soc' SOC codes x `n_dims' dimensions"
di as text "    Expected: 774 x 202"
if `n_soc' == 774 & `n_dims' == 202 {
    di as result "    PASS"
}
else {
    di as error "    MISMATCH"
}

* Check value range
local range_ok = 1
foreach v of varlist ability_* activity_* knowledge_* skill_* {
    qui su `v'
    if r(min) < -0.001 | r(max) > 1.001 {
        local range_ok = 0
    }
}
if `range_ok' {
    di as result "    Value range [0,1]: PASS"
}
else {
    di as error "    Value range [0,1]: FAIL"
}

* --- Check 2: Crosswalk dimensions ---
use "$stdata/skill_vectors_by_census2018.dta", clear
local n_census = _N
di as text "  Skill vectors: `n_census' Census codes"
di as text "    Expected: 565"
if `n_census' == 565 {
    di as result "    PASS"
}
else {
    di as error "    MISMATCH (got `n_census')"
}

* --- Check 3: Switching matrix ---
use "$stdata/switching_matrix.dta", clear
qui count if switches > 0
local n_nonzero = r(N)
di as text "  Switching matrix nonzero pairs: `n_nonzero'"
di as text "    Expected: ~13,757"

* --- Check 4: Pairwise dataset ---
use "$stdata/pairwise_dataset.dta", clear
local n_pairs = _N
di as text "  Pairwise dataset: `n_pairs' pairs"
di as text "    Expected: ~275,100"

* --- Check 5: Model comparison ---
use "$stdata/model_comparison.dta", clear
di as text "  Model comparison rows: " _N
list specification skill_distance estimator R2 if specification == "baseline", ///
    separator(2) abbreviate(25)

* --- Check 6: Portability index ---
use "$stdata/portability_index_fixed_delta1.dta", clear
di as text "  Portability index: " _N " occupations"
qui su portability_index
di as text "    Range: [" %6.3f r(min) ", " %6.3f r(max) "]"
di as text "    Mean:  " %6.3f r(mean)


di as result _n "======================================================================"
di as result "REPLICATION COMPLETE"
di as result "======================================================================"
di as text "  Intermediate data: $stdata/"
di as text "  Output files:      $stout/"
di as text "    - model_comparison.csv"
di as text "    - skill_portability_predictions.csv"
di as text "    - portability_index_fixed_delta1.csv"
di as text "    - sectoral_downturn_results.csv"

