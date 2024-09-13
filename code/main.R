options(max.print = 150)
options(width = 5000)
options(length = 50)

if (!require("pacman")) install.packages("pacman")
pacman::p_load(arrow, fixest, here, readr, readxl, zoo, data.table, margins, marginaleffects, imputeTS, tidyverse, plm)

setwd(here())

for (folder in c('data', 'data/out', 'data/out/sensitivity',
                 'data/out/projections', 'data/out/rice-simulations', 'data/tmp', 'data/tmp/temp_data', 'data/tmp/gdp_data',
                 'data/tmp/baseline_trajectories', 'estimates', 'estimates/coef', 'estimates/vcov', 'img', 'tables'
                 )) {
    ifelse(!dir.exists(folder), dir.create(folder), FALSE)
}

# Clean start. Remove intermediate files
clean <- FALSE
if (clean) {
    unlink("data/out/rice-simulations/*")  # Results of RICE runs varying ECS.
    unlink("data/out/sensitivity/*")  # Results of projections sensitivity analysis
    unlink("data/tmp/baseline_trajectories/income_ssp*")  # Baseline income trajectories by ssp and decile/GDP
    unlink("estimates/coef/*")  # regression coefficients
    unlink("estimates/vcov/*")  # regression VCOV matrix
    unlink("data/tmp/temp_data/*.parquet")  # Temperature projections varying ECS
    unlink("data/tmp/gdp_data/*.parquet")  # GDP-level income projections (with damages) varying ECS
}

source(here("code", "load_data_functions.R"))

# Load desired data and merge in one database
gdp_data <- full_join(
  #| WB_ppp | WB_mer | WB_mer_update | WB_ppp_fra | PWT |
  load_data_gdp(gdp_source = "WB_mer_update"),
  #| UDel | CRU | CRU_update |
  load_data_weather(weather_source = "CRU_update"),
  by = c("iso3", "year"))

dec_rec <- left_join(
  #| reconstructed_shares | linear_interpolation | stineman_interpolation | no_interpolation |
  load_data_deciles(deciles_source = "reconstructed_shares"),
  gdp_data,
  by = c("iso3", "year")) %>%
  filter(year >= 1960 & year <= 2020)

source(here('code/regression_functions.R'))
source(here('code/make_regression_tables.R'))

run_regressions(globaldata = dec_rec, rm_outliers = F)
make_main_regression_tables()
reg_robustness_reconstructed_shares()
make_regression_tables_reconstructed_dummy()