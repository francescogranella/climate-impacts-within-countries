require(haven)
require(fixest)
require(tidyverse)
require(here)
require(readr)
require(readxl)
require(zoo)
require(plm)
require(data.table)
require(margins)
require(marginaleffects)
require(imputeTS)

# Load country-level weather data
load_data_weather <- function(weather_source = "CRU_update"){
  if(weather_source == "UDel") {
    
    temp_data <- read.csv(here("data", "WB_UDel.csv")) %>% 
      select(countrycode, year, UDel_pop_preci, UDel_pop_temp) %>% 
      rename(
        iso3 = countrycode,
        temperature_mean = UDel_pop_temp,
        precip = UDel_pop_preci
      ) 
    
  } else if(weather_source == "CRU") {
    
    load(here("data", "climate_dataset_iso3_year_historical_pop2000weighted.Rdata"))
    
    temp_data <- climate_dataset_iso3_year
    rm(climate_dataset_iso3_year)
    
    temp_data <- temp_data %>% 
      rename(
        precip = precipitation_mean
      ) %>% 
      select(iso3, year, temperature_mean, precip)
    
  } else if(weather_source == "CRU_update") {
    
    load(here("data", "climate_dataset_iso3_year_2022_popdens2000weighted.Rdata"))
    
    temp_data <- climate_dataset_iso3_year
    rm(climate_dataset_iso3_year)
    
    temp_data <- temp_data %>% 
      rename(
        precip = precipitation_mean
      ) %>% 
      select(iso3, year, temperature_mean, precip)
  } else {stop("Please select a valid source")}

  temp_data <- temp_data %>% 
  group_by(iso3) %>% 
  mutate(temperature_mean2 = temperature_mean^2,
         diff_temp = temperature_mean - dplyr::lag(temperature_mean),
         lag_diff_temp = dplyr::lag(diff_temp),
         lag_temperature_mean = dplyr::lag(temperature_mean),
         diff_prec = precip - dplyr::lag(precip),
         lag_diff_prec = dplyr::lag(diff_prec),
         lag_precip = dplyr::lag(precip)) 

return(temp_data)
}

# Load country-level GDP data
load_data_gdp <- function(gdp_source = "WB_mer_update"){
  if(gdp_source == "WB_ppp") {
    
    gdp_data <- read.csv(here("data", "WB_gdp_cap_ppp_2017usd.csv"),
                         skip = 4) %>% 
      rename(countryname = Country.Name,
             iso3 = Country.Code) %>% 
      select(-Indicator.Code, -Indicator.Name, -X) %>% 
      pivot_longer(
        cols = "X1960":"X2022",
        names_to = "year",
        values_to = "gdppc"
      ) %>% 
      group_by(iso3) %>% 
      mutate(
        year = as.numeric(substr(year, start = 2, stop = 5)),
        lgdppc = log(gdppc),
        l1_lgdppc = dplyr::lag(lgdppc),
        gdppc_growth = lgdppc - dplyr::lag(lgdppc)
      ) 
  } else if(gdp_source == "WB_mer") {
    
    gdp_data <- read_dta(here("data", "GrowthClimateDataset.dta")) %>% 
      select(iso, year, gdpCAP_wdi, growthWDI) %>% 
      rename(
        iso3 = iso,
        gdppc_growth = growthWDI
      ) %>% 
      group_by(iso3) %>% 
      mutate(
        lgdppc = log(gdpCAP_wdi),
        l1_lgdppc = dplyr::lag(lgdppc)
      )
    
  } else if(gdp_source == "WB_mer_update") {
    
    gdp_data <- read.csv(here("data", "WB_gdpcap_2015usd.csv"),
                         skip = 4) %>% 
      rename(countryname = Country.Name,
             iso3 = Country.Code) %>% 
      select(-Indicator.Code, -Indicator.Name, -X) %>% 
      pivot_longer(
        cols = "X1960":"X2022",
        names_to = "year",
        values_to = "gdppc"
      ) %>% 
      group_by(iso3) %>% 
      mutate(
        year = as.numeric(substr(year, start = 2, stop = 5)),
        lgdppc = log(gdppc),
        l1_lgdppc = dplyr::lag(lgdppc),
        gdppc_growth = lgdppc - dplyr::lag(lgdppc)
      ) 
    
  } else if(gdp_source == "WB_ppp_fra") {
    
    gdp_data <- read.csv(here("data", "gdp_interpolated.csv")) %>% 
      rename(
        iso3 = countrycode,
        lgdppc = lppppred
      ) %>% 
      select(iso3, year, lgdppc) %>% 
      group_by(iso3) %>% 
      mutate(
        l1_lgdppc = dplyr::lag(lgdppc),
        gdppc_growth = lgdppc - dplyr::lag(lgdppc)
      )
  } else if(gdp_source == "PWT") {
    
    gdp_data <- read_excel(here("data", "pwt100.xlsx"),
                           sheet = "Data") %>% 
      rename(
        iso3 = countrycode
      ) %>% 
      select(iso3, year, rgdpo, pop) %>% 
      group_by(iso3) %>% 
      mutate(
        gdppc = rgdpo/pop,
        lgdppc = log(gdppc),
        l1_lgdppc = dplyr::lag(lgdppc),
        gdppc_growth = lgdppc - dplyr::lag(lgdppc)
      ) 
  } else {stop("Please select a valid source")}
return(gdp_data)
}

# Load country-level decile data
load_data_deciles <- function(deciles_source = "reconstructed_shares"){
  ## Merge country-level data with linearly interpolated shares
  if (deciles_source == "linear_interpolation") {
  
    deciles_data <- full_join(
                          read_dta("data/global_inequality_data.dta"),
                          fread("data/inequality_dataset_analyzed_deciles.csv") %>% pivot_wider(id_cols = c(year, iso3), names_from = dist),
                  by = c("iso3","year")
                  ) %>%
      group_by(iso3) %>%
      arrange(year) %>% 
      rename_with(toupper, starts_with("d")) %>% 
      select(iso3, year, D1:D10)
  
  ## Merge country-level data with historical reconstructed shares
  } else if (deciles_source == "reconstructed_shares"){

    deciles_data <- read.csv(here("data", "Final_Historical_data_ISO.csv")) %>% 
      rename(iso3 = iso) %>% 
      mutate(Category = toupper(Category),
             iso3 = toupper(iso3),
             dec_share = 100*Income..net.) %>% 
      select(iso3, year, Category, dec_share) %>% 
      pivot_wider(
        names_from = "Category",
        values_from = "dec_share"
      )

  ## Merge country-level data with Stineman-interpolated decile shares    
  } else if (deciles_source == "stineman_interpolation") {
      
    wiid_df <- read_excel(here("data", "../data/in/WIID_30JUN2022.xlsx")) %>%
      dplyr::rename(iso3 = c3) %>% 
      select(iso3, year, d1:d10, resource) %>%
      filter(resource == "Income (net)") %>% 
      mutate(unique_id = paste(iso3, year, sep = "_")) %>% 
      group_by(unique_id) %>% 
      slice_head(n=1) %>% 
      ungroup() %>% 
      select(-unique_id) %>% 
      rename_with(toupper, starts_with("d")) %>% 
      #First filter wiid_df to after 1960, so don't interpolate away from rest of data
      filter(year >= 1960) 
    
    df_exp <- wiid_df %>%       
      na.omit() %>% 
      group_by(iso3) %>% 
      summarise(n_obs = n())
    
    wiid_df <- wiid_df %>% 
      full_join(df_exp, by = "iso3") %>% 
      #Balance time dimension of panel by filling implicit missing with NAs
      complete(nesting(iso3, n_obs), year = full_seq(year, period = 1)) %>% 
      select(iso3, year, everything())
    
    ## Interpolate with Stineman algorithm
    stine_int <- wiid_df
    for (i in unique(stine_int$iso3)) {
      stine_int[stine_int$iso3 == i,] <- na_interpolation(stine_int[stine_int$iso3==i,],
                                                          option = "stine")
    }
    
    deciles_data <- stine_int %>%
      select(iso3,year,D1:D10)

  } else if (deciles_source == "no_interpolation") {
    
    deciles_data <- read_excel(here("data", "../data/in/WIID_30JUN2022.xlsx")) %>%
      dplyr::rename(iso3 = c3) %>% 
      select(iso3, year, d1:d10, resource) %>%
      filter(resource == "Income (net)") %>% 
      mutate(unique_id = paste(iso3, year, sep = "_")) %>% 
      group_by(unique_id) %>% 
      slice_head(n=1) %>% 
      ungroup() %>% 
      select(-unique_id) %>% 
      rename_with(toupper, starts_with("d")) %>% 
      filter(year >= 1960)
    
  } else {stop("Please select a valid source")}
return(deciles_data)
}

## Load projected decile shares from GCAM group
load_data_decile_projections <- function(deciles_source="Narayan_et_al_2023"){
  if (deciles_source == "Narayan_et_al_2023") {
    
    dec_proj <- read.csv(here("data/in", "ISO_level_projections_PC_model_projections.csv"))
    
    #keep only years that are multiple of 5 (to have 5-yr steps as in RICE50+)
    dec_proj <- dec_proj[(dec_proj$year %% 5) == 0,]
    
    # make names compatible
    dec_proj <- dec_proj %>% 
      mutate(
        dist = toupper(Category),
        n = tolower(ISO),
        t = (year - 2015)/5 + 1
      ) %>% 
      rename(
        value = pred_shares,
        ssp = sce
      ) %>% 
      select(ssp, t, n, dist, value)
  } else {stop("Please select a valid source")}
return(dec_proj)  
}