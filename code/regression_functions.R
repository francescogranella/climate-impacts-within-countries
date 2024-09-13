library(glue)
library(arrow)

get_temp <- function() {
    return(seq(-5, 35, 0.05))
}

get_precip <- function(globaldata) {
    return(mean(globaldata$precip, na.rm = T))
}

bind_temp <- function(x) {
    temp <- get_temp()
    x <- x %>% cbind(temp)
    return(x)
}

calculate_quantiles <- function(data, column_name, probs = seq(0.01, 1, 0.01)) {
    # Extract the specified column from the data
    column <- data[[column_name]]
    # Calculate quantiles for the specified column
    quantiles <- data.frame(quantile(column, probs, na.rm = TRUE))
    return(quantiles)
}

make_regression_data <- function(globaldata, decile, rm_outliers = F) {
    # Create a dataset ready for regression
    data <- globaldata %>%
      group_by(iso3) %>%
      arrange(year) %>%
      mutate(decile_pcvalue = log(get(decile) * exp(lgdppc)),
             l1_decile_pcvalue = dplyr::lag(decile_pcvalue),
             decile_pcvalue_growth = decile_pcvalue - dplyr::lag(decile_pcvalue),
             l1_growth = dplyr::lag(decile_pcvalue_growth)) %>%  #n.b. l1_growth here is first lag of decile-y growth (diff for gdppc)
      filter(!is.na(l1_growth))  # Ensure all specifications have the same sample
    if (rm_outliers == T) {
        quantiles <- calculate_quantiles(data, 'decile_pcvalue_growth')
        data %>% filter(decile_pcvalue_growth > quantiles[1,] & decile_pcvalue_growth < quantiles[100,])
    }
    return(data)
}

make_regression_country_data <- function(globaldata, rm_outliers = F) {
    # Create country-level dataset ready for regression
    data <- globaldata %>%
      group_by(iso3) %>%
      arrange(year) %>%
      mutate(l1_growth = dplyr::lag(gdppc_growth)) %>%
      filter(!is.na(l1_growth) &
               year > 1968 &
               !(iso3 == "MAR" & year == 1999) &
               !(iso3 == "ROU" & year == 1994))  # Ensure all specifications have the same sample
    if (rm_outliers == T) {
        quantiles <- calculate_quantiles(data, 'gdppc_growth')
        data %>% filter(gdppc_growth > quantiles[1,] & gdppc_growth < quantiles[100,])
    }
    return(data)
}


make_pred <- function(temp, precip, reg_results, iso3, year, l1_growth = 0.2, l1_lgdppc = NaN) {

    # Build synthetic data for given country-year..
    newdata <- data.frame(temperature_mean = temp, temperature_mean2 = temp^2,
                          precip = precip,
                          l1_growth = l1_growth,
                          l1_lgdppc = l1_lgdppc,
                          iso3 = iso3, year = year) # choose random ones

    # ..to predict GDP growth at varying temperature levels
    pred <- as_tibble(predict(reg_results, newdata = newdata,
                              interval = "confidence", level = 0.9))

    # Shift predicted GDP growth so that it has max at zero
    pred <- pred %>%
      mutate(fit_shift = fit - max(pred$fit, na.rm = T),
             lwr_shift = lwr - max(pred$fit, na.rm = T),
             upr_shift = upr - max(pred$fit, na.rm = T)) %>% # to recenter
      select(-fit, -lwr, -upr)
    return(pred)
}

run_regressions <- function(globaldata, rm_outliers = F) {
    # Function to run multiple regressions without and with adaptation, and store results
    # 'dynamic' chooses whether regression includes lagged growth.
    # 'adaptation' chooses whethet to interact temperature with GDP

    temp <- get_temp()
    precip <- get_precip(globaldata)

    formulas <- list()
    formulas[['adapt']] <- 'decile_pcvalue_growth ~ l1_growth + precip + I(precip^2) + temperature_mean + temperature_mean2 + temperature_mean:l1_lgdppc + temperature_mean2:l1_lgdppc + precip:l1_lgdppc + I(precip^2):l1_lgdppc | iso3 + year + iso3[year]'
    formulas[['adapt_lsdv']] <- 'decile_pcvalue_growth ~ l1_growth + precip + I(precip^2) + temperature_mean + temperature_mean2 + temperature_mean:l1_lgdppc + temperature_mean2:l1_lgdppc + precip:l1_lgdppc + I(precip^2):l1_lgdppc + factor(iso3) + factor(year) + factor(iso3) * year + factor(iso3) * I(year^2)'
    formulas[['BHM']] <- 'decile_pcvalue_growth ~ l1_growth + precip + I(precip^2) + temperature_mean + temperature_mean2 | iso3 + year + iso3[year] + iso3[year^2]'
    formulas[['KW']] <- 'decile_pcvalue_growth ~ l1_growth + diff_temp + lag_diff_temp + diff_temp:lag_temperature_mean + lag_diff_temp:lag_temperature_mean + lag_temperature_mean + I(lag_temperature_mean^2) + diff_prec + lag_diff_prec + diff_prec:lag_precip + lag_diff_prec:lag_precip + lag_precip + I(lag_precip^2) | iso3 + year + iso3[year]'
    formulas[['adapt-no_l1_growth']] <- 'decile_pcvalue_growth ~ precip + I(precip^2) + temperature_mean + temperature_mean2 + temperature_mean:l1_lgdppc + temperature_mean2:l1_lgdppc + precip:l1_lgdppc + I(precip^2):l1_lgdppc | iso3 + year + iso3[year]'
    formulas[['adapt_lsdv-no_l1_growth']] <- 'decile_pcvalue_growth ~ precip + I(precip^2) + temperature_mean + temperature_mean2 + temperature_mean:l1_lgdppc + temperature_mean2:l1_lgdppc + precip:l1_lgdppc + I(precip^2):l1_lgdppc + factor(iso3) + factor(year) + factor(iso3) * year + factor(iso3) * I(year^2)'
    formulas[['BHM-no_l1_growth']] <- 'decile_pcvalue_growth ~ precip + I(precip^2) + temperature_mean + temperature_mean2 | iso3 + year + iso3[year] + iso3[year^2]'
    formulas[['KW-no_l1_growth']] <- 'decile_pcvalue_growth ~ diff_temp + lag_diff_temp + diff_temp:lag_temperature_mean + lag_diff_temp:lag_temperature_mean + lag_temperature_mean + I(lag_temperature_mean^2) + diff_prec + lag_diff_prec + diff_prec:lag_precip + lag_diff_prec:lag_precip + lag_precip + I(lag_precip^2) | iso3 + year + iso3[year]'

    income_segments <- paste0("D", seq(1, 10))
    income_segments[11] <- 'GDP'

    allreg <- list()
    allpred <- list()
    allpred_usa <- list()
    allpred_mex <- list()
    allpred_ken <- list()
    for (decile in income_segments) {

        if (decile == 'GDP') {
            data <- make_regression_country_data(globaldata, rm_outliers = rm_outliers)
        } else {
            data <- make_regression_data(globaldata, decile, rm_outliers = rm_outliers)
        }
        library(arrow)
        arrow::write_parquet(data, paste0('data/out/data_', decile, '.parquet'))

        for (formula_name in names(formulas)) {
            formula <- formulas[[formula_name]]
            lsdv <- grepl('lsdv', formula_name)

            if (decile == 'GDP') {
                formula <- gsub('decile_pcvalue_growth', 'gdppc_growth', formula)
            }

            formula <- as.formula(formula)
            spec_name <- paste(formula_name, decile, sep = '-')

            if (lsdv) {
                reg_results <- lm(formula, data = data)
                allreg[[spec_name]] <- reg_results

                allpred[[spec_name]] <- make_pred(temp, precip, reg_results, 'USA', 2000)
                allpred_usa[[spec_name]] <- make_pred(temp, mean(globaldata$precip[globaldata$iso3 == "USA"], na.rm = T), reg_results, 'USA', 2000, l1_lgdppc = quantile(globaldata$lgdppc, seq(0.25, 1, 0.25), na.rm = T)["75%"])
                allpred_mex[[spec_name]] <- make_pred(temp, mean(globaldata$precip[globaldata$iso3 == "MEX"], na.rm = T), reg_results, 'USA', 2000, l1_lgdppc = quantile(globaldata$lgdppc, seq(0.25, 1, 0.25), na.rm = T)["50%"])
                allpred_ken[[spec_name]] <- make_pred(temp, mean(globaldata$precip[globaldata$iso3 == "KEN"], na.rm = T), reg_results, 'USA', 2000, l1_lgdppc = quantile(globaldata$lgdppc, seq(0.25, 1, 0.25), na.rm = T)["25%"])
            } else {
                reg_results <- feols(formula, data = data)
                allreg[[spec_name]] <- reg_results
            }
        }
    }

    # Save regression results and predictions
    saveRDS(allreg, file = 'estimates/regressions.rds')
    saveRDS(allpred, file = 'estimates/predictions.rds')
    saveRDS(allpred_usa, file = 'estimates/predictions_usa.rds')
    saveRDS(allpred_mex, file = 'estimates/predictions_mex.rds')
    saveRDS(allpred_ken, file = 'estimates/predictions_ken.rds')

    # Export coefficients and VCOV to parquet
    for (spec_name in names(allreg)) {
        reg_result = allreg[[spec_name]]
        lsdv <- grepl('lsdv', spec_name)
        if (lsdv == FALSE) {
            coeftable = reg_result$coeftable
            coeftable$varname <- row.names(coeftable)
            vcov = as.data.frame(reg_result$cov.iid)
            vcov$varname <- row.names(vcov)
        } else {
            coeftable <- as.data.frame(coefficients(reg_result))
            vcov <- as.data.frame(vcov(reg_result))
        }
        write_parquet(coeftable, paste0('estimates/coef/', spec_name, '.parquet'))
        write_parquet(vcov, paste0('estimates/vcov/', spec_name, '.parquet'))
    }

}


reg_robustness_reconstructed_shares <- function() {

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

    dec_raw <- load_data_deciles(deciles_source = "no_interpolation") %>%
      select(iso3, year) %>%
      na.omit()
    dec_raw['raw'] = 1

    decile_data_compare <- full_join(dec_rec, dec_raw)
    decile_data_compare$raw[is.na(decile_data_compare$raw)] <- 0

    formulas <- list()
    formulas[['reconstructed_dummy']] <- 'decile_pcvalue_growth ~ l1_growth + precip + I(precip^2) + temperature_mean + temperature_mean2 + temperature_mean:l1_lgdppc + temperature_mean2:l1_lgdppc + precip:l1_lgdppc + I(precip^2):l1_lgdppc | iso3 + year + iso3[year] + iso3^raw + year^raw'
    formulas[['reconstructed_dummy_interact']] <- 'decile_pcvalue_growth ~ l1_growth + precip + I(precip^2) + temperature_mean + temperature_mean2 + temperature_mean:l1_lgdppc + temperature_mean2:l1_lgdppc + precip:l1_lgdppc + I(precip^2):l1_lgdppc + temperature_mean:raw + temperature_mean2:raw + temperature_mean:l1_lgdppc:raw + temperature_mean2:l1_lgdppc:raw | iso3 + year + iso3[year]'
    income_segments <- paste0("D", seq(1, 10))
    income_segments[11] <- 'GDP'
    allreg <- list()
    for (decile in income_segments) {
        # write_dta(data, r"(C:\Users\Granella\Dropbox (CMCC)\PhD\Research\inequality-impacts-econometrics\tmp\test.dta)")
        for (formula_name in names(formulas)) {
            formula <- formulas[[formula_name]]
            spec_name <- paste(formula_name, decile, sep = '-')
            if (decile == 'GDP') {
                data <- make_regression_country_data(decile_data_compare, rm_outliers = F)
                formula <- gsub('decile_pcvalue_growth', 'gdppc_growth', formula)
            }    else {
                data <- make_regression_data(decile_data_compare, decile, rm_outliers = F)
            }
            formula <- as.formula(formula)
            allreg[[spec_name]] <- feols(formula, data = data)
        }
    }
    saveRDS(allreg, file = 'estimates/robustness-reconstructed_dummy.rds')

    for (decile in names(allreg)) {
        reg_result = allreg[[decile]]
        coeftable = reg_result$coeftable
        coeftable$varname <- row.names(coeftable)
        vcov = as.data.frame(reg_result$cov.iid)
        vcov$varname <- row.names(vcov)
        write_parquet(coeftable, paste0('estimates/coef/', spec_name, '.parquet'))
        write_parquet(vcov, paste0('estimates/vcov/', spec_name, '.parquet'))
    }

}
