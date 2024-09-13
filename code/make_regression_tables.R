make_main_regression_tables <- function() {
    allreg = readRDS('estimates/regressions.rds')

    # Adapt
    setFixest_dict(c(gdppc_growth = "GDP pc growth",
                     decile_pcvalue_growth = "Decile income growth",
                     temperature_mean = "Temperature",
                     avg_y_country = "avg. GDP",
                     "temperature_mean2" = "Temperature, Squared",
                     "temperature_mean:l1_lgdppc" = "Temperature X GDP($t-1$)",
                     "temperature_mean2:l1_lgdppc" = "Temperature Sq. X GDP($t-1$)",
                     iso3 = "Country",
                     year = "Year"))

    etable(allreg[c(paste0("adapt-", c(paste0("D", 1:10), "GDP")))], keep = "%temperature",
           # title = "BHM-Adapt specification",
           file = "tables/adapt.tex",
           replace = T)

    # BHM
    setFixest_dict(c(gdppc_growth = "GDP pc growth",
                     decile_pcvalue_growth = "Decile income growth",
                     temperature_mean = "Temperature",
                     temperature_mean2 = "Temperature, Squared",
                     iso3 = "Country",
                     year = "Year"))

    etable(allreg[c(paste0("BHM-", c(paste0("D", 1:10), "GDP")))], keep = "%temperature",
           # title = "Burke, Hsiang and Miguel (2015) specification",
           file = "tables/bhm.tex",
           replace = T)

    # KW
    setFixest_dict(c(gdppc_growth = "GDP pc growth",
                     decile_pcvalue_growth = "Decile income growth",
                     diff_temp = "$\\Delta$temperature",
                     lag_diff_temp = "$\\Delta$temperature, t-1",
                     "diff_temp:lag_temperature_mean" = "$\\Delta$temperature, t *temperature, t-1",
                     "lag_diff_temp:lag_temperature_mean" = "$\\Delta$temperature, t-1 *temperature, t-1",
                     iso3 = "Country",
                     year = "Year"))

    etable(allreg[c(paste0("KW-", c(paste0("D", 1:10), "GDP")))], keep = c("%diff_temp", "%lag_diff_temp"),
           # title = "Kalkuhl and Wenz (2020) specification",
           file = "tables/kw.tex",
           replace = T)

    # Adapt without l1_growth
    setFixest_dict(c(gdppc_growth = "GDP pc growth",
                     decile_pcvalue_growth = "Decile income growth",
                     temperature_mean = "Temperature",
                     avg_y_country = "avg. GDP",
                     "I(temperature_mean^2)" = "Temperature, Squared",
                     "temperature_mean:l1_lgdppc" = "Temperature X GDP($t-1$)",
                     "I(temperature_mean^2):l1_lgdppc" = "Temperature Sq. X GDP($t-1$)",
                     iso3 = "Country",
                     year = "Year"))

    etable(allreg[c(paste0("adapt-no_l1_growth-", c(paste0("D", 1:10), "GDP")))], keep = "%temperature",
           # title = "BHM-Adapt specification, no lagged income growth",
           file = "tables/adapt-no_l1_growth.tex",
           replace = T)

    # BHM without l1_growth
    setFixest_dict(c(gdppc_growth = "GDP pc growth",
                     decile_pcvalue_growth = "Decile income growth",
                     temperature_mean = "Temperature",
                     temperature_mean2 = "Temperature, Squared",
                     iso3 = "Country",
                     year = "Year"))

    etable(allreg[c(paste0("BHM-no_l1_growth-", c(paste0("D", 1:10), "GDP")))], keep = "%temperature",
           # title = "Burke, Hsiang and Miguel (2015) specification, no lagged income growth",
           file = "tables/bhm-no_l1_growth.tex",
           replace = T)

    # KW
    setFixest_dict(c(gdppc_growth = "GDP pc growth",
                     decile_pcvalue_growth = "Decile income growth",
                     diff_temp = "$\\Delta$temperature",
                     lag_diff_temp = "$\\Delta$temperature, t-1",
                     "diff_temp:lag_temperature_mean" = "$\\Delta$temperature, t *temperature, t-1",
                     "lag_diff_temp:lag_temperature_mean" = "$\\Delta$temperature, t-1 *temperature, t-1",
                     iso3 = "Country",
                     year = "Year"))

    etable(allreg[c(paste0("KW-no_l1_growth-", c(paste0("D", 1:10), "GDP")))], keep = c("%diff_temp", "%lag_diff_temp"),
           # title = "Kalkuhl and Wenz (2020) specification, no lagged income growth",
           file = "tables/kw-no_l1_growth.tex",
           replace = T)

}

make_regression_tables_reconstructed_dummy <- function() {
    allreg = readRDS('estimates/robustness-reconstructed_dummy.rds')

    setFixest_dict(c(gdppc_growth = "GDP pc growth",
                     decile_pcvalue_growth = "Decile income growth",
                     temperature_mean = "Temperature",
                     avg_y_country = "avg. GDP",
                     "temperature_mean2" = "Temperature, Squared",
                     "temperature_mean:l1_lgdppc" = "Temperature X GDP($t-1$)",
                     "temperature_mean2:l1_lgdppc" = "Temperature Sq. X GDP($t-1$)",
                     iso3 = "Country",
                     year = "Year",
                     raw = "Reconstructed share"))

    etable(allreg[c(paste0("reconstructed_dummy-", c(paste0("D", 1:10), "GDP")))],
           keep = "%temperature",
           # title = "Including fixed effects for recunstructed shares",
           file = "tables/adapt_reconstructed_dummy.tex",
           replace = T)

    setFixest_dict(c(gdppc_growth = "GDP pc growth",
                     decile_pcvalue_growth = "Decile income growth",
                     temperature_mean = "Temperature",
                     avg_y_country = "avg. GDP",
                     "temperature_mean2" = "Temperature, Squared",
                     "temperature_mean:l1_lgdppc" = "Temperature X GDP($t-1$)",
                     "temperature_mean2:l1_lgdppc" = "Temperature Sq. X GDP($t-1$)",
                     "temperature_mean:l1_lgdppc:raw" = "Temperature X GDP($t-1$) X Reconstructed share",
                     "temperature_mean2:l1_lgdppc:raw" = "Temperature Sq. X GDP($t-1$) X Reconstructed share",
                     iso3 = "Country",
                     year = "Year",
                     l1_lgdppc = "GDP($t-1$)",
                     raw = "Reconstructed share"))

    etable(allreg[c(paste0("reconstructed_dummy_interact-", c(paste0("D", 1:10), "GDP")))],
           keep = "%temperature",
           # title = "Including fixed effects for recunstructed shares",
           file = "tables/adapt_reconstructed_dummy_interact.tex",
           replace = T)
}