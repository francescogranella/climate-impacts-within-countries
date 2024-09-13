require(tidyverse)
require(ggpubr)
require(plm)
require(ggrepel)
library(here)
require(ggpubr)
require(latex2exp)
options(scipen = 999)

# Load data
data_ela <- read.csv(paste0(here(), "/data/out/projections/SSP3_adapt.csv"))
#declare as panel
data_ela <- pdata.frame(data_ela, index = c("n", "t"))
rownames(data_ela) <- NULL

#generate countryyear
data_ela <- data_ela %>% mutate(countryyear = paste0(n, t))

summary(lm("log(damages) ~ 1 + log(income) + factor(t) + factor(n)", data = data_ela %>% filter(!is.na(damages) & !is.na(income) & damages != 0 & income != 0)))

#mean from monte carlo analysis of all elasticities
xi_pos_within <- 1.01
xi_pos_between <- 1.15
xi_pos_total <- 0.81
xi_neg_within <- 0.64
xi_neg_between <- 0.91
xi_neg_total <- 0.52

#where to show labels
label_x_pos <- 2
label_x_neg <- -2





#try to combine damages with benefits: within country
data_within_demeaned <- data_ela %>%
  filter(income > 0 & damages > 0) %>%
  group_by(countryyear) %>%
  mutate(damages_demeaned = log(damages) - mean(log(damages)), income_demeaned = log(income) - mean(log(income))) %>%
  ungroup() %>%
  select(countryyear, damages_demeaned, income_demeaned, income, pop, t, n)

#negative damages
data_within_demeaned_neg <- data_ela %>%
  filter(income > 0 & damages < 0) %>%
  group_by(countryyear) %>%
  mutate(damages_demeaned = log(-damages) - mean(log(-damages)), income_demeaned = log(income) - mean(log(income))) %>%
  ungroup() %>%
  select(countryyear, damages_demeaned, income_demeaned, income, t, n)

#between country benefits and damages
data_within_demeaned_between <- data_ela %>%
  filter(income > 0 & gdp_damages > 0) %>%
  group_by(t) %>%
  mutate(damages_demeaned = log(gdp_damages) - mean(log(gdp_damages)), income_demeaned = log(gdp) - mean(log(gdp))) %>%
  ungroup() %>%
  select(pop, gdp, damages_demeaned, income_demeaned, t, n) %>%
  distinct()
data_within_demeaned_between_neg <- data_ela %>%
  filter(income > 0 & gdp_damages < 0) %>%
  group_by(t) %>%
  mutate(damages_demeaned = log(-gdp_damages) - mean(log(-gdp_damages)), income_demeaned = log(gdp) - mean(log(gdp))) %>%
  ungroup() %>%
  select(pop, gdp, damages_demeaned, income_demeaned, t, n) %>%
  distinct()

#total distribution
data_within_demeaned_total <- data_ela %>%
  filter(income > 0 & damages > 0) %>%
  group_by(t) %>%
  mutate(damages_demeaned = log(damages) - mean(log(damages)), income_demeaned = log(income) - mean(log(income))) %>%
  ungroup() %>%
  select(pop, gdp, income, damages_demeaned, income_demeaned, t, n) %>%
  distinct()
data_within_demeaned_total_neg <- data_ela %>%
  filter(income > 0 & damages < 0) %>%
  group_by(t) %>%
  mutate(damages_demeaned = log(-damages) - mean(log(-damages)), income_demeaned = log(income) - mean(log(income))) %>%
  ungroup() %>%
  select(pop, gdp, income, damages_demeaned, income_demeaned, t, n) %>%
  distinct()


#adjust in the following three plots the color aesthetics to be red for Damages and blue for Benefits
p_within_both <- ggplot(data_within_demeaned %>% filter(t == 18), aes(x = income_demeaned, y = damages_demeaned, color = "Damage")) +
  geom_point() +
  theme_minimal() +
  scale_color_viridis_c() +
  labs(color = "Income per capita", x = "Income relative to mean [$]", y = "Impacts relative to mean  [$]") +
  theme(legend.position = "bottom") +
  geom_abline(aes(slope = 1, intercept = 0, linetype = "neutral")) +
  scale_linetype_manual(values = c("neutral" = "dashed", "estimated" = "solid"), name = "") +
  geom_point(data = data_within_demeaned_neg %>% filter(t == 18), aes(x = income_demeaned, y = damages_demeaned, color = "Benefit")) +
  theme_minimal() +
  geom_abline(aes(intercept = 0, slope = xi_neg_within, linetype = "estimated"), color = "sienna3") +
  labs(color = "", x = "Relative income", y = "Relative impacts") +
  geom_abline(intercept = 0, slope = 1.01, linetype = "solid", color = "skyblue3") +
  theme(legend.position = "bottom") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  geom_label_repel(data = data.frame(income_demeaned = 2, damages_demeaned = label_x_pos * xi_neg_within, gdp = 1), label = TeX(paste0("$\\xi^{-}=", xi_neg_within, "$")), color = "sienna3", size = 4) +
  geom_label_repel(data = data.frame(income_demeaned = -2, damages_demeaned = label_x_neg * 1.01, gdp = 1), label = TeX(paste0("$\\xi^{+}=", xi_pos_within, "$")), color = "skyblue3", size = 4) +
  scale_color_manual(values = c("Benefit" = "skyblue3", "Damage" = "sienna3"), name = "") +
  scale_x_continuous() +
  scale_y_continuous()

p_between_both <- ggplot(data_within_demeaned_between %>% filter(t == 18), aes(x = income_demeaned, y = damages_demeaned, color = "Damage")) +
  geom_abline(aes(slope = 1, intercept = 0, linetype = "neutral")) +
  geom_point() +
  geom_point(data = data_within_demeaned_between_neg %>% filter(t == 18), aes(x = income_demeaned, y = damages_demeaned, color = "Benefit")) +
  ggrepel::geom_text_repel(aes(label = toupper(n))) +
  theme_minimal() +
  labs(color = "", x = "Relative income", y = "Relative impacts") +
  geom_abline(aes(intercept = 0, slope = xi_neg_between, linetype = "estimated"), color = "sienna3") +
  theme(legend.position = "bottom") +
  geom_label_repel(data = data.frame(income_demeaned = 2, damages_demeaned = label_x_pos * xi_neg_between, gdp = 1), label = TeX(paste0("$\\xi^{-}=", xi_neg_between, "$")), color = "sienna3", size = 4) +
  scale_linetype_manual(values = c("neutral" = "dashed", "estimated" = "solid"), name = "") +
  ggrepel::geom_text_repel(data = data_within_demeaned_between_neg %>% filter(t == 18), aes(x = income_demeaned, y = damages_demeaned, color = "Benefit", label = toupper(n))) +
  scale_color_manual(values = c("Benefit" = "skyblue3", "Damage" = "sienna3"), name = "") +
  geom_abline(intercept = 0, slope = xi_pos_between, linetype = "solid", color = "skyblue3") +
  geom_label_repel(data = data.frame(income_demeaned = -2, damages_demeaned = label_x_neg * xi_pos_between, gdp = 1), label = TeX(paste0("$\\xi^{+}=", xi_pos_between, "$")), color = "skyblue3", size = 4)

p_total_both <- ggplot(data_within_demeaned_total %>% filter(t == 18), aes(x = income_demeaned, y = damages_demeaned, color = "Damage")) +
  geom_abline(aes(slope = 1, intercept = 0, linetype = "neutral")) +
  geom_point() +
  geom_point(data = data_within_demeaned_total_neg %>% filter(t == 18), aes(x = income_demeaned, y = damages_demeaned, color = "Benefit")) +
  theme_minimal() +
  labs(color = "", x = "Relative income", y = "Relative impacts") +
  geom_abline(aes(intercept = 0, slope = xi_neg_total, linetype = "estimated"), color = "sienna3") +
  theme(legend.position = "bottom") +
  geom_abline(intercept = 0, slope = xi_pos_total, linetype = "solid", color = "skyblue3") +
  geom_label_repel(data = data.frame(income_demeaned = 2, damages_demeaned = label_x_pos * xi_neg_total, gdp = 1), label = TeX(paste0("$\\xi^{-}=", xi_neg_total, "$")), color = "sienna3", size = 4) +
  scale_linetype_manual(values = c("neutral" = "dashed", "estimated" = "solid"), name = "") +
  scale_color_manual(values = c("Benefit" = "skyblue3", "Damage" = "sienna3"), name = "") +
  geom_label_repel(data = data.frame(income_demeaned = -2, damages_demeaned = label_x_neg * xi_pos_total, gdp = 1), label = TeX(paste0("$\\xi^{+}=", xi_pos_total, "$")), color = "skyblue3", size = 4)

ggarrange(ggplot() + theme_minimal(), p_within_both + ggtitle("Within country inequality"), p_between_both + ggtitle("Between country inequality"), p_total_both + ggtitle("Total Inequality"), ncol = 2, nrow = 2, common.legend = TRUE, legend = "bottom", labels = c("", "A", "B", "C"), font.label = list(size = 12, color = "black", face = "bold"), align = "hv", widths = c(1, 1), heights = c(1, 1))
ggsave("img/inequality_decomposition_both_twocolors.png", width = 12, height = 8, dpi = 300, bg = "white")
ggarrange(ggarrange(p_within_both + ggtitle("Within country inequality"), p_between_both + ggtitle("Between country inequality"),
                    ncol = 2, nrow = 1, common.legend = TRUE, legend = "none", labels = c("A", "B"), font.label = list(size = 12, color = "black", face = "bold"), align = "hv", widths = c(1, 1), heights = c(1)
), p_total_both + ggtitle("Total Inequality"), ncol = 1, nrow = 2, common.legend = TRUE, legend = "bottom", labels = c("", "C"), font.label = list(size = 12, color = "black", face = "bold"), align = "hv", widths = c(1))
ggsave("img/inequality_decomposition_both_twocolors_one_figure.png", width = 12, height = 8, dpi = 300, bg = "white")






