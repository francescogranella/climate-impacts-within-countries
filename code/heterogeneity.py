import gdxpds
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy.highlevel import dmatrices
from stargazer.stargazer import Stargazer, LineLocation

import context

context.pdsettings()

ssp = 3
specification = 'adapt'
df = pd.read_parquet(context.projectpath() / f'data/out/projections/SSP{ssp}_{specification}.parquet')

# Temperature data
gdx_path = context.projectpath() / f'data/out/rice-simulations/results_climsens_ssp{ssp}_0.gdx'
dfs_dict = gdxpds.read_gdx.to_dataframe(gdx_path, 'TEMP_REGION')
temp_data = dfs_dict['TEMP_REGION'].rename(columns={'Level': 'temp'})[['t', 'n', 'temp']]
temp_data['t'] = temp_data['t'].astype(int)

# Merge
df = pd.merge(df, temp_data, on=['t', 'n'], how='inner')

# Exclude t=1, because there are no damages
df = df[df.t > 1]

results = []
for _df in [
    df[(df.damages >= 0) & (df.gdp < df.gdp.median())], df[(df.damages >= 0) & (df.gdp >= df.gdp.median())],
    df[(df.damages >= 0) & (df.temp < df.temp.median())], df[(df.damages >= 0) & (df.temp >= df.temp.median())],
    df[(df.damages < 0) & (df.gdp < df.gdp.median())], df[(df.damages < 0) & (df.gdp >= df.gdp.median())],
    df[(df.damages < 0) & (df.temp < df.temp.median())], df[(df.damages < 0) & (df.temp >= df.temp.median())],
]:
    # Log (of absolute value for negative damages)
    _df = _df.assign(y=np.log(np.abs(_df.damages)))
    _df = _df.assign(x=np.log(_df.income))
    # Demean
    _df = _df.assign(y=_df.y - _df.groupby(['n', 't']).y.transform('mean'))
    _df = _df.assign(x=_df.x - _df.groupby(['n', 't']).x.transform('mean'))
    # Regress
    r = sm.OLS(*dmatrices('y ~ x', data=_df, return_type='dataframe'), missing='drop').fit(cov_type='HC3')
    results.append(r)

# Collect model results and render the regression table
stargazer = Stargazer(results)
stargazer.dep_var_name = ''
stargazer.dependent_variable = 'Log(Impacts)'
stargazer.add_line('', [r'< median GDP', r'> median GDP', r'< median temp.', r'> median temp.',
                        r'< median GDP', r'> median GDP', r'< median temp.', r'> median temp.', ],
                   LineLocation.HEADER_BOTTOM)
stargazer.custom_columns(['Damages', 'Benefits'], [4, 4])
stargazer.show_model_numbers(True)
stargazer.covariate_order(['x'])
stargazer.rename_covariates({'x': 'Log(Income)'})
stargazer.significant_digits(2)
stargazer.add_line('Country-Year FE', ['Y'] * len(results))
stargazer.show_degrees_of_freedom(False)
stargazer.show_residual_std_err = False
stargazer.show_f_statistic = False
stargazer.append_notes(False)
stargazer.show_notes = False
with open(context.projectpath() / 'tables/heterogeneity.tex', 'w') as f:
    f.write(stargazer.render_latex(only_tabular=True))
