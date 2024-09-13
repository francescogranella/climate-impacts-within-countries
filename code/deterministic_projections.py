# %% imports
import gdxpds
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy.highlevel import dmatrices
from tqdm import tqdm

import context
from utils import gdp_projection, deciles_projections

print(np.__version__)
pd.set_option('use_inf_as_na', True)
context.pdsettings()
tqdm.pandas()

# %% Settings
ssp = 3
rcp = '70'
sampled_coef_index = 0
l = []
for specification in ['adapt', 'BHM', 'KW', 'reconstructed_dummy', 'adapt-no_l1_growth', 'BHM-no_l1_growth', 'KW-no_l1_growth']:
    #
    print(specification)
    gdx_path = context.projectpath() / f'data/out/rice-simulations/results_climsens_ssp{ssp}_0.gdx'
    dfs_dict = gdxpds.read_gdx.to_dataframe(gdx_path, 'TEMP_REGION')
    temp_data = dfs_dict['TEMP_REGION'].rename(columns={'Level': 'temp'})[['t', 'n', 'temp']]
    temp_data['t'] = temp_data['t'].astype(int)

    # GDP per capita with impacts. It will be the interaction term in the projections of decile-level income
    gdp_df = gdp_projection(ssp, temp_data, specification)

    # Projections
    proj_df = deciles_projections(ssp, rcp, gdp_df, temp_data, specification=specification,
                                  sampled_coef_index=sampled_coef_index)


    # Exclude t=1, because there are no damages
    proj_df = proj_df[proj_df.t > 1]

    # Elasticity method 1: Damages on income + country-time FE (Within)
    abs_xi = []
    for _df in [proj_df[proj_df.damages >= 0], proj_df[proj_df.damages < 0]]:
        # Log (of absolute value for negative damages)
        _df = _df.assign(y=np.log(np.abs(_df.damages)))
        _df = _df.assign(x=np.log(_df.income))
        # Demean
        _df = _df.assign(y=_df.y - _df.groupby(['n', 't']).y.transform('mean'))
        _df = _df.assign(x=_df.x - _df.groupby(['n', 't']).x.transform('mean'))
        # Regress
        xi = sm.OLS(*dmatrices('y ~ x', data=_df, return_type='dataframe'), missing='drop').fit().params['x']
        abs_xi.append(xi)
    abs_posxi, abs_negxi = abs_xi

    abs_xi_df = pd.DataFrame(
        dict(specification=specification, ssp=ssp, sampled_coef_index=sampled_coef_index, method='absolute-within',
             posxi=abs_posxi, negxi=abs_negxi), index=[0])

    # Elasticity method 2: Aggregate damages on aggregate income (Between)
    bw_t_xi = []
    # Aggregate
    for _df in [proj_df[proj_df.damages >= 0], proj_df[proj_df.damages < 0]]:
        _df = _df.groupby(['n', 't'])[['damages', 'income']].sum().reset_index()
        # Log (of absolute value for negative damages)
        _df = _df.assign(y=np.log(np.abs(_df.damages)))
        _df = _df.assign(x=np.log(_df.income))
        # Regress
        xi = sm.OLS(*dmatrices('y ~ x + C(t)', data=_df, return_type='dataframe'), missing='drop').fit().params['x']
        bw_t_xi.append(xi)
    bw_t_posxi, bw_t_negxi = bw_t_xi

    bw_t_xi_df = pd.DataFrame(
        dict(specification=specification, ssp=ssp, sampled_coef_index=sampled_coef_index, method='absolute-between_t',
             posxi=bw_t_posxi, negxi=bw_t_negxi, ), index=[0])

    l.append(pd.concat([abs_xi_df, bw_t_xi_df]))

print(pd.concat(l))
