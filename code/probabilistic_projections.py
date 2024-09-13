# %% imports
"""
Make projections
"""
import platform
from pathlib import Path

import gdxpds
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from patsy.highlevel import dmatrices
from tqdm import tqdm

import context
from utils import gdp_projection, deciles_projections

print(np.__version__)
pd.set_option('use_inf_as_na', True)
context.pdsettings()
tqdm.pandas()


# %% Wrapper function used for parallelization
def wrapper_deciles_projections(ssp, rcp, gdp_data, temp_data, specification, sampled_coef_index):
    """
    Wrapper function used for parallelization
    :param ssp:
    :param rcp:
    :param gdp_data:
    :param temp_data:
    :param sampled_coef_index: what coefs from sample are to be sed
    :return:
    """
    proj_df = deciles_projections(ssp, rcp, gdp_data, temp_data, specification=specification, sampled_coef_index=sampled_coef_index)

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
        dict(ssp=ssp, rcp=rcp, specification=specification, sampled_coef_index=sampled_coef_index, temp_run_index=temp_run_index,
             method='absolute-within', posxi=abs_posxi, negxi=abs_negxi), index=[0])

    # Elasticity method 2: Aggregate damages on aggregate income (Between)
    bw_t_xi = []
    # Aggregate
    for _df in [proj_df[proj_df.damages >= 0], proj_df[proj_df.damages < 0]]:
        _df = _df.groupby(['n', 't'])[['damages', 'income']].sum().reset_index()
        # If there are no damages or no benefits, elasticity is NA
        if len(_df) == 0:
            bw_t_xi.append(np.nan)
        else:
            # Log (of absolute value for negative damages)
            _df = _df.assign(y=np.log(np.abs(_df.damages)))
            _df = _df.assign(x=np.log(_df.income))
            # Regress
            xi = sm.OLS(*dmatrices('y ~ x + C(t)', data=_df, return_type='dataframe'), missing='drop').fit().params['x']
            bw_t_xi.append(xi)
    bw_t_posxi, bw_t_negxi = bw_t_xi

    bw_t_xi_df = pd.DataFrame(
        dict(ssp=ssp, rcp=rcp, specification=specification, sampled_coef_index=sampled_coef_index, temp_run_index=temp_run_index,
             method='absolute-between_t', posxi=bw_t_posxi, negxi=bw_t_negxi), index=[0])

    return pd.concat([abs_xi_df, bw_t_xi_df])


# %% MC
ssps = [3, 2]
specifications = [ 'adapt', 'BHM', 'KW', 'reconstructed_dummy', 'adapt-no_l1_growth', 'bhm-no_l1_growth', 'kw-no_l1_growth']
for specification in specifications:
    for ssp in ssps:
        rcp = {2: '45', 3: '70'}[ssp]

        # Where are RICE runs located?
        if platform.system() == 'Windows':
            temp_runs = list(Path(r"C:\Users\Granella\Dropbox (CMCC)\PhD\Research\RICE50x\results_impacts").glob(f'results_climsens_ssp{ssp}*'))
        else:
            temp_runs = list(
                Path(context.projectpath() / 'data/out/rice-simulations/').glob(f'results_climsens_ssp{ssp}*')
            )

        # For every RICE run...
        temp_runs = sorted(temp_runs)
        for _i, temp_run in (pbar := tqdm(enumerate(temp_runs), total=len(temp_runs))):
            temp_run_index = int(temp_run.stem.split('_')[-1])
            pbar.set_description(f'SSP{ssp}. Spec "{specification}", Temperature index #{temp_run_index}')

            temp_run_file_path = context.projectpath() / f'data/out/sensitivity/elasticity_ssp{ssp}_{specification}_{temp_run_index}.parquet'

            # If file not there yet
            if temp_run_file_path.is_file():
                continue

            # Read temperature trajectories
            dfs_dict = gdxpds.read_gdx.to_dataframe(temp_run, 'TEMP_REGION')
            temp_data = dfs_dict['TEMP_REGION'].rename(columns={'Level': 'temp'})[['t', 'n', 'temp']]
            temp_data['t'] = temp_data['t'].astype(int)

            # GDP per capita with impacts. It will be the interaction term in the projections of decile-level income
            gdp_data = gdp_projection(ssp, temp_data, specification)
            temp_data.to_parquet(
                context.projectpath() / f'data/tmp/temp_data/{ssp}_{specification}_{temp_run_index}.parquet')
            gdp_data.to_parquet(
                context.projectpath() / f'data/tmp/gdp_data/{ssp}_{specification}_{temp_run_index}.parquet')

            # Decile-level income with impacts
            coef_sample_size = 500

            # n_jobs if local machine or remote server
            n_jobs = 6 if platform.system() == 'Windows' else 142

            try:
                k = Parallel(n_jobs=n_jobs)(
                    delayed(wrapper_deciles_projections)(ssp, rcp, gdp_data, temp_data, specification, i) for i in
                    tqdm(range(coef_sample_size)))
                pd.concat(k).to_parquet(temp_run_file_path)
            except np.linalg.LinAlgError:
                continue

