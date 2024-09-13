import pickle
import warnings
from typing import Literal

import gdxpds
import numpy as np
import pandas as pd
from tqdm import tqdm

import context


class IncomeTrajectory:
    def __init__(self, ssp, shares_source='GCAM'):
        """
        Choose an SSP, this class will create an object with the correct income trajectories.
        `GDP` is in trillion USD. `pop` is in million people
        :param ssp:
        """
        warnings.filterwarnings("ignore", message="Unknown file type: assumed RDS")
        # Read the GDX, extract necessary vars and rename
        dfs_dict = gdxpds.read_gdx.to_dataframes(context.projectpath() / f'data/out/rice-simulations/results_climsens_ssp{ssp}_0.gdx')
        gdp = dfs_dict['ykali'].rename(columns={'Value': 'gdp'})
        if shares_source == 'RICE':
            shares = dfs_dict['quantiles_ref'].rename(columns={'Value': 'share'})
        elif shares_source == 'GCAM':
            shares = pd.read_csv(context.projectpath() / 'data/in/ISO_level_projections_PC_model_projections.csv') \
                .rename(columns={'ISO': 'n', 'Category': 'dist', 'pred_shares': 'share', 'year': 't'}) \
                .query(f"sce=='SSP{ssp}'") \
                .assign(n=lambda x: x.n.str.lower(),
                        t=lambda x: (x.t - 2010) / 5,
                        dist=lambda x: x.dist.str.upper(), ) \
                .query("t == t // 1").assign(t=lambda x: x.t.astype(int).astype(str)) \
                .filter(['n', 't', 'dist', 'share'])
        pop = dfs_dict['pop'].rename(columns={'Value': 'pop'})
        # Combine in one dataframe
        df = pd.merge(gdp, pop, on=['t', 'n'], how='inner').merge(
            pd.pivot(shares, index=['t', 'n'], columns='dist', values='share').reset_index(),
            on=['t', 'n'], how='inner'
        )
        df['t'] = df.t.astype(int)
        # Limit to 2100
        # df = df[df.t <= 18]
        # GDP per capita of each decile
        for col in df.filter(like='D').columns:
            df[col] = df.gdp * df[col] / df['pop'] * 1e6 * 10  # Decile population is a tenth of total pop
        df['gdp'] = df['gdp'] / df['pop'] * 1e6
        self.df = df

    def decile(self, decile):
        """
        Extract income trajectory for the chosen decile
        :param decile:
        :return:
        """
        if decile not in ['gdp'] + [f'D{x}' for x in range(1, 11)]:
            raise ValueError(
                "'decile' should be one of ['gdp', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']")
        return self.df[['t', 'n', 'pop', decile]].rename(columns={decile: 'income'})


def save_income_trajectories(shares_source='GCAM'):
    """Create and save to disk income trajectories by ssp and decile/GDP"""
    for ssp in [2, 3]:
        income_traj = IncomeTrajectory(ssp, shares_source=shares_source)
        for decile in ['gdp'] + [f'D{x}' for x in range(1, 11)]:
            df = income_traj.decile(decile)
            df.to_parquet(context.projectpath() / f'data/tmp/baseline_trajectories/income_ssp{ssp}_{decile}_{shares_source}.parquet')


def get_income_trajectory(ssp, decile, shares_source='GCAM'):
    """Read income trajectories by ssp and decile/GDP from disk"""
    file_path = context.projectpath() / f'data/tmp/baseline_trajectories/income_ssp{ssp}_{decile}_{shares_source}.parquet'
    if (file_path).is_file():
        return pd.read_parquet(file_path)
    else:
        save_income_trajectories(shares_source)
        return pd.read_parquet(file_path)


class Coefs:
    def __init__(self, specification):
        # if specification not in ['adapt', 'BHM', 'KW']:
        #     raise ValueError('specification should be one of adapt, BHM, KW')
        coef_files = list((context.projectpath() / 'estimates/coef/').glob(f'{specification}-*.parquet'))
        coef_l = []
        vcov_l = []
        for coef_file in coef_files:
            model_name = coef_file.stem
            model_spec = '-'.join(model_name.split('-')[:-1])
            model_decile = model_name.split('-')[-1]
            if model_spec != specification:
                continue

            _coef = pd.read_parquet(coef_file).set_index('varname')
            _coef = _coef.filter(['Estimate']).rename(columns={'Estimate': 'coef'})
            _coef['specification'] = model_spec
            _coef['decile'] = model_decile
            coef_l.append(_coef)

            _vcov = pd.read_parquet(context.projectpath() / f'estimates/vcov/{coef_file.stem}.parquet').set_index(
                'varname')
            _vcov['specification'] = model_spec
            _vcov['decile'] = model_decile
            vcov_l.append(_vcov)

        coef = pd.concat(coef_l)
        vcov = pd.concat(vcov_l)

        if specification == 'adapt' or specification == 'adapt-no_l1_growth':
            coeflist = ['temperature_mean', 'temperature_mean2', 'temperature_mean:l1_lgdppc',
                        'temperature_mean2:l1_lgdppc']
        elif specification == 'BHM' or specification == 'BHM-no_l1_growth':
            coeflist = ['temperature_mean', 'temperature_mean2']
        elif specification == 'KW'or specification == 'KW-no_l1_growth':
            coeflist = ['diff_temp', 'lag_diff_temp', 'lag_temperature_mean', 'I(lag_temperature_mean^2)',
                        'diff_temp:lag_temperature_mean', 'lag_diff_temp:lag_temperature_mean']
        elif specification == 'reconstructed_dummy':
            coeflist = ['temperature_mean', 'temperature_mean2', 'temperature_mean:l1_lgdppc',
                        'temperature_mean2:l1_lgdppc']

        coef = coef[coef.index.isin(coeflist)]
        vcov = vcov.iloc[vcov.index.isin(coeflist), vcov.columns.isin(coeflist + ['specification', 'decile'])]

        self.coef = coef
        self.vcov = vcov
        self.coeflist = coeflist

    def get_model(self, decile):
        return self.coef[self.coef.decile == decile].drop(columns=['specification', 'decile']), \
            self.vcov[self.vcov.decile == decile].drop(columns=['specification', 'decile'])


def get_sampled_coefs(decile, specification, n=1000, seed=6759):
    """Sample regression coefficients. The point estimates are always the first value"""
    # Regression coefficients and VCOV matrices
    decile = 'GDP' if decile == 'gdp' else decile
    c = Coefs(specification)
    # for chose decile and specification
    coef, vcov = c.get_model(decile)
    # Sample from multivariate normal
    sampled_coefs = np.random.default_rng(seed=seed).multivariate_normal(coef.coef, vcov, size=n - 1)
    # Add the point estimates
    sampled_coefs = np.append(coef.values.T, sampled_coefs, axis=0)

    if False:
        with open(context.projectpath() / 'estimates/results.pickle', 'rb') as f:
            results_dict = pickle.load(f)
        coef = results_dict[decile]['coefs']
        vcov = results_dict[decile]['vcov']
        # Sample from multivariate normal
        sampled_coefs = np.random.default_rng(seed=seed).multivariate_normal(coef, vcov, size=n - 1)
        # Add the point estimates
        sampled_coefs = np.append(np.expand_dims(coef, axis=0), sampled_coefs, axis=0)

    return sampled_coefs


def _bhm_projection(df, coefs):
    """
    Projection process for BHM specification. Differently from the adaptation specificaiton, this is the same iterative process for decile income and GDP

    :param df:
    :param coefs:
    :return:
    """
    b1, b2 = coefs
    for t in range(2, 19):
        # Create lagged income with impacts
        df.loc[df.t == t, 'lag_income_with_impacts'] = df.loc[df.t == t - 1, 'income_with_impacts'].values
        row = df[df.t == t]
        # [1 + base_growth + delta_T * b1 + delta_T2 * b2] * Income_t-1
        df.loc[df.t == t, 'income_with_impacts'] = \
            (1 + row.base_growth + row.delta_temp * b1 + row.delta_temp2 * b2) * row.lag_income_with_impacts
    return df


def _kw_projection(df, coefs):
    """
    Projection process for KW specification. Differently from the adaptation specificaiton, this is the same iterative process for decile income and GDP

    :param df:
    :param coefs: see Coefs('KW').coeflist for the variables the coefficient correspond to
    :return:
    """
    b1, b2, b3, b4, b5, b6 = coefs
    # print(Coefs(specification).coeflist)
    for t in range(2, 19):
        # Create lagged income with impacts
        df.loc[df.t == t, 'lag_income_with_impacts'] = df.loc[df.t == t - 1, 'income_with_impacts'].values
        row = df[df.t == t]
        # [1 + base_growth + (T_t-T_t-1) * b1 + (T_t-1 - T_t-1) * b2 + (T_t-1 - T_0) * b3 +
        # + (T_t-1^2 - T_0^2) * b4 + (T_t-T_t-1)*T_t-1 * b5 + (T_t-1 - T_t-1)*T_t-1 * b6
        df.loc[df.t == t, 'income_with_impacts'] = \
            (1 + row.base_growth +
             row.diff_temp * b1 +
             row.lag_diff_temp * b2 +
             # Not significant
             # row.lag_delta_temp * b3 +
             # row.lag_delta_temp2 * b4 +
             row['diff_temp:lag_temp'] * b5 +
             row['lag_diff_temp:lag_temp'] * b6
             ) \
            * row.lag_income_with_impacts
    return df


def gdp_projection(ssp, temp_data, specification):
    """
    Given an SSP scenario and a coherent exogenous temperature trajectory, project GDP with damages until 2100.
    Note that coefficients are not stochastic but fixed.

    :param ssp:
    :param temp_data:
    :return:
    """
    income_data = get_income_trajectory(ssp, 'gdp')
    # Merge income data with tempearture data
    df = pd.merge(income_data, temp_data, on=['n', 't'], how='inner')
    # and sort
    df.sort_values(by=['n', 't'], inplace=True)
    # Create variables needed for projections of income-with-impacts
    # Assign at time t the baseline growth rate of income from t-1 to t
    df['base_growth'] = df.groupby('n', sort=False).income.pct_change(periods=1).fillna(
        0)  # sort=False speeds up
    # Temperature_t - Temperature_t0
    df['delta_temp'] = df.temp - df.groupby('n', sort=False).temp.transform('first')
    # Temperature_t^2 - Temperature_t0^2
    df['delta_temp2'] = df.temp ** 2 - df.groupby('n', sort=False).temp.transform('first') ** 2
    # Initialize income with impacts..
    df['income_with_impacts'] = df.income
    # ..its lag..
    df['lag_income_with_impacts'] = df.groupby('n', sort=False).income.shift(periods=1)
    df['lag_income_with_impacts'] = np.where(df.lag_income_with_impacts.isna(), df.income_with_impacts,
                                             df.lag_income_with_impacts)
    # ..and the log of lag
    df['log_lag_income_with_impacts'] = np.log(df.lag_income_with_impacts)
    # Baseline growth and Temperature deltas are null at t==1
    df['base_growth'] = df['base_growth'].fillna(0)
    df['delta_temp'] = df['delta_temp'].fillna(0)
    df['delta_temp2'] = df['delta_temp2'].fillna(0)
    # For each country, make projection
    coefs = get_sampled_coefs('gdp', specification=specification, n=1, seed=6759)[0]

    if specification == 'KW' or specification == 'KW-no_l1_growth' :
        # Temp_t - Temp_t-1
        df['diff_temp'] = df.groupby('n', sort=False).temp.diff(periods=1)
        # Temp_t-1 - Temp_t-2
        df['lag_diff_temp'] = df.groupby('n', sort=False).diff_temp.shift(periods=1)
        # Temp_t-1
        df['lag_temp'] = df.groupby('n', sort=False).temp.shift(periods=1)
        # (Temp_t - Temp_t-1)*(Temp_t-1)
        df['diff_temp:lag_temp'] = df['diff_temp'] * df['lag_temp']
        # (Temp_t-1 - Temp_t-2)*(Temp_t-1)
        df['lag_diff_temp:lag_temp'] = df['lag_diff_temp'] * df['lag_temp']
        # Temperature_t-1 - Temperature_t0
        df['lag_delta_temp'] = df.groupby('n', sort=False).delta_temp.shift(periods=1)
        # Temperature_t-1^2 - Temperature_t0^2
        df['lag_delta_temp2'] = df.groupby('n', sort=False).delta_temp2.shift(periods=1)
        # First period: fill nan with 0
        for c in ['diff_temp', 'lag_diff_temp', 'diff_temp:lag_temp', 'lag_diff_temp:lag_temp', 'lag_delta_temp',
                  'lag_delta_temp2']:
            df[c] = df[c].fillna(0)

    # Iteratively reconstruct income with impacts
    # Note: no need to perform this operation country-by-country because of how the data is set up. 🥳
    if specification == 'adapt' or specification == 'adapt-no_l1_growth':
        # Differs from decile-level projections. In the latter, the interaction term log_lag_income_with_impacts is exogenous
        # here, it is updated at each iteration
        b1, b2, b3, b4 = coefs
        for t in range(2, 19):
            # Create lagged income with impacts
            df.loc[df.t == t, 'lag_income_with_impacts'] = df.loc[df.t == t - 1, 'income_with_impacts'].values
            # Create log lagged income per capita with impacts
            df.loc[df.t == t, 'log_lag_income_pc_with_impacts'] = np.log(
                df.loc[df.t == t - 1, 'income_with_impacts'].values)
            #
            row = df[df.t == t]
            # [1 + base_growth + delta_T *  (b1 + b3 *  log(Income_t-1)) + delta_T2 * (b2 + b4 * log(Income_t-1))] * Income_t-1
            df.loc[df.t == t, 'income_with_impacts'] = \
                (1 + row.base_growth +
                 row.delta_temp * (b1 + b3 * row.log_lag_income_pc_with_impacts)
                 +
                 row.delta_temp2 * (b2 + b4 * row.log_lag_income_pc_with_impacts)
                 ) \
                * row.lag_income_with_impacts
    elif specification == 'BHM' or specification == 'BHM-no_l1_growth':
        df = _bhm_projection(df, coefs)
    if specification == 'KW' or specification == 'KW-no_l1_growth':
        df = _kw_projection(df, coefs)

    return df.assign(gdp=df.income,
                     gdppc_with_impacts=df.income_with_impacts,
                     gdp_damages=df.income - df.income_with_impacts) \
        .rename(columns={'income_with_impacts': 'gdp_with_impacts'}) \
        .filter(['t', 'n', 'gdp', 'gdppc_with_impacts', 'gdp_with_impacts', 'gdp_damages'])


def deciles_projections(ssp, rcp, gdp_data, temp_data, specification, sampled_coef_index=0):
    """
    Given an SSP scenario and a coherent exogenous temperature trajectory, project decile income with damages until 2100.

    :param ssp:
    :param rcp: [Redudant]
    :param gdp_data: Projections of GDP with impacts
    :param temp_data:
    :param sampled_coef_index: Regression coefficients are sampled from a distribution. sampled_coef_index picks from that sample. 0 is reserved for point estimates as in the paper
    :param specification:
    :return:
    """
    l = []
    for decile in [f'D{x}' for x in range(1, 11)]:
        coefs = get_sampled_coefs(decile, specification=specification, n=1000, seed=6759)[sampled_coef_index]

        # Decile-level income data
        income_data = get_income_trajectory(ssp, decile)
        # Add country-levle GDP per capita with damages. Needed in interaction terms
        income_data = pd.merge(income_data, gdp_data, on=['n', 't'], how='inner')
        # If temp_exog is True, read iso3 temperatures from GDX file (for correct SSP). Else, take temperature from sample
        # Merge income data with tempearture data
        df = pd.merge(income_data, temp_data, on=['n', 't'], how='inner')
        # and sort
        df.sort_values(by=['n', 't'], inplace=True)
        # Create variables needed for projections of income-with-impacts
        # Assign at time t the baseline growth rate of income from t-1 to t
        df['base_growth'] = df.groupby('n', sort=False).income.pct_change(periods=1).fillna(
            0)  # sort=False speeds up
        # Temperature_t - Temperature_t0
        df['delta_temp'] = df.temp - df.groupby('n', sort=False).temp.transform('first')
        # Temperature_t^2 - Temperature_t0^2
        df['delta_temp2'] = df.temp ** 2 - df.groupby('n', sort=False).temp.transform('first') ** 2
        # Initialize income with impacts..
        df['income_with_impacts'] = df.income
        # .lag GDP per capita
        df['lag_gdppc_with_impacts'] = df.groupby('n', sort=False).gdppc_with_impacts.shift(periods=1)
        df['lag_gdppc_with_impacts'] = np.where(df.lag_gdppc_with_impacts.isna(), df.gdppc_with_impacts,
                                                df.lag_gdppc_with_impacts)
        # ..and the log of lag
        df['log_lag_gdppc_with_impacts'] = np.log(df.lag_gdppc_with_impacts)
        # Baseline growth and Temperature deltas are null at t==1
        df['base_growth'] = df['base_growth'].fillna(0)
        df['delta_temp'] = df['delta_temp'].fillna(0)
        df['delta_temp2'] = df['delta_temp2'].fillna(0)

        if specification == 'KW' or specification == 'KW-no_l1_growth':
            # Temp_t - Temp_t-1
            df['diff_temp'] = df.groupby('n', sort=False).temp.diff(periods=1)
            # Temp_t-1 - Temp_t-2
            df['lag_diff_temp'] = df.groupby('n', sort=False).diff_temp.shift(periods=1)
            # Temp_t-1
            df['lag_temp'] = df.groupby('n', sort=False).temp.shift(periods=1)
            # (Temp_t - Temp_t-1)*(Temp_t-1)
            df['diff_temp:lag_temp'] = df['diff_temp'] * df['lag_temp']
            # (Temp_t-1 - Temp_t-2)*(Temp_t-1)
            df['lag_diff_temp:lag_temp'] = df['lag_diff_temp'] * df['lag_temp']
            # Temperature_t-1 - Temperature_t0
            df['lag_delta_temp'] = df.groupby('n', sort=False).delta_temp.shift(periods=1)
            # Temperature_t-1^2 - Temperature_t0^2
            df['lag_delta_temp2'] = df.groupby('n', sort=False).delta_temp2.shift(periods=1)
            # First period: fill nan with 0
            for c in ['diff_temp', 'lag_diff_temp', 'diff_temp:lag_temp', 'lag_diff_temp:lag_temp', 'lag_delta_temp',
                      'lag_delta_temp2']:
                df[c] = df[c].fillna(0)

        # For each country, make projection
        if specification == 'BHM' or specification == 'BHM-no_l1_growth':
            df = _bhm_projection(df, coefs)
        elif specification == 'KW' or specification == 'KW-no_l1_growth':
            df = _kw_projection(df, coefs)
        else:   # 'adapt' or adapt robustness or adapt-no_l1_growth
            b1, b2, b3, b4 = coefs
            for t in range(2, 19):
                # Create lagged income with impacts
                df.loc[df.t == t, 'lag_income_with_impacts'] = df.loc[df.t == t - 1, 'income_with_impacts'].values
                row = df[df.t == t]
                # [1 + base_growth + delta_T *  (b1 + b3 *  log(GDPpc_t-1)) + delta_T2 * (b2 + b4 * log(GDPpc_t-1))] * Income_t-1
                df.loc[df.t == t, 'income_with_impacts'] = \
                    (1 + row.base_growth +
                     row.delta_temp * (b1 + b3 * row.log_lag_gdppc_with_impacts)
                     +
                     row.delta_temp2 * (b2 + b4 * row.log_lag_gdppc_with_impacts)
                     ) \
                    * row.lag_income_with_impacts

        l.append(
            df.filter(
                ['t', 'n', 'pop', 'income', 'income_with_impacts', 'gdp', 'gdp_with_impacts', 'gdp_damages']).assign(
                decile=decile, ssp=ssp, rcp=rcp))
    df = pd.concat(l)
    df = df.assign(damages=df.income - df.income_with_impacts)
    # Relative income and relative damages
    df['relative_income'] = df.income / df.groupby(
        ['ssp', 'rcp', 'n', 't']).income.transform('sum')
    df['relative_income_with_impacts'] = df.income_with_impacts / df.groupby(
        ['ssp', 'rcp', 'n', 't']).income_with_impacts.transform('sum')
    df['relative_damages'] = df.damages / df.groupby(['ssp', 'rcp', 'n', 't']).damages.transform('sum')
    return df


def write_rice_climsens_bat():
    """
    Write a .bat file with the list of GAMS commands specifying the climate sensitivity.

    :return:
    """
    mu, sd, n = 1.1, 0.3, 500
    # mu, sd, n = 1.17, 0.22, 1000  # better, because has 17-83 at 2.6-3.9, and 5-95 at 2.2, 4.5
    ccss = np.append(np.array([3]), np.random.default_rng(seed=354762).lognormal(mu, sd, n))
    l = ["""cd "C:/Users/Granella/Dropbox (CMCC)/PhD/Research/RICE50x"""""]
    for ssp in [2, 3]:
        for i, cs in enumerate(ccss):
            s = f'gams run_rice50x.gms --n=maxiso3 --baseline=ssp{ssp} --impact=off --resdir="results_impacts" --climsens={cs} --nameout=climsens_ssp{ssp}_{i}'
            l.append(s)
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.kdeplot(ccss);
    plt.show()
    with open(context.projectpath() / 'data/out/rice_probabilistic_temp.bat', 'w') as f:
        f.write('\n'.join(l))

