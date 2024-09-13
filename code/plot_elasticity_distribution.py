# %% imports
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats.mstats import winsorize
from tqdm import tqdm

import context

print(np.__version__)
pd.set_option('use_inf_as_na', True)
context.pdsettings()
tqdm.pandas()

# %% Main results
for ssp in [3]:
    for method in ['absolute-within', 'absolute-between_t_n', 'absolute-between_t', 'relative']:
        data = pd.concat([pd.read_parquet(x) for x in
                          tqdm(
                              list((context.projectpath() / f'data/out/sensitivity').glob(
                                  f'elasticity_ssp{ssp}_*.parquet'))
                          )
                          ])
        #  Filtering
        # Plot within-country elasticities
        data = data[data.method == method]

        # Winsorize extreme values likely caused by imbalances in deciles with benefits/damages
        limits = [0.001, 0.001]  # very extremes of the distribution
        data['posxi'] = winsorize(data['posxi'], limits)
        data['negxi'] = winsorize(data['negxi'], limits)

        # Econometric uncertainty. Original temp run
        stat_df = data[data.temp_run_index == 0]
        temp_df = data[data.sampled_coef_index == 0]

        #  Plot
        # Store means, used as bars
        xi_means = data.groupby('specification')[['posxi', 'negxi']].mean().reset_index()
        full_spec_means = {}
        for ci, specification in enumerate(['adapt', 'BHM', 'KW']):
            full_spec_df = data[data['specification'] == specification]
            full_spec_means[specification] = {'posxi': full_spec_df['posxi'].mean(),
                                              'negxi': full_spec_df['negxi'].mean()}

        # Figure
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 5), sharex=True, sharey=True, constrained_layout=False,
                                dpi=200)
        for ci, specification in enumerate(['adapt', 'BHM', 'KW']):
            # subset specification
            stat_spec_df = stat_df[stat_df['specification'] == specification]
            temp_spec_df = temp_df[temp_df['specification'] == specification]
            full_spec_df = data[data['specification'] == specification]

            sns.kdeplot(stat_spec_df['posxi'], ax=axs[0, ci], alpha=1., label='Econometric uncertainty')
            sns.kdeplot(temp_spec_df['posxi'], ax=axs[0, ci], alpha=1., label='Climate uncertainty')
            sns.kdeplot(full_spec_df['posxi'], ax=axs[0, ci], alpha=1., label='Full uncertainty', c='k',
                        linestyle='dashed',
                        linewidth=2)

            sns.kdeplot(stat_spec_df['negxi'], ax=axs[1, ci], alpha=1., label='Econometric uncertainty')
            sns.kdeplot(temp_spec_df['negxi'], ax=axs[1, ci], alpha=1., label='Climate uncertainty')
            sns.kdeplot(full_spec_df['negxi'], ax=axs[1, ci], alpha=1., label='Full uncertainty', c='k',
                        linestyle='dashed',
                        linewidth=2)
            # Means in bars
            for s in ['adapt', 'BHM', 'KW']:
                bottom_bar_heights = xi_means[xi_means.specification == s].iloc[0]
                if s == specification:
                    axs[0, ci].axvline(bottom_bar_heights['posxi'], 0.02, .075, c='k', linewidth=4, clip_on=False,
                                       zorder=1)
                    axs[1, ci].axvline(bottom_bar_heights['negxi'], 0.02, .075, c='k', linewidth=4, clip_on=False,
                                       zorder=1)
                else:
                    axs[0, ci].axvline(bottom_bar_heights['posxi'], 0.02, .075, c='silver', linewidth=4, clip_on=False,
                                       zorder=0)
                    axs[1, ci].axvline(bottom_bar_heights['negxi'], 0.02, .075, c='silver', linewidth=4, clip_on=False,
                                       zorder=0)

            # Title
            axs[0, ci].set_title(specification.replace('adapt', 'BHM-Adapt'))

        axs[0, 0].set_ylabel('Climate\ndamages', )
        axs[1, 0].set_ylabel('Climate\nbenefits', )
        axs[1, 0].set_xlabel(r'')
        axs[1, 1].set_xlabel(r'')
        axs[1, 2].set_xlabel(r'')

        handles, labels = axs[0, 1].get_legend_handles_labels()
        fig.legend(handles[::-1], labels[::-1], frameon=True, framealpha=1, edgecolor='none', fontsize=10,
                   loc='lower center',
                   bbox_to_anchor=(0.5, -.1), ncols=3)

        for ax in axs.flatten():
            ax.axvline(1, c='k', linewidth=0.75, linestyle='dashed')
            ax.axhline(0, c='k', linewidth=0.75, zorder=0)
            ax.set_ylim(-0.75, 4)
            ax.set_xlim(-0.25, 2.25)
            ax.spines['left'].set_bounds(0, 4)
            ax.spines[['top', 'right']].set_visible(False)

        fig.supxlabel(r'$\xi$')
        fig.supylabel(r'Density')
        plt.tight_layout()
        plt.savefig(context.projectpath() / f'img/elasticity_distribution_ssp{ssp}_{method}.png', bbox_inches='tight')
        plt.show()
        # End figure

        # Summary tables
        main_data = data[data.specification.isin(['adapt', 'BHM', 'KW'])]
        posxi_summary = pd.concat([
            main_data.groupby(['ssp', 'rcp', 'specification']).posxi.mean().rename('Mean estimate').to_frame(),
            main_data.groupby(['ssp', 'rcp', 'specification']).posxi.quantile([0.25, 0.75]).unstack().apply(
                lambda x: f'[{np.round(x[0.25], 2)}, {np.round(x[0.75], 2)}]', axis=1).rename(
                'Full uncertainty IQR').to_frame(),
            main_data.groupby(['ssp', 'rcp', 'specification']).posxi.quantile([0.025, 0.975]).unstack().apply(
                lambda x: f'[{np.round(x[0.025], 2)}, {np.round(x[0.975], 2)}]', axis=1).rename(
                'Full uncertainty 95\% range').to_frame(),
            main_data.assign(empty_line='').groupby(['ssp', 'rcp', 'specification']).empty_line.first().rename(
                '').to_frame(),
            main_data[main_data.temp_run_index == 0].groupby(['ssp', 'rcp', 'specification']).posxi.std().rename(
                'Econometric uncertainty $\sigma$').to_frame(),
            main_data[main_data.sampled_coef_index == 0].groupby(['ssp', 'rcp', 'specification']).posxi.std().rename(
                'Climate uncertainty $\sigma$').to_frame(),
            main_data.groupby(['ssp', 'rcp', 'specification']).posxi.std().rename(
                'Full uncertainty $\sigma$').to_frame(),
        ], axis=1).droplevel([0, 1]).T.reset_index().filter(['index', 'adapt', 'BHM', 'KW']).rename(
            columns={'index': '', 'adapt': 'BHM-Adapt'})

        negxi_summary = pd.concat([
            main_data.groupby(['ssp', 'rcp', 'specification']).negxi.mean().rename('Mean estimate').to_frame(),
            main_data.groupby(['ssp', 'rcp', 'specification']).negxi.quantile([0.25, 0.75]).unstack().apply(
                lambda x: f'[{np.round(x[0.25], 2)}, {np.round(x[0.75], 2)}]', axis=1).rename(
                'Full uncertainty IQR').to_frame(),
            main_data.groupby(['ssp', 'rcp', 'specification']).negxi.quantile([0.025, 0.975]).unstack().apply(
                lambda x: f'[{np.round(x[0.025], 2)}, {np.round(x[0.975], 2)}]', axis=1).rename(
                'Full uncertainty 95\% range').to_frame(),
            main_data.assign(empty_line='').groupby(['ssp', 'rcp', 'specification']).empty_line.first().rename(
                '').to_frame(),
            main_data[main_data.temp_run_index == 0].groupby(['ssp', 'rcp', 'specification']).negxi.std().rename(
                'Econometric uncertainty $\sigma$').to_frame(),
            main_data[main_data.sampled_coef_index == 0].groupby(['ssp', 'rcp', 'specification']).negxi.std().rename(
                'Climate uncertainty $\sigma$').to_frame(),
            main_data.groupby(['ssp', 'rcp', 'specification']).negxi.std().rename(
                'Full uncertainty $\sigma$').to_frame(),
        ], axis=1).droplevel([0, 1]).T.reset_index().filter(['index', 'adapt', 'BHM', 'KW']).rename(
            columns={'index': '', 'adapt': 'BHM-Adapt'})

        posxi_summary = posxi_summary.set_index('')
        posxi_summary.columns = pd.MultiIndex.from_tuples(list(product(['Damages'], list(posxi_summary.columns))),
                                                          names=['', ''])
        negxi_summary = negxi_summary.set_index('')
        negxi_summary.columns = pd.MultiIndex.from_tuples(list(product(['Benefits'], list(negxi_summary.columns))),
                                                          names=["", ""])

        latex = pd.concat([posxi_summary, negxi_summary], axis=1).to_latex(multicolumn_format='c',
                                                                           column_format='lcccccc', float_format="%.2f")

        with open(context.projectpath() / f'tables/xi_summary_ssp{ssp}_{method}.tex', 'w') as f:
            f.write(latex)
# %% Robustness
for ssp in [3]:
    for method in ['absolute-within', 'absolute-between_t_n', 'absolute-between_t', 'relative']:
        for specification in ['reconstructed_dummy', 'adapt-no_l1_growth']:
            data = pd.concat([pd.read_parquet(x) for x in
                              tqdm(
                                  list((context.projectpath() / f'data/out/sensitivity').glob(
                                      f'elasticity_ssp{ssp}_*.parquet'))
                              )
                              ])
            #  Filtering
            # Plot within-country elasticities
            data = data[data.method == method]

            # Winsorize extreme values likely caused by imbalances in deciles with benefits/damages
            limits = [0.001, 0.001]  # very extremes of the distribution
            data['posxi'] = winsorize(data['posxi'], limits)
            data['negxi'] = winsorize(data['negxi'], limits)

            # Econometric uncertainty. Original temp run
            stat_df = data[data.temp_run_index == 0]
            temp_df = data[data.sampled_coef_index == 0]

            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 4), sharex=True, sharey=True, constrained_layout=False,
                                    dpi=200)

            stat_spec_df = stat_df[stat_df['specification'] == specification]
            temp_spec_df = temp_df[temp_df['specification'] == specification]
            full_spec_df = data[data['specification'] == specification]

            sns.kdeplot(stat_spec_df['posxi'], ax=axs[0], alpha=1., label='Econometric uncertainty')
            sns.kdeplot(temp_spec_df['posxi'], ax=axs[0], alpha=1., label='Climate uncertainty')
            sns.kdeplot(full_spec_df['posxi'], ax=axs[0], alpha=1., label='Full uncertainty', c='k', linestyle='dashed',
                        linewidth=2)

            sns.kdeplot(stat_spec_df['negxi'], ax=axs[1], alpha=1., label='Econometric uncertainty')
            sns.kdeplot(temp_spec_df['negxi'], ax=axs[1], alpha=1., label='Climate uncertainty')
            sns.kdeplot(full_spec_df['negxi'], ax=axs[1], alpha=1., label='Full uncertainty', c='k', linestyle='dashed',
                        linewidth=2)

            axs[0].axvline(full_spec_df['posxi'].mean(), 0.02, .075, c='k', linewidth=4, clip_on=False)

            axs[1].axvline(full_spec_df['negxi'].mean(), 0.02, .075, c='k', linewidth=4, clip_on=False)

            axs[0].set_ylabel('Climate\ndamages', )
            axs[1].set_ylabel('Climate\nbenefits', )
            axs[1].set_xlabel(r'')
            axs[0].legend(frameon=True, framealpha=1, edgecolor='none', fontsize=10)
            for ax in axs.flatten():
                ax.axvline(1, c='k', linewidth=0.75, linestyle='dashed')
                ax.axhline(0, c='k', linewidth=0.75, zorder=0)
                ax.set_ylim(-0.75, 4)
                ax.set_xlim(-0.25, 2.25)
                ax.spines['left'].set_bounds(0, 4)
                ax.spines[['top', 'right']].set_visible(False)
            fig.supxlabel(r'$\xi$')
            fig.supylabel(r'Density')
            plt.savefig(context.projectpath() / f'img/elasticity_distribution_ssp{ssp}_{method}_{specification}.png')
            plt.tight_layout()
            plt.show()

            robustness_data = data[data.specification == specification]
            posxi_summary = pd.concat([
                robustness_data.groupby(['ssp', 'rcp', 'specification']).posxi.mean().rename('Mean estimate').to_frame(),
                robustness_data.groupby(['ssp', 'rcp', 'specification']).posxi.quantile([0.25, 0.75]).unstack().apply(
                    lambda x: f'[{np.round(x[0.25], 2)}, {np.round(x[0.75], 2)}]', axis=1).rename(
                    'Full uncertainty IQR').to_frame(),
                robustness_data.groupby(['ssp', 'rcp', 'specification']).posxi.quantile([0.025, 0.975]).unstack().apply(
                    lambda x: f'[{np.round(x[0.025], 2)}, {np.round(x[0.975], 2)}]', axis=1).rename(
                    'Full uncertainty 95\% range').to_frame(),
                robustness_data.assign(empty_line='').groupby(['ssp', 'rcp', 'specification']).empty_line.first().rename(
                    '').to_frame(),
                robustness_data[robustness_data.temp_run_index == 0].groupby(
                    ['ssp', 'rcp', 'specification']).posxi.std().rename(
                    'Econometric uncertainty $\sigma$').to_frame(),
                robustness_data[robustness_data.sampled_coef_index == 0].groupby(
                    ['ssp', 'rcp', 'specification']).posxi.std().rename(
                    'Climate uncertainty $\sigma$').to_frame(),
                robustness_data.groupby(['ssp', 'rcp', 'specification']).posxi.std().rename(
                    'Full uncertainty $\sigma$').to_frame(),
            ], axis=1).droplevel([0, 1]).T.reset_index().filter(['index', specification]).rename(
                columns={'index': '', specification: 'BHM-Adapt'})

            negxi_summary = pd.concat([
                robustness_data.groupby(['ssp', 'rcp', 'specification']).negxi.mean().rename('Mean estimate').to_frame(),
                robustness_data.groupby(['ssp', 'rcp', 'specification']).negxi.quantile([0.25, 0.75]).unstack().apply(
                    lambda x: f'[{np.round(x[0.25], 2)}, {np.round(x[0.75], 2)}]', axis=1).rename(
                    'Full uncertainty IQR').to_frame(),
                robustness_data.groupby(['ssp', 'rcp', 'specification']).negxi.quantile([0.025, 0.975]).unstack().apply(
                    lambda x: f'[{np.round(x[0.025], 2)}, {np.round(x[0.975], 2)}]', axis=1).rename(
                    'Full uncertainty 95\% range').to_frame(),
                robustness_data.assign(empty_line='').groupby(['ssp', 'rcp', 'specification']).empty_line.first().rename(
                    '').to_frame(),
                robustness_data[robustness_data.temp_run_index == 0].groupby(
                    ['ssp', 'rcp', 'specification']).negxi.std().rename(
                    'Econometric uncertainty $\sigma$').to_frame(),
                robustness_data[robustness_data.sampled_coef_index == 0].groupby(
                    ['ssp', 'rcp', 'specification']).negxi.std().rename(
                    'Climate uncertainty $\sigma$').to_frame(),
                robustness_data.groupby(['ssp', 'rcp', 'specification']).negxi.std().rename(
                    'Full uncertainty $\sigma$').to_frame(),
            ], axis=1).droplevel([0, 1]).T.reset_index().filter(['index', specification]).rename(
                columns={'index': '', specification: 'BHM-Adapt'})

            posxi_summary = posxi_summary.set_index('')
            posxi_summary.columns = pd.MultiIndex.from_tuples(list(product(['Damages'], list(posxi_summary.columns))),
                                                              names=['', ''])
            negxi_summary = negxi_summary.set_index('')
            negxi_summary.columns = pd.MultiIndex.from_tuples(list(product(['Benefits'], list(negxi_summary.columns))),
                                                              names=["", ""])

            latex = pd.concat([posxi_summary, negxi_summary], axis=1).to_latex(multicolumn_format='c',
                                                                               column_format='lccc', float_format="%.2f")

            with open(context.projectpath() / f'tables/xi_summary_ssp{ssp}_{method}_{specification}.tex', 'w') as f:
                f.write(latex)
