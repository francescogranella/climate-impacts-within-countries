# %% Imports
from pathlib import Path

import cartopy as cart
import cartopy.crs as ccrs
import gdxpds
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from tqdm import tqdm

import context

context.pdsettings()


# %% Functions
def gini(x):
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad / np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Credits: https://stackoverflow.com/a/20528097
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    # plt.register_cmap(cmap=newcmap)

    return newcmap


# %% Temperature distribution
temp_runs = list(
    Path(r"C:\Users\Granella\Dropbox (CMCC)\PhD\Research\RICE50x\results_impacts").glob(
        f'results_climsens_ssp*')
)

temp_region_l = []
tatm_l = []
for i, temp_run in (pbar := tqdm(enumerate(temp_runs), total=len(temp_runs))):
    dfs_dict = gdxpds.read_gdx.to_dataframe(temp_run, 'TEMP_REGION')
    _dat = dfs_dict['TEMP_REGION'].rename(columns={'Level': 'temp'})[['t', 'n', 'temp']]
    _dat['t'] = _dat['t'].astype(int)
    _dat['ssp'] = int(temp_run.stem.split('_')[2][-1])
    _dat['temp_run_index'] = int(temp_run.stem.split('_')[3])
    temp_region_l.append(_dat)

    dfs_dict = gdxpds.read_gdx.to_dataframe(temp_run, 'TATM')
    _dat = dfs_dict['TATM'].rename(columns={'Level': 'temp'})[['t', 'temp']]
    _dat['t'] = _dat['t'].astype(int)
    _dat['ssp'] = int(temp_run.stem.split('_')[2][-1])
    _dat['temp_run_index'] = int(temp_run.stem.split('_')[3])
    tatm_l.append(_dat)
temp_region_data = pd.concat(temp_region_l)
tatm_data = pd.concat(tatm_l)

temp_region_data.to_parquet(context.projectpath() / 'data/out/probabilistic_temp_region.parquet')
temp_region_data = pd.read_parquet(context.projectpath() / 'data/out/probabilistic_temp_region.parquet')
tatm_data.to_parquet(context.projectpath() / 'data/out/probabilistic_tatm.parquet')
tatm_data = pd.read_parquet(context.projectpath() / 'data/out/probabilistic_tatm.parquet')

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4), sharex=False, sharey=False, dpi=200)
for idx, g in temp_region_data.groupby('temp_run_index'):
    sns.kdeplot(g.query('t==18 & ssp==3').temp, ax=ax, color='peachpuff', alpha=0.1, linewidth=0.25)
sns.kdeplot(temp_region_data.query('t==18  & ssp==3 & temp_run_index==0').temp, ax=ax, color='tab:orange')
ax.spines[['top', 'right']].set_visible(False)
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}°"))
ax.set_xlabel('Temperature, °C')
plt.suptitle('Country-level mean temperatures\nin 2100 under SSP3')
plt.savefig(context.projectpath() / f'img/TEMP_REGION_distribution.png')
plt.show()

sns.set_palette("tab10")
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)
for ax, (ssp, g) in zip(axs, tatm_data.groupby('ssp')):
    sns.kdeplot(g.query('t==10').temp, ax=ax, )
    sns.kdeplot(g.query('t==18').temp, ax=ax, )
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('Temperature anomaly, °C')
    ax.set_title(f'SSP{ssp}')
axs[0].text(2.8, 1.8, '2050')
axs[0].text(4.75, 0.3, '2100')
axs[1].text(3, 0.6, '2050')
axs[1].text(4.75, 0.3, '2100')
ax.spines[['top', 'right']].set_visible(False)
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("+{x:.0f}°"))
plt.suptitle('Global mean temperature anomaly')
plt.tight_layout()
plt.savefig(context.projectpath() / f'img/GMT_distribution.png')
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), sharex=False, sharey=False)
ax.broken_barh([(0, 1 / 3), (4, 1 / 3)], (0, 1), facecolors='tab:blue')
ax.broken_barh([(2, 1 / 3), (3, 1 / 3)], (2, 1), facecolors='tab:blue')
plt.show()

# %% Violinplot main projection
ssp = 3
specifications = ['adapt', 'BHM', 'KW']
for specification in specifications:
    df = pd.read_parquet(context.projectpath() / f'data/out/projections/SSP{ssp}_{specification}.parquet')
    df['damages_pct'] = - df.damages / df.income

    df = df.query('t==18').pivot(index='n', columns='decile', values='damages_pct')
    df = df[[f'D{x}' for x in range(1, 11)]]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharex=False, sharey=False)
    parts = ax.violinplot(df, showmedians=True, showextrema=False)
    for d in range(1, 11):
        ax.scatter(np.repeat(d, len(df)) + np.random.uniform(-0.15, 0.15, len(df)), df[f'D{d}'], alpha=0.8, c='k', s=3)
        ax.plot([d - 0.25, d + 0.25], [df[f'D{d}'].median(), df[f'D{d}'].median()], c='tab:orange', linewidth=4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    ax.axhline(0, c='k')
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels(df.columns)
    ax.set_xlabel('Decile')
    ax.set_ylabel('Impacts in 2100 [% of income]')
    plt.tight_layout()
    plt.savefig(context.projectpath() / f'img/violinplot_{specification}.png')
    plt.show()


# %% Maps D1-D1, Delta Gini

def _add_background(ax):
    ax.set_extent([-180, 180, -63, 80], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.1)
    ax.add_feature(cart.feature.OCEAN, edgecolor="white", facecolor=[0.9, 0.9, 0.9])
    ax.add_feature(cart.feature.BORDERS, edgecolor="black", linewidth=0.1)
    return ax

ssp = 3
specification = 'adapt'
df = pd.read_parquet(context.projectpath() / f'data/out/projections/SSP{ssp}_{specification}.parquet')
df = df[df.t == 18]

d1d10 = df.assign(damages_to_income=df.damages / df.income).pivot(index=['n'], columns=['decile'],
                                                                  values=['damages_to_income'])
d1d10.columns = [x[1] for x in d1d10.columns]
d1d10 = (d1d10.D1 - d1d10.D10).rename('val').reset_index()
d1d10['n'] = d1d10['n'].str.upper()

gini_with_impacts = df.groupby('n').income_with_impacts.apply(lambda x: gini(x.values)).rename('gini_with_impacts')
gini_cf = df.groupby('n').income.apply(lambda x: gini(x.values)).rename('gini')
delta_gini = (gini_with_impacts - gini_cf).rename('val').reset_index()
delta_gini['n'] = delta_gini['n'].str.upper()
delta_gini['val'] *= 100

gdf1 = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).rename(columns={'iso_a3': 'n'})
gdf1 = gdf1.to_crs(ccrs.Robinson())
gdf1 = gdf1.merge(d1d10, how='inner')

gdf2 = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).rename(columns={'iso_a3': 'n'})
gdf2 = gdf2.to_crs(ccrs.Robinson())
gdf2 = gdf2.merge(delta_gini, how='inner')

fig, axs = plt.subplots(nrows=1, ncols=2, subplot_kw={'projection': ccrs.Robinson()}, dpi=200, figsize=(8, 3))
cmap = mpl.cm.RdYlBu_r

# Panel b
ax = axs[0]
ax = _add_background(ax)
bounds = np.linspace(-0.1, 0.3, 9)
midpoint = 1 - bounds.max() / (bounds.max() + abs(bounds.min())) + 0.01
shifted_cmap = shiftedColorMap(cmap, start=0, midpoint=midpoint, stop=1)
norm = mpl.colors.BoundaryNorm(bounds, shifted_cmap.N, extend='both')
gdf1.plot(ax=ax, column='val', legend=True, norm=norm, cmap=shifted_cmap,
          legend_kwds={'label': r'$\dfrac{Damages^{1}}{Income^{1}}-\dfrac{Damages^{10}}{Income^{10}}$', 'orientation': 'horizontal',
                       'pad': 0.1,
                       'spacing': 'proportional',
                       'extend': 'both',
                       'format': lambda x, _: f"{x:+.0%}",
                       'ticks': bounds[::2],
                       'ticklocation': 'none'
                       }
          )

# Panel b
ax = axs[1]
ax = _add_background(ax)
bounds = np.linspace(-1, 5, 13)
midpoint = 1 - bounds.max() / (bounds.max() + abs(bounds.min())) + 0.01
shifted_cmap = shiftedColorMap(cmap, start=0, midpoint=midpoint, stop=1)
norm = mpl.colors.BoundaryNorm(bounds, shifted_cmap.N, extend='both')
gdf2.plot(ax=ax, column='val', legend=True, norm=norm, cmap=shifted_cmap,
          legend_kwds={'label': r"$\Delta$ Gini [0-100]", 'orientation': 'horizontal',
                       'pad': 0.1,
                       'spacing': 'proportional',
                       'extend': 'both',
                       'format': lambda x, _: f"{x:+.0f}",
                       'ticks': bounds[::2],
                       'ticklocation': 'none'
                       }
          )
for i, ax in enumerate(fig.axes[:2]):
    label = list('abcdefghijklmnopqrst')[i]
    ax.text(-0.1, 1.15, label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')
plt.tight_layout()
plt.savefig(context.projectpath() / 'img/2100_damages.pdf')
plt.show()
