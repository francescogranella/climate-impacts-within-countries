# Create and activate a python environment
conda env create --name=inequality-impacts python=3-10 -y
conda activate inequality-impacts
#Install required packages
pip install -r requirements.txt

# Run code
# Estimate damage functions
rscript code/main.R
# Cmpute the income elasticity of climate damages from point estimates of damage functions and standard ECS
python code/deterministic_projections.py
# Monte Carlo
python code/probabilistic_projections.py
# Descriptive plots for projected damages
python code/plot_distributional_results.py
# Plots and tables of income elasticity distribution from MC
python code/plot_elasticity_distribution.py
# Heterogeneity analysis
python code/heterogeneity.py
# Variance decomposition
rscript code/variance_decomposition_plot.R
