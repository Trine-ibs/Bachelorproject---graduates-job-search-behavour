# Bachelor thesis - Dynamic structural model for UI job-search behavoiur
The programs presented here together set up a structural model framework and estimate the model using generalized method of moments on imported empirical hazard data of Danish UI recipients. The estimated model parameters are stored on a class object, from which the model can be solved and model-implied hazard rates computed under counterfactual benefit schedule parameters.


## Files

| File | Description |
|---|---|
| `dyn_model_class.py` | Model setup as class object: data loading, parameter initialisation, estimation options |
| `dyn_model_funct.py` | Utility functions, search cost, value functions, solution to model with backwards induction |
| `dyn_model_estim.py` | GMM objective, optimization, standard errors |
| `helper_functions.py` | Counterfactuals, aggregation, interpolation, survival function |
| `results_log_util.ipynb` | Estimation and results for log utility |
| `results_shiftlog_util.ipynb` | Estimation and results for log-shift utility |
| `plot_data.ipynb` | Plots of empirical hazard rates, estimated models and spline fits |

## How to Run

Together `dyn_model_class.py`, `dyn_model_funct.py` and `dyn_model_estim.py` provide the entire setup needed for initialising a model class object, estimating the model parameters, and solving and storing the solution on the class object. 
`helper_functions.py` includes additional helper functions needed for simulating and plotting counterfactuals for the estimated model.

In the folder `Data/` you find the empirical hazard rates and covariance matrices for all data used in the estimation. These are automatically loaded through the class object.

1. Run `results_log_util.ipynb` for the log specification estimation
2. Run `results_shiftlog_util.ipynb` for the log-shift specification estimation

The estimated model hazards are saved to `output/` after running these estimation notebooks.

3. Run `plot_data.ipynb` for empirical hazard figures and figures comparing estimated models.
