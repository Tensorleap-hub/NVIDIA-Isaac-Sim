from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from calibration_optuna import DEFAULT_CONFIG
from calibration_optuna.data_utils import prepare_client_data_for_optimizer, \
    load_distributions_from_metadata, infer_bounds_and_types_from_metadata
from calibration_optuna.experiment_runner import ExperimentRunner


def suggestions_to_csv_format(
    suggestions: List[Tuple[str, Dict]],
    group_names: List[str],
    n_samples_to_generate: int = 1000,
    add_limit: bool = False,
) -> pd.DataFrame:
    """
    Convert optimizer suggestions from dictionary format to CSV format.

    Creates a single DataFrame where each row represents one simulation type
    from one distribution. All suggestions are combined into one table.

    Args:
        suggestions: List of (dist_id, params_dict) tuples from optimizer
                    params_dict contains shape_prob_* and {simulation}__{param} keys
        group_names: List of simulation names (e.g., ['simulation_1', 'simulation_2'])
        n_samples_to_generate: Total samples per distribution across all sources
        add_limit: mention limit reached in csv

    Returns:
        Single DataFrame with columns:
        - distribution_id: str (dist_3, dist_4, etc.)
        - simulation_type: str (simulation_1, simulation_2, etc.)
        - n_samples: int (number of samples to generate for this source)
        - {param_name}: float for each parameter (without simulation prefix)
    """
    all_rows = []

    for dist_id, params in suggestions:
        for sim_name in group_names:
            # Extract probability for this simulation
            prob_key = f'shape_prob_{sim_name}'
            prob = params.get(prob_key, 0.0)

            # Convert probability to sample count
            n_samples = max(DEFAULT_CONFIG.get('min_samples_to_return', 2), int(round(prob * n_samples_to_generate)))

            # Extract all parameters for this simulation
            param_prefix = f'{sim_name}__'
            row = {
                'distribution_id': dist_id,
                'simulation_type': sim_name,
                'n_samples': n_samples
            }
            if add_limit:
                if n_samples <= DEFAULT_CONFIG.get('min_samples_to_return', 2):
                    row['limit_reached'] = 'True'
                else:
                    row['limit_reached'] = 'False'
            # Add all simulation-specific parameters without prefix
            for key, value in params.items():
                if key.startswith(param_prefix):
                    param_name = key[len(param_prefix):]
                    row[param_name] = value

            all_rows.append(row)

    # Create single DataFrame with all rows
    return pd.DataFrame(all_rows)

def run_optimizer_iteration(
    real_embeddings: np.ndarray,
    embeddings_per_simulation: List[np.ndarray],
    metadata_per_simulation: List['pd.DataFrame']
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    High-level function: run one optimization iteration from client data format.

    This function handles the complete workflow from client's per-simulation data format
    through to optimization suggestions in CSV-ready DataFrame format.

    Workflow:
    1. Convert per-simulation client data to unified optimizer format
       (auto-generates simulation names: simulation_1, simulation_2, etc.)
    2. Extract distributions and infer bounds from metadata
    3. Create runner (or reuse existing one)
    4. Run optimization iteration
    5. Get best trials seen so far
    6. Convert both to CSV format

    Args:
        real_embeddings: Real data embeddings (M, 400)
        embeddings_per_simulation: List of synthetic embedding arrays, one per simulation type
                                   Each array has shape (n_samples_for_that_type, 400)
                                   Together they represent ONE joint distribution
        metadata_per_simulation: List of metadata DataFrames, one per simulation type
                                 Each DataFrame contains parameters for that simulation type
                                 All rows in a DataFrame should have identical parameter values
                                 Parameters are inferred from DataFrame column names

    Returns:
        Tuple of (suggestions_df, best_trials_df):
        - suggestions_df: Next iteration recommendations
        - best_trials_df: Best trials seen so far

        Both DataFrames have columns:
        - distribution_id: distribution/trial ID
        - simulation_type: simulation name (simulation_1, simulation_2, etc.)
        - shape_probability: probability for this simulation
        - {param_name}: parameter values (without simulation prefix)
    """
    if len(embeddings_per_simulation) == 0:
        raise ValueError("Source embeddings must be supplied to run the optimization")
    if len(metadata_per_simulation) == 0:
        raise ValueError("Metadata must be supplied to run the optimization")
    if real_embeddings.shape[0] < 1:
        raise ValueError("Target embeddings must be supplied to run the optimiation")
    # Hardcoded configuration
    config = DEFAULT_CONFIG
    # Convert client format to optimizer format (auto-generates group_names)
    embeddings_indices_by_dist, metadata_df, group_names = prepare_client_data_for_optimizer(
        embeddings_per_simulation, metadata_per_simulation
    )
    print("Prepared Data for Optimiztion Finished",{"Num Distributions": len(embeddings_indices_by_dist),
                    "Synthetic Size": sum([len(embeddings_per_simulation[i]) for i in range(len(embeddings_per_simulation))]),
                    "Real Size": len(real_embeddings),
                    "PCA dim": real_embeddings.shape[-1],
                    "Number of metadata columns": len(metadata_df.columns)
                        })
    # Extract distributions and infer bounds
    distributions = load_distributions_from_metadata(metadata_df)
    param_bounds, param_type = infer_bounds_and_types_from_metadata(metadata_df, group_names)
    print(str(param_bounds)[:10000])

    # Compute paramaters and dynamically override config
    simulation_number = len(embeddings_per_simulation)
    n_samples = config.get('n_samples_to_generate', 1000)
    samples_per_distribution = config.get('samples_per_distribution', 84) * simulation_number
    suggestion_to_make = int(np.clip(n_samples // samples_per_distribution,
                                     5,
                                     config.get('max_top_n_suggestions', 100)))
    top_n_best_trials = config.get('top_n_best_trials', 1)
    config['iteration_batch_size'] = suggestion_to_make
    # Create runner with dict config
    runner = ExperimentRunner(
        config=config,
        param_bounds=param_bounds,
        param_type=param_type
    )
    runner.set_real_embeddings(real_embeddings)

    # Run iteration
    suggestions = runner.run_iteration(
        current_distributions=distributions,
        embeddings_by_shape=embeddings_per_simulation,
        embeddings_indices_by_dist=embeddings_indices_by_dist
    )

    # Limit suggestions and get best trials with separate config parameters
    best_trials = runner.get_best_trials(top_n=top_n_best_trials)

    # Convert both to CSV format
    suggestions_df = suggestions_to_csv_format(suggestions, group_names, samples_per_distribution, add_limit=False)
    best_trials_df = suggestions_to_csv_format(best_trials, group_names, n_samples, add_limit=True)
    print("Finished optimizing", {"samples": n_samples,
                                                  "suggestions": len(suggestions),
                                                  "best": top_n_best_trials
                                                  })
    return suggestions_df, best_trials_df
