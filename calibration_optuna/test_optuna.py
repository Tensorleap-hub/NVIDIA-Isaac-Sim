from copy import deepcopy
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from calibration_optuna import DEFAULT_CONFIG
from calibration_optuna.experiment_runner import ExperimentRunner
from calibration_optuna.run_optimizer import run_optimizer_iteration


np.random.seed(42)


def create_grouped_dataframe(num_items, group_size_range=(40, 80), extra_string_column=False):
    data = {
        "string_col": [],
        "int_col": [],
        "float_col": []
    }
    if extra_string_column:
        data["extra_string_col"] = []

    created = 0
    while created < num_items:
        group_size = np.random.randint(group_size_range[0], group_size_range[1] + 1)
        group_size = min(group_size, num_items - created)

        g_str = f"group_{np.random.randint(1, 100)}"
        g_int = int(np.random.randint(1, 1000))
        g_float = float(np.random.random() * 100)

        data["string_col"].extend([g_str] * group_size)
        data["int_col"].extend([g_int] * group_size)
        data["float_col"].extend([g_float] * group_size)

        if extra_string_column:
            g_extra = f"extra_{np.random.randint(1, 100)}"
            data["extra_string_col"].extend([g_extra] * group_size)

        created += group_size

    return pd.DataFrame(data)


def test_optuna():
    real_embeddings = np.random.random((300, 400))
    embeddings_per_simulation = [np.random.random((200, 400)), np.random.random((300, 400))]
    df1 = create_grouped_dataframe(200, (40, 80), extra_string_column=False)
    df2 = create_grouped_dataframe(300, (40, 80), extra_string_column=True)
    metadata_per_simulation = [df1, df2]

    suggestions_df, best_trials_df = run_optimizer_iteration(
        real_embeddings,
        embeddings_per_simulation,
        metadata_per_simulation
    )

    assert not suggestions_df.empty
    assert not best_trials_df.empty


def _candidate_row(mu_list, sigma_list):
    row = {}
    for idx, (mu, sigma) in enumerate(zip(mu_list, sigma_list), start=1):
        row[f"mu{idx}"] = float(mu)
        row[f"sigma{idx}"] = float(sigma)
    return row


def _generate_candidate_embeddings(params, base_draws):
    chunks = []
    n_components = base_draws.shape[0]
    for idx in range(1, n_components + 1):
        mu = float(params[f"candidate__mu{idx}"])
        sigma = float(params[f"candidate__sigma{idx}"])
        chunks.append(mu + sigma * base_draws[idx - 1])
    return np.concatenate(chunks, axis=0)


def _candidate_params_from_distribution(params):
    candidate_params = {}
    for key, value in params.items():
        if key.startswith("candidate__"):
            candidate_params[key] = float(value)
    return candidate_params


def _prepare_iteration_inputs(current_distributions, base_draws):
    candidate_embeddings = []
    current_distributions_with_logits = []
    embeddings_indices_by_dist = {}
    start_idx = 0

    for dist_number, (dist_id, params) in enumerate(current_distributions):
        candidate_params = _candidate_params_from_distribution(params)
        distribution_params = {
            "shape_logit_candidate": 0.0,
            **candidate_params
        }
        candidate_embedding = _generate_candidate_embeddings(distribution_params, base_draws)
        end_idx = start_idx + len(candidate_embedding)

        candidate_embeddings.append(candidate_embedding)
        current_distributions_with_logits.append((dist_id, distribution_params))
        embeddings_indices_by_dist[dist_number] = [(0, np.arange(start_idx, end_idx))]
        start_idx = end_idx

    embeddings_by_shape = [np.concatenate(candidate_embeddings, axis=0)]
    return current_distributions_with_logits, embeddings_by_shape, embeddings_indices_by_dist


def _candidate_distributions_from_rows(candidate_rows):
    current_distributions = []
    for idx, row in candidate_rows.iterrows():
        params = {"shape_logit_candidate": 0.0}
        for key, value in row.items():
            params[f"candidate__{key}"] = float(value)
        current_distributions.append((f"dist_{idx}", params))
    return current_distributions


def _candidate_distributions_from_suggestions(suggestions):
    current_distributions = []
    for dist_id, params in suggestions:
        current_params = {"shape_logit_candidate": 0.0}
        for key, value in params.items():
            if key.startswith("candidate__"):
                current_params[key] = float(value)
        current_distributions.append((dist_id, current_params))
    return current_distributions


def _mean_relative_error(best_params, ref_mu, ref_sigma):
    relative_mu_errors = []
    relative_sigma_errors = []
    for idx, (mu, sigma) in enumerate(zip(ref_mu, ref_sigma), start=1):
        best_mu = float(best_params[f"candidate__mu{idx}"])
        best_sigma = float(best_params[f"candidate__sigma{idx}"])

        relative_mu_errors.append(abs(best_mu - mu) / max(abs(mu), 1.0))
        relative_sigma_errors.append(abs(best_sigma - sigma) / max(abs(sigma), 0.1))

    return float(np.mean(relative_mu_errors)), float(np.mean(relative_sigma_errors))


def test_convergens():
    ref_mu = [3.0, 1.0, 8.0, 2.0, 5.0]
    ref_sigma = [0.5, 1.0, 0.1, 0.6, 0.2]
    n_rounds = 5

    base_draws = np.random.RandomState(123).normal(size=(len(ref_mu), 100, 1))
    reference_params = {}
    for idx, (mu, sigma) in enumerate(zip(ref_mu, ref_sigma), start=1):
        reference_params[f"candidate__mu{idx}"] = float(mu)
        reference_params[f"candidate__sigma{idx}"] = float(sigma)

    real_embeddings = _generate_candidate_embeddings(reference_params, base_draws)

    candidate_rows = pd.DataFrame([
        _candidate_row([0.05, 2.0, 4.0, 1, 10.0], [0.1, 0.2, 0.3, 0.4, 0.5]),
        _candidate_row([4.0, 5.0, 6.0, 15, 1], [1.0, 2.0, 3.0, 4.0, 5.0]),
        _candidate_row([2.0, 0.0, 8.0, 0,2], [0.5, 1.4, 0.05, 0.1, 0.2]),
    ])

    param_bounds = {
        "candidate": {
            column: [float(candidate_rows[column].min()), float(candidate_rows[column].max())]
            for column in candidate_rows.columns
        }
    }
    param_type = {
        "candidate": {column: "float" for column in candidate_rows.columns}
    }

    config = deepcopy(DEFAULT_CONFIG)
    config["experiment_name"] = "gaussian_convergence"
    config["iteration_batch_size"] = 4
    config["top_n_best_trials"] = 1
    config["mmd_max_samples"] = 300
    config["optimizer"] = {
        "n_startup_trials": 3,
        "multivariate": True,
    }

    current_distributions = _candidate_distributions_from_rows(candidate_rows)

    with TemporaryDirectory() as temp_dir:
        config["experiments_base_dir"] = temp_dir
        runner = ExperimentRunner(
            config=config,
            param_bounds=param_bounds,
            param_type=param_type
        )
        runner.set_real_embeddings(real_embeddings)

        initial_distributions, embeddings_by_shape, embeddings_indices_by_dist = _prepare_iteration_inputs(
            current_distributions,
            base_draws
        )
        suggestions = runner.run_iteration(
            current_distributions=initial_distributions,
            embeddings_by_shape=embeddings_by_shape,
            embeddings_indices_by_dist=embeddings_indices_by_dist
        )
        initial_best_metric = min(
            trial.values[0] for trial in runner.optimizer.study.trials
            if trial.values is not None
        )

        for _ in range(n_rounds):
            current_distributions = _candidate_distributions_from_suggestions(suggestions)
            current_distributions, embeddings_by_shape, embeddings_indices_by_dist = _prepare_iteration_inputs(
                current_distributions,
                base_draws
            )
            suggestions = runner.run_iteration(
                current_distributions=current_distributions,
                embeddings_by_shape=embeddings_by_shape,
                embeddings_indices_by_dist=embeddings_indices_by_dist
            )

        completed_trials = [
            trial for trial in runner.optimizer.study.trials
            if trial.values is not None
        ]
        final_best_metric = min(trial.values[0] for trial in completed_trials)
        best_trials = runner.get_best_trials(top_n=1)
        mean_relative_error_mu, mean_relative_error_sigma = _mean_relative_error(best_trials[0][1], ref_mu, ref_sigma)

        assert len(suggestions) == config["iteration_batch_size"]
        assert len(completed_trials) == 3 + n_rounds * config["iteration_batch_size"]
        assert len(best_trials) == 1
        assert "shape_prob_candidate" in best_trials[0][1]
        assert "candidate__mu1" in best_trials[0][1]
        assert final_best_metric <= initial_best_metric
        assert mean_relative_error_mu <= 1
