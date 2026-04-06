"""
Data utilities for production optimizer.

Includes:
- Data format conversion (client format -> optimizer format)
- Bounds inference from metadata
- Distribution loading from metadata

No dependencies on MockDataGenerator or data generation modules.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import math
from itertools import product


def _detect_sub_distributions(metadata_df: pd.DataFrame) -> pd.Series:
    """
    Detect sub-distributions within a source by grouping identical rows.

    Rounds numeric columns to 3 decimal places before comparison.
    Returns a Series mapping each row to its sub-distribution ID (0, 1, 2, ...).
    """
    df_rounded = metadata_df.copy()
    for col in df_rounded.columns:
        if pd.api.types.is_float_dtype(df_rounded[col]):
            df_rounded[col] = df_rounded[col].round(3)

    # Use factorize to assign IDs to unique parameter sets
    return pd.factorize(df_rounded.apply(tuple, axis=1))[0]


def prepare_client_data_for_optimizer(
    embeddings_by_shape: List[np.ndarray],
    metadata_by_shape: List[pd.DataFrame]
) -> Tuple[Dict[int, List[Tuple[int, np.ndarray]]], pd.DataFrame, List[str]]:
    """
    Convert per-simulation client data to unified optimizer format.

    Creates joint distributions as Cartesian product of sub-distributions across sources.
    Sub-distributions are detected by grouping identical rows (with 3 decimal places tolerance).

    Args:
        embeddings_by_shape: List of (n_samples, 400) arrays, one per source (simulation type)
        metadata_by_shape: List of DataFrames with source-specific params
                          Sub-distributions detected by grouping identical rows
                          No distribution_idx column needed

    Returns:
        embeddings_indices_by_dist: Dict mapping distribution_id to list of (source_idx, indices) tuples
                                    Indices reference samples in original embeddings_by_shape arrays
        metadata_df: Unified DataFrame with columns:
                     - distribution_id: int (joint distribution ID from Cartesian product)
                     - shape_logit_{simulation_name}: float for each simulation
                     - {simulation_name}__{param}: value for each simulation parameter
                     One row per distribution (not per sample)
        group_names: List of auto-generated simulation names ['simulation_1', 'simulation_2', ...]
    """
    if len(embeddings_by_shape) != len(metadata_by_shape):
        raise ValueError(
            f"Mismatch: {len(embeddings_by_shape)} embedding arrays but "
            f"{len(metadata_by_shape)} metadata DataFrames"
        )

    # Auto-generate simulation names: simulation_1, simulation_2, etc.
    group_names = [f"simulation_{i+1}" for i in range(len(embeddings_by_shape))]

    # Verify DataFrames are not empty
    for i, metadata_df in enumerate(metadata_by_shape):
        if len(metadata_df) == 0:
            raise ValueError(f"Empty DataFrame for simulation_{i+1}")

    # Step 1: Detect sub-distributions within each source
    sub_dist_ids_by_source = []
    sub_dist_params_by_source = []

    for sim_name, metadata_df in zip(group_names, metadata_by_shape):
        # Detect sub-distributions
        sub_dist_ids = _detect_sub_distributions(metadata_df)
        sub_dist_ids_by_source.append(sub_dist_ids)

        # Extract unique sub-distribution parameters
        unique_sub_dists = {}
        for sub_id in range(sub_dist_ids.max() + 1):
            mask = sub_dist_ids == sub_id
            first_row = metadata_df[mask].iloc[0]

            params = {}
            for col in metadata_df.columns:
                value = first_row[col]
                if pd.api.types.is_numeric_dtype(metadata_df[col]):
                    non_null_values = metadata_df[col].dropna()
                    if len(non_null_values) > 0:
                        inferred_type = pd.api.types.infer_dtype(non_null_values, skipna=False)
                        is_integer = (inferred_type == "integer" or
                                     (inferred_type == "floating" and all(v == int(v) for v in non_null_values if pd.notna(v))))
                    else:
                        is_integer = False

                    if is_integer:
                        params[f'{sim_name}__{col}'] = int(value)
                    else:
                        params[f'{sim_name}__{col}'] = float(value)
                else:
                    params[f'{sim_name}__{col}'] = value

            unique_sub_dists[sub_id] = params

        sub_dist_params_by_source.append(unique_sub_dists)

    # Step 2: Generate Cartesian product of sub-distributions
    num_sub_dists_per_source = [len(params) for params in sub_dist_params_by_source]
    sub_dist_ranges = [range(n) for n in num_sub_dists_per_source]
    cartesian_product = list(product(*sub_dist_ranges))

    # Step 3: For each joint distribution, collect indices and build metadata
    unified_metadata_rows = []
    embeddings_indices_by_dist = {}

    for joint_dist_id, sub_dist_combo in enumerate(cartesian_product):
        # Collect indices from each source for this joint distribution
        indices_for_this_dist = []
        sample_counts_for_logits = {}

        for source_idx, (sim_name, sub_dist_id) in enumerate(zip(group_names, sub_dist_combo)):
            # Get indices of samples belonging to this sub-distribution
            mask = sub_dist_ids_by_source[source_idx] == sub_dist_id
            indices = np.where(mask)[0]
            indices_for_this_dist.append((source_idx, indices))
            sample_counts_for_logits[sim_name] = len(indices)

        # Compute shape logits from sample counts
        total_samples = sum(sample_counts_for_logits.values())
        shape_logits = {}
        for sim_name, count in sample_counts_for_logits.items():
            prob = max(count / total_samples, 1e-6)
            shape_logits[f'shape_logit_{sim_name}'] = math.log(prob)

        # Merge parameters from all sources for this joint distribution
        merged_params = {}
        for source_idx, sub_dist_id in enumerate(sub_dist_combo):
            merged_params.update(sub_dist_params_by_source[source_idx][sub_dist_id])

        # Store ONE metadata row per distribution
        unified_row = {
            'distribution_id': joint_dist_id,
            **shape_logits,
            **merged_params
        }
        unified_metadata_rows.append(unified_row)
        embeddings_indices_by_dist[joint_dist_id] = indices_for_this_dist

    # Create compact metadata DataFrame (one row per distribution)
    metadata_df = pd.DataFrame(unified_metadata_rows)

    return embeddings_indices_by_dist, metadata_df, group_names


def _count_samples_per_distribution(
    metadata_by_shape: List[pd.DataFrame],
    group_names: List[str]
) -> Dict[int, Dict[str, int]]:
    """
    Count how many samples each distribution has for each shape.

    Returns:
        {dist_id: {shape_name: count}}
        Example: {0: {'circle': 12, 'ellipse': 8, 'irregular': 5}}
    """
    # Get all unique distribution IDs across all shapes
    all_dist_ids = set()
    for df in metadata_by_shape:
        all_dist_ids.update(df['distribution_id'].unique())

    sample_counts = {}

    for dist_id in sorted(all_dist_ids):
        sample_counts[dist_id] = {}

        for shape_name, metadata_df in zip(group_names, metadata_by_shape):
            # Count rows with this distribution_id
            count = len(metadata_df[metadata_df['distribution_id'] == dist_id])
            sample_counts[dist_id][shape_name] = count

    return sample_counts


def _compute_shape_logits(
    sample_counts: Dict[int, Dict[str, int]],
    group_names: List[str]
) -> Dict[int, Dict[str, float]]:
    """
    Compute shape logits from sample counts using inverse softmax.

    Args:
        sample_counts: {dist_id: {shape: count}}
        group_names: List of shape names

    Returns:
        {dist_id: {shape_logit_{shape}: logit_value}}
    """
    logits_by_dist = {}

    for dist_id, counts in sample_counts.items():
        # Total samples for this distribution
        total = sum(counts.values())

        if total == 0:
            raise ValueError(f"Distribution {dist_id} has 0 total samples")

        # Compute probabilities
        probs = {shape: counts.get(shape, 0) / total for shape in group_names}

        # Inverse softmax: logit = log(prob)
        logits = {}
        for shape in group_names:
            prob = max(probs[shape], 1e-6)  # Avoid log(0)
            logits[f'shape_logit_{shape}'] = math.log(prob)

        logits_by_dist[dist_id] = logits

    return logits_by_dist


def _merge_shape_parameters(
    metadata_by_shape: List[pd.DataFrame],
    group_names: List[str]
) -> Dict[int, Dict[str, Any]]:
    """
    Merge shape-specific parameters into unified format with prefixes.

    Args:
        metadata_by_shape: List of DataFrames with shape-specific params
        group_names: List of shape names

    Returns:
        {dist_id: {shape__param: value}}
        Example: {0: {'circle__void_count_mean': 5.0, 'ellipse__rotation_std': 15.0, ...}}
    """
    # Get all unique distribution IDs
    all_dist_ids = set()
    for df in metadata_by_shape:
        all_dist_ids.update(df['distribution_id'].unique())

    params_by_dist = {}

    for dist_id in sorted(all_dist_ids):
        merged_params = {}

        for shape_name, metadata_df in zip(group_names, metadata_by_shape):
            # Get rows for this distribution
            shape_rows = metadata_df[metadata_df['distribution_id'] == dist_id]

            if len(shape_rows) == 0:
                # No samples for this shape in this distribution
                continue

            # Take first row (all rows within same dist_id should have identical params)
            row = shape_rows.iloc[0]

            # Add all columns except distribution_id with shape prefix
            for col in metadata_df.columns:
                if col != 'distribution_id':
                    value = row[col]
                    # Keep original type (numeric or categorical)
                    if pd.api.types.is_numeric_dtype(metadata_df[col]):
                        non_null_values = metadata_df[col].dropna()
                        if len(non_null_values) > 0:
                            inferred_type = pd.api.types.infer_dtype(non_null_values, skipna=False)
                            is_integer = (inferred_type == "integer" or
                                         (inferred_type == "floating" and all(v == int(v) for v in non_null_values if pd.notna(v))))
                        else:
                            is_integer = False

                        if is_integer:
                            merged_params[f'{shape_name}__{col}'] = int(value)
                        else:
                            merged_params[f'{shape_name}__{col}'] = float(value)
                    else:
                        merged_params[f'{shape_name}__{col}'] = value

        params_by_dist[dist_id] = merged_params

    return params_by_dist


def infer_bounds_from_dataframe(
    df: pd.DataFrame,
    param_columns: List[str] = None,
    exclude_columns: List[str] = None
) -> Dict:
    """
    Infer parameter bounds from a single DataFrame.

    Args:
        df: DataFrame with sample data (each row is a sample)
        param_columns: List of parameter column names to process. If None, uses all columns.
        exclude_columns: List of columns to exclude (e.g., ['void_shape'])

    Returns:
        Dictionary mapping parameter names to their inferred bounds:
        - Numerical columns: [min, max]
        - Categorical columns: list of unique values
    """
    if param_columns is None:
        param_columns = df.columns.tolist()

    if exclude_columns:
        param_columns = [p for p in param_columns if p not in exclude_columns]

    bounds = {}

    for param in param_columns:
        if param not in df.columns:
            print(f"Warning: Column '{param}' not found in DataFrame")
            continue

        col_data = df[param]

        # Check if column is numerical or categorical
        if pd.api.types.is_numeric_dtype(col_data):
            # Numerical column: [min, max]
            bounds[param] = [float(col_data.min()), float(col_data.max())]
        else:
            # Categorical column: list of unique values
            bounds[param] = col_data.unique().tolist()

    return bounds


def infer_bounds_and_types_from_metadata(
    metadata_df: pd.DataFrame,
    group_names: List[str]
) -> Tuple[Dict[str, Dict], Dict[str, Dict[str, str]]]:
    """
    Infer parameter bounds and types from metadata DataFrame.

    The metadata DataFrame has columns:
    - distribution_id: int (groups samples by distribution)
    - shape_logit_*: float (one per shape)
    - {shape}__{param}: float for each shape-specific parameter

    All samples from the same distribution have identical param values.
    Bounds are inferred from the range across all distributions.

    Returns:
        param_bounds: {group_name: {param_name: [min, max] or [categories]}}
        param_types: {group_name: {param_name: "int" | "float" | "categorical"}}
    """
    unique_dists = metadata_df.drop_duplicates(subset='distribution_id')
    param_bounds: Dict[str, Dict] = {}
    param_types: Dict[str, Dict[str, str]] = {}

    fixed_numeric_params: List[str] = []
    total_numeric_params = 0

    for group_name in group_names:
        group_bounds: Dict[str, Any] = {}
        group_types: Dict[str, str] = {}

        for col in unique_dists.columns:
            if col.startswith(f'{group_name}__'):
                param_name = col[len(f'{group_name}__'):]
                values = unique_dists[col]

                if pd.api.types.is_numeric_dtype(values):
                    non_null = values.dropna()
                    if len(non_null) == 0:
                        group_types[param_name] = "float"
                    else:
                        inferred_type = pd.api.types.infer_dtype(non_null, skipna=False)
                        is_integer = (inferred_type == "integer" or
                                     (inferred_type == "floating" and all(v == int(v) for v in non_null)))
                        group_types[param_name] = "int" if is_integer else "float"
                    min_val = float(values.min())
                    max_val = float(values.max())
                    if group_types[param_name] == "int":
                        min_val = int(round(min_val))
                        max_val = int(round(max_val))
                    group_bounds[param_name] = [min_val, max_val]
                    total_numeric_params += 1
                    if min_val == max_val:
                        fixed_numeric_params.append(f'{group_name}__{param_name}')
                        print(
                            "Param has a single value; bounds are fixed.",
                            {
                                "param_name": f"{group_name}__{param_name}",
                                "value": min_val
                            }
                        )
                else:
                    group_bounds[param_name] = values.unique().tolist()
                    group_types[param_name] = "categorical"

        param_bounds[group_name] = group_bounds
        param_types[group_name] = group_types

    if total_numeric_params > 0 and len(fixed_numeric_params) == total_numeric_params:
        fixed_list = ", ".join(fixed_numeric_params)
        raise ValueError(
            "All numeric parameters have fixed bounds (min == max). "
            f"Provide metadata with variability. Fixed params: {fixed_list}"
        )

    return param_bounds, param_types


def load_distributions_from_metadata(
    metadata_df: pd.DataFrame
) -> List[Tuple[str, Dict]]:
    """
    Extract distributions from metadata DataFrame.

    The metadata DataFrame has columns:
    - distribution_id: int (groups samples by distribution)
    - shape_logit_*: float (one per shape)
    - {shape}__{param}: float for each shape-specific parameter

    All samples from the same distribution have identical param values.

    Args:
        metadata_df: DataFrame with distribution_id and all joint params as columns

    Returns:
        List of (dist_id, params_dict) tuples in joint optimizer format
    """
    # Get unique distributions (one row per distribution)
    unique_dists = metadata_df.drop_duplicates(subset='distribution_id').sort_values('distribution_id')

    distributions = []

    for _, row in unique_dists.iterrows():
        dist_id = f"dist_{int(row['distribution_id'])}"

        # Extract all param columns (everything except distribution_id)
        params = {}
        for col in row.index:
            if col != 'distribution_id':
                # Keep original type (numeric or categorical)
                value = row[col]
                if pd.api.types.is_numeric_dtype(metadata_df[col]):
                    non_null_values = metadata_df[col].dropna()
                    if len(non_null_values) > 0:
                        inferred_type = pd.api.types.infer_dtype(non_null_values, skipna=False)
                        is_integer = (inferred_type == "integer" or
                                     (inferred_type == "floating" and all(v == int(v) for v in non_null_values if pd.notna(v))))
                    else:
                        is_integer = False

                    if is_integer:
                        params[col] = int(value)
                    else:
                        params[col] = float(value)
                else:
                    params[col] = value

        distributions.append((dist_id, params))

    return distributions
