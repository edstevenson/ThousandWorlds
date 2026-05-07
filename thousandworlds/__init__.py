from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[1]))

from . import data, evaluate, preprocessing, schema, spectral
from .evaluate import score, score_predictions
from .data import (
    BASELINES_RESULTS_ARCHIVES,
    BASELINES_URLS,
    BASELINES_URLS_ENVVAR,
    DATASET_PAGE_URL,
    DATA_URL,
    DATA_URL_ENVVAR,
    GRID_SHAPE,
    N_COEFFS,
    T,
    TARGET_PHYSICAL_DOMAIN,
    TARGET_GCMS,
    DataBundle,
    download,
    download_baselines,
    download_dataset,
    load,
)
from .field_spec import (
    CANONICAL_FIELD_VARIABLES,
    CANONICAL_INPUT_NAMES,
    FIELDS_ALL_OBS,
    FIELDS_COMPLETE_OBS_ONLY,
    PUBLIC_NAME_ALIASES,
    public_field_names,
    public_name,
)
from .preprocessing import (
    LinearTrend,
    Stats,
    apply_linear_trend,
    build_design_matrix,
    fit_linear_trend,
    inverse_preprocess_outputs_grid,
    inverse_transform_inputs,
    load_stats,
    normalise_spectral,
    preprocess_outputs_grid,
    remove_linear_trend,
    transform_inputs,
    unnormalise_spectral,
)
from .schema import (
    BENCHMARK_PROTOCOLS,
    BENCHMARK_SPLITS,
    BENCHMARK_SPACES,
    BENCHMARK_SUBSETS,
    PROTOCOL_TO_TEST_FILE,
    SPACE_TO_ARCHIVE_DIR,
    SUBSET_TO_ARCHIVE,
    SUBSET_TO_FIELDS,
    canonical_field_names,
    resolve_data_root,
    subset_path,
    support_path,
    supports_protocol,
)
from .spectral import (
    apply_symmetry_mask,
    build_equatorial_symmetry_mask,
    load_inverse_sht_matrix,
    load_latitude_weights,
    load_symmetry_masks,
    to_grid,
)
