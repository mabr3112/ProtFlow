'''Module to test code from protflow.runners'''
# imports
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock


# dependencies

# custom
from protflow.poses import description_from_path, filter_dataframe_by_value, filter_dataframe_by_rank, col_in_df, load_poses, get_format, combine_dataframe_score_columns, scale_series, normalize_series

####################### variables ##############################
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'poses_description': ['pose1_0001_0001', 'pose1_0001_0002', 'pose2_0001_0001', 'pose2_0001_0003', 'pose3_0001_0001', 'pose4_0001_0001'],
        'score': [10, 20, 30, 40, 50, 60],
        'score2': [20, 40, 60, 80, 100, 120],
        'group': ['A', 'A', 'A', 'B', 'B', 'B']
    })


####################### tests ##################################
@pytest.mark.parametrize("path, expect", [
    ("/path/to/my/testfile.pdb", "testfile")
])

def test_description_from_path(path, expect):
    output = description_from_path(path)
    assert output == expect

#### FILTER_DATAFRAME_BY_VALUE ####

@pytest.mark.parametrize("operator,value,expected_scores", [
    ('>', 30, [40, 50, 60]),
    ('>=', 30, [30, 40, 50, 60]),
    ('<', 30, [10, 20]),
    ('<=', 30, [10, 20, 30]),
    ('=', 30, [30]),
    ('!=', 30, [10, 20, 40, 50, 60])
])

# Test filtering with all supported operators
def test_filter_dataframe_by_value_valid(sample_df, operator, value, expected_scores):
    output_df = filter_dataframe_by_value(sample_df, col='score', value=value, operator=operator)
    assert list(output_df['score']) == expected_scores

# Test invalid operator
def test_filter_dataframe_by_value_invalid_operator(sample_df):
    with pytest.raises(KeyError, match="Invalid operator"):
        filter_dataframe_by_value(sample_df, col='score', value=30, operator='invalid')

# Test missing column
def test_filter_dataframe_by_value_missing_column(sample_df):
    with pytest.raises(KeyError):
        filter_dataframe_by_value(sample_df, col='missing_col', value=30, operator='>')

# Test empty result
def test_filter_dataframe_by_value_empty_result(sample_df):
    output_df = filter_dataframe_by_value(sample_df, col='score', value=100, operator='>')
    assert output_df.empty


#### FILTER_DATAFRAME_BY_RANK ####

# Basic top N filtering
@pytest.mark.parametrize("n,expected_scores", [
    (1, [10]),        # Ascending: smallest score
    (2, [10, 20]),    # Top 2 in ascending order
])

def test_filter_dataframe_by_rank_top_n(sample_df, n, expected_scores):
    output_df = filter_dataframe_by_rank(sample_df, col='score', n=n, ascending=True)
    assert list(output_df['score']) == expected_scores

# Top fraction filtering
@pytest.mark.parametrize("n,expected_count", [
    (0.5, 3),  # top 50% fraction
    (0.34, 2), # roughly 1/3 of rows
])
def test_filter_dataframe_by_rank_fraction(sample_df, n, expected_count):
    output_df = filter_dataframe_by_rank(sample_df, col='score', n=n, ascending=False)
    # descending: should pick top scores
    assert len(output_df) == expected_count
    assert output_df['score'].is_monotonic_decreasing


# Group filtering
def test_filter_dataframe_by_rank_grouped(sample_df):
    output_df = filter_dataframe_by_rank(sample_df, col='score', n=1, group_col='group', ascending=False)
    # Each group should retain 1 row, the highest score in each
    assert len(output_df) == 2
    assert set(output_df['score']) == {30, 60}


# Remove layers filtering
def test_filter_dataframe_by_rank_remove_layers(sample_df):
    output_df = filter_dataframe_by_rank(
        sample_df, col='score', n=1, remove_layers=1, ascending=False
    )
    # Removing last layer should group poses by prefix 'pose1', 'pose2', 'pose3'
    assert len(output_df) == 4
    # Pose1 -> max 50, Pose2 -> max 40, Pose3 -> max 60
    assert set(output_df['score']) == {20, 40, 50, 60}


# Invalid n value
def test_filter_dataframe_by_rank_invalid_n(sample_df):
    with pytest.raises(ValueError):
        filter_dataframe_by_rank(sample_df, col='score', n=0)
    with pytest.raises(ValueError):
        filter_dataframe_by_rank(sample_df, col='score', n=-1)

# Mutually exclusive group_col and remove_layers
def test_filter_dataframe_by_rank_mutually_exclusive(sample_df):
    with pytest.raises(KeyError, match="mutually exclusive"):
        filter_dataframe_by_rank(sample_df, col='score', n=1, group_col='group', remove_layers=1)


# Invalid remove_layers type
def test_filter_dataframe_by_rank_invalid_remove_layers_type(sample_df):
    with pytest.raises(TypeError):
        filter_dataframe_by_rank(sample_df, col='score', n=1, remove_layers="1")


# Missing column
def test_filter_dataframe_by_rank_missing_column(sample_df):
    with pytest.raises(KeyError):
        filter_dataframe_by_rank(sample_df, col='missing_col', n=1)

#### COL_IN_DF ####

# Single column exists
def test_col_in_df_single_column_exists(sample_df):
    # Should not raise an error
    col_in_df(sample_df, 'poses_description')


# Multiple columns exist
def test_col_in_df_multiple_columns_exist(sample_df):
    # Should not raise an error
    col_in_df(sample_df, ['poses_description', 'score'])


# Single column missing
def test_col_in_df_single_column_missing(sample_df):
    with pytest.raises(KeyError, match="Could not find missing_col"):
        col_in_df(sample_df, 'missing_col')


# Multiple columns with one missing
def test_col_in_df_multiple_columns_one_missing(sample_df):
    with pytest.raises(KeyError, match="Could not find missing_col"):
        col_in_df(sample_df, ['poses_description', 'missing_col'])


# Empty list as columns
def test_col_in_df_empty_list(sample_df):
    # Should do nothing and not raise
    col_in_df(sample_df, [])


#### LOAD_POSES ####
def test_load_poses_calls_poses_load_poses():
    mock_poses_instance = MagicMock()
    with patch("protflow.poses.Poses", return_value=mock_poses_instance) as mock_poses_class:
        result = load_poses("dummy_path.json")

        # Ensure Poses() was instantiated
        mock_poses_class.assert_called_once()

        # Ensure .load_poses was called with the path
        mock_poses_instance.load_poses.assert_called_once_with("dummy_path.json")

        # Ensure the returned value is the same as the mocked instance
        assert result == mock_poses_instance.load_poses.return_value


#### GET_FORMAT ####

# Supported formats
@pytest.mark.parametrize("filename,expected_function", [
    ("file.json", pd.read_json),
    ("file.csv", pd.read_csv),
    ("file.pickle", pd.read_pickle),
    ("file.feather", pd.read_feather),
    ("file.parquet", pd.read_parquet),
    ("file.JSON", pd.read_json),
])
def test_get_format_supported(filename, expected_function):
    func = get_format(filename)
    assert func == expected_function


# Unsupported format should raise KeyError
def test_get_format_unsupported_extension():
    with pytest.raises(KeyError):
        get_format("file.txt")


#### COMBINE_DATAFRAME_SCORE_COLUMNS ####

# Basic combination without scaling
def test_combine_dataframe_score_columns_basic(sample_df):
    # Duplicate 'score' to simulate multiple score columns
    combined = combine_dataframe_score_columns(
        sample_df, scoreterms=['score', 'score2'], weights=[1, 1], scale=False
    )

    expected = pd.Series([-2.672612, -1.603567, -0.534522, 0.534522, 1.603567, 2.672612])
    pd.testing.assert_series_equal(
        combined.reset_index(drop=True), expected, rtol=1e-6
    )


# Combination with scaling
def test_combine_dataframe_score_columns_scaled(sample_df):
    combined = combine_dataframe_score_columns(
        sample_df, scoreterms=['score', 'score2'], weights=[1, 1], scale=True
    )

    expected = pd.Series([0, 0.2, 0.4, 0.6, 0.8, 1])

    pd.testing.assert_series_equal(
        combined.reset_index(drop=True), expected, rtol=1e-6
    )


# Weight and scoreterms mismatch
def test_combine_dataframe_score_columns_weight_mismatch(sample_df):
    with pytest.raises(ValueError, match="must be equal"):
        combine_dataframe_score_columns(sample_df, scoreterms=['score', 'score2'], weights=[1])


# Non-numeric values in score column
def test_combine_dataframe_score_columns_non_numeric(sample_df):
    sample_df['score3'] = ['10', '20', 'BAD', '40', '50', '60']
    with pytest.raises(ValueError, match="must only contain float or integers"):
        combine_dataframe_score_columns(sample_df, scoreterms=['score', 'score3'], weights=[1, 1])


def test_combine_dataframe_score_columns_weighted_effect(sample_df):
  
    combined_skewed = combine_dataframe_score_columns(
        sample_df, ['score', 'score2'], [1, -2]
    )
    
    expected = pd.Series([1.336306, 0.801784, 0.267261, -0.267261, -0.801784, -1.336306])

    pd.testing.assert_series_equal(
        combined_skewed.reset_index(drop=True), expected, rtol=1e-6
    )


#### SCALE_SERIES ####

def test_scale_series_basic(sample_df):
    scaled = scale_series(sample_df['score'])
    # min should be 0 and max should be 1
    assert scaled.min() == 0
    assert scaled.max() == 1
    # Values should be increasing
    assert scaled.is_monotonic_increasing


def test_scale_series_constant():
    ser = pd.Series([5, 5, 5])
    scaled = scale_series(ser)
    # All values should be 0 if all original values are the same
    assert all(scaled == 0)


def test_scale_series_contains_nan(sample_df):
    ser = sample_df['score'].copy()
    ser.iloc[2] = float('nan')
    scaled = scale_series(ser)
    # NaNs should remain NaNs after scaling
    assert scaled.isna().sum() == 1

#### NORMALIZE_SERIES ####

def test_normalize_series_basic(sample_df):
    ser = normalize_series(sample_df['score'])
    # Median-centered: median should be around 0 after normalization
    assert abs(ser.median()) < 1e-12
    # Standard deviation should be 1
    assert abs(ser.std() - 1.0) < 1e-12


def test_normalize_series_scaled(sample_df):
    ser = normalize_series(sample_df['score'], scale=True)
    # Should be between 0 and 1 after scaling
    assert ser.min() == 0
    assert ser.max() == 1
    # Length should be the same as input
    assert len(ser) == len(sample_df)


def test_normalize_series_constant():
    ser = pd.Series([5, 5, 5])
    normalized = normalize_series(ser)
    # All values should be 0 for constant series
    assert all(normalized == 0)


def test_normalize_series_with_nan(sample_df):
    ser = sample_df['score'].copy()
    ser.iloc[2] = float('nan')
    normalized = normalize_series(ser)
    # NaNs remain NaNs
    assert normalized.isna().sum() == 1
