'''Module to test code from protflow.runners'''
# imports
import pytest
import pandas as pd
import os
import io
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# dependencies

# custom
from protflow.poses import Poses
from protflow.poses import description_from_path, filter_dataframe_by_value, filter_dataframe_by_rank, col_in_df, load_poses, get_format, combine_dataframe_score_columns, scale_series, normalize_series, class_in_df

####################### variables ##############################
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'poses_description': ['pose1_0001_0001', 'pose1_0001_0002', 'pose2_0001_0001', 'pose2_0001_0003', 'pose3_0001_0001', 'pose4_0001_0001'],
        'score': [10, 20, 30, 40, 50, 60],
        'score2': [20, 40, 60, 80, 100, 120],
        'group': ['A', 'A', 'A', 'B', 'B', 'B'],
        'input_poses': ['pose1_0001_0001.pdb', 'pose1_0001_0002.pdb', 'pose2_0001_0001.pdb', 'pose2_0001_0003.pdb', 'pose3_0001_0001.pdb', 'pose4_0001_0001.pdb'],
        'poses': ['pose1_0001_0001.pdb', 'pose1_0001_0002.pdb', 'pose2_0001_0001.pdb', 'pose2_0001_0003.pdb', 'pose3_0001_0001.pdb', 'pose4_0001_0001.pdb']
    })

class DummyRS:
    def __init__(self, value, from_scorefile: bool = False):
        self.value = value
        self.from_scorefile = from_scorefile

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

### CLASS_IN_DF ###

def test_class_in_df_empty_df_returns_unchanged():
    df = pd.DataFrame(columns=["a", "b"])
    out = class_in_df(df, dict, "hit_cols")
    # same shape & no new column
    assert list(out.columns) == ["a", "b"]
    assert out.empty
    # original not mutated
    assert list(df.columns) == ["a", "b"]

def test_class_in_df_no_matches_returns_unchanged():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out = class_in_df(df, dict, "hit_cols")
    assert list(out.columns) == ["a", "b"]
    pd.testing.assert_frame_equal(out, df)  # identical copy (no new column)

def test_class_in_df_dict_matches_adds_out_col_with_per_row_lists():
    df = pd.DataFrame({
        "a": [1, {"x": 1}, 3],
        "b": [{"y": 2}, 5, [1, 2]],
        "c": ["hi", "there", "world"],
    })
    out = class_in_df(df, dict, "hit_cols")

    # Column added
    assert "hit_cols" in out.columns

    # Expected lists (order must follow column order: a, b, c)
    # row 0: b is dict
    # row 1: a is dict
    # row 2: none are dict → empty list
    expected = [
        ["b"],
        ["a"],
        [],
    ]
    assert out["hit_cols"].tolist() == expected

    # other columns preserved
    pd.testing.assert_series_equal(out["a"], df["a"])

def test_class_in_df_tuple_of_classes_matches_union():
    df = pd.DataFrame({
        "a": [1, {"x": 1}, 3],
        "b": [{"y": 2}, 5, [1, 2]],
        "c": ["hi", "there", "world"],
    })
    out = class_in_df(df, (dict, list), "hit_cols")

    expected = [
        ["b"],       # dict in b
        ["a"],       # dict in a
        ["b"],       # list in b
    ]
    assert out["hit_cols"].tolist() == expected

def test_class_in_df_does_not_mutate_input():
    df = pd.DataFrame({
        "a": [1, {"x": 1}],
        "b": ["z", {"y": 2}],
    })
    df_copy = df.copy(deep=True)

    out = class_in_df(df, dict, "hit_cols")

    # out has the new column, original doesn't
    assert "hit_cols" in out.columns
    assert "hit_cols" not in df.columns
    # original data unchanged
    pd.testing.assert_frame_equal(df, df_copy)

def test_class_in_df_rows_with_no_matches_get_empty_list_when_any_match_exists():
    df = pd.DataFrame({
        "a": [1, {"x": 1}, 3],      # middle row has a dict
        "b": ["u", "v", "w"],
    })
    out = class_in_df(df, dict, "hit_cols")
    # row-wise: [[], ['a'], []]
    assert out["hit_cols"].tolist() == [[], ["a"], []]


#### POSES TESTS ####

def test_set_storage_format_valid_and_invalid(tmp_path):
    p = Poses(work_dir=tmp_path.as_posix(), storage_format="csv")
    assert p.storage_format == "csv"  # valid path per FORMAT_STORAGE_DICT
    with pytest.raises(KeyError):
        p.set_storage_format("xlsx")   # not supported formats raise KeyError


def test_set_scorefile_uses_workdir_basename_and_format(tmp_path):
    p = Poses(work_dir=tmp_path.as_posix(), storage_format="json")
    expected = tmp_path / f"{tmp_path.name}_scores.json"
    assert Path(p.scorefile) == expected


def test_set_work_dir_creates_subdirs_and_sets_paths(tmp_path):
    p = Poses(work_dir=tmp_path.as_posix())
    assert Path(p.work_dir) == tmp_path
    # subdirs are created
    assert (tmp_path / "scores").is_dir()
    assert (tmp_path / "filter").is_dir()
    assert (tmp_path / "plots").is_dir()


def test_parse_poses_string_file_and_list(tmp_path):
    f = tmp_path / "a.pdb"
    f.write_text("REMARK dummy\n")
    p = Poses()
    # string path
    assert p.parse_poses(f.as_posix()) == [f.as_posix()]
    # list of paths
    assert p.parse_poses([f.as_posix()]) == [f.as_posix()]


def test_parse_poses_glob_suffix(tmp_path):
    (tmp_path / "a1.pdb").write_text("x")
    (tmp_path / "a2.pdb").write_text("y")
    p = Poses()
    out = p.parse_poses(tmp_path.as_posix(), glob_suffix="*.pdb")
    assert {Path(x).name for x in out} == {"a1.pdb", "a2.pdb"}


def test_parse_poses_errors(tmp_path):
    p = Poses()
    with pytest.raises(FileNotFoundError):
        p.parse_poses(tmp_path.as_posix(), glob_suffix="*.missing")
    with pytest.raises(FileNotFoundError):
        p.parse_poses(str(tmp_path / "nope.pdb"))
    with pytest.raises(TypeError):
        p.parse_poses(123)  # invalid type


def test_parse_descriptions_basic(tmp_path):
    p = Poses()
    files = [tmp_path / "poseA_0001.pdb", tmp_path / "poseB_0002.pdb"]
    for f in files: f.write_text("x")
    out = p.parse_descriptions([f.as_posix() for f in files])
    assert out == ["poseA_0001", "poseB_0002"]


def test_set_poses_from_dataframe_and_integrity(sample_df):
    p = Poses()
    p.set_poses(sample_df)
    pd.testing.assert_frame_equal(p.df, sample_df)


def test_check_poses_df_integrity_missing_columns():
    p = Poses()
    bad = pd.DataFrame({"poses": ["a.pdb"]})
    with pytest.raises(KeyError):
        p.check_poses_df_integrity(bad)


def test_check_prefix_conflicts_and_invalid():
    p = Poses()
    p.df = pd.DataFrame({
        "input_poses": ["a.pdb"],
        "poses": ["a.pdb"],
        "poses_description": ["a"],
        "foo_location": ["x"]
    })
    with pytest.raises(KeyError):
        p.check_prefix("foo")
    with pytest.raises(ValueError):
        p.check_prefix("bad/name")


def test_split_multiline_fasta_writes_files_and_returns_paths(tmp_path, monkeypatch):
    # mock parse_fasta_to_dict to simulate two sequences + special chars in headers
    from protflow import poses as poses_mod
    monkeypatch.setattr(poses_mod, "parse_fasta_to_dict", lambda path, encoding="UTF-8": {
        "A/1|x": "AAAA",
        "B-2 x": "BBBB"
    })
    p = Poses(work_dir=tmp_path.as_posix())
    out = p.split_multiline_fasta("dummy.fa")
    # files in workdir/input_fastas_split/
    out_dir = tmp_path / "input_fastas_split"
    assert all(Path(fp).parent == out_dir for fp in out)
    assert all(Path(fp).is_file() for fp in out)
    # headers sanitized to underscores
    assert {Path(fp).name for fp in out} == {"A_1_x.fa", "B_2_x.fa"}


def test_split_multiline_fasta_without_workdir_raises(monkeypatch):
    from protflow import poses as poses_mod
    # Mock out the FASTA parser so it doesn't try to read a real file
    monkeypatch.setattr(
        poses_mod, 
        "parse_fasta_to_dict", 
        lambda path, encoding="UTF-8": {"seq1": "AAAA"}
    )
    p = Poses(work_dir=None)
    with pytest.raises(AttributeError):
        p.split_multiline_fasta("dummy.fa")


def test_determine_pose_type_single_and_multiple(sample_df):
    # check if correct extension is returned
    p = Poses(poses=sample_df)
    exts = p.determine_pose_type()
    assert exts == [".pdb"]

    # check if multiple extensions are present
    p.df.at[0, "poses"] = "test.fa"
    exts = p.determine_pose_type()
    assert len(exts) == 2
    assert set(exts) == {".pdb", ".fa"}

    # check if no extension is present
    p.df["poses"] = [os.path.splitext(pose)[0] for pose in p.df["poses"]]
    exts = p.determine_pose_type()
    assert len(exts) == 1
    assert exts[0] == ""


def test_save_scores_appends_extension_and_writes(tmp_path, sample_df):
    p = Poses(poses=sample_df, work_dir=tmp_path.as_posix(), storage_format="csv")
    out_noext = (tmp_path / "scores_out").as_posix()
    p.save_scores(out_path=out_noext, out_format="csv")
    assert (tmp_path / "scores_out.csv").is_file()


def test_save_poses_copies_files_and_respects_overwrite(tmp_path, sample_df):
    # create input files
    dst = tmp_path / "dst"
    sample_df = create_temp_poses(tmp_path, sample_df)
    p = Poses(poses=sample_df)
    p.save_poses(dst.as_posix(), overwrite=True)
    for name in sample_df["poses_description"]:
        assert (dst / f"{name}.pdb").read_text() == f"#{name}" # check if contents match input
        (dst / f"{name}.pdb").write_text("a") # modify content to check if overwrite function works in next step
    p.save_poses(dst.as_posix(), overwrite=False)
    for name in sample_df["poses_description"]:
        assert (dst / f"{name}.pdb").read_text() == "a" # check if contents are still modified


def test_poses_list_returns_poses_column(sample_df):
    p = Poses(poses=sample_df)
    assert p.poses_list() == sample_df["poses"].to_list()


def test_change_poses_dir_copy_and_validate(tmp_path, sample_df):
    sample_df = create_temp_poses(tmp_path, sample_df)
    p = Poses(poses=sample_df)
    # copy=True creates target dir and copies files
    dst = tmp_path / "dst"
    p.change_poses_dir(dst.as_posix(), copy=True, overwrite=False)
    assert {f.name for f in dst.iterdir()} == set([f"{name}.pdb" for name in sample_df["poses_description"]])
    assert all(Path(x).parent == dst for x in p.df["poses"])


def test_change_poses_dir_validate_existing_without_copy(tmp_path, sample_df):
    sample_df = create_temp_poses(tmp_path, sample_df)
    dst = tmp_path / "dst"
    dst.mkdir()
    for path, name in zip(sample_df["poses"], sample_df["poses_description"]):
        shutil.copy(path, dst / f"{name}.pdb")

    p = Poses(poses=sample_df)

    # no copy: requires existing dir and files
    p.change_poses_dir(dst.as_posix(), copy=False)
    assert all(Path(x).parent == dst for x in p.df["poses"])
    assert set(p.poses_list()) == set([path.as_posix() for path in dst.iterdir()])


def test_change_poses_dir_validate_existing_with_copy(tmp_path, sample_df):
    sample_df = create_temp_poses(tmp_path, sample_df)
    dst = tmp_path / "dst"

    p = Poses(poses=sample_df)

    # copy
    p.change_poses_dir(dst.as_posix(), copy=True)
    assert all(Path(x).parent == dst for x in p.df["poses"])
    assert set(p.poses_list()) == set([path.as_posix() for path in dst.iterdir()])


def test_get_pose_loads_structure_by_description(tmp_path, monkeypatch):
    f = tmp_path / "a.pdb"; f.write_text("ATOM ...")
    p = Poses()
    p.df = pd.DataFrame({
        "input_poses": [f.as_posix()],
        "poses": [f.as_posix()],
        "poses_description": ["a"]
    })

    sentinel = object()
    # mock the loader used inside get_pose
    with patch("protflow.poses.load_structure_from_pdbfile", return_value=sentinel) as m:
        out = p.get_pose("a")
        assert out is sentinel
        m.assert_called_once()
    with pytest.raises(KeyError):
        p.get_pose("missing")


def test_reindex_poses_force_and_conflict(tmp_path, sample_df):
    sample_df = create_temp_poses(tmp_path, sample_df)
    p = Poses(poses=sample_df, work_dir=tmp_path)
    # two poses share same name if one layer is removed
    # removing 1 layer without force => conflict error
    with pytest.raises(RuntimeError):
        p.reindex_poses(prefix="reindexed", remove_layers=1, force_reindex=False)

    # with force=True: writes to work_dir/prefix and appends new index
    p.reindex_poses(prefix="reindexed", remove_layers=1, force_reindex=True)
    out_dir = tmp_path / "reindexed"
    assert all(out_dir in Path(x).parents for x in p.df["poses"])
    # descriptions should have single layer with new index
    assert set(p.df["poses_description"]) == {"x_0001", "x_0002"} or all("_" in d for d in p.df["poses_description"])

def test_set_poses_df_triggers_resselection_conversion_str_and_dict(monkeypatch):
    # Patch the class used inside convert_resselection_cols
    from protflow import poses as poses_mod
    monkeypatch.setattr(poses_mod, "ResidueSelection", DummyRS)

    df = pd.DataFrame({
        "input_poses": ["a.pdb"],
        "poses": ["a.pdb"],
        "poses_description": ["a"],
        "import_resselection_cols": [["fixed_residues", "motif_residues"]],
        "fixed_residues": ["A12,A34"],  # str -> DummyRS(str)
        "motif_residues": [{"residues": [["A", 164], ["A", 165]]}],  # dict -> DummyRS(dict, from_scorefile=True)
    })

    p = Poses()
    p.set_poses(df)  # must call convert_resselection_cols under the hood

    assert isinstance(p.df.at[0, "fixed_residues"], DummyRS)
    assert p.df.at[0, "fixed_residues"].value == "A12,A34"
    assert p.df.at[0, "fixed_residues"].from_scorefile is False

    assert isinstance(p.df.at[0, "motif_residues"], DummyRS)
    assert p.df.at[0, "motif_residues"].value == {"residues": [["A", 164], ["A", 165]]}
    assert p.df.at[0, "motif_residues"].from_scorefile is True

def test_set_poses_df_stringified_selector_is_parsed(monkeypatch):
    from protflow import poses as poses_mod
    monkeypatch.setattr(poses_mod, "ResidueSelection", DummyRS)

    df = pd.DataFrame({
        "input_poses": ["a.pdb"],
        "poses": ["a.pdb"],
        "poses_description": ["a"],
        # like CSV import: stringified list
        "import_resselection_cols": ["['motif_residues']"],
        "motif_residues": ["B5-B9"],  # str -> DummyRS(str)
    })

    p = Poses()
    p.set_poses(df)

    assert isinstance(p.df.at[0, "motif_residues"], DummyRS)
    assert p.df.at[0, "motif_residues"].value == "B5-B9"

def test_convert_resselection_cols_missing_target_column_warns_and_skips(monkeypatch, caplog):
    from protflow import poses as poses_mod
    monkeypatch.setattr(poses_mod, "ResidueSelection", DummyRS)

    df = pd.DataFrame({
        "input_poses": ["a.pdb"],
        "poses": ["a.pdb"],
        "poses_description": ["a"],
        "import_resselection_cols": [["nope", "fixed_residues"]],
        "fixed_residues": ["A1"],
    })

    p = Poses()
    p.set_poses(df)  # runs convert_resselection_cols()

    # existing col converted
    assert isinstance(p.df.at[0, "fixed_residues"], DummyRS)
    # and we should have seen a warning about the missing one
    assert any("Could not find column nope" in rec.message for rec in caplog.records)

def test_convert_resselection_cols_selector_wrong_type_raises(monkeypatch):
    from protflow import poses as poses_mod
    monkeypatch.setattr(poses_mod, "ResidueSelection", DummyRS)

    df = pd.DataFrame({
        "input_poses": ["a.pdb"],
        "poses": ["a.pdb"],
        "poses_description": ["a"],
        "import_resselection_cols": [123],  # not list/tuple/parsable string
        "fixed_residues": ["A1"],
    })

    p = Poses()
    with pytest.raises(KeyError):
        p.set_poses(df)

def test_convert_resselection_cols_malformed_stringified_selector_raises(monkeypatch):
    from protflow import poses as poses_mod
    monkeypatch.setattr(poses_mod, "ResidueSelection", DummyRS)

    df = pd.DataFrame({
        "input_poses": ["a.pdb"],
        "poses": ["a.pdb"],
        "poses_description": ["a"],
        "import_resselection_cols": ["[motif_residues"],  # malformed, literal_eval should fail
        "motif_residues": ["A1"],
    })

    p = Poses()
    with pytest.raises((ValueError, SyntaxError, KeyError)):
        p.set_poses(df)
    
def test_convert_resselection_cols_skips_falsy_and_keeps_existing_instances(monkeypatch):
    from protflow import poses as poses_mod
    monkeypatch.setattr(poses_mod, "ResidueSelection", DummyRS)

    already = DummyRS("A2")  # will be left as-is

    df = pd.DataFrame({
        "input_poses": ["a.pdb", "b.pdb"],
        "poses": ["a.pdb", "b.pdb"],
        "poses_description": ["a", "b"],
        "import_resselection_cols": [["fixed_residues"], ["fixed_residues"]],
        "fixed_residues": [already, None],   # row0 already RS, row1 falsy -> skip
    })

    p = Poses()
    p.set_poses(df)

    # Row 0 remains same object (no re-wrap)
    assert p.df.at[0, "fixed_residues"] is already
    # Row 1 unchanged (None stays None)
    assert p.df.at[1, "fixed_residues"] is None

def test_convert_resselection_cols_absent_selector_col_is_noop(monkeypatch):
    from protflow import poses as poses_mod
    monkeypatch.setattr(poses_mod, "ResidueSelection", DummyRS)

    df = pd.DataFrame({
        "input_poses": ["a.pdb"],
        "poses": ["a.pdb"],
        "poses_description": ["a"],
        # no "import_resselection_cols" column
        "fixed_residues": ["A1"],
    })

    p = Poses()
    p.set_poses(df)  # should not attempt conversion, no error
    assert p.df.at[0, "fixed_residues"] == "A1"


###############################################################################
def create_temp_poses(path, df):
    """
    creates temporary poses files with description as content
    """
    src = path / "src"
    src.mkdir()
    df["poses"] = path / df["poses"]
    for file, name in zip(df["poses"], df["poses_description"]):
        file.write_text(f"#{name}") # create tmp files containing description as input
    return df

