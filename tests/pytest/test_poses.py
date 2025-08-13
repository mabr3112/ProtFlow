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
from protflow.poses import description_from_path, filter_dataframe_by_value, filter_dataframe_by_rank, col_in_df, load_poses, get_format, combine_dataframe_score_columns, scale_series, normalize_series

####################### variables ##############################
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'poses_description': ['pose1_0001_0001', 'pose1_0001_0002', 'pose2_0001_0001', 'pose2_0001_0003', 'pose3_0001_0001', 'pose4_0001_0001'],
        'score': [10, 20, 30, 40, 50, 60],
        'score2': [20, 40, 60, 80, 100, 120],
        'group': ['A', 'A', 'A', 'B', 'B', 'B'],
        'group2': ["pose1.pdb", "pose2.pdb", "pose3.pdb", "pose4.pdb", "pose5.pdb", "pose6.pdb"],
        'input_poses': ['pose1.pdb', 'pose1.pdb', 'pose2.pdb', 'pose2.pdb', 'pose3.pdb', 'pose4.pdb'],
        'poses': ['pose1_0001_0001.pdb', 'pose1_0001_0002.pdb', 'pose2_0001_0001.pdb', 'pose2_0001_0003.pdb', 'pose3_0001_0001.pdb', 'pose4_0001_0001.pdb']
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
    assert set(p.df["poses"].to_list()) == set([path.as_posix() for path in dst.iterdir()])


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


@pytest.mark.parametrize("group_col,remove_layers,expect", [
    ("input_poses", None, ['pose1_0001.pdb', 'pose1_0002.pdb', 'pose2_0001.pdb', 'pose2_0002.pdb', 'pose3_0001.pdb', 'pose4_0001.pdb']),
    (None, 1, ['pose1_0001_0001.pdb', 'pose1_0001_0002.pdb', 'pose2_0001_0001.pdb', 'pose2_0001_0002.pdb', 'pose3_0001_0001.pdb', 'pose4_0001_0001.pdb']),
])


def test_reindex_poses_force_and_conflict(tmp_path, sample_df, group_col, remove_layers, expect):
    sample_df = create_temp_poses(tmp_path, sample_df)
    p = Poses(poses=sample_df, work_dir=tmp_path.as_posix())
    # poses share same name if using input_poses as group_col => conflict error
    with pytest.raises(RuntimeError):
        p.reindex_poses(prefix="reindexed", group_col=group_col, remove_layers=remove_layers, force_reindex=False)

    # with force=True: writes to work_dir/prefix and appends new index
    p.reindex_poses(prefix="reindexed", group_col=group_col, remove_layers=remove_layers, force_reindex=True)

    out_dir = tmp_path / "reindexed"
    # descriptions should have single layer with new index
    assert p.df["poses"].to_list() == [(out_dir / filename).as_posix() for filename in expect]


def test_reindex_poses_group_col_and_remove_layers():
    p = Poses()
    # check if error is raised if both group col and remove layers are set
    with pytest.raises(KeyError):
        p.reindex_poses(prefix="reindexed", group_col="input_poses", remove_layers=2)

@pytest.mark.parametrize("group_col,remove_layers,expect", [
    ("group2", None, ["pose1.pdb", "pose2.pdb", "pose3.pdb", "pose4.pdb", "pose5.pdb", "pose6.pdb"])
])


def test_reindex_poses(tmp_path, sample_df, group_col, remove_layers, expect):
    # check without force reindex
    sample_df = create_temp_poses(tmp_path, sample_df)
    p = Poses(poses=sample_df, work_dir=tmp_path.as_posix())

    p.reindex_poses(prefix="reindexed", group_col=group_col, remove_layers=remove_layers)

    out_dir = tmp_path / "reindexed"
    # check if filenames match
    assert p.df["poses"].to_list() == [(out_dir / filename).as_posix() for filename in expect]


def test_duplicate_poses_creates_copies_and_updates_df(tmp_path, sample_df):
    sample_df = create_temp_poses(tmp_path, sample_df)
    p = Poses(poses=sample_df, work_dir=tmp_path)
    out = tmp_path / "dups"
    p.duplicate_poses(out.as_posix(), n_duplicates=3, overwrite=True)
    # df length == 3 copies
    assert len(p.df) == len(sample_df) * 3


def test_reset_poses_unique_and_force(sample_df):
    p = Poses(poses=sample_df) 
    # without force_reset_df -> mismatch raises
    with pytest.raises(RuntimeError):
        p.reset_poses()

    # with force_reset_df -> rebuild df
    p.reset_poses(force_reset_df=True)
    assert set(p.df.columns) == set(["poses", "poses_description", "input_poses"])
    assert set(p.df["poses"]) == set(sample_df["input_poses"])


def test_set_motif_type_check_and_registers_column(sample_df, monkeypatch):
    # mock ResidueSelection type used for isinstance checks
    from protflow import poses as poses_mod
    class DummyRS: pass
    monkeypatch.setattr(poses_mod, "ResidueSelection", DummyRS)

    p = Poses(poses=sample_df)
    p.df["motifs"] = [DummyRS() for _ in p.poses_list()]
    p.set_motif("motifs")
    assert "motifs" in p.motifs

    p.df.loc[0, "motifs"] = "not_a_motif"
    with pytest.raises(TypeError):
        p.set_motif("motifs")


def test_convert_pdb_to_fasta_writes_files_and_optionally_updates(tmp_path, monkeypatch):
    # input .pdbs
    f1 = tmp_path / "a.pdb"; f1.write_text("ATOM")
    f2 = tmp_path / "b.pdb"; f2.write_text("ATOM")
    p = Poses(work_dir=tmp_path.as_posix())
    p.df = pd.DataFrame({
        "input_poses": [f1.as_posix(), f2.as_posix()],
        "poses": [f1.as_posix(), f2.as_posix()],
        "poses_description": ["a","b"]
    })
    # mock sequence extraction chain
    with patch("protflow.poses.load_structure_from_pdbfile") as load_mock, \
         patch("protflow.poses.get_sequence_from_pose", side_effect=["AAAA", "BBBB", "CCCC", "DDDD"]) as seq_mock: # 4 elements in side effect because convert_pdb is called twice!
        p.convert_pdb_to_fasta(prefix="conv", update_poses=False)
        fasta_dir = tmp_path / "conv_fasta_location"
        assert (fasta_dir / "a.fasta").read_text().strip().endswith("AAAA")
        assert (fasta_dir / "b.fasta").read_text().strip().endswith("BBBB")
        assert "conv_fasta_location" in p.df.columns[3]  # new column added

        # update_poses=True replaces poses with fasta paths
        p.convert_pdb_to_fasta(prefix="conv2", update_poses=True)
        assert all(Path(x).suffix == ".fasta" for x in p.df["poses"])


def test_convert_pdb_to_fasta_raises_if_not_pdb(tmp_path):
    f = tmp_path / "a.fa"; f.write_text(">a\nAAAA")
    p = Poses(work_dir=tmp_path.as_posix())
    p.df = pd.DataFrame({
        "input_poses": [f.as_posix()],
        "poses": [f.as_posix()],
        "poses_description": ["a"]
    })
    with pytest.raises(RuntimeError):
        p.convert_pdb_to_fasta(prefix="x")


def test_filter_poses_by_rank_writes_filtered_file(tmp_path, sample_df):
    p = Poses(poses=sample_df, work_dir=tmp_path.as_posix())
    p.filter_poses_by_rank(n=1, score_col="score", prefix="top", plot=False, overwrite=True, ascending=False)
    out = tmp_path / "filter" / "top_filter.json"  # default storage_format is json
    assert out.is_file()
    # df should now be filtered to one row (highest score overall)
    assert len(p.df) == 1
    assert p.df["score"].iloc[0] == sample_df["score"].max()


def test_filter_poses_by_value_prefix_requires_workdir(sample_df):
    p = Poses(poses=sample_df)  # no work_dir -> no filter_dir
    with pytest.raises(AttributeError):
        p.filter_poses_by_value(score_col="s", value=1.5, operator=">", prefix="cut")


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
    df["poses"] = [pose.as_posix() for pose in df["poses"]]

    return df