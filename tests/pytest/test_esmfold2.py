"""CPU regressions for ESMFold2 input preparation and result integration."""

import ast
import copy
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from protflow.jobstarters import JobStarter
from protflow.poses import Poses
from protflow.tools.esmfold2 import ESMFold2, ESMFold2Params


def _poses(tmp_path):
    paths = []
    for name in ('first', 'second'):
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps({'sequences': [{'protein': {'id': 'A', 'sequence': 'ACDE', 'msa': None}}]}))
        paths.append(str(path))
    return Poses(poses=paths, work_dir=str(tmp_path / 'runs'))


def _runner(jobstarter=None):
    runner = ESMFold2.__new__(ESMFold2)
    runner.name = 'ESMFold2'
    runner.index_layers = 1
    runner.esmfold2_python = 'python'
    runner.esmfold2_path = 'run_esmfold2.py'
    runner.pre_cmd = ''
    runner.jobstarter = jobstarter
    return runner


def test_pose_options_are_independent_and_do_not_mutate_inputs(tmp_path):
    poses = _poses(tmp_path)
    overrides = [{'num_loops': 5, 'num_diffusion_samples': 2}, {}]
    poses.df['fold_options'] = overrides
    global_options = {'num_loops': 3, 'num_sampling_steps': 200}
    before = copy.deepcopy((global_options, overrides))
    files = _runner()._parse_poses(poses, str(tmp_path), 2, global_options, 'fold_options')
    records = {key: value for path in files for key, value in json.loads(Path(path).read_text()).items()}
    assert records['first']['options'] == {'num_loops': 5, 'num_sampling_steps': 200, 'num_diffusion_samples': 2}
    assert records['second']['options'] == global_options
    assert (global_options, overrides) == before


@pytest.mark.parametrize('options', [[], '', '--num_loops 3', 0])
def test_options_must_be_dictionaries(tmp_path, options):
    with pytest.raises(KeyError, match='dict'):
        _runner()._parse_poses(_poses(tmp_path), str(tmp_path), 1, options)


def test_invalid_pose_option_is_rejected(tmp_path):
    poses = _poses(tmp_path)
    poses.df['fold_options'] = [{}, None]
    with pytest.raises(KeyError, match='dict'):
        _runner()._parse_poses(poses, str(tmp_path), 1, {}, 'fold_options')


@pytest.mark.parametrize('ligand_type,values,expected', [
    ('smiles', ['CCO', 'CC(=O)O'], ['CCO', 'CC(=O)O']),
    ('ccd', ['ATP', 'ADP'], [['ATP'], ['ADP']]),
])
@pytest.mark.parametrize('from_column', [False, True])
def test_params_preserve_ligand_types_and_default_protein_fields(tmp_path, ligand_type, values, expected, from_column):
    poses = _poses(tmp_path)
    poses.df['ligand_value'] = values
    params = ESMFold2Params()
    params.add_protein(id='B', sequence='GGGG')
    params.add_ligand(id='Z', ligand='ligand_value' if from_column else values[0],
                      ligand_type=ligand_type, poses_cols=['ligand'] if from_column else None)
    params.generate_json_files(poses, str(tmp_path / 'params'))
    for i, path in enumerate(poses.poses_list()):
        entries = json.loads(Path(path).read_text())['sequences']
        assert entries[1]['protein']['msa'] is None
        assert entries[1]['protein']['modifications'] is None
        assert entries[2]['ligand'][ligand_type] == expected[i if from_column else 0]


@pytest.mark.parametrize('modifications', [None, [], [{'position': 0, 'ccd': 'MSE'}]])
def test_worker_parses_optional_modifications_without_model_imports(modifications):
    # Exercise the worker's parser without importing model weights or GPU packages.
    worker = Path(__file__).resolve().parents[2] / 'protflow/tools/runners_auxiliary_scripts/run_esmfold2.py'
    tree = ast.parse(worker.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'parse_seq_entry')
    namespace = {name: SimpleNamespace for name in ('ProteinInput', 'DNAInput', 'RNAInput', 'LigandInput', 'Modification')}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(worker), 'exec'), namespace)
    parsed = namespace['parse_seq_entry']({'protein': {'id': 'A', 'sequence': 'ACDE', 'modifications': modifications}})
    if modifications:
        assert parsed.modifications[0].ccd == 'MSE'
    else:
        assert parsed.modifications == modifications


class FakePredictionJobs(JobStarter):
    def __init__(self):
        self.max_cores = 2
        self.starts = 0

    def start(self, cmds, jobname, output_path, wait=True):
        self.starts += 1
        for cmd in cmds:
            args = shlex.split(cmd)
            batch = Path(args[args.index('--input_json') + 1])
            output = Path(args[args.index('--output_dir') + 1])
            rows = []
            for name, record in json.loads(batch.read_text()).items():
                for index in range(record['options'].get('num_diffusion_samples', 1)):
                    description = f'{name}_{index + 1:04d}'
                    location = output / f'{description}.cif'
                    location.write_text('data_prediction\n')
                    rows.append({'description': description, 'location': str(location), 'ptm': 0.8})
            pd.DataFrame(rows).to_json(output / f'{batch.stem}_out.json')


def test_run_merges_multiple_predictions_and_reuses_cached_scores(tmp_path):
    poses = _poses(tmp_path)
    poses.df['fold_options'] = [{'num_diffusion_samples': 2}, {}]
    original_df = poses.df.copy(deep=True)
    jobs = FakePredictionJobs()
    runner = _runner(jobs)
    result = runner.run(poses, 'prediction', options={'num_loops': 3}, pose_options='fold_options')
    assert result.df['poses_description'].tolist() == ['first_0001', 'first_0002', 'second_0001']
    assert result.df['prediction_ptm'].tolist() == [0.8, 0.8, 0.8]
    assert all(Path(path).is_file() for path in result.poses_list())
    fresh = Poses(poses=original_df, work_dir=str(tmp_path / 'runs'))
    cached = runner.run(fresh, 'prediction', options={'num_loops': 10})
    assert len(cached) == 3
    assert jobs.starts == 1
