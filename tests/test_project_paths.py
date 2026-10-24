from pathlib import Path
import tempfile
import unittest

from project_paths import recorded_path, portable_path, PROJECT_ROOT


class RecordedPathTests(unittest.TestCase):
    def test_windows_and_posix_recordings_relocate_without_editing_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root/'outputs/embodiment/run/model.pt'
            target.parent.mkdir(parents=True)
            target.write_bytes(b'frozen checkpoint')
            for old in (r'D:\old\ArmSelf-perception\outputs\embodiment\run\model.pt',
                        '/old/machine/outputs/embodiment/run/model.pt',
                        'outputs/embodiment/run/model.pt'):
                self.assertEqual(recorded_path(old, root), target)
            self.assertEqual(target.read_bytes(), b'frozen checkpoint')

    def test_new_checkout_wins_over_existing_original_checkout(self):
        with tempfile.TemporaryDirectory() as old, tempfile.TemporaryDirectory() as new:
            for root, content in ((Path(old), b'old'), (Path(new), b'new')):
                target = root/'outputs/run/metrics.json'
                target.parent.mkdir(parents=True)
                target.write_bytes(content)
            self.assertEqual(recorded_path(Path(old)/'outputs/run/metrics.json', new).read_bytes(), b'new')

    def test_external_existing_path_remains_usable_and_missing_path_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'external.pt'
            target.touch()
            self.assertEqual(recorded_path(target), target)
            with self.assertRaisesRegex(FileNotFoundError, 'Restore the experiment bundles'):
                recorded_path('outputs/not-an-installed-experiment', directory)

    def test_new_project_metadata_uses_posix_relative_paths(self):
        self.assertEqual(portable_path(PROJECT_ROOT/'outputs/run'), 'outputs/run')
