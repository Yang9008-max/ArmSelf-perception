import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.artifacts import checked_target, restore, sha256


class ArtifactRestoreTests(unittest.TestCase):
    def test_restore_preserves_existing_research_and_verifies_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = 'outputs/run/frozen.pt'
            archive = root/'release.zip'
            payload = b'checkpoint bytes'
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr(name, payload)
            bundle = dict(asset=archive.name, sha256=sha256(archive), files=[
                dict(path=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())])
            restore(archive, bundle, root)
            restore(archive, bundle, root)
            (root/name).write_bytes(b'local new experiment')
            with self.assertRaisesRegex(FileExistsError, 'Preserving different existing file'):
                restore(archive, bundle, root)
            self.assertEqual((root/name).read_bytes(), b'local new experiment')
            archive.write_bytes(b'corrupted download')
            with self.assertRaisesRegex(ValueError, 'Archive SHA-256 mismatch'):
                restore(archive, bundle, root)

    def test_archive_paths_cannot_escape_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('../outside', 'outputs/../../outside', '/outputs/absolute',
                         'outputs/C:/escape', 'outputs/..\\escape'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    checked_target(root, name)
