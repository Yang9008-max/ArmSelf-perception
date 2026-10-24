import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from scripts.artifacts import checked_target, download, download_url, restore, sha256


class ArtifactRestoreTests(unittest.TestCase):
    def test_interrupted_download_keeps_received_bytes_and_resumes_at_offset(self):
        class Dropped(io.BytesIO):
            status = 200

            def read1(self, size=-1):
                if self.tell() == 0:
                    return super().read1(3)
                raise OSError('connection dropped mid-download')

        class Remaining(io.BytesIO):
            status = 206

        payload = b'abcdef'
        bundle = dict(asset='model.zip', bytes=6, sha256=hashlib.sha256(payload).hexdigest(),
                      url='https://example.invalid/model.zip')
        with tempfile.TemporaryDirectory() as directory:
            with patch('scripts.artifacts.urlopen', side_effect=[Dropped(payload), Remaining(payload[3:])]) as opener, \
                    patch('scripts.artifacts.time.sleep'):
                path = download(bundle, Path(directory))
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(opener.call_args_list[1].args[0].get_header('Range'), 'bytes=3-')

    def test_public_download_selects_the_named_complete_api_asset(self):
        bundle = dict(url='https://github.com/owner/repo/releases/download/snapshot/model.zip',
                      asset='model.zip', bytes=42)
        response = dict(assets=[dict(name='other.zip', size=42, state='uploaded', url='wrong'),
                                dict(name='model.zip', size=42, state='uploaded',
                                     url='https://api.github.com/repos/owner/repo/releases/assets/7')])
        with patch('scripts.artifacts.urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
            self.assertEqual(download_url(bundle), response['assets'][1]['url'])
        response['assets'][1]['size'] = 41
        with patch('scripts.artifacts.urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                download_url(bundle)

    def test_api_unavailable_falls_back_to_original_public_url(self):
        bundle = dict(url='https://github.com/owner/repo/releases/download/snapshot/model.zip',
                      asset='model.zip', bytes=42)
        with patch('scripts.artifacts.urlopen', side_effect=OSError('API rate limit')):
            self.assertEqual(download_url(bundle), bundle['url'])

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
