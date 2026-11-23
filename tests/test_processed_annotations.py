from pathlib import Path
import tempfile
import unittest
from multiview_data.processed import ProcessedVCIDataset, ANNOTATION_PATHS


class AnnotationAdapterTests(unittest.TestCase):
    def test_optional_products_and_explicit_label_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = root/'one'
            (subject/'images').mkdir(parents=True)
            for name, relative in ANNOTATION_PATHS.items():
                path = subject/relative
                path.parent.mkdir(parents=True,exist_ok=True)
                if name in ('semantic_general','semantic_detailed','semantic_confidence'):
                    path.mkdir()
                else:
                    path.write_text('{}')
            required = tuple(ANNOTATION_PATHS)
            data = ProcessedVCIDataset(root,required_keys=required).subjects[0]
            self.assertFalse(data['missing_keys'])
            self.assertEqual(data['labels'],data['semantic_general'])
            (subject/'labels').mkdir()
            data = ProcessedVCIDataset(root,required_keys=required).subjects[0]
            self.assertEqual(data['labels'],subject/'labels')
