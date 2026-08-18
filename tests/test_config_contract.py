import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DataModuleConfigTest(unittest.TestCase):
    def test_datamodule_does_not_mutate_structured_config(self):
        from LipModel_MFM.datamodule.data_module_CCS import DataModule_CCS

        cfg = SimpleNamespace(
            trainer=SimpleNamespace(num_nodes=1),
            data=SimpleNamespace(num_workers=0),
        )
        module = DataModule_CCS(cfg)
        self.assertIs(module.cfg, cfg)
        self.assertFalse(hasattr(cfg, "gpus"))

    def test_video_modality_does_not_construct_audio_transform(self):
        from LipModel_MFM.datamodule.data_module_CCS import DataModule_CCS

        cfg = SimpleNamespace(
            trainer=SimpleNamespace(num_nodes=1),
            data=SimpleNamespace(modality="video"),
            decode=SimpleNamespace(snr_target=None),
        )
        module = DataModule_CCS(cfg)

        with mock.patch(
            "LipModel_MFM.datamodule.data_module_CCS.AudioTransform",
            side_effect=AssertionError("audio transform should stay disabled"),
        ), mock.patch(
            "LipModel_MFM.datamodule.data_module_CCS.VideoTransform",
            return_value="video-transform",
        ):
            audio_transform, video_transform = module._transforms("test")

        self.assertIsNone(audio_transform)
        self.assertEqual(video_transform, "video-transform")


if __name__ == "__main__":
    unittest.main()
