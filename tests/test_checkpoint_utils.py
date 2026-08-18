import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CheckpointUtilsTest(unittest.TestCase):
    def test_extracts_lightning_state_dict(self):
        from LipModel_MFM.checkpoint_utils import extract_state_dict

        tensor = torch.tensor([1.0])
        state = extract_state_dict(
            {"state_dict": {"model.weight": tensor}}
        )
        self.assertIs(state["model.weight"], tensor)

    def test_strips_only_inner_model_prefix(self):
        from LipModel_MFM.checkpoint_utils import extract_model_state_dict

        state = extract_model_state_dict(
            {
                "state_dict": {
                    "model.encoder.weight": torch.tensor([1.0]),
                    "other": torch.tensor([2.0]),
                }
            }
        )
        self.assertEqual(list(state), ["encoder.weight"])


if __name__ == "__main__":
    unittest.main()
