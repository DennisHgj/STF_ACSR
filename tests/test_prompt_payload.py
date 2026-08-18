import unittest
from pathlib import Path
from types import SimpleNamespace

from CSPM.CustomizedPromptTemplate import (
    Frame,
    HandRecognition,
    generate_recognition_single,
)


class FakeCompletions:
    def __init__(self, parsed):
        self.parsed = parsed
        self.request = None

    def parse(self, **request):
        self.request = request
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(parsed=self.parsed))
            ],
            usage=SimpleNamespace(total_tokens=321),
        )


def fake_client(parsed):
    completions = FakeCompletions(parsed)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return client, completions


class PromptPayloadTest(unittest.TestCase):
    def setUp(self):
        self.support_set = (
            Path(__file__).parents[1] / "CSPM" / "support_set"
        )

    def test_supported_image_and_structured_output_schema(self):
        parsed = HandRecognition(
            recog_results=[
                Frame(
                    frame_id=12,
                    hand_position=3,
                    hand_shape=4,
                    reasoning_process=(
                        "The fingers point to the chin; all are straight."
                    ),
                )
            ]
        )
        client, completions = fake_client(parsed)

        results, usage = generate_recognition_single(
            ["ZmFrZS1qcGVn"],
            support_set_path=str(self.support_set),
            frame_ids=[12],
            model="gpt-4o",
            client=client,
        )

        self.assertEqual(results[0]["hand_shape"], 4)
        self.assertEqual(usage, [321])
        request = completions.request
        self.assertEqual(request["model"], "gpt-4o")
        self.assertIs(request["response_format"], HandRecognition)
        content = request["messages"][1]["content"]
        image_parts = [
            item for item in content if item["type"] == "image_url"
        ]
        self.assertEqual(len(image_parts), 81)
        self.assertTrue(
            all(
                item["image_url"]["url"].startswith(
                    "data:image/jpeg;base64,"
                )
                and item["image_url"]["detail"] == "high"
                for item in image_parts
            )
        )

    def test_frame_id_mismatch_is_rejected(self):
        parsed = HandRecognition(
            recog_results=[
                Frame(
                    frame_id=99,
                    hand_position=0,
                    hand_shape=0,
                    reasoning_process="Mismatch fixture.",
                )
            ]
        )
        client, _ = fake_client(parsed)
        with self.assertRaisesRegex(ValueError, "frame IDs"):
            generate_recognition_single(
                ["ZmFrZS1qcGVn"],
                support_set_path=str(self.support_set),
                frame_ids=[12],
                client=client,
            )


if __name__ == "__main__":
    unittest.main()
