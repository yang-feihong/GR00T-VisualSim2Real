import unittest

from omegaconf import OmegaConf

from gr00t.rl.utils.config_utils import configure_headless_physics
from gr00t.rl.eval_agent_trl import make_json_serializable


def make_config(*, headless=True, render_results=False, cameras=False, physics_only=True):
    return OmegaConf.create(
        {
            "headless": headless,
            "simulator": {
                "config": {
                    "render_results": render_results,
                    "physics_only": physics_only,
                    "cameras": {"enable_cameras": cameras},
                }
            },
        }
    )


class HeadlessPhysicsConfigTest(unittest.TestCase):
    def test_physics_only_disables_visual_scene_objects(self):
        config = make_config()

        self.assertTrue(configure_headless_physics(config))
        self.assertTrue(config.simulator.config.disable_visual_markers)
        self.assertTrue(config.simulator.config.disable_visual_materials)

    def test_physics_only_rejects_rendering_modes(self):
        cases = [
            {"headless": False},
            {"render_results": True},
            {"cameras": True},
        ]
        for overrides in cases:
            with self.subTest(**overrides):
                config = make_config(**overrides)
                with self.assertRaisesRegex(ValueError, "physics_only requires"):
                    configure_headless_physics(config)

    def test_eval_metrics_are_json_serializable(self):
        import torch

        converted = make_json_serializable(
            {"rate": torch.tensor(0.5), "flags": torch.tensor([True, False])}
        )

        self.assertEqual(converted, {"rate": 0.5, "flags": [True, False]})


if __name__ == "__main__":
    unittest.main()
