import ast
import unittest
from pathlib import Path


RL_ROOT = Path(__file__).resolve().parents[1]
DOOR_SOURCE = RL_ROOT / "envs/door/door_open_homie.py"
COMMAND_SOURCE = RL_ROOT / "envs/base_task/b2z1_command_base.py"
DELTA_ACTION_SOURCE = RL_ROOT / "envs/base_task/delta_action_base.py"
ENV_CONFIG = RL_ROOT / "config/env/door_open_b2z1.yaml"
EXP_CONFIG = RL_ROOT / "config/exp/wbmanip/door_open_b2z1_lstm.yaml"
OBS_CONFIG = RL_ROOT / "config/obs/wbmanip/door_open_b2z1.yaml"
REWARD_CONFIG = RL_ROOT / "config/rewards/wbmanip/reward_door_open_b2z1.yaml"
ROBOT_CONFIG = RL_ROOT / "config/robot/b2z1/b2z1.yaml"


def class_method_names(source_path, class_name):
    module = ast.parse(source_path.read_text())
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.name
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def class_method_node(source_path, class_name, method_name):
    module = ast.parse(source_path.read_text())
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )


class B2Z1PureRLContractTest(unittest.TestCase):
    def test_door_task_exposes_six_explicit_stages_without_pregrasp(self):
        task_source = DOOR_SOURCE.read_text()
        env_config = ENV_CONFIG.read_text()
        six_stage_source = ast.get_source_segment(
            task_source,
            next(
                node
                for node in ast.parse(task_source).body
                if isinstance(node, ast.ClassDef) and node.name == "B2Z1DoorSixStage"
            ),
        )
        self.assertIn("class B2Z1DoorSixStage", task_source)
        self.assertNotIn("class B2Z1DoorSevenStage", task_source)
        self.assertIn("STAGE_WALK_TO_DOOR = 0", task_source)
        self.assertIn("STAGE_GRASP = 1", six_stage_source)
        self.assertNotIn("STAGE_PREGRASP", six_stage_source)
        self.assertNotIn("pregrasp_target", six_stage_source)
        self.assertIn(
            "max_stage_time: [1000, 1000, 500, 750, 1500, 1000]",
            env_config,
        )
        self.assertIn(
            "stage_reward_scale: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]",
            env_config,
        )

    def test_environment_has_no_scripted_policy_api(self):
        methods = class_method_names(DOOR_SOURCE, "B2Z1DoorSixStage")
        self.assertNotIn("get_policy_action_reference", methods)
        self.assertNotIn("get_policy_action_supervision", methods)
        self.assertFalse(
            any("reference" in name for name in methods),
            sorted(name for name in methods if "reference" in name),
        )

    def test_door_configuration_has_no_scripted_action_controls(self):
        env_config = ENV_CONFIG.read_text()
        exp_config = EXP_CONFIG.read_text()
        for forbidden in (
            "reference_",
            "action_residual_obs_key",
            "action_residual_stage_masks",
            "policy_action_supervision_coef",
            "policy_action_supervision_huber_beta",
        ):
            self.assertNotIn(forbidden, env_config + exp_config)

    def test_exploration_uses_one_stage_independent_action_semantics(self):
        exp_config = EXP_CONFIG.read_text()
        self.assertNotIn("action_noise_stage_obs_key", exp_config)
        self.assertNotIn("action_noise_stage_scales", exp_config)
        self.assertIn(
            "max_noise_std: [0.20, 0.20, 0.12, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.10]",
            exp_config,
        )
        self.assertIn(
            "temporal_action_noise_correlated_fraction: [0.9, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
            exp_config,
        )
        self.assertIn("state_dependent_temporal_action_noise: True", exp_config)
        self.assertIn(
            "state_dependent_temporal_action_noise_max_fraction: [0.99, 0.99, 0.99, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
            exp_config,
        )

    def test_observation_and_reward_do_not_carry_script_targets(self):
        observation_config = OBS_CONFIG.read_text()
        reward_config = REWARD_CONFIG.read_text()
        self.assertIn("b2z1_task_geometry: 26", observation_config)
        for forbidden in (
            "action_reference",
            "policy_action_supervision",
            "approach_base_command",
            "approach_ee_command",
            "command_rate",
            "command_limit",
        ):
            self.assertNotIn(forbidden, observation_config + reward_config)

    def test_stage_is_visible_to_actor_and_critic_as_six_way_one_hot(self):
        observation_config = OBS_CONFIG.read_text()
        env_config = ENV_CONFIG.read_text()
        self.assertIn("stage,", observation_config)
        self.assertGreaterEqual(observation_config.count("stage,"), 2)
        self.assertIn(
            "stage: ${eval:'len(${env.config.max_stage_time})'}",
            observation_config,
        )
        self.assertIn(
            "max_stage_time: [1000, 1000, 500, 750, 1500, 1000]",
            env_config,
        )

    def test_fixed_stage_training_uses_procedural_states_without_snapshot_source(self):
        task_source = DOOR_SOURCE.read_text()
        env_config = ENV_CONFIG.read_text()
        self.assertIn("staged_reset_fixed_stage: null", env_config)
        self.assertIn("staged_reset_initial_state_source: null", env_config)
        self.assertIn("staged_reset_checkpoint_samples_per_env: 16", env_config)
        self.assertIn(
            "return (self.STAGE_GRASP, self.STAGE_PUSH_OPEN, self.STAGE_THROUGH)",
            task_source,
        )
        self.assertIn(
            "return (self.STAGE_OPEN_LATCH, self.STAGE_REPOSITION)",
            task_source,
        )

    def test_failed_steady_inverse_is_not_available_to_the_task(self):
        task_source = DOOR_SOURCE.read_text()
        command_source = COMMAND_SOURCE.read_text()
        for forbidden in (
            "steady_inverse",
            "external_joint_effort",
            "_compute_b2z1_steady_ee_command_world",
        ):
            self.assertNotIn(forbidden, task_source + command_source)

    def test_arm_ik_uses_lab30_orientation_error_without_extra_gain(self):
        command_source = COMMAND_SOURCE.read_text()
        robot_config = ROBOT_CONFIG.read_text()
        self.assertNotIn("lowlevel_ik_orientation_gain", command_source + robot_config)

        refresh_target = class_method_node(
            COMMAND_SOURCE,
            "B2Z1CommandBase",
            "_refresh_b2z1_arm_target",
        )
        refresh_source = ast.get_source_segment(command_source, refresh_target)
        self.assertIn(
            "drot = self._orientation_error_xyzw(goal_quat, ee_orn)",
            refresh_source,
        )

    def test_all_ten_actions_share_one_stage_independent_command_map(self):
        env_config = ENV_CONFIG.read_text()
        self.assertIn(
            "delta_action_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]",
            env_config,
        )
        self.assertNotIn("zero_vel:", env_config)
        self.assertNotIn("zero_finger:", env_config)

        mapper = class_method_node(
            DOOR_SOURCE,
            "B2Z1DoorSixStage",
            "_canonical_action_to_physical_commands",
        )
        mapper_names = {
            node.id for node in ast.walk(mapper) if isinstance(node, ast.Name)
        }
        mapper_attributes = {
            node.attr for node in ast.walk(mapper) if isinstance(node, ast.Attribute)
        }
        self.assertNotIn("stage_buf", mapper_names | mapper_attributes)
        self.assertNotIn("door_open_lr", mapper_names | mapper_attributes)

        mapper_source = ast.get_source_segment(DOOR_SOURCE.read_text(), mapper)
        for action_slice in (
            "canonical_action[:, :3]",
            "canonical_action[:, 3:6]",
            "canonical_action[:, 6:9]",
            "canonical_action[:, 9]",
        ):
            self.assertIn(action_slice, mapper_source)

        delta_step = class_method_node(DELTA_ACTION_SOURCE, "DeltaActionBase", "step")
        delta_step_source = ast.get_source_segment(
            DELTA_ACTION_SOURCE.read_text(), delta_step
        )
        self.assertIn(
            "self._integrate_policy_delta_actions(", delta_step_source
        )


if __name__ == "__main__":
    unittest.main()
