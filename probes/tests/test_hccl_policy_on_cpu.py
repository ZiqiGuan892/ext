"""Check the supplied HCCL environment contract without importing the NPU stack."""

import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "hccl_policy.py"
spec = importlib.util.spec_from_file_location("hccl_policy_under_test", SOURCE)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


@pytest.mark.parametrize("profile,value", [("aiv", "AIV"), ("host", "HOST"), ("host_ts", "HOST_TS"),
                                          ("unset", None), ("custom", "AI_CPU")])
def test_declared_policy_matches(profile, value):
    env = {"HUANYU_HCCL_PROFILE": profile}
    if value is not None:
        env["HCCL_OP_EXPANSION_MODE"] = value
    assert policy.assert_hccl_policy(env, quiet_ok=True)["ok"]


@pytest.mark.parametrize("profile,value", [("aiv", "HOST"), ("aiv", None), ("unset", "AIV"),
                                          ("custom", "AICPU"), ("invalid", "AIV")])
def test_mismatches_fail_before_training(profile, value):
    env = {"HUANYU_HCCL_PROFILE": profile}
    if value is not None:
        env["HCCL_OP_EXPANSION_MODE"] = value
    with pytest.raises(RuntimeError, match="策略校验失败"):
        policy.assert_hccl_policy(env)


def test_undeclared_profile_does_not_enforce_policy():
    assert policy.assert_hccl_policy({"HCCL_OP_EXPANSION_MODE": "foreign-value"})["ok"]
