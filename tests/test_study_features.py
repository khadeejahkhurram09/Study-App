import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("app_module", ROOT / "app.py")
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)


def test_grade_level_options_cover_primary_to_a_level():
    assert "Kindergarten" in app.GRADE_LEVEL_OPTIONS
    assert "O Level Year 1" in app.GRADE_LEVEL_OPTIONS
    assert "A Level" in app.GRADE_LEVEL_OPTIONS


def test_intervention_plan_flags_low_performance():
    plan = app.build_intervention_plan(38, 60)
    assert plan[0]["type"] == "past_papers"
    assert "guided revision" in plan[0]["message"].lower()


def test_intervention_plan_is_empty_when_threshold_is_met():
    plan = app.build_intervention_plan(80, 60)
    assert plan == []
