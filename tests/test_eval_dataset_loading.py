import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluate_math", ROOT / "eval" / "evaluate_math.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_load_local_eval_jsonl_and_normalize_examples():
    rows = module.load_jsonl_or_json(str(ROOT / "tests" / "fixtures" / "custom_eval.jsonl"))
    assert len(rows) == 2

    problem, answer, question_id = module.normalize_eval_example(rows[0], "aimo", 0)
    assert problem == "What is 1+1?"
    assert answer == "2"
    assert question_id == "c1"
    assert module.eval_metadata(rows[0]) == {"perturbation": "clean"}

    problem, answer, question_id = module.normalize_eval_example(rows[1], "rrb-aime", 1)
    assert problem == "What is 2+2?"
    assert answer == "4"
    assert question_id == 1
