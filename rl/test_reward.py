import json

import pytest

from rl.reward import compute_reward, iou_reward
from scripts.score import parse_output


def gt(*boxes):
    return [{"label": lab, "bbox": tuple(b)} for lab, b in boxes]


def out(*boxes):
    return json.dumps([{"label": lab, "bbox_2d": list(b)} for lab, b in boxes])


POTHOLE = ("pothole", (100, 100, 200, 200))
CRACK = ("crack", (500, 500, 700, 600))


def test_perfect():
    assert compute_reward(out(POTHOLE, CRACK), gt(POTHOLE, CRACK)) == pytest.approx(1.0)


def test_broken_json():
    assert compute_reward('[{"label": "pothole", "bbox_2d": [1, 2', gt(POTHOLE)) == 0.0
    assert compute_reward("Based on the image, there is a pothole.", gt(POTHOLE)) == 0.0


def test_wrong_label():
    assert compute_reward(out(("crack", POTHOLE[1])), gt(POTHOLE)) == pytest.approx(0.2)


def test_extra_box_penalized():
    assert compute_reward(out(POTHOLE, CRACK), gt(POTHOLE)) == pytest.approx(0.6)


def test_both_empty():
    assert compute_reward("[]", []) == 1.0


def test_empty_prediction_on_damaged_image():
    # Hack "luôn trả []": chỉ nhận điểm định dạng
    assert compute_reward("[]", gt(POTHOLE, CRACK)) == pytest.approx(0.2)


def test_giant_box_gets_no_iou():
    # Hack "box khổng lồ": IoU với box nhỏ < 0.5 nên không được ghép
    assert compute_reward(out(("pothole", (0, 0, 1000, 1000))), gt(POTHOLE)) == pytest.approx(0.2)


def test_low_iou_not_matched_and_partial_iou_scaled():
    shifted = ("pothole", (150, 100, 250, 200))  # IoU = 1/3 < 0.5
    assert compute_reward(out(shifted), gt(POTHOLE)) == pytest.approx(0.2)
    close = ("pothole", (110, 100, 210, 200))    # IoU = 90/110
    assert compute_reward(out(close), gt(POTHOLE)) == pytest.approx(0.2 + 0.8 * 90 / 110)


def test_greedy_one_to_one_matching():
    # Hai dự đoán trùng một GT: chỉ một cặp được tính, cái còn lại là box thừa
    assert compute_reward(out(POTHOLE, POTHOLE), gt(POTHOLE)) == pytest.approx(0.6)


def test_markdown_fence_accepted():
    assert compute_reward("```json\n" + out(POTHOLE) + "\n```", gt(POTHOLE)) == pytest.approx(1.0)


def test_trl_interface_conversational():
    answer = out(POTHOLE)
    completions = [[{"role": "assistant", "content": answer}], [{"role": "assistant", "content": "oops"}], answer]
    assert iou_reward(completions, answer=[answer] * 3) == pytest.approx([1.0, 0.0, 1.0])
    assert parse_output(answer)[0]["label"] == "pothole"
