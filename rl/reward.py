"""Reward IoU kiểm chứng được cho GRPO (E5). Python thuần, không cần GPU."""
from scripts.score import iou_xyxy, parse_output

IOU_THRESHOLD = 0.5


def compute_reward(completion, gt_boxes):
    """completion: chuỗi model sinh ra. gt_boxes: list[{"label", "bbox": (x1, y1, x2, y2)}] thang 0–1000.

    0.0 nếu JSON hỏng; 1.0 nếu cả GT và dự đoán đều rỗng; ngược lại
    0.2 + 0.8 * tổng IoU các cặp khớp / max(số GT, số dự đoán, 1).
    """
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0
    total_iou = 0.0
    for lab in {g["label"] for g in gt_boxes} | {p["label"] for p in pred}:
        g = [x["bbox"] for x in gt_boxes if x["label"] == lab]
        p = [x["bbox"] for x in pred if x["label"] == lab]
        pairs = sorted(((iou_xyxy(a, b), i, j) for i, a in enumerate(g) for j, b in enumerate(p)), reverse=True)
        used_g, used_p = set(), set()
        for v, i, j in pairs:  # ghép đôi tham lam theo IoU giảm dần, cùng lớp
            if v < IOU_THRESHOLD:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            total_iou += v
    # Mẫu số max(...) phạt box bịa (thừa) lẫn box bỏ sót
    return 0.2 + 0.8 * total_iou / max(len(gt_boxes), len(pred), 1)


def _completion_text(completion):
    """trl truyền completion dạng chuỗi, hoặc dạng hội thoại [{"role": "assistant", "content": ...}]."""
    if isinstance(completion, str):
        return completion
    content = completion[-1]["content"]
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content


def iou_reward(completions, answer, **kwargs):
    """Hàm reward theo giao diện trl.GRPOTrainer; `answer` là cột đáp án JSON (thang 0–1000) của dataset."""
    return [compute_reward(_completion_text(c), parse_output(a)) for c, a in zip(completions, answer)]
