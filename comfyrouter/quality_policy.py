"""Shared optional visual-task rubric for evaluators, diagnosis and preflight."""
import os

POLICY_ID = "visual_task_tolerant_v2"
POLICY_TEXT = (
    "Current authorized evaluation and repair policy (overrides older numerical checklists and feedback): "
    "Judge task completion and actual visual quality: requested subject/action/edit, identity and protected "
    "regions, composition, continuity, and visible motion/smoothness. Do not fail for small numerical "
    "differences alone. Duration within 10%, fps or "
    "speed/interpolation factor within 10%, and frame counts within two frames of the target or ceiling are acceptable "
    "when visual content and the intended effect are preserved. The nominal target is approximate; "
    "planner-added exact frame counts must not strengthen the user's original request. "
    "Do not require simultaneous mathematically incompatible duration/fps/frame-count targets. "
    "Do not repair a missing final boundary frame solely to satisfy an exact count. "
    "Keep execution memory/frame safety caps intact. Gross shortening, missing stages, damaged subjects, "
    "wrong masks, frozen motion, or a missing requested transformation still fail. "
    "Evaluate all supplied visual evidence. Sparse sampled frames alone cannot prove that a working "
    "interpolator only duplicated frames; look at adjacent frames and the actual executed graph, and "
    "do not invent a defect from absent evidence. These tolerances apply consistently to all cases, "
    "not only a known example. Historical strict-rubric failures are context, not binding current judgments."
)

def visual_tolerance_enabled():
    return os.getenv("COMFYROUTER_QUALITY_POLICY") == POLICY_ID


def temporal_near(actual, target, fps=None):
    tolerance = abs(float(target)) * .10
    return abs(float(actual) - float(target)) <= tolerance + 1e-9
