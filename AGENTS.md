# Repository Agent Instructions

- Keep YOLO26 TensorFlow changes focused on object detection parity, training stability, and reproducible benchmarking.
- After every completed code, script, test, or documentation change, run relevant checks/tests when feasible.
- After every completed change, stage, commit, and push the current branch unless the user explicitly says not to commit or not to push.
- Do not add CPU fallback paths to Linux COCO training or benchmark runners; they should fail fast without TensorFlow GPU.
