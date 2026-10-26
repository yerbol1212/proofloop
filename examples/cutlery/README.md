# Fork and spoon capture preset

Use **one fork in LEFT** and **one spoon in RIGHT**. The [capture guide](../../docs/CAPTURE.md) gives the continuous recording sequence: correct placement, move the fork RIGHT, cover both objects, uncover and move the fork back LEFT. Keep AirPods, glasses and extra cutlery out of view.

`requirements.txt` contains the explicit placement requirements. `concepts.json` maps the inventory IDs to the concrete SAM concepts `fork` and `spoon`. `protocol.json` is an **authored, unapproved starter**, not a Spark-generated result or a tested physical scene. Its `unique_target: true` flags express proposed assumptions; the operator must verify that the scene contains one visually distinct target instance of each class before approving a session. This is a placement contract, not a model-inferred exact count.

No camera region file is included. After recording, inspect an extracted frame and define LEFT/RIGHT normalized rectangles for that actual displayed image. Do not copy the generated tools scene's calibration. The complete [session approval](../../docs/SESSIONS.md) binds reviewed frames, their timestamps, the protocol, concepts and those camera rectangles before live inference.

To draft the same requirements with Spark instead of using the authored starter:

```bash
.venv/bin/python -m proofloop.compiler \
  --requirements 'The unique fork must be located in left and the unique spoon in right.' \
  --objects fork spoon --regions left right \
  --predicate-kind location --unique-objects fork spoon
```

Generated-image feasibility results, if recorded separately, do not validate your phone, lighting, metal reflections or physical cutlery. The actual phone-camera run remains a separate check. The earlier screwdriver/pliers images and reports remain unchanged as historical evidence.
