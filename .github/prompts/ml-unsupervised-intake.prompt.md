---
name: "ML Unsupervised Intake"
description: "Intake and plan unsupervised ML work from CSV with approval-gated execution and documentation."
argument-hint: "Dataset path, objective, preferred technique (clustering/PCA), instruction set, deliverables"
agent: "agent"
---
Use the `ml-core-workflow` skill for this task.

Plan and execute an unsupervised machine learning workflow from CSV data.

Inputs:
- Dataset path: ${input:dataset_path}
- Objective (segmentation, anomaly detection, dimensionality reduction, etc.): ${input:objective}
- Preferred technique (clustering, PCA, or auto): ${input:technique}
- Optional instruction set (step-by-step rules from user): ${input:instruction_set}
- Expected deliverables: ${input:deliverables}

Execution requirements:
1. Propose one recommended step at a time.
2. Wait for approval before each execution step.
3. Create a feature branch first for new work.
4. Document preprocessing, feature handling, and unsupervised evaluation rationale.
5. Produce PR-ready summary and validation notes.

If mandatory inputs are missing, ask concise clarification questions first.
