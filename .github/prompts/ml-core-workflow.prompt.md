---
name: "ML Core Workflow"
description: "Plan and execute an ML project step by step with approvals, branching, and documentation."
argument-hint: "Dataset path, technique intent (supervised/unsupervised), deliverables"
agent: "agent"
---
Use the `ml-core-workflow` skill for this task.

Build and document a machine learning project using an approval-gated workflow.

Inputs:
- Dataset path: ${input:dataset_path}
- Technique intent: ${input:technique_intent}
- Target column (if supervised): ${input:target_column}
- Expected deliverables: ${input:deliverables}

Execution requirements:
1. Propose one recommended step at a time.
2. Wait for approval before executing each step.
3. Create a feature branch first for new work.
4. Keep outputs PR-ready with concise validation notes.

If required inputs are missing, ask focused clarification questions first.
