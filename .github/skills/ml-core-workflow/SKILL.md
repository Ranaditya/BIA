---
name: ml-core-workflow
user-invocable: true
description: "End-to-end machine learning workflow controller for CSV-based projects. Use when users ask to build ML projects step by step, plan first then execute after approval, document ML tasks, enforce approval gates, create feature branches, or prepare PR-ready summaries across supervised or unsupervised learning. Trigger phrases include: build ML pipeline from CSV, develop and document ML project, step-by-step ML implementation, supervised vs unsupervised ML workflow, create ML project structure, and PR-ready ML documentation."
---

# ML Core Workflow

Use this skill as the process controller for machine learning tasks.

## Use When

- User provides a dataset and instructions to build an ML solution.
- User asks for step-by-step planning and approval before execution.
- User asks for ML-specific documentation and PR-ready output.
- User needs consistent workflow standards across multiple ML techniques.

## Do Not Use When

- User only asks for a single formula or quick conceptual definition.
- User asks only for cloud deployment without model-building workflow.
- A technique-specific skill is explicitly requested and applicable.

## Scope

- Process orchestration for supervised and unsupervised ML tasks.
- Planning, approval gates, structure standards, documentation standards.
- Hand-off routing to technique-specific skills.

## Required Intake

Collect or confirm:

1. Dataset path
2. Target column if supervised
3. Technique intent (regression, classification, clustering, or unknown)
4. Evaluation expectation and deliverables

## Execution Rules

1. Propose one step at a time with recommendation.
2. Wait for user approval before each execution step.
3. For new work, create a feature branch first.
4. Keep code modular where appropriate.
5. Keep documentation updated with each significant change.
6. Validate changes by running the workflow.

## Standard Delivery Pattern

1. Plan with assumptions and options.
2. Execute approved step only.
3. Report outcome and next recommended step.
4. Continue approval loop until completion.

## Documentation Standard

For each ML task, include:

- Objective and dataset details
- Preprocessing decisions
- Modeling approach and rationale
- Metrics and interpretation
- How to run and verify
- PR summary with what changed and how it was validated

## Routing Guidance

- If task is supervised regression, use ml-supervised-regression-workflow.
- If task is supervised classification, route to a classification-focused skill when available.
- If task is unsupervised clustering or dimensionality reduction, route to unsupervised skill when available.
- If ambiguous, ask one concise clarifying question and continue.

## Trigger Examples

- "I have a CSV and need a full ML project step by step."
- "Plan and document an ML workflow, then execute after each approval."
- "Set up a supervised/unsupervised ML project with proper README and PR notes."
