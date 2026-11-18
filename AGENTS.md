# PathSnap working instructions

Implement only the stage explicitly approved in the current user request. Do not
advance to the next stage without explicit approval. Keep all data processing,
ML and future server operations on Laptop 1; Laptop 2 is an independent HTTP UI.
Never alter raw source files, create artificial labels/results or leak references
into candidate/inference decisions. Preserve original node, arc and observation IDs.

For EVERY stage, automatically maintain all three READMEs:
`README.md`, `laptop1_backend/README.md`, `laptop2_frontend/README.md`.
Preserve previous-stage explanations, describe new modules/functions and their
inputs/outputs, update exact Windows commands, dependencies, actual outputs,
verified test/performance results, limitations, remaining work and the progress
checklist. Add a short changelog entry. Distinguish planned functionality from
working code. No stage is complete until these documents accurately match the
executed implementation. Never claim unexecuted tests or jobs succeeded.

For every remaining stage, also create/update a separate detailed
docs/steps/STEP_XX_README.md with objective, rationale, algorithms, tools, files,
function input/output contracts, execution flow, exact PowerShell commands,
actual outputs/results/tests, issues, assumptions, completion and next-stage
connection. Link all stage references from all three main READMEs. Preserve
historical stage explanations and clearly identify current status. Cleaning
stages require a measured method/purpose/technique/affected-record/handling table.
