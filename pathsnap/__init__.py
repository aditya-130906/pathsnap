"""PathSnap shared preprocessing and training package.

Modules
-------
data        strict readers for .track/.nodes/.arcs/.route (route only when explicitly requested)
geo         per-trip metric projection and point-to-segment geometry
candidates  inference-compatible candidate arcs and features (never reads .route)
labels      SEPARATE observation-label approximation from .route (training/evaluation only)
build       builds reproducible candidate/label CSVs for the train and student_test partitions
eda         dataset inventory and diagnostics
metrics     candidate- and observation-level metrics (numpy only)
training    shared five-model training: GroupKFold CV, student-test evaluation, artifacts
"""
