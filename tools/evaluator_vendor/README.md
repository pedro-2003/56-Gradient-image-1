# Vendored evaluator (verbatim)

Unmodified copies of `validator/evaluation/{denoising_mse,image_denoising,image_test_data,
image_encoder,image_flow_adapter,image_artifacts,evaluation_logging}.py` and
`validator/evaluation/evaluators/diffusion.py` from rayonlabs/G.O.D at commit fb8cff2
(2026-09-23). Used only by `tools/local_evaluate.py` so G-PARITY can run in the venv without the
validator's docker image. The only change is the import prefix `validator.evaluation.` -> `evaluator_vendor.` (sed);
logic is untouched. Re-copy from upstream (and re-apply that sed) when the evaluator changes.
