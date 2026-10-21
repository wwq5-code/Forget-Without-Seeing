# SMFA upstream provenance

- Repository: <https://github.com/zeng-zhen/S-MLLMUn>
- Downloaded: 2026-10-05
- Source archive: `https://codeload.github.com/zeng-zhen/S-MLLMUn/tar.gz/refs/heads/main`
- Archive SHA-256: `ec6265ca3ed02a67a49c3af1efeb5438559a80e32569307868b4a4211fd02b35`
- Vendored directory: `On_FIUBench/third_party/S-MLLMUn`

The upstream files are kept unchanged. The FIUBench integration lives in
`On_FIUBench/fiubench_llava_scheme_b_core.py`.

The integration is explicitly a **matched-projector adaptation**, not a native
full-model reproduction. Upstream SMFA trains multimodal and text-only LoRA
adapters over language-model projection layers plus a retaining adapter, then
sculpts their effective weight updates. This repository's controlled experiment
keeps the LLaVA vision tower, language model, and native multimodal projector
frozen and trains only the pre-existing low-rank projector residual coefficient
(`lora_b`) with a fixed `lora_a`. It therefore uses the upstream multimodal
IDK/retain objective, applies the same sign-conflict/relative-magnitude rule to
the effective projector weight update (`delta_B @ lora_a`), and least-squares
projects the sculpted result back to the fixed-rank parameterization. There is
no meaningful text-only path through this projector residual.
