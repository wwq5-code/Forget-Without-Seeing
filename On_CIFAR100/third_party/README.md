# External CLIP unlearning baselines

Downloaded on 2026-10-04 for provenance and comparison with
`On_CIFAR100/unlearning_clip_by_subliminal.py`.

## CLIPErase

- Upstream: https://github.com/Tianyu-yang-anna/ClipErase-ACL
- Revision: `859fc63b2e9404d7d45d76bed2e3d2f482b59ec6`
- Downloaded archive SHA-256: `9d29ba0ee203ca5b8aef1325a639413e98bd5ddd01adfe187cc32eab5dc57e33`
- Local directory: `ClipErase-ACL/`
- License: MIT (see `ClipErase-ACL/LICENSE`)

## Zero-Shot Class Unlearning in CLIP with Synthetic Samples

- Upstream: https://github.com/akres001/Zero-Shot-Class-Unlearning-in-CLIP-with-Synthetic-Samples
- Revision: `0fd58edac67dddc5283cf12d2e8feedba5b7050d`
- Downloaded archive SHA-256: `9cf0a17ec825be22497853c95c6c0168d998f6240bebeb3f22d4d91ff57aed0f`
- Local directory: `Zero-Shot-Class-Unlearning-in-CLIP-with-Synthetic-Samples/`
- License: no license file was present at this revision. The downloaded source is
  retained unmodified for inspection; do not redistribute it without confirming
  permission with the upstream authors.

## Integration scope

The comparison entry point uses independent matched adaptations rather than
importing either repository at runtime. All methods start from the same
personalized CLIP checkpoint and use the same CIFAR-100 test split and behavior
metrics. CLIPErase keeps its forget/retain/consistency objective and receives
the private forget-trigger records. The zero-shot baseline keeps synthetic
target generation, joint Lipschitz optimization, and gradient-saliency layer
selection, and does not receive private forget records.

The comparison output also contains the repository's own functional-message
method at the same forget-record count with `epsilon=inf`, giving a non-private
utility reference against the two non-DP baselines. Finite-epsilon results stay
in the separate DP sweep output.

These results are a comparison on this repository's trigger-association
unlearning task, not an unmodified reproduction of either paper's original
class-deletion benchmark.

Run both matched baselines with:

```bash
python On_CIFAR100/unlearning_clip_by_subliminal.py \
  --sweep comparison \
  --data-root /path/to/cifar \
  --cache-dir /path/to/open_clip_cache \
  --print-progress
```

Use `--comparison-methods cliperase` or
`--comparison-methods zero_shot_lipschitz` to run one method. By default the
comparison JSON is written to
`On_CIFAR100/unlearning_clip_by_subliminal_104_for_comparison.txt`.
