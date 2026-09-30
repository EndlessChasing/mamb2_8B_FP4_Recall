# License scope: Mamb2_8B_FP4_Recall

This distribution combines components with different licenses. The root
GPL-3.0 text does not relicense the Apache-2.0 base weights or third-party code.

| Component | License | Complete license text |
| --- | --- | --- |
| Modified NVIDIA base: FP4 codes/scales and repaired393 FP16 tensors | Apache-2.0 | [Apache-2.0](licenses/Apache-2.0.txt) |
| Project runtime, packing/repair/training/evaluation code and final Resurface adapter | GPL-3.0 | [GPL-3.0](licenses/GPL-3.0.txt) |
| Preserved StateQuant reference source, unused in state execution | Apache-2.0 | [StateQuant license](licenses/StateQuant-Apache-2.0.txt) |
| Separately installed Mamba/PyTorch/Triton and other dependencies | Their upstream licenses | Retain each installed package's own notices |
| WikiText/Wikipedia text, if reproduced separately | Applicable upstream text terms | [Pinned dataset card](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md) |

EndlessChasing releases its base-weight modifications under Apache-2.0,
including the fixed repaired manifest
`2d2208b690c2ec0def90578d8590cc122bb1e4542b0d49ae4417a48f72b0b73c`.
The new adapter identified by SHA256
`dca3cda5a9ab8a665abd1a355c6e5b294acee023d2a1a5a8e20f4e84c9674580`
is distributed under GPL-3.0 with its included implementation/training source.
Earlier artifact-specific family grants identify different weights/adapters.

These licenses permit redistribution subject to their terms, including
preserving notices and providing applicable GPL-covered source. No research-only
Quamba weight license applies. See [third-party notices](THIRD_PARTY_NOTICES.md)
for attribution, modifications and exact upstream provenance.
