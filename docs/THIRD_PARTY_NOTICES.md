# Third-party notices: Mamb2_8B_FP4_Recall

Paths are relative to the complete bundle root. The HF snapshot places this
root under `release/`; the GitHub runtime archive and 119 weight assets assemble
into `mamba2-8b-fp4g16-repaired-s16-resurface-v1/`.

## Applicable licenses

The modified base weight files, including the repaired retained FP16 shard,
are Apache-2.0. EndlessChasing's modification grant is described in
[LICENSES.md](https://github.com/EndlessChasing/mamb2_8B_FP4_Recall/blob/main/LICENSES.md); complete text is [licenses/Apache-2.0.txt](https://github.com/EndlessChasing/mamb2_8B_FP4_Recall/blob/v0.1.0-fp4g16-repaired-s16-resurface/licenses/Apache-2.0.txt)
and the canonical [reference/w4/WEIGHTS_LICENSE.txt](https://github.com/EndlessChasing/mamb2_8B_FP4_Recall/blob/v0.1.0-fp4g16-repaired-s16-resurface/reference/w4/WEIGHTS_LICENSE.txt).
The inherited Recall/project implementation and this new Resurface adapter
are GPL-3.0, with complete text in [LICENSE](https://github.com/EndlessChasing/mamb2_8B_FP4_Recall/blob/v0.1.0-fp4g16-repaired-s16-resurface/LICENSE) and
[licenses/GPL-3.0.txt](https://github.com/EndlessChasing/mamb2_8B_FP4_Recall/blob/v0.1.0-fp4g16-repaired-s16-resurface/licenses/GPL-3.0.txt). Preserved upstream source retains
its own license and attribution. The code license does not relicense underlying
Apache weights or dataset text; the base weight license does not grant an
Apache license to this adapter/runtime.

## NVIDIA base and tokenizer: Apache-2.0

The source is [`nvidia/mamba2-8b-3t-4k`, revision
`b915550c63ba9359f88f44d1f6a600d85af27302`](https://huggingface.co/nvidia/mamba2-8b-3t-4k/tree/b915550c63ba9359f88f44d1f6a600d85af27302).
Its [pinned model card](https://huggingface.co/nvidia/mamba2-8b-3t-4k/blob/b915550c63ba9359f88f44d1f6a600d85af27302/README.md)
declares Apache-2.0 and identifies the pure Mamba2 model. Credit Roger Waleffe
and colleagues, [*An Empirical Study of Mamba-based Language Models*](https://arxiv.org/abs/2406.07887), 2024;
the pinned card gives the complete author list and original citation.

**Modification notice:** EndlessChasing converts the Megatron source to the
native model configuration and FP16 reference. It independently quantizes114
large matrices to E2M1 FP4,16-element groups, E4M3FN block scales and FP32
global scales using a fixed reconstruction-error range search. Base-only
TRAIN finetuning then updates393 existing retained FP16 tensors; step 448
was selected from a fixed four-checkpoint pool by full validation NLL. No FP4
matrix/code/scale file changes during that repair, and all 507 base tensors
remain frozen during new adapter training. The original authors do not
endorse these modifications.

The modified base is identified by `weights/weight_manifest.json` SHA256
`2d2208b690c2ec0def90578d8590cc122bb1e4542b0d49ae4417a48f72b0b73c`.
The original conversion receipt and parent manifest are preserved beside it.
No Quamba quantized checkpoint or research-only weight package is used.
The original checkpoint SHA256 is
`47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`.

The included source tokenizer is
`tokenizer/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model`,
SHA256 `5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09`,
from the same pinned NVIDIA repository.

## Mamba implementation and installed dependencies

Native SSD and recurrent execution use the separately installed
[`state-spaces/mamba`](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa),
measured as mamba-ssm2.3.2.post1, under Apache-2.0, copyright2023 Tri Dao
and Albert Gu. See its [original license](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/LICENSE)
and [Mamba2/SSD paper](https://arxiv.org/abs/2405.21060). These dependencies are
installed separately; preserve their complete licenses/notices. Apache text
included for the base does not replace a dependency's own attribution notice.
PyTorch, Triton and other dependencies likewise retain their upstream terms.

## StateQuant reference: preserved source only

The source closure retains the StateQuant reference kernel from the
user-supplied archive, commit`22156f74edf428fd192948d75a0dabf3cf152d48`,
copyright2026 Kun Yue, Apache-2.0. Its license is preserved at
[reference/statequant/LICENSE](https://github.com/EndlessChasing/mamb2_8B_FP4_Recall/blob/v0.1.0-fp4g16-repaired-s16-resurface/reference/statequant/LICENSE), with an identical
copy at [licenses/StateQuant-Apache-2.0.txt](https://github.com/EndlessChasing/mamb2_8B_FP4_Recall/blob/v0.1.0-fp4g16-repaired-s16-resurface/licenses/StateQuant-Apache-2.0.txt).
See [StateQuant](https://github.com/Oso1106/StateQuant) for the upstream work.
This model executes native FP16 state; no StateQuant controller, table,
predictor or state_config checkpoint is part of its inference configuration.
Retaining an imported helper's source dependency is not a state-quantization claim.

## Recall and new Resurface adapter: GPL-3.0

The inherited Recall implementation comes from
[`EndlessChasing/mamb2_8B_Recall`](https://github.com/EndlessChasing/mamb2_8B_Recall)
under GPL-3.0. New project implementation, repair/packing/training/evaluation
source and the newly trained adapter are distributed under GPL-3.0.
The complete corresponding project source is included in the bundle and tagged
GitHub source repository. The adapter's SHA256 is
`dca3cda5a9ab8a665abd1a355c6e5b294acee023d2a1a5a8e20f4e84c9674580`.

The independently implemented gated post-D correction is inspired by
[*Resurface: Multi-Binding Recall Is Latent in Mamba's State*](https://github.com/Oso1106/Resurface-Multi-Binding-Recall-Is-Latent-in-Mamba-s-State).
Its insertion site and training binding are specific to this repaired base;
it is not an exact reproduction of the upstream insertion site or an
interchangeable adapter. Earlier Apache grants in family notices identify
other named artifacts and do not define the license of this newly trained adapter.

## WikiText attribution and separate data scope

Training/evaluation uses WikiText by Stephen Merity, Caiming Xiong, James
Bradbury and Richard Socher, [*Pointer Sentinel Mixture Models*](https://arxiv.org/abs/1609.07843), 2016,
and articles by Wikipedia contributors. The pinned
[`Salesforce/wikitext` dataset card](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md)
lists CC BY-SA3.0/GFDL metadata while its Licensing Information section links
CC BY-SA4.0. This notice retains that upstream discrepancy and attribution;
consult the pinned card and original contributor history for text terms.
The complete prose corpus is not included in this model download. Reproducing
text/token excerpts does not change their underlying attribution or terms.
Synthetic numeric recall cases, token hashes and model statistics are separate
from underlying prose text. GPL-covered code does not relicense dataset text.

When redistributing, preserve this notice, full applicable licenses, original
attributions, the modification notices and manifests identifying the artifacts.
