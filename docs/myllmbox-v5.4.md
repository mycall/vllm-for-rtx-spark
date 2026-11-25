# MyLLMBox v5.4 on Windows ARM64

The first experiment for concurrent agents is **BF16 target KV at a fixed memory budget**. This tests upstream's claim that deeper draft tokens are accepted more often. The Windows `gpu-96gb-draft-kv-bf16.yaml` profile already uses BF16 for the MTP draft layer, while the target still uses FP8. Run it once as shipped and once with `--kv-cache-dtype auto` to select the model's BF16 dtype. Keep its 18,000,000,000-byte pool, depth 3–7 and HEAD2 fixed. Measure acceptance by draft position and how many concurrent histories fit. [Windows baseline](../configs/qwen3.8-flash-next/gpu-96gb-draft-kv-bf16.yaml) · [Upstream v5.4 recipe](https://github.com/myllmbox/qwen38-flash-next-recipe/blob/v5.4/recipe.yaml)

## What needs porting

The Windows source pin is v5.3.1. The public v5.4 tag, `d392cad` (October 27), changes the recipe, launcher and README and points to a new Linux image. Some implementations, including the depth library, ship in the image. [Windows source lock](../sources.lock.json) · [Upstream tag diff](https://github.com/myllmbox/qwen38-flash-next-recipe/compare/v5.3.1...v5.4)

| Component | Upstream v5.4 | Windows port work |
|---|---|---|
| Attention KV | BF16 target and draft. Upstream measured 881,260 **shared** tokens with a 27 GB KV budget and a 409,600-token per-request limit. | Measure the pool reported at startup and preemptions under load at the Windows budget. |
| Dynamic MTP | Starting depth 3. The maximum is 9 for one live request, 7 for 2–3, 6 for 4–7 and 5 for 8 or more. | The Windows host reads a JSON policy and selects a batch depth from per-request depths. Add the live-request ceilings and align CUDA graph capture sizes with the resulting verification widths. Raising the YAML maximum alone misses the concurrency policy. [Depth selection](../src/recipes/qwen3_8/helpers/mbx_depth.py) |
| Draft output head | An NVFP4 copy scores the full vocabulary. Verification uses the checkpoint's BF16 head. | The Windows HEAD2 path uses a rank-256 scorer and rescores 2,048 candidates. A full-head implementation needs a separate draft weight, load-time quantization with bounded temporary memory, and a comparison of token acceptance. [Current draft head](../src/recipes/qwen3_8/helpers/mbx_draft_head2.py) |
| GDN and agent-turn reuse | v5.4 selects BF16 recurrent-state storage. GDN and PLE prefill checkpoints for resuming an appended turn were already in v5.3.1. | Test recurrent-state storage separately from attention KV. The shipped GDN prefill kernels use FP32 state and checkpoint buffers. Check cold and cached answers, cached-token counts and follow-up time to first token. [Prefill kernel contract](../src/kernels/cute/gdn_aot.py) · [v5.3.1 release notes](https://github.com/myllmbox/qwen38-flash-next-recipe/tree/v5.3.1) |

**Native GPU check, November 24, 2026.** On an RTX Spark N1X with the Windows `0.32.0+rtx_spark.0.1.2` wheel, Marlin ran a synthetic NVFP4 projection with the checkpoint's 248,320 vocabulary rows and 2,560 hidden features. At batch sizes 1, 4 and 8, every logit matched the independently decoded weights in eager execution and CUDA graph replay for the one-hot inputs. This covers the projection. Quantizing real weights, integrating the drafter and measuring generation are separate experiments. Reproduce from the installed checkout with `.\.venv\Scripts\python.exe tests\probe_nvfp4_draft_head.py`. [GPU probe](../tests/probe_nvfp4_draft_head.py)

## Lineage and comparison points

This Windows port derives from MyLLMBox. Another direct derivative, Alssndr0's DFlash2 image, uses MyLLMBox v5.3 and the **same NVIDIA FP4-Local target**. On one DGX Spark it reported 59.48 tok/s with DFlash2 versus 56.12 with tuned MTP across 177 paired single-stream agent requests. This provides a same-checkpoint control for the Windows port. [Alssndr0's replay and image patches](https://github.com/Alssndr0/qwen38-flash-next-recipe/tree/e0f4d37e82e68770670bc588ed1ab2673acdd3e7/dflash2)

| Other development lines | Relationship | GitHub stars, Nov 24 snapshot |
|---|---|---:|
| [MiaAI-Lab single Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark) | Independent single-Spark competitor, including TensorFold. | 647 |
| [MiaAI-Lab dual Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) | Independent two-Spark recipe. Its hardware topology differs from this port's single N1X. | 419 |
| [roozich](https://github.com/roozich/Qwen3.8-Flash-Spark) | Originated from blazux. Adds an incoai DFlash2 profile with four draft tokens, a full vocabulary and the V2 runner. | 5 |
| [Saren-Arterius AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound) | Also a blazux fork. Contributed GB10 kernels and PLE improvements used by roozich. | 52 |
| [styles01/sparkrun-recipes](https://github.com/styles01/sparkrun-recipes) | Integrates and benchmarks recipes across multiple engines. | 96 |

## Published eight-stream results

- On October 22, Alfonso Fortunato measured **MyLLMBox v5.3** at 136.9 aggregate tok/s, the highest of six configurations on his DGX Spark. TensorFold led single-stream code decode at 67.2 tok/s. This comparison predates v5.4. [Six-configuration comparison](https://alfonsofortunato.com/writing/six-ways-to-serve-qwen3-8-flash-next-on-one-dgx-spark)
- On a GX10, roozich reported 176.1 aggregate tok/s with DFlash2 K4 versus 167.0 with MTP K2 on the same target and serving settings. The reported gain went from 26% at one stream to 5% at eight. The recipe uses an NVFP4 target layout that differs from this project's FP4-Local checkpoint. Its port also needs Qwen4Exp hidden-state taps and QSA ring handling. [Matched GX10 measurements](https://github.com/roozich/Qwen3.8-Flash-Spark/blob/df9052833eb5933fb3b0508a609def87a81e3d69/docs/BENCHMARKS.md#dflash-profile-measurements) · [DFlash integration patch](https://github.com/roozich/Qwen3.8-Flash-Spark/blob/df9052833eb5933fb3b0508a609def87a81e3d69/src/patch_dflash_qwen4exp.py)

## Paired N1X benchmark

`n1x-ai` already runs a MyLLMBox v5.4 Linux image under Windows Subsystem for Linux Containers (WSLC) with an AutoRound target and DFlash2. Its historical native FP4-Local MTP and WSLC DFlash2 profiles both passed 72/72 inventory tasks. At four clients they delivered 0.537 and 0.811 correct tasks/s respectively. Weights, runtime, cache and GPU partition differed (96 vs 88 GiB). Those rates establish deployed-profile baselines. [MyLLMBox Linux runtime](https://github.com/mycall/n1x-ai/blob/38cd44291cd3f0a74a7aa0bbd9a5f46630ff1c1b/wiki/MyLLMBox.md) · [Native agent results and conditions](https://github.com/mycall/n1x-ai/blob/38cd44291cd3f0a74a7aa0bbd9a5f46630ff1c1b/wiki/Native-vLLM.md#measured-agent-throughput)

For a paired comparison, use the same FP4-Local revision, N1X partition, power mode, KV budget, prompts and sampling. Measure native v5.3.1-derived MTP first, then target BF16 KV, then the depth and head changes individually. Compare the resulting MTP configuration with the incoai DFlash2 K4 drafter on that same target. Record shared KV capacity, per-position acceptance and preemptions alongside aggregate decode tok/s. Use the thinking-enabled inventory suite to score correct tool tasks/s and the incremental runs to measure turn resumption.

Run the speed client's offline preview from the `n1x-ai` checkout.

```powershell
.\benchmark.ps1 -Profile qwen38fn-myllmbox -Suite speed -SpeedPlanOnly `
    -SpeedDepths 4096 -SpeedConcurrency 1,4,8 -SpeedMode both -SpeedRepeats 3 -SpeedOutputTokens 128
```

The preview produced six cold/incremental cells with 78 measured requests planned and 39 prefix-priming requests. End-to-end native v5.4 and matched DFlash2 timings are open. [Speed protocol](https://github.com/mycall/n1x-ai/blob/38cd44291cd3f0a74a7aa0bbd9a5f46630ff1c1b/wiki/Speed-benchmarks.md)
