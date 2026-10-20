# Experimental: NVIDIA DGX Spark (GB10, aarch64)

The DGX Spark pairs an ARM CPU (10 Cortex-X925 + 10 Cortex-A725 cores) with a Blackwell GPU (GB10, sm_121); the CPU
and the GPU share 128 GB of unified memory. Strata runs there from a source build. It is community-tested, on one Spark,
with Qwen3.8-Flash-Next **IQ2_XS** and Unsloth's **UD-Q4_K_XL** installed by `setup.sh`. The other setup models use
expert formats that have an ARM code path (below) but have not been run on a Spark. There is no ready-made ARM engine.

## Rebased for v0.1.41, and IQ3_S validated (2026-10-08)

This support was rebased onto v0.1.41 (commit `fb58e0d`) after upstream's history cleanup, which also required two
small drift fixes for the new base — the guards marked "DGX Spark GB10 (aarch64)" in
`src/kernels/cpu/expert_layout.cpp` (the x86 cpuid / oracle-flag / `cpu_features()` block) and
`src/kernels/cpu/native_expert.cpp` (the `iq*_supported` / `kq256_supported` answers and the `q8k_quant_avx2`
quantizer). With those, **IQ3_S** also runs on a Spark, and was measured on one (256K context, `--parallel 4`,
`--kv int8`, all 24,576 experts in the cache):

- decode 58.8-66.0 tok/s, prompt 103-111 tok/s on short requests, MTP accepting 13/14 and 60/79 proposals;
- a 127K-token long prompt read at 1512 tok/s prefill, with a needle-in-haystack marker retrieved correctly from
  that depth (decode 57.4 tok/s during it).

## Install with setup (recommended)

```sh
./setup.sh
```

The same questions as on a PC. What setup does differently on a Spark:

- **GPU:** the GB10 reports no memory of its own (`nvidia-smi` shows `[N/A]`), so setup counts the system's RAM as
  the GPU's.
- **Engine:** compiled on the Spark, once (the ready-made engine is x86-64). It needs the CUDA 13 toolkit and a C++
  compiler; both were already installed on the Spark it was tested on. ggml's CPU code is compiled with the
  features read from `/proc/cpuinfo` (`-march=armv8.6-a+dotprod+i8mm+fp16` on the GB10): gcc's `-mcpu=native`
  finds none on this CPU, which would leave ggml without its dot-product instructions.
- **Start settings:** `--mmap-experts` and no KV streaming (see below).

## Docker

Not on ARM yet. The repository's `Dockerfile` packages a `linux-x86_64` engine and Terminal, so an image built on the
Spark is not usable. Use setup, or the manual build:

```sh
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121 -DSTRATA_GGML_DIR=third_party/llama.cpp \
  -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm+fp16
cmake --build build --target strata
```

## How it runs with unified memory

- **Every expert on the GPU.** With `--expert-cache auto` the cache holds all 24,576 experts (33.0 GiB for IQ2_XS, 71.7 GiB for
  UD-Q4_K_XL), so
  the CPU computes none: the per-token expert work is the GPU's alone.
- **`--mmap-experts`:** the experts are read from the GGUF in place. A RAM arena of the experts (the default on a PC)
  would be a second copy of the expert cache in the same memory.
- **The cache's size counts reclaimable memory.** `cudaMemGetInfo` reports only free RAM, not the page cache, which
  here holds mostly the GGUF's own pages and gives way to the GPU's allocation. On a unified memory system the engine sizes
  the cache from `MemAvailable` less 6 GiB for everything else; `STRATA_UMA_HEADROOM_GIB=N` changes that margin.
  That sizing is the same one an integrated GPU uses on x86 (Strix Halo included).
- **No KV streaming:** it moves KV to RAM to free VRAM, and there is no separate VRAM.

## Speed

Qwen3.8-Flash-Next IQ2_XS on a DGX Spark (driver 580.173.02, CUDA 13.0, Ubuntu 24.04), started with setup's own
`run-iq2_xs.sh` (128K context, `--kv int8`). [llama-benchy](https://github.com/eugr/llama-benchy) 0.4.0, 3 runs per
row after a warm-up, `--no-cache`, one request at a time, 128 tokens written per request. Measured on the
community port this support was taken from, rebased onto Strata around 0.1.38; a current engine can differ.

| Prompt | Context | Prompt tokens | Prefill | Decode | MTP acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 0 | 2,049 | 957 tok/s | 55.7 tok/s | 72% |
| 8K | 0 | 8,194 | 1,357 tok/s | 59.7 tok/s | 74% |
| 2K | 16K | 18,433 | 1,410 tok/s | 63.7 tok/s | 73% |
| 8K | 16K | 24,577 | 1,503 tok/s | 57.3 tok/s | 80% |
| 2K | 64K | 67,586 | 1,538 tok/s | 56.2 tok/s | 77% |
| 8K | 64K | 73,729 | 1,548 tok/s | 56.4 tok/s | 75% |

Prefill and decode are the server's own (`strata serve: prompt ... read in ... generated in ...` in the
log), averaged over the 3 runs. llama-benchy's `pp` column cannot time Strata's server: the first chunk of the stream
is sent before the prompt is read, so benchy's time to first response is a few milliseconds. Its time to the first
token is right (2.18 s at 2K, 6.11 s at 8K, 47.8 s at 8K after 64K).

Short chat answers decode at 68-79 tok/s (a coding, a factual and a maths question, `reasoning_effort` low).

**UD-Q4_K_XL** (`./setup.sh --family unsloth`, 262K context, `--kv int8`): all 24,576 experts in the GPU's cache,
82.2 GiB of GPU memory for the whole engine; the 28.8 GB PLE table stays on the SSD. On a unified memory system setup
keeps no RAM budget of experts (`--resident-budget-gib`, the 64 GB PC's setting): it would be a second copy of them
in the memory the expert cache uses. The same benchmark:

| Prompt | Context | Prompt tokens | Prefill | Decode | MTP acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 0 | 2,049 | 590 tok/s | 38.4 tok/s | 74% |
| 8K | 0 | 8,193 | 1,117 tok/s | 39.2 tok/s | 72% |
| 2K | 16K | 18,433 | 1,058 tok/s | 41.8 tok/s | 80% |
| 8K | 16K | 24,577 | 1,173 tok/s | 39.7 tok/s | 78% |
| 2K | 64K | 67,585 | 1,165 tok/s | 39.4 tok/s | 79% |
| 8K | 64K | 73,729 | 1,187 tok/s | 41.5 tok/s | 81% |

## When the CPU computes experts

With fewer cache slots than experts (a smaller `--expert-cache`), the CPU computes the rest. On aarch64 every expert
format, Q2_0 included, goes through ggml-cpu's NEON dot products; Strata's own multi-token kernels for the x86 CPUs
(AVX-512 / AVX2) have no ARM version yet. Measured on IQ2_XS with a 2,048-slot cache, where the CPU computed ~22 of the
~29 experts each layer routed: 19.7 tok/s, against 82.9 tok/s with every expert on the GPU, and the same greedy
answer token for token.

## An ARM server with a discrete card (NVIDIA L4)

Here most experts do not fit on the GPU, so the ARM CPU computes the ones the cache misses.

Measured on 2026-10-20: a KVM guest with 32 Neoverse-N1 cores in one socket (`asimddp`, no `i8mm`, SVE or `bf16`),
124 GB of RAM and an NVIDIA L4 (24 GB, PCIe 4.0 x16, 24.7 GB/s host to device by the engine's probe), driver
610.43.02. The engine is 0.1.46.1 with this port (engine sources as in commit `11db774b`; setup's source hash
`4c204374574acd2b`), compiled by `./setup.sh --build` in the `nvidia/cuda:13.0.0-devel-ubuntu24.04` image (GCC 13.3,
CUDA 13.0) with `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`. Model: Qwen3.8-Flash-Next **IQ3_XXS**, 64K context, the
config setup wrote (`--kv int8`, MTP on, no images). The engine's CPU pool has 30 workers and the host thread. The
GPU's cache holds 9,657 of the 24,576 experts (15.65 GiB).

Each request starts with a fresh nonce, so none reuses another's prompt cache. 512 tokens written per request; 3 runs.
The rates are the server's (`timings` in the answer).

| | Result |
| --- | ---: |
| Decode, one request | 52.7, 53.3, 59.3 tok/s |
| Two requests at once (`--parallel 2`), total | 51.1, 51.4, 51.9 tok/s (26.4-27.7 each) |
| A 33,025-token prompt, read in | 22.7, 23.0, 23.1 s (1,430-1,454 tok/s) |

- **The CPU computes few experts.** The cache is filled from the routing profile, so the experts that are used most
  are on the GPU. In the decode runs, 93.3-96.2% of a request's experts were cache hits, 0.3-0.8% were read by the GPU
  over PCIe, and the CPU computed 3.8-6.6% (`/metrics`: `hit_rate`, `pcie_share`). With two requests at once it
  computed 4-17%.
- **The CPU is not the limit.** `perf` over a 512-token answer: ggml-cpu's NEON dot products took 5.3% of the
  engine's samples (`ggml_vec_dot_iq3_s_q8_K` 1.5%), and the pool's workers spent 88% waiting for work. The GPU's SM
  use was 97-100% (`nvidia-smi dmon`).
- **`--parallel 2` gains nothing here:** the two requests together decode about as fast as one alone.
- **Checks on the same runs:** an OpenAI tool call (`get_weather`, `{"city":"Lyon","unit":"celsius"}`), a marker in
  the middle of the 33K-token prompt retrieved in 3 of 3 runs, and a one-sentence answer.

**A VM that shows every vCPU as its own socket gets one pool worker.** The same VM first ran with 32 sockets of one
core. Every vCPU then had its own `physical_package_id`, so the engine kept its pool on the host thread's socket:
`CPU pool: 1 workers by default (32 physical cores, ...; bounded by the host's CPU socket)`. Decode was 34.5, 39.9 and
40.3 tok/s, and `--parallel 2` 34.1-35.0 tok/s in total. `"--pool-workers", "15"` in the `args` of the config setup
writes gave 49.2, 50.1 and 53.5 tok/s (7 workers: 48.9 median, 31: 50.8). Give such a VM one socket, or set
`--pool-workers`. The engine's PCIe probe measured 20.6 GB/s on that VM; why it differs from 24.7 GB/s was not
checked (one guess, not measured: memory fragmentation on that VM).

## Not covered

- Models other than IQ2_XS, IQ3_XXS, IQ3_S and UD-Q4_K_XL: not run (their formats are covered: IQ1_M, IQ2_XXS,
  IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL and Q2_0 experts all have ggml NEON paths).
- Images (`--vision`): built with the same CPU flags, not tested.
- ARM CPUs without a CUDA GPU: not tested. One ARM server with a discrete card was run (above).
- The canonical Q2_0 pack (`experts.bin` in Strata's own layout) is x86-only. Setup's Q2_0 size on a Spark is the
  native GGUF pack, whose Q2_0 rows use ggml.

## Tests

`expert_multi_test` checks the AVX-512 Q2_0 kernel and is registered on x86 only, as is `iq_avx2_parity`.
`expert_parity --selftest` skips where the CPU has no AVX-512.
