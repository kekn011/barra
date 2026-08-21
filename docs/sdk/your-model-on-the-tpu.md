# Your model on the Pixel TPU

barra gives you something unusual: a **complete, on-device compile-and-run path**
for the Tensor G3 edge TPU. No cloud service, no vendor SDK on your PC — you
export a TFLite model, compile it *on the phone* with `barrac`, and run it
through `libbarra`, zero-copy, from your own C program.

This guide covers the rules that make a model compile and run well. They were
found empirically (this TPU has no public toolchain); stick to them and the
path is reliable.

## The 60-second version

```sh
# on the node
barrac mymodel.tflite            # -> mymodel.package (30-120 s)
```

```c
#include <barra.h>
barra_tpu t;  barra_tpu_open(&t);
barra_zbuf in, out;
barra_zc_alloc(&in,  IN_BYTES);  barra_zc_alloc(&out, OUT_BYTES);
memcpy(in.map, my_input, IN_BYTES);
barra_tpu_infer(&t, /*model_id=*/0, &in, &out, NULL);
/* result is in out.map — no copies happened anywhere */
```

Packages are served by the `tpud` bridge; the **model id is the position in
the load order** of the tpud instance you talk to (see "Serving packages"
below).

## Export rules (TFLite side)

The on-device compiler consumes standard **fully quantized** TFLite flatbuffers.
What works:

1. **Fixed shapes only.** No dynamic dimensions. Pick your batch/sequence size
   at export time; pad at runtime (the cost of zero rows is usually negligible).
2. **Matrix multiply = 1×1 convolution.** Export dense layers as
   `tf.nn.conv2d` over a `[1, M, 1, K]` input with a `[1, 1, K, N]` kernel.
   That is the shape the TPU's convolution engine eats natively.
3. **Full integer quantization** with a representative dataset:
   * **int8** (`tf.lite.Optimize.DEFAULT`, int8 in/out): fastest, coarsest.
   * **16×8** (`EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8`,
     int16 in/out): ~15 bits of activation precision at int8-weight speed.
     Zero points are 0 by design. Per-tensor calibration is usually enough at
     ±32767 — but calibrate with data that covers your real value ranges, or
     outputs will saturate silently (cosine metrics barely notice; end-to-end
     quality does).
4. **~200 operators** are supported (conv/matmul, elementwise, softmax,
   normalization, attention-shaped compositions …). If the compiler rejects a
   graph, `barrac` prints the compiler log — look for the first unsupported op
   and restructure.

## Size limits (learned the hard way)

* **≤ ~16 MB of weights per package.** Beyond that the on-device compiler
  hangs without a result. Split big layers into chunks (e.g. a 2560→9728
  matmul as 2 × 2560→4864) and run the chunks as separate inferences.
* **int16 outputs: ≤ 2048 columns per package at M=512.** Wider int16 outputs
  fail with "unrecoverable compilation errors" — chunk the output dimension.
  (int8 outputs go wider.)
* **~150 packages loadable at once.** The TPU firmware has a fixed internal
  pool for graph registrations (~157 graphs); plan your model count
  accordingly and merge small sibling layers into one package where possible.
* **~2 GB total pinned buffers** (TPU IOMMU) across all loaded packages.

## Serving packages

The base image runs a system `tpud` for the default socket. For your own model
set, start a dedicated instance with your packages (order = model ids):

```sh
su -c 'TPU_FENCE=1 LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 \
  /data/adb/baseos/llm/tpud-attn /path/tpu.sock pkg0.package pkg1.package …'
```

then point libbarra at it with `BARRA_SOCK_DIR=<dir containing tpu.sock>`.
Both instances can coexist (the ~157-graph firmware pool is shared).

## Performance notes

* Measured dense-matmul throughput: **~4.4 TOPS int8** on transformer-shaped
  1×1 convs (M=512); ~720 GFLOPS-equivalent on int8 elementwise mixes.
* **Pipeline your inferences.** A single blocking infer pays ~6 ms of latency;
  with `barra_tpu_submit`/`barra_tpu_wait` and two jobs in flight the per-infer
  cost drops to ~4.4 ms on medium packages.
* Zero-copy is the default worldview: allocate `barra_zbuf`s once, import them
  once, and let TPU/GPU/DSP read and write the same dmabuf. Copying tensors
  through sockets is the slow path and never necessary.
* The TPU clock gates aggressively; `barra-smi` showing 0 MHz at idle is
  normal. Under load expect ~1.1 GHz.

## Worked example

The barra LLM stack itself is the reference customer of this path: the
attention projections of a 4-billion-parameter transformer run as 144
16×8-quantized packages (4 per layer, k/v merged to respect the graph-pool
limit), compiled entirely on-device, and land within noise of the fp16
baseline in perplexity. The generator scripts live in `src/ggml-barra/attn/`
in the barra repository — they are a good template for chunking, calibration
and metadata handling.
