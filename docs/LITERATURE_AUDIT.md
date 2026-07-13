# Literature and progress-PDF audit

The local planning source was `音视频推理加速 (1).pdf` (48 pages, created
2026-07-06, SHA-256
`7eee5092f51059d28f230bb0895a8f728bfb0b4771730e08de42940ab3bddfa0`).
Pages 39-48 were checked both by text extraction and page rendering.

## Boundary between historical results and FastA2V

Pages 45-48 describe results from an older Ovi/SVG/cache implementation. The
current FastA2V repository does not inherit that implementation. These numbers
are comparison targets only:

| PDF page | Historical configuration | Reported result |
| --- | --- | --- |
| 45 | Ovi + SVG, 960 x 960, 10 seconds | 1.35x end-to-end, 1.9x video self-attention, PSNR 25.6 |
| 45 | Protective sparse routing | 1.32x, PSNR 25.8 |
| 46 | SVG + CFG cache | 1.65x, PSNR 24.27 |
| 46 | SVG + block cache variants | 1.34-1.35x, PSNR 24.33-24.40 |
| 47 | 960 x 960, 5 seconds, SVG | 1.17x, PSNR 24.63 |
| 47 | 960 x 960, 5 seconds, CFG cache | 1.46x, PSNR 24.42 |
| 47 | Aggressive block-cache variants | 1.74-1.94x, PSNR 22.99-23.52 |

Those pages do not record a source revision, complete environment, fixed seed,
per-stage timing, peak memory, raw output, or machine-readable results. They
therefore cannot serve as a reproducible baseline.

## Constraints carried into the new experiment

- Page 48 reports that sparse attention on the audio branch caused silent
  output. FastA2V keeps audio self-attention dense.
- Video-text, audio-text, audio-to-video, and video-to-audio cross-attention
  remain dense in the first implementation.
- The old block cache reused outputs two or three times. FastA2V's first block
  cache will permit at most one consecutive reuse and will keep conditional and
  unconditional state separate.
- The old 5-second result used 960 x 960. The new baseline uses the official
  `720x720_5s` checkpoint at 720 x 720.
- The PDF shows that sparse video self-attention can accelerate a module while
  producing a much smaller end-to-end gain. Every method must therefore report
  both denoising and total generation time.

## Relevant local papers

The local `Essay/` directory contains 16 papers. The first implementation phase
uses these as follows:

| Paper | Use in FastA2V |
| --- | --- |
| SageAttention | Quantized attention background and SpargeAttn dependency context |
| Sparse VideoGen 2 | SVG comparison and later feasibility study |
| PreciseCache | Block-cache design reference |
| VMoBA | Long-sequence sparse-attention reference, not a first-stage method |
| TurboDiffusion | Combined acceleration survey; training components are excluded |
| DMD, DMD2, DiagDistill, GPD, WaDi, rCM | Excluded because this project does not train or distill |
| PipeDiT | Deferred because the first baseline is single-GPU |
| LESA, TABM | Deferred because they introduce learned or more complex cache policies |

The local folder does not contain standalone Ovi, LTX-2, SpargeAttn, or Radial
Attention papers. Repository revisions and APIs for those methods must be
verified from their official project sources.

## Official implementation sources

- Ovi: <https://github.com/character-ai/Ovi>
- SpargeAttn: <https://github.com/thu-ml/SpargeAttn>
- Radial Attention: <https://github.com/mit-han-lab/radial-attention>
- Sparse VideoGen: <https://github.com/svg-project/Sparse-VideoGen>
- LTX-2: <https://github.com/Lightricks/LTX-2>

