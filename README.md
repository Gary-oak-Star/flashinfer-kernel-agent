# FlashInfer AI Kernel Generation Agent

AI-powered CUDA kernel generator for LLM operations using Triton/CuTe.

## Competition Info
- **Competition**: MLSys 2026 FlashInfer NVIDIA Track
- **Deadline**: April 24, 2026
- **Prize**: NVIDIA GPU hardware + MLSys registration
- **Link**: https://mlsys26.flashinfer.ai/

## Targets
1. Fused MoE (FP8 Mixture-of-Experts)
2. Gated Delta Net (Qwen3-Next)

## Approach
Multi-stage kernel generation:
1. Analyze kernel specification
2. Generate initial Triton/CUDA code
3. Compile and benchmark
4. Profile bottlenecks
5. Iteratively optimize

## Setup
```bash
pip install -r requirements.txt
# Modal compute credits provided by organizers
```

## Status
- [x] Project setup
- [ ] Clone FlashInfer baseline
- [ ] Implement generation pipeline
- [ ] Test on sample kernels
- [ ] Optimize for B200
