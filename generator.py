#!/usr/bin/env python3
"""
FlashInfer Kernel Generation Agent
"""
import os
import google.generativeai as genai

def generate_kernel(spec):
    """Generate optimized kernel from specification"""
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""You are an expert CUDA/Triton kernel developer.

Task: Generate an optimized GPU kernel for the following specification.

Specification:
{spec}

Requirements:
- Target: NVIDIA Blackwell B200 GPU
- Use Triton for ease of generation
- Optimize for:
  * Memory coalescing
  * Shared memory usage
  * Warp-level primitives
  * FP8 tensor cores (if applicable)

Output the complete Triton kernel code with comments.
"""
    
    response = model.generate_content(prompt)
    return response.text

def main():
    print("FlashInfer Kernel Generator")
    print("=" * 50)
    
    # Example: MoE kernel spec
    spec = """
    Fused Mixture-of-Experts kernel:
    - Input: FP8 tensor (batch_size, seq_len, hidden_dim)
    - Experts: 8 experts, each (hidden_dim, expert_dim)
    - Top-K: Select top 2 experts per token
    - Output: FP8 tensor (batch_size, seq_len, hidden_dim)
    """
    
    print("Generating kernel...")
    kernel_code = generate_kernel(spec)
    
    print("\nGenerated Kernel:")
    print(kernel_code)
    
    # Save to file
    with open('generated_kernel.py', 'w') as f:
        f.write(kernel_code)
    
    print("\n✅ Kernel saved to generated_kernel.py")

if __name__ == '__main__':
    main()
