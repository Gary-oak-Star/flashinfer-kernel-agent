#!/usr/bin/env python3
"""
Test Gemini's ability to generate simple GPU kernels
"""
import os
import json
import urllib.request

def call_gemini(prompt):
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("⚠️  No GOOGLE_API_KEY, using mock")
        return "# Mock kernel\nimport triton\n\n# TODO: Implement"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    req = urllib.request.Request(url, json.dumps(data).encode(), {'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            result = json.loads(r.read())
            return result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        print(f"API Error: {e}")
        return None

def test_simple_kernel():
    """Test generating a simple vector add kernel"""
    print("🧪 Test 1: Simple Vector Add Kernel\n")
    
    prompt = """You are an expert in GPU programming with Triton.

Task: Generate a Triton kernel for vector addition.

Requirements:
- Input: two vectors A and B of size N
- Output: vector C = A + B
- Use Triton's @triton.jit decorator
- Include proper block size configuration
- Add comments explaining key optimizations

Generate the complete Triton kernel code:"""
    
    print("📤 Calling Gemini API...")
    response = call_gemini(prompt)
    
    if response:
        print("✅ Kernel generated!\n")
        print("=" * 60)
        
        # Extract code
        if '```python' in response:
            code = response.split('```python')[1].split('```')[0].strip()
        elif '```' in response:
            code = response.split('```')[1].split('```')[0].strip()
        else:
            code = response
        
        print(code)
        print("=" * 60)
        
        # Save to file
        with open('kernels/vector_add.py', 'w') as f:
            f.write(code)
        
        print("\n💾 Saved to kernels/vector_add.py")
        
        # Basic validation
        has_triton = 'triton' in code.lower()
        has_jit = '@triton.jit' in code or '@jit' in code
        has_function = 'def ' in code
        
        print(f"\n📊 Validation:")
        print(f"  Has Triton import: {has_triton}")
        print(f"  Has @jit decorator: {has_jit}")
        print(f"  Has function def: {has_function}")
        
        if has_triton and has_jit and has_function:
            print("\n✅ Basic structure looks correct!")
            return True
        else:
            print("\n⚠️  Missing some components")
            return False
    else:
        print("❌ Failed to generate kernel")
        return False

def test_moe_spec():
    """Test understanding MoE kernel requirements"""
    print("\n🧪 Test 2: Understanding MoE Requirements\n")
    
    prompt = """Explain the key optimization challenges for a Mixture-of-Experts (MoE) kernel on GPU:

Requirements:
- FP8 precision
- Top-K routing (select top 2 experts per token)
- 8 experts total
- Minimize memory bandwidth
- Maximize tensor core utilization

List the top 3 optimization strategies:"""
    
    print("📤 Asking about MoE optimization...")
    response = call_gemini(prompt)
    
    if response:
        print("\n📝 AI Response:")
        print("-" * 60)
        print(response[:500] + "..." if len(response) > 500 else response)
        print("-" * 60)
        return True
    return False

if __name__ == '__main__':
    print("🎯 FlashInfer Kernel Generation Tests\n")
    
    # Test 1
    success1 = test_simple_kernel()
    
    # Test 2
    success2 = test_moe_spec()
    
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"  Simple kernel generation: {'✅' if success1 else '❌'}")
    print(f"  MoE understanding: {'✅' if success2 else '❌'}")
    print("=" * 60)
    
    if success1 and success2:
        print("\n🎉 FlashInfer Agent基础能力验证成功！")
        print("\n📋 下一步:")
        print("  1. 测试编译生成的 kernel")
        print("  2. 添加 benchmark 循环")
        print("  3. 实现迭代优化")
    else:
        print("\n⚠️  需要调试 API 或 prompt")
