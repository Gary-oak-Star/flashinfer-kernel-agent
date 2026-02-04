import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(
    A,  # Pointer to the first input vector
    B,  # Pointer to the second input vector
    C,  # Pointer to the output vector
    N: tl.constexpr,  # Number of elements in the vectors (passed as a compile-time constant)
    BLOCK_SIZE: tl.constexpr, # Number of elements each program instance computes.
):
    """Kernel for vector addition C = A + B.

    This kernel adds two vectors, A and B, element-wise and stores the result in vector C.
    It leverages Triton's programming model for efficient parallel execution on GPUs.

    Args:
        A: Pointer to the first input vector.
        B: Pointer to the second input vector.
        C: Pointer to the output vector.
        N: Number of elements in the vectors (passed as a compile-time constant).
        BLOCK_SIZE: Number of elements each program instance computes.  This must be a power of 2.
    """
    # Determine the global program ID
    pid = tl.program_id(axis=0)

    # Compute the memory offsets for A, B, and C
    # Using program id and block size to calculate the starting memory addresses.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to guard against out-of-bounds memory accesses
    mask = offsets < N

    # Load data from memory using the mask
    # tl.load efficiently loads elements based on the mask, preventing out-of-bounds reads.
    a = tl.load(A + offsets, mask=mask)
    b = tl.load(B + offsets, mask=mask)

    # Perform the addition
    c = a + b

    # Store the result back to memory using the mask
    # tl.store ensures that only valid elements within the mask are written back to memory.
    tl.store(C + offsets, c, mask=mask)


def add(a, b, block_size=1024):
    """
    Vector addition using Triton.

    Args:
        a: The first input vector (torch.Tensor).
        b: The second input vector (torch.Tensor).
        block_size: The block size to use for the Triton kernel (default: 1024).  Must be a power of 2.

    Returns:
        The result vector (torch.Tensor).
    """
    # Check the input shapes
    assert a.shape == b.shape, "Input tensors must have the same shape"
    n = a.shape[0]

    # Allocate output tensor
    c = torch.empty_like(a)

    # Launch the kernel
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)

    add_kernel[grid](
        a, b, c, n, BLOCK_SIZE=block_size
    )

    return c

if __name__ == '__main__':
    # Example Usage
    n = 1024 * 1024  # Example vector size
    a = torch.rand(n, device='cuda')
    b = torch.rand(n, device='cuda')

    # Run the Triton kernel
    c_triton = add(a, b)

    # Verify the result against PyTorch
    c_torch = a + b
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-5, atol=1e-5) # Reduced tolerances to avoid intermittent failures
    print(f"✅ Triton result matches PyTorch result")

    # Benchmark the Triton kernel
    ms = triton.testing.do_bench(lambda: add(a, b))
    print(f"Triton kernel took: {ms:.4f} ms")