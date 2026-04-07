import torch
import torchvision.models as models
import time


def check_gpu_info():
    print("=" * 50)
    print("1. SYSTEM & GPU CAPABILITIES")
    print("=" * 50)
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your drivers.")
        exit()

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"GPU Name:          {props.name}")
    print(f"Compute Capability:{props.major}.{props.minor}")
    print(f"Total VRAM:        {props.total_memory / (1024**3):.2f} GB")
    print(f"Multi-Processors:  {props.multi_processor_count}")
    print("-" * 50)


def benchmark_gemm(size=8192, iterations=50):
    print("\n" + "=" * 50)
    print(f"2. RAW COMPUTE (Matrix Multiplication {size}x{size})")
    print("=" * 50)

    # Operations for multiplying two NxN matrices: 2 * N^3
    ops_per_iteration = 2.0 * (size**3)

    for dtype, name in [
        (torch.float32, "FP32 (Standard)"),
        (torch.float16, "FP16 (Tensor Cores)"),
    ]:
        A = torch.randn(size, size, device="cuda", dtype=dtype)
        B = torch.randn(size, size, device="cuda", dtype=dtype)

        # Warmup (critical to spin up GPU clocks and initialize CUDA context)
        for _ in range(10):
            _ = torch.matmul(A, B)
        torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            _ = torch.matmul(A, B)
        torch.cuda.synchronize()
        end_time = time.time()

        avg_time = (end_time - start_time) / iterations
        tflops = (ops_per_iteration / avg_time) / 1e12

        print(f"[{name}]")
        print(f"Avg Time per op: {avg_time*1000:.2f} ms")
        print(f"Throughput:      {tflops:.2f} TFLOPS")
        print("-" * 30)


def benchmark_resnet(batch_size=64, iterations=30):
    print("\n" + "=" * 50)
    print(f"3. REAL-WORLD WORKLOAD (ResNet50, Batch Size {batch_size})")
    print("=" * 50)

    model = models.resnet50().cuda()
    model.eval()  # Set to inference mode

    # Use FP16 using PyTorch AMP (Automatic Mixed Precision)
    # T4 is heavily optimized for mixed precision inference
    scaler = torch.cuda.amp.GradScaler()
    inputs = torch.randn(batch_size, 3, 224, 224, device="cuda")

    # Warmup
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(5):
            _ = model(inputs)
    torch.cuda.synchronize()

    # Benchmark Forward Pass
    start_time = time.time()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(iterations):
            _ = model(inputs)
    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    images_per_sec = (batch_size * iterations) / total_time

    print(f"Mode:             Mixed Precision (FP16/FP32)")
    print(f"Total Time:       {total_time:.2f} seconds")
    print(f"Throughput:       {images_per_sec:.2f} Images / second")


def benchmark_memory_bandwidth(size_mb=500, iterations=50):
    print("\n" + "=" * 50)
    print(f"4. VRAM BANDWIDTH TEST ({size_mb} MB Transfer)")
    print("=" * 50)

    # 500 MB of float32 data
    elements = (size_mb * 1024 * 1024) // 4

    # CPU to GPU
    cpu_tensor = torch.randn(elements, dtype=torch.float32)

    # Warmup
    gpu_tensor = cpu_tensor.to("cuda")
    torch.cuda.synchronize()

    # Host to Device
    start = time.time()
    for _ in range(iterations):
        _ = cpu_tensor.to("cuda")
    torch.cuda.synchronize()
    h2d_time = (time.time() - start) / iterations
    h2d_bw = size_mb / h2d_time / 1024  # GB/s

    # Device to Device
    start = time.time()
    for _ in range(iterations):
        _ = gpu_tensor.clone()
    torch.cuda.synchronize()
    d2d_time = (time.time() - start) / iterations
    d2d_bw = size_mb / d2d_time / 1024  # GB/s

    print(f"CPU -> GPU (PCIe):  {h2d_bw:.2f} GB/s")
    print(f"GPU -> GPU (VRAM):  {d2d_bw:.2f} GB/s")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    check_gpu_info()
    benchmark_gemm()
    benchmark_resnet()
    benchmark_memory_bandwidth()
