package main

import (
	"fmt"
	"unsafe"

	"github.com/mumax/3/cuda/cu"
)

const kernelSource = `
extern "C" __global__ void countTriangles(int* adjMatrix, int* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int count = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            count += adjMatrix[idx * n + i] * adjMatrix[i * n + j] * adjMatrix[j * n + idx];
        }
    }
    result[idx] = count / 6; // Divide by 6 to account for overcounting
}
`

func main() {
	// Define adjacency matrix for portals (5x5)
	adjMatrix := []int32{
		0, 1, 1, 1, 0,
		1, 0, 1, 0, 1,
		1, 1, 0, 1, 0,
		1, 0, 1, 0, 1,
		0, 1, 0, 1, 0,
	}
	n := int32(5) // Number of portals

	// Initialize CUDA
	cu.Init(0)
	device := cu.DeviceGet(0)
	ctx := cu.CtxCreate(0, device)
	defer ctx.Destroy()

	// Allocate device memory
	dAdjMatrix := cu.MemAlloc(int64(len(adjMatrix) * 4))
	defer cu.MemFree(dAdjMatrix)

	dResult := cu.MemAlloc(int64(n * 4))
	defer cu.MemFree(dResult)

	// Copy adjacency matrix to GPU
	cu.MemcpyHtoD(dAdjMatrix, unsafe.Pointer(&adjMatrix[0]), int64(len(adjMatrix)*4))

	// Compile kernel
	module := cu.ModuleLoadData(kernelSource) // No Unload() method, automatically cleaned up

	kernel := cu.ModuleGetFunction(module, "countTriangles") // Correct function retrieval

	// Kernel Launch Configuration
	blockSize := 256
	gridSize := (int(n) + blockSize - 1) / blockSize // Ensures full coverage

	// 🚀 **Fixed: Pass arguments as a slice of unsafe.Pointer** 🚀
	args := []unsafe.Pointer{
		unsafe.Pointer(&dAdjMatrix),
		unsafe.Pointer(&dResult),
		unsafe.Pointer(&n),
	}

	cu.LaunchKernel(
		kernel,
		gridSize, 1, 1, // Grid dimensions
		blockSize, 1, 1, // Block dimensions
		0, cu.Stream(0), // Shared memory, Stream
		args, // Pass arguments as a slice of unsafe.Pointer
	)

	// Retrieve result from GPU
	result := make([]int32, n)
	cu.MemcpyDtoH(unsafe.Pointer(&result[0]), dResult, int64(n*4))

	// Calculate total triangles
	totalTriangles := 0
	for _, count := range result {
		totalTriangles += int(count)
	}
	fmt.Printf("Total number of triangles (fields): %d\n", totalTriangles)
}
