package main

import (
	"fmt"
	"log"
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
    result[idx] = count / 6; // Divide by 6 to account for overcounting
}
`

func main() {
	// Adjacency matrix for portals
	adjMatrix := []int{
		0, 1, 1, 1, 0,
		1, 0, 1, 0, 1,
		1, 1, 0, 1, 0,
		1, 0, 1, 0, 1,
		0, 1, 0, 1, 0,
	}
	n := 5 // Number of portals

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
	// Copy adjacency matrix to device
	err := cu.MemcpyHtoD(dAdjMatrix, unsafe.Pointer(&adjMatrix[0]), int64(len(adjMatrix)*4))
	if err != nil {
		log.Fatalf("Failed to copy adjacency matrix to device: %v", err)
	}
	}

	// Compile kernel
	module, err := cu.ModuleLoadData(kernelSource)
	if err != nil {
		log.Fatalf("Failed to load kernel module: %v", err)
	}
	defer module.Unload()

	kernel, err := module.GetFunction("countTriangles")
	if err != nil {
		log.Fatalf("Failed to get kernel function: %v", err)
	}

	// Launch kernel
	blockSize := 256
	gridSize := (n + blockSize - 1) / blockSize
	args := [][]interface{}{
		{dAdjMatrix},
		{dResult},
		{n},
	}
	err = kernel.Launch(args, gridSize, 1, 1, blockSize, 1, 1, 0, nil)
	if err != nil {
		log.Fatalf("Failed to launch kernel: %v", err)
	}

	// Retrieve result
	result := make([]int, n)
	err = cu.MemcpyDtoH(result, dResult)
	if err != nil {
		log.Fatalf("Failed to copy result from device: %v", err)
	}

	// Output result
	totalTriangles := 0
	for _, count := range result {
		totalTriangles += count
	}
	fmt.Printf("Total number of triangles (fields): %d\n", totalTriangles)
}
