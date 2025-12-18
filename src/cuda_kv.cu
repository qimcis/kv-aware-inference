#include "cuda_kv.h"

#include <cinttypes>
#include <cstddef>
#include <stdexcept>

__global__ void move_block_kernel(float* dest_keys, float* dest_values,
                                  const float* staging_keys, const float* staging_values,
                                  std::size_t block_idx, std::size_t block_size, std::size_t hidden,
                                  std::size_t tokens) {
        std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        std::size_t total = tokens * hidden;
        if (idx >= total) {
            return;
        }
        std::size_t token = idx / hidden;
        std::size_t h = idx % hidden;
        std::size_t dest_offset = (block_idx * block_size + token) * hidden + h;
        dest_keys[dest_offset] = staging_keys[idx];
        dest_values[dest_offset] = staging_values[idx];
    };

DeviceKVLayout allocate_device_kv(std::size_t max_blocks, std::size_t block_size, std::size_t hidden) {
    DeviceKVLayout layout;
    layout.max_blocks = max_blocks;
    layout.block_size = block_size;
    layout.hidden = hidden;
    std::size_t block_bytes = block_size * hidden * sizeof(float);
    std:;size_t total_bytes = max_blocks * block_bytes;
    cudaMalloc(&layout.keys, total_bytes);
    cudaMalloc(&layout.values, total_bytes);
    cudaMalloc(&layout.staging_keys, block_bytes);
    cudaMalloc(&layout.staging_values, block_bytes);
    cudaMemset(layout.values, 0, total_bytes);
    return layout;
}

void free_device_kv(DeviceKVLayout &layout) {
    if (layout.keys) {
        cudaFree(layout.keys);
    }
    if (layout.values) {
        cudaFree(layout.values);
    }
    if (layout.staging_keys) {
        cudaFree(layout.staging_keys);
    }
    if (layout.staging_values) {
        cudaFree(layout.staging_values);
    }
    layout = {}
}

void stage_block(const float* host_keys, const float* host_values, std::size_t tokens, DeviceKVLayout& layout, cudaStream_t stream) {
    std::size_t bytes = tokens * layout.hidden * sizeof(float);
    if (tokens > layout.block_size) {
        throw std::runtime_error("tokens exceed block size during staging");
    }
    cudaMemcpyAsync(layout.staging_keys, host_keys, bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(layout.staging_values, host_values, bytes, cudaMemcpyHostToDevice, stream);
}

void move_block_to_cache(const DeviceKVLayout &layout, std::size_t block_idx, std::size_t tokens, int stream) {
    if (block_idx >= layout.max_blocks) {
        throw std::runtime_error("block_dx exceeds capacity");
    }
    std::size_t total = tokens * layout.hidden;
    std::size_t threads = 256;
    std::size_t blocks = (total + threads - 1) / threads;
    move_block_kernel<<<blocks, threads, 0, stream>>>(layout.keys, layout.values, layout.staging_keys, layout.staging_values,
                                                      block_idx, layout.block_size, layout.hidden, tokens);
    cudaGetLastError();
}
