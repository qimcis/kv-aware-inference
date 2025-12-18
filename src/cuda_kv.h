// CUDA component used to move kv blocks into device slots
#pragma once

#include <cstddef>
#include <cuda_runtime.h>

struct DeviceKVLayout {
    float* keys = nullptr;
    float* values = nullptr;
    float* staging_keys = nullptr;
    float* staging_values = nullptr;
    std::size_t max_blocks = 0;
    std::size_t block_size = 0;
    std::size_t hidden = 0;
};

DeviceKVLayout allocate_device_kv(std::size_t max_blocks, std::size_t block_size, std::size_t hidden);

void free_device_kv(DeviceKVLayout& layout);

// copy cpu-generated k/v into the staging area, async
void stage_block(const float* host_keys, const float* host_values, std::size_t tokens, DeviceKVLayout& layout, cudaStream_t stream);

// kernel launcher that writes the staged data into a block slot on device
void move_block_to_cache(const DeviceKVLayout& layout, std::size_t block_idx, std::size_t tokens, cudaStream_t stream);
