// === base name ===
kernel_49acf988a6

// === header ===
void launcher_kernel_49acf988a6(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_49acf988a6(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_49acf988a6, block.x * block.y * block.z, 1536 * sizeof(float));
        CHECK_ERR;
        if (blocksPerSM > 0) {
          gridsize = smCount * blocksPerSM;
        }
        else {
          gridsize = smCount;
        }
      }
      
  dim3 grid (std::min(gridsize, numElements0), 1, 1);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_49acf988a6, cudaFuncAttributeMaxDynamicSharedMemorySize, 1536 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_49acf988a6<<<grid,block,1536 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_49acf988a6(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 32×13(32×13) {0..32}×{0..13} strided
    // m1 32×13(32×13) {0..32}×{0..13} strided
    // m2 13×13(13×13) {0..13}×{0..13} strided
    // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{6..13})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{10..13})[0, -1]×m2 13×13(13×13) {0..13}×{0..13} strided({10..13}×{6..13})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[192 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 416 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 416 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 169 + 0 + m2_extraOffset];
          float r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v17_lead = v10_lead + (v11_i0 * 32);
            #pragma unroll
            for (int32_t v12_i1 = 10; v12_i1 < 13; ++v12_i1) {
              float v20_data = __ldcg(&glb_m1[(v17_lead + (v12_i1 * 32))]);
              r0[(v11_i0 + (v12_i1 - 10))] = v20_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 5; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 32], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 32], 4);
            __pipeline_commit();
          }
          if (threadIdx.x < 9) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 160], &glb_m2[0 + 0 + 1 * threadIdx.x + 160], 4);
            __pipeline_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[7]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 32), (6, 13)] [(10, 13)]
          float ir1[7]{};
          float v31_data = r0[0];
          float v32_data = s0[88];
          float v34_data = ir1[0];
          ir1[0] = (v34_data + (v31_data * v32_data));
          float v37_data = s0[101];
          float v39_data = ir1[1];
          ir1[1] = (v39_data + (v31_data * v37_data));
          float v42_data = s0[114];
          float v44_data = ir1[2];
          ir1[2] = (v44_data + (v31_data * v42_data));
          float v47_data = s0[127];
          float v49_data = ir1[3];
          ir1[3] = (v49_data + (v31_data * v47_data));
          float v52_data = s0[140];
          float v54_data = ir1[4];
          ir1[4] = (v54_data + (v31_data * v52_data));
          float v57_data = s0[153];
          float v59_data = ir1[5];
          ir1[5] = (v59_data + (v31_data * v57_data));
          float v62_data = s0[166];
          float v64_data = ir1[6];
          ir1[6] = (v64_data + (v31_data * v62_data));
          float v69_data = r0[1];
          float v70_data = s0[89];
          float v72_data = ir1[0];
          ir1[0] = (v72_data + (v69_data * v70_data));
          float v75_data = s0[102];
          float v77_data = ir1[1];
          ir1[1] = (v77_data + (v69_data * v75_data));
          float v80_data = s0[115];
          float v82_data = ir1[2];
          ir1[2] = (v82_data + (v69_data * v80_data));
          float v85_data = s0[128];
          float v87_data = ir1[3];
          ir1[3] = (v87_data + (v69_data * v85_data));
          float v90_data = s0[141];
          float v92_data = ir1[4];
          ir1[4] = (v92_data + (v69_data * v90_data));
          float v95_data = s0[154];
          float v97_data = ir1[5];
          ir1[5] = (v97_data + (v69_data * v95_data));
          float v100_data = s0[167];
          float v102_data = ir1[6];
          ir1[6] = (v102_data + (v69_data * v100_data));
          float v107_data = r0[2];
          float v108_data = s0[90];
          float v110_data = ir1[0];
          ir1[0] = (v110_data + (v107_data * v108_data));
          float v113_data = s0[103];
          float v115_data = ir1[1];
          ir1[1] = (v115_data + (v107_data * v113_data));
          float v118_data = s0[116];
          float v120_data = ir1[2];
          ir1[2] = (v120_data + (v107_data * v118_data));
          float v123_data = s0[129];
          float v125_data = ir1[3];
          ir1[3] = (v125_data + (v107_data * v123_data));
          float v128_data = s0[142];
          float v130_data = ir1[4];
          ir1[4] = (v130_data + (v107_data * v128_data));
          float v133_data = s0[155];
          float v135_data = ir1[5];
          ir1[5] = (v135_data + (v107_data * v133_data));
          float v138_data = s0[168];
          float v140_data = ir1[6];
          ir1[6] = (v140_data + (v107_data * v138_data));
          #pragma unroll
          for (int32_t v145_n0 = 0; v145_n0 < 1; ++v145_n0) {
            #pragma unroll
            for (int32_t v146_n1 = 6; v146_n1 < 13; ++v146_n1) {
              int32_t v148_a = v145_n0 + (v146_n1 - 6);
              float v149_data = ir1[v148_a];
              r1[v148_a] = v149_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v155_i0 = 0; v155_i0 < 1; ++v155_i0) {
            int32_t v160_lead = v155_i0 * 32;
            glb_m0[(v10_lead + v160_lead)] = 0.0f;
            glb_m0[((v10_lead + v160_lead) + 32)] = 0.0f;
            glb_m0[((v10_lead + v160_lead) + 64)] = 0.0f;
            glb_m0[((v10_lead + v160_lead) + 96)] = 0.0f;
            glb_m0[((v10_lead + v160_lead) + 128)] = 0.0f;
            glb_m0[((v10_lead + v160_lead) + 160)] = 0.0f;
            float v199_data = r1[v155_i0];
            glb_m0[((v10_lead + v160_lead) + 192)] = v199_data;
            float v207_data = r1[(v155_i0 + 1)];
            glb_m0[((v10_lead + v160_lead) + 224)] = v207_data;
            float v215_data = r1[(v155_i0 + 2)];
            glb_m0[((v10_lead + v160_lead) + 256)] = v215_data;
            float v223_data = r1[(v155_i0 + 3)];
            glb_m0[((v10_lead + v160_lead) + 288)] = v223_data;
            float v231_data = r1[(v155_i0 + 4)];
            glb_m0[((v10_lead + v160_lead) + 320)] = v231_data;
            float v239_data = r1[(v155_i0 + 5)];
            glb_m0[((v10_lead + v160_lead) + 352)] = v239_data;
            float v247_data = r1[(v155_i0 + 6)];
            glb_m0[((v10_lead + v160_lead) + 384)] = v247_data;
          }
          __syncwarp();
        }
      }
    }
  }
}

