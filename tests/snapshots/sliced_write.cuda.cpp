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
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 416 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 416 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 169 + 0 + m2_extraOffset];
          float r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v3_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v4_i0 = 0; v4_i0 < 1; ++v4_i0) {
            int32_t v9_lead = v4_i0 * 32;
            int32_t v10_lead = v3_lead + v9_lead;
            int32_t v17_lead = v3_lead + v9_lead;
            #pragma unroll
            for (int32_t v5_i1 = 10; v5_i1 < 13; ++v5_i1) {
              int32_t v11_a = v5_i1 * 32;
              int32_t v12_a = v10_lead + v11_a;
              float v20_data = __ldcg(&glb_m1[(v17_lead + v11_a)]);
              int32_t v22_a = v4_i0 + (v5_i1 - 10);
              r0[v22_a] = v20_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 5; i += 1) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 32], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 32], cuda::aligned_size_t<4>(4), pipeline);
            }
            if (threadIdx.x < 9) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 160], &glb_m2[0 + 0 + 1 * threadIdx.x + 160], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r1[7]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 32), (6, 13)] [(10, 13)]
          float ir1[7]{};
          float v28_data = r0[0];
          float v29_data = s0[88];
          float v31_data = ir1[0];
          ir1[0] = (v31_data + (v28_data * v29_data));
          float v34_data = s0[101];
          float v36_data = ir1[1];
          ir1[1] = (v36_data + (v28_data * v34_data));
          float v39_data = s0[114];
          float v41_data = ir1[2];
          ir1[2] = (v41_data + (v28_data * v39_data));
          float v44_data = s0[127];
          float v46_data = ir1[3];
          ir1[3] = (v46_data + (v28_data * v44_data));
          float v49_data = s0[140];
          float v51_data = ir1[4];
          ir1[4] = (v51_data + (v28_data * v49_data));
          float v54_data = s0[153];
          float v56_data = ir1[5];
          ir1[5] = (v56_data + (v28_data * v54_data));
          float v59_data = s0[166];
          float v61_data = ir1[6];
          ir1[6] = (v61_data + (v28_data * v59_data));
          float v66_data = r0[1];
          float v67_data = s0[89];
          float v69_data = ir1[0];
          ir1[0] = (v69_data + (v66_data * v67_data));
          float v72_data = s0[102];
          float v74_data = ir1[1];
          ir1[1] = (v74_data + (v66_data * v72_data));
          float v77_data = s0[115];
          float v79_data = ir1[2];
          ir1[2] = (v79_data + (v66_data * v77_data));
          float v82_data = s0[128];
          float v84_data = ir1[3];
          ir1[3] = (v84_data + (v66_data * v82_data));
          float v87_data = s0[141];
          float v89_data = ir1[4];
          ir1[4] = (v89_data + (v66_data * v87_data));
          float v92_data = s0[154];
          float v94_data = ir1[5];
          ir1[5] = (v94_data + (v66_data * v92_data));
          float v97_data = s0[167];
          float v99_data = ir1[6];
          ir1[6] = (v99_data + (v66_data * v97_data));
          float v104_data = r0[2];
          float v105_data = s0[90];
          float v107_data = ir1[0];
          ir1[0] = (v107_data + (v104_data * v105_data));
          float v110_data = s0[103];
          float v112_data = ir1[1];
          ir1[1] = (v112_data + (v104_data * v110_data));
          float v115_data = s0[116];
          float v117_data = ir1[2];
          ir1[2] = (v117_data + (v104_data * v115_data));
          float v120_data = s0[129];
          float v122_data = ir1[3];
          ir1[3] = (v122_data + (v104_data * v120_data));
          float v125_data = s0[142];
          float v127_data = ir1[4];
          ir1[4] = (v127_data + (v104_data * v125_data));
          float v130_data = s0[155];
          float v132_data = ir1[5];
          ir1[5] = (v132_data + (v104_data * v130_data));
          float v135_data = s0[168];
          float v137_data = ir1[6];
          ir1[6] = (v137_data + (v104_data * v135_data));
          #pragma unroll
          for (int32_t v142_n0 = 0; v142_n0 < 1; ++v142_n0) {
            #pragma unroll
            for (int32_t v143_n1 = 6; v143_n1 < 13; ++v143_n1) {
              int32_t v144_a = v143_n1 - 6;
              int32_t v145_a = v142_n0 + v144_a;
              int32_t v147_a = v142_n0 + v144_a;
              float v148_data = ir1[v147_a];
              int32_t v150_a = v142_n0 + v144_a;
              r1[v147_a] = v148_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v156_i0 = 0; v156_i0 < 1; ++v156_i0) {
            int32_t v161_lead = v156_i0 * 32;
            int32_t v163_a = (v3_lead + v161_lead) + 0;
            glb_m0[v163_a] = 0.0f;
            int32_t v170_a = (v3_lead + v161_lead) + 32;
            glb_m0[v170_a] = 0.0f;
            int32_t v177_a = (v3_lead + v161_lead) + 64;
            glb_m0[v177_a] = 0.0f;
            int32_t v184_a = (v3_lead + v161_lead) + 96;
            glb_m0[v184_a] = 0.0f;
            int32_t v191_a = (v3_lead + v161_lead) + 128;
            glb_m0[v191_a] = 0.0f;
            int32_t v198_a = (v3_lead + v161_lead) + 160;
            glb_m0[v198_a] = 0.0f;
            int32_t v199_a = v156_i0 + 0;
            float v201_data = r1[v156_i0];
            int32_t v207_a = (v3_lead + v161_lead) + 192;
            glb_m0[v207_a] = v201_data;
            int32_t v208_a = v156_i0 + 1;
            float v210_data = r1[(v156_i0 + 1)];
            int32_t v216_a = (v3_lead + v161_lead) + 224;
            glb_m0[v216_a] = v210_data;
            int32_t v217_a = v156_i0 + 2;
            float v219_data = r1[(v156_i0 + 2)];
            int32_t v225_a = (v3_lead + v161_lead) + 256;
            glb_m0[v225_a] = v219_data;
            int32_t v226_a = v156_i0 + 3;
            float v228_data = r1[(v156_i0 + 3)];
            int32_t v234_a = (v3_lead + v161_lead) + 288;
            glb_m0[v234_a] = v228_data;
            int32_t v235_a = v156_i0 + 4;
            float v237_data = r1[(v156_i0 + 4)];
            int32_t v243_a = (v3_lead + v161_lead) + 320;
            glb_m0[v243_a] = v237_data;
            int32_t v244_a = v156_i0 + 5;
            float v246_data = r1[(v156_i0 + 5)];
            int32_t v252_a = (v3_lead + v161_lead) + 352;
            glb_m0[v252_a] = v246_data;
            int32_t v253_a = v156_i0 + 6;
            float v255_data = r1[(v156_i0 + 6)];
            int32_t v261_a = (v3_lead + v161_lead) + 384;
            glb_m0[v261_a] = v255_data;
          }
          __syncwarp();
        }
      }
    }
  }
}

