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
          int32_t v2_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v9_lead = v2_lead + (v3_i0 * 32);
            #pragma unroll
            for (int32_t v4_i1 = 10; v4_i1 < 13; ++v4_i1) {
              int32_t v11_a = v9_lead + (v4_i1 * 32);
              float v12_data;
              {
                v12_data = __ldcg(&glb_m1[v11_a]);
              }
              int32_t v14_a = v3_i0 + (v4_i1 - 10);
              r0[v14_a] = v12_data;
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
          {
            // r1 = +(r0 * s0) + None
            // [(0, 32), (6, 13)] [(10, 13)]
            float ir1[7]{};
            float v18_data = r0[0];
            float v19_data = s0[88];
            float v21_data = ir1[0];
            ir1[0] = (v21_data + (v18_data * v19_data));
            float v24_data = s0[101];
            float v26_data = ir1[1];
            ir1[1] = (v26_data + (v18_data * v24_data));
            float v29_data = s0[114];
            float v31_data = ir1[2];
            ir1[2] = (v31_data + (v18_data * v29_data));
            float v34_data = s0[127];
            float v36_data = ir1[3];
            ir1[3] = (v36_data + (v18_data * v34_data));
            float v39_data = s0[140];
            float v41_data = ir1[4];
            ir1[4] = (v41_data + (v18_data * v39_data));
            float v44_data = s0[153];
            float v46_data = ir1[5];
            ir1[5] = (v46_data + (v18_data * v44_data));
            float v49_data = s0[166];
            float v51_data = ir1[6];
            ir1[6] = (v51_data + (v18_data * v49_data));
            float v56_data = r0[1];
            float v57_data = s0[89];
            float v59_data = ir1[0];
            ir1[0] = (v59_data + (v56_data * v57_data));
            float v62_data = s0[102];
            float v64_data = ir1[1];
            ir1[1] = (v64_data + (v56_data * v62_data));
            float v67_data = s0[115];
            float v69_data = ir1[2];
            ir1[2] = (v69_data + (v56_data * v67_data));
            float v72_data = s0[128];
            float v74_data = ir1[3];
            ir1[3] = (v74_data + (v56_data * v72_data));
            float v77_data = s0[141];
            float v79_data = ir1[4];
            ir1[4] = (v79_data + (v56_data * v77_data));
            float v82_data = s0[154];
            float v84_data = ir1[5];
            ir1[5] = (v84_data + (v56_data * v82_data));
            float v87_data = s0[167];
            float v89_data = ir1[6];
            ir1[6] = (v89_data + (v56_data * v87_data));
            float v94_data = r0[2];
            float v95_data = s0[90];
            float v97_data = ir1[0];
            ir1[0] = (v97_data + (v94_data * v95_data));
            float v100_data = s0[103];
            float v102_data = ir1[1];
            ir1[1] = (v102_data + (v94_data * v100_data));
            float v105_data = s0[116];
            float v107_data = ir1[2];
            ir1[2] = (v107_data + (v94_data * v105_data));
            float v110_data = s0[129];
            float v112_data = ir1[3];
            ir1[3] = (v112_data + (v94_data * v110_data));
            float v115_data = s0[142];
            float v117_data = ir1[4];
            ir1[4] = (v117_data + (v94_data * v115_data));
            float v120_data = s0[155];
            float v122_data = ir1[5];
            ir1[5] = (v122_data + (v94_data * v120_data));
            float v125_data = s0[168];
            float v127_data = ir1[6];
            ir1[6] = (v127_data + (v94_data * v125_data));
            #pragma unroll
            for (int32_t v132_n0 = 0; v132_n0 < 1; ++v132_n0) {
              #pragma unroll
              for (int32_t v133_n1 = 6; v133_n1 < 13; ++v133_n1) {
                int32_t v134_a = v133_n1 - 6;
                int32_t v135_a = v132_n0 + v134_a;
                float v136_data = ir1[v135_a];
                int32_t v138_a = v132_n0 + v134_a;
                r1[v138_a] = v136_data;
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          int32_t v141_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v142_i0 = 0; v142_i0 < 1; ++v142_i0) {
            int32_t v147_lead = v142_i0 * 32;
            int32_t v149_a = (v141_lead + v147_lead) + 0;
            glb_m0[v149_a] = 0.0f;
            int32_t v156_a = (v141_lead + v147_lead) + 32;
            glb_m0[v156_a] = 0.0f;
            int32_t v163_a = (v141_lead + v147_lead) + 64;
            glb_m0[v163_a] = 0.0f;
            int32_t v170_a = (v141_lead + v147_lead) + 96;
            glb_m0[v170_a] = 0.0f;
            int32_t v177_a = (v141_lead + v147_lead) + 128;
            glb_m0[v177_a] = 0.0f;
            int32_t v184_a = (v141_lead + v147_lead) + 160;
            glb_m0[v184_a] = 0.0f;
            int32_t v185_a = v142_i0 + 0;
            float v186_data = r1[v185_a];
            int32_t v192_a = (v141_lead + v147_lead) + 192;
            glb_m0[v192_a] = v186_data;
            int32_t v193_a = v142_i0 + 1;
            float v194_data = r1[v193_a];
            int32_t v200_a = (v141_lead + v147_lead) + 224;
            glb_m0[v200_a] = v194_data;
            int32_t v201_a = v142_i0 + 2;
            float v202_data = r1[v201_a];
            int32_t v208_a = (v141_lead + v147_lead) + 256;
            glb_m0[v208_a] = v202_data;
            int32_t v209_a = v142_i0 + 3;
            float v210_data = r1[v209_a];
            int32_t v216_a = (v141_lead + v147_lead) + 288;
            glb_m0[v216_a] = v210_data;
            int32_t v217_a = v142_i0 + 4;
            float v218_data = r1[v217_a];
            int32_t v224_a = (v141_lead + v147_lead) + 320;
            glb_m0[v224_a] = v218_data;
            int32_t v225_a = v142_i0 + 5;
            float v226_data = r1[v225_a];
            int32_t v232_a = (v141_lead + v147_lead) + 352;
            glb_m0[v232_a] = v226_data;
            int32_t v233_a = v142_i0 + 6;
            float v234_data = r1[v233_a];
            int32_t v240_a = (v141_lead + v147_lead) + 384;
            glb_m0[v240_a] = v234_data;
          }
          __syncwarp();
        }
      }
    }
  }
}

