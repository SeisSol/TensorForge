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
          {
            // r1 = +(r0 * s0) + None
            // [(0, 32), (6, 13)] [(10, 13)]
            float ir1[7]{};
            float v27_data = r0[0];
            float v28_data = s0[88];
            float v30_data = ir1[0];
            ir1[0] = (v30_data + (v27_data * v28_data));
            float v33_data = s0[101];
            float v35_data = ir1[1];
            ir1[1] = (v35_data + (v27_data * v33_data));
            float v38_data = s0[114];
            float v40_data = ir1[2];
            ir1[2] = (v40_data + (v27_data * v38_data));
            float v43_data = s0[127];
            float v45_data = ir1[3];
            ir1[3] = (v45_data + (v27_data * v43_data));
            float v48_data = s0[140];
            float v50_data = ir1[4];
            ir1[4] = (v50_data + (v27_data * v48_data));
            float v53_data = s0[153];
            float v55_data = ir1[5];
            ir1[5] = (v55_data + (v27_data * v53_data));
            float v58_data = s0[166];
            float v60_data = ir1[6];
            ir1[6] = (v60_data + (v27_data * v58_data));
            float v65_data = r0[1];
            float v66_data = s0[89];
            float v68_data = ir1[0];
            ir1[0] = (v68_data + (v65_data * v66_data));
            float v71_data = s0[102];
            float v73_data = ir1[1];
            ir1[1] = (v73_data + (v65_data * v71_data));
            float v76_data = s0[115];
            float v78_data = ir1[2];
            ir1[2] = (v78_data + (v65_data * v76_data));
            float v81_data = s0[128];
            float v83_data = ir1[3];
            ir1[3] = (v83_data + (v65_data * v81_data));
            float v86_data = s0[141];
            float v88_data = ir1[4];
            ir1[4] = (v88_data + (v65_data * v86_data));
            float v91_data = s0[154];
            float v93_data = ir1[5];
            ir1[5] = (v93_data + (v65_data * v91_data));
            float v96_data = s0[167];
            float v98_data = ir1[6];
            ir1[6] = (v98_data + (v65_data * v96_data));
            float v103_data = r0[2];
            float v104_data = s0[90];
            float v106_data = ir1[0];
            ir1[0] = (v106_data + (v103_data * v104_data));
            float v109_data = s0[103];
            float v111_data = ir1[1];
            ir1[1] = (v111_data + (v103_data * v109_data));
            float v114_data = s0[116];
            float v116_data = ir1[2];
            ir1[2] = (v116_data + (v103_data * v114_data));
            float v119_data = s0[129];
            float v121_data = ir1[3];
            ir1[3] = (v121_data + (v103_data * v119_data));
            float v124_data = s0[142];
            float v126_data = ir1[4];
            ir1[4] = (v126_data + (v103_data * v124_data));
            float v129_data = s0[155];
            float v131_data = ir1[5];
            ir1[5] = (v131_data + (v103_data * v129_data));
            float v134_data = s0[168];
            float v136_data = ir1[6];
            ir1[6] = (v136_data + (v103_data * v134_data));
            #pragma unroll
            for (int32_t v141_n0 = 0; v141_n0 < 1; ++v141_n0) {
              #pragma unroll
              for (int32_t v142_n1 = 6; v142_n1 < 13; ++v142_n1) {
                int32_t v143_a = v142_n1 - 6;
                int32_t v144_a = v141_n0 + v143_a;
                int32_t v146_a = v141_n0 + v143_a;
                float v147_data = ir1[v146_a];
                int32_t v149_a = v141_n0 + v143_a;
                r1[v146_a] = v147_data;
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v155_i0 = 0; v155_i0 < 1; ++v155_i0) {
            int32_t v160_lead = v155_i0 * 32;
            int32_t v162_a = (v3_lead + v160_lead) + 0;
            glb_m0[v162_a] = 0.0f;
            int32_t v169_a = (v3_lead + v160_lead) + 32;
            glb_m0[v169_a] = 0.0f;
            int32_t v176_a = (v3_lead + v160_lead) + 64;
            glb_m0[v176_a] = 0.0f;
            int32_t v183_a = (v3_lead + v160_lead) + 96;
            glb_m0[v183_a] = 0.0f;
            int32_t v190_a = (v3_lead + v160_lead) + 128;
            glb_m0[v190_a] = 0.0f;
            int32_t v197_a = (v3_lead + v160_lead) + 160;
            glb_m0[v197_a] = 0.0f;
            int32_t v198_a = v155_i0 + 0;
            float v200_data = r1[v155_i0];
            int32_t v206_a = (v3_lead + v160_lead) + 192;
            glb_m0[v206_a] = v200_data;
            int32_t v207_a = v155_i0 + 1;
            float v209_data = r1[(v155_i0 + 1)];
            int32_t v215_a = (v3_lead + v160_lead) + 224;
            glb_m0[v215_a] = v209_data;
            int32_t v216_a = v155_i0 + 2;
            float v218_data = r1[(v155_i0 + 2)];
            int32_t v224_a = (v3_lead + v160_lead) + 256;
            glb_m0[v224_a] = v218_data;
            int32_t v225_a = v155_i0 + 3;
            float v227_data = r1[(v155_i0 + 3)];
            int32_t v233_a = (v3_lead + v160_lead) + 288;
            glb_m0[v233_a] = v227_data;
            int32_t v234_a = v155_i0 + 4;
            float v236_data = r1[(v155_i0 + 4)];
            int32_t v242_a = (v3_lead + v160_lead) + 320;
            glb_m0[v242_a] = v236_data;
            int32_t v243_a = v155_i0 + 5;
            float v245_data = r1[(v155_i0 + 5)];
            int32_t v251_a = (v3_lead + v160_lead) + 352;
            glb_m0[v251_a] = v245_data;
            int32_t v252_a = v155_i0 + 6;
            float v254_data = r1[(v155_i0 + 6)];
            int32_t v260_a = (v3_lead + v160_lead) + 384;
            glb_m0[v260_a] = v254_data;
          }
          __syncwarp();
        }
      }
    }
  }
}

