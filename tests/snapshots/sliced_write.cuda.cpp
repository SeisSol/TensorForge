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
            int32_t v8_lead = v3_i0 * 32;
            int32_t v9_lead = v2_lead + v8_lead;
            int32_t v16_lead = v2_lead + v8_lead;
            #pragma unroll
            for (int32_t v4_i1 = 10; v4_i1 < 13; ++v4_i1) {
              int32_t v10_a = v4_i1 * 32;
              int32_t v11_a = v9_lead + v10_a;
              float v19_data = __ldcg(&glb_m1[(v16_lead + v10_a)]);
              int32_t v21_a = v3_i0 + (v4_i1 - 10);
              r0[v21_a] = v19_data;
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
            float v25_data = r0[0];
            float v26_data = s0[88];
            float v28_data = ir1[0];
            ir1[0] = (v28_data + (v25_data * v26_data));
            float v31_data = s0[101];
            float v33_data = ir1[1];
            ir1[1] = (v33_data + (v25_data * v31_data));
            float v36_data = s0[114];
            float v38_data = ir1[2];
            ir1[2] = (v38_data + (v25_data * v36_data));
            float v41_data = s0[127];
            float v43_data = ir1[3];
            ir1[3] = (v43_data + (v25_data * v41_data));
            float v46_data = s0[140];
            float v48_data = ir1[4];
            ir1[4] = (v48_data + (v25_data * v46_data));
            float v51_data = s0[153];
            float v53_data = ir1[5];
            ir1[5] = (v53_data + (v25_data * v51_data));
            float v56_data = s0[166];
            float v58_data = ir1[6];
            ir1[6] = (v58_data + (v25_data * v56_data));
            float v63_data = r0[1];
            float v64_data = s0[89];
            float v66_data = ir1[0];
            ir1[0] = (v66_data + (v63_data * v64_data));
            float v69_data = s0[102];
            float v71_data = ir1[1];
            ir1[1] = (v71_data + (v63_data * v69_data));
            float v74_data = s0[115];
            float v76_data = ir1[2];
            ir1[2] = (v76_data + (v63_data * v74_data));
            float v79_data = s0[128];
            float v81_data = ir1[3];
            ir1[3] = (v81_data + (v63_data * v79_data));
            float v84_data = s0[141];
            float v86_data = ir1[4];
            ir1[4] = (v86_data + (v63_data * v84_data));
            float v89_data = s0[154];
            float v91_data = ir1[5];
            ir1[5] = (v91_data + (v63_data * v89_data));
            float v94_data = s0[167];
            float v96_data = ir1[6];
            ir1[6] = (v96_data + (v63_data * v94_data));
            float v101_data = r0[2];
            float v102_data = s0[90];
            float v104_data = ir1[0];
            ir1[0] = (v104_data + (v101_data * v102_data));
            float v107_data = s0[103];
            float v109_data = ir1[1];
            ir1[1] = (v109_data + (v101_data * v107_data));
            float v112_data = s0[116];
            float v114_data = ir1[2];
            ir1[2] = (v114_data + (v101_data * v112_data));
            float v117_data = s0[129];
            float v119_data = ir1[3];
            ir1[3] = (v119_data + (v101_data * v117_data));
            float v122_data = s0[142];
            float v124_data = ir1[4];
            ir1[4] = (v124_data + (v101_data * v122_data));
            float v127_data = s0[155];
            float v129_data = ir1[5];
            ir1[5] = (v129_data + (v101_data * v127_data));
            float v132_data = s0[168];
            float v134_data = ir1[6];
            ir1[6] = (v134_data + (v101_data * v132_data));
            #pragma unroll
            for (int32_t v139_n0 = 0; v139_n0 < 1; ++v139_n0) {
              #pragma unroll
              for (int32_t v140_n1 = 6; v140_n1 < 13; ++v140_n1) {
                int32_t v141_a = v140_n1 - 6;
                int32_t v142_a = v139_n0 + v141_a;
                int32_t v144_a = v139_n0 + v141_a;
                float v145_data = ir1[v144_a];
                int32_t v147_a = v139_n0 + v141_a;
                r1[v144_a] = v145_data;
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v153_i0 = 0; v153_i0 < 1; ++v153_i0) {
            int32_t v158_lead = v153_i0 * 32;
            int32_t v160_a = (v2_lead + v158_lead) + 0;
            glb_m0[v160_a] = 0.0f;
            int32_t v167_a = (v2_lead + v158_lead) + 32;
            glb_m0[v167_a] = 0.0f;
            int32_t v174_a = (v2_lead + v158_lead) + 64;
            glb_m0[v174_a] = 0.0f;
            int32_t v181_a = (v2_lead + v158_lead) + 96;
            glb_m0[v181_a] = 0.0f;
            int32_t v188_a = (v2_lead + v158_lead) + 128;
            glb_m0[v188_a] = 0.0f;
            int32_t v195_a = (v2_lead + v158_lead) + 160;
            glb_m0[v195_a] = 0.0f;
            int32_t v196_a = v153_i0 + 0;
            float v198_data = r1[v153_i0];
            int32_t v204_a = (v2_lead + v158_lead) + 192;
            glb_m0[v204_a] = v198_data;
            int32_t v205_a = v153_i0 + 1;
            float v207_data = r1[(v153_i0 + 1)];
            int32_t v213_a = (v2_lead + v158_lead) + 224;
            glb_m0[v213_a] = v207_data;
            int32_t v214_a = v153_i0 + 2;
            float v216_data = r1[(v153_i0 + 2)];
            int32_t v222_a = (v2_lead + v158_lead) + 256;
            glb_m0[v222_a] = v216_data;
            int32_t v223_a = v153_i0 + 3;
            float v225_data = r1[(v153_i0 + 3)];
            int32_t v231_a = (v2_lead + v158_lead) + 288;
            glb_m0[v231_a] = v225_data;
            int32_t v232_a = v153_i0 + 4;
            float v234_data = r1[(v153_i0 + 4)];
            int32_t v240_a = (v2_lead + v158_lead) + 320;
            glb_m0[v240_a] = v234_data;
            int32_t v241_a = v153_i0 + 5;
            float v243_data = r1[(v153_i0 + 5)];
            int32_t v249_a = (v2_lead + v158_lead) + 352;
            glb_m0[v249_a] = v243_data;
            int32_t v250_a = v153_i0 + 6;
            float v252_data = r1[(v153_i0 + 6)];
            int32_t v258_a = (v2_lead + v158_lead) + 384;
            glb_m0[v258_a] = v252_data;
          }
          __syncwarp();
        }
      }
    }
  }
}

