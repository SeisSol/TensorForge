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
          float v29_data = r0[0];
          float v30_data = s0[88];
          float v32_data = ir1[0];
          ir1[0] = (v32_data + (v29_data * v30_data));
          float v35_data = s0[101];
          float v37_data = ir1[1];
          ir1[1] = (v37_data + (v29_data * v35_data));
          float v40_data = s0[114];
          float v42_data = ir1[2];
          ir1[2] = (v42_data + (v29_data * v40_data));
          float v45_data = s0[127];
          float v47_data = ir1[3];
          ir1[3] = (v47_data + (v29_data * v45_data));
          float v50_data = s0[140];
          float v52_data = ir1[4];
          ir1[4] = (v52_data + (v29_data * v50_data));
          float v55_data = s0[153];
          float v57_data = ir1[5];
          ir1[5] = (v57_data + (v29_data * v55_data));
          float v60_data = s0[166];
          float v62_data = ir1[6];
          ir1[6] = (v62_data + (v29_data * v60_data));
          float v67_data = r0[1];
          float v68_data = s0[89];
          float v70_data = ir1[0];
          ir1[0] = (v70_data + (v67_data * v68_data));
          float v73_data = s0[102];
          float v75_data = ir1[1];
          ir1[1] = (v75_data + (v67_data * v73_data));
          float v78_data = s0[115];
          float v80_data = ir1[2];
          ir1[2] = (v80_data + (v67_data * v78_data));
          float v83_data = s0[128];
          float v85_data = ir1[3];
          ir1[3] = (v85_data + (v67_data * v83_data));
          float v88_data = s0[141];
          float v90_data = ir1[4];
          ir1[4] = (v90_data + (v67_data * v88_data));
          float v93_data = s0[154];
          float v95_data = ir1[5];
          ir1[5] = (v95_data + (v67_data * v93_data));
          float v98_data = s0[167];
          float v100_data = ir1[6];
          ir1[6] = (v100_data + (v67_data * v98_data));
          float v105_data = r0[2];
          float v106_data = s0[90];
          float v108_data = ir1[0];
          ir1[0] = (v108_data + (v105_data * v106_data));
          float v111_data = s0[103];
          float v113_data = ir1[1];
          ir1[1] = (v113_data + (v105_data * v111_data));
          float v116_data = s0[116];
          float v118_data = ir1[2];
          ir1[2] = (v118_data + (v105_data * v116_data));
          float v121_data = s0[129];
          float v123_data = ir1[3];
          ir1[3] = (v123_data + (v105_data * v121_data));
          float v126_data = s0[142];
          float v128_data = ir1[4];
          ir1[4] = (v128_data + (v105_data * v126_data));
          float v131_data = s0[155];
          float v133_data = ir1[5];
          ir1[5] = (v133_data + (v105_data * v131_data));
          float v136_data = s0[168];
          float v138_data = ir1[6];
          ir1[6] = (v138_data + (v105_data * v136_data));
          #pragma unroll
          for (int32_t v143_n0 = 0; v143_n0 < 1; ++v143_n0) {
            #pragma unroll
            for (int32_t v144_n1 = 6; v144_n1 < 13; ++v144_n1) {
              int32_t v145_a = v144_n1 - 6;
              int32_t v146_a = v143_n0 + v145_a;
              int32_t v148_a = v143_n0 + v145_a;
              float v149_data = ir1[v148_a];
              int32_t v151_a = v143_n0 + v145_a;
              r1[v148_a] = v149_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v157_i0 = 0; v157_i0 < 1; ++v157_i0) {
            int32_t v162_lead = v157_i0 * 32;
            int32_t v164_a = (v3_lead + v162_lead) + 0;
            glb_m0[v164_a] = 0.0f;
            int32_t v171_a = (v3_lead + v162_lead) + 32;
            glb_m0[v171_a] = 0.0f;
            int32_t v178_a = (v3_lead + v162_lead) + 64;
            glb_m0[v178_a] = 0.0f;
            int32_t v185_a = (v3_lead + v162_lead) + 96;
            glb_m0[v185_a] = 0.0f;
            int32_t v192_a = (v3_lead + v162_lead) + 128;
            glb_m0[v192_a] = 0.0f;
            int32_t v199_a = (v3_lead + v162_lead) + 160;
            glb_m0[v199_a] = 0.0f;
            int32_t v200_a = v157_i0 + 0;
            float v202_data = r1[v157_i0];
            int32_t v208_a = (v3_lead + v162_lead) + 192;
            glb_m0[v208_a] = v202_data;
            int32_t v209_a = v157_i0 + 1;
            float v211_data = r1[(v157_i0 + 1)];
            int32_t v217_a = (v3_lead + v162_lead) + 224;
            glb_m0[v217_a] = v211_data;
            int32_t v218_a = v157_i0 + 2;
            float v220_data = r1[(v157_i0 + 2)];
            int32_t v226_a = (v3_lead + v162_lead) + 256;
            glb_m0[v226_a] = v220_data;
            int32_t v227_a = v157_i0 + 3;
            float v229_data = r1[(v157_i0 + 3)];
            int32_t v235_a = (v3_lead + v162_lead) + 288;
            glb_m0[v235_a] = v229_data;
            int32_t v236_a = v157_i0 + 4;
            float v238_data = r1[(v157_i0 + 4)];
            int32_t v244_a = (v3_lead + v162_lead) + 320;
            glb_m0[v244_a] = v238_data;
            int32_t v245_a = v157_i0 + 5;
            float v247_data = r1[(v157_i0 + 5)];
            int32_t v253_a = (v3_lead + v162_lead) + 352;
            glb_m0[v253_a] = v247_data;
            int32_t v254_a = v157_i0 + 6;
            float v256_data = r1[(v157_i0 + 6)];
            int32_t v262_a = (v3_lead + v162_lead) + 384;
            glb_m0[v262_a] = v256_data;
          }
          __syncwarp();
        }
      }
    }
  }
}

