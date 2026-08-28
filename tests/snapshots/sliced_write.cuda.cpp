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
          int32_t v6_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
            int32_t v12_lead = v7_i0 * 32;
            int32_t v13_lead = v6_lead + v12_lead;
            int32_t v20_lead = v6_lead + v12_lead;
            #pragma unroll
            for (int32_t v8_i1 = 10; v8_i1 < 13; ++v8_i1) {
              int32_t v14_a = v8_i1 * 32;
              int32_t v15_a = v13_lead + v14_a;
              float v23_data = __ldcg(&glb_m1[(v20_lead + v14_a)]);
              int32_t v25_a = v7_i0 + (v8_i1 - 10);
              r0[v25_a] = v23_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
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
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[7]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 32), (6, 13)] [(10, 13)]
          float ir1[7]{};
          float v34_data = r0[0];
          float v35_data = s0[88];
          float v37_data = ir1[0];
          ir1[0] = (v37_data + (v34_data * v35_data));
          float v40_data = s0[101];
          float v42_data = ir1[1];
          ir1[1] = (v42_data + (v34_data * v40_data));
          float v45_data = s0[114];
          float v47_data = ir1[2];
          ir1[2] = (v47_data + (v34_data * v45_data));
          float v50_data = s0[127];
          float v52_data = ir1[3];
          ir1[3] = (v52_data + (v34_data * v50_data));
          float v55_data = s0[140];
          float v57_data = ir1[4];
          ir1[4] = (v57_data + (v34_data * v55_data));
          float v60_data = s0[153];
          float v62_data = ir1[5];
          ir1[5] = (v62_data + (v34_data * v60_data));
          float v65_data = s0[166];
          float v67_data = ir1[6];
          ir1[6] = (v67_data + (v34_data * v65_data));
          float v72_data = r0[1];
          float v73_data = s0[89];
          float v75_data = ir1[0];
          ir1[0] = (v75_data + (v72_data * v73_data));
          float v78_data = s0[102];
          float v80_data = ir1[1];
          ir1[1] = (v80_data + (v72_data * v78_data));
          float v83_data = s0[115];
          float v85_data = ir1[2];
          ir1[2] = (v85_data + (v72_data * v83_data));
          float v88_data = s0[128];
          float v90_data = ir1[3];
          ir1[3] = (v90_data + (v72_data * v88_data));
          float v93_data = s0[141];
          float v95_data = ir1[4];
          ir1[4] = (v95_data + (v72_data * v93_data));
          float v98_data = s0[154];
          float v100_data = ir1[5];
          ir1[5] = (v100_data + (v72_data * v98_data));
          float v103_data = s0[167];
          float v105_data = ir1[6];
          ir1[6] = (v105_data + (v72_data * v103_data));
          float v110_data = r0[2];
          float v111_data = s0[90];
          float v113_data = ir1[0];
          ir1[0] = (v113_data + (v110_data * v111_data));
          float v116_data = s0[103];
          float v118_data = ir1[1];
          ir1[1] = (v118_data + (v110_data * v116_data));
          float v121_data = s0[116];
          float v123_data = ir1[2];
          ir1[2] = (v123_data + (v110_data * v121_data));
          float v126_data = s0[129];
          float v128_data = ir1[3];
          ir1[3] = (v128_data + (v110_data * v126_data));
          float v131_data = s0[142];
          float v133_data = ir1[4];
          ir1[4] = (v133_data + (v110_data * v131_data));
          float v136_data = s0[155];
          float v138_data = ir1[5];
          ir1[5] = (v138_data + (v110_data * v136_data));
          float v141_data = s0[168];
          float v143_data = ir1[6];
          ir1[6] = (v143_data + (v110_data * v141_data));
          #pragma unroll
          for (int32_t v148_n0 = 0; v148_n0 < 1; ++v148_n0) {
            #pragma unroll
            for (int32_t v149_n1 = 6; v149_n1 < 13; ++v149_n1) {
              int32_t v150_a = v149_n1 - 6;
              int32_t v151_a = v148_n0 + v150_a;
              int32_t v153_a = v148_n0 + v150_a;
              float v154_data = ir1[v153_a];
              r1[v153_a] = v154_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v160_i0 = 0; v160_i0 < 1; ++v160_i0) {
            int32_t v165_lead = v160_i0 * 32;
            glb_m0[(v6_lead + v165_lead)] = 0.0f;
            glb_m0[((v6_lead + v165_lead) + 32)] = 0.0f;
            glb_m0[((v6_lead + v165_lead) + 64)] = 0.0f;
            glb_m0[((v6_lead + v165_lead) + 96)] = 0.0f;
            glb_m0[((v6_lead + v165_lead) + 128)] = 0.0f;
            glb_m0[((v6_lead + v165_lead) + 160)] = 0.0f;
            int32_t v203_a = v160_i0 + 0;
            float v205_data = r1[v160_i0];
            glb_m0[((v6_lead + v165_lead) + 192)] = v205_data;
            int32_t v212_a = v160_i0 + 1;
            float v214_data = r1[(v160_i0 + 1)];
            glb_m0[((v6_lead + v165_lead) + 224)] = v214_data;
            int32_t v221_a = v160_i0 + 2;
            float v223_data = r1[(v160_i0 + 2)];
            glb_m0[((v6_lead + v165_lead) + 256)] = v223_data;
            int32_t v230_a = v160_i0 + 3;
            float v232_data = r1[(v160_i0 + 3)];
            glb_m0[((v6_lead + v165_lead) + 288)] = v232_data;
            int32_t v239_a = v160_i0 + 4;
            float v241_data = r1[(v160_i0 + 4)];
            glb_m0[((v6_lead + v165_lead) + 320)] = v241_data;
            int32_t v248_a = v160_i0 + 5;
            float v250_data = r1[(v160_i0 + 5)];
            glb_m0[((v6_lead + v165_lead) + 352)] = v250_data;
            int32_t v257_a = v160_i0 + 6;
            float v259_data = r1[(v160_i0 + 6)];
            glb_m0[((v6_lead + v165_lead) + 384)] = v259_data;
          }
          __syncwarp();
        }
      }
    }
  }
}

