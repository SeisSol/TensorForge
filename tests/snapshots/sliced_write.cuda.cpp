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
          float v32_data = r0[0];
          float v33_data = s0[88];
          float v35_data = ir1[0];
          ir1[0] = (v35_data + (v32_data * v33_data));
          float v38_data = s0[101];
          float v40_data = ir1[1];
          ir1[1] = (v40_data + (v32_data * v38_data));
          float v43_data = s0[114];
          float v45_data = ir1[2];
          ir1[2] = (v45_data + (v32_data * v43_data));
          float v48_data = s0[127];
          float v50_data = ir1[3];
          ir1[3] = (v50_data + (v32_data * v48_data));
          float v53_data = s0[140];
          float v55_data = ir1[4];
          ir1[4] = (v55_data + (v32_data * v53_data));
          float v58_data = s0[153];
          float v60_data = ir1[5];
          ir1[5] = (v60_data + (v32_data * v58_data));
          float v63_data = s0[166];
          float v65_data = ir1[6];
          ir1[6] = (v65_data + (v32_data * v63_data));
          float v70_data = r0[1];
          float v71_data = s0[89];
          float v73_data = ir1[0];
          ir1[0] = (v73_data + (v70_data * v71_data));
          float v76_data = s0[102];
          float v78_data = ir1[1];
          ir1[1] = (v78_data + (v70_data * v76_data));
          float v81_data = s0[115];
          float v83_data = ir1[2];
          ir1[2] = (v83_data + (v70_data * v81_data));
          float v86_data = s0[128];
          float v88_data = ir1[3];
          ir1[3] = (v88_data + (v70_data * v86_data));
          float v91_data = s0[141];
          float v93_data = ir1[4];
          ir1[4] = (v93_data + (v70_data * v91_data));
          float v96_data = s0[154];
          float v98_data = ir1[5];
          ir1[5] = (v98_data + (v70_data * v96_data));
          float v101_data = s0[167];
          float v103_data = ir1[6];
          ir1[6] = (v103_data + (v70_data * v101_data));
          float v108_data = r0[2];
          float v109_data = s0[90];
          float v111_data = ir1[0];
          ir1[0] = (v111_data + (v108_data * v109_data));
          float v114_data = s0[103];
          float v116_data = ir1[1];
          ir1[1] = (v116_data + (v108_data * v114_data));
          float v119_data = s0[116];
          float v121_data = ir1[2];
          ir1[2] = (v121_data + (v108_data * v119_data));
          float v124_data = s0[129];
          float v126_data = ir1[3];
          ir1[3] = (v126_data + (v108_data * v124_data));
          float v129_data = s0[142];
          float v131_data = ir1[4];
          ir1[4] = (v131_data + (v108_data * v129_data));
          float v134_data = s0[155];
          float v136_data = ir1[5];
          ir1[5] = (v136_data + (v108_data * v134_data));
          float v139_data = s0[168];
          float v141_data = ir1[6];
          ir1[6] = (v141_data + (v108_data * v139_data));
          #pragma unroll
          for (int32_t v146_n0 = 0; v146_n0 < 1; ++v146_n0) {
            #pragma unroll
            for (int32_t v147_n1 = 6; v147_n1 < 13; ++v147_n1) {
              int32_t v148_a = v147_n1 - 6;
              int32_t v149_a = v146_n0 + v148_a;
              int32_t v151_a = v146_n0 + v148_a;
              float v152_data = ir1[v151_a];
              int32_t v154_a = v146_n0 + v148_a;
              r1[v151_a] = v152_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v160_i0 = 0; v160_i0 < 1; ++v160_i0) {
            int32_t v165_lead = v160_i0 * 32;
            int32_t v167_a = (v6_lead + v165_lead) + 0;
            glb_m0[v167_a] = 0.0f;
            int32_t v174_a = (v6_lead + v165_lead) + 32;
            glb_m0[v174_a] = 0.0f;
            int32_t v181_a = (v6_lead + v165_lead) + 64;
            glb_m0[v181_a] = 0.0f;
            int32_t v188_a = (v6_lead + v165_lead) + 96;
            glb_m0[v188_a] = 0.0f;
            int32_t v195_a = (v6_lead + v165_lead) + 128;
            glb_m0[v195_a] = 0.0f;
            int32_t v202_a = (v6_lead + v165_lead) + 160;
            glb_m0[v202_a] = 0.0f;
            int32_t v203_a = v160_i0 + 0;
            float v205_data = r1[v160_i0];
            int32_t v211_a = (v6_lead + v165_lead) + 192;
            glb_m0[v211_a] = v205_data;
            int32_t v212_a = v160_i0 + 1;
            float v214_data = r1[(v160_i0 + 1)];
            int32_t v220_a = (v6_lead + v165_lead) + 224;
            glb_m0[v220_a] = v214_data;
            int32_t v221_a = v160_i0 + 2;
            float v223_data = r1[(v160_i0 + 2)];
            int32_t v229_a = (v6_lead + v165_lead) + 256;
            glb_m0[v229_a] = v223_data;
            int32_t v230_a = v160_i0 + 3;
            float v232_data = r1[(v160_i0 + 3)];
            int32_t v238_a = (v6_lead + v165_lead) + 288;
            glb_m0[v238_a] = v232_data;
            int32_t v239_a = v160_i0 + 4;
            float v241_data = r1[(v160_i0 + 4)];
            int32_t v247_a = (v6_lead + v165_lead) + 320;
            glb_m0[v247_a] = v241_data;
            int32_t v248_a = v160_i0 + 5;
            float v250_data = r1[(v160_i0 + 5)];
            int32_t v256_a = (v6_lead + v165_lead) + 352;
            glb_m0[v256_a] = v250_data;
            int32_t v257_a = v160_i0 + 6;
            float v259_data = r1[(v160_i0 + 6)];
            int32_t v265_a = (v6_lead + v165_lead) + 384;
            glb_m0[v265_a] = v259_data;
          }
          __syncwarp();
        }
      }
    }
  }
}

