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
            int32_t v16_lead = v11_i0 * 32;
            int32_t v17_lead = v10_lead + v16_lead;
            int32_t v24_lead = v10_lead + v16_lead;
            #pragma unroll
            for (int32_t v12_i1 = 10; v12_i1 < 13; ++v12_i1) {
              int32_t v18_a = v12_i1 * 32;
              int32_t v19_a = v17_lead + v18_a;
              float v27_data = __ldcg(&glb_m1[(v24_lead + v18_a)]);
              r0[(v11_i0 + (v12_i1 - 10))] = v27_data;
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
          float v38_data = r0[0];
          float v39_data = s0[88];
          float v41_data = ir1[0];
          ir1[0] = (v41_data + (v38_data * v39_data));
          float v44_data = s0[101];
          float v46_data = ir1[1];
          ir1[1] = (v46_data + (v38_data * v44_data));
          float v49_data = s0[114];
          float v51_data = ir1[2];
          ir1[2] = (v51_data + (v38_data * v49_data));
          float v54_data = s0[127];
          float v56_data = ir1[3];
          ir1[3] = (v56_data + (v38_data * v54_data));
          float v59_data = s0[140];
          float v61_data = ir1[4];
          ir1[4] = (v61_data + (v38_data * v59_data));
          float v64_data = s0[153];
          float v66_data = ir1[5];
          ir1[5] = (v66_data + (v38_data * v64_data));
          float v69_data = s0[166];
          float v71_data = ir1[6];
          ir1[6] = (v71_data + (v38_data * v69_data));
          float v76_data = r0[1];
          float v77_data = s0[89];
          float v79_data = ir1[0];
          ir1[0] = (v79_data + (v76_data * v77_data));
          float v82_data = s0[102];
          float v84_data = ir1[1];
          ir1[1] = (v84_data + (v76_data * v82_data));
          float v87_data = s0[115];
          float v89_data = ir1[2];
          ir1[2] = (v89_data + (v76_data * v87_data));
          float v92_data = s0[128];
          float v94_data = ir1[3];
          ir1[3] = (v94_data + (v76_data * v92_data));
          float v97_data = s0[141];
          float v99_data = ir1[4];
          ir1[4] = (v99_data + (v76_data * v97_data));
          float v102_data = s0[154];
          float v104_data = ir1[5];
          ir1[5] = (v104_data + (v76_data * v102_data));
          float v107_data = s0[167];
          float v109_data = ir1[6];
          ir1[6] = (v109_data + (v76_data * v107_data));
          float v114_data = r0[2];
          float v115_data = s0[90];
          float v117_data = ir1[0];
          ir1[0] = (v117_data + (v114_data * v115_data));
          float v120_data = s0[103];
          float v122_data = ir1[1];
          ir1[1] = (v122_data + (v114_data * v120_data));
          float v125_data = s0[116];
          float v127_data = ir1[2];
          ir1[2] = (v127_data + (v114_data * v125_data));
          float v130_data = s0[129];
          float v132_data = ir1[3];
          ir1[3] = (v132_data + (v114_data * v130_data));
          float v135_data = s0[142];
          float v137_data = ir1[4];
          ir1[4] = (v137_data + (v114_data * v135_data));
          float v140_data = s0[155];
          float v142_data = ir1[5];
          ir1[5] = (v142_data + (v114_data * v140_data));
          float v145_data = s0[168];
          float v147_data = ir1[6];
          ir1[6] = (v147_data + (v114_data * v145_data));
          #pragma unroll
          for (int32_t v152_n0 = 0; v152_n0 < 1; ++v152_n0) {
            #pragma unroll
            for (int32_t v153_n1 = 6; v153_n1 < 13; ++v153_n1) {
              int32_t v154_a = v153_n1 - 6;
              int32_t v155_a = v152_n0 + v154_a;
              int32_t v157_a = v152_n0 + v154_a;
              float v158_data = ir1[v157_a];
              r1[v157_a] = v158_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v164_i0 = 0; v164_i0 < 1; ++v164_i0) {
            int32_t v169_lead = v164_i0 * 32;
            glb_m0[(v10_lead + v169_lead)] = 0.0f;
            glb_m0[((v10_lead + v169_lead) + 32)] = 0.0f;
            glb_m0[((v10_lead + v169_lead) + 64)] = 0.0f;
            glb_m0[((v10_lead + v169_lead) + 96)] = 0.0f;
            glb_m0[((v10_lead + v169_lead) + 128)] = 0.0f;
            glb_m0[((v10_lead + v169_lead) + 160)] = 0.0f;
            int32_t v207_a = v164_i0 + 0;
            float v209_data = r1[v164_i0];
            glb_m0[((v10_lead + v169_lead) + 192)] = v209_data;
            int32_t v216_a = v164_i0 + 1;
            float v218_data = r1[(v164_i0 + 1)];
            glb_m0[((v10_lead + v169_lead) + 224)] = v218_data;
            int32_t v225_a = v164_i0 + 2;
            float v227_data = r1[(v164_i0 + 2)];
            glb_m0[((v10_lead + v169_lead) + 256)] = v227_data;
            int32_t v234_a = v164_i0 + 3;
            float v236_data = r1[(v164_i0 + 3)];
            glb_m0[((v10_lead + v169_lead) + 288)] = v236_data;
            int32_t v243_a = v164_i0 + 4;
            float v245_data = r1[(v164_i0 + 4)];
            glb_m0[((v10_lead + v169_lead) + 320)] = v245_data;
            int32_t v252_a = v164_i0 + 5;
            float v254_data = r1[(v164_i0 + 5)];
            glb_m0[((v10_lead + v169_lead) + 352)] = v254_data;
            int32_t v261_a = v164_i0 + 6;
            float v263_data = r1[(v164_i0 + 6)];
            glb_m0[((v10_lead + v169_lead) + 384)] = v263_data;
          }
          __syncwarp();
        }
      }
    }
  }
}

