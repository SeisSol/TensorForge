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
          int32_t v7_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v8_i0 = 0; v8_i0 < 1; ++v8_i0) {
            int32_t v13_lead = v8_i0 * 32;
            int32_t v14_lead = v7_lead + v13_lead;
            int32_t v21_lead = v7_lead + v13_lead;
            #pragma unroll
            for (int32_t v9_i1 = 10; v9_i1 < 13; ++v9_i1) {
              int32_t v15_a = v9_i1 * 32;
              int32_t v16_a = v14_lead + v15_a;
              float v24_data = __ldcg(&glb_m1[(v21_lead + v15_a)]);
              r0[(v8_i0 + (v9_i1 - 10))] = v24_data;
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
          float v35_data = r0[0];
          float v36_data = s0[88];
          float v38_data = ir1[0];
          ir1[0] = (v38_data + (v35_data * v36_data));
          float v41_data = s0[101];
          float v43_data = ir1[1];
          ir1[1] = (v43_data + (v35_data * v41_data));
          float v46_data = s0[114];
          float v48_data = ir1[2];
          ir1[2] = (v48_data + (v35_data * v46_data));
          float v51_data = s0[127];
          float v53_data = ir1[3];
          ir1[3] = (v53_data + (v35_data * v51_data));
          float v56_data = s0[140];
          float v58_data = ir1[4];
          ir1[4] = (v58_data + (v35_data * v56_data));
          float v61_data = s0[153];
          float v63_data = ir1[5];
          ir1[5] = (v63_data + (v35_data * v61_data));
          float v66_data = s0[166];
          float v68_data = ir1[6];
          ir1[6] = (v68_data + (v35_data * v66_data));
          float v73_data = r0[1];
          float v74_data = s0[89];
          float v76_data = ir1[0];
          ir1[0] = (v76_data + (v73_data * v74_data));
          float v79_data = s0[102];
          float v81_data = ir1[1];
          ir1[1] = (v81_data + (v73_data * v79_data));
          float v84_data = s0[115];
          float v86_data = ir1[2];
          ir1[2] = (v86_data + (v73_data * v84_data));
          float v89_data = s0[128];
          float v91_data = ir1[3];
          ir1[3] = (v91_data + (v73_data * v89_data));
          float v94_data = s0[141];
          float v96_data = ir1[4];
          ir1[4] = (v96_data + (v73_data * v94_data));
          float v99_data = s0[154];
          float v101_data = ir1[5];
          ir1[5] = (v101_data + (v73_data * v99_data));
          float v104_data = s0[167];
          float v106_data = ir1[6];
          ir1[6] = (v106_data + (v73_data * v104_data));
          float v111_data = r0[2];
          float v112_data = s0[90];
          float v114_data = ir1[0];
          ir1[0] = (v114_data + (v111_data * v112_data));
          float v117_data = s0[103];
          float v119_data = ir1[1];
          ir1[1] = (v119_data + (v111_data * v117_data));
          float v122_data = s0[116];
          float v124_data = ir1[2];
          ir1[2] = (v124_data + (v111_data * v122_data));
          float v127_data = s0[129];
          float v129_data = ir1[3];
          ir1[3] = (v129_data + (v111_data * v127_data));
          float v132_data = s0[142];
          float v134_data = ir1[4];
          ir1[4] = (v134_data + (v111_data * v132_data));
          float v137_data = s0[155];
          float v139_data = ir1[5];
          ir1[5] = (v139_data + (v111_data * v137_data));
          float v142_data = s0[168];
          float v144_data = ir1[6];
          ir1[6] = (v144_data + (v111_data * v142_data));
          #pragma unroll
          for (int32_t v149_n0 = 0; v149_n0 < 1; ++v149_n0) {
            #pragma unroll
            for (int32_t v150_n1 = 6; v150_n1 < 13; ++v150_n1) {
              int32_t v151_a = v150_n1 - 6;
              int32_t v152_a = v149_n0 + v151_a;
              int32_t v154_a = v149_n0 + v151_a;
              float v155_data = ir1[v154_a];
              r1[v154_a] = v155_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v161_i0 = 0; v161_i0 < 1; ++v161_i0) {
            int32_t v166_lead = v161_i0 * 32;
            glb_m0[(v7_lead + v166_lead)] = 0.0f;
            glb_m0[((v7_lead + v166_lead) + 32)] = 0.0f;
            glb_m0[((v7_lead + v166_lead) + 64)] = 0.0f;
            glb_m0[((v7_lead + v166_lead) + 96)] = 0.0f;
            glb_m0[((v7_lead + v166_lead) + 128)] = 0.0f;
            glb_m0[((v7_lead + v166_lead) + 160)] = 0.0f;
            int32_t v204_a = v161_i0 + 0;
            float v206_data = r1[v161_i0];
            glb_m0[((v7_lead + v166_lead) + 192)] = v206_data;
            int32_t v213_a = v161_i0 + 1;
            float v215_data = r1[(v161_i0 + 1)];
            glb_m0[((v7_lead + v166_lead) + 224)] = v215_data;
            int32_t v222_a = v161_i0 + 2;
            float v224_data = r1[(v161_i0 + 2)];
            glb_m0[((v7_lead + v166_lead) + 256)] = v224_data;
            int32_t v231_a = v161_i0 + 3;
            float v233_data = r1[(v161_i0 + 3)];
            glb_m0[((v7_lead + v166_lead) + 288)] = v233_data;
            int32_t v240_a = v161_i0 + 4;
            float v242_data = r1[(v161_i0 + 4)];
            glb_m0[((v7_lead + v166_lead) + 320)] = v242_data;
            int32_t v249_a = v161_i0 + 5;
            float v251_data = r1[(v161_i0 + 5)];
            glb_m0[((v7_lead + v166_lead) + 352)] = v251_data;
            int32_t v258_a = v161_i0 + 6;
            float v260_data = r1[(v161_i0 + 6)];
            glb_m0[((v7_lead + v166_lead) + 384)] = v260_data;
          }
          __syncwarp();
        }
      }
    }
  }
}

