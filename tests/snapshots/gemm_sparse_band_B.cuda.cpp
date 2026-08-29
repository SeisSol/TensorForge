// === base name ===
kernel_30948bd44e

// === header ===
void launcher_kernel_30948bd44e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_30948bd44e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_30948bd44e, block.x * block.y * block.z, 1280 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_30948bd44e, cudaFuncAttributeMaxDynamicSharedMemorySize, 1280 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_30948bd44e<<<grid,block,1280 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_30948bd44e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×16(16×16) {0..16}×{0..16} strided
    // m1 16×16(16×16) {0..16}×{0..16} strided
    // m2 16×16(16×16) {0..16}×{0..16} strided
    // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[80 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v13_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v14_i0 = 0; v14_i0 < 1; ++v14_i0) {
            int32_t v20_lead = v13_lead + (v14_i0 * 16);
            #pragma unroll
            for (int32_t v15_i1 = 0; v15_i1 < 16; ++v15_i1) {
              float v23_data = __ldcg(&glb_m1[(v20_lead + (v15_i1 * 16))]);
              r0[(v14_i0 + v15_i1)] = v23_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m2[0 + 0 + 1 * threadIdx.x + 16], 4);
          __pipeline_commit();
          if (threadIdx.x < 14) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
            __pipeline_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[16]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float ir1[16]{};
          float v34_data = r0[0];
          float v38_data = s0[(0 ^ ((0 >> 5) & 31))];
          float v40_data = ir1[0];
          ir1[0] = (v40_data + (v34_data * v38_data));
          float v46_data = s0[(2 ^ ((2 >> 5) & 31))];
          float v48_data = ir1[1];
          ir1[1] = (v48_data + (v34_data * v46_data));
          float v67_data = r0[1];
          float v71_data = s0[(1 ^ ((1 >> 5) & 31))];
          float v73_data = ir1[0];
          ir1[0] = (v73_data + (v67_data * v71_data));
          float v79_data = s0[(3 ^ ((3 >> 5) & 31))];
          float v81_data = ir1[1];
          ir1[1] = (v81_data + (v67_data * v79_data));
          float v87_data = s0[(5 ^ ((5 >> 5) & 31))];
          float v89_data = ir1[2];
          ir1[2] = (v89_data + (v67_data * v87_data));
          float v107_data = r0[2];
          float v112_data = s0[(4 ^ ((4 >> 5) & 31))];
          float v114_data = ir1[1];
          ir1[1] = (v114_data + (v107_data * v112_data));
          float v120_data = s0[(6 ^ ((6 >> 5) & 31))];
          float v122_data = ir1[2];
          ir1[2] = (v122_data + (v107_data * v120_data));
          float v128_data = s0[(8 ^ ((8 >> 5) & 31))];
          float v130_data = ir1[3];
          ir1[3] = (v130_data + (v107_data * v128_data));
          float v147_data = r0[3];
          float v153_data = s0[(7 ^ ((7 >> 5) & 31))];
          float v155_data = ir1[2];
          ir1[2] = (v155_data + (v147_data * v153_data));
          float v161_data = s0[(9 ^ ((9 >> 5) & 31))];
          float v163_data = ir1[3];
          ir1[3] = (v163_data + (v147_data * v161_data));
          float v169_data = s0[(11 ^ ((11 >> 5) & 31))];
          float v171_data = ir1[4];
          ir1[4] = (v171_data + (v147_data * v169_data));
          float v187_data = r0[4];
          float v194_data = s0[(10 ^ ((10 >> 5) & 31))];
          float v196_data = ir1[3];
          ir1[3] = (v196_data + (v187_data * v194_data));
          float v202_data = s0[(12 ^ ((12 >> 5) & 31))];
          float v204_data = ir1[4];
          ir1[4] = (v204_data + (v187_data * v202_data));
          float v210_data = s0[(14 ^ ((14 >> 5) & 31))];
          float v212_data = ir1[5];
          ir1[5] = (v212_data + (v187_data * v210_data));
          float v227_data = r0[5];
          float v235_data = s0[(13 ^ ((13 >> 5) & 31))];
          float v237_data = ir1[4];
          ir1[4] = (v237_data + (v227_data * v235_data));
          float v243_data = s0[(15 ^ ((15 >> 5) & 31))];
          float v245_data = ir1[5];
          ir1[5] = (v245_data + (v227_data * v243_data));
          float v251_data = s0[(17 ^ ((17 >> 5) & 31))];
          float v253_data = ir1[6];
          ir1[6] = (v253_data + (v227_data * v251_data));
          float v267_data = r0[6];
          float v276_data = s0[(16 ^ ((16 >> 5) & 31))];
          float v278_data = ir1[5];
          ir1[5] = (v278_data + (v267_data * v276_data));
          float v284_data = s0[(18 ^ ((18 >> 5) & 31))];
          float v286_data = ir1[6];
          ir1[6] = (v286_data + (v267_data * v284_data));
          float v292_data = s0[(20 ^ ((20 >> 5) & 31))];
          float v294_data = ir1[7];
          ir1[7] = (v294_data + (v267_data * v292_data));
          float v307_data = r0[7];
          float v317_data = s0[(19 ^ ((19 >> 5) & 31))];
          float v319_data = ir1[6];
          ir1[6] = (v319_data + (v307_data * v317_data));
          float v325_data = s0[(21 ^ ((21 >> 5) & 31))];
          float v327_data = ir1[7];
          ir1[7] = (v327_data + (v307_data * v325_data));
          float v333_data = s0[(23 ^ ((23 >> 5) & 31))];
          float v335_data = ir1[8];
          ir1[8] = (v335_data + (v307_data * v333_data));
          float v347_data = r0[8];
          float v358_data = s0[(22 ^ ((22 >> 5) & 31))];
          float v360_data = ir1[7];
          ir1[7] = (v360_data + (v347_data * v358_data));
          float v366_data = s0[(24 ^ ((24 >> 5) & 31))];
          float v368_data = ir1[8];
          ir1[8] = (v368_data + (v347_data * v366_data));
          float v374_data = s0[(26 ^ ((26 >> 5) & 31))];
          float v376_data = ir1[9];
          ir1[9] = (v376_data + (v347_data * v374_data));
          float v387_data = r0[9];
          float v399_data = s0[(25 ^ ((25 >> 5) & 31))];
          float v401_data = ir1[8];
          ir1[8] = (v401_data + (v387_data * v399_data));
          float v407_data = s0[(27 ^ ((27 >> 5) & 31))];
          float v409_data = ir1[9];
          ir1[9] = (v409_data + (v387_data * v407_data));
          float v415_data = s0[(29 ^ ((29 >> 5) & 31))];
          float v417_data = ir1[10];
          ir1[10] = (v417_data + (v387_data * v415_data));
          float v427_data = r0[10];
          float v440_data = s0[(28 ^ ((28 >> 5) & 31))];
          float v442_data = ir1[9];
          ir1[9] = (v442_data + (v427_data * v440_data));
          float v448_data = s0[(30 ^ ((30 >> 5) & 31))];
          float v450_data = ir1[10];
          ir1[10] = (v450_data + (v427_data * v448_data));
          float v456_data = s0[(32 ^ ((32 >> 5) & 31))];
          float v458_data = ir1[11];
          ir1[11] = (v458_data + (v427_data * v456_data));
          float v467_data = r0[11];
          float v481_data = s0[(31 ^ ((31 >> 5) & 31))];
          float v483_data = ir1[10];
          ir1[10] = (v483_data + (v467_data * v481_data));
          float v489_data = s0[(33 ^ ((33 >> 5) & 31))];
          float v491_data = ir1[11];
          ir1[11] = (v491_data + (v467_data * v489_data));
          float v497_data = s0[(35 ^ ((35 >> 5) & 31))];
          float v499_data = ir1[12];
          ir1[12] = (v499_data + (v467_data * v497_data));
          float v507_data = r0[12];
          float v522_data = s0[(34 ^ ((34 >> 5) & 31))];
          float v524_data = ir1[11];
          ir1[11] = (v524_data + (v507_data * v522_data));
          float v530_data = s0[(36 ^ ((36 >> 5) & 31))];
          float v532_data = ir1[12];
          ir1[12] = (v532_data + (v507_data * v530_data));
          float v538_data = s0[(38 ^ ((38 >> 5) & 31))];
          float v540_data = ir1[13];
          ir1[13] = (v540_data + (v507_data * v538_data));
          float v547_data = r0[13];
          float v563_data = s0[(37 ^ ((37 >> 5) & 31))];
          float v565_data = ir1[12];
          ir1[12] = (v565_data + (v547_data * v563_data));
          float v571_data = s0[(39 ^ ((39 >> 5) & 31))];
          float v573_data = ir1[13];
          ir1[13] = (v573_data + (v547_data * v571_data));
          float v579_data = s0[(41 ^ ((41 >> 5) & 31))];
          float v581_data = ir1[14];
          ir1[14] = (v581_data + (v547_data * v579_data));
          float v587_data = r0[14];
          float v604_data = s0[(40 ^ ((40 >> 5) & 31))];
          float v606_data = ir1[13];
          ir1[13] = (v606_data + (v587_data * v604_data));
          float v612_data = s0[(42 ^ ((42 >> 5) & 31))];
          float v614_data = ir1[14];
          ir1[14] = (v614_data + (v587_data * v612_data));
          float v620_data = s0[(44 ^ ((44 >> 5) & 31))];
          float v622_data = ir1[15];
          ir1[15] = (v622_data + (v587_data * v620_data));
          float v627_data = r0[15];
          float v645_data = s0[(43 ^ ((43 >> 5) & 31))];
          float v647_data = ir1[14];
          ir1[14] = (v647_data + (v627_data * v645_data));
          float v653_data = s0[(45 ^ ((45 >> 5) & 31))];
          float v655_data = ir1[15];
          ir1[15] = (v655_data + (v627_data * v653_data));
          #pragma unroll
          for (int32_t v660_n0 = 0; v660_n0 < 1; ++v660_n0) {
            #pragma unroll
            for (int32_t v661_n1 = 0; v661_n1 < 16; ++v661_n1) {
              int32_t v662_a = v660_n0 + v661_n1;
              float v663_data = ir1[v662_a];
              r1[v662_a] = v663_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v668_i0 = 0; v668_i0 < 1; ++v668_i0) {
            int32_t v676_lead = v13_lead + (v668_i0 * 16);
            #pragma unroll
            for (int32_t v669_i1 = 0; v669_i1 < 16; ++v669_i1) {
              float v671_data = r1[(v668_i0 + v669_i1)];
              glb_m0[(v676_lead + (v669_i1 * 16))] = v671_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

