// === base name ===
kernel_417e1ddcc4

// === header ===
void launcher_kernel_417e1ddcc4(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_417e1ddcc4(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_417e1ddcc4, block.x * block.y * block.z, 1024 * sizeof(double));
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
        cudaFuncSetAttribute(kernel_kernel_417e1ddcc4, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024 * sizeof(double));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_417e1ddcc4<<<grid,block,1024 * sizeof(double),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_417e1ddcc4(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      double* tempShrMem = &localShrMem0[48];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v13_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v14_i0 = 0; v14_i0 < 1; ++v14_i0) {
            int32_t v20_lead = v13_lead + (v14_i0 * 16);
            #pragma unroll
            for (int32_t v15_i1 = 0; v15_i1 < 16; ++v15_i1) {
              double v23_data = __ldcg(&glb_m1[(v20_lead + (v15_i1 * 16))]);
              r0[(v14_i0 + v15_i1)] = v23_data;
            }
          }
          double* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 8);
          __pipeline_commit();
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m2[0 + 0 + 1 * threadIdx.x + 16], 8);
          __pipeline_commit();
          if (threadIdx.x < 14) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m2[0 + 0 + 1 * threadIdx.x + 32], 8);
            __pipeline_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          double r1[16]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          double ir1[16]{};
          double v34_data = r0[0];
          double v38_data = s0[(0 ^ ((0 >> 5) & 31))];
          double v40_data = ir1[0];
          ir1[0] = (v40_data + (v34_data * v38_data));
          double v46_data = s0[(2 ^ ((2 >> 5) & 31))];
          double v48_data = ir1[1];
          ir1[1] = (v48_data + (v34_data * v46_data));
          double v67_data = r0[1];
          double v71_data = s0[(1 ^ ((1 >> 5) & 31))];
          double v73_data = ir1[0];
          ir1[0] = (v73_data + (v67_data * v71_data));
          double v79_data = s0[(3 ^ ((3 >> 5) & 31))];
          double v81_data = ir1[1];
          ir1[1] = (v81_data + (v67_data * v79_data));
          double v87_data = s0[(5 ^ ((5 >> 5) & 31))];
          double v89_data = ir1[2];
          ir1[2] = (v89_data + (v67_data * v87_data));
          double v107_data = r0[2];
          double v112_data = s0[(4 ^ ((4 >> 5) & 31))];
          double v114_data = ir1[1];
          ir1[1] = (v114_data + (v107_data * v112_data));
          double v120_data = s0[(6 ^ ((6 >> 5) & 31))];
          double v122_data = ir1[2];
          ir1[2] = (v122_data + (v107_data * v120_data));
          double v128_data = s0[(8 ^ ((8 >> 5) & 31))];
          double v130_data = ir1[3];
          ir1[3] = (v130_data + (v107_data * v128_data));
          double v147_data = r0[3];
          double v153_data = s0[(7 ^ ((7 >> 5) & 31))];
          double v155_data = ir1[2];
          ir1[2] = (v155_data + (v147_data * v153_data));
          double v161_data = s0[(9 ^ ((9 >> 5) & 31))];
          double v163_data = ir1[3];
          ir1[3] = (v163_data + (v147_data * v161_data));
          double v169_data = s0[(11 ^ ((11 >> 5) & 31))];
          double v171_data = ir1[4];
          ir1[4] = (v171_data + (v147_data * v169_data));
          double v187_data = r0[4];
          double v194_data = s0[(10 ^ ((10 >> 5) & 31))];
          double v196_data = ir1[3];
          ir1[3] = (v196_data + (v187_data * v194_data));
          double v202_data = s0[(12 ^ ((12 >> 5) & 31))];
          double v204_data = ir1[4];
          ir1[4] = (v204_data + (v187_data * v202_data));
          double v210_data = s0[(14 ^ ((14 >> 5) & 31))];
          double v212_data = ir1[5];
          ir1[5] = (v212_data + (v187_data * v210_data));
          double v227_data = r0[5];
          double v235_data = s0[(13 ^ ((13 >> 5) & 31))];
          double v237_data = ir1[4];
          ir1[4] = (v237_data + (v227_data * v235_data));
          double v243_data = s0[(15 ^ ((15 >> 5) & 31))];
          double v245_data = ir1[5];
          ir1[5] = (v245_data + (v227_data * v243_data));
          double v251_data = s0[(17 ^ ((17 >> 5) & 31))];
          double v253_data = ir1[6];
          ir1[6] = (v253_data + (v227_data * v251_data));
          double v267_data = r0[6];
          double v276_data = s0[(16 ^ ((16 >> 5) & 31))];
          double v278_data = ir1[5];
          ir1[5] = (v278_data + (v267_data * v276_data));
          double v284_data = s0[(18 ^ ((18 >> 5) & 31))];
          double v286_data = ir1[6];
          ir1[6] = (v286_data + (v267_data * v284_data));
          double v292_data = s0[(20 ^ ((20 >> 5) & 31))];
          double v294_data = ir1[7];
          ir1[7] = (v294_data + (v267_data * v292_data));
          double v307_data = r0[7];
          double v317_data = s0[(19 ^ ((19 >> 5) & 31))];
          double v319_data = ir1[6];
          ir1[6] = (v319_data + (v307_data * v317_data));
          double v325_data = s0[(21 ^ ((21 >> 5) & 31))];
          double v327_data = ir1[7];
          ir1[7] = (v327_data + (v307_data * v325_data));
          double v333_data = s0[(23 ^ ((23 >> 5) & 31))];
          double v335_data = ir1[8];
          ir1[8] = (v335_data + (v307_data * v333_data));
          double v347_data = r0[8];
          double v358_data = s0[(22 ^ ((22 >> 5) & 31))];
          double v360_data = ir1[7];
          ir1[7] = (v360_data + (v347_data * v358_data));
          double v366_data = s0[(24 ^ ((24 >> 5) & 31))];
          double v368_data = ir1[8];
          ir1[8] = (v368_data + (v347_data * v366_data));
          double v374_data = s0[(26 ^ ((26 >> 5) & 31))];
          double v376_data = ir1[9];
          ir1[9] = (v376_data + (v347_data * v374_data));
          double v387_data = r0[9];
          double v399_data = s0[(25 ^ ((25 >> 5) & 31))];
          double v401_data = ir1[8];
          ir1[8] = (v401_data + (v387_data * v399_data));
          double v407_data = s0[(27 ^ ((27 >> 5) & 31))];
          double v409_data = ir1[9];
          ir1[9] = (v409_data + (v387_data * v407_data));
          double v415_data = s0[(29 ^ ((29 >> 5) & 31))];
          double v417_data = ir1[10];
          ir1[10] = (v417_data + (v387_data * v415_data));
          double v427_data = r0[10];
          double v440_data = s0[(28 ^ ((28 >> 5) & 31))];
          double v442_data = ir1[9];
          ir1[9] = (v442_data + (v427_data * v440_data));
          double v448_data = s0[(30 ^ ((30 >> 5) & 31))];
          double v450_data = ir1[10];
          ir1[10] = (v450_data + (v427_data * v448_data));
          double v456_data = s0[(32 ^ ((32 >> 5) & 31))];
          double v458_data = ir1[11];
          ir1[11] = (v458_data + (v427_data * v456_data));
          double v467_data = r0[11];
          double v481_data = s0[(31 ^ ((31 >> 5) & 31))];
          double v483_data = ir1[10];
          ir1[10] = (v483_data + (v467_data * v481_data));
          double v489_data = s0[(33 ^ ((33 >> 5) & 31))];
          double v491_data = ir1[11];
          ir1[11] = (v491_data + (v467_data * v489_data));
          double v497_data = s0[(35 ^ ((35 >> 5) & 31))];
          double v499_data = ir1[12];
          ir1[12] = (v499_data + (v467_data * v497_data));
          double v507_data = r0[12];
          double v522_data = s0[(34 ^ ((34 >> 5) & 31))];
          double v524_data = ir1[11];
          ir1[11] = (v524_data + (v507_data * v522_data));
          double v530_data = s0[(36 ^ ((36 >> 5) & 31))];
          double v532_data = ir1[12];
          ir1[12] = (v532_data + (v507_data * v530_data));
          double v538_data = s0[(38 ^ ((38 >> 5) & 31))];
          double v540_data = ir1[13];
          ir1[13] = (v540_data + (v507_data * v538_data));
          double v547_data = r0[13];
          double v563_data = s0[(37 ^ ((37 >> 5) & 31))];
          double v565_data = ir1[12];
          ir1[12] = (v565_data + (v547_data * v563_data));
          double v571_data = s0[(39 ^ ((39 >> 5) & 31))];
          double v573_data = ir1[13];
          ir1[13] = (v573_data + (v547_data * v571_data));
          double v579_data = s0[(41 ^ ((41 >> 5) & 31))];
          double v581_data = ir1[14];
          ir1[14] = (v581_data + (v547_data * v579_data));
          double v587_data = r0[14];
          double v604_data = s0[(40 ^ ((40 >> 5) & 31))];
          double v606_data = ir1[13];
          ir1[13] = (v606_data + (v587_data * v604_data));
          double v612_data = s0[(42 ^ ((42 >> 5) & 31))];
          double v614_data = ir1[14];
          ir1[14] = (v614_data + (v587_data * v612_data));
          double v620_data = s0[(44 ^ ((44 >> 5) & 31))];
          double v622_data = ir1[15];
          ir1[15] = (v622_data + (v587_data * v620_data));
          double v627_data = r0[15];
          double v645_data = s0[(43 ^ ((43 >> 5) & 31))];
          double v647_data = ir1[14];
          ir1[14] = (v647_data + (v627_data * v645_data));
          double v653_data = s0[(45 ^ ((45 >> 5) & 31))];
          double v655_data = ir1[15];
          ir1[15] = (v655_data + (v627_data * v653_data));
          #pragma unroll
          for (int32_t v660_n0 = 0; v660_n0 < 1; ++v660_n0) {
            #pragma unroll
            for (int32_t v661_n1 = 0; v661_n1 < 16; ++v661_n1) {
              int32_t v662_a = v660_n0 + v661_n1;
              double v663_data = ir1[v662_a];
              r1[v662_a] = v663_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v668_i0 = 0; v668_i0 < 1; ++v668_i0) {
            int32_t v676_lead = v13_lead + (v668_i0 * 16);
            #pragma unroll
            for (int32_t v669_i1 = 0; v669_i1 < 16; ++v669_i1) {
              double v671_data = r1[(v668_i0 + v669_i1)];
              glb_m0[(v676_lead + (v669_i1 * 16))] = v671_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

