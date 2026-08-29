// === base name ===
kernel_7ab185b978

// === header ===
void launcher_kernel_7ab185b978(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_7ab185b978(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_7ab185b978, block.x * block.y * block.z, 3072 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_7ab185b978, cudaFuncAttributeMaxDynamicSharedMemorySize, 3072 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_7ab185b978<<<grid,block,3072 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_7ab185b978(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 32×9(32×9) {0..32}×{0..9} pointer_based
    // m1 16×9(16×9) {0..16}×{0..9} pointer_based
    // m2 16×9(16×9) {0..16}×{0..9} pointer_based
    // m3 32×9(32×9) {0..32}×{0..9} pointer_based
    // m4 9×9(9×9) {0..9}×{0..9} pointer_based
    // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] = m0 32×9(32×9) {0..32}×{0..9} pointer_based({0..32}×{0..9})[0, 1]
    // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] += m1 16×9(16×9) {0..16}×{0..9} pointer_based({0..16}×{0..9})[0, 1]
    // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] += m2 16×9(16×9) {0..16}×{0..9} pointer_based({0..16}×{0..9})[0, 1]
    // m3 32×9(32×9) {0..32}×{0..9} pointer_based({0..32}×{0..9})[0, 1] = t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, -1]×m4 9×9(9×9) {0..9}×{0..9} pointer_based({0..9}×{0..9})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[384 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[384];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0][0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0][0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0][0 + m4_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v15_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v16_i0 = 0; v16_i0 < 1; ++v16_i0) {
            int32_t v22_lead = v15_lead + (v16_i0 * 32);
            #pragma unroll
            for (int32_t v17_i1 = 0; v17_i1 < 9; ++v17_i1) {
              float v25_data = __ldcg(&glb_m0[(v22_lead + (v17_i1 * 32))]);
              r0[(v16_i0 + v17_i1)] = v25_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          if (v15_lead < 16) {
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 9; ++v32_i1) {
              float v40_data = __ldcg(&glb_m1[(v15_lead + (v32_i1 * 16))]);
              r2[v32_i1] = v40_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          float v46_data = r0[0];
          float v47_data = r1[0];
          r1[0] = (v47_data + v46_data);
          float v49_data = r0[1];
          float v50_data = r1[1];
          r1[1] = (v50_data + v49_data);
          float v52_data = r0[2];
          float v53_data = r1[2];
          r1[2] = (v53_data + v52_data);
          float v55_data = r0[3];
          float v56_data = r1[3];
          r1[3] = (v56_data + v55_data);
          float v58_data = r0[4];
          float v59_data = r1[4];
          r1[4] = (v59_data + v58_data);
          float v61_data = r0[5];
          float v62_data = r1[5];
          r1[5] = (v62_data + v61_data);
          float v64_data = r0[6];
          float v65_data = r1[6];
          r1[6] = (v65_data + v64_data);
          float v67_data = r0[7];
          float v68_data = r1[7];
          r1[7] = (v68_data + v67_data);
          float v70_data = r0[8];
          float v71_data = r1[8];
          r1[8] = (v71_data + v70_data);
          float* __restrict__ s0 = &localShrMem0[96];
          // s0 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v77_i0 = 0; v77_i0 < 1; ++v77_i0) {
            int32_t v85_lead = v15_lead + (v77_i0 * 32);
            #pragma unroll
            for (int32_t v78_i1 = 0; v78_i1 < 9; ++v78_i1) {
              float v80_data = r1[(v77_i0 + v78_i1)];
              int32_t v87_a = v85_lead + (v78_i1 * 32);
              s0[(v87_a ^ ((v87_a >> 5) & 31))] = v80_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v15_lead < 16) {
            #pragma unroll
            for (int32_t v96_i1 = 0; v96_i1 < 9; ++v96_i1) {
              float v104_data = __ldcg(&glb_m2[(v15_lead + (v96_i1 * 16))]);
              r4[v96_i1] = v104_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          __syncwarp();
          // r3 = +(r2) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir3[9]{};
          if (v15_lead < 16) {
            float v112_data = r2[0];
            float v113_data = ir3[0];
            ir3[0] = (v113_data + v112_data);
            float v115_data = r2[1];
            float v116_data = ir3[1];
            ir3[1] = (v116_data + v115_data);
            float v118_data = r2[2];
            float v119_data = ir3[2];
            ir3[2] = (v119_data + v118_data);
            float v121_data = r2[3];
            float v122_data = ir3[3];
            ir3[3] = (v122_data + v121_data);
            float v124_data = r2[4];
            float v125_data = ir3[4];
            ir3[4] = (v125_data + v124_data);
            float v127_data = r2[5];
            float v128_data = ir3[5];
            ir3[5] = (v128_data + v127_data);
            float v130_data = r2[6];
            float v131_data = ir3[6];
            ir3[6] = (v131_data + v130_data);
            float v133_data = r2[7];
            float v134_data = ir3[7];
            ir3[7] = (v134_data + v133_data);
            float v136_data = r2[8];
            float v137_data = ir3[8];
            ir3[8] = (v137_data + v136_data);
          }
          if (v15_lead < 16) {
            #pragma unroll
            for (int32_t v143_n1 = 0; v143_n1 < 9; ++v143_n1) {
              float v145_data = ir3[v143_n1];
              int32_t v152_a = v15_lead + (v143_n1 * 32);
              float v156_data = s0[(v152_a ^ ((v152_a >> 5) & 31))];
              r3[v143_n1] = (v156_data + v145_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r3);
          if (v15_lead < 16) {
            #pragma unroll
            for (int32_t v163_i1 = 0; v163_i1 < 9; ++v163_i1) {
              float v165_data = r3[v163_i1];
              int32_t v172_a = v15_lead + (v163_i1 * 32);
              s0[(v172_a ^ ((v172_a >> 5) & 31))] = v165_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          __syncwarp();
          // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir5[9]{};
          if (v15_lead < 16) {
            float v182_data = r4[0];
            float v183_data = ir5[0];
            ir5[0] = (v183_data + v182_data);
            float v185_data = r4[1];
            float v186_data = ir5[1];
            ir5[1] = (v186_data + v185_data);
            float v188_data = r4[2];
            float v189_data = ir5[2];
            ir5[2] = (v189_data + v188_data);
            float v191_data = r4[3];
            float v192_data = ir5[3];
            ir5[3] = (v192_data + v191_data);
            float v194_data = r4[4];
            float v195_data = ir5[4];
            ir5[4] = (v195_data + v194_data);
            float v197_data = r4[5];
            float v198_data = ir5[5];
            ir5[5] = (v198_data + v197_data);
            float v200_data = r4[6];
            float v201_data = ir5[6];
            ir5[6] = (v201_data + v200_data);
            float v203_data = r4[7];
            float v204_data = ir5[7];
            ir5[7] = (v204_data + v203_data);
            float v206_data = r4[8];
            float v207_data = ir5[8];
            ir5[8] = (v207_data + v206_data);
          }
          if (v15_lead < 16) {
            #pragma unroll
            for (int32_t v213_n1 = 0; v213_n1 < 9; ++v213_n1) {
              float v215_data = ir5[v213_n1];
              int32_t v222_a = v15_lead + (v213_n1 * 32);
              float v226_data = s0[(v222_a ^ ((v222_a >> 5) & 31))];
              r5[v213_n1] = (v226_data + v215_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r5);
          if (v15_lead < 16) {
            #pragma unroll
            for (int32_t v233_i1 = 0; v233_i1 < 9; ++v233_i1) {
              float v235_data = r5[v233_i1];
              int32_t v242_a = v15_lead + (v233_i1 * 32);
              s0[(v242_a ^ ((v242_a >> 5) & 31))] = v235_data;
            }
          }
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = load{g>s}(glb_m4[0, 1])
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 0], &glb_m4[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 32], &glb_m4[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_commit();
          if (threadIdx.x < 17) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 64], &glb_m4[0 + 0 + 1 * threadIdx.x + 64], 4);
            __pipeline_commit();
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r6[9]{};
          __syncwarp();
          // r6 = +(s0 * s1) + None
          // [(0, 32), (0, 9)] [(0, 9)]
          float ir6[9]{};
          float v264_data = s0[(v15_lead ^ ((v15_lead >> 5) & 31))];
          float v265_data = s1[0];
          float v267_data = ir6[0];
          ir6[0] = (v267_data + (v264_data * v265_data));
          float v278_data = s0[(v15_lead ^ ((v15_lead >> 5) & 31))];
          float v279_data = s1[9];
          float v281_data = ir6[1];
          ir6[1] = (v281_data + (v278_data * v279_data));
          float v292_data = s0[(v15_lead ^ ((v15_lead >> 5) & 31))];
          float v293_data = s1[18];
          float v295_data = ir6[2];
          ir6[2] = (v295_data + (v292_data * v293_data));
          float v306_data = s0[(v15_lead ^ ((v15_lead >> 5) & 31))];
          float v307_data = s1[27];
          float v309_data = ir6[3];
          ir6[3] = (v309_data + (v306_data * v307_data));
          float v320_data = s0[(v15_lead ^ ((v15_lead >> 5) & 31))];
          float v321_data = s1[36];
          float v323_data = ir6[4];
          ir6[4] = (v323_data + (v320_data * v321_data));
          float v334_data = s0[(v15_lead ^ ((v15_lead >> 5) & 31))];
          float v335_data = s1[45];
          float v337_data = ir6[5];
          ir6[5] = (v337_data + (v334_data * v335_data));
          float v348_data = s0[(v15_lead ^ ((v15_lead >> 5) & 31))];
          float v349_data = s1[54];
          float v351_data = ir6[6];
          ir6[6] = (v351_data + (v348_data * v349_data));
          float v362_data = s0[(v15_lead ^ ((v15_lead >> 5) & 31))];
          float v363_data = s1[63];
          float v365_data = ir6[7];
          ir6[7] = (v365_data + (v362_data * v363_data));
          float v376_data = s0[(v15_lead ^ ((v15_lead >> 5) & 31))];
          float v377_data = s1[72];
          float v379_data = ir6[8];
          ir6[8] = (v379_data + (v376_data * v377_data));
          int32_t v389_a = v15_lead + 32;
          float v393_data = s0[(v389_a ^ ((v389_a >> 5) & 31))];
          float v394_data = s1[1];
          float v396_data = ir6[0];
          ir6[0] = (v396_data + (v393_data * v394_data));
          int32_t v403_a = v15_lead + 32;
          float v407_data = s0[(v403_a ^ ((v403_a >> 5) & 31))];
          float v408_data = s1[10];
          float v410_data = ir6[1];
          ir6[1] = (v410_data + (v407_data * v408_data));
          int32_t v417_a = v15_lead + 32;
          float v421_data = s0[(v417_a ^ ((v417_a >> 5) & 31))];
          float v422_data = s1[19];
          float v424_data = ir6[2];
          ir6[2] = (v424_data + (v421_data * v422_data));
          int32_t v431_a = v15_lead + 32;
          float v435_data = s0[(v431_a ^ ((v431_a >> 5) & 31))];
          float v436_data = s1[28];
          float v438_data = ir6[3];
          ir6[3] = (v438_data + (v435_data * v436_data));
          int32_t v445_a = v15_lead + 32;
          float v449_data = s0[(v445_a ^ ((v445_a >> 5) & 31))];
          float v450_data = s1[37];
          float v452_data = ir6[4];
          ir6[4] = (v452_data + (v449_data * v450_data));
          int32_t v459_a = v15_lead + 32;
          float v463_data = s0[(v459_a ^ ((v459_a >> 5) & 31))];
          float v464_data = s1[46];
          float v466_data = ir6[5];
          ir6[5] = (v466_data + (v463_data * v464_data));
          int32_t v473_a = v15_lead + 32;
          float v477_data = s0[(v473_a ^ ((v473_a >> 5) & 31))];
          float v478_data = s1[55];
          float v480_data = ir6[6];
          ir6[6] = (v480_data + (v477_data * v478_data));
          int32_t v487_a = v15_lead + 32;
          float v491_data = s0[(v487_a ^ ((v487_a >> 5) & 31))];
          float v492_data = s1[64];
          float v494_data = ir6[7];
          ir6[7] = (v494_data + (v491_data * v492_data));
          int32_t v501_a = v15_lead + 32;
          float v505_data = s0[(v501_a ^ ((v501_a >> 5) & 31))];
          float v506_data = s1[73];
          float v508_data = ir6[8];
          ir6[8] = (v508_data + (v505_data * v506_data));
          int32_t v518_a = v15_lead + 64;
          float v522_data = s0[(v518_a ^ ((v518_a >> 5) & 31))];
          float v523_data = s1[2];
          float v525_data = ir6[0];
          ir6[0] = (v525_data + (v522_data * v523_data));
          int32_t v532_a = v15_lead + 64;
          float v536_data = s0[(v532_a ^ ((v532_a >> 5) & 31))];
          float v537_data = s1[11];
          float v539_data = ir6[1];
          ir6[1] = (v539_data + (v536_data * v537_data));
          int32_t v546_a = v15_lead + 64;
          float v550_data = s0[(v546_a ^ ((v546_a >> 5) & 31))];
          float v551_data = s1[20];
          float v553_data = ir6[2];
          ir6[2] = (v553_data + (v550_data * v551_data));
          int32_t v560_a = v15_lead + 64;
          float v564_data = s0[(v560_a ^ ((v560_a >> 5) & 31))];
          float v565_data = s1[29];
          float v567_data = ir6[3];
          ir6[3] = (v567_data + (v564_data * v565_data));
          int32_t v574_a = v15_lead + 64;
          float v578_data = s0[(v574_a ^ ((v574_a >> 5) & 31))];
          float v579_data = s1[38];
          float v581_data = ir6[4];
          ir6[4] = (v581_data + (v578_data * v579_data));
          int32_t v588_a = v15_lead + 64;
          float v592_data = s0[(v588_a ^ ((v588_a >> 5) & 31))];
          float v593_data = s1[47];
          float v595_data = ir6[5];
          ir6[5] = (v595_data + (v592_data * v593_data));
          int32_t v602_a = v15_lead + 64;
          float v606_data = s0[(v602_a ^ ((v602_a >> 5) & 31))];
          float v607_data = s1[56];
          float v609_data = ir6[6];
          ir6[6] = (v609_data + (v606_data * v607_data));
          int32_t v616_a = v15_lead + 64;
          float v620_data = s0[(v616_a ^ ((v616_a >> 5) & 31))];
          float v621_data = s1[65];
          float v623_data = ir6[7];
          ir6[7] = (v623_data + (v620_data * v621_data));
          int32_t v630_a = v15_lead + 64;
          float v634_data = s0[(v630_a ^ ((v630_a >> 5) & 31))];
          float v635_data = s1[74];
          float v637_data = ir6[8];
          ir6[8] = (v637_data + (v634_data * v635_data));
          int32_t v647_a = v15_lead + 96;
          float v651_data = s0[(v647_a ^ ((v647_a >> 5) & 31))];
          float v652_data = s1[3];
          float v654_data = ir6[0];
          ir6[0] = (v654_data + (v651_data * v652_data));
          int32_t v661_a = v15_lead + 96;
          float v665_data = s0[(v661_a ^ ((v661_a >> 5) & 31))];
          float v666_data = s1[12];
          float v668_data = ir6[1];
          ir6[1] = (v668_data + (v665_data * v666_data));
          int32_t v675_a = v15_lead + 96;
          float v679_data = s0[(v675_a ^ ((v675_a >> 5) & 31))];
          float v680_data = s1[21];
          float v682_data = ir6[2];
          ir6[2] = (v682_data + (v679_data * v680_data));
          int32_t v689_a = v15_lead + 96;
          float v693_data = s0[(v689_a ^ ((v689_a >> 5) & 31))];
          float v694_data = s1[30];
          float v696_data = ir6[3];
          ir6[3] = (v696_data + (v693_data * v694_data));
          int32_t v703_a = v15_lead + 96;
          float v707_data = s0[(v703_a ^ ((v703_a >> 5) & 31))];
          float v708_data = s1[39];
          float v710_data = ir6[4];
          ir6[4] = (v710_data + (v707_data * v708_data));
          int32_t v717_a = v15_lead + 96;
          float v721_data = s0[(v717_a ^ ((v717_a >> 5) & 31))];
          float v722_data = s1[48];
          float v724_data = ir6[5];
          ir6[5] = (v724_data + (v721_data * v722_data));
          int32_t v731_a = v15_lead + 96;
          float v735_data = s0[(v731_a ^ ((v731_a >> 5) & 31))];
          float v736_data = s1[57];
          float v738_data = ir6[6];
          ir6[6] = (v738_data + (v735_data * v736_data));
          int32_t v745_a = v15_lead + 96;
          float v749_data = s0[(v745_a ^ ((v745_a >> 5) & 31))];
          float v750_data = s1[66];
          float v752_data = ir6[7];
          ir6[7] = (v752_data + (v749_data * v750_data));
          int32_t v759_a = v15_lead + 96;
          float v763_data = s0[(v759_a ^ ((v759_a >> 5) & 31))];
          float v764_data = s1[75];
          float v766_data = ir6[8];
          ir6[8] = (v766_data + (v763_data * v764_data));
          int32_t v776_a = v15_lead + 128;
          float v780_data = s0[(v776_a ^ ((v776_a >> 5) & 31))];
          float v781_data = s1[4];
          float v783_data = ir6[0];
          ir6[0] = (v783_data + (v780_data * v781_data));
          int32_t v790_a = v15_lead + 128;
          float v794_data = s0[(v790_a ^ ((v790_a >> 5) & 31))];
          float v795_data = s1[13];
          float v797_data = ir6[1];
          ir6[1] = (v797_data + (v794_data * v795_data));
          int32_t v804_a = v15_lead + 128;
          float v808_data = s0[(v804_a ^ ((v804_a >> 5) & 31))];
          float v809_data = s1[22];
          float v811_data = ir6[2];
          ir6[2] = (v811_data + (v808_data * v809_data));
          int32_t v818_a = v15_lead + 128;
          float v822_data = s0[(v818_a ^ ((v818_a >> 5) & 31))];
          float v823_data = s1[31];
          float v825_data = ir6[3];
          ir6[3] = (v825_data + (v822_data * v823_data));
          int32_t v832_a = v15_lead + 128;
          float v836_data = s0[(v832_a ^ ((v832_a >> 5) & 31))];
          float v837_data = s1[40];
          float v839_data = ir6[4];
          ir6[4] = (v839_data + (v836_data * v837_data));
          int32_t v846_a = v15_lead + 128;
          float v850_data = s0[(v846_a ^ ((v846_a >> 5) & 31))];
          float v851_data = s1[49];
          float v853_data = ir6[5];
          ir6[5] = (v853_data + (v850_data * v851_data));
          int32_t v860_a = v15_lead + 128;
          float v864_data = s0[(v860_a ^ ((v860_a >> 5) & 31))];
          float v865_data = s1[58];
          float v867_data = ir6[6];
          ir6[6] = (v867_data + (v864_data * v865_data));
          int32_t v874_a = v15_lead + 128;
          float v878_data = s0[(v874_a ^ ((v874_a >> 5) & 31))];
          float v879_data = s1[67];
          float v881_data = ir6[7];
          ir6[7] = (v881_data + (v878_data * v879_data));
          int32_t v888_a = v15_lead + 128;
          float v892_data = s0[(v888_a ^ ((v888_a >> 5) & 31))];
          float v893_data = s1[76];
          float v895_data = ir6[8];
          ir6[8] = (v895_data + (v892_data * v893_data));
          int32_t v905_a = v15_lead + 160;
          float v909_data = s0[(v905_a ^ ((v905_a >> 5) & 31))];
          float v910_data = s1[5];
          float v912_data = ir6[0];
          ir6[0] = (v912_data + (v909_data * v910_data));
          int32_t v919_a = v15_lead + 160;
          float v923_data = s0[(v919_a ^ ((v919_a >> 5) & 31))];
          float v924_data = s1[14];
          float v926_data = ir6[1];
          ir6[1] = (v926_data + (v923_data * v924_data));
          int32_t v933_a = v15_lead + 160;
          float v937_data = s0[(v933_a ^ ((v933_a >> 5) & 31))];
          float v938_data = s1[23];
          float v940_data = ir6[2];
          ir6[2] = (v940_data + (v937_data * v938_data));
          int32_t v947_a = v15_lead + 160;
          float v951_data = s0[(v947_a ^ ((v947_a >> 5) & 31))];
          float v952_data = s1[32];
          float v954_data = ir6[3];
          ir6[3] = (v954_data + (v951_data * v952_data));
          int32_t v961_a = v15_lead + 160;
          float v965_data = s0[(v961_a ^ ((v961_a >> 5) & 31))];
          float v966_data = s1[41];
          float v968_data = ir6[4];
          ir6[4] = (v968_data + (v965_data * v966_data));
          int32_t v975_a = v15_lead + 160;
          float v979_data = s0[(v975_a ^ ((v975_a >> 5) & 31))];
          float v980_data = s1[50];
          float v982_data = ir6[5];
          ir6[5] = (v982_data + (v979_data * v980_data));
          int32_t v989_a = v15_lead + 160;
          float v993_data = s0[(v989_a ^ ((v989_a >> 5) & 31))];
          float v994_data = s1[59];
          float v996_data = ir6[6];
          ir6[6] = (v996_data + (v993_data * v994_data));
          int32_t v1003_a = v15_lead + 160;
          float v1007_data = s0[(v1003_a ^ ((v1003_a >> 5) & 31))];
          float v1008_data = s1[68];
          float v1010_data = ir6[7];
          ir6[7] = (v1010_data + (v1007_data * v1008_data));
          int32_t v1017_a = v15_lead + 160;
          float v1021_data = s0[(v1017_a ^ ((v1017_a >> 5) & 31))];
          float v1022_data = s1[77];
          float v1024_data = ir6[8];
          ir6[8] = (v1024_data + (v1021_data * v1022_data));
          int32_t v1034_a = v15_lead + 192;
          float v1038_data = s0[(v1034_a ^ ((v1034_a >> 5) & 31))];
          float v1039_data = s1[6];
          float v1041_data = ir6[0];
          ir6[0] = (v1041_data + (v1038_data * v1039_data));
          int32_t v1048_a = v15_lead + 192;
          float v1052_data = s0[(v1048_a ^ ((v1048_a >> 5) & 31))];
          float v1053_data = s1[15];
          float v1055_data = ir6[1];
          ir6[1] = (v1055_data + (v1052_data * v1053_data));
          int32_t v1062_a = v15_lead + 192;
          float v1066_data = s0[(v1062_a ^ ((v1062_a >> 5) & 31))];
          float v1067_data = s1[24];
          float v1069_data = ir6[2];
          ir6[2] = (v1069_data + (v1066_data * v1067_data));
          int32_t v1076_a = v15_lead + 192;
          float v1080_data = s0[(v1076_a ^ ((v1076_a >> 5) & 31))];
          float v1081_data = s1[33];
          float v1083_data = ir6[3];
          ir6[3] = (v1083_data + (v1080_data * v1081_data));
          int32_t v1090_a = v15_lead + 192;
          float v1094_data = s0[(v1090_a ^ ((v1090_a >> 5) & 31))];
          float v1095_data = s1[42];
          float v1097_data = ir6[4];
          ir6[4] = (v1097_data + (v1094_data * v1095_data));
          int32_t v1104_a = v15_lead + 192;
          float v1108_data = s0[(v1104_a ^ ((v1104_a >> 5) & 31))];
          float v1109_data = s1[51];
          float v1111_data = ir6[5];
          ir6[5] = (v1111_data + (v1108_data * v1109_data));
          int32_t v1118_a = v15_lead + 192;
          float v1122_data = s0[(v1118_a ^ ((v1118_a >> 5) & 31))];
          float v1123_data = s1[60];
          float v1125_data = ir6[6];
          ir6[6] = (v1125_data + (v1122_data * v1123_data));
          int32_t v1132_a = v15_lead + 192;
          float v1136_data = s0[(v1132_a ^ ((v1132_a >> 5) & 31))];
          float v1137_data = s1[69];
          float v1139_data = ir6[7];
          ir6[7] = (v1139_data + (v1136_data * v1137_data));
          int32_t v1146_a = v15_lead + 192;
          float v1150_data = s0[(v1146_a ^ ((v1146_a >> 5) & 31))];
          float v1151_data = s1[78];
          float v1153_data = ir6[8];
          ir6[8] = (v1153_data + (v1150_data * v1151_data));
          int32_t v1163_a = v15_lead + 224;
          float v1167_data = s0[(v1163_a ^ ((v1163_a >> 5) & 31))];
          float v1168_data = s1[7];
          float v1170_data = ir6[0];
          ir6[0] = (v1170_data + (v1167_data * v1168_data));
          int32_t v1177_a = v15_lead + 224;
          float v1181_data = s0[(v1177_a ^ ((v1177_a >> 5) & 31))];
          float v1182_data = s1[16];
          float v1184_data = ir6[1];
          ir6[1] = (v1184_data + (v1181_data * v1182_data));
          int32_t v1191_a = v15_lead + 224;
          float v1195_data = s0[(v1191_a ^ ((v1191_a >> 5) & 31))];
          float v1196_data = s1[25];
          float v1198_data = ir6[2];
          ir6[2] = (v1198_data + (v1195_data * v1196_data));
          int32_t v1205_a = v15_lead + 224;
          float v1209_data = s0[(v1205_a ^ ((v1205_a >> 5) & 31))];
          float v1210_data = s1[34];
          float v1212_data = ir6[3];
          ir6[3] = (v1212_data + (v1209_data * v1210_data));
          int32_t v1219_a = v15_lead + 224;
          float v1223_data = s0[(v1219_a ^ ((v1219_a >> 5) & 31))];
          float v1224_data = s1[43];
          float v1226_data = ir6[4];
          ir6[4] = (v1226_data + (v1223_data * v1224_data));
          int32_t v1233_a = v15_lead + 224;
          float v1237_data = s0[(v1233_a ^ ((v1233_a >> 5) & 31))];
          float v1238_data = s1[52];
          float v1240_data = ir6[5];
          ir6[5] = (v1240_data + (v1237_data * v1238_data));
          int32_t v1247_a = v15_lead + 224;
          float v1251_data = s0[(v1247_a ^ ((v1247_a >> 5) & 31))];
          float v1252_data = s1[61];
          float v1254_data = ir6[6];
          ir6[6] = (v1254_data + (v1251_data * v1252_data));
          int32_t v1261_a = v15_lead + 224;
          float v1265_data = s0[(v1261_a ^ ((v1261_a >> 5) & 31))];
          float v1266_data = s1[70];
          float v1268_data = ir6[7];
          ir6[7] = (v1268_data + (v1265_data * v1266_data));
          int32_t v1275_a = v15_lead + 224;
          float v1279_data = s0[(v1275_a ^ ((v1275_a >> 5) & 31))];
          float v1280_data = s1[79];
          float v1282_data = ir6[8];
          ir6[8] = (v1282_data + (v1279_data * v1280_data));
          int32_t v1292_a = v15_lead + 256;
          float v1296_data = s0[(v1292_a ^ ((v1292_a >> 5) & 31))];
          float v1297_data = s1[8];
          float v1299_data = ir6[0];
          ir6[0] = (v1299_data + (v1296_data * v1297_data));
          int32_t v1306_a = v15_lead + 256;
          float v1310_data = s0[(v1306_a ^ ((v1306_a >> 5) & 31))];
          float v1311_data = s1[17];
          float v1313_data = ir6[1];
          ir6[1] = (v1313_data + (v1310_data * v1311_data));
          int32_t v1320_a = v15_lead + 256;
          float v1324_data = s0[(v1320_a ^ ((v1320_a >> 5) & 31))];
          float v1325_data = s1[26];
          float v1327_data = ir6[2];
          ir6[2] = (v1327_data + (v1324_data * v1325_data));
          int32_t v1334_a = v15_lead + 256;
          float v1338_data = s0[(v1334_a ^ ((v1334_a >> 5) & 31))];
          float v1339_data = s1[35];
          float v1341_data = ir6[3];
          ir6[3] = (v1341_data + (v1338_data * v1339_data));
          int32_t v1348_a = v15_lead + 256;
          float v1352_data = s0[(v1348_a ^ ((v1348_a >> 5) & 31))];
          float v1353_data = s1[44];
          float v1355_data = ir6[4];
          ir6[4] = (v1355_data + (v1352_data * v1353_data));
          int32_t v1362_a = v15_lead + 256;
          float v1366_data = s0[(v1362_a ^ ((v1362_a >> 5) & 31))];
          float v1367_data = s1[53];
          float v1369_data = ir6[5];
          ir6[5] = (v1369_data + (v1366_data * v1367_data));
          int32_t v1376_a = v15_lead + 256;
          float v1380_data = s0[(v1376_a ^ ((v1376_a >> 5) & 31))];
          float v1381_data = s1[62];
          float v1383_data = ir6[6];
          ir6[6] = (v1383_data + (v1380_data * v1381_data));
          int32_t v1390_a = v15_lead + 256;
          float v1394_data = s0[(v1390_a ^ ((v1390_a >> 5) & 31))];
          float v1395_data = s1[71];
          float v1397_data = ir6[7];
          ir6[7] = (v1397_data + (v1394_data * v1395_data));
          int32_t v1404_a = v15_lead + 256;
          float v1408_data = s0[(v1404_a ^ ((v1404_a >> 5) & 31))];
          float v1409_data = s1[80];
          float v1411_data = ir6[8];
          ir6[8] = (v1411_data + (v1408_data * v1409_data));
          #pragma unroll
          for (int32_t v1416_n0 = 0; v1416_n0 < 1; ++v1416_n0) {
            #pragma unroll
            for (int32_t v1417_n1 = 0; v1417_n1 < 9; ++v1417_n1) {
              int32_t v1418_a = v1416_n0 + v1417_n1;
              float v1419_data = ir6[v1418_a];
              r6[v1418_a] = v1419_data;
            }
          }
          // glb_m3 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v1424_i0 = 0; v1424_i0 < 1; ++v1424_i0) {
            int32_t v1432_lead = v15_lead + (v1424_i0 * 32);
            #pragma unroll
            for (int32_t v1425_i1 = 0; v1425_i1 < 9; ++v1425_i1) {
              float v1427_data = r6[(v1424_i0 + v1425_i1)];
              glb_m3[(v1432_lead + (v1425_i1 * 32))] = v1427_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

