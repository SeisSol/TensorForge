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
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0][0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0][0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0][0 + m4_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v9_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v10_i0 = 0; v10_i0 < 1; ++v10_i0) {
            int32_t v15_lead = v10_i0 * 32;
            int32_t v16_lead = v9_lead + v15_lead;
            int32_t v23_lead = v9_lead + v15_lead;
            #pragma unroll
            for (int32_t v11_i1 = 0; v11_i1 < 9; ++v11_i1) {
              int32_t v17_a = v11_i1 * 32;
              int32_t v18_a = v16_lead + v17_a;
              float v26_data = __ldcg(&glb_m0[(v23_lead + v17_a)]);
              r0[(v10_i0 + v11_i1)] = v26_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          if (v9_lead < 16) {
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 9; ++v33_i1) {
              int32_t v39_a = v33_i1 * 16;
              int32_t v40_a = v9_lead + v39_a;
              float v48_data = __ldcg(&glb_m1[(v9_lead + v39_a)]);
              r2[v33_i1] = v48_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          float v54_data = r0[0];
          float v55_data = r1[0];
          r1[0] = (v55_data + v54_data);
          float v57_data = r0[1];
          float v58_data = r1[1];
          r1[1] = (v58_data + v57_data);
          float v60_data = r0[2];
          float v61_data = r1[2];
          r1[2] = (v61_data + v60_data);
          float v63_data = r0[3];
          float v64_data = r1[3];
          r1[3] = (v64_data + v63_data);
          float v66_data = r0[4];
          float v67_data = r1[4];
          r1[4] = (v67_data + v66_data);
          float v69_data = r0[5];
          float v70_data = r1[5];
          r1[5] = (v70_data + v69_data);
          float v72_data = r0[6];
          float v73_data = r1[6];
          r1[6] = (v73_data + v72_data);
          float v75_data = r0[7];
          float v76_data = r1[7];
          r1[7] = (v76_data + v75_data);
          float v78_data = r0[8];
          float v79_data = r1[8];
          r1[8] = (v79_data + v78_data);
          float* __restrict__ s0 = &localShrMem0[96];
          // s0 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v85_i0 = 0; v85_i0 < 1; ++v85_i0) {
            int32_t v94_lead = v9_lead + (v85_i0 * 32);
            #pragma unroll
            for (int32_t v86_i1 = 0; v86_i1 < 9; ++v86_i1) {
              int32_t v87_a = v85_i0 + v86_i1;
              float v89_data = r1[(v85_i0 + v86_i1)];
              s0[(v94_lead + (v86_i1 * 32))] = v89_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v9_lead < 16) {
            #pragma unroll
            for (int32_t v102_i1 = 0; v102_i1 < 9; ++v102_i1) {
              int32_t v108_a = v102_i1 * 16;
              int32_t v109_a = v9_lead + v108_a;
              float v117_data = __ldcg(&glb_m2[(v9_lead + v108_a)]);
              r4[v102_i1] = v117_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          __syncwarp();
          // r3 = +(r2) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir3[9]{};
          if (v9_lead < 16) {
            float v125_data = r2[0];
            float v126_data = ir3[0];
            ir3[0] = (v126_data + v125_data);
            float v128_data = r2[1];
            float v129_data = ir3[1];
            ir3[1] = (v129_data + v128_data);
            float v131_data = r2[2];
            float v132_data = ir3[2];
            ir3[2] = (v132_data + v131_data);
            float v134_data = r2[3];
            float v135_data = ir3[3];
            ir3[3] = (v135_data + v134_data);
            float v137_data = r2[4];
            float v138_data = ir3[4];
            ir3[4] = (v138_data + v137_data);
            float v140_data = r2[5];
            float v141_data = ir3[5];
            ir3[5] = (v141_data + v140_data);
            float v143_data = r2[6];
            float v144_data = ir3[6];
            ir3[6] = (v144_data + v143_data);
            float v146_data = r2[7];
            float v147_data = ir3[7];
            ir3[7] = (v147_data + v146_data);
            float v149_data = r2[8];
            float v150_data = ir3[8];
            ir3[8] = (v150_data + v149_data);
          }
          if (v9_lead < 16) {
            #pragma unroll
            for (int32_t v156_n1 = 0; v156_n1 < 9; ++v156_n1) {
              int32_t v157_a = 0 + v156_n1;
              float v159_data = ir3[v156_n1];
              int32_t v165_a = v156_n1 * 32;
              int32_t v166_a = v9_lead + v165_a;
              float v174_data = s0[(v9_lead + v165_a)];
              r3[v156_n1] = (v174_data + v159_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r3);
          if (v9_lead < 16) {
            #pragma unroll
            for (int32_t v181_i1 = 0; v181_i1 < 9; ++v181_i1) {
              int32_t v182_a = 0 + v181_i1;
              float v184_data = r3[v181_i1];
              s0[(v9_lead + (v181_i1 * 32))] = v184_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          __syncwarp();
          // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir5[9]{};
          if (v9_lead < 16) {
            float v198_data = r4[0];
            float v199_data = ir5[0];
            ir5[0] = (v199_data + v198_data);
            float v201_data = r4[1];
            float v202_data = ir5[1];
            ir5[1] = (v202_data + v201_data);
            float v204_data = r4[2];
            float v205_data = ir5[2];
            ir5[2] = (v205_data + v204_data);
            float v207_data = r4[3];
            float v208_data = ir5[3];
            ir5[3] = (v208_data + v207_data);
            float v210_data = r4[4];
            float v211_data = ir5[4];
            ir5[4] = (v211_data + v210_data);
            float v213_data = r4[5];
            float v214_data = ir5[5];
            ir5[5] = (v214_data + v213_data);
            float v216_data = r4[6];
            float v217_data = ir5[6];
            ir5[6] = (v217_data + v216_data);
            float v219_data = r4[7];
            float v220_data = ir5[7];
            ir5[7] = (v220_data + v219_data);
            float v222_data = r4[8];
            float v223_data = ir5[8];
            ir5[8] = (v223_data + v222_data);
          }
          if (v9_lead < 16) {
            #pragma unroll
            for (int32_t v229_n1 = 0; v229_n1 < 9; ++v229_n1) {
              int32_t v230_a = 0 + v229_n1;
              float v232_data = ir5[v229_n1];
              int32_t v238_a = v229_n1 * 32;
              int32_t v239_a = v9_lead + v238_a;
              float v247_data = s0[(v9_lead + v238_a)];
              r5[v229_n1] = (v247_data + v232_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r5);
          if (v9_lead < 16) {
            #pragma unroll
            for (int32_t v254_i1 = 0; v254_i1 < 9; ++v254_i1) {
              int32_t v255_a = 0 + v254_i1;
              float v257_data = r5[v254_i1];
              s0[(v9_lead + (v254_i1 * 32))] = v257_data;
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
          int32_t v279_a = v9_lead + 0;
          float v286_data = s0[v9_lead];
          float v287_data = s1[0];
          float v289_data = ir6[0];
          ir6[0] = (v289_data + (v286_data * v287_data));
          int32_t v296_a = v9_lead + 0;
          float v303_data = s0[v9_lead];
          float v304_data = s1[9];
          float v306_data = ir6[1];
          ir6[1] = (v306_data + (v303_data * v304_data));
          int32_t v313_a = v9_lead + 0;
          float v320_data = s0[v9_lead];
          float v321_data = s1[18];
          float v323_data = ir6[2];
          ir6[2] = (v323_data + (v320_data * v321_data));
          int32_t v330_a = v9_lead + 0;
          float v337_data = s0[v9_lead];
          float v338_data = s1[27];
          float v340_data = ir6[3];
          ir6[3] = (v340_data + (v337_data * v338_data));
          int32_t v347_a = v9_lead + 0;
          float v354_data = s0[v9_lead];
          float v355_data = s1[36];
          float v357_data = ir6[4];
          ir6[4] = (v357_data + (v354_data * v355_data));
          int32_t v364_a = v9_lead + 0;
          float v371_data = s0[v9_lead];
          float v372_data = s1[45];
          float v374_data = ir6[5];
          ir6[5] = (v374_data + (v371_data * v372_data));
          int32_t v381_a = v9_lead + 0;
          float v388_data = s0[v9_lead];
          float v389_data = s1[54];
          float v391_data = ir6[6];
          ir6[6] = (v391_data + (v388_data * v389_data));
          int32_t v398_a = v9_lead + 0;
          float v405_data = s0[v9_lead];
          float v406_data = s1[63];
          float v408_data = ir6[7];
          ir6[7] = (v408_data + (v405_data * v406_data));
          int32_t v415_a = v9_lead + 0;
          float v422_data = s0[v9_lead];
          float v423_data = s1[72];
          float v425_data = ir6[8];
          ir6[8] = (v425_data + (v422_data * v423_data));
          int32_t v435_a = v9_lead + 32;
          float v442_data = s0[(v9_lead + 32)];
          float v443_data = s1[1];
          float v445_data = ir6[0];
          ir6[0] = (v445_data + (v442_data * v443_data));
          int32_t v452_a = v9_lead + 32;
          float v459_data = s0[(v9_lead + 32)];
          float v460_data = s1[10];
          float v462_data = ir6[1];
          ir6[1] = (v462_data + (v459_data * v460_data));
          int32_t v469_a = v9_lead + 32;
          float v476_data = s0[(v9_lead + 32)];
          float v477_data = s1[19];
          float v479_data = ir6[2];
          ir6[2] = (v479_data + (v476_data * v477_data));
          int32_t v486_a = v9_lead + 32;
          float v493_data = s0[(v9_lead + 32)];
          float v494_data = s1[28];
          float v496_data = ir6[3];
          ir6[3] = (v496_data + (v493_data * v494_data));
          int32_t v503_a = v9_lead + 32;
          float v510_data = s0[(v9_lead + 32)];
          float v511_data = s1[37];
          float v513_data = ir6[4];
          ir6[4] = (v513_data + (v510_data * v511_data));
          int32_t v520_a = v9_lead + 32;
          float v527_data = s0[(v9_lead + 32)];
          float v528_data = s1[46];
          float v530_data = ir6[5];
          ir6[5] = (v530_data + (v527_data * v528_data));
          int32_t v537_a = v9_lead + 32;
          float v544_data = s0[(v9_lead + 32)];
          float v545_data = s1[55];
          float v547_data = ir6[6];
          ir6[6] = (v547_data + (v544_data * v545_data));
          int32_t v554_a = v9_lead + 32;
          float v561_data = s0[(v9_lead + 32)];
          float v562_data = s1[64];
          float v564_data = ir6[7];
          ir6[7] = (v564_data + (v561_data * v562_data));
          int32_t v571_a = v9_lead + 32;
          float v578_data = s0[(v9_lead + 32)];
          float v579_data = s1[73];
          float v581_data = ir6[8];
          ir6[8] = (v581_data + (v578_data * v579_data));
          int32_t v591_a = v9_lead + 64;
          float v598_data = s0[(v9_lead + 64)];
          float v599_data = s1[2];
          float v601_data = ir6[0];
          ir6[0] = (v601_data + (v598_data * v599_data));
          int32_t v608_a = v9_lead + 64;
          float v615_data = s0[(v9_lead + 64)];
          float v616_data = s1[11];
          float v618_data = ir6[1];
          ir6[1] = (v618_data + (v615_data * v616_data));
          int32_t v625_a = v9_lead + 64;
          float v632_data = s0[(v9_lead + 64)];
          float v633_data = s1[20];
          float v635_data = ir6[2];
          ir6[2] = (v635_data + (v632_data * v633_data));
          int32_t v642_a = v9_lead + 64;
          float v649_data = s0[(v9_lead + 64)];
          float v650_data = s1[29];
          float v652_data = ir6[3];
          ir6[3] = (v652_data + (v649_data * v650_data));
          int32_t v659_a = v9_lead + 64;
          float v666_data = s0[(v9_lead + 64)];
          float v667_data = s1[38];
          float v669_data = ir6[4];
          ir6[4] = (v669_data + (v666_data * v667_data));
          int32_t v676_a = v9_lead + 64;
          float v683_data = s0[(v9_lead + 64)];
          float v684_data = s1[47];
          float v686_data = ir6[5];
          ir6[5] = (v686_data + (v683_data * v684_data));
          int32_t v693_a = v9_lead + 64;
          float v700_data = s0[(v9_lead + 64)];
          float v701_data = s1[56];
          float v703_data = ir6[6];
          ir6[6] = (v703_data + (v700_data * v701_data));
          int32_t v710_a = v9_lead + 64;
          float v717_data = s0[(v9_lead + 64)];
          float v718_data = s1[65];
          float v720_data = ir6[7];
          ir6[7] = (v720_data + (v717_data * v718_data));
          int32_t v727_a = v9_lead + 64;
          float v734_data = s0[(v9_lead + 64)];
          float v735_data = s1[74];
          float v737_data = ir6[8];
          ir6[8] = (v737_data + (v734_data * v735_data));
          int32_t v747_a = v9_lead + 96;
          float v754_data = s0[(v9_lead + 96)];
          float v755_data = s1[3];
          float v757_data = ir6[0];
          ir6[0] = (v757_data + (v754_data * v755_data));
          int32_t v764_a = v9_lead + 96;
          float v771_data = s0[(v9_lead + 96)];
          float v772_data = s1[12];
          float v774_data = ir6[1];
          ir6[1] = (v774_data + (v771_data * v772_data));
          int32_t v781_a = v9_lead + 96;
          float v788_data = s0[(v9_lead + 96)];
          float v789_data = s1[21];
          float v791_data = ir6[2];
          ir6[2] = (v791_data + (v788_data * v789_data));
          int32_t v798_a = v9_lead + 96;
          float v805_data = s0[(v9_lead + 96)];
          float v806_data = s1[30];
          float v808_data = ir6[3];
          ir6[3] = (v808_data + (v805_data * v806_data));
          int32_t v815_a = v9_lead + 96;
          float v822_data = s0[(v9_lead + 96)];
          float v823_data = s1[39];
          float v825_data = ir6[4];
          ir6[4] = (v825_data + (v822_data * v823_data));
          int32_t v832_a = v9_lead + 96;
          float v839_data = s0[(v9_lead + 96)];
          float v840_data = s1[48];
          float v842_data = ir6[5];
          ir6[5] = (v842_data + (v839_data * v840_data));
          int32_t v849_a = v9_lead + 96;
          float v856_data = s0[(v9_lead + 96)];
          float v857_data = s1[57];
          float v859_data = ir6[6];
          ir6[6] = (v859_data + (v856_data * v857_data));
          int32_t v866_a = v9_lead + 96;
          float v873_data = s0[(v9_lead + 96)];
          float v874_data = s1[66];
          float v876_data = ir6[7];
          ir6[7] = (v876_data + (v873_data * v874_data));
          int32_t v883_a = v9_lead + 96;
          float v890_data = s0[(v9_lead + 96)];
          float v891_data = s1[75];
          float v893_data = ir6[8];
          ir6[8] = (v893_data + (v890_data * v891_data));
          int32_t v903_a = v9_lead + 128;
          float v910_data = s0[(v9_lead + 128)];
          float v911_data = s1[4];
          float v913_data = ir6[0];
          ir6[0] = (v913_data + (v910_data * v911_data));
          int32_t v920_a = v9_lead + 128;
          float v927_data = s0[(v9_lead + 128)];
          float v928_data = s1[13];
          float v930_data = ir6[1];
          ir6[1] = (v930_data + (v927_data * v928_data));
          int32_t v937_a = v9_lead + 128;
          float v944_data = s0[(v9_lead + 128)];
          float v945_data = s1[22];
          float v947_data = ir6[2];
          ir6[2] = (v947_data + (v944_data * v945_data));
          int32_t v954_a = v9_lead + 128;
          float v961_data = s0[(v9_lead + 128)];
          float v962_data = s1[31];
          float v964_data = ir6[3];
          ir6[3] = (v964_data + (v961_data * v962_data));
          int32_t v971_a = v9_lead + 128;
          float v978_data = s0[(v9_lead + 128)];
          float v979_data = s1[40];
          float v981_data = ir6[4];
          ir6[4] = (v981_data + (v978_data * v979_data));
          int32_t v988_a = v9_lead + 128;
          float v995_data = s0[(v9_lead + 128)];
          float v996_data = s1[49];
          float v998_data = ir6[5];
          ir6[5] = (v998_data + (v995_data * v996_data));
          int32_t v1005_a = v9_lead + 128;
          float v1012_data = s0[(v9_lead + 128)];
          float v1013_data = s1[58];
          float v1015_data = ir6[6];
          ir6[6] = (v1015_data + (v1012_data * v1013_data));
          int32_t v1022_a = v9_lead + 128;
          float v1029_data = s0[(v9_lead + 128)];
          float v1030_data = s1[67];
          float v1032_data = ir6[7];
          ir6[7] = (v1032_data + (v1029_data * v1030_data));
          int32_t v1039_a = v9_lead + 128;
          float v1046_data = s0[(v9_lead + 128)];
          float v1047_data = s1[76];
          float v1049_data = ir6[8];
          ir6[8] = (v1049_data + (v1046_data * v1047_data));
          int32_t v1059_a = v9_lead + 160;
          float v1066_data = s0[(v9_lead + 160)];
          float v1067_data = s1[5];
          float v1069_data = ir6[0];
          ir6[0] = (v1069_data + (v1066_data * v1067_data));
          int32_t v1076_a = v9_lead + 160;
          float v1083_data = s0[(v9_lead + 160)];
          float v1084_data = s1[14];
          float v1086_data = ir6[1];
          ir6[1] = (v1086_data + (v1083_data * v1084_data));
          int32_t v1093_a = v9_lead + 160;
          float v1100_data = s0[(v9_lead + 160)];
          float v1101_data = s1[23];
          float v1103_data = ir6[2];
          ir6[2] = (v1103_data + (v1100_data * v1101_data));
          int32_t v1110_a = v9_lead + 160;
          float v1117_data = s0[(v9_lead + 160)];
          float v1118_data = s1[32];
          float v1120_data = ir6[3];
          ir6[3] = (v1120_data + (v1117_data * v1118_data));
          int32_t v1127_a = v9_lead + 160;
          float v1134_data = s0[(v9_lead + 160)];
          float v1135_data = s1[41];
          float v1137_data = ir6[4];
          ir6[4] = (v1137_data + (v1134_data * v1135_data));
          int32_t v1144_a = v9_lead + 160;
          float v1151_data = s0[(v9_lead + 160)];
          float v1152_data = s1[50];
          float v1154_data = ir6[5];
          ir6[5] = (v1154_data + (v1151_data * v1152_data));
          int32_t v1161_a = v9_lead + 160;
          float v1168_data = s0[(v9_lead + 160)];
          float v1169_data = s1[59];
          float v1171_data = ir6[6];
          ir6[6] = (v1171_data + (v1168_data * v1169_data));
          int32_t v1178_a = v9_lead + 160;
          float v1185_data = s0[(v9_lead + 160)];
          float v1186_data = s1[68];
          float v1188_data = ir6[7];
          ir6[7] = (v1188_data + (v1185_data * v1186_data));
          int32_t v1195_a = v9_lead + 160;
          float v1202_data = s0[(v9_lead + 160)];
          float v1203_data = s1[77];
          float v1205_data = ir6[8];
          ir6[8] = (v1205_data + (v1202_data * v1203_data));
          int32_t v1215_a = v9_lead + 192;
          float v1222_data = s0[(v9_lead + 192)];
          float v1223_data = s1[6];
          float v1225_data = ir6[0];
          ir6[0] = (v1225_data + (v1222_data * v1223_data));
          int32_t v1232_a = v9_lead + 192;
          float v1239_data = s0[(v9_lead + 192)];
          float v1240_data = s1[15];
          float v1242_data = ir6[1];
          ir6[1] = (v1242_data + (v1239_data * v1240_data));
          int32_t v1249_a = v9_lead + 192;
          float v1256_data = s0[(v9_lead + 192)];
          float v1257_data = s1[24];
          float v1259_data = ir6[2];
          ir6[2] = (v1259_data + (v1256_data * v1257_data));
          int32_t v1266_a = v9_lead + 192;
          float v1273_data = s0[(v9_lead + 192)];
          float v1274_data = s1[33];
          float v1276_data = ir6[3];
          ir6[3] = (v1276_data + (v1273_data * v1274_data));
          int32_t v1283_a = v9_lead + 192;
          float v1290_data = s0[(v9_lead + 192)];
          float v1291_data = s1[42];
          float v1293_data = ir6[4];
          ir6[4] = (v1293_data + (v1290_data * v1291_data));
          int32_t v1300_a = v9_lead + 192;
          float v1307_data = s0[(v9_lead + 192)];
          float v1308_data = s1[51];
          float v1310_data = ir6[5];
          ir6[5] = (v1310_data + (v1307_data * v1308_data));
          int32_t v1317_a = v9_lead + 192;
          float v1324_data = s0[(v9_lead + 192)];
          float v1325_data = s1[60];
          float v1327_data = ir6[6];
          ir6[6] = (v1327_data + (v1324_data * v1325_data));
          int32_t v1334_a = v9_lead + 192;
          float v1341_data = s0[(v9_lead + 192)];
          float v1342_data = s1[69];
          float v1344_data = ir6[7];
          ir6[7] = (v1344_data + (v1341_data * v1342_data));
          int32_t v1351_a = v9_lead + 192;
          float v1358_data = s0[(v9_lead + 192)];
          float v1359_data = s1[78];
          float v1361_data = ir6[8];
          ir6[8] = (v1361_data + (v1358_data * v1359_data));
          int32_t v1371_a = v9_lead + 224;
          float v1378_data = s0[(v9_lead + 224)];
          float v1379_data = s1[7];
          float v1381_data = ir6[0];
          ir6[0] = (v1381_data + (v1378_data * v1379_data));
          int32_t v1388_a = v9_lead + 224;
          float v1395_data = s0[(v9_lead + 224)];
          float v1396_data = s1[16];
          float v1398_data = ir6[1];
          ir6[1] = (v1398_data + (v1395_data * v1396_data));
          int32_t v1405_a = v9_lead + 224;
          float v1412_data = s0[(v9_lead + 224)];
          float v1413_data = s1[25];
          float v1415_data = ir6[2];
          ir6[2] = (v1415_data + (v1412_data * v1413_data));
          int32_t v1422_a = v9_lead + 224;
          float v1429_data = s0[(v9_lead + 224)];
          float v1430_data = s1[34];
          float v1432_data = ir6[3];
          ir6[3] = (v1432_data + (v1429_data * v1430_data));
          int32_t v1439_a = v9_lead + 224;
          float v1446_data = s0[(v9_lead + 224)];
          float v1447_data = s1[43];
          float v1449_data = ir6[4];
          ir6[4] = (v1449_data + (v1446_data * v1447_data));
          int32_t v1456_a = v9_lead + 224;
          float v1463_data = s0[(v9_lead + 224)];
          float v1464_data = s1[52];
          float v1466_data = ir6[5];
          ir6[5] = (v1466_data + (v1463_data * v1464_data));
          int32_t v1473_a = v9_lead + 224;
          float v1480_data = s0[(v9_lead + 224)];
          float v1481_data = s1[61];
          float v1483_data = ir6[6];
          ir6[6] = (v1483_data + (v1480_data * v1481_data));
          int32_t v1490_a = v9_lead + 224;
          float v1497_data = s0[(v9_lead + 224)];
          float v1498_data = s1[70];
          float v1500_data = ir6[7];
          ir6[7] = (v1500_data + (v1497_data * v1498_data));
          int32_t v1507_a = v9_lead + 224;
          float v1514_data = s0[(v9_lead + 224)];
          float v1515_data = s1[79];
          float v1517_data = ir6[8];
          ir6[8] = (v1517_data + (v1514_data * v1515_data));
          int32_t v1527_a = v9_lead + 256;
          float v1534_data = s0[(v9_lead + 256)];
          float v1535_data = s1[8];
          float v1537_data = ir6[0];
          ir6[0] = (v1537_data + (v1534_data * v1535_data));
          int32_t v1544_a = v9_lead + 256;
          float v1551_data = s0[(v9_lead + 256)];
          float v1552_data = s1[17];
          float v1554_data = ir6[1];
          ir6[1] = (v1554_data + (v1551_data * v1552_data));
          int32_t v1561_a = v9_lead + 256;
          float v1568_data = s0[(v9_lead + 256)];
          float v1569_data = s1[26];
          float v1571_data = ir6[2];
          ir6[2] = (v1571_data + (v1568_data * v1569_data));
          int32_t v1578_a = v9_lead + 256;
          float v1585_data = s0[(v9_lead + 256)];
          float v1586_data = s1[35];
          float v1588_data = ir6[3];
          ir6[3] = (v1588_data + (v1585_data * v1586_data));
          int32_t v1595_a = v9_lead + 256;
          float v1602_data = s0[(v9_lead + 256)];
          float v1603_data = s1[44];
          float v1605_data = ir6[4];
          ir6[4] = (v1605_data + (v1602_data * v1603_data));
          int32_t v1612_a = v9_lead + 256;
          float v1619_data = s0[(v9_lead + 256)];
          float v1620_data = s1[53];
          float v1622_data = ir6[5];
          ir6[5] = (v1622_data + (v1619_data * v1620_data));
          int32_t v1629_a = v9_lead + 256;
          float v1636_data = s0[(v9_lead + 256)];
          float v1637_data = s1[62];
          float v1639_data = ir6[6];
          ir6[6] = (v1639_data + (v1636_data * v1637_data));
          int32_t v1646_a = v9_lead + 256;
          float v1653_data = s0[(v9_lead + 256)];
          float v1654_data = s1[71];
          float v1656_data = ir6[7];
          ir6[7] = (v1656_data + (v1653_data * v1654_data));
          int32_t v1663_a = v9_lead + 256;
          float v1670_data = s0[(v9_lead + 256)];
          float v1671_data = s1[80];
          float v1673_data = ir6[8];
          ir6[8] = (v1673_data + (v1670_data * v1671_data));
          #pragma unroll
          for (int32_t v1678_n0 = 0; v1678_n0 < 1; ++v1678_n0) {
            #pragma unroll
            for (int32_t v1679_n1 = 0; v1679_n1 < 9; ++v1679_n1) {
              int32_t v1680_a = v1678_n0 + v1679_n1;
              int32_t v1681_a = v1678_n0 + v1679_n1;
              float v1682_data = ir6[v1681_a];
              r6[v1681_a] = v1682_data;
            }
          }
          // glb_m3 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v1687_i0 = 0; v1687_i0 < 1; ++v1687_i0) {
            int32_t v1696_lead = v9_lead + (v1687_i0 * 32);
            #pragma unroll
            for (int32_t v1688_i1 = 0; v1688_i1 < 9; ++v1688_i1) {
              int32_t v1689_a = v1687_i0 + v1688_i1;
              float v1691_data = r6[(v1687_i0 + v1688_i1)];
              glb_m3[(v1696_lead + (v1688_i1 * 32))] = v1691_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

