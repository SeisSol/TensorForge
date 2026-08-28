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
          int32_t v12_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
            int32_t v19_lead = v12_lead + (v13_i0 * 32);
            #pragma unroll
            for (int32_t v14_i1 = 0; v14_i1 < 9; ++v14_i1) {
              float v22_data = __ldcg(&glb_m0[(v19_lead + (v14_i1 * 32))]);
              r0[(v13_i0 + v14_i1)] = v22_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 9; ++v29_i1) {
              float v37_data = __ldcg(&glb_m1[(v12_lead + (v29_i1 * 16))]);
              r2[v29_i1] = v37_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          float v43_data = r0[0];
          float v44_data = r1[0];
          r1[0] = (v44_data + v43_data);
          float v46_data = r0[1];
          float v47_data = r1[1];
          r1[1] = (v47_data + v46_data);
          float v49_data = r0[2];
          float v50_data = r1[2];
          r1[2] = (v50_data + v49_data);
          float v52_data = r0[3];
          float v53_data = r1[3];
          r1[3] = (v53_data + v52_data);
          float v55_data = r0[4];
          float v56_data = r1[4];
          r1[4] = (v56_data + v55_data);
          float v58_data = r0[5];
          float v59_data = r1[5];
          r1[5] = (v59_data + v58_data);
          float v61_data = r0[6];
          float v62_data = r1[6];
          r1[6] = (v62_data + v61_data);
          float v64_data = r0[7];
          float v65_data = r1[7];
          r1[7] = (v65_data + v64_data);
          float v67_data = r0[8];
          float v68_data = r1[8];
          r1[8] = (v68_data + v67_data);
          float* __restrict__ s0 = &localShrMem0[96];
          // s0 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v74_i0 = 0; v74_i0 < 1; ++v74_i0) {
            int32_t v82_lead = v12_lead + (v74_i0 * 32);
            #pragma unroll
            for (int32_t v75_i1 = 0; v75_i1 < 9; ++v75_i1) {
              float v77_data = r1[(v74_i0 + v75_i1)];
              s0[(v82_lead + (v75_i1 * 32))] = v77_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v90_i1 = 0; v90_i1 < 9; ++v90_i1) {
              float v98_data = __ldcg(&glb_m2[(v12_lead + (v90_i1 * 16))]);
              r4[v90_i1] = v98_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          __syncwarp();
          // r3 = +(r2) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir3[9]{};
          if (v12_lead < 16) {
            float v106_data = r2[0];
            float v107_data = ir3[0];
            ir3[0] = (v107_data + v106_data);
            float v109_data = r2[1];
            float v110_data = ir3[1];
            ir3[1] = (v110_data + v109_data);
            float v112_data = r2[2];
            float v113_data = ir3[2];
            ir3[2] = (v113_data + v112_data);
            float v115_data = r2[3];
            float v116_data = ir3[3];
            ir3[3] = (v116_data + v115_data);
            float v118_data = r2[4];
            float v119_data = ir3[4];
            ir3[4] = (v119_data + v118_data);
            float v121_data = r2[5];
            float v122_data = ir3[5];
            ir3[5] = (v122_data + v121_data);
            float v124_data = r2[6];
            float v125_data = ir3[6];
            ir3[6] = (v125_data + v124_data);
            float v127_data = r2[7];
            float v128_data = ir3[7];
            ir3[7] = (v128_data + v127_data);
            float v130_data = r2[8];
            float v131_data = ir3[8];
            ir3[8] = (v131_data + v130_data);
          }
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v137_n1 = 0; v137_n1 < 9; ++v137_n1) {
              float v139_data = ir3[v137_n1];
              float v147_data = s0[(v12_lead + (v137_n1 * 32))];
              r3[v137_n1] = (v147_data + v139_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r3);
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v154_i1 = 0; v154_i1 < 9; ++v154_i1) {
              float v156_data = r3[v154_i1];
              s0[(v12_lead + (v154_i1 * 32))] = v156_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          __syncwarp();
          // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir5[9]{};
          if (v12_lead < 16) {
            float v170_data = r4[0];
            float v171_data = ir5[0];
            ir5[0] = (v171_data + v170_data);
            float v173_data = r4[1];
            float v174_data = ir5[1];
            ir5[1] = (v174_data + v173_data);
            float v176_data = r4[2];
            float v177_data = ir5[2];
            ir5[2] = (v177_data + v176_data);
            float v179_data = r4[3];
            float v180_data = ir5[3];
            ir5[3] = (v180_data + v179_data);
            float v182_data = r4[4];
            float v183_data = ir5[4];
            ir5[4] = (v183_data + v182_data);
            float v185_data = r4[5];
            float v186_data = ir5[5];
            ir5[5] = (v186_data + v185_data);
            float v188_data = r4[6];
            float v189_data = ir5[6];
            ir5[6] = (v189_data + v188_data);
            float v191_data = r4[7];
            float v192_data = ir5[7];
            ir5[7] = (v192_data + v191_data);
            float v194_data = r4[8];
            float v195_data = ir5[8];
            ir5[8] = (v195_data + v194_data);
          }
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v201_n1 = 0; v201_n1 < 9; ++v201_n1) {
              float v203_data = ir5[v201_n1];
              float v211_data = s0[(v12_lead + (v201_n1 * 32))];
              r5[v201_n1] = (v211_data + v203_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r5);
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v218_i1 = 0; v218_i1 < 9; ++v218_i1) {
              float v220_data = r5[v218_i1];
              s0[(v12_lead + (v218_i1 * 32))] = v220_data;
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
          float v243_data = s0[v12_lead];
          float v244_data = s1[0];
          float v246_data = ir6[0];
          ir6[0] = (v246_data + (v243_data * v244_data));
          float v254_data = s0[v12_lead];
          float v255_data = s1[9];
          float v257_data = ir6[1];
          ir6[1] = (v257_data + (v254_data * v255_data));
          float v265_data = s0[v12_lead];
          float v266_data = s1[18];
          float v268_data = ir6[2];
          ir6[2] = (v268_data + (v265_data * v266_data));
          float v276_data = s0[v12_lead];
          float v277_data = s1[27];
          float v279_data = ir6[3];
          ir6[3] = (v279_data + (v276_data * v277_data));
          float v287_data = s0[v12_lead];
          float v288_data = s1[36];
          float v290_data = ir6[4];
          ir6[4] = (v290_data + (v287_data * v288_data));
          float v298_data = s0[v12_lead];
          float v299_data = s1[45];
          float v301_data = ir6[5];
          ir6[5] = (v301_data + (v298_data * v299_data));
          float v309_data = s0[v12_lead];
          float v310_data = s1[54];
          float v312_data = ir6[6];
          ir6[6] = (v312_data + (v309_data * v310_data));
          float v320_data = s0[v12_lead];
          float v321_data = s1[63];
          float v323_data = ir6[7];
          ir6[7] = (v323_data + (v320_data * v321_data));
          float v331_data = s0[v12_lead];
          float v332_data = s1[72];
          float v334_data = ir6[8];
          ir6[8] = (v334_data + (v331_data * v332_data));
          float v345_data = s0[(v12_lead + 32)];
          float v346_data = s1[1];
          float v348_data = ir6[0];
          ir6[0] = (v348_data + (v345_data * v346_data));
          float v356_data = s0[(v12_lead + 32)];
          float v357_data = s1[10];
          float v359_data = ir6[1];
          ir6[1] = (v359_data + (v356_data * v357_data));
          float v367_data = s0[(v12_lead + 32)];
          float v368_data = s1[19];
          float v370_data = ir6[2];
          ir6[2] = (v370_data + (v367_data * v368_data));
          float v378_data = s0[(v12_lead + 32)];
          float v379_data = s1[28];
          float v381_data = ir6[3];
          ir6[3] = (v381_data + (v378_data * v379_data));
          float v389_data = s0[(v12_lead + 32)];
          float v390_data = s1[37];
          float v392_data = ir6[4];
          ir6[4] = (v392_data + (v389_data * v390_data));
          float v400_data = s0[(v12_lead + 32)];
          float v401_data = s1[46];
          float v403_data = ir6[5];
          ir6[5] = (v403_data + (v400_data * v401_data));
          float v411_data = s0[(v12_lead + 32)];
          float v412_data = s1[55];
          float v414_data = ir6[6];
          ir6[6] = (v414_data + (v411_data * v412_data));
          float v422_data = s0[(v12_lead + 32)];
          float v423_data = s1[64];
          float v425_data = ir6[7];
          ir6[7] = (v425_data + (v422_data * v423_data));
          float v433_data = s0[(v12_lead + 32)];
          float v434_data = s1[73];
          float v436_data = ir6[8];
          ir6[8] = (v436_data + (v433_data * v434_data));
          float v447_data = s0[(v12_lead + 64)];
          float v448_data = s1[2];
          float v450_data = ir6[0];
          ir6[0] = (v450_data + (v447_data * v448_data));
          float v458_data = s0[(v12_lead + 64)];
          float v459_data = s1[11];
          float v461_data = ir6[1];
          ir6[1] = (v461_data + (v458_data * v459_data));
          float v469_data = s0[(v12_lead + 64)];
          float v470_data = s1[20];
          float v472_data = ir6[2];
          ir6[2] = (v472_data + (v469_data * v470_data));
          float v480_data = s0[(v12_lead + 64)];
          float v481_data = s1[29];
          float v483_data = ir6[3];
          ir6[3] = (v483_data + (v480_data * v481_data));
          float v491_data = s0[(v12_lead + 64)];
          float v492_data = s1[38];
          float v494_data = ir6[4];
          ir6[4] = (v494_data + (v491_data * v492_data));
          float v502_data = s0[(v12_lead + 64)];
          float v503_data = s1[47];
          float v505_data = ir6[5];
          ir6[5] = (v505_data + (v502_data * v503_data));
          float v513_data = s0[(v12_lead + 64)];
          float v514_data = s1[56];
          float v516_data = ir6[6];
          ir6[6] = (v516_data + (v513_data * v514_data));
          float v524_data = s0[(v12_lead + 64)];
          float v525_data = s1[65];
          float v527_data = ir6[7];
          ir6[7] = (v527_data + (v524_data * v525_data));
          float v535_data = s0[(v12_lead + 64)];
          float v536_data = s1[74];
          float v538_data = ir6[8];
          ir6[8] = (v538_data + (v535_data * v536_data));
          float v549_data = s0[(v12_lead + 96)];
          float v550_data = s1[3];
          float v552_data = ir6[0];
          ir6[0] = (v552_data + (v549_data * v550_data));
          float v560_data = s0[(v12_lead + 96)];
          float v561_data = s1[12];
          float v563_data = ir6[1];
          ir6[1] = (v563_data + (v560_data * v561_data));
          float v571_data = s0[(v12_lead + 96)];
          float v572_data = s1[21];
          float v574_data = ir6[2];
          ir6[2] = (v574_data + (v571_data * v572_data));
          float v582_data = s0[(v12_lead + 96)];
          float v583_data = s1[30];
          float v585_data = ir6[3];
          ir6[3] = (v585_data + (v582_data * v583_data));
          float v593_data = s0[(v12_lead + 96)];
          float v594_data = s1[39];
          float v596_data = ir6[4];
          ir6[4] = (v596_data + (v593_data * v594_data));
          float v604_data = s0[(v12_lead + 96)];
          float v605_data = s1[48];
          float v607_data = ir6[5];
          ir6[5] = (v607_data + (v604_data * v605_data));
          float v615_data = s0[(v12_lead + 96)];
          float v616_data = s1[57];
          float v618_data = ir6[6];
          ir6[6] = (v618_data + (v615_data * v616_data));
          float v626_data = s0[(v12_lead + 96)];
          float v627_data = s1[66];
          float v629_data = ir6[7];
          ir6[7] = (v629_data + (v626_data * v627_data));
          float v637_data = s0[(v12_lead + 96)];
          float v638_data = s1[75];
          float v640_data = ir6[8];
          ir6[8] = (v640_data + (v637_data * v638_data));
          float v651_data = s0[(v12_lead + 128)];
          float v652_data = s1[4];
          float v654_data = ir6[0];
          ir6[0] = (v654_data + (v651_data * v652_data));
          float v662_data = s0[(v12_lead + 128)];
          float v663_data = s1[13];
          float v665_data = ir6[1];
          ir6[1] = (v665_data + (v662_data * v663_data));
          float v673_data = s0[(v12_lead + 128)];
          float v674_data = s1[22];
          float v676_data = ir6[2];
          ir6[2] = (v676_data + (v673_data * v674_data));
          float v684_data = s0[(v12_lead + 128)];
          float v685_data = s1[31];
          float v687_data = ir6[3];
          ir6[3] = (v687_data + (v684_data * v685_data));
          float v695_data = s0[(v12_lead + 128)];
          float v696_data = s1[40];
          float v698_data = ir6[4];
          ir6[4] = (v698_data + (v695_data * v696_data));
          float v706_data = s0[(v12_lead + 128)];
          float v707_data = s1[49];
          float v709_data = ir6[5];
          ir6[5] = (v709_data + (v706_data * v707_data));
          float v717_data = s0[(v12_lead + 128)];
          float v718_data = s1[58];
          float v720_data = ir6[6];
          ir6[6] = (v720_data + (v717_data * v718_data));
          float v728_data = s0[(v12_lead + 128)];
          float v729_data = s1[67];
          float v731_data = ir6[7];
          ir6[7] = (v731_data + (v728_data * v729_data));
          float v739_data = s0[(v12_lead + 128)];
          float v740_data = s1[76];
          float v742_data = ir6[8];
          ir6[8] = (v742_data + (v739_data * v740_data));
          float v753_data = s0[(v12_lead + 160)];
          float v754_data = s1[5];
          float v756_data = ir6[0];
          ir6[0] = (v756_data + (v753_data * v754_data));
          float v764_data = s0[(v12_lead + 160)];
          float v765_data = s1[14];
          float v767_data = ir6[1];
          ir6[1] = (v767_data + (v764_data * v765_data));
          float v775_data = s0[(v12_lead + 160)];
          float v776_data = s1[23];
          float v778_data = ir6[2];
          ir6[2] = (v778_data + (v775_data * v776_data));
          float v786_data = s0[(v12_lead + 160)];
          float v787_data = s1[32];
          float v789_data = ir6[3];
          ir6[3] = (v789_data + (v786_data * v787_data));
          float v797_data = s0[(v12_lead + 160)];
          float v798_data = s1[41];
          float v800_data = ir6[4];
          ir6[4] = (v800_data + (v797_data * v798_data));
          float v808_data = s0[(v12_lead + 160)];
          float v809_data = s1[50];
          float v811_data = ir6[5];
          ir6[5] = (v811_data + (v808_data * v809_data));
          float v819_data = s0[(v12_lead + 160)];
          float v820_data = s1[59];
          float v822_data = ir6[6];
          ir6[6] = (v822_data + (v819_data * v820_data));
          float v830_data = s0[(v12_lead + 160)];
          float v831_data = s1[68];
          float v833_data = ir6[7];
          ir6[7] = (v833_data + (v830_data * v831_data));
          float v841_data = s0[(v12_lead + 160)];
          float v842_data = s1[77];
          float v844_data = ir6[8];
          ir6[8] = (v844_data + (v841_data * v842_data));
          float v855_data = s0[(v12_lead + 192)];
          float v856_data = s1[6];
          float v858_data = ir6[0];
          ir6[0] = (v858_data + (v855_data * v856_data));
          float v866_data = s0[(v12_lead + 192)];
          float v867_data = s1[15];
          float v869_data = ir6[1];
          ir6[1] = (v869_data + (v866_data * v867_data));
          float v877_data = s0[(v12_lead + 192)];
          float v878_data = s1[24];
          float v880_data = ir6[2];
          ir6[2] = (v880_data + (v877_data * v878_data));
          float v888_data = s0[(v12_lead + 192)];
          float v889_data = s1[33];
          float v891_data = ir6[3];
          ir6[3] = (v891_data + (v888_data * v889_data));
          float v899_data = s0[(v12_lead + 192)];
          float v900_data = s1[42];
          float v902_data = ir6[4];
          ir6[4] = (v902_data + (v899_data * v900_data));
          float v910_data = s0[(v12_lead + 192)];
          float v911_data = s1[51];
          float v913_data = ir6[5];
          ir6[5] = (v913_data + (v910_data * v911_data));
          float v921_data = s0[(v12_lead + 192)];
          float v922_data = s1[60];
          float v924_data = ir6[6];
          ir6[6] = (v924_data + (v921_data * v922_data));
          float v932_data = s0[(v12_lead + 192)];
          float v933_data = s1[69];
          float v935_data = ir6[7];
          ir6[7] = (v935_data + (v932_data * v933_data));
          float v943_data = s0[(v12_lead + 192)];
          float v944_data = s1[78];
          float v946_data = ir6[8];
          ir6[8] = (v946_data + (v943_data * v944_data));
          float v957_data = s0[(v12_lead + 224)];
          float v958_data = s1[7];
          float v960_data = ir6[0];
          ir6[0] = (v960_data + (v957_data * v958_data));
          float v968_data = s0[(v12_lead + 224)];
          float v969_data = s1[16];
          float v971_data = ir6[1];
          ir6[1] = (v971_data + (v968_data * v969_data));
          float v979_data = s0[(v12_lead + 224)];
          float v980_data = s1[25];
          float v982_data = ir6[2];
          ir6[2] = (v982_data + (v979_data * v980_data));
          float v990_data = s0[(v12_lead + 224)];
          float v991_data = s1[34];
          float v993_data = ir6[3];
          ir6[3] = (v993_data + (v990_data * v991_data));
          float v1001_data = s0[(v12_lead + 224)];
          float v1002_data = s1[43];
          float v1004_data = ir6[4];
          ir6[4] = (v1004_data + (v1001_data * v1002_data));
          float v1012_data = s0[(v12_lead + 224)];
          float v1013_data = s1[52];
          float v1015_data = ir6[5];
          ir6[5] = (v1015_data + (v1012_data * v1013_data));
          float v1023_data = s0[(v12_lead + 224)];
          float v1024_data = s1[61];
          float v1026_data = ir6[6];
          ir6[6] = (v1026_data + (v1023_data * v1024_data));
          float v1034_data = s0[(v12_lead + 224)];
          float v1035_data = s1[70];
          float v1037_data = ir6[7];
          ir6[7] = (v1037_data + (v1034_data * v1035_data));
          float v1045_data = s0[(v12_lead + 224)];
          float v1046_data = s1[79];
          float v1048_data = ir6[8];
          ir6[8] = (v1048_data + (v1045_data * v1046_data));
          float v1059_data = s0[(v12_lead + 256)];
          float v1060_data = s1[8];
          float v1062_data = ir6[0];
          ir6[0] = (v1062_data + (v1059_data * v1060_data));
          float v1070_data = s0[(v12_lead + 256)];
          float v1071_data = s1[17];
          float v1073_data = ir6[1];
          ir6[1] = (v1073_data + (v1070_data * v1071_data));
          float v1081_data = s0[(v12_lead + 256)];
          float v1082_data = s1[26];
          float v1084_data = ir6[2];
          ir6[2] = (v1084_data + (v1081_data * v1082_data));
          float v1092_data = s0[(v12_lead + 256)];
          float v1093_data = s1[35];
          float v1095_data = ir6[3];
          ir6[3] = (v1095_data + (v1092_data * v1093_data));
          float v1103_data = s0[(v12_lead + 256)];
          float v1104_data = s1[44];
          float v1106_data = ir6[4];
          ir6[4] = (v1106_data + (v1103_data * v1104_data));
          float v1114_data = s0[(v12_lead + 256)];
          float v1115_data = s1[53];
          float v1117_data = ir6[5];
          ir6[5] = (v1117_data + (v1114_data * v1115_data));
          float v1125_data = s0[(v12_lead + 256)];
          float v1126_data = s1[62];
          float v1128_data = ir6[6];
          ir6[6] = (v1128_data + (v1125_data * v1126_data));
          float v1136_data = s0[(v12_lead + 256)];
          float v1137_data = s1[71];
          float v1139_data = ir6[7];
          ir6[7] = (v1139_data + (v1136_data * v1137_data));
          float v1147_data = s0[(v12_lead + 256)];
          float v1148_data = s1[80];
          float v1150_data = ir6[8];
          ir6[8] = (v1150_data + (v1147_data * v1148_data));
          #pragma unroll
          for (int32_t v1155_n0 = 0; v1155_n0 < 1; ++v1155_n0) {
            #pragma unroll
            for (int32_t v1156_n1 = 0; v1156_n1 < 9; ++v1156_n1) {
              int32_t v1157_a = v1155_n0 + v1156_n1;
              float v1158_data = ir6[v1157_a];
              r6[v1157_a] = v1158_data;
            }
          }
          // glb_m3 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v1163_i0 = 0; v1163_i0 < 1; ++v1163_i0) {
            int32_t v1171_lead = v12_lead + (v1163_i0 * 32);
            #pragma unroll
            for (int32_t v1164_i1 = 0; v1164_i1 < 9; ++v1164_i1) {
              float v1166_data = r6[(v1163_i0 + v1164_i1)];
              glb_m3[(v1171_lead + (v1164_i1 * 32))] = v1166_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

