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
      float* __restrict__ s0 = &localShrMem0[96];
      float* __restrict__ s1 = &localShrMem0[0];
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
          int32_t v17_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v18_i0 = 0; v18_i0 < 1; ++v18_i0) {
            int32_t v24_lead = v17_lead + (v18_i0 * 32);
            #pragma unroll
            for (int32_t v19_i1 = 0; v19_i1 < 9; ++v19_i1) {
              float v27_data = __ldcg(&glb_m0[(v24_lead + (v19_i1 * 32))]);
              r0[(v18_i0 + v19_i1)] = v27_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          if (v17_lead < 16) {
            #pragma unroll
            for (int32_t v34_i1 = 0; v34_i1 < 9; ++v34_i1) {
              float v42_data = __ldcg(&glb_m1[(v17_lead + (v34_i1 * 16))]);
              r2[v34_i1] = v42_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          float v48_data = r0[0];
          float v49_data = r1[0];
          r1[0] = (v49_data + v48_data);
          float v51_data = r0[1];
          float v52_data = r1[1];
          r1[1] = (v52_data + v51_data);
          float v54_data = r0[2];
          float v55_data = r1[2];
          r1[2] = (v55_data + v54_data);
          float v57_data = r0[3];
          float v58_data = r1[3];
          r1[3] = (v58_data + v57_data);
          float v60_data = r0[4];
          float v61_data = r1[4];
          r1[4] = (v61_data + v60_data);
          float v63_data = r0[5];
          float v64_data = r1[5];
          r1[5] = (v64_data + v63_data);
          float v66_data = r0[6];
          float v67_data = r1[6];
          r1[6] = (v67_data + v66_data);
          float v69_data = r0[7];
          float v70_data = r1[7];
          r1[7] = (v70_data + v69_data);
          float v72_data = r0[8];
          float v73_data = r1[8];
          r1[8] = (v73_data + v72_data);
          // s0 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v78_i0 = 0; v78_i0 < 1; ++v78_i0) {
            int32_t v86_lead = v17_lead + (v78_i0 * 32);
            #pragma unroll
            for (int32_t v79_i1 = 0; v79_i1 < 9; ++v79_i1) {
              float v81_data = r1[(v78_i0 + v79_i1)];
              s0[(v86_lead + (v79_i1 * 32))] = v81_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v17_lead < 16) {
            #pragma unroll
            for (int32_t v94_i1 = 0; v94_i1 < 9; ++v94_i1) {
              float v102_data = __ldcg(&glb_m2[(v17_lead + (v94_i1 * 16))]);
              r4[v94_i1] = v102_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          __syncwarp();
          // r3 = +(r2) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir3[9]{};
          if (v17_lead < 16) {
            float v110_data = r2[0];
            float v111_data = ir3[0];
            ir3[0] = (v111_data + v110_data);
            float v113_data = r2[1];
            float v114_data = ir3[1];
            ir3[1] = (v114_data + v113_data);
            float v116_data = r2[2];
            float v117_data = ir3[2];
            ir3[2] = (v117_data + v116_data);
            float v119_data = r2[3];
            float v120_data = ir3[3];
            ir3[3] = (v120_data + v119_data);
            float v122_data = r2[4];
            float v123_data = ir3[4];
            ir3[4] = (v123_data + v122_data);
            float v125_data = r2[5];
            float v126_data = ir3[5];
            ir3[5] = (v126_data + v125_data);
            float v128_data = r2[6];
            float v129_data = ir3[6];
            ir3[6] = (v129_data + v128_data);
            float v131_data = r2[7];
            float v132_data = ir3[7];
            ir3[7] = (v132_data + v131_data);
            float v134_data = r2[8];
            float v135_data = ir3[8];
            ir3[8] = (v135_data + v134_data);
          }
          if (v17_lead < 16) {
            #pragma unroll
            for (int32_t v141_n1 = 0; v141_n1 < 9; ++v141_n1) {
              float v143_data = ir3[v141_n1];
              float v151_data = s0[(v17_lead + (v141_n1 * 32))];
              r3[v141_n1] = (v151_data + v143_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r3);
          if (v17_lead < 16) {
            #pragma unroll
            for (int32_t v158_i1 = 0; v158_i1 < 9; ++v158_i1) {
              float v160_data = r3[v158_i1];
              s0[(v17_lead + (v158_i1 * 32))] = v160_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          __syncwarp();
          // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir5[9]{};
          if (v17_lead < 16) {
            float v174_data = r4[0];
            float v175_data = ir5[0];
            ir5[0] = (v175_data + v174_data);
            float v177_data = r4[1];
            float v178_data = ir5[1];
            ir5[1] = (v178_data + v177_data);
            float v180_data = r4[2];
            float v181_data = ir5[2];
            ir5[2] = (v181_data + v180_data);
            float v183_data = r4[3];
            float v184_data = ir5[3];
            ir5[3] = (v184_data + v183_data);
            float v186_data = r4[4];
            float v187_data = ir5[4];
            ir5[4] = (v187_data + v186_data);
            float v189_data = r4[5];
            float v190_data = ir5[5];
            ir5[5] = (v190_data + v189_data);
            float v192_data = r4[6];
            float v193_data = ir5[6];
            ir5[6] = (v193_data + v192_data);
            float v195_data = r4[7];
            float v196_data = ir5[7];
            ir5[7] = (v196_data + v195_data);
            float v198_data = r4[8];
            float v199_data = ir5[8];
            ir5[8] = (v199_data + v198_data);
          }
          if (v17_lead < 16) {
            #pragma unroll
            for (int32_t v205_n1 = 0; v205_n1 < 9; ++v205_n1) {
              float v207_data = ir5[v205_n1];
              float v215_data = s0[(v17_lead + (v205_n1 * 32))];
              r5[v205_n1] = (v215_data + v207_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r5);
          if (v17_lead < 16) {
            #pragma unroll
            for (int32_t v222_i1 = 0; v222_i1 < 9; ++v222_i1) {
              float v224_data = r5[v222_i1];
              s0[(v17_lead + (v222_i1 * 32))] = v224_data;
            }
          }
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
          float v246_data = s0[v17_lead];
          float v247_data = s1[0];
          float v249_data = ir6[0];
          ir6[0] = (v249_data + (v246_data * v247_data));
          float v257_data = s0[v17_lead];
          float v258_data = s1[9];
          float v260_data = ir6[1];
          ir6[1] = (v260_data + (v257_data * v258_data));
          float v268_data = s0[v17_lead];
          float v269_data = s1[18];
          float v271_data = ir6[2];
          ir6[2] = (v271_data + (v268_data * v269_data));
          float v279_data = s0[v17_lead];
          float v280_data = s1[27];
          float v282_data = ir6[3];
          ir6[3] = (v282_data + (v279_data * v280_data));
          float v290_data = s0[v17_lead];
          float v291_data = s1[36];
          float v293_data = ir6[4];
          ir6[4] = (v293_data + (v290_data * v291_data));
          float v301_data = s0[v17_lead];
          float v302_data = s1[45];
          float v304_data = ir6[5];
          ir6[5] = (v304_data + (v301_data * v302_data));
          float v312_data = s0[v17_lead];
          float v313_data = s1[54];
          float v315_data = ir6[6];
          ir6[6] = (v315_data + (v312_data * v313_data));
          float v323_data = s0[v17_lead];
          float v324_data = s1[63];
          float v326_data = ir6[7];
          ir6[7] = (v326_data + (v323_data * v324_data));
          float v334_data = s0[v17_lead];
          float v335_data = s1[72];
          float v337_data = ir6[8];
          ir6[8] = (v337_data + (v334_data * v335_data));
          float v348_data = s0[(v17_lead + 32)];
          float v349_data = s1[1];
          float v351_data = ir6[0];
          ir6[0] = (v351_data + (v348_data * v349_data));
          float v359_data = s0[(v17_lead + 32)];
          float v360_data = s1[10];
          float v362_data = ir6[1];
          ir6[1] = (v362_data + (v359_data * v360_data));
          float v370_data = s0[(v17_lead + 32)];
          float v371_data = s1[19];
          float v373_data = ir6[2];
          ir6[2] = (v373_data + (v370_data * v371_data));
          float v381_data = s0[(v17_lead + 32)];
          float v382_data = s1[28];
          float v384_data = ir6[3];
          ir6[3] = (v384_data + (v381_data * v382_data));
          float v392_data = s0[(v17_lead + 32)];
          float v393_data = s1[37];
          float v395_data = ir6[4];
          ir6[4] = (v395_data + (v392_data * v393_data));
          float v403_data = s0[(v17_lead + 32)];
          float v404_data = s1[46];
          float v406_data = ir6[5];
          ir6[5] = (v406_data + (v403_data * v404_data));
          float v414_data = s0[(v17_lead + 32)];
          float v415_data = s1[55];
          float v417_data = ir6[6];
          ir6[6] = (v417_data + (v414_data * v415_data));
          float v425_data = s0[(v17_lead + 32)];
          float v426_data = s1[64];
          float v428_data = ir6[7];
          ir6[7] = (v428_data + (v425_data * v426_data));
          float v436_data = s0[(v17_lead + 32)];
          float v437_data = s1[73];
          float v439_data = ir6[8];
          ir6[8] = (v439_data + (v436_data * v437_data));
          float v450_data = s0[(v17_lead + 64)];
          float v451_data = s1[2];
          float v453_data = ir6[0];
          ir6[0] = (v453_data + (v450_data * v451_data));
          float v461_data = s0[(v17_lead + 64)];
          float v462_data = s1[11];
          float v464_data = ir6[1];
          ir6[1] = (v464_data + (v461_data * v462_data));
          float v472_data = s0[(v17_lead + 64)];
          float v473_data = s1[20];
          float v475_data = ir6[2];
          ir6[2] = (v475_data + (v472_data * v473_data));
          float v483_data = s0[(v17_lead + 64)];
          float v484_data = s1[29];
          float v486_data = ir6[3];
          ir6[3] = (v486_data + (v483_data * v484_data));
          float v494_data = s0[(v17_lead + 64)];
          float v495_data = s1[38];
          float v497_data = ir6[4];
          ir6[4] = (v497_data + (v494_data * v495_data));
          float v505_data = s0[(v17_lead + 64)];
          float v506_data = s1[47];
          float v508_data = ir6[5];
          ir6[5] = (v508_data + (v505_data * v506_data));
          float v516_data = s0[(v17_lead + 64)];
          float v517_data = s1[56];
          float v519_data = ir6[6];
          ir6[6] = (v519_data + (v516_data * v517_data));
          float v527_data = s0[(v17_lead + 64)];
          float v528_data = s1[65];
          float v530_data = ir6[7];
          ir6[7] = (v530_data + (v527_data * v528_data));
          float v538_data = s0[(v17_lead + 64)];
          float v539_data = s1[74];
          float v541_data = ir6[8];
          ir6[8] = (v541_data + (v538_data * v539_data));
          float v552_data = s0[(v17_lead + 96)];
          float v553_data = s1[3];
          float v555_data = ir6[0];
          ir6[0] = (v555_data + (v552_data * v553_data));
          float v563_data = s0[(v17_lead + 96)];
          float v564_data = s1[12];
          float v566_data = ir6[1];
          ir6[1] = (v566_data + (v563_data * v564_data));
          float v574_data = s0[(v17_lead + 96)];
          float v575_data = s1[21];
          float v577_data = ir6[2];
          ir6[2] = (v577_data + (v574_data * v575_data));
          float v585_data = s0[(v17_lead + 96)];
          float v586_data = s1[30];
          float v588_data = ir6[3];
          ir6[3] = (v588_data + (v585_data * v586_data));
          float v596_data = s0[(v17_lead + 96)];
          float v597_data = s1[39];
          float v599_data = ir6[4];
          ir6[4] = (v599_data + (v596_data * v597_data));
          float v607_data = s0[(v17_lead + 96)];
          float v608_data = s1[48];
          float v610_data = ir6[5];
          ir6[5] = (v610_data + (v607_data * v608_data));
          float v618_data = s0[(v17_lead + 96)];
          float v619_data = s1[57];
          float v621_data = ir6[6];
          ir6[6] = (v621_data + (v618_data * v619_data));
          float v629_data = s0[(v17_lead + 96)];
          float v630_data = s1[66];
          float v632_data = ir6[7];
          ir6[7] = (v632_data + (v629_data * v630_data));
          float v640_data = s0[(v17_lead + 96)];
          float v641_data = s1[75];
          float v643_data = ir6[8];
          ir6[8] = (v643_data + (v640_data * v641_data));
          float v654_data = s0[(v17_lead + 128)];
          float v655_data = s1[4];
          float v657_data = ir6[0];
          ir6[0] = (v657_data + (v654_data * v655_data));
          float v665_data = s0[(v17_lead + 128)];
          float v666_data = s1[13];
          float v668_data = ir6[1];
          ir6[1] = (v668_data + (v665_data * v666_data));
          float v676_data = s0[(v17_lead + 128)];
          float v677_data = s1[22];
          float v679_data = ir6[2];
          ir6[2] = (v679_data + (v676_data * v677_data));
          float v687_data = s0[(v17_lead + 128)];
          float v688_data = s1[31];
          float v690_data = ir6[3];
          ir6[3] = (v690_data + (v687_data * v688_data));
          float v698_data = s0[(v17_lead + 128)];
          float v699_data = s1[40];
          float v701_data = ir6[4];
          ir6[4] = (v701_data + (v698_data * v699_data));
          float v709_data = s0[(v17_lead + 128)];
          float v710_data = s1[49];
          float v712_data = ir6[5];
          ir6[5] = (v712_data + (v709_data * v710_data));
          float v720_data = s0[(v17_lead + 128)];
          float v721_data = s1[58];
          float v723_data = ir6[6];
          ir6[6] = (v723_data + (v720_data * v721_data));
          float v731_data = s0[(v17_lead + 128)];
          float v732_data = s1[67];
          float v734_data = ir6[7];
          ir6[7] = (v734_data + (v731_data * v732_data));
          float v742_data = s0[(v17_lead + 128)];
          float v743_data = s1[76];
          float v745_data = ir6[8];
          ir6[8] = (v745_data + (v742_data * v743_data));
          float v756_data = s0[(v17_lead + 160)];
          float v757_data = s1[5];
          float v759_data = ir6[0];
          ir6[0] = (v759_data + (v756_data * v757_data));
          float v767_data = s0[(v17_lead + 160)];
          float v768_data = s1[14];
          float v770_data = ir6[1];
          ir6[1] = (v770_data + (v767_data * v768_data));
          float v778_data = s0[(v17_lead + 160)];
          float v779_data = s1[23];
          float v781_data = ir6[2];
          ir6[2] = (v781_data + (v778_data * v779_data));
          float v789_data = s0[(v17_lead + 160)];
          float v790_data = s1[32];
          float v792_data = ir6[3];
          ir6[3] = (v792_data + (v789_data * v790_data));
          float v800_data = s0[(v17_lead + 160)];
          float v801_data = s1[41];
          float v803_data = ir6[4];
          ir6[4] = (v803_data + (v800_data * v801_data));
          float v811_data = s0[(v17_lead + 160)];
          float v812_data = s1[50];
          float v814_data = ir6[5];
          ir6[5] = (v814_data + (v811_data * v812_data));
          float v822_data = s0[(v17_lead + 160)];
          float v823_data = s1[59];
          float v825_data = ir6[6];
          ir6[6] = (v825_data + (v822_data * v823_data));
          float v833_data = s0[(v17_lead + 160)];
          float v834_data = s1[68];
          float v836_data = ir6[7];
          ir6[7] = (v836_data + (v833_data * v834_data));
          float v844_data = s0[(v17_lead + 160)];
          float v845_data = s1[77];
          float v847_data = ir6[8];
          ir6[8] = (v847_data + (v844_data * v845_data));
          float v858_data = s0[(v17_lead + 192)];
          float v859_data = s1[6];
          float v861_data = ir6[0];
          ir6[0] = (v861_data + (v858_data * v859_data));
          float v869_data = s0[(v17_lead + 192)];
          float v870_data = s1[15];
          float v872_data = ir6[1];
          ir6[1] = (v872_data + (v869_data * v870_data));
          float v880_data = s0[(v17_lead + 192)];
          float v881_data = s1[24];
          float v883_data = ir6[2];
          ir6[2] = (v883_data + (v880_data * v881_data));
          float v891_data = s0[(v17_lead + 192)];
          float v892_data = s1[33];
          float v894_data = ir6[3];
          ir6[3] = (v894_data + (v891_data * v892_data));
          float v902_data = s0[(v17_lead + 192)];
          float v903_data = s1[42];
          float v905_data = ir6[4];
          ir6[4] = (v905_data + (v902_data * v903_data));
          float v913_data = s0[(v17_lead + 192)];
          float v914_data = s1[51];
          float v916_data = ir6[5];
          ir6[5] = (v916_data + (v913_data * v914_data));
          float v924_data = s0[(v17_lead + 192)];
          float v925_data = s1[60];
          float v927_data = ir6[6];
          ir6[6] = (v927_data + (v924_data * v925_data));
          float v935_data = s0[(v17_lead + 192)];
          float v936_data = s1[69];
          float v938_data = ir6[7];
          ir6[7] = (v938_data + (v935_data * v936_data));
          float v946_data = s0[(v17_lead + 192)];
          float v947_data = s1[78];
          float v949_data = ir6[8];
          ir6[8] = (v949_data + (v946_data * v947_data));
          float v960_data = s0[(v17_lead + 224)];
          float v961_data = s1[7];
          float v963_data = ir6[0];
          ir6[0] = (v963_data + (v960_data * v961_data));
          float v971_data = s0[(v17_lead + 224)];
          float v972_data = s1[16];
          float v974_data = ir6[1];
          ir6[1] = (v974_data + (v971_data * v972_data));
          float v982_data = s0[(v17_lead + 224)];
          float v983_data = s1[25];
          float v985_data = ir6[2];
          ir6[2] = (v985_data + (v982_data * v983_data));
          float v993_data = s0[(v17_lead + 224)];
          float v994_data = s1[34];
          float v996_data = ir6[3];
          ir6[3] = (v996_data + (v993_data * v994_data));
          float v1004_data = s0[(v17_lead + 224)];
          float v1005_data = s1[43];
          float v1007_data = ir6[4];
          ir6[4] = (v1007_data + (v1004_data * v1005_data));
          float v1015_data = s0[(v17_lead + 224)];
          float v1016_data = s1[52];
          float v1018_data = ir6[5];
          ir6[5] = (v1018_data + (v1015_data * v1016_data));
          float v1026_data = s0[(v17_lead + 224)];
          float v1027_data = s1[61];
          float v1029_data = ir6[6];
          ir6[6] = (v1029_data + (v1026_data * v1027_data));
          float v1037_data = s0[(v17_lead + 224)];
          float v1038_data = s1[70];
          float v1040_data = ir6[7];
          ir6[7] = (v1040_data + (v1037_data * v1038_data));
          float v1048_data = s0[(v17_lead + 224)];
          float v1049_data = s1[79];
          float v1051_data = ir6[8];
          ir6[8] = (v1051_data + (v1048_data * v1049_data));
          float v1062_data = s0[(v17_lead + 256)];
          float v1063_data = s1[8];
          float v1065_data = ir6[0];
          ir6[0] = (v1065_data + (v1062_data * v1063_data));
          float v1073_data = s0[(v17_lead + 256)];
          float v1074_data = s1[17];
          float v1076_data = ir6[1];
          ir6[1] = (v1076_data + (v1073_data * v1074_data));
          float v1084_data = s0[(v17_lead + 256)];
          float v1085_data = s1[26];
          float v1087_data = ir6[2];
          ir6[2] = (v1087_data + (v1084_data * v1085_data));
          float v1095_data = s0[(v17_lead + 256)];
          float v1096_data = s1[35];
          float v1098_data = ir6[3];
          ir6[3] = (v1098_data + (v1095_data * v1096_data));
          float v1106_data = s0[(v17_lead + 256)];
          float v1107_data = s1[44];
          float v1109_data = ir6[4];
          ir6[4] = (v1109_data + (v1106_data * v1107_data));
          float v1117_data = s0[(v17_lead + 256)];
          float v1118_data = s1[53];
          float v1120_data = ir6[5];
          ir6[5] = (v1120_data + (v1117_data * v1118_data));
          float v1128_data = s0[(v17_lead + 256)];
          float v1129_data = s1[62];
          float v1131_data = ir6[6];
          ir6[6] = (v1131_data + (v1128_data * v1129_data));
          float v1139_data = s0[(v17_lead + 256)];
          float v1140_data = s1[71];
          float v1142_data = ir6[7];
          ir6[7] = (v1142_data + (v1139_data * v1140_data));
          float v1150_data = s0[(v17_lead + 256)];
          float v1151_data = s1[80];
          float v1153_data = ir6[8];
          ir6[8] = (v1153_data + (v1150_data * v1151_data));
          #pragma unroll
          for (int32_t v1158_n0 = 0; v1158_n0 < 1; ++v1158_n0) {
            #pragma unroll
            for (int32_t v1159_n1 = 0; v1159_n1 < 9; ++v1159_n1) {
              int32_t v1160_a = v1158_n0 + v1159_n1;
              float v1161_data = ir6[v1160_a];
              r6[v1160_a] = v1161_data;
            }
          }
          // glb_m3 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v1166_i0 = 0; v1166_i0 < 1; ++v1166_i0) {
            int32_t v1174_lead = v17_lead + (v1166_i0 * 32);
            #pragma unroll
            for (int32_t v1167_i1 = 0; v1167_i1 < 9; ++v1167_i1) {
              float v1169_data = r6[(v1166_i0 + v1167_i1)];
              glb_m3[(v1174_lead + (v1167_i1 * 32))] = v1169_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

