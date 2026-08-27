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
          int32_t v3_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v4_i0 = 0; v4_i0 < 1; ++v4_i0) {
            int32_t v9_lead = v4_i0 * 32;
            int32_t v10_lead = v3_lead + v9_lead;
            int32_t v17_lead = v3_lead + v9_lead;
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 9; ++v5_i1) {
              int32_t v11_a = v5_i1 * 32;
              int32_t v12_a = v10_lead + v11_a;
              float v20_data = __ldcg(&glb_m0[(v17_lead + v11_a)]);
              int32_t v21_a = v4_i0 + v5_i1;
              r0[v21_a] = v20_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 9; ++v27_i1) {
              int32_t v33_a = v27_i1 * 16;
              int32_t v34_a = v3_lead + v33_a;
              float v42_data = __ldcg(&glb_m1[(v3_lead + v33_a)]);
              int32_t v43_a = 0 + v27_i1;
              r2[v43_a] = v42_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          auto& ir1 = r1;
          float v48_data = r0[0];
          float v49_data = ir1[0];
          ir1[0] = (v49_data + v48_data);
          float v51_data = r0[1];
          float v52_data = ir1[1];
          ir1[1] = (v52_data + v51_data);
          float v54_data = r0[2];
          float v55_data = ir1[2];
          ir1[2] = (v55_data + v54_data);
          float v57_data = r0[3];
          float v58_data = ir1[3];
          ir1[3] = (v58_data + v57_data);
          float v60_data = r0[4];
          float v61_data = ir1[4];
          ir1[4] = (v61_data + v60_data);
          float v63_data = r0[5];
          float v64_data = ir1[5];
          ir1[5] = (v64_data + v63_data);
          float v66_data = r0[6];
          float v67_data = ir1[6];
          ir1[6] = (v67_data + v66_data);
          float v69_data = r0[7];
          float v70_data = ir1[7];
          ir1[7] = (v70_data + v69_data);
          float v72_data = r0[8];
          float v73_data = ir1[8];
          ir1[8] = (v73_data + v72_data);
          float* __restrict__ s0 = &localShrMem0[96];
          // s0 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v79_i0 = 0; v79_i0 < 1; ++v79_i0) {
            int32_t v88_lead = v3_lead + (v79_i0 * 32);
            #pragma unroll
            for (int32_t v80_i1 = 0; v80_i1 < 9; ++v80_i1) {
              int32_t v81_a = v79_i0 + v80_i1;
              float v83_data = r1[(v79_i0 + v80_i1)];
              int32_t v90_a = v88_lead + (v80_i1 * 32);
              s0[v90_a] = v83_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v96_i1 = 0; v96_i1 < 9; ++v96_i1) {
              int32_t v102_a = v96_i1 * 16;
              int32_t v103_a = v3_lead + v102_a;
              float v111_data = __ldcg(&glb_m2[(v3_lead + v102_a)]);
              int32_t v112_a = 0 + v96_i1;
              r4[v112_a] = v111_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          __syncwarp();
          // r3 = +(r2) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir3[9]{};
          if (v3_lead < 16) {
            float v119_data = r2[0];
            float v120_data = ir3[0];
            ir3[0] = (v120_data + v119_data);
            float v122_data = r2[1];
            float v123_data = ir3[1];
            ir3[1] = (v123_data + v122_data);
            float v125_data = r2[2];
            float v126_data = ir3[2];
            ir3[2] = (v126_data + v125_data);
            float v128_data = r2[3];
            float v129_data = ir3[3];
            ir3[3] = (v129_data + v128_data);
            float v131_data = r2[4];
            float v132_data = ir3[4];
            ir3[4] = (v132_data + v131_data);
            float v134_data = r2[5];
            float v135_data = ir3[5];
            ir3[5] = (v135_data + v134_data);
            float v137_data = r2[6];
            float v138_data = ir3[6];
            ir3[6] = (v138_data + v137_data);
            float v140_data = r2[7];
            float v141_data = ir3[7];
            ir3[7] = (v141_data + v140_data);
            float v143_data = r2[8];
            float v144_data = ir3[8];
            ir3[8] = (v144_data + v143_data);
          }
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v150_n1 = 0; v150_n1 < 9; ++v150_n1) {
              int32_t v151_a = 0 + v150_n1;
              float v153_data = ir3[v150_n1];
              int32_t v159_a = v150_n1 * 32;
              int32_t v160_a = v3_lead + v159_a;
              float v168_data = s0[(v3_lead + v159_a)];
              int32_t v170_a = 0 + v150_n1;
              r3[v150_n1] = (v168_data + v153_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r3);
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v176_i1 = 0; v176_i1 < 9; ++v176_i1) {
              int32_t v177_a = 0 + v176_i1;
              float v179_data = r3[v176_i1];
              int32_t v186_a = v3_lead + (v176_i1 * 32);
              s0[v186_a] = v179_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          __syncwarp();
          // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir5[9]{};
          if (v3_lead < 16) {
            float v193_data = r4[0];
            float v194_data = ir5[0];
            ir5[0] = (v194_data + v193_data);
            float v196_data = r4[1];
            float v197_data = ir5[1];
            ir5[1] = (v197_data + v196_data);
            float v199_data = r4[2];
            float v200_data = ir5[2];
            ir5[2] = (v200_data + v199_data);
            float v202_data = r4[3];
            float v203_data = ir5[3];
            ir5[3] = (v203_data + v202_data);
            float v205_data = r4[4];
            float v206_data = ir5[4];
            ir5[4] = (v206_data + v205_data);
            float v208_data = r4[5];
            float v209_data = ir5[5];
            ir5[5] = (v209_data + v208_data);
            float v211_data = r4[6];
            float v212_data = ir5[6];
            ir5[6] = (v212_data + v211_data);
            float v214_data = r4[7];
            float v215_data = ir5[7];
            ir5[7] = (v215_data + v214_data);
            float v217_data = r4[8];
            float v218_data = ir5[8];
            ir5[8] = (v218_data + v217_data);
          }
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v224_n1 = 0; v224_n1 < 9; ++v224_n1) {
              int32_t v225_a = 0 + v224_n1;
              float v227_data = ir5[v224_n1];
              int32_t v233_a = v224_n1 * 32;
              int32_t v234_a = v3_lead + v233_a;
              float v242_data = s0[(v3_lead + v233_a)];
              int32_t v244_a = 0 + v224_n1;
              r5[v224_n1] = (v242_data + v227_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r5);
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v250_i1 = 0; v250_i1 < 9; ++v250_i1) {
              int32_t v251_a = 0 + v250_i1;
              float v253_data = r5[v250_i1];
              int32_t v260_a = v3_lead + (v250_i1 * 32);
              s0[v260_a] = v253_data;
            }
          }
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = load{g>s}(glb_m4[0, 1])
          pipeline.producer_acquire();
          cuda::memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 0], &glb_m4[0 + 0 + 1 * threadIdx.x + 0], cuda::aligned_size_t<4>(4), pipeline);
          cuda::memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 32], &glb_m4[0 + 0 + 1 * threadIdx.x + 32], cuda::aligned_size_t<4>(4), pipeline);
          if (threadIdx.x < 17) {
            cuda::memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 64], &glb_m4[0 + 0 + 1 * threadIdx.x + 64], cuda::aligned_size_t<4>(4), pipeline);
          }
          __syncwarp();
          pipeline.producer_commit();
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r6[9]{};
          __syncwarp();
          // r6 = +(s0 * s1) + None
          // [(0, 32), (0, 9)] [(0, 9)]
          float ir6[9]{};
          int32_t v272_a = v3_lead + 0;
          float v279_data = s0[v3_lead];
          float v280_data = s1[0];
          float v282_data = ir6[0];
          ir6[0] = (v282_data + (v279_data * v280_data));
          int32_t v289_a = v3_lead + 0;
          float v296_data = s0[v3_lead];
          float v297_data = s1[9];
          float v299_data = ir6[1];
          ir6[1] = (v299_data + (v296_data * v297_data));
          int32_t v306_a = v3_lead + 0;
          float v313_data = s0[v3_lead];
          float v314_data = s1[18];
          float v316_data = ir6[2];
          ir6[2] = (v316_data + (v313_data * v314_data));
          int32_t v323_a = v3_lead + 0;
          float v330_data = s0[v3_lead];
          float v331_data = s1[27];
          float v333_data = ir6[3];
          ir6[3] = (v333_data + (v330_data * v331_data));
          int32_t v340_a = v3_lead + 0;
          float v347_data = s0[v3_lead];
          float v348_data = s1[36];
          float v350_data = ir6[4];
          ir6[4] = (v350_data + (v347_data * v348_data));
          int32_t v357_a = v3_lead + 0;
          float v364_data = s0[v3_lead];
          float v365_data = s1[45];
          float v367_data = ir6[5];
          ir6[5] = (v367_data + (v364_data * v365_data));
          int32_t v374_a = v3_lead + 0;
          float v381_data = s0[v3_lead];
          float v382_data = s1[54];
          float v384_data = ir6[6];
          ir6[6] = (v384_data + (v381_data * v382_data));
          int32_t v391_a = v3_lead + 0;
          float v398_data = s0[v3_lead];
          float v399_data = s1[63];
          float v401_data = ir6[7];
          ir6[7] = (v401_data + (v398_data * v399_data));
          int32_t v408_a = v3_lead + 0;
          float v415_data = s0[v3_lead];
          float v416_data = s1[72];
          float v418_data = ir6[8];
          ir6[8] = (v418_data + (v415_data * v416_data));
          int32_t v428_a = v3_lead + 32;
          float v435_data = s0[(v3_lead + 32)];
          float v436_data = s1[1];
          float v438_data = ir6[0];
          ir6[0] = (v438_data + (v435_data * v436_data));
          int32_t v445_a = v3_lead + 32;
          float v452_data = s0[(v3_lead + 32)];
          float v453_data = s1[10];
          float v455_data = ir6[1];
          ir6[1] = (v455_data + (v452_data * v453_data));
          int32_t v462_a = v3_lead + 32;
          float v469_data = s0[(v3_lead + 32)];
          float v470_data = s1[19];
          float v472_data = ir6[2];
          ir6[2] = (v472_data + (v469_data * v470_data));
          int32_t v479_a = v3_lead + 32;
          float v486_data = s0[(v3_lead + 32)];
          float v487_data = s1[28];
          float v489_data = ir6[3];
          ir6[3] = (v489_data + (v486_data * v487_data));
          int32_t v496_a = v3_lead + 32;
          float v503_data = s0[(v3_lead + 32)];
          float v504_data = s1[37];
          float v506_data = ir6[4];
          ir6[4] = (v506_data + (v503_data * v504_data));
          int32_t v513_a = v3_lead + 32;
          float v520_data = s0[(v3_lead + 32)];
          float v521_data = s1[46];
          float v523_data = ir6[5];
          ir6[5] = (v523_data + (v520_data * v521_data));
          int32_t v530_a = v3_lead + 32;
          float v537_data = s0[(v3_lead + 32)];
          float v538_data = s1[55];
          float v540_data = ir6[6];
          ir6[6] = (v540_data + (v537_data * v538_data));
          int32_t v547_a = v3_lead + 32;
          float v554_data = s0[(v3_lead + 32)];
          float v555_data = s1[64];
          float v557_data = ir6[7];
          ir6[7] = (v557_data + (v554_data * v555_data));
          int32_t v564_a = v3_lead + 32;
          float v571_data = s0[(v3_lead + 32)];
          float v572_data = s1[73];
          float v574_data = ir6[8];
          ir6[8] = (v574_data + (v571_data * v572_data));
          int32_t v584_a = v3_lead + 64;
          float v591_data = s0[(v3_lead + 64)];
          float v592_data = s1[2];
          float v594_data = ir6[0];
          ir6[0] = (v594_data + (v591_data * v592_data));
          int32_t v601_a = v3_lead + 64;
          float v608_data = s0[(v3_lead + 64)];
          float v609_data = s1[11];
          float v611_data = ir6[1];
          ir6[1] = (v611_data + (v608_data * v609_data));
          int32_t v618_a = v3_lead + 64;
          float v625_data = s0[(v3_lead + 64)];
          float v626_data = s1[20];
          float v628_data = ir6[2];
          ir6[2] = (v628_data + (v625_data * v626_data));
          int32_t v635_a = v3_lead + 64;
          float v642_data = s0[(v3_lead + 64)];
          float v643_data = s1[29];
          float v645_data = ir6[3];
          ir6[3] = (v645_data + (v642_data * v643_data));
          int32_t v652_a = v3_lead + 64;
          float v659_data = s0[(v3_lead + 64)];
          float v660_data = s1[38];
          float v662_data = ir6[4];
          ir6[4] = (v662_data + (v659_data * v660_data));
          int32_t v669_a = v3_lead + 64;
          float v676_data = s0[(v3_lead + 64)];
          float v677_data = s1[47];
          float v679_data = ir6[5];
          ir6[5] = (v679_data + (v676_data * v677_data));
          int32_t v686_a = v3_lead + 64;
          float v693_data = s0[(v3_lead + 64)];
          float v694_data = s1[56];
          float v696_data = ir6[6];
          ir6[6] = (v696_data + (v693_data * v694_data));
          int32_t v703_a = v3_lead + 64;
          float v710_data = s0[(v3_lead + 64)];
          float v711_data = s1[65];
          float v713_data = ir6[7];
          ir6[7] = (v713_data + (v710_data * v711_data));
          int32_t v720_a = v3_lead + 64;
          float v727_data = s0[(v3_lead + 64)];
          float v728_data = s1[74];
          float v730_data = ir6[8];
          ir6[8] = (v730_data + (v727_data * v728_data));
          int32_t v740_a = v3_lead + 96;
          float v747_data = s0[(v3_lead + 96)];
          float v748_data = s1[3];
          float v750_data = ir6[0];
          ir6[0] = (v750_data + (v747_data * v748_data));
          int32_t v757_a = v3_lead + 96;
          float v764_data = s0[(v3_lead + 96)];
          float v765_data = s1[12];
          float v767_data = ir6[1];
          ir6[1] = (v767_data + (v764_data * v765_data));
          int32_t v774_a = v3_lead + 96;
          float v781_data = s0[(v3_lead + 96)];
          float v782_data = s1[21];
          float v784_data = ir6[2];
          ir6[2] = (v784_data + (v781_data * v782_data));
          int32_t v791_a = v3_lead + 96;
          float v798_data = s0[(v3_lead + 96)];
          float v799_data = s1[30];
          float v801_data = ir6[3];
          ir6[3] = (v801_data + (v798_data * v799_data));
          int32_t v808_a = v3_lead + 96;
          float v815_data = s0[(v3_lead + 96)];
          float v816_data = s1[39];
          float v818_data = ir6[4];
          ir6[4] = (v818_data + (v815_data * v816_data));
          int32_t v825_a = v3_lead + 96;
          float v832_data = s0[(v3_lead + 96)];
          float v833_data = s1[48];
          float v835_data = ir6[5];
          ir6[5] = (v835_data + (v832_data * v833_data));
          int32_t v842_a = v3_lead + 96;
          float v849_data = s0[(v3_lead + 96)];
          float v850_data = s1[57];
          float v852_data = ir6[6];
          ir6[6] = (v852_data + (v849_data * v850_data));
          int32_t v859_a = v3_lead + 96;
          float v866_data = s0[(v3_lead + 96)];
          float v867_data = s1[66];
          float v869_data = ir6[7];
          ir6[7] = (v869_data + (v866_data * v867_data));
          int32_t v876_a = v3_lead + 96;
          float v883_data = s0[(v3_lead + 96)];
          float v884_data = s1[75];
          float v886_data = ir6[8];
          ir6[8] = (v886_data + (v883_data * v884_data));
          int32_t v896_a = v3_lead + 128;
          float v903_data = s0[(v3_lead + 128)];
          float v904_data = s1[4];
          float v906_data = ir6[0];
          ir6[0] = (v906_data + (v903_data * v904_data));
          int32_t v913_a = v3_lead + 128;
          float v920_data = s0[(v3_lead + 128)];
          float v921_data = s1[13];
          float v923_data = ir6[1];
          ir6[1] = (v923_data + (v920_data * v921_data));
          int32_t v930_a = v3_lead + 128;
          float v937_data = s0[(v3_lead + 128)];
          float v938_data = s1[22];
          float v940_data = ir6[2];
          ir6[2] = (v940_data + (v937_data * v938_data));
          int32_t v947_a = v3_lead + 128;
          float v954_data = s0[(v3_lead + 128)];
          float v955_data = s1[31];
          float v957_data = ir6[3];
          ir6[3] = (v957_data + (v954_data * v955_data));
          int32_t v964_a = v3_lead + 128;
          float v971_data = s0[(v3_lead + 128)];
          float v972_data = s1[40];
          float v974_data = ir6[4];
          ir6[4] = (v974_data + (v971_data * v972_data));
          int32_t v981_a = v3_lead + 128;
          float v988_data = s0[(v3_lead + 128)];
          float v989_data = s1[49];
          float v991_data = ir6[5];
          ir6[5] = (v991_data + (v988_data * v989_data));
          int32_t v998_a = v3_lead + 128;
          float v1005_data = s0[(v3_lead + 128)];
          float v1006_data = s1[58];
          float v1008_data = ir6[6];
          ir6[6] = (v1008_data + (v1005_data * v1006_data));
          int32_t v1015_a = v3_lead + 128;
          float v1022_data = s0[(v3_lead + 128)];
          float v1023_data = s1[67];
          float v1025_data = ir6[7];
          ir6[7] = (v1025_data + (v1022_data * v1023_data));
          int32_t v1032_a = v3_lead + 128;
          float v1039_data = s0[(v3_lead + 128)];
          float v1040_data = s1[76];
          float v1042_data = ir6[8];
          ir6[8] = (v1042_data + (v1039_data * v1040_data));
          int32_t v1052_a = v3_lead + 160;
          float v1059_data = s0[(v3_lead + 160)];
          float v1060_data = s1[5];
          float v1062_data = ir6[0];
          ir6[0] = (v1062_data + (v1059_data * v1060_data));
          int32_t v1069_a = v3_lead + 160;
          float v1076_data = s0[(v3_lead + 160)];
          float v1077_data = s1[14];
          float v1079_data = ir6[1];
          ir6[1] = (v1079_data + (v1076_data * v1077_data));
          int32_t v1086_a = v3_lead + 160;
          float v1093_data = s0[(v3_lead + 160)];
          float v1094_data = s1[23];
          float v1096_data = ir6[2];
          ir6[2] = (v1096_data + (v1093_data * v1094_data));
          int32_t v1103_a = v3_lead + 160;
          float v1110_data = s0[(v3_lead + 160)];
          float v1111_data = s1[32];
          float v1113_data = ir6[3];
          ir6[3] = (v1113_data + (v1110_data * v1111_data));
          int32_t v1120_a = v3_lead + 160;
          float v1127_data = s0[(v3_lead + 160)];
          float v1128_data = s1[41];
          float v1130_data = ir6[4];
          ir6[4] = (v1130_data + (v1127_data * v1128_data));
          int32_t v1137_a = v3_lead + 160;
          float v1144_data = s0[(v3_lead + 160)];
          float v1145_data = s1[50];
          float v1147_data = ir6[5];
          ir6[5] = (v1147_data + (v1144_data * v1145_data));
          int32_t v1154_a = v3_lead + 160;
          float v1161_data = s0[(v3_lead + 160)];
          float v1162_data = s1[59];
          float v1164_data = ir6[6];
          ir6[6] = (v1164_data + (v1161_data * v1162_data));
          int32_t v1171_a = v3_lead + 160;
          float v1178_data = s0[(v3_lead + 160)];
          float v1179_data = s1[68];
          float v1181_data = ir6[7];
          ir6[7] = (v1181_data + (v1178_data * v1179_data));
          int32_t v1188_a = v3_lead + 160;
          float v1195_data = s0[(v3_lead + 160)];
          float v1196_data = s1[77];
          float v1198_data = ir6[8];
          ir6[8] = (v1198_data + (v1195_data * v1196_data));
          int32_t v1208_a = v3_lead + 192;
          float v1215_data = s0[(v3_lead + 192)];
          float v1216_data = s1[6];
          float v1218_data = ir6[0];
          ir6[0] = (v1218_data + (v1215_data * v1216_data));
          int32_t v1225_a = v3_lead + 192;
          float v1232_data = s0[(v3_lead + 192)];
          float v1233_data = s1[15];
          float v1235_data = ir6[1];
          ir6[1] = (v1235_data + (v1232_data * v1233_data));
          int32_t v1242_a = v3_lead + 192;
          float v1249_data = s0[(v3_lead + 192)];
          float v1250_data = s1[24];
          float v1252_data = ir6[2];
          ir6[2] = (v1252_data + (v1249_data * v1250_data));
          int32_t v1259_a = v3_lead + 192;
          float v1266_data = s0[(v3_lead + 192)];
          float v1267_data = s1[33];
          float v1269_data = ir6[3];
          ir6[3] = (v1269_data + (v1266_data * v1267_data));
          int32_t v1276_a = v3_lead + 192;
          float v1283_data = s0[(v3_lead + 192)];
          float v1284_data = s1[42];
          float v1286_data = ir6[4];
          ir6[4] = (v1286_data + (v1283_data * v1284_data));
          int32_t v1293_a = v3_lead + 192;
          float v1300_data = s0[(v3_lead + 192)];
          float v1301_data = s1[51];
          float v1303_data = ir6[5];
          ir6[5] = (v1303_data + (v1300_data * v1301_data));
          int32_t v1310_a = v3_lead + 192;
          float v1317_data = s0[(v3_lead + 192)];
          float v1318_data = s1[60];
          float v1320_data = ir6[6];
          ir6[6] = (v1320_data + (v1317_data * v1318_data));
          int32_t v1327_a = v3_lead + 192;
          float v1334_data = s0[(v3_lead + 192)];
          float v1335_data = s1[69];
          float v1337_data = ir6[7];
          ir6[7] = (v1337_data + (v1334_data * v1335_data));
          int32_t v1344_a = v3_lead + 192;
          float v1351_data = s0[(v3_lead + 192)];
          float v1352_data = s1[78];
          float v1354_data = ir6[8];
          ir6[8] = (v1354_data + (v1351_data * v1352_data));
          int32_t v1364_a = v3_lead + 224;
          float v1371_data = s0[(v3_lead + 224)];
          float v1372_data = s1[7];
          float v1374_data = ir6[0];
          ir6[0] = (v1374_data + (v1371_data * v1372_data));
          int32_t v1381_a = v3_lead + 224;
          float v1388_data = s0[(v3_lead + 224)];
          float v1389_data = s1[16];
          float v1391_data = ir6[1];
          ir6[1] = (v1391_data + (v1388_data * v1389_data));
          int32_t v1398_a = v3_lead + 224;
          float v1405_data = s0[(v3_lead + 224)];
          float v1406_data = s1[25];
          float v1408_data = ir6[2];
          ir6[2] = (v1408_data + (v1405_data * v1406_data));
          int32_t v1415_a = v3_lead + 224;
          float v1422_data = s0[(v3_lead + 224)];
          float v1423_data = s1[34];
          float v1425_data = ir6[3];
          ir6[3] = (v1425_data + (v1422_data * v1423_data));
          int32_t v1432_a = v3_lead + 224;
          float v1439_data = s0[(v3_lead + 224)];
          float v1440_data = s1[43];
          float v1442_data = ir6[4];
          ir6[4] = (v1442_data + (v1439_data * v1440_data));
          int32_t v1449_a = v3_lead + 224;
          float v1456_data = s0[(v3_lead + 224)];
          float v1457_data = s1[52];
          float v1459_data = ir6[5];
          ir6[5] = (v1459_data + (v1456_data * v1457_data));
          int32_t v1466_a = v3_lead + 224;
          float v1473_data = s0[(v3_lead + 224)];
          float v1474_data = s1[61];
          float v1476_data = ir6[6];
          ir6[6] = (v1476_data + (v1473_data * v1474_data));
          int32_t v1483_a = v3_lead + 224;
          float v1490_data = s0[(v3_lead + 224)];
          float v1491_data = s1[70];
          float v1493_data = ir6[7];
          ir6[7] = (v1493_data + (v1490_data * v1491_data));
          int32_t v1500_a = v3_lead + 224;
          float v1507_data = s0[(v3_lead + 224)];
          float v1508_data = s1[79];
          float v1510_data = ir6[8];
          ir6[8] = (v1510_data + (v1507_data * v1508_data));
          int32_t v1520_a = v3_lead + 256;
          float v1527_data = s0[(v3_lead + 256)];
          float v1528_data = s1[8];
          float v1530_data = ir6[0];
          ir6[0] = (v1530_data + (v1527_data * v1528_data));
          int32_t v1537_a = v3_lead + 256;
          float v1544_data = s0[(v3_lead + 256)];
          float v1545_data = s1[17];
          float v1547_data = ir6[1];
          ir6[1] = (v1547_data + (v1544_data * v1545_data));
          int32_t v1554_a = v3_lead + 256;
          float v1561_data = s0[(v3_lead + 256)];
          float v1562_data = s1[26];
          float v1564_data = ir6[2];
          ir6[2] = (v1564_data + (v1561_data * v1562_data));
          int32_t v1571_a = v3_lead + 256;
          float v1578_data = s0[(v3_lead + 256)];
          float v1579_data = s1[35];
          float v1581_data = ir6[3];
          ir6[3] = (v1581_data + (v1578_data * v1579_data));
          int32_t v1588_a = v3_lead + 256;
          float v1595_data = s0[(v3_lead + 256)];
          float v1596_data = s1[44];
          float v1598_data = ir6[4];
          ir6[4] = (v1598_data + (v1595_data * v1596_data));
          int32_t v1605_a = v3_lead + 256;
          float v1612_data = s0[(v3_lead + 256)];
          float v1613_data = s1[53];
          float v1615_data = ir6[5];
          ir6[5] = (v1615_data + (v1612_data * v1613_data));
          int32_t v1622_a = v3_lead + 256;
          float v1629_data = s0[(v3_lead + 256)];
          float v1630_data = s1[62];
          float v1632_data = ir6[6];
          ir6[6] = (v1632_data + (v1629_data * v1630_data));
          int32_t v1639_a = v3_lead + 256;
          float v1646_data = s0[(v3_lead + 256)];
          float v1647_data = s1[71];
          float v1649_data = ir6[7];
          ir6[7] = (v1649_data + (v1646_data * v1647_data));
          int32_t v1656_a = v3_lead + 256;
          float v1663_data = s0[(v3_lead + 256)];
          float v1664_data = s1[80];
          float v1666_data = ir6[8];
          ir6[8] = (v1666_data + (v1663_data * v1664_data));
          #pragma unroll
          for (int32_t v1671_n0 = 0; v1671_n0 < 1; ++v1671_n0) {
            #pragma unroll
            for (int32_t v1672_n1 = 0; v1672_n1 < 9; ++v1672_n1) {
              int32_t v1673_a = v1671_n0 + v1672_n1;
              int32_t v1674_a = v1671_n0 + v1672_n1;
              float v1675_data = ir6[v1674_a];
              int32_t v1676_a = v1671_n0 + v1672_n1;
              r6[v1674_a] = v1675_data;
            }
          }
          // glb_m3 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v1681_i0 = 0; v1681_i0 < 1; ++v1681_i0) {
            int32_t v1690_lead = v3_lead + (v1681_i0 * 32);
            #pragma unroll
            for (int32_t v1682_i1 = 0; v1682_i1 < 9; ++v1682_i1) {
              int32_t v1683_a = v1681_i0 + v1682_i1;
              float v1685_data = r6[(v1681_i0 + v1682_i1)];
              int32_t v1692_a = v1690_lead + (v1682_i1 * 32);
              glb_m3[v1692_a] = v1685_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

