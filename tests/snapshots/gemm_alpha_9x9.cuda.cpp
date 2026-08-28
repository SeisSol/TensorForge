// === base name ===
kernel_08a27dccde

// === header ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_08a27dccde, block.x * block.y * block.z, 1792 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_08a27dccde, cudaFuncAttributeMaxDynamicSharedMemorySize, 1792 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_08a27dccde<<<grid,block,1792 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 9×9(9×9) {0..9}×{0..9} strided
    // m1 9×9(9×9) {0..9}×{0..9} strided
    // m2 9×9(9×9) {0..9}×{0..9} strided
    // m3 ()  scalar
    // m0 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, 1] = m1 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, -1]×m2 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[-1, 1]×m3 ()  scalar()[]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[112 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[96];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 81 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 81 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 81 + 0 + m2_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 16;
          if (v10_lead < 9) {
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 9; ++v12_i1) {
              float v20_data = __ldcg(&glb_m1[(v10_lead + (v12_i1 * 9))]);
              r0[v12_i1] = v20_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 5; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
            __pipeline_commit();
          }
          if (threadIdx.x < 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 80], &glb_m2[0 + 0 + 1 * threadIdx.x + 80], 4);
            __pipeline_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[9]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 9), (0, 9)] [(0, 9)]
          float ir1[9]{};
          if (v10_lead < 9) {
            float v31_data = r0[0];
            float v32_data = s0[0];
            float v34_data = ir1[0];
            ir1[0] = (v34_data + (v31_data * v32_data));
            float v37_data = s0[9];
            float v39_data = ir1[1];
            ir1[1] = (v39_data + (v31_data * v37_data));
            float v42_data = s0[18];
            float v44_data = ir1[2];
            ir1[2] = (v44_data + (v31_data * v42_data));
            float v47_data = s0[27];
            float v49_data = ir1[3];
            ir1[3] = (v49_data + (v31_data * v47_data));
            float v52_data = s0[36];
            float v54_data = ir1[4];
            ir1[4] = (v54_data + (v31_data * v52_data));
            float v57_data = s0[45];
            float v59_data = ir1[5];
            ir1[5] = (v59_data + (v31_data * v57_data));
            float v62_data = s0[54];
            float v64_data = ir1[6];
            ir1[6] = (v64_data + (v31_data * v62_data));
            float v67_data = s0[63];
            float v69_data = ir1[7];
            ir1[7] = (v69_data + (v31_data * v67_data));
            float v72_data = s0[72];
            float v74_data = ir1[8];
            ir1[8] = (v74_data + (v31_data * v72_data));
          }
          if (v10_lead < 9) {
            float v80_data = r0[1];
            float v81_data = s0[1];
            float v83_data = ir1[0];
            ir1[0] = (v83_data + (v80_data * v81_data));
            float v86_data = s0[10];
            float v88_data = ir1[1];
            ir1[1] = (v88_data + (v80_data * v86_data));
            float v91_data = s0[19];
            float v93_data = ir1[2];
            ir1[2] = (v93_data + (v80_data * v91_data));
            float v96_data = s0[28];
            float v98_data = ir1[3];
            ir1[3] = (v98_data + (v80_data * v96_data));
            float v101_data = s0[37];
            float v103_data = ir1[4];
            ir1[4] = (v103_data + (v80_data * v101_data));
            float v106_data = s0[46];
            float v108_data = ir1[5];
            ir1[5] = (v108_data + (v80_data * v106_data));
            float v111_data = s0[55];
            float v113_data = ir1[6];
            ir1[6] = (v113_data + (v80_data * v111_data));
            float v116_data = s0[64];
            float v118_data = ir1[7];
            ir1[7] = (v118_data + (v80_data * v116_data));
            float v121_data = s0[73];
            float v123_data = ir1[8];
            ir1[8] = (v123_data + (v80_data * v121_data));
          }
          if (v10_lead < 9) {
            float v129_data = r0[2];
            float v130_data = s0[2];
            float v132_data = ir1[0];
            ir1[0] = (v132_data + (v129_data * v130_data));
            float v135_data = s0[11];
            float v137_data = ir1[1];
            ir1[1] = (v137_data + (v129_data * v135_data));
            float v140_data = s0[20];
            float v142_data = ir1[2];
            ir1[2] = (v142_data + (v129_data * v140_data));
            float v145_data = s0[29];
            float v147_data = ir1[3];
            ir1[3] = (v147_data + (v129_data * v145_data));
            float v150_data = s0[38];
            float v152_data = ir1[4];
            ir1[4] = (v152_data + (v129_data * v150_data));
            float v155_data = s0[47];
            float v157_data = ir1[5];
            ir1[5] = (v157_data + (v129_data * v155_data));
            float v160_data = s0[56];
            float v162_data = ir1[6];
            ir1[6] = (v162_data + (v129_data * v160_data));
            float v165_data = s0[65];
            float v167_data = ir1[7];
            ir1[7] = (v167_data + (v129_data * v165_data));
            float v170_data = s0[74];
            float v172_data = ir1[8];
            ir1[8] = (v172_data + (v129_data * v170_data));
          }
          if (v10_lead < 9) {
            float v178_data = r0[3];
            float v179_data = s0[3];
            float v181_data = ir1[0];
            ir1[0] = (v181_data + (v178_data * v179_data));
            float v184_data = s0[12];
            float v186_data = ir1[1];
            ir1[1] = (v186_data + (v178_data * v184_data));
            float v189_data = s0[21];
            float v191_data = ir1[2];
            ir1[2] = (v191_data + (v178_data * v189_data));
            float v194_data = s0[30];
            float v196_data = ir1[3];
            ir1[3] = (v196_data + (v178_data * v194_data));
            float v199_data = s0[39];
            float v201_data = ir1[4];
            ir1[4] = (v201_data + (v178_data * v199_data));
            float v204_data = s0[48];
            float v206_data = ir1[5];
            ir1[5] = (v206_data + (v178_data * v204_data));
            float v209_data = s0[57];
            float v211_data = ir1[6];
            ir1[6] = (v211_data + (v178_data * v209_data));
            float v214_data = s0[66];
            float v216_data = ir1[7];
            ir1[7] = (v216_data + (v178_data * v214_data));
            float v219_data = s0[75];
            float v221_data = ir1[8];
            ir1[8] = (v221_data + (v178_data * v219_data));
          }
          if (v10_lead < 9) {
            float v227_data = r0[4];
            float v228_data = s0[4];
            float v230_data = ir1[0];
            ir1[0] = (v230_data + (v227_data * v228_data));
            float v233_data = s0[13];
            float v235_data = ir1[1];
            ir1[1] = (v235_data + (v227_data * v233_data));
            float v238_data = s0[22];
            float v240_data = ir1[2];
            ir1[2] = (v240_data + (v227_data * v238_data));
            float v243_data = s0[31];
            float v245_data = ir1[3];
            ir1[3] = (v245_data + (v227_data * v243_data));
            float v248_data = s0[40];
            float v250_data = ir1[4];
            ir1[4] = (v250_data + (v227_data * v248_data));
            float v253_data = s0[49];
            float v255_data = ir1[5];
            ir1[5] = (v255_data + (v227_data * v253_data));
            float v258_data = s0[58];
            float v260_data = ir1[6];
            ir1[6] = (v260_data + (v227_data * v258_data));
            float v263_data = s0[67];
            float v265_data = ir1[7];
            ir1[7] = (v265_data + (v227_data * v263_data));
            float v268_data = s0[76];
            float v270_data = ir1[8];
            ir1[8] = (v270_data + (v227_data * v268_data));
          }
          if (v10_lead < 9) {
            float v276_data = r0[5];
            float v277_data = s0[5];
            float v279_data = ir1[0];
            ir1[0] = (v279_data + (v276_data * v277_data));
            float v282_data = s0[14];
            float v284_data = ir1[1];
            ir1[1] = (v284_data + (v276_data * v282_data));
            float v287_data = s0[23];
            float v289_data = ir1[2];
            ir1[2] = (v289_data + (v276_data * v287_data));
            float v292_data = s0[32];
            float v294_data = ir1[3];
            ir1[3] = (v294_data + (v276_data * v292_data));
            float v297_data = s0[41];
            float v299_data = ir1[4];
            ir1[4] = (v299_data + (v276_data * v297_data));
            float v302_data = s0[50];
            float v304_data = ir1[5];
            ir1[5] = (v304_data + (v276_data * v302_data));
            float v307_data = s0[59];
            float v309_data = ir1[6];
            ir1[6] = (v309_data + (v276_data * v307_data));
            float v312_data = s0[68];
            float v314_data = ir1[7];
            ir1[7] = (v314_data + (v276_data * v312_data));
            float v317_data = s0[77];
            float v319_data = ir1[8];
            ir1[8] = (v319_data + (v276_data * v317_data));
          }
          if (v10_lead < 9) {
            float v325_data = r0[6];
            float v326_data = s0[6];
            float v328_data = ir1[0];
            ir1[0] = (v328_data + (v325_data * v326_data));
            float v331_data = s0[15];
            float v333_data = ir1[1];
            ir1[1] = (v333_data + (v325_data * v331_data));
            float v336_data = s0[24];
            float v338_data = ir1[2];
            ir1[2] = (v338_data + (v325_data * v336_data));
            float v341_data = s0[33];
            float v343_data = ir1[3];
            ir1[3] = (v343_data + (v325_data * v341_data));
            float v346_data = s0[42];
            float v348_data = ir1[4];
            ir1[4] = (v348_data + (v325_data * v346_data));
            float v351_data = s0[51];
            float v353_data = ir1[5];
            ir1[5] = (v353_data + (v325_data * v351_data));
            float v356_data = s0[60];
            float v358_data = ir1[6];
            ir1[6] = (v358_data + (v325_data * v356_data));
            float v361_data = s0[69];
            float v363_data = ir1[7];
            ir1[7] = (v363_data + (v325_data * v361_data));
            float v366_data = s0[78];
            float v368_data = ir1[8];
            ir1[8] = (v368_data + (v325_data * v366_data));
          }
          if (v10_lead < 9) {
            float v374_data = r0[7];
            float v375_data = s0[7];
            float v377_data = ir1[0];
            ir1[0] = (v377_data + (v374_data * v375_data));
            float v380_data = s0[16];
            float v382_data = ir1[1];
            ir1[1] = (v382_data + (v374_data * v380_data));
            float v385_data = s0[25];
            float v387_data = ir1[2];
            ir1[2] = (v387_data + (v374_data * v385_data));
            float v390_data = s0[34];
            float v392_data = ir1[3];
            ir1[3] = (v392_data + (v374_data * v390_data));
            float v395_data = s0[43];
            float v397_data = ir1[4];
            ir1[4] = (v397_data + (v374_data * v395_data));
            float v400_data = s0[52];
            float v402_data = ir1[5];
            ir1[5] = (v402_data + (v374_data * v400_data));
            float v405_data = s0[61];
            float v407_data = ir1[6];
            ir1[6] = (v407_data + (v374_data * v405_data));
            float v410_data = s0[70];
            float v412_data = ir1[7];
            ir1[7] = (v412_data + (v374_data * v410_data));
            float v415_data = s0[79];
            float v417_data = ir1[8];
            ir1[8] = (v417_data + (v374_data * v415_data));
          }
          if (v10_lead < 9) {
            float v423_data = r0[8];
            float v424_data = s0[8];
            float v426_data = ir1[0];
            ir1[0] = (v426_data + (v423_data * v424_data));
            float v429_data = s0[17];
            float v431_data = ir1[1];
            ir1[1] = (v431_data + (v423_data * v429_data));
            float v434_data = s0[26];
            float v436_data = ir1[2];
            ir1[2] = (v436_data + (v423_data * v434_data));
            float v439_data = s0[35];
            float v441_data = ir1[3];
            ir1[3] = (v441_data + (v423_data * v439_data));
            float v444_data = s0[44];
            float v446_data = ir1[4];
            ir1[4] = (v446_data + (v423_data * v444_data));
            float v449_data = s0[53];
            float v451_data = ir1[5];
            ir1[5] = (v451_data + (v423_data * v449_data));
            float v454_data = s0[62];
            float v456_data = ir1[6];
            ir1[6] = (v456_data + (v423_data * v454_data));
            float v459_data = s0[71];
            float v461_data = ir1[7];
            ir1[7] = (v461_data + (v423_data * v459_data));
            float v464_data = s0[80];
            float v466_data = ir1[8];
            ir1[8] = (v466_data + (v423_data * v464_data));
          }
          if (v10_lead < 9) {
            #pragma unroll
            for (int32_t v473_n1 = 0; v473_n1 < 9; ++v473_n1) {
              float v475_data = ir1[v473_n1];
              r1[v473_n1] = (v475_data * 13.0f);
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v10_lead < 9) {
            #pragma unroll
            for (int32_t v482_i1 = 0; v482_i1 < 9; ++v482_i1) {
              float v484_data = r1[v482_i1];
              glb_m0[(v10_lead + (v482_i1 * 9))] = v484_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

