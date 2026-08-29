// === base name ===
kernel_ead773dd51

// === header ===
void launcher_kernel_ead773dd51(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_ead773dd51(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_ead773dd51, block.x * block.y * block.z, 2304 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_ead773dd51, cudaFuncAttributeMaxDynamicSharedMemorySize, 2304 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_ead773dd51<<<grid,block,2304 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_ead773dd51(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 12×8(12×8) {0..12}×{0..8} strided
    // m1 32×16(32×16) {0..32}×{0..16} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 32×16(32×16) {0..32}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[144 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[128];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 512 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 16;
          if (v10_lead < 12) {
            int32_t v18_off = v10_lead + 4;
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 16; ++v12_i1) {
              float v21_data = __ldcg(&glb_m1[(v18_off + (v12_i1 * 32))]);
              r0[v12_i1] = v21_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
            __pipeline_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 8)] [(0, 16)]
          float ir1[8]{};
          if (v10_lead < 12) {
            float v31_data = r0[0];
            float v32_data = s0[0];
            float v34_data = ir1[0];
            ir1[0] = (v34_data + (v31_data * v32_data));
            float v37_data = s0[16];
            float v39_data = ir1[1];
            ir1[1] = (v39_data + (v31_data * v37_data));
            float v42_data = s0[33];
            float v44_data = ir1[2];
            ir1[2] = (v44_data + (v31_data * v42_data));
            float v47_data = s0[49];
            float v49_data = ir1[3];
            ir1[3] = (v49_data + (v31_data * v47_data));
            float v52_data = s0[66];
            float v54_data = ir1[4];
            ir1[4] = (v54_data + (v31_data * v52_data));
            float v57_data = s0[82];
            float v59_data = ir1[5];
            ir1[5] = (v59_data + (v31_data * v57_data));
            float v62_data = s0[99];
            float v64_data = ir1[6];
            ir1[6] = (v64_data + (v31_data * v62_data));
            float v67_data = s0[115];
            float v69_data = ir1[7];
            ir1[7] = (v69_data + (v31_data * v67_data));
          }
          if (v10_lead < 12) {
            float v75_data = r0[1];
            float v76_data = s0[1];
            float v78_data = ir1[0];
            ir1[0] = (v78_data + (v75_data * v76_data));
            float v81_data = s0[17];
            float v83_data = ir1[1];
            ir1[1] = (v83_data + (v75_data * v81_data));
            float v86_data = s0[32];
            float v88_data = ir1[2];
            ir1[2] = (v88_data + (v75_data * v86_data));
            float v91_data = s0[48];
            float v93_data = ir1[3];
            ir1[3] = (v93_data + (v75_data * v91_data));
            float v96_data = s0[67];
            float v98_data = ir1[4];
            ir1[4] = (v98_data + (v75_data * v96_data));
            float v101_data = s0[83];
            float v103_data = ir1[5];
            ir1[5] = (v103_data + (v75_data * v101_data));
            float v106_data = s0[98];
            float v108_data = ir1[6];
            ir1[6] = (v108_data + (v75_data * v106_data));
            float v111_data = s0[114];
            float v113_data = ir1[7];
            ir1[7] = (v113_data + (v75_data * v111_data));
          }
          if (v10_lead < 12) {
            float v119_data = r0[2];
            float v120_data = s0[2];
            float v122_data = ir1[0];
            ir1[0] = (v122_data + (v119_data * v120_data));
            float v125_data = s0[18];
            float v127_data = ir1[1];
            ir1[1] = (v127_data + (v119_data * v125_data));
            float v130_data = s0[35];
            float v132_data = ir1[2];
            ir1[2] = (v132_data + (v119_data * v130_data));
            float v135_data = s0[51];
            float v137_data = ir1[3];
            ir1[3] = (v137_data + (v119_data * v135_data));
            float v140_data = s0[64];
            float v142_data = ir1[4];
            ir1[4] = (v142_data + (v119_data * v140_data));
            float v145_data = s0[80];
            float v147_data = ir1[5];
            ir1[5] = (v147_data + (v119_data * v145_data));
            float v150_data = s0[97];
            float v152_data = ir1[6];
            ir1[6] = (v152_data + (v119_data * v150_data));
            float v155_data = s0[113];
            float v157_data = ir1[7];
            ir1[7] = (v157_data + (v119_data * v155_data));
          }
          if (v10_lead < 12) {
            float v163_data = r0[3];
            float v164_data = s0[3];
            float v166_data = ir1[0];
            ir1[0] = (v166_data + (v163_data * v164_data));
            float v169_data = s0[19];
            float v171_data = ir1[1];
            ir1[1] = (v171_data + (v163_data * v169_data));
            float v174_data = s0[34];
            float v176_data = ir1[2];
            ir1[2] = (v176_data + (v163_data * v174_data));
            float v179_data = s0[50];
            float v181_data = ir1[3];
            ir1[3] = (v181_data + (v163_data * v179_data));
            float v184_data = s0[65];
            float v186_data = ir1[4];
            ir1[4] = (v186_data + (v163_data * v184_data));
            float v189_data = s0[81];
            float v191_data = ir1[5];
            ir1[5] = (v191_data + (v163_data * v189_data));
            float v194_data = s0[96];
            float v196_data = ir1[6];
            ir1[6] = (v196_data + (v163_data * v194_data));
            float v199_data = s0[112];
            float v201_data = ir1[7];
            ir1[7] = (v201_data + (v163_data * v199_data));
          }
          if (v10_lead < 12) {
            float v207_data = r0[4];
            float v208_data = s0[4];
            float v210_data = ir1[0];
            ir1[0] = (v210_data + (v207_data * v208_data));
            float v213_data = s0[20];
            float v215_data = ir1[1];
            ir1[1] = (v215_data + (v207_data * v213_data));
            float v218_data = s0[37];
            float v220_data = ir1[2];
            ir1[2] = (v220_data + (v207_data * v218_data));
            float v223_data = s0[53];
            float v225_data = ir1[3];
            ir1[3] = (v225_data + (v207_data * v223_data));
            float v228_data = s0[70];
            float v230_data = ir1[4];
            ir1[4] = (v230_data + (v207_data * v228_data));
            float v233_data = s0[86];
            float v235_data = ir1[5];
            ir1[5] = (v235_data + (v207_data * v233_data));
            float v238_data = s0[103];
            float v240_data = ir1[6];
            ir1[6] = (v240_data + (v207_data * v238_data));
            float v243_data = s0[119];
            float v245_data = ir1[7];
            ir1[7] = (v245_data + (v207_data * v243_data));
          }
          if (v10_lead < 12) {
            float v251_data = r0[5];
            float v252_data = s0[5];
            float v254_data = ir1[0];
            ir1[0] = (v254_data + (v251_data * v252_data));
            float v257_data = s0[21];
            float v259_data = ir1[1];
            ir1[1] = (v259_data + (v251_data * v257_data));
            float v262_data = s0[36];
            float v264_data = ir1[2];
            ir1[2] = (v264_data + (v251_data * v262_data));
            float v267_data = s0[52];
            float v269_data = ir1[3];
            ir1[3] = (v269_data + (v251_data * v267_data));
            float v272_data = s0[71];
            float v274_data = ir1[4];
            ir1[4] = (v274_data + (v251_data * v272_data));
            float v277_data = s0[87];
            float v279_data = ir1[5];
            ir1[5] = (v279_data + (v251_data * v277_data));
            float v282_data = s0[102];
            float v284_data = ir1[6];
            ir1[6] = (v284_data + (v251_data * v282_data));
            float v287_data = s0[118];
            float v289_data = ir1[7];
            ir1[7] = (v289_data + (v251_data * v287_data));
          }
          if (v10_lead < 12) {
            float v295_data = r0[6];
            float v296_data = s0[6];
            float v298_data = ir1[0];
            ir1[0] = (v298_data + (v295_data * v296_data));
            float v301_data = s0[22];
            float v303_data = ir1[1];
            ir1[1] = (v303_data + (v295_data * v301_data));
            float v306_data = s0[39];
            float v308_data = ir1[2];
            ir1[2] = (v308_data + (v295_data * v306_data));
            float v311_data = s0[55];
            float v313_data = ir1[3];
            ir1[3] = (v313_data + (v295_data * v311_data));
            float v316_data = s0[68];
            float v318_data = ir1[4];
            ir1[4] = (v318_data + (v295_data * v316_data));
            float v321_data = s0[84];
            float v323_data = ir1[5];
            ir1[5] = (v323_data + (v295_data * v321_data));
            float v326_data = s0[101];
            float v328_data = ir1[6];
            ir1[6] = (v328_data + (v295_data * v326_data));
            float v331_data = s0[117];
            float v333_data = ir1[7];
            ir1[7] = (v333_data + (v295_data * v331_data));
          }
          if (v10_lead < 12) {
            float v339_data = r0[7];
            float v340_data = s0[7];
            float v342_data = ir1[0];
            ir1[0] = (v342_data + (v339_data * v340_data));
            float v345_data = s0[23];
            float v347_data = ir1[1];
            ir1[1] = (v347_data + (v339_data * v345_data));
            float v350_data = s0[38];
            float v352_data = ir1[2];
            ir1[2] = (v352_data + (v339_data * v350_data));
            float v355_data = s0[54];
            float v357_data = ir1[3];
            ir1[3] = (v357_data + (v339_data * v355_data));
            float v360_data = s0[69];
            float v362_data = ir1[4];
            ir1[4] = (v362_data + (v339_data * v360_data));
            float v365_data = s0[85];
            float v367_data = ir1[5];
            ir1[5] = (v367_data + (v339_data * v365_data));
            float v370_data = s0[100];
            float v372_data = ir1[6];
            ir1[6] = (v372_data + (v339_data * v370_data));
            float v375_data = s0[116];
            float v377_data = ir1[7];
            ir1[7] = (v377_data + (v339_data * v375_data));
          }
          if (v10_lead < 12) {
            float v383_data = r0[8];
            float v384_data = s0[8];
            float v386_data = ir1[0];
            ir1[0] = (v386_data + (v383_data * v384_data));
            float v389_data = s0[24];
            float v391_data = ir1[1];
            ir1[1] = (v391_data + (v383_data * v389_data));
            float v394_data = s0[41];
            float v396_data = ir1[2];
            ir1[2] = (v396_data + (v383_data * v394_data));
            float v399_data = s0[57];
            float v401_data = ir1[3];
            ir1[3] = (v401_data + (v383_data * v399_data));
            float v404_data = s0[74];
            float v406_data = ir1[4];
            ir1[4] = (v406_data + (v383_data * v404_data));
            float v409_data = s0[90];
            float v411_data = ir1[5];
            ir1[5] = (v411_data + (v383_data * v409_data));
            float v414_data = s0[107];
            float v416_data = ir1[6];
            ir1[6] = (v416_data + (v383_data * v414_data));
            float v419_data = s0[123];
            float v421_data = ir1[7];
            ir1[7] = (v421_data + (v383_data * v419_data));
          }
          if (v10_lead < 12) {
            float v427_data = r0[9];
            float v428_data = s0[9];
            float v430_data = ir1[0];
            ir1[0] = (v430_data + (v427_data * v428_data));
            float v433_data = s0[25];
            float v435_data = ir1[1];
            ir1[1] = (v435_data + (v427_data * v433_data));
            float v438_data = s0[40];
            float v440_data = ir1[2];
            ir1[2] = (v440_data + (v427_data * v438_data));
            float v443_data = s0[56];
            float v445_data = ir1[3];
            ir1[3] = (v445_data + (v427_data * v443_data));
            float v448_data = s0[75];
            float v450_data = ir1[4];
            ir1[4] = (v450_data + (v427_data * v448_data));
            float v453_data = s0[91];
            float v455_data = ir1[5];
            ir1[5] = (v455_data + (v427_data * v453_data));
            float v458_data = s0[106];
            float v460_data = ir1[6];
            ir1[6] = (v460_data + (v427_data * v458_data));
            float v463_data = s0[122];
            float v465_data = ir1[7];
            ir1[7] = (v465_data + (v427_data * v463_data));
          }
          if (v10_lead < 12) {
            float v471_data = r0[10];
            float v472_data = s0[10];
            float v474_data = ir1[0];
            ir1[0] = (v474_data + (v471_data * v472_data));
            float v477_data = s0[26];
            float v479_data = ir1[1];
            ir1[1] = (v479_data + (v471_data * v477_data));
            float v482_data = s0[43];
            float v484_data = ir1[2];
            ir1[2] = (v484_data + (v471_data * v482_data));
            float v487_data = s0[59];
            float v489_data = ir1[3];
            ir1[3] = (v489_data + (v471_data * v487_data));
            float v492_data = s0[72];
            float v494_data = ir1[4];
            ir1[4] = (v494_data + (v471_data * v492_data));
            float v497_data = s0[88];
            float v499_data = ir1[5];
            ir1[5] = (v499_data + (v471_data * v497_data));
            float v502_data = s0[105];
            float v504_data = ir1[6];
            ir1[6] = (v504_data + (v471_data * v502_data));
            float v507_data = s0[121];
            float v509_data = ir1[7];
            ir1[7] = (v509_data + (v471_data * v507_data));
          }
          if (v10_lead < 12) {
            float v515_data = r0[11];
            float v516_data = s0[11];
            float v518_data = ir1[0];
            ir1[0] = (v518_data + (v515_data * v516_data));
            float v521_data = s0[27];
            float v523_data = ir1[1];
            ir1[1] = (v523_data + (v515_data * v521_data));
            float v526_data = s0[42];
            float v528_data = ir1[2];
            ir1[2] = (v528_data + (v515_data * v526_data));
            float v531_data = s0[58];
            float v533_data = ir1[3];
            ir1[3] = (v533_data + (v515_data * v531_data));
            float v536_data = s0[73];
            float v538_data = ir1[4];
            ir1[4] = (v538_data + (v515_data * v536_data));
            float v541_data = s0[89];
            float v543_data = ir1[5];
            ir1[5] = (v543_data + (v515_data * v541_data));
            float v546_data = s0[104];
            float v548_data = ir1[6];
            ir1[6] = (v548_data + (v515_data * v546_data));
            float v551_data = s0[120];
            float v553_data = ir1[7];
            ir1[7] = (v553_data + (v515_data * v551_data));
          }
          if (v10_lead < 12) {
            float v559_data = r0[12];
            float v560_data = s0[12];
            float v562_data = ir1[0];
            ir1[0] = (v562_data + (v559_data * v560_data));
            float v565_data = s0[28];
            float v567_data = ir1[1];
            ir1[1] = (v567_data + (v559_data * v565_data));
            float v570_data = s0[45];
            float v572_data = ir1[2];
            ir1[2] = (v572_data + (v559_data * v570_data));
            float v575_data = s0[61];
            float v577_data = ir1[3];
            ir1[3] = (v577_data + (v559_data * v575_data));
            float v580_data = s0[78];
            float v582_data = ir1[4];
            ir1[4] = (v582_data + (v559_data * v580_data));
            float v585_data = s0[94];
            float v587_data = ir1[5];
            ir1[5] = (v587_data + (v559_data * v585_data));
            float v590_data = s0[111];
            float v592_data = ir1[6];
            ir1[6] = (v592_data + (v559_data * v590_data));
            float v595_data = s0[127];
            float v597_data = ir1[7];
            ir1[7] = (v597_data + (v559_data * v595_data));
          }
          if (v10_lead < 12) {
            float v603_data = r0[13];
            float v604_data = s0[13];
            float v606_data = ir1[0];
            ir1[0] = (v606_data + (v603_data * v604_data));
            float v609_data = s0[29];
            float v611_data = ir1[1];
            ir1[1] = (v611_data + (v603_data * v609_data));
            float v614_data = s0[44];
            float v616_data = ir1[2];
            ir1[2] = (v616_data + (v603_data * v614_data));
            float v619_data = s0[60];
            float v621_data = ir1[3];
            ir1[3] = (v621_data + (v603_data * v619_data));
            float v624_data = s0[79];
            float v626_data = ir1[4];
            ir1[4] = (v626_data + (v603_data * v624_data));
            float v629_data = s0[95];
            float v631_data = ir1[5];
            ir1[5] = (v631_data + (v603_data * v629_data));
            float v634_data = s0[110];
            float v636_data = ir1[6];
            ir1[6] = (v636_data + (v603_data * v634_data));
            float v639_data = s0[126];
            float v641_data = ir1[7];
            ir1[7] = (v641_data + (v603_data * v639_data));
          }
          if (v10_lead < 12) {
            float v647_data = r0[14];
            float v648_data = s0[14];
            float v650_data = ir1[0];
            ir1[0] = (v650_data + (v647_data * v648_data));
            float v653_data = s0[30];
            float v655_data = ir1[1];
            ir1[1] = (v655_data + (v647_data * v653_data));
            float v658_data = s0[47];
            float v660_data = ir1[2];
            ir1[2] = (v660_data + (v647_data * v658_data));
            float v663_data = s0[63];
            float v665_data = ir1[3];
            ir1[3] = (v665_data + (v647_data * v663_data));
            float v668_data = s0[76];
            float v670_data = ir1[4];
            ir1[4] = (v670_data + (v647_data * v668_data));
            float v673_data = s0[92];
            float v675_data = ir1[5];
            ir1[5] = (v675_data + (v647_data * v673_data));
            float v678_data = s0[109];
            float v680_data = ir1[6];
            ir1[6] = (v680_data + (v647_data * v678_data));
            float v683_data = s0[125];
            float v685_data = ir1[7];
            ir1[7] = (v685_data + (v647_data * v683_data));
          }
          if (v10_lead < 12) {
            float v691_data = r0[15];
            float v692_data = s0[15];
            float v694_data = ir1[0];
            ir1[0] = (v694_data + (v691_data * v692_data));
            float v697_data = s0[31];
            float v699_data = ir1[1];
            ir1[1] = (v699_data + (v691_data * v697_data));
            float v702_data = s0[46];
            float v704_data = ir1[2];
            ir1[2] = (v704_data + (v691_data * v702_data));
            float v707_data = s0[62];
            float v709_data = ir1[3];
            ir1[3] = (v709_data + (v691_data * v707_data));
            float v712_data = s0[77];
            float v714_data = ir1[4];
            ir1[4] = (v714_data + (v691_data * v712_data));
            float v717_data = s0[93];
            float v719_data = ir1[5];
            ir1[5] = (v719_data + (v691_data * v717_data));
            float v722_data = s0[108];
            float v724_data = ir1[6];
            ir1[6] = (v724_data + (v691_data * v722_data));
            float v727_data = s0[124];
            float v729_data = ir1[7];
            ir1[7] = (v729_data + (v691_data * v727_data));
          }
          if (v10_lead < 12) {
            #pragma unroll
            for (int32_t v735_n1 = 0; v735_n1 < 8; ++v735_n1) {
              float v737_data = ir1[v735_n1];
              r1[v735_n1] = v737_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v10_lead < 12) {
            #pragma unroll
            for (int32_t v743_i1 = 0; v743_i1 < 8; ++v743_i1) {
              float v745_data = r1[v743_i1];
              glb_m0[(v10_lead + (v743_i1 * 12))] = v745_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

