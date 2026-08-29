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
          int32_t v13_lead = threadIdx.x % 16;
          if (v13_lead < 12) {
            int32_t v21_off = v13_lead + 4;
            #pragma unroll
            for (int32_t v15_i1 = 0; v15_i1 < 16; ++v15_i1) {
              float v24_data = __ldcg(&glb_m1[(v21_off + (v15_i1 * 32))]);
              r0[v15_i1] = v24_data;
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
          if (v13_lead < 12) {
            float v34_data = r0[0];
            float v35_data = s0[0];
            float v37_data = ir1[0];
            ir1[0] = (v37_data + (v34_data * v35_data));
            float v40_data = s0[16];
            float v42_data = ir1[1];
            ir1[1] = (v42_data + (v34_data * v40_data));
            float v45_data = s0[33];
            float v47_data = ir1[2];
            ir1[2] = (v47_data + (v34_data * v45_data));
            float v50_data = s0[49];
            float v52_data = ir1[3];
            ir1[3] = (v52_data + (v34_data * v50_data));
            float v55_data = s0[66];
            float v57_data = ir1[4];
            ir1[4] = (v57_data + (v34_data * v55_data));
            float v60_data = s0[82];
            float v62_data = ir1[5];
            ir1[5] = (v62_data + (v34_data * v60_data));
            float v65_data = s0[99];
            float v67_data = ir1[6];
            ir1[6] = (v67_data + (v34_data * v65_data));
            float v70_data = s0[115];
            float v72_data = ir1[7];
            ir1[7] = (v72_data + (v34_data * v70_data));
          }
          if (v13_lead < 12) {
            float v78_data = r0[1];
            float v79_data = s0[1];
            float v81_data = ir1[0];
            ir1[0] = (v81_data + (v78_data * v79_data));
            float v84_data = s0[17];
            float v86_data = ir1[1];
            ir1[1] = (v86_data + (v78_data * v84_data));
            float v89_data = s0[32];
            float v91_data = ir1[2];
            ir1[2] = (v91_data + (v78_data * v89_data));
            float v94_data = s0[48];
            float v96_data = ir1[3];
            ir1[3] = (v96_data + (v78_data * v94_data));
            float v99_data = s0[67];
            float v101_data = ir1[4];
            ir1[4] = (v101_data + (v78_data * v99_data));
            float v104_data = s0[83];
            float v106_data = ir1[5];
            ir1[5] = (v106_data + (v78_data * v104_data));
            float v109_data = s0[98];
            float v111_data = ir1[6];
            ir1[6] = (v111_data + (v78_data * v109_data));
            float v114_data = s0[114];
            float v116_data = ir1[7];
            ir1[7] = (v116_data + (v78_data * v114_data));
          }
          if (v13_lead < 12) {
            float v122_data = r0[2];
            float v123_data = s0[2];
            float v125_data = ir1[0];
            ir1[0] = (v125_data + (v122_data * v123_data));
            float v128_data = s0[18];
            float v130_data = ir1[1];
            ir1[1] = (v130_data + (v122_data * v128_data));
            float v133_data = s0[35];
            float v135_data = ir1[2];
            ir1[2] = (v135_data + (v122_data * v133_data));
            float v138_data = s0[51];
            float v140_data = ir1[3];
            ir1[3] = (v140_data + (v122_data * v138_data));
            float v143_data = s0[64];
            float v145_data = ir1[4];
            ir1[4] = (v145_data + (v122_data * v143_data));
            float v148_data = s0[80];
            float v150_data = ir1[5];
            ir1[5] = (v150_data + (v122_data * v148_data));
            float v153_data = s0[97];
            float v155_data = ir1[6];
            ir1[6] = (v155_data + (v122_data * v153_data));
            float v158_data = s0[113];
            float v160_data = ir1[7];
            ir1[7] = (v160_data + (v122_data * v158_data));
          }
          if (v13_lead < 12) {
            float v166_data = r0[3];
            float v167_data = s0[3];
            float v169_data = ir1[0];
            ir1[0] = (v169_data + (v166_data * v167_data));
            float v172_data = s0[19];
            float v174_data = ir1[1];
            ir1[1] = (v174_data + (v166_data * v172_data));
            float v177_data = s0[34];
            float v179_data = ir1[2];
            ir1[2] = (v179_data + (v166_data * v177_data));
            float v182_data = s0[50];
            float v184_data = ir1[3];
            ir1[3] = (v184_data + (v166_data * v182_data));
            float v187_data = s0[65];
            float v189_data = ir1[4];
            ir1[4] = (v189_data + (v166_data * v187_data));
            float v192_data = s0[81];
            float v194_data = ir1[5];
            ir1[5] = (v194_data + (v166_data * v192_data));
            float v197_data = s0[96];
            float v199_data = ir1[6];
            ir1[6] = (v199_data + (v166_data * v197_data));
            float v202_data = s0[112];
            float v204_data = ir1[7];
            ir1[7] = (v204_data + (v166_data * v202_data));
          }
          if (v13_lead < 12) {
            float v210_data = r0[4];
            float v211_data = s0[4];
            float v213_data = ir1[0];
            ir1[0] = (v213_data + (v210_data * v211_data));
            float v216_data = s0[20];
            float v218_data = ir1[1];
            ir1[1] = (v218_data + (v210_data * v216_data));
            float v221_data = s0[37];
            float v223_data = ir1[2];
            ir1[2] = (v223_data + (v210_data * v221_data));
            float v226_data = s0[53];
            float v228_data = ir1[3];
            ir1[3] = (v228_data + (v210_data * v226_data));
            float v231_data = s0[70];
            float v233_data = ir1[4];
            ir1[4] = (v233_data + (v210_data * v231_data));
            float v236_data = s0[86];
            float v238_data = ir1[5];
            ir1[5] = (v238_data + (v210_data * v236_data));
            float v241_data = s0[103];
            float v243_data = ir1[6];
            ir1[6] = (v243_data + (v210_data * v241_data));
            float v246_data = s0[119];
            float v248_data = ir1[7];
            ir1[7] = (v248_data + (v210_data * v246_data));
          }
          if (v13_lead < 12) {
            float v254_data = r0[5];
            float v255_data = s0[5];
            float v257_data = ir1[0];
            ir1[0] = (v257_data + (v254_data * v255_data));
            float v260_data = s0[21];
            float v262_data = ir1[1];
            ir1[1] = (v262_data + (v254_data * v260_data));
            float v265_data = s0[36];
            float v267_data = ir1[2];
            ir1[2] = (v267_data + (v254_data * v265_data));
            float v270_data = s0[52];
            float v272_data = ir1[3];
            ir1[3] = (v272_data + (v254_data * v270_data));
            float v275_data = s0[71];
            float v277_data = ir1[4];
            ir1[4] = (v277_data + (v254_data * v275_data));
            float v280_data = s0[87];
            float v282_data = ir1[5];
            ir1[5] = (v282_data + (v254_data * v280_data));
            float v285_data = s0[102];
            float v287_data = ir1[6];
            ir1[6] = (v287_data + (v254_data * v285_data));
            float v290_data = s0[118];
            float v292_data = ir1[7];
            ir1[7] = (v292_data + (v254_data * v290_data));
          }
          if (v13_lead < 12) {
            float v298_data = r0[6];
            float v299_data = s0[6];
            float v301_data = ir1[0];
            ir1[0] = (v301_data + (v298_data * v299_data));
            float v304_data = s0[22];
            float v306_data = ir1[1];
            ir1[1] = (v306_data + (v298_data * v304_data));
            float v309_data = s0[39];
            float v311_data = ir1[2];
            ir1[2] = (v311_data + (v298_data * v309_data));
            float v314_data = s0[55];
            float v316_data = ir1[3];
            ir1[3] = (v316_data + (v298_data * v314_data));
            float v319_data = s0[68];
            float v321_data = ir1[4];
            ir1[4] = (v321_data + (v298_data * v319_data));
            float v324_data = s0[84];
            float v326_data = ir1[5];
            ir1[5] = (v326_data + (v298_data * v324_data));
            float v329_data = s0[101];
            float v331_data = ir1[6];
            ir1[6] = (v331_data + (v298_data * v329_data));
            float v334_data = s0[117];
            float v336_data = ir1[7];
            ir1[7] = (v336_data + (v298_data * v334_data));
          }
          if (v13_lead < 12) {
            float v342_data = r0[7];
            float v343_data = s0[7];
            float v345_data = ir1[0];
            ir1[0] = (v345_data + (v342_data * v343_data));
            float v348_data = s0[23];
            float v350_data = ir1[1];
            ir1[1] = (v350_data + (v342_data * v348_data));
            float v353_data = s0[38];
            float v355_data = ir1[2];
            ir1[2] = (v355_data + (v342_data * v353_data));
            float v358_data = s0[54];
            float v360_data = ir1[3];
            ir1[3] = (v360_data + (v342_data * v358_data));
            float v363_data = s0[69];
            float v365_data = ir1[4];
            ir1[4] = (v365_data + (v342_data * v363_data));
            float v368_data = s0[85];
            float v370_data = ir1[5];
            ir1[5] = (v370_data + (v342_data * v368_data));
            float v373_data = s0[100];
            float v375_data = ir1[6];
            ir1[6] = (v375_data + (v342_data * v373_data));
            float v378_data = s0[116];
            float v380_data = ir1[7];
            ir1[7] = (v380_data + (v342_data * v378_data));
          }
          if (v13_lead < 12) {
            float v386_data = r0[8];
            float v387_data = s0[8];
            float v389_data = ir1[0];
            ir1[0] = (v389_data + (v386_data * v387_data));
            float v392_data = s0[24];
            float v394_data = ir1[1];
            ir1[1] = (v394_data + (v386_data * v392_data));
            float v397_data = s0[41];
            float v399_data = ir1[2];
            ir1[2] = (v399_data + (v386_data * v397_data));
            float v402_data = s0[57];
            float v404_data = ir1[3];
            ir1[3] = (v404_data + (v386_data * v402_data));
            float v407_data = s0[74];
            float v409_data = ir1[4];
            ir1[4] = (v409_data + (v386_data * v407_data));
            float v412_data = s0[90];
            float v414_data = ir1[5];
            ir1[5] = (v414_data + (v386_data * v412_data));
            float v417_data = s0[107];
            float v419_data = ir1[6];
            ir1[6] = (v419_data + (v386_data * v417_data));
            float v422_data = s0[123];
            float v424_data = ir1[7];
            ir1[7] = (v424_data + (v386_data * v422_data));
          }
          if (v13_lead < 12) {
            float v430_data = r0[9];
            float v431_data = s0[9];
            float v433_data = ir1[0];
            ir1[0] = (v433_data + (v430_data * v431_data));
            float v436_data = s0[25];
            float v438_data = ir1[1];
            ir1[1] = (v438_data + (v430_data * v436_data));
            float v441_data = s0[40];
            float v443_data = ir1[2];
            ir1[2] = (v443_data + (v430_data * v441_data));
            float v446_data = s0[56];
            float v448_data = ir1[3];
            ir1[3] = (v448_data + (v430_data * v446_data));
            float v451_data = s0[75];
            float v453_data = ir1[4];
            ir1[4] = (v453_data + (v430_data * v451_data));
            float v456_data = s0[91];
            float v458_data = ir1[5];
            ir1[5] = (v458_data + (v430_data * v456_data));
            float v461_data = s0[106];
            float v463_data = ir1[6];
            ir1[6] = (v463_data + (v430_data * v461_data));
            float v466_data = s0[122];
            float v468_data = ir1[7];
            ir1[7] = (v468_data + (v430_data * v466_data));
          }
          if (v13_lead < 12) {
            float v474_data = r0[10];
            float v475_data = s0[10];
            float v477_data = ir1[0];
            ir1[0] = (v477_data + (v474_data * v475_data));
            float v480_data = s0[26];
            float v482_data = ir1[1];
            ir1[1] = (v482_data + (v474_data * v480_data));
            float v485_data = s0[43];
            float v487_data = ir1[2];
            ir1[2] = (v487_data + (v474_data * v485_data));
            float v490_data = s0[59];
            float v492_data = ir1[3];
            ir1[3] = (v492_data + (v474_data * v490_data));
            float v495_data = s0[72];
            float v497_data = ir1[4];
            ir1[4] = (v497_data + (v474_data * v495_data));
            float v500_data = s0[88];
            float v502_data = ir1[5];
            ir1[5] = (v502_data + (v474_data * v500_data));
            float v505_data = s0[105];
            float v507_data = ir1[6];
            ir1[6] = (v507_data + (v474_data * v505_data));
            float v510_data = s0[121];
            float v512_data = ir1[7];
            ir1[7] = (v512_data + (v474_data * v510_data));
          }
          if (v13_lead < 12) {
            float v518_data = r0[11];
            float v519_data = s0[11];
            float v521_data = ir1[0];
            ir1[0] = (v521_data + (v518_data * v519_data));
            float v524_data = s0[27];
            float v526_data = ir1[1];
            ir1[1] = (v526_data + (v518_data * v524_data));
            float v529_data = s0[42];
            float v531_data = ir1[2];
            ir1[2] = (v531_data + (v518_data * v529_data));
            float v534_data = s0[58];
            float v536_data = ir1[3];
            ir1[3] = (v536_data + (v518_data * v534_data));
            float v539_data = s0[73];
            float v541_data = ir1[4];
            ir1[4] = (v541_data + (v518_data * v539_data));
            float v544_data = s0[89];
            float v546_data = ir1[5];
            ir1[5] = (v546_data + (v518_data * v544_data));
            float v549_data = s0[104];
            float v551_data = ir1[6];
            ir1[6] = (v551_data + (v518_data * v549_data));
            float v554_data = s0[120];
            float v556_data = ir1[7];
            ir1[7] = (v556_data + (v518_data * v554_data));
          }
          if (v13_lead < 12) {
            float v562_data = r0[12];
            float v563_data = s0[12];
            float v565_data = ir1[0];
            ir1[0] = (v565_data + (v562_data * v563_data));
            float v568_data = s0[28];
            float v570_data = ir1[1];
            ir1[1] = (v570_data + (v562_data * v568_data));
            float v573_data = s0[45];
            float v575_data = ir1[2];
            ir1[2] = (v575_data + (v562_data * v573_data));
            float v578_data = s0[61];
            float v580_data = ir1[3];
            ir1[3] = (v580_data + (v562_data * v578_data));
            float v583_data = s0[78];
            float v585_data = ir1[4];
            ir1[4] = (v585_data + (v562_data * v583_data));
            float v588_data = s0[94];
            float v590_data = ir1[5];
            ir1[5] = (v590_data + (v562_data * v588_data));
            float v593_data = s0[111];
            float v595_data = ir1[6];
            ir1[6] = (v595_data + (v562_data * v593_data));
            float v598_data = s0[127];
            float v600_data = ir1[7];
            ir1[7] = (v600_data + (v562_data * v598_data));
          }
          if (v13_lead < 12) {
            float v606_data = r0[13];
            float v607_data = s0[13];
            float v609_data = ir1[0];
            ir1[0] = (v609_data + (v606_data * v607_data));
            float v612_data = s0[29];
            float v614_data = ir1[1];
            ir1[1] = (v614_data + (v606_data * v612_data));
            float v617_data = s0[44];
            float v619_data = ir1[2];
            ir1[2] = (v619_data + (v606_data * v617_data));
            float v622_data = s0[60];
            float v624_data = ir1[3];
            ir1[3] = (v624_data + (v606_data * v622_data));
            float v627_data = s0[79];
            float v629_data = ir1[4];
            ir1[4] = (v629_data + (v606_data * v627_data));
            float v632_data = s0[95];
            float v634_data = ir1[5];
            ir1[5] = (v634_data + (v606_data * v632_data));
            float v637_data = s0[110];
            float v639_data = ir1[6];
            ir1[6] = (v639_data + (v606_data * v637_data));
            float v642_data = s0[126];
            float v644_data = ir1[7];
            ir1[7] = (v644_data + (v606_data * v642_data));
          }
          if (v13_lead < 12) {
            float v650_data = r0[14];
            float v651_data = s0[14];
            float v653_data = ir1[0];
            ir1[0] = (v653_data + (v650_data * v651_data));
            float v656_data = s0[30];
            float v658_data = ir1[1];
            ir1[1] = (v658_data + (v650_data * v656_data));
            float v661_data = s0[47];
            float v663_data = ir1[2];
            ir1[2] = (v663_data + (v650_data * v661_data));
            float v666_data = s0[63];
            float v668_data = ir1[3];
            ir1[3] = (v668_data + (v650_data * v666_data));
            float v671_data = s0[76];
            float v673_data = ir1[4];
            ir1[4] = (v673_data + (v650_data * v671_data));
            float v676_data = s0[92];
            float v678_data = ir1[5];
            ir1[5] = (v678_data + (v650_data * v676_data));
            float v681_data = s0[109];
            float v683_data = ir1[6];
            ir1[6] = (v683_data + (v650_data * v681_data));
            float v686_data = s0[125];
            float v688_data = ir1[7];
            ir1[7] = (v688_data + (v650_data * v686_data));
          }
          if (v13_lead < 12) {
            float v694_data = r0[15];
            float v695_data = s0[15];
            float v697_data = ir1[0];
            ir1[0] = (v697_data + (v694_data * v695_data));
            float v700_data = s0[31];
            float v702_data = ir1[1];
            ir1[1] = (v702_data + (v694_data * v700_data));
            float v705_data = s0[46];
            float v707_data = ir1[2];
            ir1[2] = (v707_data + (v694_data * v705_data));
            float v710_data = s0[62];
            float v712_data = ir1[3];
            ir1[3] = (v712_data + (v694_data * v710_data));
            float v715_data = s0[77];
            float v717_data = ir1[4];
            ir1[4] = (v717_data + (v694_data * v715_data));
            float v720_data = s0[93];
            float v722_data = ir1[5];
            ir1[5] = (v722_data + (v694_data * v720_data));
            float v725_data = s0[108];
            float v727_data = ir1[6];
            ir1[6] = (v727_data + (v694_data * v725_data));
            float v730_data = s0[124];
            float v732_data = ir1[7];
            ir1[7] = (v732_data + (v694_data * v730_data));
          }
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v738_n1 = 0; v738_n1 < 8; ++v738_n1) {
              float v740_data = ir1[v738_n1];
              r1[v738_n1] = v740_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v746_i1 = 0; v746_i1 < 8; ++v746_i1) {
              float v748_data = r1[v746_i1];
              glb_m0[(v13_lead + (v746_i1 * 12))] = v748_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

