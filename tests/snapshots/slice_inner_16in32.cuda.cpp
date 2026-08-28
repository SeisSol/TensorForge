// === base name ===
kernel_87f2838a59

// === header ===
void launcher_kernel_87f2838a59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_87f2838a59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_87f2838a59, block.x * block.y * block.z, 2304 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_87f2838a59, cudaFuncAttributeMaxDynamicSharedMemorySize, 2304 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_87f2838a59<<<grid,block,2304 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_87f2838a59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×8(16×8) {0..16}×{0..8} strided
    // m1 32×32(32×32) {0..32}×{0..32} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[0, 1] = m1 32×32(32×32) {0..32}×{0..32} strided({0..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 1024 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v7_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v8_i0 = 0; v8_i0 < 1; ++v8_i0) {
            int32_t v13_lead = v8_i0 * 16;
            int32_t v15_off = (v7_lead + v13_lead) + 8;
            int32_t v23_off = (v7_lead + v13_lead) + 8;
            #pragma unroll
            for (int32_t v9_i1 = 8; v9_i1 < 24; ++v9_i1) {
              int32_t v16_a = v9_i1 * 32;
              int32_t v17_a = v15_off + v16_a;
              float v26_data = __ldcg(&glb_m1[(v23_off + v16_a)]);
              r0[(v8_i0 + (v9_i1 - 8))] = v26_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            #pragma unroll
            for (int32_t i = 0; i < 8; i += 1) {
              __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
              __pipeline_commit();
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 16), (0, 8)] [(0, 16)]
          float ir1[8]{};
          float v36_data = r0[0];
          float v37_data = s0[0];
          float v39_data = ir1[0];
          ir1[0] = (v39_data + (v36_data * v37_data));
          float v42_data = s0[16];
          float v44_data = ir1[1];
          ir1[1] = (v44_data + (v36_data * v42_data));
          float v47_data = s0[32];
          float v49_data = ir1[2];
          ir1[2] = (v49_data + (v36_data * v47_data));
          float v52_data = s0[48];
          float v54_data = ir1[3];
          ir1[3] = (v54_data + (v36_data * v52_data));
          float v57_data = s0[64];
          float v59_data = ir1[4];
          ir1[4] = (v59_data + (v36_data * v57_data));
          float v62_data = s0[80];
          float v64_data = ir1[5];
          ir1[5] = (v64_data + (v36_data * v62_data));
          float v67_data = s0[96];
          float v69_data = ir1[6];
          ir1[6] = (v69_data + (v36_data * v67_data));
          float v72_data = s0[112];
          float v74_data = ir1[7];
          ir1[7] = (v74_data + (v36_data * v72_data));
          float v79_data = r0[1];
          float v80_data = s0[1];
          float v82_data = ir1[0];
          ir1[0] = (v82_data + (v79_data * v80_data));
          float v85_data = s0[17];
          float v87_data = ir1[1];
          ir1[1] = (v87_data + (v79_data * v85_data));
          float v90_data = s0[33];
          float v92_data = ir1[2];
          ir1[2] = (v92_data + (v79_data * v90_data));
          float v95_data = s0[49];
          float v97_data = ir1[3];
          ir1[3] = (v97_data + (v79_data * v95_data));
          float v100_data = s0[65];
          float v102_data = ir1[4];
          ir1[4] = (v102_data + (v79_data * v100_data));
          float v105_data = s0[81];
          float v107_data = ir1[5];
          ir1[5] = (v107_data + (v79_data * v105_data));
          float v110_data = s0[97];
          float v112_data = ir1[6];
          ir1[6] = (v112_data + (v79_data * v110_data));
          float v115_data = s0[113];
          float v117_data = ir1[7];
          ir1[7] = (v117_data + (v79_data * v115_data));
          float v122_data = r0[2];
          float v123_data = s0[2];
          float v125_data = ir1[0];
          ir1[0] = (v125_data + (v122_data * v123_data));
          float v128_data = s0[18];
          float v130_data = ir1[1];
          ir1[1] = (v130_data + (v122_data * v128_data));
          float v133_data = s0[34];
          float v135_data = ir1[2];
          ir1[2] = (v135_data + (v122_data * v133_data));
          float v138_data = s0[50];
          float v140_data = ir1[3];
          ir1[3] = (v140_data + (v122_data * v138_data));
          float v143_data = s0[66];
          float v145_data = ir1[4];
          ir1[4] = (v145_data + (v122_data * v143_data));
          float v148_data = s0[82];
          float v150_data = ir1[5];
          ir1[5] = (v150_data + (v122_data * v148_data));
          float v153_data = s0[98];
          float v155_data = ir1[6];
          ir1[6] = (v155_data + (v122_data * v153_data));
          float v158_data = s0[114];
          float v160_data = ir1[7];
          ir1[7] = (v160_data + (v122_data * v158_data));
          float v165_data = r0[3];
          float v166_data = s0[3];
          float v168_data = ir1[0];
          ir1[0] = (v168_data + (v165_data * v166_data));
          float v171_data = s0[19];
          float v173_data = ir1[1];
          ir1[1] = (v173_data + (v165_data * v171_data));
          float v176_data = s0[35];
          float v178_data = ir1[2];
          ir1[2] = (v178_data + (v165_data * v176_data));
          float v181_data = s0[51];
          float v183_data = ir1[3];
          ir1[3] = (v183_data + (v165_data * v181_data));
          float v186_data = s0[67];
          float v188_data = ir1[4];
          ir1[4] = (v188_data + (v165_data * v186_data));
          float v191_data = s0[83];
          float v193_data = ir1[5];
          ir1[5] = (v193_data + (v165_data * v191_data));
          float v196_data = s0[99];
          float v198_data = ir1[6];
          ir1[6] = (v198_data + (v165_data * v196_data));
          float v201_data = s0[115];
          float v203_data = ir1[7];
          ir1[7] = (v203_data + (v165_data * v201_data));
          float v208_data = r0[4];
          float v209_data = s0[4];
          float v211_data = ir1[0];
          ir1[0] = (v211_data + (v208_data * v209_data));
          float v214_data = s0[20];
          float v216_data = ir1[1];
          ir1[1] = (v216_data + (v208_data * v214_data));
          float v219_data = s0[36];
          float v221_data = ir1[2];
          ir1[2] = (v221_data + (v208_data * v219_data));
          float v224_data = s0[52];
          float v226_data = ir1[3];
          ir1[3] = (v226_data + (v208_data * v224_data));
          float v229_data = s0[68];
          float v231_data = ir1[4];
          ir1[4] = (v231_data + (v208_data * v229_data));
          float v234_data = s0[84];
          float v236_data = ir1[5];
          ir1[5] = (v236_data + (v208_data * v234_data));
          float v239_data = s0[100];
          float v241_data = ir1[6];
          ir1[6] = (v241_data + (v208_data * v239_data));
          float v244_data = s0[116];
          float v246_data = ir1[7];
          ir1[7] = (v246_data + (v208_data * v244_data));
          float v251_data = r0[5];
          float v252_data = s0[5];
          float v254_data = ir1[0];
          ir1[0] = (v254_data + (v251_data * v252_data));
          float v257_data = s0[21];
          float v259_data = ir1[1];
          ir1[1] = (v259_data + (v251_data * v257_data));
          float v262_data = s0[37];
          float v264_data = ir1[2];
          ir1[2] = (v264_data + (v251_data * v262_data));
          float v267_data = s0[53];
          float v269_data = ir1[3];
          ir1[3] = (v269_data + (v251_data * v267_data));
          float v272_data = s0[69];
          float v274_data = ir1[4];
          ir1[4] = (v274_data + (v251_data * v272_data));
          float v277_data = s0[85];
          float v279_data = ir1[5];
          ir1[5] = (v279_data + (v251_data * v277_data));
          float v282_data = s0[101];
          float v284_data = ir1[6];
          ir1[6] = (v284_data + (v251_data * v282_data));
          float v287_data = s0[117];
          float v289_data = ir1[7];
          ir1[7] = (v289_data + (v251_data * v287_data));
          float v294_data = r0[6];
          float v295_data = s0[6];
          float v297_data = ir1[0];
          ir1[0] = (v297_data + (v294_data * v295_data));
          float v300_data = s0[22];
          float v302_data = ir1[1];
          ir1[1] = (v302_data + (v294_data * v300_data));
          float v305_data = s0[38];
          float v307_data = ir1[2];
          ir1[2] = (v307_data + (v294_data * v305_data));
          float v310_data = s0[54];
          float v312_data = ir1[3];
          ir1[3] = (v312_data + (v294_data * v310_data));
          float v315_data = s0[70];
          float v317_data = ir1[4];
          ir1[4] = (v317_data + (v294_data * v315_data));
          float v320_data = s0[86];
          float v322_data = ir1[5];
          ir1[5] = (v322_data + (v294_data * v320_data));
          float v325_data = s0[102];
          float v327_data = ir1[6];
          ir1[6] = (v327_data + (v294_data * v325_data));
          float v330_data = s0[118];
          float v332_data = ir1[7];
          ir1[7] = (v332_data + (v294_data * v330_data));
          float v337_data = r0[7];
          float v338_data = s0[7];
          float v340_data = ir1[0];
          ir1[0] = (v340_data + (v337_data * v338_data));
          float v343_data = s0[23];
          float v345_data = ir1[1];
          ir1[1] = (v345_data + (v337_data * v343_data));
          float v348_data = s0[39];
          float v350_data = ir1[2];
          ir1[2] = (v350_data + (v337_data * v348_data));
          float v353_data = s0[55];
          float v355_data = ir1[3];
          ir1[3] = (v355_data + (v337_data * v353_data));
          float v358_data = s0[71];
          float v360_data = ir1[4];
          ir1[4] = (v360_data + (v337_data * v358_data));
          float v363_data = s0[87];
          float v365_data = ir1[5];
          ir1[5] = (v365_data + (v337_data * v363_data));
          float v368_data = s0[103];
          float v370_data = ir1[6];
          ir1[6] = (v370_data + (v337_data * v368_data));
          float v373_data = s0[119];
          float v375_data = ir1[7];
          ir1[7] = (v375_data + (v337_data * v373_data));
          float v380_data = r0[8];
          float v381_data = s0[8];
          float v383_data = ir1[0];
          ir1[0] = (v383_data + (v380_data * v381_data));
          float v386_data = s0[24];
          float v388_data = ir1[1];
          ir1[1] = (v388_data + (v380_data * v386_data));
          float v391_data = s0[40];
          float v393_data = ir1[2];
          ir1[2] = (v393_data + (v380_data * v391_data));
          float v396_data = s0[56];
          float v398_data = ir1[3];
          ir1[3] = (v398_data + (v380_data * v396_data));
          float v401_data = s0[72];
          float v403_data = ir1[4];
          ir1[4] = (v403_data + (v380_data * v401_data));
          float v406_data = s0[88];
          float v408_data = ir1[5];
          ir1[5] = (v408_data + (v380_data * v406_data));
          float v411_data = s0[104];
          float v413_data = ir1[6];
          ir1[6] = (v413_data + (v380_data * v411_data));
          float v416_data = s0[120];
          float v418_data = ir1[7];
          ir1[7] = (v418_data + (v380_data * v416_data));
          float v423_data = r0[9];
          float v424_data = s0[9];
          float v426_data = ir1[0];
          ir1[0] = (v426_data + (v423_data * v424_data));
          float v429_data = s0[25];
          float v431_data = ir1[1];
          ir1[1] = (v431_data + (v423_data * v429_data));
          float v434_data = s0[41];
          float v436_data = ir1[2];
          ir1[2] = (v436_data + (v423_data * v434_data));
          float v439_data = s0[57];
          float v441_data = ir1[3];
          ir1[3] = (v441_data + (v423_data * v439_data));
          float v444_data = s0[73];
          float v446_data = ir1[4];
          ir1[4] = (v446_data + (v423_data * v444_data));
          float v449_data = s0[89];
          float v451_data = ir1[5];
          ir1[5] = (v451_data + (v423_data * v449_data));
          float v454_data = s0[105];
          float v456_data = ir1[6];
          ir1[6] = (v456_data + (v423_data * v454_data));
          float v459_data = s0[121];
          float v461_data = ir1[7];
          ir1[7] = (v461_data + (v423_data * v459_data));
          float v466_data = r0[10];
          float v467_data = s0[10];
          float v469_data = ir1[0];
          ir1[0] = (v469_data + (v466_data * v467_data));
          float v472_data = s0[26];
          float v474_data = ir1[1];
          ir1[1] = (v474_data + (v466_data * v472_data));
          float v477_data = s0[42];
          float v479_data = ir1[2];
          ir1[2] = (v479_data + (v466_data * v477_data));
          float v482_data = s0[58];
          float v484_data = ir1[3];
          ir1[3] = (v484_data + (v466_data * v482_data));
          float v487_data = s0[74];
          float v489_data = ir1[4];
          ir1[4] = (v489_data + (v466_data * v487_data));
          float v492_data = s0[90];
          float v494_data = ir1[5];
          ir1[5] = (v494_data + (v466_data * v492_data));
          float v497_data = s0[106];
          float v499_data = ir1[6];
          ir1[6] = (v499_data + (v466_data * v497_data));
          float v502_data = s0[122];
          float v504_data = ir1[7];
          ir1[7] = (v504_data + (v466_data * v502_data));
          float v509_data = r0[11];
          float v510_data = s0[11];
          float v512_data = ir1[0];
          ir1[0] = (v512_data + (v509_data * v510_data));
          float v515_data = s0[27];
          float v517_data = ir1[1];
          ir1[1] = (v517_data + (v509_data * v515_data));
          float v520_data = s0[43];
          float v522_data = ir1[2];
          ir1[2] = (v522_data + (v509_data * v520_data));
          float v525_data = s0[59];
          float v527_data = ir1[3];
          ir1[3] = (v527_data + (v509_data * v525_data));
          float v530_data = s0[75];
          float v532_data = ir1[4];
          ir1[4] = (v532_data + (v509_data * v530_data));
          float v535_data = s0[91];
          float v537_data = ir1[5];
          ir1[5] = (v537_data + (v509_data * v535_data));
          float v540_data = s0[107];
          float v542_data = ir1[6];
          ir1[6] = (v542_data + (v509_data * v540_data));
          float v545_data = s0[123];
          float v547_data = ir1[7];
          ir1[7] = (v547_data + (v509_data * v545_data));
          float v552_data = r0[12];
          float v553_data = s0[12];
          float v555_data = ir1[0];
          ir1[0] = (v555_data + (v552_data * v553_data));
          float v558_data = s0[28];
          float v560_data = ir1[1];
          ir1[1] = (v560_data + (v552_data * v558_data));
          float v563_data = s0[44];
          float v565_data = ir1[2];
          ir1[2] = (v565_data + (v552_data * v563_data));
          float v568_data = s0[60];
          float v570_data = ir1[3];
          ir1[3] = (v570_data + (v552_data * v568_data));
          float v573_data = s0[76];
          float v575_data = ir1[4];
          ir1[4] = (v575_data + (v552_data * v573_data));
          float v578_data = s0[92];
          float v580_data = ir1[5];
          ir1[5] = (v580_data + (v552_data * v578_data));
          float v583_data = s0[108];
          float v585_data = ir1[6];
          ir1[6] = (v585_data + (v552_data * v583_data));
          float v588_data = s0[124];
          float v590_data = ir1[7];
          ir1[7] = (v590_data + (v552_data * v588_data));
          float v595_data = r0[13];
          float v596_data = s0[13];
          float v598_data = ir1[0];
          ir1[0] = (v598_data + (v595_data * v596_data));
          float v601_data = s0[29];
          float v603_data = ir1[1];
          ir1[1] = (v603_data + (v595_data * v601_data));
          float v606_data = s0[45];
          float v608_data = ir1[2];
          ir1[2] = (v608_data + (v595_data * v606_data));
          float v611_data = s0[61];
          float v613_data = ir1[3];
          ir1[3] = (v613_data + (v595_data * v611_data));
          float v616_data = s0[77];
          float v618_data = ir1[4];
          ir1[4] = (v618_data + (v595_data * v616_data));
          float v621_data = s0[93];
          float v623_data = ir1[5];
          ir1[5] = (v623_data + (v595_data * v621_data));
          float v626_data = s0[109];
          float v628_data = ir1[6];
          ir1[6] = (v628_data + (v595_data * v626_data));
          float v631_data = s0[125];
          float v633_data = ir1[7];
          ir1[7] = (v633_data + (v595_data * v631_data));
          float v638_data = r0[14];
          float v639_data = s0[14];
          float v641_data = ir1[0];
          ir1[0] = (v641_data + (v638_data * v639_data));
          float v644_data = s0[30];
          float v646_data = ir1[1];
          ir1[1] = (v646_data + (v638_data * v644_data));
          float v649_data = s0[46];
          float v651_data = ir1[2];
          ir1[2] = (v651_data + (v638_data * v649_data));
          float v654_data = s0[62];
          float v656_data = ir1[3];
          ir1[3] = (v656_data + (v638_data * v654_data));
          float v659_data = s0[78];
          float v661_data = ir1[4];
          ir1[4] = (v661_data + (v638_data * v659_data));
          float v664_data = s0[94];
          float v666_data = ir1[5];
          ir1[5] = (v666_data + (v638_data * v664_data));
          float v669_data = s0[110];
          float v671_data = ir1[6];
          ir1[6] = (v671_data + (v638_data * v669_data));
          float v674_data = s0[126];
          float v676_data = ir1[7];
          ir1[7] = (v676_data + (v638_data * v674_data));
          float v681_data = r0[15];
          float v682_data = s0[15];
          float v684_data = ir1[0];
          ir1[0] = (v684_data + (v681_data * v682_data));
          float v687_data = s0[31];
          float v689_data = ir1[1];
          ir1[1] = (v689_data + (v681_data * v687_data));
          float v692_data = s0[47];
          float v694_data = ir1[2];
          ir1[2] = (v694_data + (v681_data * v692_data));
          float v697_data = s0[63];
          float v699_data = ir1[3];
          ir1[3] = (v699_data + (v681_data * v697_data));
          float v702_data = s0[79];
          float v704_data = ir1[4];
          ir1[4] = (v704_data + (v681_data * v702_data));
          float v707_data = s0[95];
          float v709_data = ir1[5];
          ir1[5] = (v709_data + (v681_data * v707_data));
          float v712_data = s0[111];
          float v714_data = ir1[6];
          ir1[6] = (v714_data + (v681_data * v712_data));
          float v717_data = s0[127];
          float v719_data = ir1[7];
          ir1[7] = (v719_data + (v681_data * v717_data));
          #pragma unroll
          for (int32_t v724_n0 = 0; v724_n0 < 1; ++v724_n0) {
            #pragma unroll
            for (int32_t v725_n1 = 0; v725_n1 < 8; ++v725_n1) {
              int32_t v726_a = v724_n0 + v725_n1;
              int32_t v727_a = v724_n0 + v725_n1;
              float v728_data = ir1[v727_a];
              r1[v727_a] = v728_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v733_i0 = 0; v733_i0 < 1; ++v733_i0) {
            int32_t v742_lead = v7_lead + (v733_i0 * 16);
            #pragma unroll
            for (int32_t v734_i1 = 0; v734_i1 < 8; ++v734_i1) {
              int32_t v735_a = v733_i0 + v734_i1;
              float v737_data = r1[(v733_i0 + v734_i1)];
              glb_m0[(v742_lead + (v734_i1 * 16))] = v737_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

