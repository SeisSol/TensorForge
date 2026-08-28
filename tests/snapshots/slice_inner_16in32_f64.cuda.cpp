// === base name ===
kernel_3d37ccf0b0

// === header ===
void launcher_kernel_3d37ccf0b0(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_3d37ccf0b0(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_3d37ccf0b0, block.x * block.y * block.z, 2304 * sizeof(double));
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
        cudaFuncSetAttribute(kernel_kernel_3d37ccf0b0, cudaFuncAttributeMaxDynamicSharedMemorySize, 2304 * sizeof(double));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_3d37ccf0b0<<<grid,block,2304 * sizeof(double),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_3d37ccf0b0(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[144 * threadIdx.y + 0];
      double* tempShrMem = &localShrMem0[128];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[batchId0 * 1024 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          double r0[16]{};
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
              double v26_data = __ldcg(&glb_m1[(v23_off + v16_a)]);
              r0[(v8_i0 + (v9_i1 - 8))] = v26_data;
            }
          }
          double* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            #pragma unroll
            for (int32_t i = 0; i < 8; i += 1) {
              __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 8);
              __pipeline_commit();
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          double r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 16), (0, 8)] [(0, 16)]
          double ir1[8]{};
          double v36_data = r0[0];
          double v37_data = s0[0];
          double v39_data = ir1[0];
          ir1[0] = (v39_data + (v36_data * v37_data));
          double v42_data = s0[16];
          double v44_data = ir1[1];
          ir1[1] = (v44_data + (v36_data * v42_data));
          double v47_data = s0[32];
          double v49_data = ir1[2];
          ir1[2] = (v49_data + (v36_data * v47_data));
          double v52_data = s0[48];
          double v54_data = ir1[3];
          ir1[3] = (v54_data + (v36_data * v52_data));
          double v57_data = s0[64];
          double v59_data = ir1[4];
          ir1[4] = (v59_data + (v36_data * v57_data));
          double v62_data = s0[80];
          double v64_data = ir1[5];
          ir1[5] = (v64_data + (v36_data * v62_data));
          double v67_data = s0[96];
          double v69_data = ir1[6];
          ir1[6] = (v69_data + (v36_data * v67_data));
          double v72_data = s0[112];
          double v74_data = ir1[7];
          ir1[7] = (v74_data + (v36_data * v72_data));
          double v79_data = r0[1];
          double v80_data = s0[1];
          double v82_data = ir1[0];
          ir1[0] = (v82_data + (v79_data * v80_data));
          double v85_data = s0[17];
          double v87_data = ir1[1];
          ir1[1] = (v87_data + (v79_data * v85_data));
          double v90_data = s0[33];
          double v92_data = ir1[2];
          ir1[2] = (v92_data + (v79_data * v90_data));
          double v95_data = s0[49];
          double v97_data = ir1[3];
          ir1[3] = (v97_data + (v79_data * v95_data));
          double v100_data = s0[65];
          double v102_data = ir1[4];
          ir1[4] = (v102_data + (v79_data * v100_data));
          double v105_data = s0[81];
          double v107_data = ir1[5];
          ir1[5] = (v107_data + (v79_data * v105_data));
          double v110_data = s0[97];
          double v112_data = ir1[6];
          ir1[6] = (v112_data + (v79_data * v110_data));
          double v115_data = s0[113];
          double v117_data = ir1[7];
          ir1[7] = (v117_data + (v79_data * v115_data));
          double v122_data = r0[2];
          double v123_data = s0[2];
          double v125_data = ir1[0];
          ir1[0] = (v125_data + (v122_data * v123_data));
          double v128_data = s0[18];
          double v130_data = ir1[1];
          ir1[1] = (v130_data + (v122_data * v128_data));
          double v133_data = s0[34];
          double v135_data = ir1[2];
          ir1[2] = (v135_data + (v122_data * v133_data));
          double v138_data = s0[50];
          double v140_data = ir1[3];
          ir1[3] = (v140_data + (v122_data * v138_data));
          double v143_data = s0[66];
          double v145_data = ir1[4];
          ir1[4] = (v145_data + (v122_data * v143_data));
          double v148_data = s0[82];
          double v150_data = ir1[5];
          ir1[5] = (v150_data + (v122_data * v148_data));
          double v153_data = s0[98];
          double v155_data = ir1[6];
          ir1[6] = (v155_data + (v122_data * v153_data));
          double v158_data = s0[114];
          double v160_data = ir1[7];
          ir1[7] = (v160_data + (v122_data * v158_data));
          double v165_data = r0[3];
          double v166_data = s0[3];
          double v168_data = ir1[0];
          ir1[0] = (v168_data + (v165_data * v166_data));
          double v171_data = s0[19];
          double v173_data = ir1[1];
          ir1[1] = (v173_data + (v165_data * v171_data));
          double v176_data = s0[35];
          double v178_data = ir1[2];
          ir1[2] = (v178_data + (v165_data * v176_data));
          double v181_data = s0[51];
          double v183_data = ir1[3];
          ir1[3] = (v183_data + (v165_data * v181_data));
          double v186_data = s0[67];
          double v188_data = ir1[4];
          ir1[4] = (v188_data + (v165_data * v186_data));
          double v191_data = s0[83];
          double v193_data = ir1[5];
          ir1[5] = (v193_data + (v165_data * v191_data));
          double v196_data = s0[99];
          double v198_data = ir1[6];
          ir1[6] = (v198_data + (v165_data * v196_data));
          double v201_data = s0[115];
          double v203_data = ir1[7];
          ir1[7] = (v203_data + (v165_data * v201_data));
          double v208_data = r0[4];
          double v209_data = s0[4];
          double v211_data = ir1[0];
          ir1[0] = (v211_data + (v208_data * v209_data));
          double v214_data = s0[20];
          double v216_data = ir1[1];
          ir1[1] = (v216_data + (v208_data * v214_data));
          double v219_data = s0[36];
          double v221_data = ir1[2];
          ir1[2] = (v221_data + (v208_data * v219_data));
          double v224_data = s0[52];
          double v226_data = ir1[3];
          ir1[3] = (v226_data + (v208_data * v224_data));
          double v229_data = s0[68];
          double v231_data = ir1[4];
          ir1[4] = (v231_data + (v208_data * v229_data));
          double v234_data = s0[84];
          double v236_data = ir1[5];
          ir1[5] = (v236_data + (v208_data * v234_data));
          double v239_data = s0[100];
          double v241_data = ir1[6];
          ir1[6] = (v241_data + (v208_data * v239_data));
          double v244_data = s0[116];
          double v246_data = ir1[7];
          ir1[7] = (v246_data + (v208_data * v244_data));
          double v251_data = r0[5];
          double v252_data = s0[5];
          double v254_data = ir1[0];
          ir1[0] = (v254_data + (v251_data * v252_data));
          double v257_data = s0[21];
          double v259_data = ir1[1];
          ir1[1] = (v259_data + (v251_data * v257_data));
          double v262_data = s0[37];
          double v264_data = ir1[2];
          ir1[2] = (v264_data + (v251_data * v262_data));
          double v267_data = s0[53];
          double v269_data = ir1[3];
          ir1[3] = (v269_data + (v251_data * v267_data));
          double v272_data = s0[69];
          double v274_data = ir1[4];
          ir1[4] = (v274_data + (v251_data * v272_data));
          double v277_data = s0[85];
          double v279_data = ir1[5];
          ir1[5] = (v279_data + (v251_data * v277_data));
          double v282_data = s0[101];
          double v284_data = ir1[6];
          ir1[6] = (v284_data + (v251_data * v282_data));
          double v287_data = s0[117];
          double v289_data = ir1[7];
          ir1[7] = (v289_data + (v251_data * v287_data));
          double v294_data = r0[6];
          double v295_data = s0[6];
          double v297_data = ir1[0];
          ir1[0] = (v297_data + (v294_data * v295_data));
          double v300_data = s0[22];
          double v302_data = ir1[1];
          ir1[1] = (v302_data + (v294_data * v300_data));
          double v305_data = s0[38];
          double v307_data = ir1[2];
          ir1[2] = (v307_data + (v294_data * v305_data));
          double v310_data = s0[54];
          double v312_data = ir1[3];
          ir1[3] = (v312_data + (v294_data * v310_data));
          double v315_data = s0[70];
          double v317_data = ir1[4];
          ir1[4] = (v317_data + (v294_data * v315_data));
          double v320_data = s0[86];
          double v322_data = ir1[5];
          ir1[5] = (v322_data + (v294_data * v320_data));
          double v325_data = s0[102];
          double v327_data = ir1[6];
          ir1[6] = (v327_data + (v294_data * v325_data));
          double v330_data = s0[118];
          double v332_data = ir1[7];
          ir1[7] = (v332_data + (v294_data * v330_data));
          double v337_data = r0[7];
          double v338_data = s0[7];
          double v340_data = ir1[0];
          ir1[0] = (v340_data + (v337_data * v338_data));
          double v343_data = s0[23];
          double v345_data = ir1[1];
          ir1[1] = (v345_data + (v337_data * v343_data));
          double v348_data = s0[39];
          double v350_data = ir1[2];
          ir1[2] = (v350_data + (v337_data * v348_data));
          double v353_data = s0[55];
          double v355_data = ir1[3];
          ir1[3] = (v355_data + (v337_data * v353_data));
          double v358_data = s0[71];
          double v360_data = ir1[4];
          ir1[4] = (v360_data + (v337_data * v358_data));
          double v363_data = s0[87];
          double v365_data = ir1[5];
          ir1[5] = (v365_data + (v337_data * v363_data));
          double v368_data = s0[103];
          double v370_data = ir1[6];
          ir1[6] = (v370_data + (v337_data * v368_data));
          double v373_data = s0[119];
          double v375_data = ir1[7];
          ir1[7] = (v375_data + (v337_data * v373_data));
          double v380_data = r0[8];
          double v381_data = s0[8];
          double v383_data = ir1[0];
          ir1[0] = (v383_data + (v380_data * v381_data));
          double v386_data = s0[24];
          double v388_data = ir1[1];
          ir1[1] = (v388_data + (v380_data * v386_data));
          double v391_data = s0[40];
          double v393_data = ir1[2];
          ir1[2] = (v393_data + (v380_data * v391_data));
          double v396_data = s0[56];
          double v398_data = ir1[3];
          ir1[3] = (v398_data + (v380_data * v396_data));
          double v401_data = s0[72];
          double v403_data = ir1[4];
          ir1[4] = (v403_data + (v380_data * v401_data));
          double v406_data = s0[88];
          double v408_data = ir1[5];
          ir1[5] = (v408_data + (v380_data * v406_data));
          double v411_data = s0[104];
          double v413_data = ir1[6];
          ir1[6] = (v413_data + (v380_data * v411_data));
          double v416_data = s0[120];
          double v418_data = ir1[7];
          ir1[7] = (v418_data + (v380_data * v416_data));
          double v423_data = r0[9];
          double v424_data = s0[9];
          double v426_data = ir1[0];
          ir1[0] = (v426_data + (v423_data * v424_data));
          double v429_data = s0[25];
          double v431_data = ir1[1];
          ir1[1] = (v431_data + (v423_data * v429_data));
          double v434_data = s0[41];
          double v436_data = ir1[2];
          ir1[2] = (v436_data + (v423_data * v434_data));
          double v439_data = s0[57];
          double v441_data = ir1[3];
          ir1[3] = (v441_data + (v423_data * v439_data));
          double v444_data = s0[73];
          double v446_data = ir1[4];
          ir1[4] = (v446_data + (v423_data * v444_data));
          double v449_data = s0[89];
          double v451_data = ir1[5];
          ir1[5] = (v451_data + (v423_data * v449_data));
          double v454_data = s0[105];
          double v456_data = ir1[6];
          ir1[6] = (v456_data + (v423_data * v454_data));
          double v459_data = s0[121];
          double v461_data = ir1[7];
          ir1[7] = (v461_data + (v423_data * v459_data));
          double v466_data = r0[10];
          double v467_data = s0[10];
          double v469_data = ir1[0];
          ir1[0] = (v469_data + (v466_data * v467_data));
          double v472_data = s0[26];
          double v474_data = ir1[1];
          ir1[1] = (v474_data + (v466_data * v472_data));
          double v477_data = s0[42];
          double v479_data = ir1[2];
          ir1[2] = (v479_data + (v466_data * v477_data));
          double v482_data = s0[58];
          double v484_data = ir1[3];
          ir1[3] = (v484_data + (v466_data * v482_data));
          double v487_data = s0[74];
          double v489_data = ir1[4];
          ir1[4] = (v489_data + (v466_data * v487_data));
          double v492_data = s0[90];
          double v494_data = ir1[5];
          ir1[5] = (v494_data + (v466_data * v492_data));
          double v497_data = s0[106];
          double v499_data = ir1[6];
          ir1[6] = (v499_data + (v466_data * v497_data));
          double v502_data = s0[122];
          double v504_data = ir1[7];
          ir1[7] = (v504_data + (v466_data * v502_data));
          double v509_data = r0[11];
          double v510_data = s0[11];
          double v512_data = ir1[0];
          ir1[0] = (v512_data + (v509_data * v510_data));
          double v515_data = s0[27];
          double v517_data = ir1[1];
          ir1[1] = (v517_data + (v509_data * v515_data));
          double v520_data = s0[43];
          double v522_data = ir1[2];
          ir1[2] = (v522_data + (v509_data * v520_data));
          double v525_data = s0[59];
          double v527_data = ir1[3];
          ir1[3] = (v527_data + (v509_data * v525_data));
          double v530_data = s0[75];
          double v532_data = ir1[4];
          ir1[4] = (v532_data + (v509_data * v530_data));
          double v535_data = s0[91];
          double v537_data = ir1[5];
          ir1[5] = (v537_data + (v509_data * v535_data));
          double v540_data = s0[107];
          double v542_data = ir1[6];
          ir1[6] = (v542_data + (v509_data * v540_data));
          double v545_data = s0[123];
          double v547_data = ir1[7];
          ir1[7] = (v547_data + (v509_data * v545_data));
          double v552_data = r0[12];
          double v553_data = s0[12];
          double v555_data = ir1[0];
          ir1[0] = (v555_data + (v552_data * v553_data));
          double v558_data = s0[28];
          double v560_data = ir1[1];
          ir1[1] = (v560_data + (v552_data * v558_data));
          double v563_data = s0[44];
          double v565_data = ir1[2];
          ir1[2] = (v565_data + (v552_data * v563_data));
          double v568_data = s0[60];
          double v570_data = ir1[3];
          ir1[3] = (v570_data + (v552_data * v568_data));
          double v573_data = s0[76];
          double v575_data = ir1[4];
          ir1[4] = (v575_data + (v552_data * v573_data));
          double v578_data = s0[92];
          double v580_data = ir1[5];
          ir1[5] = (v580_data + (v552_data * v578_data));
          double v583_data = s0[108];
          double v585_data = ir1[6];
          ir1[6] = (v585_data + (v552_data * v583_data));
          double v588_data = s0[124];
          double v590_data = ir1[7];
          ir1[7] = (v590_data + (v552_data * v588_data));
          double v595_data = r0[13];
          double v596_data = s0[13];
          double v598_data = ir1[0];
          ir1[0] = (v598_data + (v595_data * v596_data));
          double v601_data = s0[29];
          double v603_data = ir1[1];
          ir1[1] = (v603_data + (v595_data * v601_data));
          double v606_data = s0[45];
          double v608_data = ir1[2];
          ir1[2] = (v608_data + (v595_data * v606_data));
          double v611_data = s0[61];
          double v613_data = ir1[3];
          ir1[3] = (v613_data + (v595_data * v611_data));
          double v616_data = s0[77];
          double v618_data = ir1[4];
          ir1[4] = (v618_data + (v595_data * v616_data));
          double v621_data = s0[93];
          double v623_data = ir1[5];
          ir1[5] = (v623_data + (v595_data * v621_data));
          double v626_data = s0[109];
          double v628_data = ir1[6];
          ir1[6] = (v628_data + (v595_data * v626_data));
          double v631_data = s0[125];
          double v633_data = ir1[7];
          ir1[7] = (v633_data + (v595_data * v631_data));
          double v638_data = r0[14];
          double v639_data = s0[14];
          double v641_data = ir1[0];
          ir1[0] = (v641_data + (v638_data * v639_data));
          double v644_data = s0[30];
          double v646_data = ir1[1];
          ir1[1] = (v646_data + (v638_data * v644_data));
          double v649_data = s0[46];
          double v651_data = ir1[2];
          ir1[2] = (v651_data + (v638_data * v649_data));
          double v654_data = s0[62];
          double v656_data = ir1[3];
          ir1[3] = (v656_data + (v638_data * v654_data));
          double v659_data = s0[78];
          double v661_data = ir1[4];
          ir1[4] = (v661_data + (v638_data * v659_data));
          double v664_data = s0[94];
          double v666_data = ir1[5];
          ir1[5] = (v666_data + (v638_data * v664_data));
          double v669_data = s0[110];
          double v671_data = ir1[6];
          ir1[6] = (v671_data + (v638_data * v669_data));
          double v674_data = s0[126];
          double v676_data = ir1[7];
          ir1[7] = (v676_data + (v638_data * v674_data));
          double v681_data = r0[15];
          double v682_data = s0[15];
          double v684_data = ir1[0];
          ir1[0] = (v684_data + (v681_data * v682_data));
          double v687_data = s0[31];
          double v689_data = ir1[1];
          ir1[1] = (v689_data + (v681_data * v687_data));
          double v692_data = s0[47];
          double v694_data = ir1[2];
          ir1[2] = (v694_data + (v681_data * v692_data));
          double v697_data = s0[63];
          double v699_data = ir1[3];
          ir1[3] = (v699_data + (v681_data * v697_data));
          double v702_data = s0[79];
          double v704_data = ir1[4];
          ir1[4] = (v704_data + (v681_data * v702_data));
          double v707_data = s0[95];
          double v709_data = ir1[5];
          ir1[5] = (v709_data + (v681_data * v707_data));
          double v712_data = s0[111];
          double v714_data = ir1[6];
          ir1[6] = (v714_data + (v681_data * v712_data));
          double v717_data = s0[127];
          double v719_data = ir1[7];
          ir1[7] = (v719_data + (v681_data * v717_data));
          #pragma unroll
          for (int32_t v724_n0 = 0; v724_n0 < 1; ++v724_n0) {
            #pragma unroll
            for (int32_t v725_n1 = 0; v725_n1 < 8; ++v725_n1) {
              int32_t v726_a = v724_n0 + v725_n1;
              int32_t v727_a = v724_n0 + v725_n1;
              double v728_data = ir1[v727_a];
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
              double v737_data = r1[(v733_i0 + v734_i1)];
              glb_m0[(v742_lead + (v734_i1 * 16))] = v737_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

