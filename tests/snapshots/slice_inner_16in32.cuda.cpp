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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 1024 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v18_off = (v10_lead + (v11_i0 * 16)) + 8;
            #pragma unroll
            for (int32_t v12_i1 = 8; v12_i1 < 24; ++v12_i1) {
              float v21_data = __ldcg(&glb_m1[(v18_off + (v12_i1 * 32))]);
              r0[(v11_i0 + (v12_i1 - 8))] = v21_data;
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
          // [(0, 16), (0, 8)] [(0, 16)]
          float ir1[8]{};
          float v31_data = r0[0];
          float v32_data = s0[0];
          float v34_data = ir1[0];
          ir1[0] = (v34_data + (v31_data * v32_data));
          float v37_data = s0[16];
          float v39_data = ir1[1];
          ir1[1] = (v39_data + (v31_data * v37_data));
          float v42_data = s0[32];
          float v44_data = ir1[2];
          ir1[2] = (v44_data + (v31_data * v42_data));
          float v47_data = s0[48];
          float v49_data = ir1[3];
          ir1[3] = (v49_data + (v31_data * v47_data));
          float v52_data = s0[64];
          float v54_data = ir1[4];
          ir1[4] = (v54_data + (v31_data * v52_data));
          float v57_data = s0[80];
          float v59_data = ir1[5];
          ir1[5] = (v59_data + (v31_data * v57_data));
          float v62_data = s0[96];
          float v64_data = ir1[6];
          ir1[6] = (v64_data + (v31_data * v62_data));
          float v67_data = s0[112];
          float v69_data = ir1[7];
          ir1[7] = (v69_data + (v31_data * v67_data));
          float v74_data = r0[1];
          float v75_data = s0[1];
          float v77_data = ir1[0];
          ir1[0] = (v77_data + (v74_data * v75_data));
          float v80_data = s0[17];
          float v82_data = ir1[1];
          ir1[1] = (v82_data + (v74_data * v80_data));
          float v85_data = s0[33];
          float v87_data = ir1[2];
          ir1[2] = (v87_data + (v74_data * v85_data));
          float v90_data = s0[49];
          float v92_data = ir1[3];
          ir1[3] = (v92_data + (v74_data * v90_data));
          float v95_data = s0[65];
          float v97_data = ir1[4];
          ir1[4] = (v97_data + (v74_data * v95_data));
          float v100_data = s0[81];
          float v102_data = ir1[5];
          ir1[5] = (v102_data + (v74_data * v100_data));
          float v105_data = s0[97];
          float v107_data = ir1[6];
          ir1[6] = (v107_data + (v74_data * v105_data));
          float v110_data = s0[113];
          float v112_data = ir1[7];
          ir1[7] = (v112_data + (v74_data * v110_data));
          float v117_data = r0[2];
          float v118_data = s0[2];
          float v120_data = ir1[0];
          ir1[0] = (v120_data + (v117_data * v118_data));
          float v123_data = s0[18];
          float v125_data = ir1[1];
          ir1[1] = (v125_data + (v117_data * v123_data));
          float v128_data = s0[34];
          float v130_data = ir1[2];
          ir1[2] = (v130_data + (v117_data * v128_data));
          float v133_data = s0[50];
          float v135_data = ir1[3];
          ir1[3] = (v135_data + (v117_data * v133_data));
          float v138_data = s0[66];
          float v140_data = ir1[4];
          ir1[4] = (v140_data + (v117_data * v138_data));
          float v143_data = s0[82];
          float v145_data = ir1[5];
          ir1[5] = (v145_data + (v117_data * v143_data));
          float v148_data = s0[98];
          float v150_data = ir1[6];
          ir1[6] = (v150_data + (v117_data * v148_data));
          float v153_data = s0[114];
          float v155_data = ir1[7];
          ir1[7] = (v155_data + (v117_data * v153_data));
          float v160_data = r0[3];
          float v161_data = s0[3];
          float v163_data = ir1[0];
          ir1[0] = (v163_data + (v160_data * v161_data));
          float v166_data = s0[19];
          float v168_data = ir1[1];
          ir1[1] = (v168_data + (v160_data * v166_data));
          float v171_data = s0[35];
          float v173_data = ir1[2];
          ir1[2] = (v173_data + (v160_data * v171_data));
          float v176_data = s0[51];
          float v178_data = ir1[3];
          ir1[3] = (v178_data + (v160_data * v176_data));
          float v181_data = s0[67];
          float v183_data = ir1[4];
          ir1[4] = (v183_data + (v160_data * v181_data));
          float v186_data = s0[83];
          float v188_data = ir1[5];
          ir1[5] = (v188_data + (v160_data * v186_data));
          float v191_data = s0[99];
          float v193_data = ir1[6];
          ir1[6] = (v193_data + (v160_data * v191_data));
          float v196_data = s0[115];
          float v198_data = ir1[7];
          ir1[7] = (v198_data + (v160_data * v196_data));
          float v203_data = r0[4];
          float v204_data = s0[4];
          float v206_data = ir1[0];
          ir1[0] = (v206_data + (v203_data * v204_data));
          float v209_data = s0[20];
          float v211_data = ir1[1];
          ir1[1] = (v211_data + (v203_data * v209_data));
          float v214_data = s0[36];
          float v216_data = ir1[2];
          ir1[2] = (v216_data + (v203_data * v214_data));
          float v219_data = s0[52];
          float v221_data = ir1[3];
          ir1[3] = (v221_data + (v203_data * v219_data));
          float v224_data = s0[68];
          float v226_data = ir1[4];
          ir1[4] = (v226_data + (v203_data * v224_data));
          float v229_data = s0[84];
          float v231_data = ir1[5];
          ir1[5] = (v231_data + (v203_data * v229_data));
          float v234_data = s0[100];
          float v236_data = ir1[6];
          ir1[6] = (v236_data + (v203_data * v234_data));
          float v239_data = s0[116];
          float v241_data = ir1[7];
          ir1[7] = (v241_data + (v203_data * v239_data));
          float v246_data = r0[5];
          float v247_data = s0[5];
          float v249_data = ir1[0];
          ir1[0] = (v249_data + (v246_data * v247_data));
          float v252_data = s0[21];
          float v254_data = ir1[1];
          ir1[1] = (v254_data + (v246_data * v252_data));
          float v257_data = s0[37];
          float v259_data = ir1[2];
          ir1[2] = (v259_data + (v246_data * v257_data));
          float v262_data = s0[53];
          float v264_data = ir1[3];
          ir1[3] = (v264_data + (v246_data * v262_data));
          float v267_data = s0[69];
          float v269_data = ir1[4];
          ir1[4] = (v269_data + (v246_data * v267_data));
          float v272_data = s0[85];
          float v274_data = ir1[5];
          ir1[5] = (v274_data + (v246_data * v272_data));
          float v277_data = s0[101];
          float v279_data = ir1[6];
          ir1[6] = (v279_data + (v246_data * v277_data));
          float v282_data = s0[117];
          float v284_data = ir1[7];
          ir1[7] = (v284_data + (v246_data * v282_data));
          float v289_data = r0[6];
          float v290_data = s0[6];
          float v292_data = ir1[0];
          ir1[0] = (v292_data + (v289_data * v290_data));
          float v295_data = s0[22];
          float v297_data = ir1[1];
          ir1[1] = (v297_data + (v289_data * v295_data));
          float v300_data = s0[38];
          float v302_data = ir1[2];
          ir1[2] = (v302_data + (v289_data * v300_data));
          float v305_data = s0[54];
          float v307_data = ir1[3];
          ir1[3] = (v307_data + (v289_data * v305_data));
          float v310_data = s0[70];
          float v312_data = ir1[4];
          ir1[4] = (v312_data + (v289_data * v310_data));
          float v315_data = s0[86];
          float v317_data = ir1[5];
          ir1[5] = (v317_data + (v289_data * v315_data));
          float v320_data = s0[102];
          float v322_data = ir1[6];
          ir1[6] = (v322_data + (v289_data * v320_data));
          float v325_data = s0[118];
          float v327_data = ir1[7];
          ir1[7] = (v327_data + (v289_data * v325_data));
          float v332_data = r0[7];
          float v333_data = s0[7];
          float v335_data = ir1[0];
          ir1[0] = (v335_data + (v332_data * v333_data));
          float v338_data = s0[23];
          float v340_data = ir1[1];
          ir1[1] = (v340_data + (v332_data * v338_data));
          float v343_data = s0[39];
          float v345_data = ir1[2];
          ir1[2] = (v345_data + (v332_data * v343_data));
          float v348_data = s0[55];
          float v350_data = ir1[3];
          ir1[3] = (v350_data + (v332_data * v348_data));
          float v353_data = s0[71];
          float v355_data = ir1[4];
          ir1[4] = (v355_data + (v332_data * v353_data));
          float v358_data = s0[87];
          float v360_data = ir1[5];
          ir1[5] = (v360_data + (v332_data * v358_data));
          float v363_data = s0[103];
          float v365_data = ir1[6];
          ir1[6] = (v365_data + (v332_data * v363_data));
          float v368_data = s0[119];
          float v370_data = ir1[7];
          ir1[7] = (v370_data + (v332_data * v368_data));
          float v375_data = r0[8];
          float v376_data = s0[8];
          float v378_data = ir1[0];
          ir1[0] = (v378_data + (v375_data * v376_data));
          float v381_data = s0[24];
          float v383_data = ir1[1];
          ir1[1] = (v383_data + (v375_data * v381_data));
          float v386_data = s0[40];
          float v388_data = ir1[2];
          ir1[2] = (v388_data + (v375_data * v386_data));
          float v391_data = s0[56];
          float v393_data = ir1[3];
          ir1[3] = (v393_data + (v375_data * v391_data));
          float v396_data = s0[72];
          float v398_data = ir1[4];
          ir1[4] = (v398_data + (v375_data * v396_data));
          float v401_data = s0[88];
          float v403_data = ir1[5];
          ir1[5] = (v403_data + (v375_data * v401_data));
          float v406_data = s0[104];
          float v408_data = ir1[6];
          ir1[6] = (v408_data + (v375_data * v406_data));
          float v411_data = s0[120];
          float v413_data = ir1[7];
          ir1[7] = (v413_data + (v375_data * v411_data));
          float v418_data = r0[9];
          float v419_data = s0[9];
          float v421_data = ir1[0];
          ir1[0] = (v421_data + (v418_data * v419_data));
          float v424_data = s0[25];
          float v426_data = ir1[1];
          ir1[1] = (v426_data + (v418_data * v424_data));
          float v429_data = s0[41];
          float v431_data = ir1[2];
          ir1[2] = (v431_data + (v418_data * v429_data));
          float v434_data = s0[57];
          float v436_data = ir1[3];
          ir1[3] = (v436_data + (v418_data * v434_data));
          float v439_data = s0[73];
          float v441_data = ir1[4];
          ir1[4] = (v441_data + (v418_data * v439_data));
          float v444_data = s0[89];
          float v446_data = ir1[5];
          ir1[5] = (v446_data + (v418_data * v444_data));
          float v449_data = s0[105];
          float v451_data = ir1[6];
          ir1[6] = (v451_data + (v418_data * v449_data));
          float v454_data = s0[121];
          float v456_data = ir1[7];
          ir1[7] = (v456_data + (v418_data * v454_data));
          float v461_data = r0[10];
          float v462_data = s0[10];
          float v464_data = ir1[0];
          ir1[0] = (v464_data + (v461_data * v462_data));
          float v467_data = s0[26];
          float v469_data = ir1[1];
          ir1[1] = (v469_data + (v461_data * v467_data));
          float v472_data = s0[42];
          float v474_data = ir1[2];
          ir1[2] = (v474_data + (v461_data * v472_data));
          float v477_data = s0[58];
          float v479_data = ir1[3];
          ir1[3] = (v479_data + (v461_data * v477_data));
          float v482_data = s0[74];
          float v484_data = ir1[4];
          ir1[4] = (v484_data + (v461_data * v482_data));
          float v487_data = s0[90];
          float v489_data = ir1[5];
          ir1[5] = (v489_data + (v461_data * v487_data));
          float v492_data = s0[106];
          float v494_data = ir1[6];
          ir1[6] = (v494_data + (v461_data * v492_data));
          float v497_data = s0[122];
          float v499_data = ir1[7];
          ir1[7] = (v499_data + (v461_data * v497_data));
          float v504_data = r0[11];
          float v505_data = s0[11];
          float v507_data = ir1[0];
          ir1[0] = (v507_data + (v504_data * v505_data));
          float v510_data = s0[27];
          float v512_data = ir1[1];
          ir1[1] = (v512_data + (v504_data * v510_data));
          float v515_data = s0[43];
          float v517_data = ir1[2];
          ir1[2] = (v517_data + (v504_data * v515_data));
          float v520_data = s0[59];
          float v522_data = ir1[3];
          ir1[3] = (v522_data + (v504_data * v520_data));
          float v525_data = s0[75];
          float v527_data = ir1[4];
          ir1[4] = (v527_data + (v504_data * v525_data));
          float v530_data = s0[91];
          float v532_data = ir1[5];
          ir1[5] = (v532_data + (v504_data * v530_data));
          float v535_data = s0[107];
          float v537_data = ir1[6];
          ir1[6] = (v537_data + (v504_data * v535_data));
          float v540_data = s0[123];
          float v542_data = ir1[7];
          ir1[7] = (v542_data + (v504_data * v540_data));
          float v547_data = r0[12];
          float v548_data = s0[12];
          float v550_data = ir1[0];
          ir1[0] = (v550_data + (v547_data * v548_data));
          float v553_data = s0[28];
          float v555_data = ir1[1];
          ir1[1] = (v555_data + (v547_data * v553_data));
          float v558_data = s0[44];
          float v560_data = ir1[2];
          ir1[2] = (v560_data + (v547_data * v558_data));
          float v563_data = s0[60];
          float v565_data = ir1[3];
          ir1[3] = (v565_data + (v547_data * v563_data));
          float v568_data = s0[76];
          float v570_data = ir1[4];
          ir1[4] = (v570_data + (v547_data * v568_data));
          float v573_data = s0[92];
          float v575_data = ir1[5];
          ir1[5] = (v575_data + (v547_data * v573_data));
          float v578_data = s0[108];
          float v580_data = ir1[6];
          ir1[6] = (v580_data + (v547_data * v578_data));
          float v583_data = s0[124];
          float v585_data = ir1[7];
          ir1[7] = (v585_data + (v547_data * v583_data));
          float v590_data = r0[13];
          float v591_data = s0[13];
          float v593_data = ir1[0];
          ir1[0] = (v593_data + (v590_data * v591_data));
          float v596_data = s0[29];
          float v598_data = ir1[1];
          ir1[1] = (v598_data + (v590_data * v596_data));
          float v601_data = s0[45];
          float v603_data = ir1[2];
          ir1[2] = (v603_data + (v590_data * v601_data));
          float v606_data = s0[61];
          float v608_data = ir1[3];
          ir1[3] = (v608_data + (v590_data * v606_data));
          float v611_data = s0[77];
          float v613_data = ir1[4];
          ir1[4] = (v613_data + (v590_data * v611_data));
          float v616_data = s0[93];
          float v618_data = ir1[5];
          ir1[5] = (v618_data + (v590_data * v616_data));
          float v621_data = s0[109];
          float v623_data = ir1[6];
          ir1[6] = (v623_data + (v590_data * v621_data));
          float v626_data = s0[125];
          float v628_data = ir1[7];
          ir1[7] = (v628_data + (v590_data * v626_data));
          float v633_data = r0[14];
          float v634_data = s0[14];
          float v636_data = ir1[0];
          ir1[0] = (v636_data + (v633_data * v634_data));
          float v639_data = s0[30];
          float v641_data = ir1[1];
          ir1[1] = (v641_data + (v633_data * v639_data));
          float v644_data = s0[46];
          float v646_data = ir1[2];
          ir1[2] = (v646_data + (v633_data * v644_data));
          float v649_data = s0[62];
          float v651_data = ir1[3];
          ir1[3] = (v651_data + (v633_data * v649_data));
          float v654_data = s0[78];
          float v656_data = ir1[4];
          ir1[4] = (v656_data + (v633_data * v654_data));
          float v659_data = s0[94];
          float v661_data = ir1[5];
          ir1[5] = (v661_data + (v633_data * v659_data));
          float v664_data = s0[110];
          float v666_data = ir1[6];
          ir1[6] = (v666_data + (v633_data * v664_data));
          float v669_data = s0[126];
          float v671_data = ir1[7];
          ir1[7] = (v671_data + (v633_data * v669_data));
          float v676_data = r0[15];
          float v677_data = s0[15];
          float v679_data = ir1[0];
          ir1[0] = (v679_data + (v676_data * v677_data));
          float v682_data = s0[31];
          float v684_data = ir1[1];
          ir1[1] = (v684_data + (v676_data * v682_data));
          float v687_data = s0[47];
          float v689_data = ir1[2];
          ir1[2] = (v689_data + (v676_data * v687_data));
          float v692_data = s0[63];
          float v694_data = ir1[3];
          ir1[3] = (v694_data + (v676_data * v692_data));
          float v697_data = s0[79];
          float v699_data = ir1[4];
          ir1[4] = (v699_data + (v676_data * v697_data));
          float v702_data = s0[95];
          float v704_data = ir1[5];
          ir1[5] = (v704_data + (v676_data * v702_data));
          float v707_data = s0[111];
          float v709_data = ir1[6];
          ir1[6] = (v709_data + (v676_data * v707_data));
          float v712_data = s0[127];
          float v714_data = ir1[7];
          ir1[7] = (v714_data + (v676_data * v712_data));
          #pragma unroll
          for (int32_t v719_n0 = 0; v719_n0 < 1; ++v719_n0) {
            #pragma unroll
            for (int32_t v720_n1 = 0; v720_n1 < 8; ++v720_n1) {
              int32_t v721_a = v719_n0 + v720_n1;
              float v722_data = ir1[v721_a];
              r1[v721_a] = v722_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v727_i0 = 0; v727_i0 < 1; ++v727_i0) {
            int32_t v735_lead = v10_lead + (v727_i0 * 16);
            #pragma unroll
            for (int32_t v728_i1 = 0; v728_i1 < 8; ++v728_i1) {
              float v730_data = r1[(v727_i0 + v728_i1)];
              glb_m0[(v735_lead + (v728_i1 * 16))] = v730_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

