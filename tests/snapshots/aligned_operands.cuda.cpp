// === base name ===
kernel_21138a3fa2

// === header ===
void launcher_kernel_21138a3fa2(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_21138a3fa2(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_21138a3fa2, block.x * block.y * block.z, 2304 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_21138a3fa2, cudaFuncAttributeMaxDynamicSharedMemorySize, 2304 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_21138a3fa2<<<grid,block,2304 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_21138a3fa2(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×8(16×8) {0..16}×{0..8} strided
    // m1 16×16(16×16) {0..16}×{0..16} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v7_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v8_i0 = 0; v8_i0 < 1; ++v8_i0) {
            int32_t v13_lead = v8_i0 * 16;
            int32_t v14_lead = v7_lead + v13_lead;
            int32_t v21_lead = v7_lead + v13_lead;
            #pragma unroll
            for (int32_t v9_i1 = 0; v9_i1 < 16; ++v9_i1) {
              int32_t v15_a = v9_i1 * 16;
              int32_t v16_a = v14_lead + v15_a;
              float v24_data = __ldcg(&glb_m1[(v21_lead + v15_a)]);
              r0[(v8_i0 + v9_i1)] = v24_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 4 * threadIdx.x + 0], &glb_m2[0 + 0 + 4 * threadIdx.x + 0], 16);
          __pipeline_commit();
          __pipeline_memcpy_async(&s0[0 + 0 + 4 * threadIdx.x + 64], &glb_m2[0 + 0 + 4 * threadIdx.x + 64], 16);
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 16), (0, 8)] [(0, 16)]
          float ir1[8]{};
          float v34_data = r0[0];
          float v35_data = s0[0];
          float v37_data = ir1[0];
          ir1[0] = (v37_data + (v34_data * v35_data));
          float v40_data = s0[16];
          float v42_data = ir1[1];
          ir1[1] = (v42_data + (v34_data * v40_data));
          float v45_data = s0[32];
          float v47_data = ir1[2];
          ir1[2] = (v47_data + (v34_data * v45_data));
          float v50_data = s0[48];
          float v52_data = ir1[3];
          ir1[3] = (v52_data + (v34_data * v50_data));
          float v55_data = s0[64];
          float v57_data = ir1[4];
          ir1[4] = (v57_data + (v34_data * v55_data));
          float v60_data = s0[80];
          float v62_data = ir1[5];
          ir1[5] = (v62_data + (v34_data * v60_data));
          float v65_data = s0[96];
          float v67_data = ir1[6];
          ir1[6] = (v67_data + (v34_data * v65_data));
          float v70_data = s0[112];
          float v72_data = ir1[7];
          ir1[7] = (v72_data + (v34_data * v70_data));
          float v77_data = r0[1];
          float v78_data = s0[1];
          float v80_data = ir1[0];
          ir1[0] = (v80_data + (v77_data * v78_data));
          float v83_data = s0[17];
          float v85_data = ir1[1];
          ir1[1] = (v85_data + (v77_data * v83_data));
          float v88_data = s0[33];
          float v90_data = ir1[2];
          ir1[2] = (v90_data + (v77_data * v88_data));
          float v93_data = s0[49];
          float v95_data = ir1[3];
          ir1[3] = (v95_data + (v77_data * v93_data));
          float v98_data = s0[65];
          float v100_data = ir1[4];
          ir1[4] = (v100_data + (v77_data * v98_data));
          float v103_data = s0[81];
          float v105_data = ir1[5];
          ir1[5] = (v105_data + (v77_data * v103_data));
          float v108_data = s0[97];
          float v110_data = ir1[6];
          ir1[6] = (v110_data + (v77_data * v108_data));
          float v113_data = s0[113];
          float v115_data = ir1[7];
          ir1[7] = (v115_data + (v77_data * v113_data));
          float v120_data = r0[2];
          float v121_data = s0[2];
          float v123_data = ir1[0];
          ir1[0] = (v123_data + (v120_data * v121_data));
          float v126_data = s0[18];
          float v128_data = ir1[1];
          ir1[1] = (v128_data + (v120_data * v126_data));
          float v131_data = s0[34];
          float v133_data = ir1[2];
          ir1[2] = (v133_data + (v120_data * v131_data));
          float v136_data = s0[50];
          float v138_data = ir1[3];
          ir1[3] = (v138_data + (v120_data * v136_data));
          float v141_data = s0[66];
          float v143_data = ir1[4];
          ir1[4] = (v143_data + (v120_data * v141_data));
          float v146_data = s0[82];
          float v148_data = ir1[5];
          ir1[5] = (v148_data + (v120_data * v146_data));
          float v151_data = s0[98];
          float v153_data = ir1[6];
          ir1[6] = (v153_data + (v120_data * v151_data));
          float v156_data = s0[114];
          float v158_data = ir1[7];
          ir1[7] = (v158_data + (v120_data * v156_data));
          float v163_data = r0[3];
          float v164_data = s0[3];
          float v166_data = ir1[0];
          ir1[0] = (v166_data + (v163_data * v164_data));
          float v169_data = s0[19];
          float v171_data = ir1[1];
          ir1[1] = (v171_data + (v163_data * v169_data));
          float v174_data = s0[35];
          float v176_data = ir1[2];
          ir1[2] = (v176_data + (v163_data * v174_data));
          float v179_data = s0[51];
          float v181_data = ir1[3];
          ir1[3] = (v181_data + (v163_data * v179_data));
          float v184_data = s0[67];
          float v186_data = ir1[4];
          ir1[4] = (v186_data + (v163_data * v184_data));
          float v189_data = s0[83];
          float v191_data = ir1[5];
          ir1[5] = (v191_data + (v163_data * v189_data));
          float v194_data = s0[99];
          float v196_data = ir1[6];
          ir1[6] = (v196_data + (v163_data * v194_data));
          float v199_data = s0[115];
          float v201_data = ir1[7];
          ir1[7] = (v201_data + (v163_data * v199_data));
          float v206_data = r0[4];
          float v207_data = s0[4];
          float v209_data = ir1[0];
          ir1[0] = (v209_data + (v206_data * v207_data));
          float v212_data = s0[20];
          float v214_data = ir1[1];
          ir1[1] = (v214_data + (v206_data * v212_data));
          float v217_data = s0[36];
          float v219_data = ir1[2];
          ir1[2] = (v219_data + (v206_data * v217_data));
          float v222_data = s0[52];
          float v224_data = ir1[3];
          ir1[3] = (v224_data + (v206_data * v222_data));
          float v227_data = s0[68];
          float v229_data = ir1[4];
          ir1[4] = (v229_data + (v206_data * v227_data));
          float v232_data = s0[84];
          float v234_data = ir1[5];
          ir1[5] = (v234_data + (v206_data * v232_data));
          float v237_data = s0[100];
          float v239_data = ir1[6];
          ir1[6] = (v239_data + (v206_data * v237_data));
          float v242_data = s0[116];
          float v244_data = ir1[7];
          ir1[7] = (v244_data + (v206_data * v242_data));
          float v249_data = r0[5];
          float v250_data = s0[5];
          float v252_data = ir1[0];
          ir1[0] = (v252_data + (v249_data * v250_data));
          float v255_data = s0[21];
          float v257_data = ir1[1];
          ir1[1] = (v257_data + (v249_data * v255_data));
          float v260_data = s0[37];
          float v262_data = ir1[2];
          ir1[2] = (v262_data + (v249_data * v260_data));
          float v265_data = s0[53];
          float v267_data = ir1[3];
          ir1[3] = (v267_data + (v249_data * v265_data));
          float v270_data = s0[69];
          float v272_data = ir1[4];
          ir1[4] = (v272_data + (v249_data * v270_data));
          float v275_data = s0[85];
          float v277_data = ir1[5];
          ir1[5] = (v277_data + (v249_data * v275_data));
          float v280_data = s0[101];
          float v282_data = ir1[6];
          ir1[6] = (v282_data + (v249_data * v280_data));
          float v285_data = s0[117];
          float v287_data = ir1[7];
          ir1[7] = (v287_data + (v249_data * v285_data));
          float v292_data = r0[6];
          float v293_data = s0[6];
          float v295_data = ir1[0];
          ir1[0] = (v295_data + (v292_data * v293_data));
          float v298_data = s0[22];
          float v300_data = ir1[1];
          ir1[1] = (v300_data + (v292_data * v298_data));
          float v303_data = s0[38];
          float v305_data = ir1[2];
          ir1[2] = (v305_data + (v292_data * v303_data));
          float v308_data = s0[54];
          float v310_data = ir1[3];
          ir1[3] = (v310_data + (v292_data * v308_data));
          float v313_data = s0[70];
          float v315_data = ir1[4];
          ir1[4] = (v315_data + (v292_data * v313_data));
          float v318_data = s0[86];
          float v320_data = ir1[5];
          ir1[5] = (v320_data + (v292_data * v318_data));
          float v323_data = s0[102];
          float v325_data = ir1[6];
          ir1[6] = (v325_data + (v292_data * v323_data));
          float v328_data = s0[118];
          float v330_data = ir1[7];
          ir1[7] = (v330_data + (v292_data * v328_data));
          float v335_data = r0[7];
          float v336_data = s0[7];
          float v338_data = ir1[0];
          ir1[0] = (v338_data + (v335_data * v336_data));
          float v341_data = s0[23];
          float v343_data = ir1[1];
          ir1[1] = (v343_data + (v335_data * v341_data));
          float v346_data = s0[39];
          float v348_data = ir1[2];
          ir1[2] = (v348_data + (v335_data * v346_data));
          float v351_data = s0[55];
          float v353_data = ir1[3];
          ir1[3] = (v353_data + (v335_data * v351_data));
          float v356_data = s0[71];
          float v358_data = ir1[4];
          ir1[4] = (v358_data + (v335_data * v356_data));
          float v361_data = s0[87];
          float v363_data = ir1[5];
          ir1[5] = (v363_data + (v335_data * v361_data));
          float v366_data = s0[103];
          float v368_data = ir1[6];
          ir1[6] = (v368_data + (v335_data * v366_data));
          float v371_data = s0[119];
          float v373_data = ir1[7];
          ir1[7] = (v373_data + (v335_data * v371_data));
          float v378_data = r0[8];
          float v379_data = s0[8];
          float v381_data = ir1[0];
          ir1[0] = (v381_data + (v378_data * v379_data));
          float v384_data = s0[24];
          float v386_data = ir1[1];
          ir1[1] = (v386_data + (v378_data * v384_data));
          float v389_data = s0[40];
          float v391_data = ir1[2];
          ir1[2] = (v391_data + (v378_data * v389_data));
          float v394_data = s0[56];
          float v396_data = ir1[3];
          ir1[3] = (v396_data + (v378_data * v394_data));
          float v399_data = s0[72];
          float v401_data = ir1[4];
          ir1[4] = (v401_data + (v378_data * v399_data));
          float v404_data = s0[88];
          float v406_data = ir1[5];
          ir1[5] = (v406_data + (v378_data * v404_data));
          float v409_data = s0[104];
          float v411_data = ir1[6];
          ir1[6] = (v411_data + (v378_data * v409_data));
          float v414_data = s0[120];
          float v416_data = ir1[7];
          ir1[7] = (v416_data + (v378_data * v414_data));
          float v421_data = r0[9];
          float v422_data = s0[9];
          float v424_data = ir1[0];
          ir1[0] = (v424_data + (v421_data * v422_data));
          float v427_data = s0[25];
          float v429_data = ir1[1];
          ir1[1] = (v429_data + (v421_data * v427_data));
          float v432_data = s0[41];
          float v434_data = ir1[2];
          ir1[2] = (v434_data + (v421_data * v432_data));
          float v437_data = s0[57];
          float v439_data = ir1[3];
          ir1[3] = (v439_data + (v421_data * v437_data));
          float v442_data = s0[73];
          float v444_data = ir1[4];
          ir1[4] = (v444_data + (v421_data * v442_data));
          float v447_data = s0[89];
          float v449_data = ir1[5];
          ir1[5] = (v449_data + (v421_data * v447_data));
          float v452_data = s0[105];
          float v454_data = ir1[6];
          ir1[6] = (v454_data + (v421_data * v452_data));
          float v457_data = s0[121];
          float v459_data = ir1[7];
          ir1[7] = (v459_data + (v421_data * v457_data));
          float v464_data = r0[10];
          float v465_data = s0[10];
          float v467_data = ir1[0];
          ir1[0] = (v467_data + (v464_data * v465_data));
          float v470_data = s0[26];
          float v472_data = ir1[1];
          ir1[1] = (v472_data + (v464_data * v470_data));
          float v475_data = s0[42];
          float v477_data = ir1[2];
          ir1[2] = (v477_data + (v464_data * v475_data));
          float v480_data = s0[58];
          float v482_data = ir1[3];
          ir1[3] = (v482_data + (v464_data * v480_data));
          float v485_data = s0[74];
          float v487_data = ir1[4];
          ir1[4] = (v487_data + (v464_data * v485_data));
          float v490_data = s0[90];
          float v492_data = ir1[5];
          ir1[5] = (v492_data + (v464_data * v490_data));
          float v495_data = s0[106];
          float v497_data = ir1[6];
          ir1[6] = (v497_data + (v464_data * v495_data));
          float v500_data = s0[122];
          float v502_data = ir1[7];
          ir1[7] = (v502_data + (v464_data * v500_data));
          float v507_data = r0[11];
          float v508_data = s0[11];
          float v510_data = ir1[0];
          ir1[0] = (v510_data + (v507_data * v508_data));
          float v513_data = s0[27];
          float v515_data = ir1[1];
          ir1[1] = (v515_data + (v507_data * v513_data));
          float v518_data = s0[43];
          float v520_data = ir1[2];
          ir1[2] = (v520_data + (v507_data * v518_data));
          float v523_data = s0[59];
          float v525_data = ir1[3];
          ir1[3] = (v525_data + (v507_data * v523_data));
          float v528_data = s0[75];
          float v530_data = ir1[4];
          ir1[4] = (v530_data + (v507_data * v528_data));
          float v533_data = s0[91];
          float v535_data = ir1[5];
          ir1[5] = (v535_data + (v507_data * v533_data));
          float v538_data = s0[107];
          float v540_data = ir1[6];
          ir1[6] = (v540_data + (v507_data * v538_data));
          float v543_data = s0[123];
          float v545_data = ir1[7];
          ir1[7] = (v545_data + (v507_data * v543_data));
          float v550_data = r0[12];
          float v551_data = s0[12];
          float v553_data = ir1[0];
          ir1[0] = (v553_data + (v550_data * v551_data));
          float v556_data = s0[28];
          float v558_data = ir1[1];
          ir1[1] = (v558_data + (v550_data * v556_data));
          float v561_data = s0[44];
          float v563_data = ir1[2];
          ir1[2] = (v563_data + (v550_data * v561_data));
          float v566_data = s0[60];
          float v568_data = ir1[3];
          ir1[3] = (v568_data + (v550_data * v566_data));
          float v571_data = s0[76];
          float v573_data = ir1[4];
          ir1[4] = (v573_data + (v550_data * v571_data));
          float v576_data = s0[92];
          float v578_data = ir1[5];
          ir1[5] = (v578_data + (v550_data * v576_data));
          float v581_data = s0[108];
          float v583_data = ir1[6];
          ir1[6] = (v583_data + (v550_data * v581_data));
          float v586_data = s0[124];
          float v588_data = ir1[7];
          ir1[7] = (v588_data + (v550_data * v586_data));
          float v593_data = r0[13];
          float v594_data = s0[13];
          float v596_data = ir1[0];
          ir1[0] = (v596_data + (v593_data * v594_data));
          float v599_data = s0[29];
          float v601_data = ir1[1];
          ir1[1] = (v601_data + (v593_data * v599_data));
          float v604_data = s0[45];
          float v606_data = ir1[2];
          ir1[2] = (v606_data + (v593_data * v604_data));
          float v609_data = s0[61];
          float v611_data = ir1[3];
          ir1[3] = (v611_data + (v593_data * v609_data));
          float v614_data = s0[77];
          float v616_data = ir1[4];
          ir1[4] = (v616_data + (v593_data * v614_data));
          float v619_data = s0[93];
          float v621_data = ir1[5];
          ir1[5] = (v621_data + (v593_data * v619_data));
          float v624_data = s0[109];
          float v626_data = ir1[6];
          ir1[6] = (v626_data + (v593_data * v624_data));
          float v629_data = s0[125];
          float v631_data = ir1[7];
          ir1[7] = (v631_data + (v593_data * v629_data));
          float v636_data = r0[14];
          float v637_data = s0[14];
          float v639_data = ir1[0];
          ir1[0] = (v639_data + (v636_data * v637_data));
          float v642_data = s0[30];
          float v644_data = ir1[1];
          ir1[1] = (v644_data + (v636_data * v642_data));
          float v647_data = s0[46];
          float v649_data = ir1[2];
          ir1[2] = (v649_data + (v636_data * v647_data));
          float v652_data = s0[62];
          float v654_data = ir1[3];
          ir1[3] = (v654_data + (v636_data * v652_data));
          float v657_data = s0[78];
          float v659_data = ir1[4];
          ir1[4] = (v659_data + (v636_data * v657_data));
          float v662_data = s0[94];
          float v664_data = ir1[5];
          ir1[5] = (v664_data + (v636_data * v662_data));
          float v667_data = s0[110];
          float v669_data = ir1[6];
          ir1[6] = (v669_data + (v636_data * v667_data));
          float v672_data = s0[126];
          float v674_data = ir1[7];
          ir1[7] = (v674_data + (v636_data * v672_data));
          float v679_data = r0[15];
          float v680_data = s0[15];
          float v682_data = ir1[0];
          ir1[0] = (v682_data + (v679_data * v680_data));
          float v685_data = s0[31];
          float v687_data = ir1[1];
          ir1[1] = (v687_data + (v679_data * v685_data));
          float v690_data = s0[47];
          float v692_data = ir1[2];
          ir1[2] = (v692_data + (v679_data * v690_data));
          float v695_data = s0[63];
          float v697_data = ir1[3];
          ir1[3] = (v697_data + (v679_data * v695_data));
          float v700_data = s0[79];
          float v702_data = ir1[4];
          ir1[4] = (v702_data + (v679_data * v700_data));
          float v705_data = s0[95];
          float v707_data = ir1[5];
          ir1[5] = (v707_data + (v679_data * v705_data));
          float v710_data = s0[111];
          float v712_data = ir1[6];
          ir1[6] = (v712_data + (v679_data * v710_data));
          float v715_data = s0[127];
          float v717_data = ir1[7];
          ir1[7] = (v717_data + (v679_data * v715_data));
          #pragma unroll
          for (int32_t v722_n0 = 0; v722_n0 < 1; ++v722_n0) {
            #pragma unroll
            for (int32_t v723_n1 = 0; v723_n1 < 8; ++v723_n1) {
              int32_t v724_a = v722_n0 + v723_n1;
              int32_t v725_a = v722_n0 + v723_n1;
              float v726_data = ir1[v725_a];
              r1[v725_a] = v726_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v731_i0 = 0; v731_i0 < 1; ++v731_i0) {
            int32_t v740_lead = v7_lead + (v731_i0 * 16);
            #pragma unroll
            for (int32_t v732_i1 = 0; v732_i1 < 8; ++v732_i1) {
              int32_t v733_a = v731_i0 + v732_i1;
              float v735_data = r1[(v731_i0 + v732_i1)];
              glb_m0[(v740_lead + (v732_i1 * 16))] = v735_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

