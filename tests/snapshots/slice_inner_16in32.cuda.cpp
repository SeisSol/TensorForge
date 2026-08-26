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
          int32_t v3_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v4_i0 = 0; v4_i0 < 1; ++v4_i0) {
            int32_t v9_lead = v4_i0 * 16;
            int32_t v11_off = (v3_lead + v9_lead) + 8;
            int32_t v19_off = (v3_lead + v9_lead) + 8;
            #pragma unroll
            for (int32_t v5_i1 = 8; v5_i1 < 24; ++v5_i1) {
              int32_t v12_a = v5_i1 * 32;
              int32_t v13_a = v11_off + v12_a;
              float v22_data = __ldcg(&glb_m1[(v19_off + v12_a)]);
              int32_t v24_a = v4_i0 + (v5_i1 - 8);
              r0[v24_a] = v22_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 8; i += 1) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r1[8]{};
          __syncwarp();
          {
            // r1 = +(r0 * s0) + None
            // [(0, 16), (0, 8)] [(0, 16)]
            float ir1[8]{};
            float v29_data = r0[0];
            float v30_data = s0[0];
            float v32_data = ir1[0];
            ir1[0] = (v32_data + (v29_data * v30_data));
            float v35_data = s0[16];
            float v37_data = ir1[1];
            ir1[1] = (v37_data + (v29_data * v35_data));
            float v40_data = s0[32];
            float v42_data = ir1[2];
            ir1[2] = (v42_data + (v29_data * v40_data));
            float v45_data = s0[48];
            float v47_data = ir1[3];
            ir1[3] = (v47_data + (v29_data * v45_data));
            float v50_data = s0[64];
            float v52_data = ir1[4];
            ir1[4] = (v52_data + (v29_data * v50_data));
            float v55_data = s0[80];
            float v57_data = ir1[5];
            ir1[5] = (v57_data + (v29_data * v55_data));
            float v60_data = s0[96];
            float v62_data = ir1[6];
            ir1[6] = (v62_data + (v29_data * v60_data));
            float v65_data = s0[112];
            float v67_data = ir1[7];
            ir1[7] = (v67_data + (v29_data * v65_data));
            float v72_data = r0[1];
            float v73_data = s0[1];
            float v75_data = ir1[0];
            ir1[0] = (v75_data + (v72_data * v73_data));
            float v78_data = s0[17];
            float v80_data = ir1[1];
            ir1[1] = (v80_data + (v72_data * v78_data));
            float v83_data = s0[33];
            float v85_data = ir1[2];
            ir1[2] = (v85_data + (v72_data * v83_data));
            float v88_data = s0[49];
            float v90_data = ir1[3];
            ir1[3] = (v90_data + (v72_data * v88_data));
            float v93_data = s0[65];
            float v95_data = ir1[4];
            ir1[4] = (v95_data + (v72_data * v93_data));
            float v98_data = s0[81];
            float v100_data = ir1[5];
            ir1[5] = (v100_data + (v72_data * v98_data));
            float v103_data = s0[97];
            float v105_data = ir1[6];
            ir1[6] = (v105_data + (v72_data * v103_data));
            float v108_data = s0[113];
            float v110_data = ir1[7];
            ir1[7] = (v110_data + (v72_data * v108_data));
            float v115_data = r0[2];
            float v116_data = s0[2];
            float v118_data = ir1[0];
            ir1[0] = (v118_data + (v115_data * v116_data));
            float v121_data = s0[18];
            float v123_data = ir1[1];
            ir1[1] = (v123_data + (v115_data * v121_data));
            float v126_data = s0[34];
            float v128_data = ir1[2];
            ir1[2] = (v128_data + (v115_data * v126_data));
            float v131_data = s0[50];
            float v133_data = ir1[3];
            ir1[3] = (v133_data + (v115_data * v131_data));
            float v136_data = s0[66];
            float v138_data = ir1[4];
            ir1[4] = (v138_data + (v115_data * v136_data));
            float v141_data = s0[82];
            float v143_data = ir1[5];
            ir1[5] = (v143_data + (v115_data * v141_data));
            float v146_data = s0[98];
            float v148_data = ir1[6];
            ir1[6] = (v148_data + (v115_data * v146_data));
            float v151_data = s0[114];
            float v153_data = ir1[7];
            ir1[7] = (v153_data + (v115_data * v151_data));
            float v158_data = r0[3];
            float v159_data = s0[3];
            float v161_data = ir1[0];
            ir1[0] = (v161_data + (v158_data * v159_data));
            float v164_data = s0[19];
            float v166_data = ir1[1];
            ir1[1] = (v166_data + (v158_data * v164_data));
            float v169_data = s0[35];
            float v171_data = ir1[2];
            ir1[2] = (v171_data + (v158_data * v169_data));
            float v174_data = s0[51];
            float v176_data = ir1[3];
            ir1[3] = (v176_data + (v158_data * v174_data));
            float v179_data = s0[67];
            float v181_data = ir1[4];
            ir1[4] = (v181_data + (v158_data * v179_data));
            float v184_data = s0[83];
            float v186_data = ir1[5];
            ir1[5] = (v186_data + (v158_data * v184_data));
            float v189_data = s0[99];
            float v191_data = ir1[6];
            ir1[6] = (v191_data + (v158_data * v189_data));
            float v194_data = s0[115];
            float v196_data = ir1[7];
            ir1[7] = (v196_data + (v158_data * v194_data));
            float v201_data = r0[4];
            float v202_data = s0[4];
            float v204_data = ir1[0];
            ir1[0] = (v204_data + (v201_data * v202_data));
            float v207_data = s0[20];
            float v209_data = ir1[1];
            ir1[1] = (v209_data + (v201_data * v207_data));
            float v212_data = s0[36];
            float v214_data = ir1[2];
            ir1[2] = (v214_data + (v201_data * v212_data));
            float v217_data = s0[52];
            float v219_data = ir1[3];
            ir1[3] = (v219_data + (v201_data * v217_data));
            float v222_data = s0[68];
            float v224_data = ir1[4];
            ir1[4] = (v224_data + (v201_data * v222_data));
            float v227_data = s0[84];
            float v229_data = ir1[5];
            ir1[5] = (v229_data + (v201_data * v227_data));
            float v232_data = s0[100];
            float v234_data = ir1[6];
            ir1[6] = (v234_data + (v201_data * v232_data));
            float v237_data = s0[116];
            float v239_data = ir1[7];
            ir1[7] = (v239_data + (v201_data * v237_data));
            float v244_data = r0[5];
            float v245_data = s0[5];
            float v247_data = ir1[0];
            ir1[0] = (v247_data + (v244_data * v245_data));
            float v250_data = s0[21];
            float v252_data = ir1[1];
            ir1[1] = (v252_data + (v244_data * v250_data));
            float v255_data = s0[37];
            float v257_data = ir1[2];
            ir1[2] = (v257_data + (v244_data * v255_data));
            float v260_data = s0[53];
            float v262_data = ir1[3];
            ir1[3] = (v262_data + (v244_data * v260_data));
            float v265_data = s0[69];
            float v267_data = ir1[4];
            ir1[4] = (v267_data + (v244_data * v265_data));
            float v270_data = s0[85];
            float v272_data = ir1[5];
            ir1[5] = (v272_data + (v244_data * v270_data));
            float v275_data = s0[101];
            float v277_data = ir1[6];
            ir1[6] = (v277_data + (v244_data * v275_data));
            float v280_data = s0[117];
            float v282_data = ir1[7];
            ir1[7] = (v282_data + (v244_data * v280_data));
            float v287_data = r0[6];
            float v288_data = s0[6];
            float v290_data = ir1[0];
            ir1[0] = (v290_data + (v287_data * v288_data));
            float v293_data = s0[22];
            float v295_data = ir1[1];
            ir1[1] = (v295_data + (v287_data * v293_data));
            float v298_data = s0[38];
            float v300_data = ir1[2];
            ir1[2] = (v300_data + (v287_data * v298_data));
            float v303_data = s0[54];
            float v305_data = ir1[3];
            ir1[3] = (v305_data + (v287_data * v303_data));
            float v308_data = s0[70];
            float v310_data = ir1[4];
            ir1[4] = (v310_data + (v287_data * v308_data));
            float v313_data = s0[86];
            float v315_data = ir1[5];
            ir1[5] = (v315_data + (v287_data * v313_data));
            float v318_data = s0[102];
            float v320_data = ir1[6];
            ir1[6] = (v320_data + (v287_data * v318_data));
            float v323_data = s0[118];
            float v325_data = ir1[7];
            ir1[7] = (v325_data + (v287_data * v323_data));
            float v330_data = r0[7];
            float v331_data = s0[7];
            float v333_data = ir1[0];
            ir1[0] = (v333_data + (v330_data * v331_data));
            float v336_data = s0[23];
            float v338_data = ir1[1];
            ir1[1] = (v338_data + (v330_data * v336_data));
            float v341_data = s0[39];
            float v343_data = ir1[2];
            ir1[2] = (v343_data + (v330_data * v341_data));
            float v346_data = s0[55];
            float v348_data = ir1[3];
            ir1[3] = (v348_data + (v330_data * v346_data));
            float v351_data = s0[71];
            float v353_data = ir1[4];
            ir1[4] = (v353_data + (v330_data * v351_data));
            float v356_data = s0[87];
            float v358_data = ir1[5];
            ir1[5] = (v358_data + (v330_data * v356_data));
            float v361_data = s0[103];
            float v363_data = ir1[6];
            ir1[6] = (v363_data + (v330_data * v361_data));
            float v366_data = s0[119];
            float v368_data = ir1[7];
            ir1[7] = (v368_data + (v330_data * v366_data));
            float v373_data = r0[8];
            float v374_data = s0[8];
            float v376_data = ir1[0];
            ir1[0] = (v376_data + (v373_data * v374_data));
            float v379_data = s0[24];
            float v381_data = ir1[1];
            ir1[1] = (v381_data + (v373_data * v379_data));
            float v384_data = s0[40];
            float v386_data = ir1[2];
            ir1[2] = (v386_data + (v373_data * v384_data));
            float v389_data = s0[56];
            float v391_data = ir1[3];
            ir1[3] = (v391_data + (v373_data * v389_data));
            float v394_data = s0[72];
            float v396_data = ir1[4];
            ir1[4] = (v396_data + (v373_data * v394_data));
            float v399_data = s0[88];
            float v401_data = ir1[5];
            ir1[5] = (v401_data + (v373_data * v399_data));
            float v404_data = s0[104];
            float v406_data = ir1[6];
            ir1[6] = (v406_data + (v373_data * v404_data));
            float v409_data = s0[120];
            float v411_data = ir1[7];
            ir1[7] = (v411_data + (v373_data * v409_data));
            float v416_data = r0[9];
            float v417_data = s0[9];
            float v419_data = ir1[0];
            ir1[0] = (v419_data + (v416_data * v417_data));
            float v422_data = s0[25];
            float v424_data = ir1[1];
            ir1[1] = (v424_data + (v416_data * v422_data));
            float v427_data = s0[41];
            float v429_data = ir1[2];
            ir1[2] = (v429_data + (v416_data * v427_data));
            float v432_data = s0[57];
            float v434_data = ir1[3];
            ir1[3] = (v434_data + (v416_data * v432_data));
            float v437_data = s0[73];
            float v439_data = ir1[4];
            ir1[4] = (v439_data + (v416_data * v437_data));
            float v442_data = s0[89];
            float v444_data = ir1[5];
            ir1[5] = (v444_data + (v416_data * v442_data));
            float v447_data = s0[105];
            float v449_data = ir1[6];
            ir1[6] = (v449_data + (v416_data * v447_data));
            float v452_data = s0[121];
            float v454_data = ir1[7];
            ir1[7] = (v454_data + (v416_data * v452_data));
            float v459_data = r0[10];
            float v460_data = s0[10];
            float v462_data = ir1[0];
            ir1[0] = (v462_data + (v459_data * v460_data));
            float v465_data = s0[26];
            float v467_data = ir1[1];
            ir1[1] = (v467_data + (v459_data * v465_data));
            float v470_data = s0[42];
            float v472_data = ir1[2];
            ir1[2] = (v472_data + (v459_data * v470_data));
            float v475_data = s0[58];
            float v477_data = ir1[3];
            ir1[3] = (v477_data + (v459_data * v475_data));
            float v480_data = s0[74];
            float v482_data = ir1[4];
            ir1[4] = (v482_data + (v459_data * v480_data));
            float v485_data = s0[90];
            float v487_data = ir1[5];
            ir1[5] = (v487_data + (v459_data * v485_data));
            float v490_data = s0[106];
            float v492_data = ir1[6];
            ir1[6] = (v492_data + (v459_data * v490_data));
            float v495_data = s0[122];
            float v497_data = ir1[7];
            ir1[7] = (v497_data + (v459_data * v495_data));
            float v502_data = r0[11];
            float v503_data = s0[11];
            float v505_data = ir1[0];
            ir1[0] = (v505_data + (v502_data * v503_data));
            float v508_data = s0[27];
            float v510_data = ir1[1];
            ir1[1] = (v510_data + (v502_data * v508_data));
            float v513_data = s0[43];
            float v515_data = ir1[2];
            ir1[2] = (v515_data + (v502_data * v513_data));
            float v518_data = s0[59];
            float v520_data = ir1[3];
            ir1[3] = (v520_data + (v502_data * v518_data));
            float v523_data = s0[75];
            float v525_data = ir1[4];
            ir1[4] = (v525_data + (v502_data * v523_data));
            float v528_data = s0[91];
            float v530_data = ir1[5];
            ir1[5] = (v530_data + (v502_data * v528_data));
            float v533_data = s0[107];
            float v535_data = ir1[6];
            ir1[6] = (v535_data + (v502_data * v533_data));
            float v538_data = s0[123];
            float v540_data = ir1[7];
            ir1[7] = (v540_data + (v502_data * v538_data));
            float v545_data = r0[12];
            float v546_data = s0[12];
            float v548_data = ir1[0];
            ir1[0] = (v548_data + (v545_data * v546_data));
            float v551_data = s0[28];
            float v553_data = ir1[1];
            ir1[1] = (v553_data + (v545_data * v551_data));
            float v556_data = s0[44];
            float v558_data = ir1[2];
            ir1[2] = (v558_data + (v545_data * v556_data));
            float v561_data = s0[60];
            float v563_data = ir1[3];
            ir1[3] = (v563_data + (v545_data * v561_data));
            float v566_data = s0[76];
            float v568_data = ir1[4];
            ir1[4] = (v568_data + (v545_data * v566_data));
            float v571_data = s0[92];
            float v573_data = ir1[5];
            ir1[5] = (v573_data + (v545_data * v571_data));
            float v576_data = s0[108];
            float v578_data = ir1[6];
            ir1[6] = (v578_data + (v545_data * v576_data));
            float v581_data = s0[124];
            float v583_data = ir1[7];
            ir1[7] = (v583_data + (v545_data * v581_data));
            float v588_data = r0[13];
            float v589_data = s0[13];
            float v591_data = ir1[0];
            ir1[0] = (v591_data + (v588_data * v589_data));
            float v594_data = s0[29];
            float v596_data = ir1[1];
            ir1[1] = (v596_data + (v588_data * v594_data));
            float v599_data = s0[45];
            float v601_data = ir1[2];
            ir1[2] = (v601_data + (v588_data * v599_data));
            float v604_data = s0[61];
            float v606_data = ir1[3];
            ir1[3] = (v606_data + (v588_data * v604_data));
            float v609_data = s0[77];
            float v611_data = ir1[4];
            ir1[4] = (v611_data + (v588_data * v609_data));
            float v614_data = s0[93];
            float v616_data = ir1[5];
            ir1[5] = (v616_data + (v588_data * v614_data));
            float v619_data = s0[109];
            float v621_data = ir1[6];
            ir1[6] = (v621_data + (v588_data * v619_data));
            float v624_data = s0[125];
            float v626_data = ir1[7];
            ir1[7] = (v626_data + (v588_data * v624_data));
            float v631_data = r0[14];
            float v632_data = s0[14];
            float v634_data = ir1[0];
            ir1[0] = (v634_data + (v631_data * v632_data));
            float v637_data = s0[30];
            float v639_data = ir1[1];
            ir1[1] = (v639_data + (v631_data * v637_data));
            float v642_data = s0[46];
            float v644_data = ir1[2];
            ir1[2] = (v644_data + (v631_data * v642_data));
            float v647_data = s0[62];
            float v649_data = ir1[3];
            ir1[3] = (v649_data + (v631_data * v647_data));
            float v652_data = s0[78];
            float v654_data = ir1[4];
            ir1[4] = (v654_data + (v631_data * v652_data));
            float v657_data = s0[94];
            float v659_data = ir1[5];
            ir1[5] = (v659_data + (v631_data * v657_data));
            float v662_data = s0[110];
            float v664_data = ir1[6];
            ir1[6] = (v664_data + (v631_data * v662_data));
            float v667_data = s0[126];
            float v669_data = ir1[7];
            ir1[7] = (v669_data + (v631_data * v667_data));
            float v674_data = r0[15];
            float v675_data = s0[15];
            float v677_data = ir1[0];
            ir1[0] = (v677_data + (v674_data * v675_data));
            float v680_data = s0[31];
            float v682_data = ir1[1];
            ir1[1] = (v682_data + (v674_data * v680_data));
            float v685_data = s0[47];
            float v687_data = ir1[2];
            ir1[2] = (v687_data + (v674_data * v685_data));
            float v690_data = s0[63];
            float v692_data = ir1[3];
            ir1[3] = (v692_data + (v674_data * v690_data));
            float v695_data = s0[79];
            float v697_data = ir1[4];
            ir1[4] = (v697_data + (v674_data * v695_data));
            float v700_data = s0[95];
            float v702_data = ir1[5];
            ir1[5] = (v702_data + (v674_data * v700_data));
            float v705_data = s0[111];
            float v707_data = ir1[6];
            ir1[6] = (v707_data + (v674_data * v705_data));
            float v710_data = s0[127];
            float v712_data = ir1[7];
            ir1[7] = (v712_data + (v674_data * v710_data));
            #pragma unroll
            for (int32_t v717_n0 = 0; v717_n0 < 1; ++v717_n0) {
              #pragma unroll
              for (int32_t v718_n1 = 0; v718_n1 < 8; ++v718_n1) {
                int32_t v719_a = v717_n0 + v718_n1;
                int32_t v720_a = v717_n0 + v718_n1;
                float v721_data = ir1[v720_a];
                int32_t v722_a = v717_n0 + v718_n1;
                r1[v720_a] = v721_data;
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v727_i0 = 0; v727_i0 < 1; ++v727_i0) {
            int32_t v736_lead = v3_lead + (v727_i0 * 16);
            #pragma unroll
            for (int32_t v728_i1 = 0; v728_i1 < 8; ++v728_i1) {
              int32_t v729_a = v727_i0 + v728_i1;
              float v731_data = r1[(v727_i0 + v728_i1)];
              int32_t v738_a = v736_lead + (v728_i1 * 16);
              glb_m0[v738_a] = v731_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

