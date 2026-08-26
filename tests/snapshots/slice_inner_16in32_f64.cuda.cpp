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
              double v22_data = __ldcg(&glb_m1[(v19_off + v12_a)]);
              int32_t v24_a = v4_i0 + (v5_i1 - 8);
              r0[v24_a] = v22_data;
            }
          }
          double* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 8; i += 1) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<8>(8), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          double r1[8]{};
          __syncwarp();
          {
            // r1 = +(r0 * s0) + None
            // [(0, 16), (0, 8)] [(0, 16)]
            double ir1[8]{};
            double v29_data = r0[0];
            double v30_data = s0[0];
            double v32_data = ir1[0];
            ir1[0] = (v32_data + (v29_data * v30_data));
            double v35_data = s0[16];
            double v37_data = ir1[1];
            ir1[1] = (v37_data + (v29_data * v35_data));
            double v40_data = s0[32];
            double v42_data = ir1[2];
            ir1[2] = (v42_data + (v29_data * v40_data));
            double v45_data = s0[48];
            double v47_data = ir1[3];
            ir1[3] = (v47_data + (v29_data * v45_data));
            double v50_data = s0[64];
            double v52_data = ir1[4];
            ir1[4] = (v52_data + (v29_data * v50_data));
            double v55_data = s0[80];
            double v57_data = ir1[5];
            ir1[5] = (v57_data + (v29_data * v55_data));
            double v60_data = s0[96];
            double v62_data = ir1[6];
            ir1[6] = (v62_data + (v29_data * v60_data));
            double v65_data = s0[112];
            double v67_data = ir1[7];
            ir1[7] = (v67_data + (v29_data * v65_data));
            double v72_data = r0[1];
            double v73_data = s0[1];
            double v75_data = ir1[0];
            ir1[0] = (v75_data + (v72_data * v73_data));
            double v78_data = s0[17];
            double v80_data = ir1[1];
            ir1[1] = (v80_data + (v72_data * v78_data));
            double v83_data = s0[33];
            double v85_data = ir1[2];
            ir1[2] = (v85_data + (v72_data * v83_data));
            double v88_data = s0[49];
            double v90_data = ir1[3];
            ir1[3] = (v90_data + (v72_data * v88_data));
            double v93_data = s0[65];
            double v95_data = ir1[4];
            ir1[4] = (v95_data + (v72_data * v93_data));
            double v98_data = s0[81];
            double v100_data = ir1[5];
            ir1[5] = (v100_data + (v72_data * v98_data));
            double v103_data = s0[97];
            double v105_data = ir1[6];
            ir1[6] = (v105_data + (v72_data * v103_data));
            double v108_data = s0[113];
            double v110_data = ir1[7];
            ir1[7] = (v110_data + (v72_data * v108_data));
            double v115_data = r0[2];
            double v116_data = s0[2];
            double v118_data = ir1[0];
            ir1[0] = (v118_data + (v115_data * v116_data));
            double v121_data = s0[18];
            double v123_data = ir1[1];
            ir1[1] = (v123_data + (v115_data * v121_data));
            double v126_data = s0[34];
            double v128_data = ir1[2];
            ir1[2] = (v128_data + (v115_data * v126_data));
            double v131_data = s0[50];
            double v133_data = ir1[3];
            ir1[3] = (v133_data + (v115_data * v131_data));
            double v136_data = s0[66];
            double v138_data = ir1[4];
            ir1[4] = (v138_data + (v115_data * v136_data));
            double v141_data = s0[82];
            double v143_data = ir1[5];
            ir1[5] = (v143_data + (v115_data * v141_data));
            double v146_data = s0[98];
            double v148_data = ir1[6];
            ir1[6] = (v148_data + (v115_data * v146_data));
            double v151_data = s0[114];
            double v153_data = ir1[7];
            ir1[7] = (v153_data + (v115_data * v151_data));
            double v158_data = r0[3];
            double v159_data = s0[3];
            double v161_data = ir1[0];
            ir1[0] = (v161_data + (v158_data * v159_data));
            double v164_data = s0[19];
            double v166_data = ir1[1];
            ir1[1] = (v166_data + (v158_data * v164_data));
            double v169_data = s0[35];
            double v171_data = ir1[2];
            ir1[2] = (v171_data + (v158_data * v169_data));
            double v174_data = s0[51];
            double v176_data = ir1[3];
            ir1[3] = (v176_data + (v158_data * v174_data));
            double v179_data = s0[67];
            double v181_data = ir1[4];
            ir1[4] = (v181_data + (v158_data * v179_data));
            double v184_data = s0[83];
            double v186_data = ir1[5];
            ir1[5] = (v186_data + (v158_data * v184_data));
            double v189_data = s0[99];
            double v191_data = ir1[6];
            ir1[6] = (v191_data + (v158_data * v189_data));
            double v194_data = s0[115];
            double v196_data = ir1[7];
            ir1[7] = (v196_data + (v158_data * v194_data));
            double v201_data = r0[4];
            double v202_data = s0[4];
            double v204_data = ir1[0];
            ir1[0] = (v204_data + (v201_data * v202_data));
            double v207_data = s0[20];
            double v209_data = ir1[1];
            ir1[1] = (v209_data + (v201_data * v207_data));
            double v212_data = s0[36];
            double v214_data = ir1[2];
            ir1[2] = (v214_data + (v201_data * v212_data));
            double v217_data = s0[52];
            double v219_data = ir1[3];
            ir1[3] = (v219_data + (v201_data * v217_data));
            double v222_data = s0[68];
            double v224_data = ir1[4];
            ir1[4] = (v224_data + (v201_data * v222_data));
            double v227_data = s0[84];
            double v229_data = ir1[5];
            ir1[5] = (v229_data + (v201_data * v227_data));
            double v232_data = s0[100];
            double v234_data = ir1[6];
            ir1[6] = (v234_data + (v201_data * v232_data));
            double v237_data = s0[116];
            double v239_data = ir1[7];
            ir1[7] = (v239_data + (v201_data * v237_data));
            double v244_data = r0[5];
            double v245_data = s0[5];
            double v247_data = ir1[0];
            ir1[0] = (v247_data + (v244_data * v245_data));
            double v250_data = s0[21];
            double v252_data = ir1[1];
            ir1[1] = (v252_data + (v244_data * v250_data));
            double v255_data = s0[37];
            double v257_data = ir1[2];
            ir1[2] = (v257_data + (v244_data * v255_data));
            double v260_data = s0[53];
            double v262_data = ir1[3];
            ir1[3] = (v262_data + (v244_data * v260_data));
            double v265_data = s0[69];
            double v267_data = ir1[4];
            ir1[4] = (v267_data + (v244_data * v265_data));
            double v270_data = s0[85];
            double v272_data = ir1[5];
            ir1[5] = (v272_data + (v244_data * v270_data));
            double v275_data = s0[101];
            double v277_data = ir1[6];
            ir1[6] = (v277_data + (v244_data * v275_data));
            double v280_data = s0[117];
            double v282_data = ir1[7];
            ir1[7] = (v282_data + (v244_data * v280_data));
            double v287_data = r0[6];
            double v288_data = s0[6];
            double v290_data = ir1[0];
            ir1[0] = (v290_data + (v287_data * v288_data));
            double v293_data = s0[22];
            double v295_data = ir1[1];
            ir1[1] = (v295_data + (v287_data * v293_data));
            double v298_data = s0[38];
            double v300_data = ir1[2];
            ir1[2] = (v300_data + (v287_data * v298_data));
            double v303_data = s0[54];
            double v305_data = ir1[3];
            ir1[3] = (v305_data + (v287_data * v303_data));
            double v308_data = s0[70];
            double v310_data = ir1[4];
            ir1[4] = (v310_data + (v287_data * v308_data));
            double v313_data = s0[86];
            double v315_data = ir1[5];
            ir1[5] = (v315_data + (v287_data * v313_data));
            double v318_data = s0[102];
            double v320_data = ir1[6];
            ir1[6] = (v320_data + (v287_data * v318_data));
            double v323_data = s0[118];
            double v325_data = ir1[7];
            ir1[7] = (v325_data + (v287_data * v323_data));
            double v330_data = r0[7];
            double v331_data = s0[7];
            double v333_data = ir1[0];
            ir1[0] = (v333_data + (v330_data * v331_data));
            double v336_data = s0[23];
            double v338_data = ir1[1];
            ir1[1] = (v338_data + (v330_data * v336_data));
            double v341_data = s0[39];
            double v343_data = ir1[2];
            ir1[2] = (v343_data + (v330_data * v341_data));
            double v346_data = s0[55];
            double v348_data = ir1[3];
            ir1[3] = (v348_data + (v330_data * v346_data));
            double v351_data = s0[71];
            double v353_data = ir1[4];
            ir1[4] = (v353_data + (v330_data * v351_data));
            double v356_data = s0[87];
            double v358_data = ir1[5];
            ir1[5] = (v358_data + (v330_data * v356_data));
            double v361_data = s0[103];
            double v363_data = ir1[6];
            ir1[6] = (v363_data + (v330_data * v361_data));
            double v366_data = s0[119];
            double v368_data = ir1[7];
            ir1[7] = (v368_data + (v330_data * v366_data));
            double v373_data = r0[8];
            double v374_data = s0[8];
            double v376_data = ir1[0];
            ir1[0] = (v376_data + (v373_data * v374_data));
            double v379_data = s0[24];
            double v381_data = ir1[1];
            ir1[1] = (v381_data + (v373_data * v379_data));
            double v384_data = s0[40];
            double v386_data = ir1[2];
            ir1[2] = (v386_data + (v373_data * v384_data));
            double v389_data = s0[56];
            double v391_data = ir1[3];
            ir1[3] = (v391_data + (v373_data * v389_data));
            double v394_data = s0[72];
            double v396_data = ir1[4];
            ir1[4] = (v396_data + (v373_data * v394_data));
            double v399_data = s0[88];
            double v401_data = ir1[5];
            ir1[5] = (v401_data + (v373_data * v399_data));
            double v404_data = s0[104];
            double v406_data = ir1[6];
            ir1[6] = (v406_data + (v373_data * v404_data));
            double v409_data = s0[120];
            double v411_data = ir1[7];
            ir1[7] = (v411_data + (v373_data * v409_data));
            double v416_data = r0[9];
            double v417_data = s0[9];
            double v419_data = ir1[0];
            ir1[0] = (v419_data + (v416_data * v417_data));
            double v422_data = s0[25];
            double v424_data = ir1[1];
            ir1[1] = (v424_data + (v416_data * v422_data));
            double v427_data = s0[41];
            double v429_data = ir1[2];
            ir1[2] = (v429_data + (v416_data * v427_data));
            double v432_data = s0[57];
            double v434_data = ir1[3];
            ir1[3] = (v434_data + (v416_data * v432_data));
            double v437_data = s0[73];
            double v439_data = ir1[4];
            ir1[4] = (v439_data + (v416_data * v437_data));
            double v442_data = s0[89];
            double v444_data = ir1[5];
            ir1[5] = (v444_data + (v416_data * v442_data));
            double v447_data = s0[105];
            double v449_data = ir1[6];
            ir1[6] = (v449_data + (v416_data * v447_data));
            double v452_data = s0[121];
            double v454_data = ir1[7];
            ir1[7] = (v454_data + (v416_data * v452_data));
            double v459_data = r0[10];
            double v460_data = s0[10];
            double v462_data = ir1[0];
            ir1[0] = (v462_data + (v459_data * v460_data));
            double v465_data = s0[26];
            double v467_data = ir1[1];
            ir1[1] = (v467_data + (v459_data * v465_data));
            double v470_data = s0[42];
            double v472_data = ir1[2];
            ir1[2] = (v472_data + (v459_data * v470_data));
            double v475_data = s0[58];
            double v477_data = ir1[3];
            ir1[3] = (v477_data + (v459_data * v475_data));
            double v480_data = s0[74];
            double v482_data = ir1[4];
            ir1[4] = (v482_data + (v459_data * v480_data));
            double v485_data = s0[90];
            double v487_data = ir1[5];
            ir1[5] = (v487_data + (v459_data * v485_data));
            double v490_data = s0[106];
            double v492_data = ir1[6];
            ir1[6] = (v492_data + (v459_data * v490_data));
            double v495_data = s0[122];
            double v497_data = ir1[7];
            ir1[7] = (v497_data + (v459_data * v495_data));
            double v502_data = r0[11];
            double v503_data = s0[11];
            double v505_data = ir1[0];
            ir1[0] = (v505_data + (v502_data * v503_data));
            double v508_data = s0[27];
            double v510_data = ir1[1];
            ir1[1] = (v510_data + (v502_data * v508_data));
            double v513_data = s0[43];
            double v515_data = ir1[2];
            ir1[2] = (v515_data + (v502_data * v513_data));
            double v518_data = s0[59];
            double v520_data = ir1[3];
            ir1[3] = (v520_data + (v502_data * v518_data));
            double v523_data = s0[75];
            double v525_data = ir1[4];
            ir1[4] = (v525_data + (v502_data * v523_data));
            double v528_data = s0[91];
            double v530_data = ir1[5];
            ir1[5] = (v530_data + (v502_data * v528_data));
            double v533_data = s0[107];
            double v535_data = ir1[6];
            ir1[6] = (v535_data + (v502_data * v533_data));
            double v538_data = s0[123];
            double v540_data = ir1[7];
            ir1[7] = (v540_data + (v502_data * v538_data));
            double v545_data = r0[12];
            double v546_data = s0[12];
            double v548_data = ir1[0];
            ir1[0] = (v548_data + (v545_data * v546_data));
            double v551_data = s0[28];
            double v553_data = ir1[1];
            ir1[1] = (v553_data + (v545_data * v551_data));
            double v556_data = s0[44];
            double v558_data = ir1[2];
            ir1[2] = (v558_data + (v545_data * v556_data));
            double v561_data = s0[60];
            double v563_data = ir1[3];
            ir1[3] = (v563_data + (v545_data * v561_data));
            double v566_data = s0[76];
            double v568_data = ir1[4];
            ir1[4] = (v568_data + (v545_data * v566_data));
            double v571_data = s0[92];
            double v573_data = ir1[5];
            ir1[5] = (v573_data + (v545_data * v571_data));
            double v576_data = s0[108];
            double v578_data = ir1[6];
            ir1[6] = (v578_data + (v545_data * v576_data));
            double v581_data = s0[124];
            double v583_data = ir1[7];
            ir1[7] = (v583_data + (v545_data * v581_data));
            double v588_data = r0[13];
            double v589_data = s0[13];
            double v591_data = ir1[0];
            ir1[0] = (v591_data + (v588_data * v589_data));
            double v594_data = s0[29];
            double v596_data = ir1[1];
            ir1[1] = (v596_data + (v588_data * v594_data));
            double v599_data = s0[45];
            double v601_data = ir1[2];
            ir1[2] = (v601_data + (v588_data * v599_data));
            double v604_data = s0[61];
            double v606_data = ir1[3];
            ir1[3] = (v606_data + (v588_data * v604_data));
            double v609_data = s0[77];
            double v611_data = ir1[4];
            ir1[4] = (v611_data + (v588_data * v609_data));
            double v614_data = s0[93];
            double v616_data = ir1[5];
            ir1[5] = (v616_data + (v588_data * v614_data));
            double v619_data = s0[109];
            double v621_data = ir1[6];
            ir1[6] = (v621_data + (v588_data * v619_data));
            double v624_data = s0[125];
            double v626_data = ir1[7];
            ir1[7] = (v626_data + (v588_data * v624_data));
            double v631_data = r0[14];
            double v632_data = s0[14];
            double v634_data = ir1[0];
            ir1[0] = (v634_data + (v631_data * v632_data));
            double v637_data = s0[30];
            double v639_data = ir1[1];
            ir1[1] = (v639_data + (v631_data * v637_data));
            double v642_data = s0[46];
            double v644_data = ir1[2];
            ir1[2] = (v644_data + (v631_data * v642_data));
            double v647_data = s0[62];
            double v649_data = ir1[3];
            ir1[3] = (v649_data + (v631_data * v647_data));
            double v652_data = s0[78];
            double v654_data = ir1[4];
            ir1[4] = (v654_data + (v631_data * v652_data));
            double v657_data = s0[94];
            double v659_data = ir1[5];
            ir1[5] = (v659_data + (v631_data * v657_data));
            double v662_data = s0[110];
            double v664_data = ir1[6];
            ir1[6] = (v664_data + (v631_data * v662_data));
            double v667_data = s0[126];
            double v669_data = ir1[7];
            ir1[7] = (v669_data + (v631_data * v667_data));
            double v674_data = r0[15];
            double v675_data = s0[15];
            double v677_data = ir1[0];
            ir1[0] = (v677_data + (v674_data * v675_data));
            double v680_data = s0[31];
            double v682_data = ir1[1];
            ir1[1] = (v682_data + (v674_data * v680_data));
            double v685_data = s0[47];
            double v687_data = ir1[2];
            ir1[2] = (v687_data + (v674_data * v685_data));
            double v690_data = s0[63];
            double v692_data = ir1[3];
            ir1[3] = (v692_data + (v674_data * v690_data));
            double v695_data = s0[79];
            double v697_data = ir1[4];
            ir1[4] = (v697_data + (v674_data * v695_data));
            double v700_data = s0[95];
            double v702_data = ir1[5];
            ir1[5] = (v702_data + (v674_data * v700_data));
            double v705_data = s0[111];
            double v707_data = ir1[6];
            ir1[6] = (v707_data + (v674_data * v705_data));
            double v710_data = s0[127];
            double v712_data = ir1[7];
            ir1[7] = (v712_data + (v674_data * v710_data));
            #pragma unroll
            for (int32_t v717_n0 = 0; v717_n0 < 1; ++v717_n0) {
              #pragma unroll
              for (int32_t v718_n1 = 0; v718_n1 < 8; ++v718_n1) {
                int32_t v719_a = v717_n0 + v718_n1;
                int32_t v720_a = v717_n0 + v718_n1;
                double v721_data = ir1[v720_a];
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
              double v731_data = r1[(v727_i0 + v728_i1)];
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

