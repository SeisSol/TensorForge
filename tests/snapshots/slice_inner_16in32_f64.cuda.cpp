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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[batchId0 * 1024 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v16_lead = v11_i0 * 16;
            int32_t v18_off = (v10_lead + v16_lead) + 8;
            int32_t v26_off = (v10_lead + v16_lead) + 8;
            #pragma unroll
            for (int32_t v12_i1 = 8; v12_i1 < 24; ++v12_i1) {
              int32_t v19_a = v12_i1 * 32;
              int32_t v20_a = v18_off + v19_a;
              double v29_data = __ldcg(&glb_m1[(v26_off + v19_a)]);
              r0[(v11_i0 + (v12_i1 - 8))] = v29_data;
            }
          }
          double* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 8);
            __pipeline_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          double r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 16), (0, 8)] [(0, 16)]
          double ir1[8]{};
          double v39_data = r0[0];
          double v40_data = s0[0];
          double v42_data = ir1[0];
          ir1[0] = (v42_data + (v39_data * v40_data));
          double v45_data = s0[16];
          double v47_data = ir1[1];
          ir1[1] = (v47_data + (v39_data * v45_data));
          double v50_data = s0[32];
          double v52_data = ir1[2];
          ir1[2] = (v52_data + (v39_data * v50_data));
          double v55_data = s0[48];
          double v57_data = ir1[3];
          ir1[3] = (v57_data + (v39_data * v55_data));
          double v60_data = s0[64];
          double v62_data = ir1[4];
          ir1[4] = (v62_data + (v39_data * v60_data));
          double v65_data = s0[80];
          double v67_data = ir1[5];
          ir1[5] = (v67_data + (v39_data * v65_data));
          double v70_data = s0[96];
          double v72_data = ir1[6];
          ir1[6] = (v72_data + (v39_data * v70_data));
          double v75_data = s0[112];
          double v77_data = ir1[7];
          ir1[7] = (v77_data + (v39_data * v75_data));
          double v82_data = r0[1];
          double v83_data = s0[1];
          double v85_data = ir1[0];
          ir1[0] = (v85_data + (v82_data * v83_data));
          double v88_data = s0[17];
          double v90_data = ir1[1];
          ir1[1] = (v90_data + (v82_data * v88_data));
          double v93_data = s0[33];
          double v95_data = ir1[2];
          ir1[2] = (v95_data + (v82_data * v93_data));
          double v98_data = s0[49];
          double v100_data = ir1[3];
          ir1[3] = (v100_data + (v82_data * v98_data));
          double v103_data = s0[65];
          double v105_data = ir1[4];
          ir1[4] = (v105_data + (v82_data * v103_data));
          double v108_data = s0[81];
          double v110_data = ir1[5];
          ir1[5] = (v110_data + (v82_data * v108_data));
          double v113_data = s0[97];
          double v115_data = ir1[6];
          ir1[6] = (v115_data + (v82_data * v113_data));
          double v118_data = s0[113];
          double v120_data = ir1[7];
          ir1[7] = (v120_data + (v82_data * v118_data));
          double v125_data = r0[2];
          double v126_data = s0[2];
          double v128_data = ir1[0];
          ir1[0] = (v128_data + (v125_data * v126_data));
          double v131_data = s0[18];
          double v133_data = ir1[1];
          ir1[1] = (v133_data + (v125_data * v131_data));
          double v136_data = s0[34];
          double v138_data = ir1[2];
          ir1[2] = (v138_data + (v125_data * v136_data));
          double v141_data = s0[50];
          double v143_data = ir1[3];
          ir1[3] = (v143_data + (v125_data * v141_data));
          double v146_data = s0[66];
          double v148_data = ir1[4];
          ir1[4] = (v148_data + (v125_data * v146_data));
          double v151_data = s0[82];
          double v153_data = ir1[5];
          ir1[5] = (v153_data + (v125_data * v151_data));
          double v156_data = s0[98];
          double v158_data = ir1[6];
          ir1[6] = (v158_data + (v125_data * v156_data));
          double v161_data = s0[114];
          double v163_data = ir1[7];
          ir1[7] = (v163_data + (v125_data * v161_data));
          double v168_data = r0[3];
          double v169_data = s0[3];
          double v171_data = ir1[0];
          ir1[0] = (v171_data + (v168_data * v169_data));
          double v174_data = s0[19];
          double v176_data = ir1[1];
          ir1[1] = (v176_data + (v168_data * v174_data));
          double v179_data = s0[35];
          double v181_data = ir1[2];
          ir1[2] = (v181_data + (v168_data * v179_data));
          double v184_data = s0[51];
          double v186_data = ir1[3];
          ir1[3] = (v186_data + (v168_data * v184_data));
          double v189_data = s0[67];
          double v191_data = ir1[4];
          ir1[4] = (v191_data + (v168_data * v189_data));
          double v194_data = s0[83];
          double v196_data = ir1[5];
          ir1[5] = (v196_data + (v168_data * v194_data));
          double v199_data = s0[99];
          double v201_data = ir1[6];
          ir1[6] = (v201_data + (v168_data * v199_data));
          double v204_data = s0[115];
          double v206_data = ir1[7];
          ir1[7] = (v206_data + (v168_data * v204_data));
          double v211_data = r0[4];
          double v212_data = s0[4];
          double v214_data = ir1[0];
          ir1[0] = (v214_data + (v211_data * v212_data));
          double v217_data = s0[20];
          double v219_data = ir1[1];
          ir1[1] = (v219_data + (v211_data * v217_data));
          double v222_data = s0[36];
          double v224_data = ir1[2];
          ir1[2] = (v224_data + (v211_data * v222_data));
          double v227_data = s0[52];
          double v229_data = ir1[3];
          ir1[3] = (v229_data + (v211_data * v227_data));
          double v232_data = s0[68];
          double v234_data = ir1[4];
          ir1[4] = (v234_data + (v211_data * v232_data));
          double v237_data = s0[84];
          double v239_data = ir1[5];
          ir1[5] = (v239_data + (v211_data * v237_data));
          double v242_data = s0[100];
          double v244_data = ir1[6];
          ir1[6] = (v244_data + (v211_data * v242_data));
          double v247_data = s0[116];
          double v249_data = ir1[7];
          ir1[7] = (v249_data + (v211_data * v247_data));
          double v254_data = r0[5];
          double v255_data = s0[5];
          double v257_data = ir1[0];
          ir1[0] = (v257_data + (v254_data * v255_data));
          double v260_data = s0[21];
          double v262_data = ir1[1];
          ir1[1] = (v262_data + (v254_data * v260_data));
          double v265_data = s0[37];
          double v267_data = ir1[2];
          ir1[2] = (v267_data + (v254_data * v265_data));
          double v270_data = s0[53];
          double v272_data = ir1[3];
          ir1[3] = (v272_data + (v254_data * v270_data));
          double v275_data = s0[69];
          double v277_data = ir1[4];
          ir1[4] = (v277_data + (v254_data * v275_data));
          double v280_data = s0[85];
          double v282_data = ir1[5];
          ir1[5] = (v282_data + (v254_data * v280_data));
          double v285_data = s0[101];
          double v287_data = ir1[6];
          ir1[6] = (v287_data + (v254_data * v285_data));
          double v290_data = s0[117];
          double v292_data = ir1[7];
          ir1[7] = (v292_data + (v254_data * v290_data));
          double v297_data = r0[6];
          double v298_data = s0[6];
          double v300_data = ir1[0];
          ir1[0] = (v300_data + (v297_data * v298_data));
          double v303_data = s0[22];
          double v305_data = ir1[1];
          ir1[1] = (v305_data + (v297_data * v303_data));
          double v308_data = s0[38];
          double v310_data = ir1[2];
          ir1[2] = (v310_data + (v297_data * v308_data));
          double v313_data = s0[54];
          double v315_data = ir1[3];
          ir1[3] = (v315_data + (v297_data * v313_data));
          double v318_data = s0[70];
          double v320_data = ir1[4];
          ir1[4] = (v320_data + (v297_data * v318_data));
          double v323_data = s0[86];
          double v325_data = ir1[5];
          ir1[5] = (v325_data + (v297_data * v323_data));
          double v328_data = s0[102];
          double v330_data = ir1[6];
          ir1[6] = (v330_data + (v297_data * v328_data));
          double v333_data = s0[118];
          double v335_data = ir1[7];
          ir1[7] = (v335_data + (v297_data * v333_data));
          double v340_data = r0[7];
          double v341_data = s0[7];
          double v343_data = ir1[0];
          ir1[0] = (v343_data + (v340_data * v341_data));
          double v346_data = s0[23];
          double v348_data = ir1[1];
          ir1[1] = (v348_data + (v340_data * v346_data));
          double v351_data = s0[39];
          double v353_data = ir1[2];
          ir1[2] = (v353_data + (v340_data * v351_data));
          double v356_data = s0[55];
          double v358_data = ir1[3];
          ir1[3] = (v358_data + (v340_data * v356_data));
          double v361_data = s0[71];
          double v363_data = ir1[4];
          ir1[4] = (v363_data + (v340_data * v361_data));
          double v366_data = s0[87];
          double v368_data = ir1[5];
          ir1[5] = (v368_data + (v340_data * v366_data));
          double v371_data = s0[103];
          double v373_data = ir1[6];
          ir1[6] = (v373_data + (v340_data * v371_data));
          double v376_data = s0[119];
          double v378_data = ir1[7];
          ir1[7] = (v378_data + (v340_data * v376_data));
          double v383_data = r0[8];
          double v384_data = s0[8];
          double v386_data = ir1[0];
          ir1[0] = (v386_data + (v383_data * v384_data));
          double v389_data = s0[24];
          double v391_data = ir1[1];
          ir1[1] = (v391_data + (v383_data * v389_data));
          double v394_data = s0[40];
          double v396_data = ir1[2];
          ir1[2] = (v396_data + (v383_data * v394_data));
          double v399_data = s0[56];
          double v401_data = ir1[3];
          ir1[3] = (v401_data + (v383_data * v399_data));
          double v404_data = s0[72];
          double v406_data = ir1[4];
          ir1[4] = (v406_data + (v383_data * v404_data));
          double v409_data = s0[88];
          double v411_data = ir1[5];
          ir1[5] = (v411_data + (v383_data * v409_data));
          double v414_data = s0[104];
          double v416_data = ir1[6];
          ir1[6] = (v416_data + (v383_data * v414_data));
          double v419_data = s0[120];
          double v421_data = ir1[7];
          ir1[7] = (v421_data + (v383_data * v419_data));
          double v426_data = r0[9];
          double v427_data = s0[9];
          double v429_data = ir1[0];
          ir1[0] = (v429_data + (v426_data * v427_data));
          double v432_data = s0[25];
          double v434_data = ir1[1];
          ir1[1] = (v434_data + (v426_data * v432_data));
          double v437_data = s0[41];
          double v439_data = ir1[2];
          ir1[2] = (v439_data + (v426_data * v437_data));
          double v442_data = s0[57];
          double v444_data = ir1[3];
          ir1[3] = (v444_data + (v426_data * v442_data));
          double v447_data = s0[73];
          double v449_data = ir1[4];
          ir1[4] = (v449_data + (v426_data * v447_data));
          double v452_data = s0[89];
          double v454_data = ir1[5];
          ir1[5] = (v454_data + (v426_data * v452_data));
          double v457_data = s0[105];
          double v459_data = ir1[6];
          ir1[6] = (v459_data + (v426_data * v457_data));
          double v462_data = s0[121];
          double v464_data = ir1[7];
          ir1[7] = (v464_data + (v426_data * v462_data));
          double v469_data = r0[10];
          double v470_data = s0[10];
          double v472_data = ir1[0];
          ir1[0] = (v472_data + (v469_data * v470_data));
          double v475_data = s0[26];
          double v477_data = ir1[1];
          ir1[1] = (v477_data + (v469_data * v475_data));
          double v480_data = s0[42];
          double v482_data = ir1[2];
          ir1[2] = (v482_data + (v469_data * v480_data));
          double v485_data = s0[58];
          double v487_data = ir1[3];
          ir1[3] = (v487_data + (v469_data * v485_data));
          double v490_data = s0[74];
          double v492_data = ir1[4];
          ir1[4] = (v492_data + (v469_data * v490_data));
          double v495_data = s0[90];
          double v497_data = ir1[5];
          ir1[5] = (v497_data + (v469_data * v495_data));
          double v500_data = s0[106];
          double v502_data = ir1[6];
          ir1[6] = (v502_data + (v469_data * v500_data));
          double v505_data = s0[122];
          double v507_data = ir1[7];
          ir1[7] = (v507_data + (v469_data * v505_data));
          double v512_data = r0[11];
          double v513_data = s0[11];
          double v515_data = ir1[0];
          ir1[0] = (v515_data + (v512_data * v513_data));
          double v518_data = s0[27];
          double v520_data = ir1[1];
          ir1[1] = (v520_data + (v512_data * v518_data));
          double v523_data = s0[43];
          double v525_data = ir1[2];
          ir1[2] = (v525_data + (v512_data * v523_data));
          double v528_data = s0[59];
          double v530_data = ir1[3];
          ir1[3] = (v530_data + (v512_data * v528_data));
          double v533_data = s0[75];
          double v535_data = ir1[4];
          ir1[4] = (v535_data + (v512_data * v533_data));
          double v538_data = s0[91];
          double v540_data = ir1[5];
          ir1[5] = (v540_data + (v512_data * v538_data));
          double v543_data = s0[107];
          double v545_data = ir1[6];
          ir1[6] = (v545_data + (v512_data * v543_data));
          double v548_data = s0[123];
          double v550_data = ir1[7];
          ir1[7] = (v550_data + (v512_data * v548_data));
          double v555_data = r0[12];
          double v556_data = s0[12];
          double v558_data = ir1[0];
          ir1[0] = (v558_data + (v555_data * v556_data));
          double v561_data = s0[28];
          double v563_data = ir1[1];
          ir1[1] = (v563_data + (v555_data * v561_data));
          double v566_data = s0[44];
          double v568_data = ir1[2];
          ir1[2] = (v568_data + (v555_data * v566_data));
          double v571_data = s0[60];
          double v573_data = ir1[3];
          ir1[3] = (v573_data + (v555_data * v571_data));
          double v576_data = s0[76];
          double v578_data = ir1[4];
          ir1[4] = (v578_data + (v555_data * v576_data));
          double v581_data = s0[92];
          double v583_data = ir1[5];
          ir1[5] = (v583_data + (v555_data * v581_data));
          double v586_data = s0[108];
          double v588_data = ir1[6];
          ir1[6] = (v588_data + (v555_data * v586_data));
          double v591_data = s0[124];
          double v593_data = ir1[7];
          ir1[7] = (v593_data + (v555_data * v591_data));
          double v598_data = r0[13];
          double v599_data = s0[13];
          double v601_data = ir1[0];
          ir1[0] = (v601_data + (v598_data * v599_data));
          double v604_data = s0[29];
          double v606_data = ir1[1];
          ir1[1] = (v606_data + (v598_data * v604_data));
          double v609_data = s0[45];
          double v611_data = ir1[2];
          ir1[2] = (v611_data + (v598_data * v609_data));
          double v614_data = s0[61];
          double v616_data = ir1[3];
          ir1[3] = (v616_data + (v598_data * v614_data));
          double v619_data = s0[77];
          double v621_data = ir1[4];
          ir1[4] = (v621_data + (v598_data * v619_data));
          double v624_data = s0[93];
          double v626_data = ir1[5];
          ir1[5] = (v626_data + (v598_data * v624_data));
          double v629_data = s0[109];
          double v631_data = ir1[6];
          ir1[6] = (v631_data + (v598_data * v629_data));
          double v634_data = s0[125];
          double v636_data = ir1[7];
          ir1[7] = (v636_data + (v598_data * v634_data));
          double v641_data = r0[14];
          double v642_data = s0[14];
          double v644_data = ir1[0];
          ir1[0] = (v644_data + (v641_data * v642_data));
          double v647_data = s0[30];
          double v649_data = ir1[1];
          ir1[1] = (v649_data + (v641_data * v647_data));
          double v652_data = s0[46];
          double v654_data = ir1[2];
          ir1[2] = (v654_data + (v641_data * v652_data));
          double v657_data = s0[62];
          double v659_data = ir1[3];
          ir1[3] = (v659_data + (v641_data * v657_data));
          double v662_data = s0[78];
          double v664_data = ir1[4];
          ir1[4] = (v664_data + (v641_data * v662_data));
          double v667_data = s0[94];
          double v669_data = ir1[5];
          ir1[5] = (v669_data + (v641_data * v667_data));
          double v672_data = s0[110];
          double v674_data = ir1[6];
          ir1[6] = (v674_data + (v641_data * v672_data));
          double v677_data = s0[126];
          double v679_data = ir1[7];
          ir1[7] = (v679_data + (v641_data * v677_data));
          double v684_data = r0[15];
          double v685_data = s0[15];
          double v687_data = ir1[0];
          ir1[0] = (v687_data + (v684_data * v685_data));
          double v690_data = s0[31];
          double v692_data = ir1[1];
          ir1[1] = (v692_data + (v684_data * v690_data));
          double v695_data = s0[47];
          double v697_data = ir1[2];
          ir1[2] = (v697_data + (v684_data * v695_data));
          double v700_data = s0[63];
          double v702_data = ir1[3];
          ir1[3] = (v702_data + (v684_data * v700_data));
          double v705_data = s0[79];
          double v707_data = ir1[4];
          ir1[4] = (v707_data + (v684_data * v705_data));
          double v710_data = s0[95];
          double v712_data = ir1[5];
          ir1[5] = (v712_data + (v684_data * v710_data));
          double v715_data = s0[111];
          double v717_data = ir1[6];
          ir1[6] = (v717_data + (v684_data * v715_data));
          double v720_data = s0[127];
          double v722_data = ir1[7];
          ir1[7] = (v722_data + (v684_data * v720_data));
          #pragma unroll
          for (int32_t v727_n0 = 0; v727_n0 < 1; ++v727_n0) {
            #pragma unroll
            for (int32_t v728_n1 = 0; v728_n1 < 8; ++v728_n1) {
              int32_t v729_a = v727_n0 + v728_n1;
              int32_t v730_a = v727_n0 + v728_n1;
              double v731_data = ir1[v730_a];
              r1[v730_a] = v731_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v736_i0 = 0; v736_i0 < 1; ++v736_i0) {
            int32_t v745_lead = v10_lead + (v736_i0 * 16);
            #pragma unroll
            for (int32_t v737_i1 = 0; v737_i1 < 8; ++v737_i1) {
              int32_t v738_a = v736_i0 + v737_i1;
              double v740_data = r1[(v736_i0 + v737_i1)];
              glb_m0[(v745_lead + (v737_i1 * 16))] = v740_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

