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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v16_lead = v11_i0 * 16;
            int32_t v17_lead = v10_lead + v16_lead;
            int32_t v24_lead = v10_lead + v16_lead;
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 16; ++v12_i1) {
              int32_t v18_a = v12_i1 * 16;
              int32_t v19_a = v17_lead + v18_a;
              float v27_data = __ldcg(&glb_m1[(v24_lead + v18_a)]);
              r0[(v11_i0 + v12_i1)] = v27_data;
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
          float v37_data = r0[0];
          float v38_data = s0[0];
          float v40_data = ir1[0];
          ir1[0] = (v40_data + (v37_data * v38_data));
          float v43_data = s0[16];
          float v45_data = ir1[1];
          ir1[1] = (v45_data + (v37_data * v43_data));
          float v48_data = s0[32];
          float v50_data = ir1[2];
          ir1[2] = (v50_data + (v37_data * v48_data));
          float v53_data = s0[48];
          float v55_data = ir1[3];
          ir1[3] = (v55_data + (v37_data * v53_data));
          float v58_data = s0[64];
          float v60_data = ir1[4];
          ir1[4] = (v60_data + (v37_data * v58_data));
          float v63_data = s0[80];
          float v65_data = ir1[5];
          ir1[5] = (v65_data + (v37_data * v63_data));
          float v68_data = s0[96];
          float v70_data = ir1[6];
          ir1[6] = (v70_data + (v37_data * v68_data));
          float v73_data = s0[112];
          float v75_data = ir1[7];
          ir1[7] = (v75_data + (v37_data * v73_data));
          float v80_data = r0[1];
          float v81_data = s0[1];
          float v83_data = ir1[0];
          ir1[0] = (v83_data + (v80_data * v81_data));
          float v86_data = s0[17];
          float v88_data = ir1[1];
          ir1[1] = (v88_data + (v80_data * v86_data));
          float v91_data = s0[33];
          float v93_data = ir1[2];
          ir1[2] = (v93_data + (v80_data * v91_data));
          float v96_data = s0[49];
          float v98_data = ir1[3];
          ir1[3] = (v98_data + (v80_data * v96_data));
          float v101_data = s0[65];
          float v103_data = ir1[4];
          ir1[4] = (v103_data + (v80_data * v101_data));
          float v106_data = s0[81];
          float v108_data = ir1[5];
          ir1[5] = (v108_data + (v80_data * v106_data));
          float v111_data = s0[97];
          float v113_data = ir1[6];
          ir1[6] = (v113_data + (v80_data * v111_data));
          float v116_data = s0[113];
          float v118_data = ir1[7];
          ir1[7] = (v118_data + (v80_data * v116_data));
          float v123_data = r0[2];
          float v124_data = s0[2];
          float v126_data = ir1[0];
          ir1[0] = (v126_data + (v123_data * v124_data));
          float v129_data = s0[18];
          float v131_data = ir1[1];
          ir1[1] = (v131_data + (v123_data * v129_data));
          float v134_data = s0[34];
          float v136_data = ir1[2];
          ir1[2] = (v136_data + (v123_data * v134_data));
          float v139_data = s0[50];
          float v141_data = ir1[3];
          ir1[3] = (v141_data + (v123_data * v139_data));
          float v144_data = s0[66];
          float v146_data = ir1[4];
          ir1[4] = (v146_data + (v123_data * v144_data));
          float v149_data = s0[82];
          float v151_data = ir1[5];
          ir1[5] = (v151_data + (v123_data * v149_data));
          float v154_data = s0[98];
          float v156_data = ir1[6];
          ir1[6] = (v156_data + (v123_data * v154_data));
          float v159_data = s0[114];
          float v161_data = ir1[7];
          ir1[7] = (v161_data + (v123_data * v159_data));
          float v166_data = r0[3];
          float v167_data = s0[3];
          float v169_data = ir1[0];
          ir1[0] = (v169_data + (v166_data * v167_data));
          float v172_data = s0[19];
          float v174_data = ir1[1];
          ir1[1] = (v174_data + (v166_data * v172_data));
          float v177_data = s0[35];
          float v179_data = ir1[2];
          ir1[2] = (v179_data + (v166_data * v177_data));
          float v182_data = s0[51];
          float v184_data = ir1[3];
          ir1[3] = (v184_data + (v166_data * v182_data));
          float v187_data = s0[67];
          float v189_data = ir1[4];
          ir1[4] = (v189_data + (v166_data * v187_data));
          float v192_data = s0[83];
          float v194_data = ir1[5];
          ir1[5] = (v194_data + (v166_data * v192_data));
          float v197_data = s0[99];
          float v199_data = ir1[6];
          ir1[6] = (v199_data + (v166_data * v197_data));
          float v202_data = s0[115];
          float v204_data = ir1[7];
          ir1[7] = (v204_data + (v166_data * v202_data));
          float v209_data = r0[4];
          float v210_data = s0[4];
          float v212_data = ir1[0];
          ir1[0] = (v212_data + (v209_data * v210_data));
          float v215_data = s0[20];
          float v217_data = ir1[1];
          ir1[1] = (v217_data + (v209_data * v215_data));
          float v220_data = s0[36];
          float v222_data = ir1[2];
          ir1[2] = (v222_data + (v209_data * v220_data));
          float v225_data = s0[52];
          float v227_data = ir1[3];
          ir1[3] = (v227_data + (v209_data * v225_data));
          float v230_data = s0[68];
          float v232_data = ir1[4];
          ir1[4] = (v232_data + (v209_data * v230_data));
          float v235_data = s0[84];
          float v237_data = ir1[5];
          ir1[5] = (v237_data + (v209_data * v235_data));
          float v240_data = s0[100];
          float v242_data = ir1[6];
          ir1[6] = (v242_data + (v209_data * v240_data));
          float v245_data = s0[116];
          float v247_data = ir1[7];
          ir1[7] = (v247_data + (v209_data * v245_data));
          float v252_data = r0[5];
          float v253_data = s0[5];
          float v255_data = ir1[0];
          ir1[0] = (v255_data + (v252_data * v253_data));
          float v258_data = s0[21];
          float v260_data = ir1[1];
          ir1[1] = (v260_data + (v252_data * v258_data));
          float v263_data = s0[37];
          float v265_data = ir1[2];
          ir1[2] = (v265_data + (v252_data * v263_data));
          float v268_data = s0[53];
          float v270_data = ir1[3];
          ir1[3] = (v270_data + (v252_data * v268_data));
          float v273_data = s0[69];
          float v275_data = ir1[4];
          ir1[4] = (v275_data + (v252_data * v273_data));
          float v278_data = s0[85];
          float v280_data = ir1[5];
          ir1[5] = (v280_data + (v252_data * v278_data));
          float v283_data = s0[101];
          float v285_data = ir1[6];
          ir1[6] = (v285_data + (v252_data * v283_data));
          float v288_data = s0[117];
          float v290_data = ir1[7];
          ir1[7] = (v290_data + (v252_data * v288_data));
          float v295_data = r0[6];
          float v296_data = s0[6];
          float v298_data = ir1[0];
          ir1[0] = (v298_data + (v295_data * v296_data));
          float v301_data = s0[22];
          float v303_data = ir1[1];
          ir1[1] = (v303_data + (v295_data * v301_data));
          float v306_data = s0[38];
          float v308_data = ir1[2];
          ir1[2] = (v308_data + (v295_data * v306_data));
          float v311_data = s0[54];
          float v313_data = ir1[3];
          ir1[3] = (v313_data + (v295_data * v311_data));
          float v316_data = s0[70];
          float v318_data = ir1[4];
          ir1[4] = (v318_data + (v295_data * v316_data));
          float v321_data = s0[86];
          float v323_data = ir1[5];
          ir1[5] = (v323_data + (v295_data * v321_data));
          float v326_data = s0[102];
          float v328_data = ir1[6];
          ir1[6] = (v328_data + (v295_data * v326_data));
          float v331_data = s0[118];
          float v333_data = ir1[7];
          ir1[7] = (v333_data + (v295_data * v331_data));
          float v338_data = r0[7];
          float v339_data = s0[7];
          float v341_data = ir1[0];
          ir1[0] = (v341_data + (v338_data * v339_data));
          float v344_data = s0[23];
          float v346_data = ir1[1];
          ir1[1] = (v346_data + (v338_data * v344_data));
          float v349_data = s0[39];
          float v351_data = ir1[2];
          ir1[2] = (v351_data + (v338_data * v349_data));
          float v354_data = s0[55];
          float v356_data = ir1[3];
          ir1[3] = (v356_data + (v338_data * v354_data));
          float v359_data = s0[71];
          float v361_data = ir1[4];
          ir1[4] = (v361_data + (v338_data * v359_data));
          float v364_data = s0[87];
          float v366_data = ir1[5];
          ir1[5] = (v366_data + (v338_data * v364_data));
          float v369_data = s0[103];
          float v371_data = ir1[6];
          ir1[6] = (v371_data + (v338_data * v369_data));
          float v374_data = s0[119];
          float v376_data = ir1[7];
          ir1[7] = (v376_data + (v338_data * v374_data));
          float v381_data = r0[8];
          float v382_data = s0[8];
          float v384_data = ir1[0];
          ir1[0] = (v384_data + (v381_data * v382_data));
          float v387_data = s0[24];
          float v389_data = ir1[1];
          ir1[1] = (v389_data + (v381_data * v387_data));
          float v392_data = s0[40];
          float v394_data = ir1[2];
          ir1[2] = (v394_data + (v381_data * v392_data));
          float v397_data = s0[56];
          float v399_data = ir1[3];
          ir1[3] = (v399_data + (v381_data * v397_data));
          float v402_data = s0[72];
          float v404_data = ir1[4];
          ir1[4] = (v404_data + (v381_data * v402_data));
          float v407_data = s0[88];
          float v409_data = ir1[5];
          ir1[5] = (v409_data + (v381_data * v407_data));
          float v412_data = s0[104];
          float v414_data = ir1[6];
          ir1[6] = (v414_data + (v381_data * v412_data));
          float v417_data = s0[120];
          float v419_data = ir1[7];
          ir1[7] = (v419_data + (v381_data * v417_data));
          float v424_data = r0[9];
          float v425_data = s0[9];
          float v427_data = ir1[0];
          ir1[0] = (v427_data + (v424_data * v425_data));
          float v430_data = s0[25];
          float v432_data = ir1[1];
          ir1[1] = (v432_data + (v424_data * v430_data));
          float v435_data = s0[41];
          float v437_data = ir1[2];
          ir1[2] = (v437_data + (v424_data * v435_data));
          float v440_data = s0[57];
          float v442_data = ir1[3];
          ir1[3] = (v442_data + (v424_data * v440_data));
          float v445_data = s0[73];
          float v447_data = ir1[4];
          ir1[4] = (v447_data + (v424_data * v445_data));
          float v450_data = s0[89];
          float v452_data = ir1[5];
          ir1[5] = (v452_data + (v424_data * v450_data));
          float v455_data = s0[105];
          float v457_data = ir1[6];
          ir1[6] = (v457_data + (v424_data * v455_data));
          float v460_data = s0[121];
          float v462_data = ir1[7];
          ir1[7] = (v462_data + (v424_data * v460_data));
          float v467_data = r0[10];
          float v468_data = s0[10];
          float v470_data = ir1[0];
          ir1[0] = (v470_data + (v467_data * v468_data));
          float v473_data = s0[26];
          float v475_data = ir1[1];
          ir1[1] = (v475_data + (v467_data * v473_data));
          float v478_data = s0[42];
          float v480_data = ir1[2];
          ir1[2] = (v480_data + (v467_data * v478_data));
          float v483_data = s0[58];
          float v485_data = ir1[3];
          ir1[3] = (v485_data + (v467_data * v483_data));
          float v488_data = s0[74];
          float v490_data = ir1[4];
          ir1[4] = (v490_data + (v467_data * v488_data));
          float v493_data = s0[90];
          float v495_data = ir1[5];
          ir1[5] = (v495_data + (v467_data * v493_data));
          float v498_data = s0[106];
          float v500_data = ir1[6];
          ir1[6] = (v500_data + (v467_data * v498_data));
          float v503_data = s0[122];
          float v505_data = ir1[7];
          ir1[7] = (v505_data + (v467_data * v503_data));
          float v510_data = r0[11];
          float v511_data = s0[11];
          float v513_data = ir1[0];
          ir1[0] = (v513_data + (v510_data * v511_data));
          float v516_data = s0[27];
          float v518_data = ir1[1];
          ir1[1] = (v518_data + (v510_data * v516_data));
          float v521_data = s0[43];
          float v523_data = ir1[2];
          ir1[2] = (v523_data + (v510_data * v521_data));
          float v526_data = s0[59];
          float v528_data = ir1[3];
          ir1[3] = (v528_data + (v510_data * v526_data));
          float v531_data = s0[75];
          float v533_data = ir1[4];
          ir1[4] = (v533_data + (v510_data * v531_data));
          float v536_data = s0[91];
          float v538_data = ir1[5];
          ir1[5] = (v538_data + (v510_data * v536_data));
          float v541_data = s0[107];
          float v543_data = ir1[6];
          ir1[6] = (v543_data + (v510_data * v541_data));
          float v546_data = s0[123];
          float v548_data = ir1[7];
          ir1[7] = (v548_data + (v510_data * v546_data));
          float v553_data = r0[12];
          float v554_data = s0[12];
          float v556_data = ir1[0];
          ir1[0] = (v556_data + (v553_data * v554_data));
          float v559_data = s0[28];
          float v561_data = ir1[1];
          ir1[1] = (v561_data + (v553_data * v559_data));
          float v564_data = s0[44];
          float v566_data = ir1[2];
          ir1[2] = (v566_data + (v553_data * v564_data));
          float v569_data = s0[60];
          float v571_data = ir1[3];
          ir1[3] = (v571_data + (v553_data * v569_data));
          float v574_data = s0[76];
          float v576_data = ir1[4];
          ir1[4] = (v576_data + (v553_data * v574_data));
          float v579_data = s0[92];
          float v581_data = ir1[5];
          ir1[5] = (v581_data + (v553_data * v579_data));
          float v584_data = s0[108];
          float v586_data = ir1[6];
          ir1[6] = (v586_data + (v553_data * v584_data));
          float v589_data = s0[124];
          float v591_data = ir1[7];
          ir1[7] = (v591_data + (v553_data * v589_data));
          float v596_data = r0[13];
          float v597_data = s0[13];
          float v599_data = ir1[0];
          ir1[0] = (v599_data + (v596_data * v597_data));
          float v602_data = s0[29];
          float v604_data = ir1[1];
          ir1[1] = (v604_data + (v596_data * v602_data));
          float v607_data = s0[45];
          float v609_data = ir1[2];
          ir1[2] = (v609_data + (v596_data * v607_data));
          float v612_data = s0[61];
          float v614_data = ir1[3];
          ir1[3] = (v614_data + (v596_data * v612_data));
          float v617_data = s0[77];
          float v619_data = ir1[4];
          ir1[4] = (v619_data + (v596_data * v617_data));
          float v622_data = s0[93];
          float v624_data = ir1[5];
          ir1[5] = (v624_data + (v596_data * v622_data));
          float v627_data = s0[109];
          float v629_data = ir1[6];
          ir1[6] = (v629_data + (v596_data * v627_data));
          float v632_data = s0[125];
          float v634_data = ir1[7];
          ir1[7] = (v634_data + (v596_data * v632_data));
          float v639_data = r0[14];
          float v640_data = s0[14];
          float v642_data = ir1[0];
          ir1[0] = (v642_data + (v639_data * v640_data));
          float v645_data = s0[30];
          float v647_data = ir1[1];
          ir1[1] = (v647_data + (v639_data * v645_data));
          float v650_data = s0[46];
          float v652_data = ir1[2];
          ir1[2] = (v652_data + (v639_data * v650_data));
          float v655_data = s0[62];
          float v657_data = ir1[3];
          ir1[3] = (v657_data + (v639_data * v655_data));
          float v660_data = s0[78];
          float v662_data = ir1[4];
          ir1[4] = (v662_data + (v639_data * v660_data));
          float v665_data = s0[94];
          float v667_data = ir1[5];
          ir1[5] = (v667_data + (v639_data * v665_data));
          float v670_data = s0[110];
          float v672_data = ir1[6];
          ir1[6] = (v672_data + (v639_data * v670_data));
          float v675_data = s0[126];
          float v677_data = ir1[7];
          ir1[7] = (v677_data + (v639_data * v675_data));
          float v682_data = r0[15];
          float v683_data = s0[15];
          float v685_data = ir1[0];
          ir1[0] = (v685_data + (v682_data * v683_data));
          float v688_data = s0[31];
          float v690_data = ir1[1];
          ir1[1] = (v690_data + (v682_data * v688_data));
          float v693_data = s0[47];
          float v695_data = ir1[2];
          ir1[2] = (v695_data + (v682_data * v693_data));
          float v698_data = s0[63];
          float v700_data = ir1[3];
          ir1[3] = (v700_data + (v682_data * v698_data));
          float v703_data = s0[79];
          float v705_data = ir1[4];
          ir1[4] = (v705_data + (v682_data * v703_data));
          float v708_data = s0[95];
          float v710_data = ir1[5];
          ir1[5] = (v710_data + (v682_data * v708_data));
          float v713_data = s0[111];
          float v715_data = ir1[6];
          ir1[6] = (v715_data + (v682_data * v713_data));
          float v718_data = s0[127];
          float v720_data = ir1[7];
          ir1[7] = (v720_data + (v682_data * v718_data));
          #pragma unroll
          for (int32_t v725_n0 = 0; v725_n0 < 1; ++v725_n0) {
            #pragma unroll
            for (int32_t v726_n1 = 0; v726_n1 < 8; ++v726_n1) {
              int32_t v727_a = v725_n0 + v726_n1;
              int32_t v728_a = v725_n0 + v726_n1;
              float v729_data = ir1[v728_a];
              r1[v728_a] = v729_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v734_i0 = 0; v734_i0 < 1; ++v734_i0) {
            int32_t v743_lead = v10_lead + (v734_i0 * 16);
            #pragma unroll
            for (int32_t v735_i1 = 0; v735_i1 < 8; ++v735_i1) {
              int32_t v736_a = v734_i0 + v735_i1;
              float v738_data = r1[(v734_i0 + v735_i1)];
              glb_m0[(v743_lead + (v735_i1 * 16))] = v738_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

