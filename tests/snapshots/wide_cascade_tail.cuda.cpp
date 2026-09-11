// === base name ===
kernel_f2741f3ca1e25424

// === header ===
void launcher_kernel_f2741f3ca1e25424(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f2741f3ca1e25424(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_f2741f3ca1e25424, block.x * block.y * block.z, 896 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_f2741f3ca1e25424, cudaFuncAttributeMaxDynamicSharedMemorySize, 896 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_f2741f3ca1e25424<<<grid,block,896 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_f2741f3ca1e25424(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 24×9(24×9) {0..24}×{0..9} strided
    // m1 24×24(24×24) {0..24}×{0..24} strided
    // m2 24×9(24×9) {0..24}×{0..9} strided
    // m0 24×9(24×9) {0..24}×{0..9} strided({0..24}×{0..9})[0, 1] = m1 24×24(24×24) {0..24}×{0..24} strided({0..24}×{0..24})[0, -1]×m2 24×9(24×9) {0..24}×{0..9} strided({0..24}×{0..9})[-1, 1]
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[224 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[224];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v4_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v4_batchId0 < numElements0; v4_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v5_ahead1 = v4_batchId0 + (gridDim.x * blockDim.y);
        size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 216 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v4_batchId0 * 576 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v4_batchId0 * 216 + 0 + m2_extraOffset];
          float r0[24]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v18_lead = threadIdx.x % 32;
          if (v18_lead < 24) {
            #pragma unroll
            for (int32_t v20_i1 = 0; v20_i1 < 24; ++v20_i1) {
              float v28_data = __ldcg(&glb_m1[(v18_lead + (v20_i1 * 24))]);
              r0[v20_i1] = v28_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 4 * threadIdx.x + 0], &glb_m2[0 + 0 + 4 * threadIdx.x + 0], 16);
          __pipeline_memcpy_async(&s0[0 + 0 + 2 * threadIdx.x + 128], &glb_m2[0 + 0 + 2 * threadIdx.x + 128], 8);
          if (threadIdx.x < 24) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 192], &glb_m2[0 + 0 + 1 * threadIdx.x + 192], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[9]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 24), (0, 9)] [(0, 24)]
          float ir1[9]{};
          if (v18_lead < 24) {
            float v39_data = r0[0];
            float v40_data = s0[0];
            float v42_data = ir1[0];
            ir1[0] = (v42_data + (v39_data * v40_data));
            float v45_data = s0[24];
            float v47_data = ir1[1];
            ir1[1] = (v47_data + (v39_data * v45_data));
            float v50_data = s0[48];
            float v52_data = ir1[2];
            ir1[2] = (v52_data + (v39_data * v50_data));
            float v55_data = s0[72];
            float v57_data = ir1[3];
            ir1[3] = (v57_data + (v39_data * v55_data));
            float v60_data = s0[96];
            float v62_data = ir1[4];
            ir1[4] = (v62_data + (v39_data * v60_data));
            float v65_data = s0[120];
            float v67_data = ir1[5];
            ir1[5] = (v67_data + (v39_data * v65_data));
            float v70_data = s0[144];
            float v72_data = ir1[6];
            ir1[6] = (v72_data + (v39_data * v70_data));
            float v75_data = s0[168];
            float v77_data = ir1[7];
            ir1[7] = (v77_data + (v39_data * v75_data));
            float v80_data = s0[192];
            float v82_data = ir1[8];
            ir1[8] = (v82_data + (v39_data * v80_data));
          }
          if (v18_lead < 24) {
            float v88_data = r0[1];
            float v89_data = s0[1];
            float v91_data = ir1[0];
            ir1[0] = (v91_data + (v88_data * v89_data));
            float v94_data = s0[25];
            float v96_data = ir1[1];
            ir1[1] = (v96_data + (v88_data * v94_data));
            float v99_data = s0[49];
            float v101_data = ir1[2];
            ir1[2] = (v101_data + (v88_data * v99_data));
            float v104_data = s0[73];
            float v106_data = ir1[3];
            ir1[3] = (v106_data + (v88_data * v104_data));
            float v109_data = s0[97];
            float v111_data = ir1[4];
            ir1[4] = (v111_data + (v88_data * v109_data));
            float v114_data = s0[121];
            float v116_data = ir1[5];
            ir1[5] = (v116_data + (v88_data * v114_data));
            float v119_data = s0[145];
            float v121_data = ir1[6];
            ir1[6] = (v121_data + (v88_data * v119_data));
            float v124_data = s0[169];
            float v126_data = ir1[7];
            ir1[7] = (v126_data + (v88_data * v124_data));
            float v129_data = s0[193];
            float v131_data = ir1[8];
            ir1[8] = (v131_data + (v88_data * v129_data));
          }
          if (v18_lead < 24) {
            float v137_data = r0[2];
            float v138_data = s0[2];
            float v140_data = ir1[0];
            ir1[0] = (v140_data + (v137_data * v138_data));
            float v143_data = s0[26];
            float v145_data = ir1[1];
            ir1[1] = (v145_data + (v137_data * v143_data));
            float v148_data = s0[50];
            float v150_data = ir1[2];
            ir1[2] = (v150_data + (v137_data * v148_data));
            float v153_data = s0[74];
            float v155_data = ir1[3];
            ir1[3] = (v155_data + (v137_data * v153_data));
            float v158_data = s0[98];
            float v160_data = ir1[4];
            ir1[4] = (v160_data + (v137_data * v158_data));
            float v163_data = s0[122];
            float v165_data = ir1[5];
            ir1[5] = (v165_data + (v137_data * v163_data));
            float v168_data = s0[146];
            float v170_data = ir1[6];
            ir1[6] = (v170_data + (v137_data * v168_data));
            float v173_data = s0[170];
            float v175_data = ir1[7];
            ir1[7] = (v175_data + (v137_data * v173_data));
            float v178_data = s0[194];
            float v180_data = ir1[8];
            ir1[8] = (v180_data + (v137_data * v178_data));
          }
          if (v18_lead < 24) {
            float v186_data = r0[3];
            float v187_data = s0[3];
            float v189_data = ir1[0];
            ir1[0] = (v189_data + (v186_data * v187_data));
            float v192_data = s0[27];
            float v194_data = ir1[1];
            ir1[1] = (v194_data + (v186_data * v192_data));
            float v197_data = s0[51];
            float v199_data = ir1[2];
            ir1[2] = (v199_data + (v186_data * v197_data));
            float v202_data = s0[75];
            float v204_data = ir1[3];
            ir1[3] = (v204_data + (v186_data * v202_data));
            float v207_data = s0[99];
            float v209_data = ir1[4];
            ir1[4] = (v209_data + (v186_data * v207_data));
            float v212_data = s0[123];
            float v214_data = ir1[5];
            ir1[5] = (v214_data + (v186_data * v212_data));
            float v217_data = s0[147];
            float v219_data = ir1[6];
            ir1[6] = (v219_data + (v186_data * v217_data));
            float v222_data = s0[171];
            float v224_data = ir1[7];
            ir1[7] = (v224_data + (v186_data * v222_data));
            float v227_data = s0[195];
            float v229_data = ir1[8];
            ir1[8] = (v229_data + (v186_data * v227_data));
          }
          if (v18_lead < 24) {
            float v235_data = r0[4];
            float v236_data = s0[4];
            float v238_data = ir1[0];
            ir1[0] = (v238_data + (v235_data * v236_data));
            float v241_data = s0[28];
            float v243_data = ir1[1];
            ir1[1] = (v243_data + (v235_data * v241_data));
            float v246_data = s0[52];
            float v248_data = ir1[2];
            ir1[2] = (v248_data + (v235_data * v246_data));
            float v251_data = s0[76];
            float v253_data = ir1[3];
            ir1[3] = (v253_data + (v235_data * v251_data));
            float v256_data = s0[100];
            float v258_data = ir1[4];
            ir1[4] = (v258_data + (v235_data * v256_data));
            float v261_data = s0[124];
            float v263_data = ir1[5];
            ir1[5] = (v263_data + (v235_data * v261_data));
            float v266_data = s0[148];
            float v268_data = ir1[6];
            ir1[6] = (v268_data + (v235_data * v266_data));
            float v271_data = s0[172];
            float v273_data = ir1[7];
            ir1[7] = (v273_data + (v235_data * v271_data));
            float v276_data = s0[196];
            float v278_data = ir1[8];
            ir1[8] = (v278_data + (v235_data * v276_data));
          }
          if (v18_lead < 24) {
            float v284_data = r0[5];
            float v285_data = s0[5];
            float v287_data = ir1[0];
            ir1[0] = (v287_data + (v284_data * v285_data));
            float v290_data = s0[29];
            float v292_data = ir1[1];
            ir1[1] = (v292_data + (v284_data * v290_data));
            float v295_data = s0[53];
            float v297_data = ir1[2];
            ir1[2] = (v297_data + (v284_data * v295_data));
            float v300_data = s0[77];
            float v302_data = ir1[3];
            ir1[3] = (v302_data + (v284_data * v300_data));
            float v305_data = s0[101];
            float v307_data = ir1[4];
            ir1[4] = (v307_data + (v284_data * v305_data));
            float v310_data = s0[125];
            float v312_data = ir1[5];
            ir1[5] = (v312_data + (v284_data * v310_data));
            float v315_data = s0[149];
            float v317_data = ir1[6];
            ir1[6] = (v317_data + (v284_data * v315_data));
            float v320_data = s0[173];
            float v322_data = ir1[7];
            ir1[7] = (v322_data + (v284_data * v320_data));
            float v325_data = s0[197];
            float v327_data = ir1[8];
            ir1[8] = (v327_data + (v284_data * v325_data));
          }
          if (v18_lead < 24) {
            float v333_data = r0[6];
            float v334_data = s0[6];
            float v336_data = ir1[0];
            ir1[0] = (v336_data + (v333_data * v334_data));
            float v339_data = s0[30];
            float v341_data = ir1[1];
            ir1[1] = (v341_data + (v333_data * v339_data));
            float v344_data = s0[54];
            float v346_data = ir1[2];
            ir1[2] = (v346_data + (v333_data * v344_data));
            float v349_data = s0[78];
            float v351_data = ir1[3];
            ir1[3] = (v351_data + (v333_data * v349_data));
            float v354_data = s0[102];
            float v356_data = ir1[4];
            ir1[4] = (v356_data + (v333_data * v354_data));
            float v359_data = s0[126];
            float v361_data = ir1[5];
            ir1[5] = (v361_data + (v333_data * v359_data));
            float v364_data = s0[150];
            float v366_data = ir1[6];
            ir1[6] = (v366_data + (v333_data * v364_data));
            float v369_data = s0[174];
            float v371_data = ir1[7];
            ir1[7] = (v371_data + (v333_data * v369_data));
            float v374_data = s0[198];
            float v376_data = ir1[8];
            ir1[8] = (v376_data + (v333_data * v374_data));
          }
          if (v18_lead < 24) {
            float v382_data = r0[7];
            float v383_data = s0[7];
            float v385_data = ir1[0];
            ir1[0] = (v385_data + (v382_data * v383_data));
            float v388_data = s0[31];
            float v390_data = ir1[1];
            ir1[1] = (v390_data + (v382_data * v388_data));
            float v393_data = s0[55];
            float v395_data = ir1[2];
            ir1[2] = (v395_data + (v382_data * v393_data));
            float v398_data = s0[79];
            float v400_data = ir1[3];
            ir1[3] = (v400_data + (v382_data * v398_data));
            float v403_data = s0[103];
            float v405_data = ir1[4];
            ir1[4] = (v405_data + (v382_data * v403_data));
            float v408_data = s0[127];
            float v410_data = ir1[5];
            ir1[5] = (v410_data + (v382_data * v408_data));
            float v413_data = s0[151];
            float v415_data = ir1[6];
            ir1[6] = (v415_data + (v382_data * v413_data));
            float v418_data = s0[175];
            float v420_data = ir1[7];
            ir1[7] = (v420_data + (v382_data * v418_data));
            float v423_data = s0[199];
            float v425_data = ir1[8];
            ir1[8] = (v425_data + (v382_data * v423_data));
          }
          if (v18_lead < 24) {
            float v431_data = r0[8];
            float v432_data = s0[8];
            float v434_data = ir1[0];
            ir1[0] = (v434_data + (v431_data * v432_data));
            float v437_data = s0[32];
            float v439_data = ir1[1];
            ir1[1] = (v439_data + (v431_data * v437_data));
            float v442_data = s0[56];
            float v444_data = ir1[2];
            ir1[2] = (v444_data + (v431_data * v442_data));
            float v447_data = s0[80];
            float v449_data = ir1[3];
            ir1[3] = (v449_data + (v431_data * v447_data));
            float v452_data = s0[104];
            float v454_data = ir1[4];
            ir1[4] = (v454_data + (v431_data * v452_data));
            float v457_data = s0[128];
            float v459_data = ir1[5];
            ir1[5] = (v459_data + (v431_data * v457_data));
            float v462_data = s0[152];
            float v464_data = ir1[6];
            ir1[6] = (v464_data + (v431_data * v462_data));
            float v467_data = s0[176];
            float v469_data = ir1[7];
            ir1[7] = (v469_data + (v431_data * v467_data));
            float v472_data = s0[200];
            float v474_data = ir1[8];
            ir1[8] = (v474_data + (v431_data * v472_data));
          }
          if (v18_lead < 24) {
            float v480_data = r0[9];
            float v481_data = s0[9];
            float v483_data = ir1[0];
            ir1[0] = (v483_data + (v480_data * v481_data));
            float v486_data = s0[33];
            float v488_data = ir1[1];
            ir1[1] = (v488_data + (v480_data * v486_data));
            float v491_data = s0[57];
            float v493_data = ir1[2];
            ir1[2] = (v493_data + (v480_data * v491_data));
            float v496_data = s0[81];
            float v498_data = ir1[3];
            ir1[3] = (v498_data + (v480_data * v496_data));
            float v501_data = s0[105];
            float v503_data = ir1[4];
            ir1[4] = (v503_data + (v480_data * v501_data));
            float v506_data = s0[129];
            float v508_data = ir1[5];
            ir1[5] = (v508_data + (v480_data * v506_data));
            float v511_data = s0[153];
            float v513_data = ir1[6];
            ir1[6] = (v513_data + (v480_data * v511_data));
            float v516_data = s0[177];
            float v518_data = ir1[7];
            ir1[7] = (v518_data + (v480_data * v516_data));
            float v521_data = s0[201];
            float v523_data = ir1[8];
            ir1[8] = (v523_data + (v480_data * v521_data));
          }
          if (v18_lead < 24) {
            float v529_data = r0[10];
            float v530_data = s0[10];
            float v532_data = ir1[0];
            ir1[0] = (v532_data + (v529_data * v530_data));
            float v535_data = s0[34];
            float v537_data = ir1[1];
            ir1[1] = (v537_data + (v529_data * v535_data));
            float v540_data = s0[58];
            float v542_data = ir1[2];
            ir1[2] = (v542_data + (v529_data * v540_data));
            float v545_data = s0[82];
            float v547_data = ir1[3];
            ir1[3] = (v547_data + (v529_data * v545_data));
            float v550_data = s0[106];
            float v552_data = ir1[4];
            ir1[4] = (v552_data + (v529_data * v550_data));
            float v555_data = s0[130];
            float v557_data = ir1[5];
            ir1[5] = (v557_data + (v529_data * v555_data));
            float v560_data = s0[154];
            float v562_data = ir1[6];
            ir1[6] = (v562_data + (v529_data * v560_data));
            float v565_data = s0[178];
            float v567_data = ir1[7];
            ir1[7] = (v567_data + (v529_data * v565_data));
            float v570_data = s0[202];
            float v572_data = ir1[8];
            ir1[8] = (v572_data + (v529_data * v570_data));
          }
          if (v18_lead < 24) {
            float v578_data = r0[11];
            float v579_data = s0[11];
            float v581_data = ir1[0];
            ir1[0] = (v581_data + (v578_data * v579_data));
            float v584_data = s0[35];
            float v586_data = ir1[1];
            ir1[1] = (v586_data + (v578_data * v584_data));
            float v589_data = s0[59];
            float v591_data = ir1[2];
            ir1[2] = (v591_data + (v578_data * v589_data));
            float v594_data = s0[83];
            float v596_data = ir1[3];
            ir1[3] = (v596_data + (v578_data * v594_data));
            float v599_data = s0[107];
            float v601_data = ir1[4];
            ir1[4] = (v601_data + (v578_data * v599_data));
            float v604_data = s0[131];
            float v606_data = ir1[5];
            ir1[5] = (v606_data + (v578_data * v604_data));
            float v609_data = s0[155];
            float v611_data = ir1[6];
            ir1[6] = (v611_data + (v578_data * v609_data));
            float v614_data = s0[179];
            float v616_data = ir1[7];
            ir1[7] = (v616_data + (v578_data * v614_data));
            float v619_data = s0[203];
            float v621_data = ir1[8];
            ir1[8] = (v621_data + (v578_data * v619_data));
          }
          if (v18_lead < 24) {
            float v627_data = r0[12];
            float v628_data = s0[12];
            float v630_data = ir1[0];
            ir1[0] = (v630_data + (v627_data * v628_data));
            float v633_data = s0[36];
            float v635_data = ir1[1];
            ir1[1] = (v635_data + (v627_data * v633_data));
            float v638_data = s0[60];
            float v640_data = ir1[2];
            ir1[2] = (v640_data + (v627_data * v638_data));
            float v643_data = s0[84];
            float v645_data = ir1[3];
            ir1[3] = (v645_data + (v627_data * v643_data));
            float v648_data = s0[108];
            float v650_data = ir1[4];
            ir1[4] = (v650_data + (v627_data * v648_data));
            float v653_data = s0[132];
            float v655_data = ir1[5];
            ir1[5] = (v655_data + (v627_data * v653_data));
            float v658_data = s0[156];
            float v660_data = ir1[6];
            ir1[6] = (v660_data + (v627_data * v658_data));
            float v663_data = s0[180];
            float v665_data = ir1[7];
            ir1[7] = (v665_data + (v627_data * v663_data));
            float v668_data = s0[204];
            float v670_data = ir1[8];
            ir1[8] = (v670_data + (v627_data * v668_data));
          }
          if (v18_lead < 24) {
            float v676_data = r0[13];
            float v677_data = s0[13];
            float v679_data = ir1[0];
            ir1[0] = (v679_data + (v676_data * v677_data));
            float v682_data = s0[37];
            float v684_data = ir1[1];
            ir1[1] = (v684_data + (v676_data * v682_data));
            float v687_data = s0[61];
            float v689_data = ir1[2];
            ir1[2] = (v689_data + (v676_data * v687_data));
            float v692_data = s0[85];
            float v694_data = ir1[3];
            ir1[3] = (v694_data + (v676_data * v692_data));
            float v697_data = s0[109];
            float v699_data = ir1[4];
            ir1[4] = (v699_data + (v676_data * v697_data));
            float v702_data = s0[133];
            float v704_data = ir1[5];
            ir1[5] = (v704_data + (v676_data * v702_data));
            float v707_data = s0[157];
            float v709_data = ir1[6];
            ir1[6] = (v709_data + (v676_data * v707_data));
            float v712_data = s0[181];
            float v714_data = ir1[7];
            ir1[7] = (v714_data + (v676_data * v712_data));
            float v717_data = s0[205];
            float v719_data = ir1[8];
            ir1[8] = (v719_data + (v676_data * v717_data));
          }
          if (v18_lead < 24) {
            float v725_data = r0[14];
            float v726_data = s0[14];
            float v728_data = ir1[0];
            ir1[0] = (v728_data + (v725_data * v726_data));
            float v731_data = s0[38];
            float v733_data = ir1[1];
            ir1[1] = (v733_data + (v725_data * v731_data));
            float v736_data = s0[62];
            float v738_data = ir1[2];
            ir1[2] = (v738_data + (v725_data * v736_data));
            float v741_data = s0[86];
            float v743_data = ir1[3];
            ir1[3] = (v743_data + (v725_data * v741_data));
            float v746_data = s0[110];
            float v748_data = ir1[4];
            ir1[4] = (v748_data + (v725_data * v746_data));
            float v751_data = s0[134];
            float v753_data = ir1[5];
            ir1[5] = (v753_data + (v725_data * v751_data));
            float v756_data = s0[158];
            float v758_data = ir1[6];
            ir1[6] = (v758_data + (v725_data * v756_data));
            float v761_data = s0[182];
            float v763_data = ir1[7];
            ir1[7] = (v763_data + (v725_data * v761_data));
            float v766_data = s0[206];
            float v768_data = ir1[8];
            ir1[8] = (v768_data + (v725_data * v766_data));
          }
          if (v18_lead < 24) {
            float v774_data = r0[15];
            float v775_data = s0[15];
            float v777_data = ir1[0];
            ir1[0] = (v777_data + (v774_data * v775_data));
            float v780_data = s0[39];
            float v782_data = ir1[1];
            ir1[1] = (v782_data + (v774_data * v780_data));
            float v785_data = s0[63];
            float v787_data = ir1[2];
            ir1[2] = (v787_data + (v774_data * v785_data));
            float v790_data = s0[87];
            float v792_data = ir1[3];
            ir1[3] = (v792_data + (v774_data * v790_data));
            float v795_data = s0[111];
            float v797_data = ir1[4];
            ir1[4] = (v797_data + (v774_data * v795_data));
            float v800_data = s0[135];
            float v802_data = ir1[5];
            ir1[5] = (v802_data + (v774_data * v800_data));
            float v805_data = s0[159];
            float v807_data = ir1[6];
            ir1[6] = (v807_data + (v774_data * v805_data));
            float v810_data = s0[183];
            float v812_data = ir1[7];
            ir1[7] = (v812_data + (v774_data * v810_data));
            float v815_data = s0[207];
            float v817_data = ir1[8];
            ir1[8] = (v817_data + (v774_data * v815_data));
          }
          if (v18_lead < 24) {
            float v823_data = r0[16];
            float v824_data = s0[16];
            float v826_data = ir1[0];
            ir1[0] = (v826_data + (v823_data * v824_data));
            float v829_data = s0[40];
            float v831_data = ir1[1];
            ir1[1] = (v831_data + (v823_data * v829_data));
            float v834_data = s0[64];
            float v836_data = ir1[2];
            ir1[2] = (v836_data + (v823_data * v834_data));
            float v839_data = s0[88];
            float v841_data = ir1[3];
            ir1[3] = (v841_data + (v823_data * v839_data));
            float v844_data = s0[112];
            float v846_data = ir1[4];
            ir1[4] = (v846_data + (v823_data * v844_data));
            float v849_data = s0[136];
            float v851_data = ir1[5];
            ir1[5] = (v851_data + (v823_data * v849_data));
            float v854_data = s0[160];
            float v856_data = ir1[6];
            ir1[6] = (v856_data + (v823_data * v854_data));
            float v859_data = s0[184];
            float v861_data = ir1[7];
            ir1[7] = (v861_data + (v823_data * v859_data));
            float v864_data = s0[208];
            float v866_data = ir1[8];
            ir1[8] = (v866_data + (v823_data * v864_data));
          }
          if (v18_lead < 24) {
            float v872_data = r0[17];
            float v873_data = s0[17];
            float v875_data = ir1[0];
            ir1[0] = (v875_data + (v872_data * v873_data));
            float v878_data = s0[41];
            float v880_data = ir1[1];
            ir1[1] = (v880_data + (v872_data * v878_data));
            float v883_data = s0[65];
            float v885_data = ir1[2];
            ir1[2] = (v885_data + (v872_data * v883_data));
            float v888_data = s0[89];
            float v890_data = ir1[3];
            ir1[3] = (v890_data + (v872_data * v888_data));
            float v893_data = s0[113];
            float v895_data = ir1[4];
            ir1[4] = (v895_data + (v872_data * v893_data));
            float v898_data = s0[137];
            float v900_data = ir1[5];
            ir1[5] = (v900_data + (v872_data * v898_data));
            float v903_data = s0[161];
            float v905_data = ir1[6];
            ir1[6] = (v905_data + (v872_data * v903_data));
            float v908_data = s0[185];
            float v910_data = ir1[7];
            ir1[7] = (v910_data + (v872_data * v908_data));
            float v913_data = s0[209];
            float v915_data = ir1[8];
            ir1[8] = (v915_data + (v872_data * v913_data));
          }
          if (v18_lead < 24) {
            float v921_data = r0[18];
            float v922_data = s0[18];
            float v924_data = ir1[0];
            ir1[0] = (v924_data + (v921_data * v922_data));
            float v927_data = s0[42];
            float v929_data = ir1[1];
            ir1[1] = (v929_data + (v921_data * v927_data));
            float v932_data = s0[66];
            float v934_data = ir1[2];
            ir1[2] = (v934_data + (v921_data * v932_data));
            float v937_data = s0[90];
            float v939_data = ir1[3];
            ir1[3] = (v939_data + (v921_data * v937_data));
            float v942_data = s0[114];
            float v944_data = ir1[4];
            ir1[4] = (v944_data + (v921_data * v942_data));
            float v947_data = s0[138];
            float v949_data = ir1[5];
            ir1[5] = (v949_data + (v921_data * v947_data));
            float v952_data = s0[162];
            float v954_data = ir1[6];
            ir1[6] = (v954_data + (v921_data * v952_data));
            float v957_data = s0[186];
            float v959_data = ir1[7];
            ir1[7] = (v959_data + (v921_data * v957_data));
            float v962_data = s0[210];
            float v964_data = ir1[8];
            ir1[8] = (v964_data + (v921_data * v962_data));
          }
          if (v18_lead < 24) {
            float v970_data = r0[19];
            float v971_data = s0[19];
            float v973_data = ir1[0];
            ir1[0] = (v973_data + (v970_data * v971_data));
            float v976_data = s0[43];
            float v978_data = ir1[1];
            ir1[1] = (v978_data + (v970_data * v976_data));
            float v981_data = s0[67];
            float v983_data = ir1[2];
            ir1[2] = (v983_data + (v970_data * v981_data));
            float v986_data = s0[91];
            float v988_data = ir1[3];
            ir1[3] = (v988_data + (v970_data * v986_data));
            float v991_data = s0[115];
            float v993_data = ir1[4];
            ir1[4] = (v993_data + (v970_data * v991_data));
            float v996_data = s0[139];
            float v998_data = ir1[5];
            ir1[5] = (v998_data + (v970_data * v996_data));
            float v1001_data = s0[163];
            float v1003_data = ir1[6];
            ir1[6] = (v1003_data + (v970_data * v1001_data));
            float v1006_data = s0[187];
            float v1008_data = ir1[7];
            ir1[7] = (v1008_data + (v970_data * v1006_data));
            float v1011_data = s0[211];
            float v1013_data = ir1[8];
            ir1[8] = (v1013_data + (v970_data * v1011_data));
          }
          if (v18_lead < 24) {
            float v1019_data = r0[20];
            float v1020_data = s0[20];
            float v1022_data = ir1[0];
            ir1[0] = (v1022_data + (v1019_data * v1020_data));
            float v1025_data = s0[44];
            float v1027_data = ir1[1];
            ir1[1] = (v1027_data + (v1019_data * v1025_data));
            float v1030_data = s0[68];
            float v1032_data = ir1[2];
            ir1[2] = (v1032_data + (v1019_data * v1030_data));
            float v1035_data = s0[92];
            float v1037_data = ir1[3];
            ir1[3] = (v1037_data + (v1019_data * v1035_data));
            float v1040_data = s0[116];
            float v1042_data = ir1[4];
            ir1[4] = (v1042_data + (v1019_data * v1040_data));
            float v1045_data = s0[140];
            float v1047_data = ir1[5];
            ir1[5] = (v1047_data + (v1019_data * v1045_data));
            float v1050_data = s0[164];
            float v1052_data = ir1[6];
            ir1[6] = (v1052_data + (v1019_data * v1050_data));
            float v1055_data = s0[188];
            float v1057_data = ir1[7];
            ir1[7] = (v1057_data + (v1019_data * v1055_data));
            float v1060_data = s0[212];
            float v1062_data = ir1[8];
            ir1[8] = (v1062_data + (v1019_data * v1060_data));
          }
          if (v18_lead < 24) {
            float v1068_data = r0[21];
            float v1069_data = s0[21];
            float v1071_data = ir1[0];
            ir1[0] = (v1071_data + (v1068_data * v1069_data));
            float v1074_data = s0[45];
            float v1076_data = ir1[1];
            ir1[1] = (v1076_data + (v1068_data * v1074_data));
            float v1079_data = s0[69];
            float v1081_data = ir1[2];
            ir1[2] = (v1081_data + (v1068_data * v1079_data));
            float v1084_data = s0[93];
            float v1086_data = ir1[3];
            ir1[3] = (v1086_data + (v1068_data * v1084_data));
            float v1089_data = s0[117];
            float v1091_data = ir1[4];
            ir1[4] = (v1091_data + (v1068_data * v1089_data));
            float v1094_data = s0[141];
            float v1096_data = ir1[5];
            ir1[5] = (v1096_data + (v1068_data * v1094_data));
            float v1099_data = s0[165];
            float v1101_data = ir1[6];
            ir1[6] = (v1101_data + (v1068_data * v1099_data));
            float v1104_data = s0[189];
            float v1106_data = ir1[7];
            ir1[7] = (v1106_data + (v1068_data * v1104_data));
            float v1109_data = s0[213];
            float v1111_data = ir1[8];
            ir1[8] = (v1111_data + (v1068_data * v1109_data));
          }
          if (v18_lead < 24) {
            float v1117_data = r0[22];
            float v1118_data = s0[22];
            float v1120_data = ir1[0];
            ir1[0] = (v1120_data + (v1117_data * v1118_data));
            float v1123_data = s0[46];
            float v1125_data = ir1[1];
            ir1[1] = (v1125_data + (v1117_data * v1123_data));
            float v1128_data = s0[70];
            float v1130_data = ir1[2];
            ir1[2] = (v1130_data + (v1117_data * v1128_data));
            float v1133_data = s0[94];
            float v1135_data = ir1[3];
            ir1[3] = (v1135_data + (v1117_data * v1133_data));
            float v1138_data = s0[118];
            float v1140_data = ir1[4];
            ir1[4] = (v1140_data + (v1117_data * v1138_data));
            float v1143_data = s0[142];
            float v1145_data = ir1[5];
            ir1[5] = (v1145_data + (v1117_data * v1143_data));
            float v1148_data = s0[166];
            float v1150_data = ir1[6];
            ir1[6] = (v1150_data + (v1117_data * v1148_data));
            float v1153_data = s0[190];
            float v1155_data = ir1[7];
            ir1[7] = (v1155_data + (v1117_data * v1153_data));
            float v1158_data = s0[214];
            float v1160_data = ir1[8];
            ir1[8] = (v1160_data + (v1117_data * v1158_data));
          }
          if (v18_lead < 24) {
            float v1166_data = r0[23];
            float v1167_data = s0[23];
            float v1169_data = ir1[0];
            ir1[0] = (v1169_data + (v1166_data * v1167_data));
            float v1172_data = s0[47];
            float v1174_data = ir1[1];
            ir1[1] = (v1174_data + (v1166_data * v1172_data));
            float v1177_data = s0[71];
            float v1179_data = ir1[2];
            ir1[2] = (v1179_data + (v1166_data * v1177_data));
            float v1182_data = s0[95];
            float v1184_data = ir1[3];
            ir1[3] = (v1184_data + (v1166_data * v1182_data));
            float v1187_data = s0[119];
            float v1189_data = ir1[4];
            ir1[4] = (v1189_data + (v1166_data * v1187_data));
            float v1192_data = s0[143];
            float v1194_data = ir1[5];
            ir1[5] = (v1194_data + (v1166_data * v1192_data));
            float v1197_data = s0[167];
            float v1199_data = ir1[6];
            ir1[6] = (v1199_data + (v1166_data * v1197_data));
            float v1202_data = s0[191];
            float v1204_data = ir1[7];
            ir1[7] = (v1204_data + (v1166_data * v1202_data));
            float v1207_data = s0[215];
            float v1209_data = ir1[8];
            ir1[8] = (v1209_data + (v1166_data * v1207_data));
          }
          if (v18_lead < 24) {
            #pragma unroll
            for (int32_t v1215_n1 = 0; v1215_n1 < 9; ++v1215_n1) {
              float v1217_data = ir1[v1215_n1];
              r1[v1215_n1] = v1217_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v18_lead < 24) {
            #pragma unroll
            for (int32_t v1223_i1 = 0; v1223_i1 < 9; ++v1223_i1) {
              float v1225_data = r1[v1223_i1];
              glb_m0[(v18_lead + (v1223_i1 * 24))] = v1225_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

