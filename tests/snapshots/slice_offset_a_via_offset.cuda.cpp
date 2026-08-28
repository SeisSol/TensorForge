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
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 512 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          alignas(16) float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v6_lead = threadIdx.x % 16;
          if (v6_lead < 12) {
            int32_t v14_off = v6_lead + 4;
            int32_t v22_off = v6_lead + 4;
            #pragma unroll
            for (int32_t v8_i1 = 0; v8_i1 < 16; ++v8_i1) {
              int32_t v15_a = v8_i1 * 32;
              int32_t v16_a = v14_off + v15_a;
              float v25_data = __ldcg(&glb_m1[(v22_off + v15_a)]);
              int32_t v26_a = 0 + v8_i1;
              r0[v26_a] = v25_data;
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
          alignas(16) float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 8)] [(0, 16)]
          float ir1[8]{};
          if (v6_lead < 12) {
            float v35_data = r0[0];
            float v36_data = s0[0];
            float v38_data = ir1[0];
            ir1[0] = (v38_data + (v35_data * v36_data));
            float v41_data = s0[16];
            float v43_data = ir1[1];
            ir1[1] = (v43_data + (v35_data * v41_data));
            float v46_data = s0[32];
            float v48_data = ir1[2];
            ir1[2] = (v48_data + (v35_data * v46_data));
            float v51_data = s0[48];
            float v53_data = ir1[3];
            ir1[3] = (v53_data + (v35_data * v51_data));
            float v56_data = s0[64];
            float v58_data = ir1[4];
            ir1[4] = (v58_data + (v35_data * v56_data));
            float v61_data = s0[80];
            float v63_data = ir1[5];
            ir1[5] = (v63_data + (v35_data * v61_data));
            float v66_data = s0[96];
            float v68_data = ir1[6];
            ir1[6] = (v68_data + (v35_data * v66_data));
            float v71_data = s0[112];
            float v73_data = ir1[7];
            ir1[7] = (v73_data + (v35_data * v71_data));
          }
          if (v6_lead < 12) {
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
          }
          if (v6_lead < 12) {
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
          }
          if (v6_lead < 12) {
            float v167_data = r0[3];
            float v168_data = s0[3];
            float v170_data = ir1[0];
            ir1[0] = (v170_data + (v167_data * v168_data));
            float v173_data = s0[19];
            float v175_data = ir1[1];
            ir1[1] = (v175_data + (v167_data * v173_data));
            float v178_data = s0[35];
            float v180_data = ir1[2];
            ir1[2] = (v180_data + (v167_data * v178_data));
            float v183_data = s0[51];
            float v185_data = ir1[3];
            ir1[3] = (v185_data + (v167_data * v183_data));
            float v188_data = s0[67];
            float v190_data = ir1[4];
            ir1[4] = (v190_data + (v167_data * v188_data));
            float v193_data = s0[83];
            float v195_data = ir1[5];
            ir1[5] = (v195_data + (v167_data * v193_data));
            float v198_data = s0[99];
            float v200_data = ir1[6];
            ir1[6] = (v200_data + (v167_data * v198_data));
            float v203_data = s0[115];
            float v205_data = ir1[7];
            ir1[7] = (v205_data + (v167_data * v203_data));
          }
          if (v6_lead < 12) {
            float v211_data = r0[4];
            float v212_data = s0[4];
            float v214_data = ir1[0];
            ir1[0] = (v214_data + (v211_data * v212_data));
            float v217_data = s0[20];
            float v219_data = ir1[1];
            ir1[1] = (v219_data + (v211_data * v217_data));
            float v222_data = s0[36];
            float v224_data = ir1[2];
            ir1[2] = (v224_data + (v211_data * v222_data));
            float v227_data = s0[52];
            float v229_data = ir1[3];
            ir1[3] = (v229_data + (v211_data * v227_data));
            float v232_data = s0[68];
            float v234_data = ir1[4];
            ir1[4] = (v234_data + (v211_data * v232_data));
            float v237_data = s0[84];
            float v239_data = ir1[5];
            ir1[5] = (v239_data + (v211_data * v237_data));
            float v242_data = s0[100];
            float v244_data = ir1[6];
            ir1[6] = (v244_data + (v211_data * v242_data));
            float v247_data = s0[116];
            float v249_data = ir1[7];
            ir1[7] = (v249_data + (v211_data * v247_data));
          }
          if (v6_lead < 12) {
            float v255_data = r0[5];
            float v256_data = s0[5];
            float v258_data = ir1[0];
            ir1[0] = (v258_data + (v255_data * v256_data));
            float v261_data = s0[21];
            float v263_data = ir1[1];
            ir1[1] = (v263_data + (v255_data * v261_data));
            float v266_data = s0[37];
            float v268_data = ir1[2];
            ir1[2] = (v268_data + (v255_data * v266_data));
            float v271_data = s0[53];
            float v273_data = ir1[3];
            ir1[3] = (v273_data + (v255_data * v271_data));
            float v276_data = s0[69];
            float v278_data = ir1[4];
            ir1[4] = (v278_data + (v255_data * v276_data));
            float v281_data = s0[85];
            float v283_data = ir1[5];
            ir1[5] = (v283_data + (v255_data * v281_data));
            float v286_data = s0[101];
            float v288_data = ir1[6];
            ir1[6] = (v288_data + (v255_data * v286_data));
            float v291_data = s0[117];
            float v293_data = ir1[7];
            ir1[7] = (v293_data + (v255_data * v291_data));
          }
          if (v6_lead < 12) {
            float v299_data = r0[6];
            float v300_data = s0[6];
            float v302_data = ir1[0];
            ir1[0] = (v302_data + (v299_data * v300_data));
            float v305_data = s0[22];
            float v307_data = ir1[1];
            ir1[1] = (v307_data + (v299_data * v305_data));
            float v310_data = s0[38];
            float v312_data = ir1[2];
            ir1[2] = (v312_data + (v299_data * v310_data));
            float v315_data = s0[54];
            float v317_data = ir1[3];
            ir1[3] = (v317_data + (v299_data * v315_data));
            float v320_data = s0[70];
            float v322_data = ir1[4];
            ir1[4] = (v322_data + (v299_data * v320_data));
            float v325_data = s0[86];
            float v327_data = ir1[5];
            ir1[5] = (v327_data + (v299_data * v325_data));
            float v330_data = s0[102];
            float v332_data = ir1[6];
            ir1[6] = (v332_data + (v299_data * v330_data));
            float v335_data = s0[118];
            float v337_data = ir1[7];
            ir1[7] = (v337_data + (v299_data * v335_data));
          }
          if (v6_lead < 12) {
            float v343_data = r0[7];
            float v344_data = s0[7];
            float v346_data = ir1[0];
            ir1[0] = (v346_data + (v343_data * v344_data));
            float v349_data = s0[23];
            float v351_data = ir1[1];
            ir1[1] = (v351_data + (v343_data * v349_data));
            float v354_data = s0[39];
            float v356_data = ir1[2];
            ir1[2] = (v356_data + (v343_data * v354_data));
            float v359_data = s0[55];
            float v361_data = ir1[3];
            ir1[3] = (v361_data + (v343_data * v359_data));
            float v364_data = s0[71];
            float v366_data = ir1[4];
            ir1[4] = (v366_data + (v343_data * v364_data));
            float v369_data = s0[87];
            float v371_data = ir1[5];
            ir1[5] = (v371_data + (v343_data * v369_data));
            float v374_data = s0[103];
            float v376_data = ir1[6];
            ir1[6] = (v376_data + (v343_data * v374_data));
            float v379_data = s0[119];
            float v381_data = ir1[7];
            ir1[7] = (v381_data + (v343_data * v379_data));
          }
          if (v6_lead < 12) {
            float v387_data = r0[8];
            float v388_data = s0[8];
            float v390_data = ir1[0];
            ir1[0] = (v390_data + (v387_data * v388_data));
            float v393_data = s0[24];
            float v395_data = ir1[1];
            ir1[1] = (v395_data + (v387_data * v393_data));
            float v398_data = s0[40];
            float v400_data = ir1[2];
            ir1[2] = (v400_data + (v387_data * v398_data));
            float v403_data = s0[56];
            float v405_data = ir1[3];
            ir1[3] = (v405_data + (v387_data * v403_data));
            float v408_data = s0[72];
            float v410_data = ir1[4];
            ir1[4] = (v410_data + (v387_data * v408_data));
            float v413_data = s0[88];
            float v415_data = ir1[5];
            ir1[5] = (v415_data + (v387_data * v413_data));
            float v418_data = s0[104];
            float v420_data = ir1[6];
            ir1[6] = (v420_data + (v387_data * v418_data));
            float v423_data = s0[120];
            float v425_data = ir1[7];
            ir1[7] = (v425_data + (v387_data * v423_data));
          }
          if (v6_lead < 12) {
            float v431_data = r0[9];
            float v432_data = s0[9];
            float v434_data = ir1[0];
            ir1[0] = (v434_data + (v431_data * v432_data));
            float v437_data = s0[25];
            float v439_data = ir1[1];
            ir1[1] = (v439_data + (v431_data * v437_data));
            float v442_data = s0[41];
            float v444_data = ir1[2];
            ir1[2] = (v444_data + (v431_data * v442_data));
            float v447_data = s0[57];
            float v449_data = ir1[3];
            ir1[3] = (v449_data + (v431_data * v447_data));
            float v452_data = s0[73];
            float v454_data = ir1[4];
            ir1[4] = (v454_data + (v431_data * v452_data));
            float v457_data = s0[89];
            float v459_data = ir1[5];
            ir1[5] = (v459_data + (v431_data * v457_data));
            float v462_data = s0[105];
            float v464_data = ir1[6];
            ir1[6] = (v464_data + (v431_data * v462_data));
            float v467_data = s0[121];
            float v469_data = ir1[7];
            ir1[7] = (v469_data + (v431_data * v467_data));
          }
          if (v6_lead < 12) {
            float v475_data = r0[10];
            float v476_data = s0[10];
            float v478_data = ir1[0];
            ir1[0] = (v478_data + (v475_data * v476_data));
            float v481_data = s0[26];
            float v483_data = ir1[1];
            ir1[1] = (v483_data + (v475_data * v481_data));
            float v486_data = s0[42];
            float v488_data = ir1[2];
            ir1[2] = (v488_data + (v475_data * v486_data));
            float v491_data = s0[58];
            float v493_data = ir1[3];
            ir1[3] = (v493_data + (v475_data * v491_data));
            float v496_data = s0[74];
            float v498_data = ir1[4];
            ir1[4] = (v498_data + (v475_data * v496_data));
            float v501_data = s0[90];
            float v503_data = ir1[5];
            ir1[5] = (v503_data + (v475_data * v501_data));
            float v506_data = s0[106];
            float v508_data = ir1[6];
            ir1[6] = (v508_data + (v475_data * v506_data));
            float v511_data = s0[122];
            float v513_data = ir1[7];
            ir1[7] = (v513_data + (v475_data * v511_data));
          }
          if (v6_lead < 12) {
            float v519_data = r0[11];
            float v520_data = s0[11];
            float v522_data = ir1[0];
            ir1[0] = (v522_data + (v519_data * v520_data));
            float v525_data = s0[27];
            float v527_data = ir1[1];
            ir1[1] = (v527_data + (v519_data * v525_data));
            float v530_data = s0[43];
            float v532_data = ir1[2];
            ir1[2] = (v532_data + (v519_data * v530_data));
            float v535_data = s0[59];
            float v537_data = ir1[3];
            ir1[3] = (v537_data + (v519_data * v535_data));
            float v540_data = s0[75];
            float v542_data = ir1[4];
            ir1[4] = (v542_data + (v519_data * v540_data));
            float v545_data = s0[91];
            float v547_data = ir1[5];
            ir1[5] = (v547_data + (v519_data * v545_data));
            float v550_data = s0[107];
            float v552_data = ir1[6];
            ir1[6] = (v552_data + (v519_data * v550_data));
            float v555_data = s0[123];
            float v557_data = ir1[7];
            ir1[7] = (v557_data + (v519_data * v555_data));
          }
          if (v6_lead < 12) {
            float v563_data = r0[12];
            float v564_data = s0[12];
            float v566_data = ir1[0];
            ir1[0] = (v566_data + (v563_data * v564_data));
            float v569_data = s0[28];
            float v571_data = ir1[1];
            ir1[1] = (v571_data + (v563_data * v569_data));
            float v574_data = s0[44];
            float v576_data = ir1[2];
            ir1[2] = (v576_data + (v563_data * v574_data));
            float v579_data = s0[60];
            float v581_data = ir1[3];
            ir1[3] = (v581_data + (v563_data * v579_data));
            float v584_data = s0[76];
            float v586_data = ir1[4];
            ir1[4] = (v586_data + (v563_data * v584_data));
            float v589_data = s0[92];
            float v591_data = ir1[5];
            ir1[5] = (v591_data + (v563_data * v589_data));
            float v594_data = s0[108];
            float v596_data = ir1[6];
            ir1[6] = (v596_data + (v563_data * v594_data));
            float v599_data = s0[124];
            float v601_data = ir1[7];
            ir1[7] = (v601_data + (v563_data * v599_data));
          }
          if (v6_lead < 12) {
            float v607_data = r0[13];
            float v608_data = s0[13];
            float v610_data = ir1[0];
            ir1[0] = (v610_data + (v607_data * v608_data));
            float v613_data = s0[29];
            float v615_data = ir1[1];
            ir1[1] = (v615_data + (v607_data * v613_data));
            float v618_data = s0[45];
            float v620_data = ir1[2];
            ir1[2] = (v620_data + (v607_data * v618_data));
            float v623_data = s0[61];
            float v625_data = ir1[3];
            ir1[3] = (v625_data + (v607_data * v623_data));
            float v628_data = s0[77];
            float v630_data = ir1[4];
            ir1[4] = (v630_data + (v607_data * v628_data));
            float v633_data = s0[93];
            float v635_data = ir1[5];
            ir1[5] = (v635_data + (v607_data * v633_data));
            float v638_data = s0[109];
            float v640_data = ir1[6];
            ir1[6] = (v640_data + (v607_data * v638_data));
            float v643_data = s0[125];
            float v645_data = ir1[7];
            ir1[7] = (v645_data + (v607_data * v643_data));
          }
          if (v6_lead < 12) {
            float v651_data = r0[14];
            float v652_data = s0[14];
            float v654_data = ir1[0];
            ir1[0] = (v654_data + (v651_data * v652_data));
            float v657_data = s0[30];
            float v659_data = ir1[1];
            ir1[1] = (v659_data + (v651_data * v657_data));
            float v662_data = s0[46];
            float v664_data = ir1[2];
            ir1[2] = (v664_data + (v651_data * v662_data));
            float v667_data = s0[62];
            float v669_data = ir1[3];
            ir1[3] = (v669_data + (v651_data * v667_data));
            float v672_data = s0[78];
            float v674_data = ir1[4];
            ir1[4] = (v674_data + (v651_data * v672_data));
            float v677_data = s0[94];
            float v679_data = ir1[5];
            ir1[5] = (v679_data + (v651_data * v677_data));
            float v682_data = s0[110];
            float v684_data = ir1[6];
            ir1[6] = (v684_data + (v651_data * v682_data));
            float v687_data = s0[126];
            float v689_data = ir1[7];
            ir1[7] = (v689_data + (v651_data * v687_data));
          }
          if (v6_lead < 12) {
            float v695_data = r0[15];
            float v696_data = s0[15];
            float v698_data = ir1[0];
            ir1[0] = (v698_data + (v695_data * v696_data));
            float v701_data = s0[31];
            float v703_data = ir1[1];
            ir1[1] = (v703_data + (v695_data * v701_data));
            float v706_data = s0[47];
            float v708_data = ir1[2];
            ir1[2] = (v708_data + (v695_data * v706_data));
            float v711_data = s0[63];
            float v713_data = ir1[3];
            ir1[3] = (v713_data + (v695_data * v711_data));
            float v716_data = s0[79];
            float v718_data = ir1[4];
            ir1[4] = (v718_data + (v695_data * v716_data));
            float v721_data = s0[95];
            float v723_data = ir1[5];
            ir1[5] = (v723_data + (v695_data * v721_data));
            float v726_data = s0[111];
            float v728_data = ir1[6];
            ir1[6] = (v728_data + (v695_data * v726_data));
            float v731_data = s0[127];
            float v733_data = ir1[7];
            ir1[7] = (v733_data + (v695_data * v731_data));
          }
          if (v6_lead < 12) {
            #pragma unroll
            for (int32_t v739_n1 = 0; v739_n1 < 8; ++v739_n1) {
              int32_t v740_a = 0 + v739_n1;
              float v742_data = ir1[v739_n1];
              r1[v739_n1] = v742_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v6_lead < 12) {
            #pragma unroll
            for (int32_t v748_i1 = 0; v748_i1 < 8; ++v748_i1) {
              int32_t v749_a = 0 + v748_i1;
              float v751_data = r1[v748_i1];
              glb_m0[(v6_lead + (v748_i1 * 12))] = v751_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

