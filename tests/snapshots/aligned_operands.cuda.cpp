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
            int32_t v17_lead = v10_lead + (v11_i0 * 16);
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 16; ++v12_i1) {
              float v20_data = __ldcg(&glb_m1[(v17_lead + (v12_i1 * 16))]);
              r0[(v11_i0 + v12_i1)] = v20_data;
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
          float v30_data = r0[0];
          float v31_data = s0[0];
          float v33_data = ir1[0];
          ir1[0] = (v33_data + (v30_data * v31_data));
          float v36_data = s0[16];
          float v38_data = ir1[1];
          ir1[1] = (v38_data + (v30_data * v36_data));
          float v41_data = s0[33];
          float v43_data = ir1[2];
          ir1[2] = (v43_data + (v30_data * v41_data));
          float v46_data = s0[49];
          float v48_data = ir1[3];
          ir1[3] = (v48_data + (v30_data * v46_data));
          float v51_data = s0[66];
          float v53_data = ir1[4];
          ir1[4] = (v53_data + (v30_data * v51_data));
          float v56_data = s0[82];
          float v58_data = ir1[5];
          ir1[5] = (v58_data + (v30_data * v56_data));
          float v61_data = s0[99];
          float v63_data = ir1[6];
          ir1[6] = (v63_data + (v30_data * v61_data));
          float v66_data = s0[115];
          float v68_data = ir1[7];
          ir1[7] = (v68_data + (v30_data * v66_data));
          float v73_data = r0[1];
          float v74_data = s0[1];
          float v76_data = ir1[0];
          ir1[0] = (v76_data + (v73_data * v74_data));
          float v79_data = s0[17];
          float v81_data = ir1[1];
          ir1[1] = (v81_data + (v73_data * v79_data));
          float v84_data = s0[32];
          float v86_data = ir1[2];
          ir1[2] = (v86_data + (v73_data * v84_data));
          float v89_data = s0[48];
          float v91_data = ir1[3];
          ir1[3] = (v91_data + (v73_data * v89_data));
          float v94_data = s0[67];
          float v96_data = ir1[4];
          ir1[4] = (v96_data + (v73_data * v94_data));
          float v99_data = s0[83];
          float v101_data = ir1[5];
          ir1[5] = (v101_data + (v73_data * v99_data));
          float v104_data = s0[98];
          float v106_data = ir1[6];
          ir1[6] = (v106_data + (v73_data * v104_data));
          float v109_data = s0[114];
          float v111_data = ir1[7];
          ir1[7] = (v111_data + (v73_data * v109_data));
          float v116_data = r0[2];
          float v117_data = s0[2];
          float v119_data = ir1[0];
          ir1[0] = (v119_data + (v116_data * v117_data));
          float v122_data = s0[18];
          float v124_data = ir1[1];
          ir1[1] = (v124_data + (v116_data * v122_data));
          float v127_data = s0[35];
          float v129_data = ir1[2];
          ir1[2] = (v129_data + (v116_data * v127_data));
          float v132_data = s0[51];
          float v134_data = ir1[3];
          ir1[3] = (v134_data + (v116_data * v132_data));
          float v137_data = s0[64];
          float v139_data = ir1[4];
          ir1[4] = (v139_data + (v116_data * v137_data));
          float v142_data = s0[80];
          float v144_data = ir1[5];
          ir1[5] = (v144_data + (v116_data * v142_data));
          float v147_data = s0[97];
          float v149_data = ir1[6];
          ir1[6] = (v149_data + (v116_data * v147_data));
          float v152_data = s0[113];
          float v154_data = ir1[7];
          ir1[7] = (v154_data + (v116_data * v152_data));
          float v159_data = r0[3];
          float v160_data = s0[3];
          float v162_data = ir1[0];
          ir1[0] = (v162_data + (v159_data * v160_data));
          float v165_data = s0[19];
          float v167_data = ir1[1];
          ir1[1] = (v167_data + (v159_data * v165_data));
          float v170_data = s0[34];
          float v172_data = ir1[2];
          ir1[2] = (v172_data + (v159_data * v170_data));
          float v175_data = s0[50];
          float v177_data = ir1[3];
          ir1[3] = (v177_data + (v159_data * v175_data));
          float v180_data = s0[65];
          float v182_data = ir1[4];
          ir1[4] = (v182_data + (v159_data * v180_data));
          float v185_data = s0[81];
          float v187_data = ir1[5];
          ir1[5] = (v187_data + (v159_data * v185_data));
          float v190_data = s0[96];
          float v192_data = ir1[6];
          ir1[6] = (v192_data + (v159_data * v190_data));
          float v195_data = s0[112];
          float v197_data = ir1[7];
          ir1[7] = (v197_data + (v159_data * v195_data));
          float v202_data = r0[4];
          float v203_data = s0[4];
          float v205_data = ir1[0];
          ir1[0] = (v205_data + (v202_data * v203_data));
          float v208_data = s0[20];
          float v210_data = ir1[1];
          ir1[1] = (v210_data + (v202_data * v208_data));
          float v213_data = s0[37];
          float v215_data = ir1[2];
          ir1[2] = (v215_data + (v202_data * v213_data));
          float v218_data = s0[53];
          float v220_data = ir1[3];
          ir1[3] = (v220_data + (v202_data * v218_data));
          float v223_data = s0[70];
          float v225_data = ir1[4];
          ir1[4] = (v225_data + (v202_data * v223_data));
          float v228_data = s0[86];
          float v230_data = ir1[5];
          ir1[5] = (v230_data + (v202_data * v228_data));
          float v233_data = s0[103];
          float v235_data = ir1[6];
          ir1[6] = (v235_data + (v202_data * v233_data));
          float v238_data = s0[119];
          float v240_data = ir1[7];
          ir1[7] = (v240_data + (v202_data * v238_data));
          float v245_data = r0[5];
          float v246_data = s0[5];
          float v248_data = ir1[0];
          ir1[0] = (v248_data + (v245_data * v246_data));
          float v251_data = s0[21];
          float v253_data = ir1[1];
          ir1[1] = (v253_data + (v245_data * v251_data));
          float v256_data = s0[36];
          float v258_data = ir1[2];
          ir1[2] = (v258_data + (v245_data * v256_data));
          float v261_data = s0[52];
          float v263_data = ir1[3];
          ir1[3] = (v263_data + (v245_data * v261_data));
          float v266_data = s0[71];
          float v268_data = ir1[4];
          ir1[4] = (v268_data + (v245_data * v266_data));
          float v271_data = s0[87];
          float v273_data = ir1[5];
          ir1[5] = (v273_data + (v245_data * v271_data));
          float v276_data = s0[102];
          float v278_data = ir1[6];
          ir1[6] = (v278_data + (v245_data * v276_data));
          float v281_data = s0[118];
          float v283_data = ir1[7];
          ir1[7] = (v283_data + (v245_data * v281_data));
          float v288_data = r0[6];
          float v289_data = s0[6];
          float v291_data = ir1[0];
          ir1[0] = (v291_data + (v288_data * v289_data));
          float v294_data = s0[22];
          float v296_data = ir1[1];
          ir1[1] = (v296_data + (v288_data * v294_data));
          float v299_data = s0[39];
          float v301_data = ir1[2];
          ir1[2] = (v301_data + (v288_data * v299_data));
          float v304_data = s0[55];
          float v306_data = ir1[3];
          ir1[3] = (v306_data + (v288_data * v304_data));
          float v309_data = s0[68];
          float v311_data = ir1[4];
          ir1[4] = (v311_data + (v288_data * v309_data));
          float v314_data = s0[84];
          float v316_data = ir1[5];
          ir1[5] = (v316_data + (v288_data * v314_data));
          float v319_data = s0[101];
          float v321_data = ir1[6];
          ir1[6] = (v321_data + (v288_data * v319_data));
          float v324_data = s0[117];
          float v326_data = ir1[7];
          ir1[7] = (v326_data + (v288_data * v324_data));
          float v331_data = r0[7];
          float v332_data = s0[7];
          float v334_data = ir1[0];
          ir1[0] = (v334_data + (v331_data * v332_data));
          float v337_data = s0[23];
          float v339_data = ir1[1];
          ir1[1] = (v339_data + (v331_data * v337_data));
          float v342_data = s0[38];
          float v344_data = ir1[2];
          ir1[2] = (v344_data + (v331_data * v342_data));
          float v347_data = s0[54];
          float v349_data = ir1[3];
          ir1[3] = (v349_data + (v331_data * v347_data));
          float v352_data = s0[69];
          float v354_data = ir1[4];
          ir1[4] = (v354_data + (v331_data * v352_data));
          float v357_data = s0[85];
          float v359_data = ir1[5];
          ir1[5] = (v359_data + (v331_data * v357_data));
          float v362_data = s0[100];
          float v364_data = ir1[6];
          ir1[6] = (v364_data + (v331_data * v362_data));
          float v367_data = s0[116];
          float v369_data = ir1[7];
          ir1[7] = (v369_data + (v331_data * v367_data));
          float v374_data = r0[8];
          float v375_data = s0[8];
          float v377_data = ir1[0];
          ir1[0] = (v377_data + (v374_data * v375_data));
          float v380_data = s0[24];
          float v382_data = ir1[1];
          ir1[1] = (v382_data + (v374_data * v380_data));
          float v385_data = s0[41];
          float v387_data = ir1[2];
          ir1[2] = (v387_data + (v374_data * v385_data));
          float v390_data = s0[57];
          float v392_data = ir1[3];
          ir1[3] = (v392_data + (v374_data * v390_data));
          float v395_data = s0[74];
          float v397_data = ir1[4];
          ir1[4] = (v397_data + (v374_data * v395_data));
          float v400_data = s0[90];
          float v402_data = ir1[5];
          ir1[5] = (v402_data + (v374_data * v400_data));
          float v405_data = s0[107];
          float v407_data = ir1[6];
          ir1[6] = (v407_data + (v374_data * v405_data));
          float v410_data = s0[123];
          float v412_data = ir1[7];
          ir1[7] = (v412_data + (v374_data * v410_data));
          float v417_data = r0[9];
          float v418_data = s0[9];
          float v420_data = ir1[0];
          ir1[0] = (v420_data + (v417_data * v418_data));
          float v423_data = s0[25];
          float v425_data = ir1[1];
          ir1[1] = (v425_data + (v417_data * v423_data));
          float v428_data = s0[40];
          float v430_data = ir1[2];
          ir1[2] = (v430_data + (v417_data * v428_data));
          float v433_data = s0[56];
          float v435_data = ir1[3];
          ir1[3] = (v435_data + (v417_data * v433_data));
          float v438_data = s0[75];
          float v440_data = ir1[4];
          ir1[4] = (v440_data + (v417_data * v438_data));
          float v443_data = s0[91];
          float v445_data = ir1[5];
          ir1[5] = (v445_data + (v417_data * v443_data));
          float v448_data = s0[106];
          float v450_data = ir1[6];
          ir1[6] = (v450_data + (v417_data * v448_data));
          float v453_data = s0[122];
          float v455_data = ir1[7];
          ir1[7] = (v455_data + (v417_data * v453_data));
          float v460_data = r0[10];
          float v461_data = s0[10];
          float v463_data = ir1[0];
          ir1[0] = (v463_data + (v460_data * v461_data));
          float v466_data = s0[26];
          float v468_data = ir1[1];
          ir1[1] = (v468_data + (v460_data * v466_data));
          float v471_data = s0[43];
          float v473_data = ir1[2];
          ir1[2] = (v473_data + (v460_data * v471_data));
          float v476_data = s0[59];
          float v478_data = ir1[3];
          ir1[3] = (v478_data + (v460_data * v476_data));
          float v481_data = s0[72];
          float v483_data = ir1[4];
          ir1[4] = (v483_data + (v460_data * v481_data));
          float v486_data = s0[88];
          float v488_data = ir1[5];
          ir1[5] = (v488_data + (v460_data * v486_data));
          float v491_data = s0[105];
          float v493_data = ir1[6];
          ir1[6] = (v493_data + (v460_data * v491_data));
          float v496_data = s0[121];
          float v498_data = ir1[7];
          ir1[7] = (v498_data + (v460_data * v496_data));
          float v503_data = r0[11];
          float v504_data = s0[11];
          float v506_data = ir1[0];
          ir1[0] = (v506_data + (v503_data * v504_data));
          float v509_data = s0[27];
          float v511_data = ir1[1];
          ir1[1] = (v511_data + (v503_data * v509_data));
          float v514_data = s0[42];
          float v516_data = ir1[2];
          ir1[2] = (v516_data + (v503_data * v514_data));
          float v519_data = s0[58];
          float v521_data = ir1[3];
          ir1[3] = (v521_data + (v503_data * v519_data));
          float v524_data = s0[73];
          float v526_data = ir1[4];
          ir1[4] = (v526_data + (v503_data * v524_data));
          float v529_data = s0[89];
          float v531_data = ir1[5];
          ir1[5] = (v531_data + (v503_data * v529_data));
          float v534_data = s0[104];
          float v536_data = ir1[6];
          ir1[6] = (v536_data + (v503_data * v534_data));
          float v539_data = s0[120];
          float v541_data = ir1[7];
          ir1[7] = (v541_data + (v503_data * v539_data));
          float v546_data = r0[12];
          float v547_data = s0[12];
          float v549_data = ir1[0];
          ir1[0] = (v549_data + (v546_data * v547_data));
          float v552_data = s0[28];
          float v554_data = ir1[1];
          ir1[1] = (v554_data + (v546_data * v552_data));
          float v557_data = s0[45];
          float v559_data = ir1[2];
          ir1[2] = (v559_data + (v546_data * v557_data));
          float v562_data = s0[61];
          float v564_data = ir1[3];
          ir1[3] = (v564_data + (v546_data * v562_data));
          float v567_data = s0[78];
          float v569_data = ir1[4];
          ir1[4] = (v569_data + (v546_data * v567_data));
          float v572_data = s0[94];
          float v574_data = ir1[5];
          ir1[5] = (v574_data + (v546_data * v572_data));
          float v577_data = s0[111];
          float v579_data = ir1[6];
          ir1[6] = (v579_data + (v546_data * v577_data));
          float v582_data = s0[127];
          float v584_data = ir1[7];
          ir1[7] = (v584_data + (v546_data * v582_data));
          float v589_data = r0[13];
          float v590_data = s0[13];
          float v592_data = ir1[0];
          ir1[0] = (v592_data + (v589_data * v590_data));
          float v595_data = s0[29];
          float v597_data = ir1[1];
          ir1[1] = (v597_data + (v589_data * v595_data));
          float v600_data = s0[44];
          float v602_data = ir1[2];
          ir1[2] = (v602_data + (v589_data * v600_data));
          float v605_data = s0[60];
          float v607_data = ir1[3];
          ir1[3] = (v607_data + (v589_data * v605_data));
          float v610_data = s0[79];
          float v612_data = ir1[4];
          ir1[4] = (v612_data + (v589_data * v610_data));
          float v615_data = s0[95];
          float v617_data = ir1[5];
          ir1[5] = (v617_data + (v589_data * v615_data));
          float v620_data = s0[110];
          float v622_data = ir1[6];
          ir1[6] = (v622_data + (v589_data * v620_data));
          float v625_data = s0[126];
          float v627_data = ir1[7];
          ir1[7] = (v627_data + (v589_data * v625_data));
          float v632_data = r0[14];
          float v633_data = s0[14];
          float v635_data = ir1[0];
          ir1[0] = (v635_data + (v632_data * v633_data));
          float v638_data = s0[30];
          float v640_data = ir1[1];
          ir1[1] = (v640_data + (v632_data * v638_data));
          float v643_data = s0[47];
          float v645_data = ir1[2];
          ir1[2] = (v645_data + (v632_data * v643_data));
          float v648_data = s0[63];
          float v650_data = ir1[3];
          ir1[3] = (v650_data + (v632_data * v648_data));
          float v653_data = s0[76];
          float v655_data = ir1[4];
          ir1[4] = (v655_data + (v632_data * v653_data));
          float v658_data = s0[92];
          float v660_data = ir1[5];
          ir1[5] = (v660_data + (v632_data * v658_data));
          float v663_data = s0[109];
          float v665_data = ir1[6];
          ir1[6] = (v665_data + (v632_data * v663_data));
          float v668_data = s0[125];
          float v670_data = ir1[7];
          ir1[7] = (v670_data + (v632_data * v668_data));
          float v675_data = r0[15];
          float v676_data = s0[15];
          float v678_data = ir1[0];
          ir1[0] = (v678_data + (v675_data * v676_data));
          float v681_data = s0[31];
          float v683_data = ir1[1];
          ir1[1] = (v683_data + (v675_data * v681_data));
          float v686_data = s0[46];
          float v688_data = ir1[2];
          ir1[2] = (v688_data + (v675_data * v686_data));
          float v691_data = s0[62];
          float v693_data = ir1[3];
          ir1[3] = (v693_data + (v675_data * v691_data));
          float v696_data = s0[77];
          float v698_data = ir1[4];
          ir1[4] = (v698_data + (v675_data * v696_data));
          float v701_data = s0[93];
          float v703_data = ir1[5];
          ir1[5] = (v703_data + (v675_data * v701_data));
          float v706_data = s0[108];
          float v708_data = ir1[6];
          ir1[6] = (v708_data + (v675_data * v706_data));
          float v711_data = s0[124];
          float v713_data = ir1[7];
          ir1[7] = (v713_data + (v675_data * v711_data));
          #pragma unroll
          for (int32_t v718_n0 = 0; v718_n0 < 1; ++v718_n0) {
            #pragma unroll
            for (int32_t v719_n1 = 0; v719_n1 < 8; ++v719_n1) {
              int32_t v720_a = v718_n0 + v719_n1;
              float v721_data = ir1[v720_a];
              r1[v720_a] = v721_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v726_i0 = 0; v726_i0 < 1; ++v726_i0) {
            int32_t v734_lead = v10_lead + (v726_i0 * 16);
            #pragma unroll
            for (int32_t v727_i1 = 0; v727_i1 < 8; ++v727_i1) {
              float v729_data = r1[(v726_i0 + v727_i1)];
              glb_m0[(v734_lead + (v727_i1 * 16))] = v729_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

