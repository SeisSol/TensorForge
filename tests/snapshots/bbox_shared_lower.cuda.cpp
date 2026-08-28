// === base name ===
kernel_4b59b6f027

// === header ===
void launcher_kernel_4b59b6f027(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_4b59b6f027(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_4b59b6f027, block.x * block.y * block.z, 2304 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_4b59b6f027, cudaFuncAttributeMaxDynamicSharedMemorySize, 2304 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_4b59b6f027<<<grid,block,2304 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_4b59b6f027(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×8(12×8) {4..16}×{0..8} strided
    // m1 16×16(12×16) {4..16}×{0..16} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 16×8(12×8) {4..16}×{0..8} strided({4..16}×{0..8})[0, 1] = m1 16×16(12×16) {4..16}×{0..16} strided({4..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 16;
          if (v10_lead < 12) {
            int32_t v19_a = (v10_lead + 4) - 4;
            int32_t v28_a = (v10_lead + 4) - 4;
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 16; ++v12_i1) {
              int32_t v20_a = v12_i1 * 12;
              int32_t v21_a = v19_a + v20_a;
              float v31_data = __ldcg(&glb_m1[(v28_a + v20_a)]);
              r0[v12_i1] = v31_data;
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
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(16, 28), (0, 8)] [(0, 16)]
          float ir1[8]{};
          if (v10_lead < 12) {
            float v41_data = r0[0];
            float v42_data = s0[0];
            float v44_data = ir1[0];
            ir1[0] = (v44_data + (v41_data * v42_data));
            float v47_data = s0[16];
            float v49_data = ir1[1];
            ir1[1] = (v49_data + (v41_data * v47_data));
            float v52_data = s0[32];
            float v54_data = ir1[2];
            ir1[2] = (v54_data + (v41_data * v52_data));
            float v57_data = s0[48];
            float v59_data = ir1[3];
            ir1[3] = (v59_data + (v41_data * v57_data));
            float v62_data = s0[64];
            float v64_data = ir1[4];
            ir1[4] = (v64_data + (v41_data * v62_data));
            float v67_data = s0[80];
            float v69_data = ir1[5];
            ir1[5] = (v69_data + (v41_data * v67_data));
            float v72_data = s0[96];
            float v74_data = ir1[6];
            ir1[6] = (v74_data + (v41_data * v72_data));
            float v77_data = s0[112];
            float v79_data = ir1[7];
            ir1[7] = (v79_data + (v41_data * v77_data));
          }
          if (v10_lead < 12) {
            float v85_data = r0[1];
            float v86_data = s0[1];
            float v88_data = ir1[0];
            ir1[0] = (v88_data + (v85_data * v86_data));
            float v91_data = s0[17];
            float v93_data = ir1[1];
            ir1[1] = (v93_data + (v85_data * v91_data));
            float v96_data = s0[33];
            float v98_data = ir1[2];
            ir1[2] = (v98_data + (v85_data * v96_data));
            float v101_data = s0[49];
            float v103_data = ir1[3];
            ir1[3] = (v103_data + (v85_data * v101_data));
            float v106_data = s0[65];
            float v108_data = ir1[4];
            ir1[4] = (v108_data + (v85_data * v106_data));
            float v111_data = s0[81];
            float v113_data = ir1[5];
            ir1[5] = (v113_data + (v85_data * v111_data));
            float v116_data = s0[97];
            float v118_data = ir1[6];
            ir1[6] = (v118_data + (v85_data * v116_data));
            float v121_data = s0[113];
            float v123_data = ir1[7];
            ir1[7] = (v123_data + (v85_data * v121_data));
          }
          if (v10_lead < 12) {
            float v129_data = r0[2];
            float v130_data = s0[2];
            float v132_data = ir1[0];
            ir1[0] = (v132_data + (v129_data * v130_data));
            float v135_data = s0[18];
            float v137_data = ir1[1];
            ir1[1] = (v137_data + (v129_data * v135_data));
            float v140_data = s0[34];
            float v142_data = ir1[2];
            ir1[2] = (v142_data + (v129_data * v140_data));
            float v145_data = s0[50];
            float v147_data = ir1[3];
            ir1[3] = (v147_data + (v129_data * v145_data));
            float v150_data = s0[66];
            float v152_data = ir1[4];
            ir1[4] = (v152_data + (v129_data * v150_data));
            float v155_data = s0[82];
            float v157_data = ir1[5];
            ir1[5] = (v157_data + (v129_data * v155_data));
            float v160_data = s0[98];
            float v162_data = ir1[6];
            ir1[6] = (v162_data + (v129_data * v160_data));
            float v165_data = s0[114];
            float v167_data = ir1[7];
            ir1[7] = (v167_data + (v129_data * v165_data));
          }
          if (v10_lead < 12) {
            float v173_data = r0[3];
            float v174_data = s0[3];
            float v176_data = ir1[0];
            ir1[0] = (v176_data + (v173_data * v174_data));
            float v179_data = s0[19];
            float v181_data = ir1[1];
            ir1[1] = (v181_data + (v173_data * v179_data));
            float v184_data = s0[35];
            float v186_data = ir1[2];
            ir1[2] = (v186_data + (v173_data * v184_data));
            float v189_data = s0[51];
            float v191_data = ir1[3];
            ir1[3] = (v191_data + (v173_data * v189_data));
            float v194_data = s0[67];
            float v196_data = ir1[4];
            ir1[4] = (v196_data + (v173_data * v194_data));
            float v199_data = s0[83];
            float v201_data = ir1[5];
            ir1[5] = (v201_data + (v173_data * v199_data));
            float v204_data = s0[99];
            float v206_data = ir1[6];
            ir1[6] = (v206_data + (v173_data * v204_data));
            float v209_data = s0[115];
            float v211_data = ir1[7];
            ir1[7] = (v211_data + (v173_data * v209_data));
          }
          if (v10_lead < 12) {
            float v217_data = r0[4];
            float v218_data = s0[4];
            float v220_data = ir1[0];
            ir1[0] = (v220_data + (v217_data * v218_data));
            float v223_data = s0[20];
            float v225_data = ir1[1];
            ir1[1] = (v225_data + (v217_data * v223_data));
            float v228_data = s0[36];
            float v230_data = ir1[2];
            ir1[2] = (v230_data + (v217_data * v228_data));
            float v233_data = s0[52];
            float v235_data = ir1[3];
            ir1[3] = (v235_data + (v217_data * v233_data));
            float v238_data = s0[68];
            float v240_data = ir1[4];
            ir1[4] = (v240_data + (v217_data * v238_data));
            float v243_data = s0[84];
            float v245_data = ir1[5];
            ir1[5] = (v245_data + (v217_data * v243_data));
            float v248_data = s0[100];
            float v250_data = ir1[6];
            ir1[6] = (v250_data + (v217_data * v248_data));
            float v253_data = s0[116];
            float v255_data = ir1[7];
            ir1[7] = (v255_data + (v217_data * v253_data));
          }
          if (v10_lead < 12) {
            float v261_data = r0[5];
            float v262_data = s0[5];
            float v264_data = ir1[0];
            ir1[0] = (v264_data + (v261_data * v262_data));
            float v267_data = s0[21];
            float v269_data = ir1[1];
            ir1[1] = (v269_data + (v261_data * v267_data));
            float v272_data = s0[37];
            float v274_data = ir1[2];
            ir1[2] = (v274_data + (v261_data * v272_data));
            float v277_data = s0[53];
            float v279_data = ir1[3];
            ir1[3] = (v279_data + (v261_data * v277_data));
            float v282_data = s0[69];
            float v284_data = ir1[4];
            ir1[4] = (v284_data + (v261_data * v282_data));
            float v287_data = s0[85];
            float v289_data = ir1[5];
            ir1[5] = (v289_data + (v261_data * v287_data));
            float v292_data = s0[101];
            float v294_data = ir1[6];
            ir1[6] = (v294_data + (v261_data * v292_data));
            float v297_data = s0[117];
            float v299_data = ir1[7];
            ir1[7] = (v299_data + (v261_data * v297_data));
          }
          if (v10_lead < 12) {
            float v305_data = r0[6];
            float v306_data = s0[6];
            float v308_data = ir1[0];
            ir1[0] = (v308_data + (v305_data * v306_data));
            float v311_data = s0[22];
            float v313_data = ir1[1];
            ir1[1] = (v313_data + (v305_data * v311_data));
            float v316_data = s0[38];
            float v318_data = ir1[2];
            ir1[2] = (v318_data + (v305_data * v316_data));
            float v321_data = s0[54];
            float v323_data = ir1[3];
            ir1[3] = (v323_data + (v305_data * v321_data));
            float v326_data = s0[70];
            float v328_data = ir1[4];
            ir1[4] = (v328_data + (v305_data * v326_data));
            float v331_data = s0[86];
            float v333_data = ir1[5];
            ir1[5] = (v333_data + (v305_data * v331_data));
            float v336_data = s0[102];
            float v338_data = ir1[6];
            ir1[6] = (v338_data + (v305_data * v336_data));
            float v341_data = s0[118];
            float v343_data = ir1[7];
            ir1[7] = (v343_data + (v305_data * v341_data));
          }
          if (v10_lead < 12) {
            float v349_data = r0[7];
            float v350_data = s0[7];
            float v352_data = ir1[0];
            ir1[0] = (v352_data + (v349_data * v350_data));
            float v355_data = s0[23];
            float v357_data = ir1[1];
            ir1[1] = (v357_data + (v349_data * v355_data));
            float v360_data = s0[39];
            float v362_data = ir1[2];
            ir1[2] = (v362_data + (v349_data * v360_data));
            float v365_data = s0[55];
            float v367_data = ir1[3];
            ir1[3] = (v367_data + (v349_data * v365_data));
            float v370_data = s0[71];
            float v372_data = ir1[4];
            ir1[4] = (v372_data + (v349_data * v370_data));
            float v375_data = s0[87];
            float v377_data = ir1[5];
            ir1[5] = (v377_data + (v349_data * v375_data));
            float v380_data = s0[103];
            float v382_data = ir1[6];
            ir1[6] = (v382_data + (v349_data * v380_data));
            float v385_data = s0[119];
            float v387_data = ir1[7];
            ir1[7] = (v387_data + (v349_data * v385_data));
          }
          if (v10_lead < 12) {
            float v393_data = r0[8];
            float v394_data = s0[8];
            float v396_data = ir1[0];
            ir1[0] = (v396_data + (v393_data * v394_data));
            float v399_data = s0[24];
            float v401_data = ir1[1];
            ir1[1] = (v401_data + (v393_data * v399_data));
            float v404_data = s0[40];
            float v406_data = ir1[2];
            ir1[2] = (v406_data + (v393_data * v404_data));
            float v409_data = s0[56];
            float v411_data = ir1[3];
            ir1[3] = (v411_data + (v393_data * v409_data));
            float v414_data = s0[72];
            float v416_data = ir1[4];
            ir1[4] = (v416_data + (v393_data * v414_data));
            float v419_data = s0[88];
            float v421_data = ir1[5];
            ir1[5] = (v421_data + (v393_data * v419_data));
            float v424_data = s0[104];
            float v426_data = ir1[6];
            ir1[6] = (v426_data + (v393_data * v424_data));
            float v429_data = s0[120];
            float v431_data = ir1[7];
            ir1[7] = (v431_data + (v393_data * v429_data));
          }
          if (v10_lead < 12) {
            float v437_data = r0[9];
            float v438_data = s0[9];
            float v440_data = ir1[0];
            ir1[0] = (v440_data + (v437_data * v438_data));
            float v443_data = s0[25];
            float v445_data = ir1[1];
            ir1[1] = (v445_data + (v437_data * v443_data));
            float v448_data = s0[41];
            float v450_data = ir1[2];
            ir1[2] = (v450_data + (v437_data * v448_data));
            float v453_data = s0[57];
            float v455_data = ir1[3];
            ir1[3] = (v455_data + (v437_data * v453_data));
            float v458_data = s0[73];
            float v460_data = ir1[4];
            ir1[4] = (v460_data + (v437_data * v458_data));
            float v463_data = s0[89];
            float v465_data = ir1[5];
            ir1[5] = (v465_data + (v437_data * v463_data));
            float v468_data = s0[105];
            float v470_data = ir1[6];
            ir1[6] = (v470_data + (v437_data * v468_data));
            float v473_data = s0[121];
            float v475_data = ir1[7];
            ir1[7] = (v475_data + (v437_data * v473_data));
          }
          if (v10_lead < 12) {
            float v481_data = r0[10];
            float v482_data = s0[10];
            float v484_data = ir1[0];
            ir1[0] = (v484_data + (v481_data * v482_data));
            float v487_data = s0[26];
            float v489_data = ir1[1];
            ir1[1] = (v489_data + (v481_data * v487_data));
            float v492_data = s0[42];
            float v494_data = ir1[2];
            ir1[2] = (v494_data + (v481_data * v492_data));
            float v497_data = s0[58];
            float v499_data = ir1[3];
            ir1[3] = (v499_data + (v481_data * v497_data));
            float v502_data = s0[74];
            float v504_data = ir1[4];
            ir1[4] = (v504_data + (v481_data * v502_data));
            float v507_data = s0[90];
            float v509_data = ir1[5];
            ir1[5] = (v509_data + (v481_data * v507_data));
            float v512_data = s0[106];
            float v514_data = ir1[6];
            ir1[6] = (v514_data + (v481_data * v512_data));
            float v517_data = s0[122];
            float v519_data = ir1[7];
            ir1[7] = (v519_data + (v481_data * v517_data));
          }
          if (v10_lead < 12) {
            float v525_data = r0[11];
            float v526_data = s0[11];
            float v528_data = ir1[0];
            ir1[0] = (v528_data + (v525_data * v526_data));
            float v531_data = s0[27];
            float v533_data = ir1[1];
            ir1[1] = (v533_data + (v525_data * v531_data));
            float v536_data = s0[43];
            float v538_data = ir1[2];
            ir1[2] = (v538_data + (v525_data * v536_data));
            float v541_data = s0[59];
            float v543_data = ir1[3];
            ir1[3] = (v543_data + (v525_data * v541_data));
            float v546_data = s0[75];
            float v548_data = ir1[4];
            ir1[4] = (v548_data + (v525_data * v546_data));
            float v551_data = s0[91];
            float v553_data = ir1[5];
            ir1[5] = (v553_data + (v525_data * v551_data));
            float v556_data = s0[107];
            float v558_data = ir1[6];
            ir1[6] = (v558_data + (v525_data * v556_data));
            float v561_data = s0[123];
            float v563_data = ir1[7];
            ir1[7] = (v563_data + (v525_data * v561_data));
          }
          if (v10_lead < 12) {
            float v569_data = r0[12];
            float v570_data = s0[12];
            float v572_data = ir1[0];
            ir1[0] = (v572_data + (v569_data * v570_data));
            float v575_data = s0[28];
            float v577_data = ir1[1];
            ir1[1] = (v577_data + (v569_data * v575_data));
            float v580_data = s0[44];
            float v582_data = ir1[2];
            ir1[2] = (v582_data + (v569_data * v580_data));
            float v585_data = s0[60];
            float v587_data = ir1[3];
            ir1[3] = (v587_data + (v569_data * v585_data));
            float v590_data = s0[76];
            float v592_data = ir1[4];
            ir1[4] = (v592_data + (v569_data * v590_data));
            float v595_data = s0[92];
            float v597_data = ir1[5];
            ir1[5] = (v597_data + (v569_data * v595_data));
            float v600_data = s0[108];
            float v602_data = ir1[6];
            ir1[6] = (v602_data + (v569_data * v600_data));
            float v605_data = s0[124];
            float v607_data = ir1[7];
            ir1[7] = (v607_data + (v569_data * v605_data));
          }
          if (v10_lead < 12) {
            float v613_data = r0[13];
            float v614_data = s0[13];
            float v616_data = ir1[0];
            ir1[0] = (v616_data + (v613_data * v614_data));
            float v619_data = s0[29];
            float v621_data = ir1[1];
            ir1[1] = (v621_data + (v613_data * v619_data));
            float v624_data = s0[45];
            float v626_data = ir1[2];
            ir1[2] = (v626_data + (v613_data * v624_data));
            float v629_data = s0[61];
            float v631_data = ir1[3];
            ir1[3] = (v631_data + (v613_data * v629_data));
            float v634_data = s0[77];
            float v636_data = ir1[4];
            ir1[4] = (v636_data + (v613_data * v634_data));
            float v639_data = s0[93];
            float v641_data = ir1[5];
            ir1[5] = (v641_data + (v613_data * v639_data));
            float v644_data = s0[109];
            float v646_data = ir1[6];
            ir1[6] = (v646_data + (v613_data * v644_data));
            float v649_data = s0[125];
            float v651_data = ir1[7];
            ir1[7] = (v651_data + (v613_data * v649_data));
          }
          if (v10_lead < 12) {
            float v657_data = r0[14];
            float v658_data = s0[14];
            float v660_data = ir1[0];
            ir1[0] = (v660_data + (v657_data * v658_data));
            float v663_data = s0[30];
            float v665_data = ir1[1];
            ir1[1] = (v665_data + (v657_data * v663_data));
            float v668_data = s0[46];
            float v670_data = ir1[2];
            ir1[2] = (v670_data + (v657_data * v668_data));
            float v673_data = s0[62];
            float v675_data = ir1[3];
            ir1[3] = (v675_data + (v657_data * v673_data));
            float v678_data = s0[78];
            float v680_data = ir1[4];
            ir1[4] = (v680_data + (v657_data * v678_data));
            float v683_data = s0[94];
            float v685_data = ir1[5];
            ir1[5] = (v685_data + (v657_data * v683_data));
            float v688_data = s0[110];
            float v690_data = ir1[6];
            ir1[6] = (v690_data + (v657_data * v688_data));
            float v693_data = s0[126];
            float v695_data = ir1[7];
            ir1[7] = (v695_data + (v657_data * v693_data));
          }
          if (v10_lead < 12) {
            float v701_data = r0[15];
            float v702_data = s0[15];
            float v704_data = ir1[0];
            ir1[0] = (v704_data + (v701_data * v702_data));
            float v707_data = s0[31];
            float v709_data = ir1[1];
            ir1[1] = (v709_data + (v701_data * v707_data));
            float v712_data = s0[47];
            float v714_data = ir1[2];
            ir1[2] = (v714_data + (v701_data * v712_data));
            float v717_data = s0[63];
            float v719_data = ir1[3];
            ir1[3] = (v719_data + (v701_data * v717_data));
            float v722_data = s0[79];
            float v724_data = ir1[4];
            ir1[4] = (v724_data + (v701_data * v722_data));
            float v727_data = s0[95];
            float v729_data = ir1[5];
            ir1[5] = (v729_data + (v701_data * v727_data));
            float v732_data = s0[111];
            float v734_data = ir1[6];
            ir1[6] = (v734_data + (v701_data * v732_data));
            float v737_data = s0[127];
            float v739_data = ir1[7];
            ir1[7] = (v739_data + (v701_data * v737_data));
          }
          if (v10_lead < 12) {
            #pragma unroll
            for (int32_t v745_n1 = 0; v745_n1 < 8; ++v745_n1) {
              int32_t v746_a = 0 + v745_n1;
              float v748_data = ir1[v745_n1];
              r1[v745_n1] = v748_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v10_lead < 12) {
            int32_t v764_a = ((v10_lead + 16_i32) + -12) - 4;
            #pragma unroll
            for (int32_t v754_i1 = 0; v754_i1 < 8; ++v754_i1) {
              int32_t v755_a = 0 + v754_i1;
              float v757_data = r1[v754_i1];
              glb_m0[(v764_a + (v754_i1 * 12))] = v757_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

