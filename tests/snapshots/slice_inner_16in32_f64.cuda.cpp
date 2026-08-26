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
          int32_t v2_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v8_lead = v3_i0 * 16;
            int32_t v10_off = (v2_lead + v8_lead) + 8;
            int32_t v18_off = (v2_lead + v8_lead) + 8;
            #pragma unroll
            for (int32_t v4_i1 = 8; v4_i1 < 24; ++v4_i1) {
              int32_t v11_a = v4_i1 * 32;
              int32_t v12_a = v10_off + v11_a;
              double v21_data = __ldcg(&glb_m1[(v18_off + v11_a)]);
              int32_t v23_a = v3_i0 + (v4_i1 - 8);
              r0[v23_a] = v21_data;
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
            double v27_data = r0[0];
            double v28_data = s0[0];
            double v30_data = ir1[0];
            ir1[0] = (v30_data + (v27_data * v28_data));
            double v33_data = s0[16];
            double v35_data = ir1[1];
            ir1[1] = (v35_data + (v27_data * v33_data));
            double v38_data = s0[32];
            double v40_data = ir1[2];
            ir1[2] = (v40_data + (v27_data * v38_data));
            double v43_data = s0[48];
            double v45_data = ir1[3];
            ir1[3] = (v45_data + (v27_data * v43_data));
            double v48_data = s0[64];
            double v50_data = ir1[4];
            ir1[4] = (v50_data + (v27_data * v48_data));
            double v53_data = s0[80];
            double v55_data = ir1[5];
            ir1[5] = (v55_data + (v27_data * v53_data));
            double v58_data = s0[96];
            double v60_data = ir1[6];
            ir1[6] = (v60_data + (v27_data * v58_data));
            double v63_data = s0[112];
            double v65_data = ir1[7];
            ir1[7] = (v65_data + (v27_data * v63_data));
            double v70_data = r0[1];
            double v71_data = s0[1];
            double v73_data = ir1[0];
            ir1[0] = (v73_data + (v70_data * v71_data));
            double v76_data = s0[17];
            double v78_data = ir1[1];
            ir1[1] = (v78_data + (v70_data * v76_data));
            double v81_data = s0[33];
            double v83_data = ir1[2];
            ir1[2] = (v83_data + (v70_data * v81_data));
            double v86_data = s0[49];
            double v88_data = ir1[3];
            ir1[3] = (v88_data + (v70_data * v86_data));
            double v91_data = s0[65];
            double v93_data = ir1[4];
            ir1[4] = (v93_data + (v70_data * v91_data));
            double v96_data = s0[81];
            double v98_data = ir1[5];
            ir1[5] = (v98_data + (v70_data * v96_data));
            double v101_data = s0[97];
            double v103_data = ir1[6];
            ir1[6] = (v103_data + (v70_data * v101_data));
            double v106_data = s0[113];
            double v108_data = ir1[7];
            ir1[7] = (v108_data + (v70_data * v106_data));
            double v113_data = r0[2];
            double v114_data = s0[2];
            double v116_data = ir1[0];
            ir1[0] = (v116_data + (v113_data * v114_data));
            double v119_data = s0[18];
            double v121_data = ir1[1];
            ir1[1] = (v121_data + (v113_data * v119_data));
            double v124_data = s0[34];
            double v126_data = ir1[2];
            ir1[2] = (v126_data + (v113_data * v124_data));
            double v129_data = s0[50];
            double v131_data = ir1[3];
            ir1[3] = (v131_data + (v113_data * v129_data));
            double v134_data = s0[66];
            double v136_data = ir1[4];
            ir1[4] = (v136_data + (v113_data * v134_data));
            double v139_data = s0[82];
            double v141_data = ir1[5];
            ir1[5] = (v141_data + (v113_data * v139_data));
            double v144_data = s0[98];
            double v146_data = ir1[6];
            ir1[6] = (v146_data + (v113_data * v144_data));
            double v149_data = s0[114];
            double v151_data = ir1[7];
            ir1[7] = (v151_data + (v113_data * v149_data));
            double v156_data = r0[3];
            double v157_data = s0[3];
            double v159_data = ir1[0];
            ir1[0] = (v159_data + (v156_data * v157_data));
            double v162_data = s0[19];
            double v164_data = ir1[1];
            ir1[1] = (v164_data + (v156_data * v162_data));
            double v167_data = s0[35];
            double v169_data = ir1[2];
            ir1[2] = (v169_data + (v156_data * v167_data));
            double v172_data = s0[51];
            double v174_data = ir1[3];
            ir1[3] = (v174_data + (v156_data * v172_data));
            double v177_data = s0[67];
            double v179_data = ir1[4];
            ir1[4] = (v179_data + (v156_data * v177_data));
            double v182_data = s0[83];
            double v184_data = ir1[5];
            ir1[5] = (v184_data + (v156_data * v182_data));
            double v187_data = s0[99];
            double v189_data = ir1[6];
            ir1[6] = (v189_data + (v156_data * v187_data));
            double v192_data = s0[115];
            double v194_data = ir1[7];
            ir1[7] = (v194_data + (v156_data * v192_data));
            double v199_data = r0[4];
            double v200_data = s0[4];
            double v202_data = ir1[0];
            ir1[0] = (v202_data + (v199_data * v200_data));
            double v205_data = s0[20];
            double v207_data = ir1[1];
            ir1[1] = (v207_data + (v199_data * v205_data));
            double v210_data = s0[36];
            double v212_data = ir1[2];
            ir1[2] = (v212_data + (v199_data * v210_data));
            double v215_data = s0[52];
            double v217_data = ir1[3];
            ir1[3] = (v217_data + (v199_data * v215_data));
            double v220_data = s0[68];
            double v222_data = ir1[4];
            ir1[4] = (v222_data + (v199_data * v220_data));
            double v225_data = s0[84];
            double v227_data = ir1[5];
            ir1[5] = (v227_data + (v199_data * v225_data));
            double v230_data = s0[100];
            double v232_data = ir1[6];
            ir1[6] = (v232_data + (v199_data * v230_data));
            double v235_data = s0[116];
            double v237_data = ir1[7];
            ir1[7] = (v237_data + (v199_data * v235_data));
            double v242_data = r0[5];
            double v243_data = s0[5];
            double v245_data = ir1[0];
            ir1[0] = (v245_data + (v242_data * v243_data));
            double v248_data = s0[21];
            double v250_data = ir1[1];
            ir1[1] = (v250_data + (v242_data * v248_data));
            double v253_data = s0[37];
            double v255_data = ir1[2];
            ir1[2] = (v255_data + (v242_data * v253_data));
            double v258_data = s0[53];
            double v260_data = ir1[3];
            ir1[3] = (v260_data + (v242_data * v258_data));
            double v263_data = s0[69];
            double v265_data = ir1[4];
            ir1[4] = (v265_data + (v242_data * v263_data));
            double v268_data = s0[85];
            double v270_data = ir1[5];
            ir1[5] = (v270_data + (v242_data * v268_data));
            double v273_data = s0[101];
            double v275_data = ir1[6];
            ir1[6] = (v275_data + (v242_data * v273_data));
            double v278_data = s0[117];
            double v280_data = ir1[7];
            ir1[7] = (v280_data + (v242_data * v278_data));
            double v285_data = r0[6];
            double v286_data = s0[6];
            double v288_data = ir1[0];
            ir1[0] = (v288_data + (v285_data * v286_data));
            double v291_data = s0[22];
            double v293_data = ir1[1];
            ir1[1] = (v293_data + (v285_data * v291_data));
            double v296_data = s0[38];
            double v298_data = ir1[2];
            ir1[2] = (v298_data + (v285_data * v296_data));
            double v301_data = s0[54];
            double v303_data = ir1[3];
            ir1[3] = (v303_data + (v285_data * v301_data));
            double v306_data = s0[70];
            double v308_data = ir1[4];
            ir1[4] = (v308_data + (v285_data * v306_data));
            double v311_data = s0[86];
            double v313_data = ir1[5];
            ir1[5] = (v313_data + (v285_data * v311_data));
            double v316_data = s0[102];
            double v318_data = ir1[6];
            ir1[6] = (v318_data + (v285_data * v316_data));
            double v321_data = s0[118];
            double v323_data = ir1[7];
            ir1[7] = (v323_data + (v285_data * v321_data));
            double v328_data = r0[7];
            double v329_data = s0[7];
            double v331_data = ir1[0];
            ir1[0] = (v331_data + (v328_data * v329_data));
            double v334_data = s0[23];
            double v336_data = ir1[1];
            ir1[1] = (v336_data + (v328_data * v334_data));
            double v339_data = s0[39];
            double v341_data = ir1[2];
            ir1[2] = (v341_data + (v328_data * v339_data));
            double v344_data = s0[55];
            double v346_data = ir1[3];
            ir1[3] = (v346_data + (v328_data * v344_data));
            double v349_data = s0[71];
            double v351_data = ir1[4];
            ir1[4] = (v351_data + (v328_data * v349_data));
            double v354_data = s0[87];
            double v356_data = ir1[5];
            ir1[5] = (v356_data + (v328_data * v354_data));
            double v359_data = s0[103];
            double v361_data = ir1[6];
            ir1[6] = (v361_data + (v328_data * v359_data));
            double v364_data = s0[119];
            double v366_data = ir1[7];
            ir1[7] = (v366_data + (v328_data * v364_data));
            double v371_data = r0[8];
            double v372_data = s0[8];
            double v374_data = ir1[0];
            ir1[0] = (v374_data + (v371_data * v372_data));
            double v377_data = s0[24];
            double v379_data = ir1[1];
            ir1[1] = (v379_data + (v371_data * v377_data));
            double v382_data = s0[40];
            double v384_data = ir1[2];
            ir1[2] = (v384_data + (v371_data * v382_data));
            double v387_data = s0[56];
            double v389_data = ir1[3];
            ir1[3] = (v389_data + (v371_data * v387_data));
            double v392_data = s0[72];
            double v394_data = ir1[4];
            ir1[4] = (v394_data + (v371_data * v392_data));
            double v397_data = s0[88];
            double v399_data = ir1[5];
            ir1[5] = (v399_data + (v371_data * v397_data));
            double v402_data = s0[104];
            double v404_data = ir1[6];
            ir1[6] = (v404_data + (v371_data * v402_data));
            double v407_data = s0[120];
            double v409_data = ir1[7];
            ir1[7] = (v409_data + (v371_data * v407_data));
            double v414_data = r0[9];
            double v415_data = s0[9];
            double v417_data = ir1[0];
            ir1[0] = (v417_data + (v414_data * v415_data));
            double v420_data = s0[25];
            double v422_data = ir1[1];
            ir1[1] = (v422_data + (v414_data * v420_data));
            double v425_data = s0[41];
            double v427_data = ir1[2];
            ir1[2] = (v427_data + (v414_data * v425_data));
            double v430_data = s0[57];
            double v432_data = ir1[3];
            ir1[3] = (v432_data + (v414_data * v430_data));
            double v435_data = s0[73];
            double v437_data = ir1[4];
            ir1[4] = (v437_data + (v414_data * v435_data));
            double v440_data = s0[89];
            double v442_data = ir1[5];
            ir1[5] = (v442_data + (v414_data * v440_data));
            double v445_data = s0[105];
            double v447_data = ir1[6];
            ir1[6] = (v447_data + (v414_data * v445_data));
            double v450_data = s0[121];
            double v452_data = ir1[7];
            ir1[7] = (v452_data + (v414_data * v450_data));
            double v457_data = r0[10];
            double v458_data = s0[10];
            double v460_data = ir1[0];
            ir1[0] = (v460_data + (v457_data * v458_data));
            double v463_data = s0[26];
            double v465_data = ir1[1];
            ir1[1] = (v465_data + (v457_data * v463_data));
            double v468_data = s0[42];
            double v470_data = ir1[2];
            ir1[2] = (v470_data + (v457_data * v468_data));
            double v473_data = s0[58];
            double v475_data = ir1[3];
            ir1[3] = (v475_data + (v457_data * v473_data));
            double v478_data = s0[74];
            double v480_data = ir1[4];
            ir1[4] = (v480_data + (v457_data * v478_data));
            double v483_data = s0[90];
            double v485_data = ir1[5];
            ir1[5] = (v485_data + (v457_data * v483_data));
            double v488_data = s0[106];
            double v490_data = ir1[6];
            ir1[6] = (v490_data + (v457_data * v488_data));
            double v493_data = s0[122];
            double v495_data = ir1[7];
            ir1[7] = (v495_data + (v457_data * v493_data));
            double v500_data = r0[11];
            double v501_data = s0[11];
            double v503_data = ir1[0];
            ir1[0] = (v503_data + (v500_data * v501_data));
            double v506_data = s0[27];
            double v508_data = ir1[1];
            ir1[1] = (v508_data + (v500_data * v506_data));
            double v511_data = s0[43];
            double v513_data = ir1[2];
            ir1[2] = (v513_data + (v500_data * v511_data));
            double v516_data = s0[59];
            double v518_data = ir1[3];
            ir1[3] = (v518_data + (v500_data * v516_data));
            double v521_data = s0[75];
            double v523_data = ir1[4];
            ir1[4] = (v523_data + (v500_data * v521_data));
            double v526_data = s0[91];
            double v528_data = ir1[5];
            ir1[5] = (v528_data + (v500_data * v526_data));
            double v531_data = s0[107];
            double v533_data = ir1[6];
            ir1[6] = (v533_data + (v500_data * v531_data));
            double v536_data = s0[123];
            double v538_data = ir1[7];
            ir1[7] = (v538_data + (v500_data * v536_data));
            double v543_data = r0[12];
            double v544_data = s0[12];
            double v546_data = ir1[0];
            ir1[0] = (v546_data + (v543_data * v544_data));
            double v549_data = s0[28];
            double v551_data = ir1[1];
            ir1[1] = (v551_data + (v543_data * v549_data));
            double v554_data = s0[44];
            double v556_data = ir1[2];
            ir1[2] = (v556_data + (v543_data * v554_data));
            double v559_data = s0[60];
            double v561_data = ir1[3];
            ir1[3] = (v561_data + (v543_data * v559_data));
            double v564_data = s0[76];
            double v566_data = ir1[4];
            ir1[4] = (v566_data + (v543_data * v564_data));
            double v569_data = s0[92];
            double v571_data = ir1[5];
            ir1[5] = (v571_data + (v543_data * v569_data));
            double v574_data = s0[108];
            double v576_data = ir1[6];
            ir1[6] = (v576_data + (v543_data * v574_data));
            double v579_data = s0[124];
            double v581_data = ir1[7];
            ir1[7] = (v581_data + (v543_data * v579_data));
            double v586_data = r0[13];
            double v587_data = s0[13];
            double v589_data = ir1[0];
            ir1[0] = (v589_data + (v586_data * v587_data));
            double v592_data = s0[29];
            double v594_data = ir1[1];
            ir1[1] = (v594_data + (v586_data * v592_data));
            double v597_data = s0[45];
            double v599_data = ir1[2];
            ir1[2] = (v599_data + (v586_data * v597_data));
            double v602_data = s0[61];
            double v604_data = ir1[3];
            ir1[3] = (v604_data + (v586_data * v602_data));
            double v607_data = s0[77];
            double v609_data = ir1[4];
            ir1[4] = (v609_data + (v586_data * v607_data));
            double v612_data = s0[93];
            double v614_data = ir1[5];
            ir1[5] = (v614_data + (v586_data * v612_data));
            double v617_data = s0[109];
            double v619_data = ir1[6];
            ir1[6] = (v619_data + (v586_data * v617_data));
            double v622_data = s0[125];
            double v624_data = ir1[7];
            ir1[7] = (v624_data + (v586_data * v622_data));
            double v629_data = r0[14];
            double v630_data = s0[14];
            double v632_data = ir1[0];
            ir1[0] = (v632_data + (v629_data * v630_data));
            double v635_data = s0[30];
            double v637_data = ir1[1];
            ir1[1] = (v637_data + (v629_data * v635_data));
            double v640_data = s0[46];
            double v642_data = ir1[2];
            ir1[2] = (v642_data + (v629_data * v640_data));
            double v645_data = s0[62];
            double v647_data = ir1[3];
            ir1[3] = (v647_data + (v629_data * v645_data));
            double v650_data = s0[78];
            double v652_data = ir1[4];
            ir1[4] = (v652_data + (v629_data * v650_data));
            double v655_data = s0[94];
            double v657_data = ir1[5];
            ir1[5] = (v657_data + (v629_data * v655_data));
            double v660_data = s0[110];
            double v662_data = ir1[6];
            ir1[6] = (v662_data + (v629_data * v660_data));
            double v665_data = s0[126];
            double v667_data = ir1[7];
            ir1[7] = (v667_data + (v629_data * v665_data));
            double v672_data = r0[15];
            double v673_data = s0[15];
            double v675_data = ir1[0];
            ir1[0] = (v675_data + (v672_data * v673_data));
            double v678_data = s0[31];
            double v680_data = ir1[1];
            ir1[1] = (v680_data + (v672_data * v678_data));
            double v683_data = s0[47];
            double v685_data = ir1[2];
            ir1[2] = (v685_data + (v672_data * v683_data));
            double v688_data = s0[63];
            double v690_data = ir1[3];
            ir1[3] = (v690_data + (v672_data * v688_data));
            double v693_data = s0[79];
            double v695_data = ir1[4];
            ir1[4] = (v695_data + (v672_data * v693_data));
            double v698_data = s0[95];
            double v700_data = ir1[5];
            ir1[5] = (v700_data + (v672_data * v698_data));
            double v703_data = s0[111];
            double v705_data = ir1[6];
            ir1[6] = (v705_data + (v672_data * v703_data));
            double v708_data = s0[127];
            double v710_data = ir1[7];
            ir1[7] = (v710_data + (v672_data * v708_data));
            #pragma unroll
            for (int32_t v715_n0 = 0; v715_n0 < 1; ++v715_n0) {
              #pragma unroll
              for (int32_t v716_n1 = 0; v716_n1 < 8; ++v716_n1) {
                int32_t v717_a = v715_n0 + v716_n1;
                int32_t v718_a = v715_n0 + v716_n1;
                double v719_data = ir1[v718_a];
                int32_t v720_a = v715_n0 + v716_n1;
                r1[v718_a] = v719_data;
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v725_i0 = 0; v725_i0 < 1; ++v725_i0) {
            int32_t v734_lead = v2_lead + (v725_i0 * 16);
            #pragma unroll
            for (int32_t v726_i1 = 0; v726_i1 < 8; ++v726_i1) {
              int32_t v727_a = v725_i0 + v726_i1;
              double v729_data = r1[(v725_i0 + v726_i1)];
              int32_t v736_a = v734_lead + (v726_i1 * 16);
              glb_m0[v736_a] = v729_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

