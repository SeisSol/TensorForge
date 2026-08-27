// === base name ===
kernel_16c847f49d

// === header ===
void launcher_kernel_16c847f49d(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_16c847f49d(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_16c847f49d, block.x * block.y * block.z, 2304 * sizeof(double));
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
        cudaFuncSetAttribute(kernel_kernel_16c847f49d, cudaFuncAttributeMaxDynamicSharedMemorySize, 2304 * sizeof(double));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_16c847f49d<<<grid,block,2304 * sizeof(double),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_16c847f49d(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 12×8(12×8) {0..12}×{0..8} strided
    // m1 12×16(12×16) {0..12}×{0..16} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m1 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
          double *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v3_lead = threadIdx.x % 16;
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 16; ++v5_i1) {
              int32_t v11_a = v5_i1 * 12;
              int32_t v12_a = v3_lead + v11_a;
              double v20_data = __ldcg(&glb_m1[(v3_lead + v11_a)]);
              int32_t v21_a = 0 + v5_i1;
              r0[v21_a] = v20_data;
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
          double r1[8]{};
          // r1 = load{g>r}(glb_m0);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 8; ++v27_i1) {
              int32_t v33_a = v27_i1 * 12;
              int32_t v34_a = v3_lead + v33_a;
              double v42_data = glb_m0[(v3_lead + v33_a)];
              int32_t v43_a = 0 + v27_i1;
              r1[v43_a] = v42_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          // wait(r1 = load{g>r}(glb_m0););
          double r2[8]{};
          __syncwarp();
          // r2 = +(r0 * s0) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 16)]
          double ir2[8]{};
          if (v3_lead < 12) {
            double v50_data = r0[0];
            double v51_data = s0[0];
            double v53_data = ir2[0];
            ir2[0] = (v53_data + (v50_data * v51_data));
            double v56_data = s0[16];
            double v58_data = ir2[1];
            ir2[1] = (v58_data + (v50_data * v56_data));
            double v61_data = s0[32];
            double v63_data = ir2[2];
            ir2[2] = (v63_data + (v50_data * v61_data));
            double v66_data = s0[48];
            double v68_data = ir2[3];
            ir2[3] = (v68_data + (v50_data * v66_data));
            double v71_data = s0[64];
            double v73_data = ir2[4];
            ir2[4] = (v73_data + (v50_data * v71_data));
            double v76_data = s0[80];
            double v78_data = ir2[5];
            ir2[5] = (v78_data + (v50_data * v76_data));
            double v81_data = s0[96];
            double v83_data = ir2[6];
            ir2[6] = (v83_data + (v50_data * v81_data));
            double v86_data = s0[112];
            double v88_data = ir2[7];
            ir2[7] = (v88_data + (v50_data * v86_data));
          }
          if (v3_lead < 12) {
            double v94_data = r0[1];
            double v95_data = s0[1];
            double v97_data = ir2[0];
            ir2[0] = (v97_data + (v94_data * v95_data));
            double v100_data = s0[17];
            double v102_data = ir2[1];
            ir2[1] = (v102_data + (v94_data * v100_data));
            double v105_data = s0[33];
            double v107_data = ir2[2];
            ir2[2] = (v107_data + (v94_data * v105_data));
            double v110_data = s0[49];
            double v112_data = ir2[3];
            ir2[3] = (v112_data + (v94_data * v110_data));
            double v115_data = s0[65];
            double v117_data = ir2[4];
            ir2[4] = (v117_data + (v94_data * v115_data));
            double v120_data = s0[81];
            double v122_data = ir2[5];
            ir2[5] = (v122_data + (v94_data * v120_data));
            double v125_data = s0[97];
            double v127_data = ir2[6];
            ir2[6] = (v127_data + (v94_data * v125_data));
            double v130_data = s0[113];
            double v132_data = ir2[7];
            ir2[7] = (v132_data + (v94_data * v130_data));
          }
          if (v3_lead < 12) {
            double v138_data = r0[2];
            double v139_data = s0[2];
            double v141_data = ir2[0];
            ir2[0] = (v141_data + (v138_data * v139_data));
            double v144_data = s0[18];
            double v146_data = ir2[1];
            ir2[1] = (v146_data + (v138_data * v144_data));
            double v149_data = s0[34];
            double v151_data = ir2[2];
            ir2[2] = (v151_data + (v138_data * v149_data));
            double v154_data = s0[50];
            double v156_data = ir2[3];
            ir2[3] = (v156_data + (v138_data * v154_data));
            double v159_data = s0[66];
            double v161_data = ir2[4];
            ir2[4] = (v161_data + (v138_data * v159_data));
            double v164_data = s0[82];
            double v166_data = ir2[5];
            ir2[5] = (v166_data + (v138_data * v164_data));
            double v169_data = s0[98];
            double v171_data = ir2[6];
            ir2[6] = (v171_data + (v138_data * v169_data));
            double v174_data = s0[114];
            double v176_data = ir2[7];
            ir2[7] = (v176_data + (v138_data * v174_data));
          }
          if (v3_lead < 12) {
            double v182_data = r0[3];
            double v183_data = s0[3];
            double v185_data = ir2[0];
            ir2[0] = (v185_data + (v182_data * v183_data));
            double v188_data = s0[19];
            double v190_data = ir2[1];
            ir2[1] = (v190_data + (v182_data * v188_data));
            double v193_data = s0[35];
            double v195_data = ir2[2];
            ir2[2] = (v195_data + (v182_data * v193_data));
            double v198_data = s0[51];
            double v200_data = ir2[3];
            ir2[3] = (v200_data + (v182_data * v198_data));
            double v203_data = s0[67];
            double v205_data = ir2[4];
            ir2[4] = (v205_data + (v182_data * v203_data));
            double v208_data = s0[83];
            double v210_data = ir2[5];
            ir2[5] = (v210_data + (v182_data * v208_data));
            double v213_data = s0[99];
            double v215_data = ir2[6];
            ir2[6] = (v215_data + (v182_data * v213_data));
            double v218_data = s0[115];
            double v220_data = ir2[7];
            ir2[7] = (v220_data + (v182_data * v218_data));
          }
          if (v3_lead < 12) {
            double v226_data = r0[4];
            double v227_data = s0[4];
            double v229_data = ir2[0];
            ir2[0] = (v229_data + (v226_data * v227_data));
            double v232_data = s0[20];
            double v234_data = ir2[1];
            ir2[1] = (v234_data + (v226_data * v232_data));
            double v237_data = s0[36];
            double v239_data = ir2[2];
            ir2[2] = (v239_data + (v226_data * v237_data));
            double v242_data = s0[52];
            double v244_data = ir2[3];
            ir2[3] = (v244_data + (v226_data * v242_data));
            double v247_data = s0[68];
            double v249_data = ir2[4];
            ir2[4] = (v249_data + (v226_data * v247_data));
            double v252_data = s0[84];
            double v254_data = ir2[5];
            ir2[5] = (v254_data + (v226_data * v252_data));
            double v257_data = s0[100];
            double v259_data = ir2[6];
            ir2[6] = (v259_data + (v226_data * v257_data));
            double v262_data = s0[116];
            double v264_data = ir2[7];
            ir2[7] = (v264_data + (v226_data * v262_data));
          }
          if (v3_lead < 12) {
            double v270_data = r0[5];
            double v271_data = s0[5];
            double v273_data = ir2[0];
            ir2[0] = (v273_data + (v270_data * v271_data));
            double v276_data = s0[21];
            double v278_data = ir2[1];
            ir2[1] = (v278_data + (v270_data * v276_data));
            double v281_data = s0[37];
            double v283_data = ir2[2];
            ir2[2] = (v283_data + (v270_data * v281_data));
            double v286_data = s0[53];
            double v288_data = ir2[3];
            ir2[3] = (v288_data + (v270_data * v286_data));
            double v291_data = s0[69];
            double v293_data = ir2[4];
            ir2[4] = (v293_data + (v270_data * v291_data));
            double v296_data = s0[85];
            double v298_data = ir2[5];
            ir2[5] = (v298_data + (v270_data * v296_data));
            double v301_data = s0[101];
            double v303_data = ir2[6];
            ir2[6] = (v303_data + (v270_data * v301_data));
            double v306_data = s0[117];
            double v308_data = ir2[7];
            ir2[7] = (v308_data + (v270_data * v306_data));
          }
          if (v3_lead < 12) {
            double v314_data = r0[6];
            double v315_data = s0[6];
            double v317_data = ir2[0];
            ir2[0] = (v317_data + (v314_data * v315_data));
            double v320_data = s0[22];
            double v322_data = ir2[1];
            ir2[1] = (v322_data + (v314_data * v320_data));
            double v325_data = s0[38];
            double v327_data = ir2[2];
            ir2[2] = (v327_data + (v314_data * v325_data));
            double v330_data = s0[54];
            double v332_data = ir2[3];
            ir2[3] = (v332_data + (v314_data * v330_data));
            double v335_data = s0[70];
            double v337_data = ir2[4];
            ir2[4] = (v337_data + (v314_data * v335_data));
            double v340_data = s0[86];
            double v342_data = ir2[5];
            ir2[5] = (v342_data + (v314_data * v340_data));
            double v345_data = s0[102];
            double v347_data = ir2[6];
            ir2[6] = (v347_data + (v314_data * v345_data));
            double v350_data = s0[118];
            double v352_data = ir2[7];
            ir2[7] = (v352_data + (v314_data * v350_data));
          }
          if (v3_lead < 12) {
            double v358_data = r0[7];
            double v359_data = s0[7];
            double v361_data = ir2[0];
            ir2[0] = (v361_data + (v358_data * v359_data));
            double v364_data = s0[23];
            double v366_data = ir2[1];
            ir2[1] = (v366_data + (v358_data * v364_data));
            double v369_data = s0[39];
            double v371_data = ir2[2];
            ir2[2] = (v371_data + (v358_data * v369_data));
            double v374_data = s0[55];
            double v376_data = ir2[3];
            ir2[3] = (v376_data + (v358_data * v374_data));
            double v379_data = s0[71];
            double v381_data = ir2[4];
            ir2[4] = (v381_data + (v358_data * v379_data));
            double v384_data = s0[87];
            double v386_data = ir2[5];
            ir2[5] = (v386_data + (v358_data * v384_data));
            double v389_data = s0[103];
            double v391_data = ir2[6];
            ir2[6] = (v391_data + (v358_data * v389_data));
            double v394_data = s0[119];
            double v396_data = ir2[7];
            ir2[7] = (v396_data + (v358_data * v394_data));
          }
          if (v3_lead < 12) {
            double v402_data = r0[8];
            double v403_data = s0[8];
            double v405_data = ir2[0];
            ir2[0] = (v405_data + (v402_data * v403_data));
            double v408_data = s0[24];
            double v410_data = ir2[1];
            ir2[1] = (v410_data + (v402_data * v408_data));
            double v413_data = s0[40];
            double v415_data = ir2[2];
            ir2[2] = (v415_data + (v402_data * v413_data));
            double v418_data = s0[56];
            double v420_data = ir2[3];
            ir2[3] = (v420_data + (v402_data * v418_data));
            double v423_data = s0[72];
            double v425_data = ir2[4];
            ir2[4] = (v425_data + (v402_data * v423_data));
            double v428_data = s0[88];
            double v430_data = ir2[5];
            ir2[5] = (v430_data + (v402_data * v428_data));
            double v433_data = s0[104];
            double v435_data = ir2[6];
            ir2[6] = (v435_data + (v402_data * v433_data));
            double v438_data = s0[120];
            double v440_data = ir2[7];
            ir2[7] = (v440_data + (v402_data * v438_data));
          }
          if (v3_lead < 12) {
            double v446_data = r0[9];
            double v447_data = s0[9];
            double v449_data = ir2[0];
            ir2[0] = (v449_data + (v446_data * v447_data));
            double v452_data = s0[25];
            double v454_data = ir2[1];
            ir2[1] = (v454_data + (v446_data * v452_data));
            double v457_data = s0[41];
            double v459_data = ir2[2];
            ir2[2] = (v459_data + (v446_data * v457_data));
            double v462_data = s0[57];
            double v464_data = ir2[3];
            ir2[3] = (v464_data + (v446_data * v462_data));
            double v467_data = s0[73];
            double v469_data = ir2[4];
            ir2[4] = (v469_data + (v446_data * v467_data));
            double v472_data = s0[89];
            double v474_data = ir2[5];
            ir2[5] = (v474_data + (v446_data * v472_data));
            double v477_data = s0[105];
            double v479_data = ir2[6];
            ir2[6] = (v479_data + (v446_data * v477_data));
            double v482_data = s0[121];
            double v484_data = ir2[7];
            ir2[7] = (v484_data + (v446_data * v482_data));
          }
          if (v3_lead < 12) {
            double v490_data = r0[10];
            double v491_data = s0[10];
            double v493_data = ir2[0];
            ir2[0] = (v493_data + (v490_data * v491_data));
            double v496_data = s0[26];
            double v498_data = ir2[1];
            ir2[1] = (v498_data + (v490_data * v496_data));
            double v501_data = s0[42];
            double v503_data = ir2[2];
            ir2[2] = (v503_data + (v490_data * v501_data));
            double v506_data = s0[58];
            double v508_data = ir2[3];
            ir2[3] = (v508_data + (v490_data * v506_data));
            double v511_data = s0[74];
            double v513_data = ir2[4];
            ir2[4] = (v513_data + (v490_data * v511_data));
            double v516_data = s0[90];
            double v518_data = ir2[5];
            ir2[5] = (v518_data + (v490_data * v516_data));
            double v521_data = s0[106];
            double v523_data = ir2[6];
            ir2[6] = (v523_data + (v490_data * v521_data));
            double v526_data = s0[122];
            double v528_data = ir2[7];
            ir2[7] = (v528_data + (v490_data * v526_data));
          }
          if (v3_lead < 12) {
            double v534_data = r0[11];
            double v535_data = s0[11];
            double v537_data = ir2[0];
            ir2[0] = (v537_data + (v534_data * v535_data));
            double v540_data = s0[27];
            double v542_data = ir2[1];
            ir2[1] = (v542_data + (v534_data * v540_data));
            double v545_data = s0[43];
            double v547_data = ir2[2];
            ir2[2] = (v547_data + (v534_data * v545_data));
            double v550_data = s0[59];
            double v552_data = ir2[3];
            ir2[3] = (v552_data + (v534_data * v550_data));
            double v555_data = s0[75];
            double v557_data = ir2[4];
            ir2[4] = (v557_data + (v534_data * v555_data));
            double v560_data = s0[91];
            double v562_data = ir2[5];
            ir2[5] = (v562_data + (v534_data * v560_data));
            double v565_data = s0[107];
            double v567_data = ir2[6];
            ir2[6] = (v567_data + (v534_data * v565_data));
            double v570_data = s0[123];
            double v572_data = ir2[7];
            ir2[7] = (v572_data + (v534_data * v570_data));
          }
          if (v3_lead < 12) {
            double v578_data = r0[12];
            double v579_data = s0[12];
            double v581_data = ir2[0];
            ir2[0] = (v581_data + (v578_data * v579_data));
            double v584_data = s0[28];
            double v586_data = ir2[1];
            ir2[1] = (v586_data + (v578_data * v584_data));
            double v589_data = s0[44];
            double v591_data = ir2[2];
            ir2[2] = (v591_data + (v578_data * v589_data));
            double v594_data = s0[60];
            double v596_data = ir2[3];
            ir2[3] = (v596_data + (v578_data * v594_data));
            double v599_data = s0[76];
            double v601_data = ir2[4];
            ir2[4] = (v601_data + (v578_data * v599_data));
            double v604_data = s0[92];
            double v606_data = ir2[5];
            ir2[5] = (v606_data + (v578_data * v604_data));
            double v609_data = s0[108];
            double v611_data = ir2[6];
            ir2[6] = (v611_data + (v578_data * v609_data));
            double v614_data = s0[124];
            double v616_data = ir2[7];
            ir2[7] = (v616_data + (v578_data * v614_data));
          }
          if (v3_lead < 12) {
            double v622_data = r0[13];
            double v623_data = s0[13];
            double v625_data = ir2[0];
            ir2[0] = (v625_data + (v622_data * v623_data));
            double v628_data = s0[29];
            double v630_data = ir2[1];
            ir2[1] = (v630_data + (v622_data * v628_data));
            double v633_data = s0[45];
            double v635_data = ir2[2];
            ir2[2] = (v635_data + (v622_data * v633_data));
            double v638_data = s0[61];
            double v640_data = ir2[3];
            ir2[3] = (v640_data + (v622_data * v638_data));
            double v643_data = s0[77];
            double v645_data = ir2[4];
            ir2[4] = (v645_data + (v622_data * v643_data));
            double v648_data = s0[93];
            double v650_data = ir2[5];
            ir2[5] = (v650_data + (v622_data * v648_data));
            double v653_data = s0[109];
            double v655_data = ir2[6];
            ir2[6] = (v655_data + (v622_data * v653_data));
            double v658_data = s0[125];
            double v660_data = ir2[7];
            ir2[7] = (v660_data + (v622_data * v658_data));
          }
          if (v3_lead < 12) {
            double v666_data = r0[14];
            double v667_data = s0[14];
            double v669_data = ir2[0];
            ir2[0] = (v669_data + (v666_data * v667_data));
            double v672_data = s0[30];
            double v674_data = ir2[1];
            ir2[1] = (v674_data + (v666_data * v672_data));
            double v677_data = s0[46];
            double v679_data = ir2[2];
            ir2[2] = (v679_data + (v666_data * v677_data));
            double v682_data = s0[62];
            double v684_data = ir2[3];
            ir2[3] = (v684_data + (v666_data * v682_data));
            double v687_data = s0[78];
            double v689_data = ir2[4];
            ir2[4] = (v689_data + (v666_data * v687_data));
            double v692_data = s0[94];
            double v694_data = ir2[5];
            ir2[5] = (v694_data + (v666_data * v692_data));
            double v697_data = s0[110];
            double v699_data = ir2[6];
            ir2[6] = (v699_data + (v666_data * v697_data));
            double v702_data = s0[126];
            double v704_data = ir2[7];
            ir2[7] = (v704_data + (v666_data * v702_data));
          }
          if (v3_lead < 12) {
            double v710_data = r0[15];
            double v711_data = s0[15];
            double v713_data = ir2[0];
            ir2[0] = (v713_data + (v710_data * v711_data));
            double v716_data = s0[31];
            double v718_data = ir2[1];
            ir2[1] = (v718_data + (v710_data * v716_data));
            double v721_data = s0[47];
            double v723_data = ir2[2];
            ir2[2] = (v723_data + (v710_data * v721_data));
            double v726_data = s0[63];
            double v728_data = ir2[3];
            ir2[3] = (v728_data + (v710_data * v726_data));
            double v731_data = s0[79];
            double v733_data = ir2[4];
            ir2[4] = (v733_data + (v710_data * v731_data));
            double v736_data = s0[95];
            double v738_data = ir2[5];
            ir2[5] = (v738_data + (v710_data * v736_data));
            double v741_data = s0[111];
            double v743_data = ir2[6];
            ir2[6] = (v743_data + (v710_data * v741_data));
            double v746_data = s0[127];
            double v748_data = ir2[7];
            ir2[7] = (v748_data + (v710_data * v746_data));
          }
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v754_n1 = 0; v754_n1 < 8; ++v754_n1) {
              int32_t v755_a = 0 + v754_n1;
              double v757_data = ir2[v754_n1];
              int32_t v758_a = 0 + v754_n1;
              double v760_data = r1[v754_n1];
              int32_t v762_a = 0 + v754_n1;
              r2[v754_n1] = (v760_data + v757_data);
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v768_i1 = 0; v768_i1 < 8; ++v768_i1) {
              int32_t v769_a = 0 + v768_i1;
              double v771_data = r2[v768_i1];
              int32_t v778_a = v3_lead + (v768_i1 * 12);
              glb_m0[v778_a] = v771_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

