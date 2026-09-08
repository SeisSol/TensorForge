// === base name ===
kernel_3e24e7feaf

// === header ===
void launcher_kernel_3e24e7feaf(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_3e24e7feaf(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_3e24e7feaf(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_3e24e7feaf(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (2560, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 32×32(6×12) {0..6}×{0..12} strided
        // m1 32×32(12×12) {0..12}×{0..12} strided
        // m2 32×32(6×12) {0..6}×{0..12} strided
        // m3 32×32(12×12) {0..12}×{0..12} strided
        // m4 32×32(12×12) {0..12}×{0..12} strided
        // t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..6}×{0..12})[0, 1] = m0 32×32(6×12) {0..6}×{0..12} strided({0..6}×{0..12})[0, -1]×m1 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[-1, 1]
        // t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..6}×{0..12})[0, 1] = m2 32×32(6×12) {0..6}×{0..12} strided({0..6}×{0..12})[0, -1]×m1 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[-1, 1]
        // m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, 1] = m4 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..12}×{0..12})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[160 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[144];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[batchId0 * 144 + 0 + m4_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v10_lead = item.get_local_id(0) % 16;
              if (v10_lead < 6) {
                #pragma unroll
                for (int32_t v12_i1 = 0; v12_i1 < 12; ++v12_i1) {
                  float v20_data = glb_m0[(v10_lead + (v12_i1 * 6))];
                  r0[v12_i1] = v20_data;
                }
              }
              float r1[12]{};
              // r1 = load{g>r}(glb_m1);
              if (v10_lead < 12) {
                #pragma unroll
                for (int32_t v27_i1 = 0; v27_i1 < 12; ++v27_i1) {
                  float v35_data = glb_m1[(v10_lead + (v27_i1 * 12))];
                  r1[v27_i1] = v35_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[12]{};
              // r3 = load{g>r}(glb_m2);
              if (v10_lead < 6) {
                #pragma unroll
                for (int32_t v42_i1 = 0; v42_i1 < 12; ++v42_i1) {
                  float v50_data = glb_m2[(v10_lead + (v42_i1 * 6))];
                  r3[v42_i1] = v50_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[12]{};
              // r2 = +(r0 * r1) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              if (v10_lead < 6) {
                float v57_data = r0[0];
                float v58_data = r1[0];
                float v61_data = r2[0];
                r2[0] = (v61_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v58_data, 0))));
                float v64_data = r1[1];
                float v67_data = r2[1];
                r2[1] = (v67_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v64_data, 0))));
                float v70_data = r1[2];
                float v73_data = r2[2];
                r2[2] = (v73_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v70_data, 0))));
                float v76_data = r1[3];
                float v79_data = r2[3];
                r2[3] = (v79_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v76_data, 0))));
                float v82_data = r1[4];
                float v85_data = r2[4];
                r2[4] = (v85_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v82_data, 0))));
                float v88_data = r1[5];
                float v91_data = r2[5];
                r2[5] = (v91_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v88_data, 0))));
                float v94_data = r1[6];
                float v97_data = r2[6];
                r2[6] = (v97_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v94_data, 0))));
                float v100_data = r1[7];
                float v103_data = r2[7];
                r2[7] = (v103_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v100_data, 0))));
                float v106_data = r1[8];
                float v109_data = r2[8];
                r2[8] = (v109_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v106_data, 0))));
                float v112_data = r1[9];
                float v115_data = r2[9];
                r2[9] = (v115_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v112_data, 0))));
                float v118_data = r1[10];
                float v121_data = r2[10];
                r2[10] = (v121_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v118_data, 0))));
                float v124_data = r1[11];
                float v127_data = r2[11];
                r2[11] = (v127_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v124_data, 0))));
              }
              if (v10_lead < 6) {
                float v133_data = r0[1];
                float v134_data = r1[0];
                float v137_data = r2[0];
                r2[0] = (v137_data + (v133_data * (sycl::group_broadcast(item.get_sub_group(), v134_data, 1))));
                float v140_data = r1[1];
                float v143_data = r2[1];
                r2[1] = (v143_data + (v133_data * (sycl::group_broadcast(item.get_sub_group(), v140_data, 1))));
                float v146_data = r1[2];
                float v149_data = r2[2];
                r2[2] = (v149_data + (v133_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 1))));
                float v152_data = r1[3];
                float v155_data = r2[3];
                r2[3] = (v155_data + (v133_data * (sycl::group_broadcast(item.get_sub_group(), v152_data, 1))));
                float v158_data = r1[4];
                float v161_data = r2[4];
                r2[4] = (v161_data + (v133_data * (sycl::group_broadcast(item.get_sub_group(), v158_data, 1))));
                float v164_data = r1[5];
                float v167_data = r2[5];
                r2[5] = (v167_data + (v133_data * (sycl::group_broadcast(item.get_sub_group(), v164_data, 1))));
                float v170_data = r1[6];
                float v173_data = r2[6];
                r2[6] = (v173_data + (v133_data * (sycl::group_broadcast(item.get_sub_group(), v170_data, 1))));
                float v176_data = r1[7];
                float v179_data = r2[7];
                r2[7] = (v179_data + (v133_data * (sycl::group_broadcast(item.get_sub_group(), v176_data, 1))));
                float v182_data = r1[8];
                float v185_data = r2[8];
                r2[8] = (v185_data + (v133_data * (sycl::group_broadcast(item.get_sub_group(), v182_data, 1))));
                float v188_data = r1[9];
                float v191_data = r2[9];
                r2[9] = (v191_data + (v133_data * (sycl::group_broadcast(item.get_sub_group(), v188_data, 1))));
                float v194_data = r1[10];
                float v197_data = r2[10];
                r2[10] = (v197_data + (v133_data * (sycl::group_broadcast(item.get_sub_group(), v194_data, 1))));
                float v200_data = r1[11];
                float v203_data = r2[11];
                r2[11] = (v203_data + (v133_data * (sycl::group_broadcast(item.get_sub_group(), v200_data, 1))));
              }
              if (v10_lead < 6) {
                float v209_data = r0[2];
                float v210_data = r1[0];
                float v213_data = r2[0];
                r2[0] = (v213_data + (v209_data * (sycl::group_broadcast(item.get_sub_group(), v210_data, 2))));
                float v216_data = r1[1];
                float v219_data = r2[1];
                r2[1] = (v219_data + (v209_data * (sycl::group_broadcast(item.get_sub_group(), v216_data, 2))));
                float v222_data = r1[2];
                float v225_data = r2[2];
                r2[2] = (v225_data + (v209_data * (sycl::group_broadcast(item.get_sub_group(), v222_data, 2))));
                float v228_data = r1[3];
                float v231_data = r2[3];
                r2[3] = (v231_data + (v209_data * (sycl::group_broadcast(item.get_sub_group(), v228_data, 2))));
                float v234_data = r1[4];
                float v237_data = r2[4];
                r2[4] = (v237_data + (v209_data * (sycl::group_broadcast(item.get_sub_group(), v234_data, 2))));
                float v240_data = r1[5];
                float v243_data = r2[5];
                r2[5] = (v243_data + (v209_data * (sycl::group_broadcast(item.get_sub_group(), v240_data, 2))));
                float v246_data = r1[6];
                float v249_data = r2[6];
                r2[6] = (v249_data + (v209_data * (sycl::group_broadcast(item.get_sub_group(), v246_data, 2))));
                float v252_data = r1[7];
                float v255_data = r2[7];
                r2[7] = (v255_data + (v209_data * (sycl::group_broadcast(item.get_sub_group(), v252_data, 2))));
                float v258_data = r1[8];
                float v261_data = r2[8];
                r2[8] = (v261_data + (v209_data * (sycl::group_broadcast(item.get_sub_group(), v258_data, 2))));
                float v264_data = r1[9];
                float v267_data = r2[9];
                r2[9] = (v267_data + (v209_data * (sycl::group_broadcast(item.get_sub_group(), v264_data, 2))));
                float v270_data = r1[10];
                float v273_data = r2[10];
                r2[10] = (v273_data + (v209_data * (sycl::group_broadcast(item.get_sub_group(), v270_data, 2))));
                float v276_data = r1[11];
                float v279_data = r2[11];
                r2[11] = (v279_data + (v209_data * (sycl::group_broadcast(item.get_sub_group(), v276_data, 2))));
              }
              if (v10_lead < 6) {
                float v285_data = r0[3];
                float v286_data = r1[0];
                float v289_data = r2[0];
                r2[0] = (v289_data + (v285_data * (sycl::group_broadcast(item.get_sub_group(), v286_data, 3))));
                float v292_data = r1[1];
                float v295_data = r2[1];
                r2[1] = (v295_data + (v285_data * (sycl::group_broadcast(item.get_sub_group(), v292_data, 3))));
                float v298_data = r1[2];
                float v301_data = r2[2];
                r2[2] = (v301_data + (v285_data * (sycl::group_broadcast(item.get_sub_group(), v298_data, 3))));
                float v304_data = r1[3];
                float v307_data = r2[3];
                r2[3] = (v307_data + (v285_data * (sycl::group_broadcast(item.get_sub_group(), v304_data, 3))));
                float v310_data = r1[4];
                float v313_data = r2[4];
                r2[4] = (v313_data + (v285_data * (sycl::group_broadcast(item.get_sub_group(), v310_data, 3))));
                float v316_data = r1[5];
                float v319_data = r2[5];
                r2[5] = (v319_data + (v285_data * (sycl::group_broadcast(item.get_sub_group(), v316_data, 3))));
                float v322_data = r1[6];
                float v325_data = r2[6];
                r2[6] = (v325_data + (v285_data * (sycl::group_broadcast(item.get_sub_group(), v322_data, 3))));
                float v328_data = r1[7];
                float v331_data = r2[7];
                r2[7] = (v331_data + (v285_data * (sycl::group_broadcast(item.get_sub_group(), v328_data, 3))));
                float v334_data = r1[8];
                float v337_data = r2[8];
                r2[8] = (v337_data + (v285_data * (sycl::group_broadcast(item.get_sub_group(), v334_data, 3))));
                float v340_data = r1[9];
                float v343_data = r2[9];
                r2[9] = (v343_data + (v285_data * (sycl::group_broadcast(item.get_sub_group(), v340_data, 3))));
                float v346_data = r1[10];
                float v349_data = r2[10];
                r2[10] = (v349_data + (v285_data * (sycl::group_broadcast(item.get_sub_group(), v346_data, 3))));
                float v352_data = r1[11];
                float v355_data = r2[11];
                r2[11] = (v355_data + (v285_data * (sycl::group_broadcast(item.get_sub_group(), v352_data, 3))));
              }
              if (v10_lead < 6) {
                float v361_data = r0[4];
                float v362_data = r1[0];
                float v365_data = r2[0];
                r2[0] = (v365_data + (v361_data * (sycl::group_broadcast(item.get_sub_group(), v362_data, 4))));
                float v368_data = r1[1];
                float v371_data = r2[1];
                r2[1] = (v371_data + (v361_data * (sycl::group_broadcast(item.get_sub_group(), v368_data, 4))));
                float v374_data = r1[2];
                float v377_data = r2[2];
                r2[2] = (v377_data + (v361_data * (sycl::group_broadcast(item.get_sub_group(), v374_data, 4))));
                float v380_data = r1[3];
                float v383_data = r2[3];
                r2[3] = (v383_data + (v361_data * (sycl::group_broadcast(item.get_sub_group(), v380_data, 4))));
                float v386_data = r1[4];
                float v389_data = r2[4];
                r2[4] = (v389_data + (v361_data * (sycl::group_broadcast(item.get_sub_group(), v386_data, 4))));
                float v392_data = r1[5];
                float v395_data = r2[5];
                r2[5] = (v395_data + (v361_data * (sycl::group_broadcast(item.get_sub_group(), v392_data, 4))));
                float v398_data = r1[6];
                float v401_data = r2[6];
                r2[6] = (v401_data + (v361_data * (sycl::group_broadcast(item.get_sub_group(), v398_data, 4))));
                float v404_data = r1[7];
                float v407_data = r2[7];
                r2[7] = (v407_data + (v361_data * (sycl::group_broadcast(item.get_sub_group(), v404_data, 4))));
                float v410_data = r1[8];
                float v413_data = r2[8];
                r2[8] = (v413_data + (v361_data * (sycl::group_broadcast(item.get_sub_group(), v410_data, 4))));
                float v416_data = r1[9];
                float v419_data = r2[9];
                r2[9] = (v419_data + (v361_data * (sycl::group_broadcast(item.get_sub_group(), v416_data, 4))));
                float v422_data = r1[10];
                float v425_data = r2[10];
                r2[10] = (v425_data + (v361_data * (sycl::group_broadcast(item.get_sub_group(), v422_data, 4))));
                float v428_data = r1[11];
                float v431_data = r2[11];
                r2[11] = (v431_data + (v361_data * (sycl::group_broadcast(item.get_sub_group(), v428_data, 4))));
              }
              if (v10_lead < 6) {
                float v437_data = r0[5];
                float v438_data = r1[0];
                float v441_data = r2[0];
                r2[0] = (v441_data + (v437_data * (sycl::group_broadcast(item.get_sub_group(), v438_data, 5))));
                float v444_data = r1[1];
                float v447_data = r2[1];
                r2[1] = (v447_data + (v437_data * (sycl::group_broadcast(item.get_sub_group(), v444_data, 5))));
                float v450_data = r1[2];
                float v453_data = r2[2];
                r2[2] = (v453_data + (v437_data * (sycl::group_broadcast(item.get_sub_group(), v450_data, 5))));
                float v456_data = r1[3];
                float v459_data = r2[3];
                r2[3] = (v459_data + (v437_data * (sycl::group_broadcast(item.get_sub_group(), v456_data, 5))));
                float v462_data = r1[4];
                float v465_data = r2[4];
                r2[4] = (v465_data + (v437_data * (sycl::group_broadcast(item.get_sub_group(), v462_data, 5))));
                float v468_data = r1[5];
                float v471_data = r2[5];
                r2[5] = (v471_data + (v437_data * (sycl::group_broadcast(item.get_sub_group(), v468_data, 5))));
                float v474_data = r1[6];
                float v477_data = r2[6];
                r2[6] = (v477_data + (v437_data * (sycl::group_broadcast(item.get_sub_group(), v474_data, 5))));
                float v480_data = r1[7];
                float v483_data = r2[7];
                r2[7] = (v483_data + (v437_data * (sycl::group_broadcast(item.get_sub_group(), v480_data, 5))));
                float v486_data = r1[8];
                float v489_data = r2[8];
                r2[8] = (v489_data + (v437_data * (sycl::group_broadcast(item.get_sub_group(), v486_data, 5))));
                float v492_data = r1[9];
                float v495_data = r2[9];
                r2[9] = (v495_data + (v437_data * (sycl::group_broadcast(item.get_sub_group(), v492_data, 5))));
                float v498_data = r1[10];
                float v501_data = r2[10];
                r2[10] = (v501_data + (v437_data * (sycl::group_broadcast(item.get_sub_group(), v498_data, 5))));
                float v504_data = r1[11];
                float v507_data = r2[11];
                r2[11] = (v507_data + (v437_data * (sycl::group_broadcast(item.get_sub_group(), v504_data, 5))));
              }
              if (v10_lead < 6) {
                float v513_data = r0[6];
                float v514_data = r1[0];
                float v517_data = r2[0];
                r2[0] = (v517_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v514_data, 6))));
                float v520_data = r1[1];
                float v523_data = r2[1];
                r2[1] = (v523_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v520_data, 6))));
                float v526_data = r1[2];
                float v529_data = r2[2];
                r2[2] = (v529_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v526_data, 6))));
                float v532_data = r1[3];
                float v535_data = r2[3];
                r2[3] = (v535_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v532_data, 6))));
                float v538_data = r1[4];
                float v541_data = r2[4];
                r2[4] = (v541_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v538_data, 6))));
                float v544_data = r1[5];
                float v547_data = r2[5];
                r2[5] = (v547_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v544_data, 6))));
                float v550_data = r1[6];
                float v553_data = r2[6];
                r2[6] = (v553_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v550_data, 6))));
                float v556_data = r1[7];
                float v559_data = r2[7];
                r2[7] = (v559_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v556_data, 6))));
                float v562_data = r1[8];
                float v565_data = r2[8];
                r2[8] = (v565_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v562_data, 6))));
                float v568_data = r1[9];
                float v571_data = r2[9];
                r2[9] = (v571_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v568_data, 6))));
                float v574_data = r1[10];
                float v577_data = r2[10];
                r2[10] = (v577_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v574_data, 6))));
                float v580_data = r1[11];
                float v583_data = r2[11];
                r2[11] = (v583_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v580_data, 6))));
              }
              if (v10_lead < 6) {
                float v589_data = r0[7];
                float v590_data = r1[0];
                float v593_data = r2[0];
                r2[0] = (v593_data + (v589_data * (sycl::group_broadcast(item.get_sub_group(), v590_data, 7))));
                float v596_data = r1[1];
                float v599_data = r2[1];
                r2[1] = (v599_data + (v589_data * (sycl::group_broadcast(item.get_sub_group(), v596_data, 7))));
                float v602_data = r1[2];
                float v605_data = r2[2];
                r2[2] = (v605_data + (v589_data * (sycl::group_broadcast(item.get_sub_group(), v602_data, 7))));
                float v608_data = r1[3];
                float v611_data = r2[3];
                r2[3] = (v611_data + (v589_data * (sycl::group_broadcast(item.get_sub_group(), v608_data, 7))));
                float v614_data = r1[4];
                float v617_data = r2[4];
                r2[4] = (v617_data + (v589_data * (sycl::group_broadcast(item.get_sub_group(), v614_data, 7))));
                float v620_data = r1[5];
                float v623_data = r2[5];
                r2[5] = (v623_data + (v589_data * (sycl::group_broadcast(item.get_sub_group(), v620_data, 7))));
                float v626_data = r1[6];
                float v629_data = r2[6];
                r2[6] = (v629_data + (v589_data * (sycl::group_broadcast(item.get_sub_group(), v626_data, 7))));
                float v632_data = r1[7];
                float v635_data = r2[7];
                r2[7] = (v635_data + (v589_data * (sycl::group_broadcast(item.get_sub_group(), v632_data, 7))));
                float v638_data = r1[8];
                float v641_data = r2[8];
                r2[8] = (v641_data + (v589_data * (sycl::group_broadcast(item.get_sub_group(), v638_data, 7))));
                float v644_data = r1[9];
                float v647_data = r2[9];
                r2[9] = (v647_data + (v589_data * (sycl::group_broadcast(item.get_sub_group(), v644_data, 7))));
                float v650_data = r1[10];
                float v653_data = r2[10];
                r2[10] = (v653_data + (v589_data * (sycl::group_broadcast(item.get_sub_group(), v650_data, 7))));
                float v656_data = r1[11];
                float v659_data = r2[11];
                r2[11] = (v659_data + (v589_data * (sycl::group_broadcast(item.get_sub_group(), v656_data, 7))));
              }
              if (v10_lead < 6) {
                float v665_data = r0[8];
                float v666_data = r1[0];
                float v669_data = r2[0];
                r2[0] = (v669_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v666_data, 8))));
                float v672_data = r1[1];
                float v675_data = r2[1];
                r2[1] = (v675_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v672_data, 8))));
                float v678_data = r1[2];
                float v681_data = r2[2];
                r2[2] = (v681_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v678_data, 8))));
                float v684_data = r1[3];
                float v687_data = r2[3];
                r2[3] = (v687_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v684_data, 8))));
                float v690_data = r1[4];
                float v693_data = r2[4];
                r2[4] = (v693_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v690_data, 8))));
                float v696_data = r1[5];
                float v699_data = r2[5];
                r2[5] = (v699_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v696_data, 8))));
                float v702_data = r1[6];
                float v705_data = r2[6];
                r2[6] = (v705_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v702_data, 8))));
                float v708_data = r1[7];
                float v711_data = r2[7];
                r2[7] = (v711_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v708_data, 8))));
                float v714_data = r1[8];
                float v717_data = r2[8];
                r2[8] = (v717_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v714_data, 8))));
                float v720_data = r1[9];
                float v723_data = r2[9];
                r2[9] = (v723_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v720_data, 8))));
                float v726_data = r1[10];
                float v729_data = r2[10];
                r2[10] = (v729_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v726_data, 8))));
                float v732_data = r1[11];
                float v735_data = r2[11];
                r2[11] = (v735_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v732_data, 8))));
              }
              if (v10_lead < 6) {
                float v741_data = r0[9];
                float v742_data = r1[0];
                float v745_data = r2[0];
                r2[0] = (v745_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v742_data, 9))));
                float v748_data = r1[1];
                float v751_data = r2[1];
                r2[1] = (v751_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v748_data, 9))));
                float v754_data = r1[2];
                float v757_data = r2[2];
                r2[2] = (v757_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v754_data, 9))));
                float v760_data = r1[3];
                float v763_data = r2[3];
                r2[3] = (v763_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v760_data, 9))));
                float v766_data = r1[4];
                float v769_data = r2[4];
                r2[4] = (v769_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v766_data, 9))));
                float v772_data = r1[5];
                float v775_data = r2[5];
                r2[5] = (v775_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v772_data, 9))));
                float v778_data = r1[6];
                float v781_data = r2[6];
                r2[6] = (v781_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v778_data, 9))));
                float v784_data = r1[7];
                float v787_data = r2[7];
                r2[7] = (v787_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v784_data, 9))));
                float v790_data = r1[8];
                float v793_data = r2[8];
                r2[8] = (v793_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v790_data, 9))));
                float v796_data = r1[9];
                float v799_data = r2[9];
                r2[9] = (v799_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v796_data, 9))));
                float v802_data = r1[10];
                float v805_data = r2[10];
                r2[10] = (v805_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v802_data, 9))));
                float v808_data = r1[11];
                float v811_data = r2[11];
                r2[11] = (v811_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v808_data, 9))));
              }
              if (v10_lead < 6) {
                float v817_data = r0[10];
                float v818_data = r1[0];
                float v821_data = r2[0];
                r2[0] = (v821_data + (v817_data * (sycl::group_broadcast(item.get_sub_group(), v818_data, 10))));
                float v824_data = r1[1];
                float v827_data = r2[1];
                r2[1] = (v827_data + (v817_data * (sycl::group_broadcast(item.get_sub_group(), v824_data, 10))));
                float v830_data = r1[2];
                float v833_data = r2[2];
                r2[2] = (v833_data + (v817_data * (sycl::group_broadcast(item.get_sub_group(), v830_data, 10))));
                float v836_data = r1[3];
                float v839_data = r2[3];
                r2[3] = (v839_data + (v817_data * (sycl::group_broadcast(item.get_sub_group(), v836_data, 10))));
                float v842_data = r1[4];
                float v845_data = r2[4];
                r2[4] = (v845_data + (v817_data * (sycl::group_broadcast(item.get_sub_group(), v842_data, 10))));
                float v848_data = r1[5];
                float v851_data = r2[5];
                r2[5] = (v851_data + (v817_data * (sycl::group_broadcast(item.get_sub_group(), v848_data, 10))));
                float v854_data = r1[6];
                float v857_data = r2[6];
                r2[6] = (v857_data + (v817_data * (sycl::group_broadcast(item.get_sub_group(), v854_data, 10))));
                float v860_data = r1[7];
                float v863_data = r2[7];
                r2[7] = (v863_data + (v817_data * (sycl::group_broadcast(item.get_sub_group(), v860_data, 10))));
                float v866_data = r1[8];
                float v869_data = r2[8];
                r2[8] = (v869_data + (v817_data * (sycl::group_broadcast(item.get_sub_group(), v866_data, 10))));
                float v872_data = r1[9];
                float v875_data = r2[9];
                r2[9] = (v875_data + (v817_data * (sycl::group_broadcast(item.get_sub_group(), v872_data, 10))));
                float v878_data = r1[10];
                float v881_data = r2[10];
                r2[10] = (v881_data + (v817_data * (sycl::group_broadcast(item.get_sub_group(), v878_data, 10))));
                float v884_data = r1[11];
                float v887_data = r2[11];
                r2[11] = (v887_data + (v817_data * (sycl::group_broadcast(item.get_sub_group(), v884_data, 10))));
              }
              if (v10_lead < 6) {
                float v893_data = r0[11];
                float v894_data = r1[0];
                float v897_data = r2[0];
                r2[0] = (v897_data + (v893_data * (sycl::group_broadcast(item.get_sub_group(), v894_data, 11))));
                float v900_data = r1[1];
                float v903_data = r2[1];
                r2[1] = (v903_data + (v893_data * (sycl::group_broadcast(item.get_sub_group(), v900_data, 11))));
                float v906_data = r1[2];
                float v909_data = r2[2];
                r2[2] = (v909_data + (v893_data * (sycl::group_broadcast(item.get_sub_group(), v906_data, 11))));
                float v912_data = r1[3];
                float v915_data = r2[3];
                r2[3] = (v915_data + (v893_data * (sycl::group_broadcast(item.get_sub_group(), v912_data, 11))));
                float v918_data = r1[4];
                float v921_data = r2[4];
                r2[4] = (v921_data + (v893_data * (sycl::group_broadcast(item.get_sub_group(), v918_data, 11))));
                float v924_data = r1[5];
                float v927_data = r2[5];
                r2[5] = (v927_data + (v893_data * (sycl::group_broadcast(item.get_sub_group(), v924_data, 11))));
                float v930_data = r1[6];
                float v933_data = r2[6];
                r2[6] = (v933_data + (v893_data * (sycl::group_broadcast(item.get_sub_group(), v930_data, 11))));
                float v936_data = r1[7];
                float v939_data = r2[7];
                r2[7] = (v939_data + (v893_data * (sycl::group_broadcast(item.get_sub_group(), v936_data, 11))));
                float v942_data = r1[8];
                float v945_data = r2[8];
                r2[8] = (v945_data + (v893_data * (sycl::group_broadcast(item.get_sub_group(), v942_data, 11))));
                float v948_data = r1[9];
                float v951_data = r2[9];
                r2[9] = (v951_data + (v893_data * (sycl::group_broadcast(item.get_sub_group(), v948_data, 11))));
                float v954_data = r1[10];
                float v957_data = r2[10];
                r2[10] = (v957_data + (v893_data * (sycl::group_broadcast(item.get_sub_group(), v954_data, 11))));
                float v960_data = r1[11];
                float v963_data = r2[11];
                r2[11] = (v963_data + (v893_data * (sycl::group_broadcast(item.get_sub_group(), v960_data, 11))));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = store{r>s}(localShrMem0, r2);
              if (v10_lead < 6) {
                #pragma unroll
                for (int32_t v970_i1 = 0; v970_i1 < 12; ++v970_i1) {
                  float v972_data = r2[v970_i1];
                  int32_t v979_a = v10_lead + (v970_i1 * 12);
                  s0[(v979_a ^ ((v979_a >> 4) & 15))] = v972_data;
                }
              }
              float r5[12]{};
              // r5 = load{g>r}(glb_m4);
              if (v10_lead < 12) {
                #pragma unroll
                for (int32_t v988_i1 = 0; v988_i1 < 12; ++v988_i1) {
                  float v996_data = glb_m4[(v10_lead + (v988_i1 * 12))];
                  r5[v988_i1] = v996_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m2););
              float r4[12]{};
              // r4 = +(r3 * r1) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              float ir4[12]{};
              if (v10_lead < 6) {
                float v1004_data = r3[0];
                float v1005_data = r1[0];
                float v1008_data = ir4[0];
                ir4[0] = (v1008_data + (v1004_data * (sycl::group_broadcast(item.get_sub_group(), v1005_data, 0))));
                float v1011_data = r1[1];
                float v1014_data = ir4[1];
                ir4[1] = (v1014_data + (v1004_data * (sycl::group_broadcast(item.get_sub_group(), v1011_data, 0))));
                float v1017_data = r1[2];
                float v1020_data = ir4[2];
                ir4[2] = (v1020_data + (v1004_data * (sycl::group_broadcast(item.get_sub_group(), v1017_data, 0))));
                float v1023_data = r1[3];
                float v1026_data = ir4[3];
                ir4[3] = (v1026_data + (v1004_data * (sycl::group_broadcast(item.get_sub_group(), v1023_data, 0))));
                float v1029_data = r1[4];
                float v1032_data = ir4[4];
                ir4[4] = (v1032_data + (v1004_data * (sycl::group_broadcast(item.get_sub_group(), v1029_data, 0))));
                float v1035_data = r1[5];
                float v1038_data = ir4[5];
                ir4[5] = (v1038_data + (v1004_data * (sycl::group_broadcast(item.get_sub_group(), v1035_data, 0))));
                float v1041_data = r1[6];
                float v1044_data = ir4[6];
                ir4[6] = (v1044_data + (v1004_data * (sycl::group_broadcast(item.get_sub_group(), v1041_data, 0))));
                float v1047_data = r1[7];
                float v1050_data = ir4[7];
                ir4[7] = (v1050_data + (v1004_data * (sycl::group_broadcast(item.get_sub_group(), v1047_data, 0))));
                float v1053_data = r1[8];
                float v1056_data = ir4[8];
                ir4[8] = (v1056_data + (v1004_data * (sycl::group_broadcast(item.get_sub_group(), v1053_data, 0))));
                float v1059_data = r1[9];
                float v1062_data = ir4[9];
                ir4[9] = (v1062_data + (v1004_data * (sycl::group_broadcast(item.get_sub_group(), v1059_data, 0))));
                float v1065_data = r1[10];
                float v1068_data = ir4[10];
                ir4[10] = (v1068_data + (v1004_data * (sycl::group_broadcast(item.get_sub_group(), v1065_data, 0))));
                float v1071_data = r1[11];
                float v1074_data = ir4[11];
                ir4[11] = (v1074_data + (v1004_data * (sycl::group_broadcast(item.get_sub_group(), v1071_data, 0))));
              }
              if (v10_lead < 6) {
                float v1080_data = r3[1];
                float v1081_data = r1[0];
                float v1084_data = ir4[0];
                ir4[0] = (v1084_data + (v1080_data * (sycl::group_broadcast(item.get_sub_group(), v1081_data, 1))));
                float v1087_data = r1[1];
                float v1090_data = ir4[1];
                ir4[1] = (v1090_data + (v1080_data * (sycl::group_broadcast(item.get_sub_group(), v1087_data, 1))));
                float v1093_data = r1[2];
                float v1096_data = ir4[2];
                ir4[2] = (v1096_data + (v1080_data * (sycl::group_broadcast(item.get_sub_group(), v1093_data, 1))));
                float v1099_data = r1[3];
                float v1102_data = ir4[3];
                ir4[3] = (v1102_data + (v1080_data * (sycl::group_broadcast(item.get_sub_group(), v1099_data, 1))));
                float v1105_data = r1[4];
                float v1108_data = ir4[4];
                ir4[4] = (v1108_data + (v1080_data * (sycl::group_broadcast(item.get_sub_group(), v1105_data, 1))));
                float v1111_data = r1[5];
                float v1114_data = ir4[5];
                ir4[5] = (v1114_data + (v1080_data * (sycl::group_broadcast(item.get_sub_group(), v1111_data, 1))));
                float v1117_data = r1[6];
                float v1120_data = ir4[6];
                ir4[6] = (v1120_data + (v1080_data * (sycl::group_broadcast(item.get_sub_group(), v1117_data, 1))));
                float v1123_data = r1[7];
                float v1126_data = ir4[7];
                ir4[7] = (v1126_data + (v1080_data * (sycl::group_broadcast(item.get_sub_group(), v1123_data, 1))));
                float v1129_data = r1[8];
                float v1132_data = ir4[8];
                ir4[8] = (v1132_data + (v1080_data * (sycl::group_broadcast(item.get_sub_group(), v1129_data, 1))));
                float v1135_data = r1[9];
                float v1138_data = ir4[9];
                ir4[9] = (v1138_data + (v1080_data * (sycl::group_broadcast(item.get_sub_group(), v1135_data, 1))));
                float v1141_data = r1[10];
                float v1144_data = ir4[10];
                ir4[10] = (v1144_data + (v1080_data * (sycl::group_broadcast(item.get_sub_group(), v1141_data, 1))));
                float v1147_data = r1[11];
                float v1150_data = ir4[11];
                ir4[11] = (v1150_data + (v1080_data * (sycl::group_broadcast(item.get_sub_group(), v1147_data, 1))));
              }
              if (v10_lead < 6) {
                float v1156_data = r3[2];
                float v1157_data = r1[0];
                float v1160_data = ir4[0];
                ir4[0] = (v1160_data + (v1156_data * (sycl::group_broadcast(item.get_sub_group(), v1157_data, 2))));
                float v1163_data = r1[1];
                float v1166_data = ir4[1];
                ir4[1] = (v1166_data + (v1156_data * (sycl::group_broadcast(item.get_sub_group(), v1163_data, 2))));
                float v1169_data = r1[2];
                float v1172_data = ir4[2];
                ir4[2] = (v1172_data + (v1156_data * (sycl::group_broadcast(item.get_sub_group(), v1169_data, 2))));
                float v1175_data = r1[3];
                float v1178_data = ir4[3];
                ir4[3] = (v1178_data + (v1156_data * (sycl::group_broadcast(item.get_sub_group(), v1175_data, 2))));
                float v1181_data = r1[4];
                float v1184_data = ir4[4];
                ir4[4] = (v1184_data + (v1156_data * (sycl::group_broadcast(item.get_sub_group(), v1181_data, 2))));
                float v1187_data = r1[5];
                float v1190_data = ir4[5];
                ir4[5] = (v1190_data + (v1156_data * (sycl::group_broadcast(item.get_sub_group(), v1187_data, 2))));
                float v1193_data = r1[6];
                float v1196_data = ir4[6];
                ir4[6] = (v1196_data + (v1156_data * (sycl::group_broadcast(item.get_sub_group(), v1193_data, 2))));
                float v1199_data = r1[7];
                float v1202_data = ir4[7];
                ir4[7] = (v1202_data + (v1156_data * (sycl::group_broadcast(item.get_sub_group(), v1199_data, 2))));
                float v1205_data = r1[8];
                float v1208_data = ir4[8];
                ir4[8] = (v1208_data + (v1156_data * (sycl::group_broadcast(item.get_sub_group(), v1205_data, 2))));
                float v1211_data = r1[9];
                float v1214_data = ir4[9];
                ir4[9] = (v1214_data + (v1156_data * (sycl::group_broadcast(item.get_sub_group(), v1211_data, 2))));
                float v1217_data = r1[10];
                float v1220_data = ir4[10];
                ir4[10] = (v1220_data + (v1156_data * (sycl::group_broadcast(item.get_sub_group(), v1217_data, 2))));
                float v1223_data = r1[11];
                float v1226_data = ir4[11];
                ir4[11] = (v1226_data + (v1156_data * (sycl::group_broadcast(item.get_sub_group(), v1223_data, 2))));
              }
              if (v10_lead < 6) {
                float v1232_data = r3[3];
                float v1233_data = r1[0];
                float v1236_data = ir4[0];
                ir4[0] = (v1236_data + (v1232_data * (sycl::group_broadcast(item.get_sub_group(), v1233_data, 3))));
                float v1239_data = r1[1];
                float v1242_data = ir4[1];
                ir4[1] = (v1242_data + (v1232_data * (sycl::group_broadcast(item.get_sub_group(), v1239_data, 3))));
                float v1245_data = r1[2];
                float v1248_data = ir4[2];
                ir4[2] = (v1248_data + (v1232_data * (sycl::group_broadcast(item.get_sub_group(), v1245_data, 3))));
                float v1251_data = r1[3];
                float v1254_data = ir4[3];
                ir4[3] = (v1254_data + (v1232_data * (sycl::group_broadcast(item.get_sub_group(), v1251_data, 3))));
                float v1257_data = r1[4];
                float v1260_data = ir4[4];
                ir4[4] = (v1260_data + (v1232_data * (sycl::group_broadcast(item.get_sub_group(), v1257_data, 3))));
                float v1263_data = r1[5];
                float v1266_data = ir4[5];
                ir4[5] = (v1266_data + (v1232_data * (sycl::group_broadcast(item.get_sub_group(), v1263_data, 3))));
                float v1269_data = r1[6];
                float v1272_data = ir4[6];
                ir4[6] = (v1272_data + (v1232_data * (sycl::group_broadcast(item.get_sub_group(), v1269_data, 3))));
                float v1275_data = r1[7];
                float v1278_data = ir4[7];
                ir4[7] = (v1278_data + (v1232_data * (sycl::group_broadcast(item.get_sub_group(), v1275_data, 3))));
                float v1281_data = r1[8];
                float v1284_data = ir4[8];
                ir4[8] = (v1284_data + (v1232_data * (sycl::group_broadcast(item.get_sub_group(), v1281_data, 3))));
                float v1287_data = r1[9];
                float v1290_data = ir4[9];
                ir4[9] = (v1290_data + (v1232_data * (sycl::group_broadcast(item.get_sub_group(), v1287_data, 3))));
                float v1293_data = r1[10];
                float v1296_data = ir4[10];
                ir4[10] = (v1296_data + (v1232_data * (sycl::group_broadcast(item.get_sub_group(), v1293_data, 3))));
                float v1299_data = r1[11];
                float v1302_data = ir4[11];
                ir4[11] = (v1302_data + (v1232_data * (sycl::group_broadcast(item.get_sub_group(), v1299_data, 3))));
              }
              if (v10_lead < 6) {
                float v1308_data = r3[4];
                float v1309_data = r1[0];
                float v1312_data = ir4[0];
                ir4[0] = (v1312_data + (v1308_data * (sycl::group_broadcast(item.get_sub_group(), v1309_data, 4))));
                float v1315_data = r1[1];
                float v1318_data = ir4[1];
                ir4[1] = (v1318_data + (v1308_data * (sycl::group_broadcast(item.get_sub_group(), v1315_data, 4))));
                float v1321_data = r1[2];
                float v1324_data = ir4[2];
                ir4[2] = (v1324_data + (v1308_data * (sycl::group_broadcast(item.get_sub_group(), v1321_data, 4))));
                float v1327_data = r1[3];
                float v1330_data = ir4[3];
                ir4[3] = (v1330_data + (v1308_data * (sycl::group_broadcast(item.get_sub_group(), v1327_data, 4))));
                float v1333_data = r1[4];
                float v1336_data = ir4[4];
                ir4[4] = (v1336_data + (v1308_data * (sycl::group_broadcast(item.get_sub_group(), v1333_data, 4))));
                float v1339_data = r1[5];
                float v1342_data = ir4[5];
                ir4[5] = (v1342_data + (v1308_data * (sycl::group_broadcast(item.get_sub_group(), v1339_data, 4))));
                float v1345_data = r1[6];
                float v1348_data = ir4[6];
                ir4[6] = (v1348_data + (v1308_data * (sycl::group_broadcast(item.get_sub_group(), v1345_data, 4))));
                float v1351_data = r1[7];
                float v1354_data = ir4[7];
                ir4[7] = (v1354_data + (v1308_data * (sycl::group_broadcast(item.get_sub_group(), v1351_data, 4))));
                float v1357_data = r1[8];
                float v1360_data = ir4[8];
                ir4[8] = (v1360_data + (v1308_data * (sycl::group_broadcast(item.get_sub_group(), v1357_data, 4))));
                float v1363_data = r1[9];
                float v1366_data = ir4[9];
                ir4[9] = (v1366_data + (v1308_data * (sycl::group_broadcast(item.get_sub_group(), v1363_data, 4))));
                float v1369_data = r1[10];
                float v1372_data = ir4[10];
                ir4[10] = (v1372_data + (v1308_data * (sycl::group_broadcast(item.get_sub_group(), v1369_data, 4))));
                float v1375_data = r1[11];
                float v1378_data = ir4[11];
                ir4[11] = (v1378_data + (v1308_data * (sycl::group_broadcast(item.get_sub_group(), v1375_data, 4))));
              }
              if (v10_lead < 6) {
                float v1384_data = r3[5];
                float v1385_data = r1[0];
                float v1388_data = ir4[0];
                ir4[0] = (v1388_data + (v1384_data * (sycl::group_broadcast(item.get_sub_group(), v1385_data, 5))));
                float v1391_data = r1[1];
                float v1394_data = ir4[1];
                ir4[1] = (v1394_data + (v1384_data * (sycl::group_broadcast(item.get_sub_group(), v1391_data, 5))));
                float v1397_data = r1[2];
                float v1400_data = ir4[2];
                ir4[2] = (v1400_data + (v1384_data * (sycl::group_broadcast(item.get_sub_group(), v1397_data, 5))));
                float v1403_data = r1[3];
                float v1406_data = ir4[3];
                ir4[3] = (v1406_data + (v1384_data * (sycl::group_broadcast(item.get_sub_group(), v1403_data, 5))));
                float v1409_data = r1[4];
                float v1412_data = ir4[4];
                ir4[4] = (v1412_data + (v1384_data * (sycl::group_broadcast(item.get_sub_group(), v1409_data, 5))));
                float v1415_data = r1[5];
                float v1418_data = ir4[5];
                ir4[5] = (v1418_data + (v1384_data * (sycl::group_broadcast(item.get_sub_group(), v1415_data, 5))));
                float v1421_data = r1[6];
                float v1424_data = ir4[6];
                ir4[6] = (v1424_data + (v1384_data * (sycl::group_broadcast(item.get_sub_group(), v1421_data, 5))));
                float v1427_data = r1[7];
                float v1430_data = ir4[7];
                ir4[7] = (v1430_data + (v1384_data * (sycl::group_broadcast(item.get_sub_group(), v1427_data, 5))));
                float v1433_data = r1[8];
                float v1436_data = ir4[8];
                ir4[8] = (v1436_data + (v1384_data * (sycl::group_broadcast(item.get_sub_group(), v1433_data, 5))));
                float v1439_data = r1[9];
                float v1442_data = ir4[9];
                ir4[9] = (v1442_data + (v1384_data * (sycl::group_broadcast(item.get_sub_group(), v1439_data, 5))));
                float v1445_data = r1[10];
                float v1448_data = ir4[10];
                ir4[10] = (v1448_data + (v1384_data * (sycl::group_broadcast(item.get_sub_group(), v1445_data, 5))));
                float v1451_data = r1[11];
                float v1454_data = ir4[11];
                ir4[11] = (v1454_data + (v1384_data * (sycl::group_broadcast(item.get_sub_group(), v1451_data, 5))));
              }
              if (v10_lead < 6) {
                float v1460_data = r3[6];
                float v1461_data = r1[0];
                float v1464_data = ir4[0];
                ir4[0] = (v1464_data + (v1460_data * (sycl::group_broadcast(item.get_sub_group(), v1461_data, 6))));
                float v1467_data = r1[1];
                float v1470_data = ir4[1];
                ir4[1] = (v1470_data + (v1460_data * (sycl::group_broadcast(item.get_sub_group(), v1467_data, 6))));
                float v1473_data = r1[2];
                float v1476_data = ir4[2];
                ir4[2] = (v1476_data + (v1460_data * (sycl::group_broadcast(item.get_sub_group(), v1473_data, 6))));
                float v1479_data = r1[3];
                float v1482_data = ir4[3];
                ir4[3] = (v1482_data + (v1460_data * (sycl::group_broadcast(item.get_sub_group(), v1479_data, 6))));
                float v1485_data = r1[4];
                float v1488_data = ir4[4];
                ir4[4] = (v1488_data + (v1460_data * (sycl::group_broadcast(item.get_sub_group(), v1485_data, 6))));
                float v1491_data = r1[5];
                float v1494_data = ir4[5];
                ir4[5] = (v1494_data + (v1460_data * (sycl::group_broadcast(item.get_sub_group(), v1491_data, 6))));
                float v1497_data = r1[6];
                float v1500_data = ir4[6];
                ir4[6] = (v1500_data + (v1460_data * (sycl::group_broadcast(item.get_sub_group(), v1497_data, 6))));
                float v1503_data = r1[7];
                float v1506_data = ir4[7];
                ir4[7] = (v1506_data + (v1460_data * (sycl::group_broadcast(item.get_sub_group(), v1503_data, 6))));
                float v1509_data = r1[8];
                float v1512_data = ir4[8];
                ir4[8] = (v1512_data + (v1460_data * (sycl::group_broadcast(item.get_sub_group(), v1509_data, 6))));
                float v1515_data = r1[9];
                float v1518_data = ir4[9];
                ir4[9] = (v1518_data + (v1460_data * (sycl::group_broadcast(item.get_sub_group(), v1515_data, 6))));
                float v1521_data = r1[10];
                float v1524_data = ir4[10];
                ir4[10] = (v1524_data + (v1460_data * (sycl::group_broadcast(item.get_sub_group(), v1521_data, 6))));
                float v1527_data = r1[11];
                float v1530_data = ir4[11];
                ir4[11] = (v1530_data + (v1460_data * (sycl::group_broadcast(item.get_sub_group(), v1527_data, 6))));
              }
              if (v10_lead < 6) {
                float v1536_data = r3[7];
                float v1537_data = r1[0];
                float v1540_data = ir4[0];
                ir4[0] = (v1540_data + (v1536_data * (sycl::group_broadcast(item.get_sub_group(), v1537_data, 7))));
                float v1543_data = r1[1];
                float v1546_data = ir4[1];
                ir4[1] = (v1546_data + (v1536_data * (sycl::group_broadcast(item.get_sub_group(), v1543_data, 7))));
                float v1549_data = r1[2];
                float v1552_data = ir4[2];
                ir4[2] = (v1552_data + (v1536_data * (sycl::group_broadcast(item.get_sub_group(), v1549_data, 7))));
                float v1555_data = r1[3];
                float v1558_data = ir4[3];
                ir4[3] = (v1558_data + (v1536_data * (sycl::group_broadcast(item.get_sub_group(), v1555_data, 7))));
                float v1561_data = r1[4];
                float v1564_data = ir4[4];
                ir4[4] = (v1564_data + (v1536_data * (sycl::group_broadcast(item.get_sub_group(), v1561_data, 7))));
                float v1567_data = r1[5];
                float v1570_data = ir4[5];
                ir4[5] = (v1570_data + (v1536_data * (sycl::group_broadcast(item.get_sub_group(), v1567_data, 7))));
                float v1573_data = r1[6];
                float v1576_data = ir4[6];
                ir4[6] = (v1576_data + (v1536_data * (sycl::group_broadcast(item.get_sub_group(), v1573_data, 7))));
                float v1579_data = r1[7];
                float v1582_data = ir4[7];
                ir4[7] = (v1582_data + (v1536_data * (sycl::group_broadcast(item.get_sub_group(), v1579_data, 7))));
                float v1585_data = r1[8];
                float v1588_data = ir4[8];
                ir4[8] = (v1588_data + (v1536_data * (sycl::group_broadcast(item.get_sub_group(), v1585_data, 7))));
                float v1591_data = r1[9];
                float v1594_data = ir4[9];
                ir4[9] = (v1594_data + (v1536_data * (sycl::group_broadcast(item.get_sub_group(), v1591_data, 7))));
                float v1597_data = r1[10];
                float v1600_data = ir4[10];
                ir4[10] = (v1600_data + (v1536_data * (sycl::group_broadcast(item.get_sub_group(), v1597_data, 7))));
                float v1603_data = r1[11];
                float v1606_data = ir4[11];
                ir4[11] = (v1606_data + (v1536_data * (sycl::group_broadcast(item.get_sub_group(), v1603_data, 7))));
              }
              if (v10_lead < 6) {
                float v1612_data = r3[8];
                float v1613_data = r1[0];
                float v1616_data = ir4[0];
                ir4[0] = (v1616_data + (v1612_data * (sycl::group_broadcast(item.get_sub_group(), v1613_data, 8))));
                float v1619_data = r1[1];
                float v1622_data = ir4[1];
                ir4[1] = (v1622_data + (v1612_data * (sycl::group_broadcast(item.get_sub_group(), v1619_data, 8))));
                float v1625_data = r1[2];
                float v1628_data = ir4[2];
                ir4[2] = (v1628_data + (v1612_data * (sycl::group_broadcast(item.get_sub_group(), v1625_data, 8))));
                float v1631_data = r1[3];
                float v1634_data = ir4[3];
                ir4[3] = (v1634_data + (v1612_data * (sycl::group_broadcast(item.get_sub_group(), v1631_data, 8))));
                float v1637_data = r1[4];
                float v1640_data = ir4[4];
                ir4[4] = (v1640_data + (v1612_data * (sycl::group_broadcast(item.get_sub_group(), v1637_data, 8))));
                float v1643_data = r1[5];
                float v1646_data = ir4[5];
                ir4[5] = (v1646_data + (v1612_data * (sycl::group_broadcast(item.get_sub_group(), v1643_data, 8))));
                float v1649_data = r1[6];
                float v1652_data = ir4[6];
                ir4[6] = (v1652_data + (v1612_data * (sycl::group_broadcast(item.get_sub_group(), v1649_data, 8))));
                float v1655_data = r1[7];
                float v1658_data = ir4[7];
                ir4[7] = (v1658_data + (v1612_data * (sycl::group_broadcast(item.get_sub_group(), v1655_data, 8))));
                float v1661_data = r1[8];
                float v1664_data = ir4[8];
                ir4[8] = (v1664_data + (v1612_data * (sycl::group_broadcast(item.get_sub_group(), v1661_data, 8))));
                float v1667_data = r1[9];
                float v1670_data = ir4[9];
                ir4[9] = (v1670_data + (v1612_data * (sycl::group_broadcast(item.get_sub_group(), v1667_data, 8))));
                float v1673_data = r1[10];
                float v1676_data = ir4[10];
                ir4[10] = (v1676_data + (v1612_data * (sycl::group_broadcast(item.get_sub_group(), v1673_data, 8))));
                float v1679_data = r1[11];
                float v1682_data = ir4[11];
                ir4[11] = (v1682_data + (v1612_data * (sycl::group_broadcast(item.get_sub_group(), v1679_data, 8))));
              }
              if (v10_lead < 6) {
                float v1688_data = r3[9];
                float v1689_data = r1[0];
                float v1692_data = ir4[0];
                ir4[0] = (v1692_data + (v1688_data * (sycl::group_broadcast(item.get_sub_group(), v1689_data, 9))));
                float v1695_data = r1[1];
                float v1698_data = ir4[1];
                ir4[1] = (v1698_data + (v1688_data * (sycl::group_broadcast(item.get_sub_group(), v1695_data, 9))));
                float v1701_data = r1[2];
                float v1704_data = ir4[2];
                ir4[2] = (v1704_data + (v1688_data * (sycl::group_broadcast(item.get_sub_group(), v1701_data, 9))));
                float v1707_data = r1[3];
                float v1710_data = ir4[3];
                ir4[3] = (v1710_data + (v1688_data * (sycl::group_broadcast(item.get_sub_group(), v1707_data, 9))));
                float v1713_data = r1[4];
                float v1716_data = ir4[4];
                ir4[4] = (v1716_data + (v1688_data * (sycl::group_broadcast(item.get_sub_group(), v1713_data, 9))));
                float v1719_data = r1[5];
                float v1722_data = ir4[5];
                ir4[5] = (v1722_data + (v1688_data * (sycl::group_broadcast(item.get_sub_group(), v1719_data, 9))));
                float v1725_data = r1[6];
                float v1728_data = ir4[6];
                ir4[6] = (v1728_data + (v1688_data * (sycl::group_broadcast(item.get_sub_group(), v1725_data, 9))));
                float v1731_data = r1[7];
                float v1734_data = ir4[7];
                ir4[7] = (v1734_data + (v1688_data * (sycl::group_broadcast(item.get_sub_group(), v1731_data, 9))));
                float v1737_data = r1[8];
                float v1740_data = ir4[8];
                ir4[8] = (v1740_data + (v1688_data * (sycl::group_broadcast(item.get_sub_group(), v1737_data, 9))));
                float v1743_data = r1[9];
                float v1746_data = ir4[9];
                ir4[9] = (v1746_data + (v1688_data * (sycl::group_broadcast(item.get_sub_group(), v1743_data, 9))));
                float v1749_data = r1[10];
                float v1752_data = ir4[10];
                ir4[10] = (v1752_data + (v1688_data * (sycl::group_broadcast(item.get_sub_group(), v1749_data, 9))));
                float v1755_data = r1[11];
                float v1758_data = ir4[11];
                ir4[11] = (v1758_data + (v1688_data * (sycl::group_broadcast(item.get_sub_group(), v1755_data, 9))));
              }
              if (v10_lead < 6) {
                float v1764_data = r3[10];
                float v1765_data = r1[0];
                float v1768_data = ir4[0];
                ir4[0] = (v1768_data + (v1764_data * (sycl::group_broadcast(item.get_sub_group(), v1765_data, 10))));
                float v1771_data = r1[1];
                float v1774_data = ir4[1];
                ir4[1] = (v1774_data + (v1764_data * (sycl::group_broadcast(item.get_sub_group(), v1771_data, 10))));
                float v1777_data = r1[2];
                float v1780_data = ir4[2];
                ir4[2] = (v1780_data + (v1764_data * (sycl::group_broadcast(item.get_sub_group(), v1777_data, 10))));
                float v1783_data = r1[3];
                float v1786_data = ir4[3];
                ir4[3] = (v1786_data + (v1764_data * (sycl::group_broadcast(item.get_sub_group(), v1783_data, 10))));
                float v1789_data = r1[4];
                float v1792_data = ir4[4];
                ir4[4] = (v1792_data + (v1764_data * (sycl::group_broadcast(item.get_sub_group(), v1789_data, 10))));
                float v1795_data = r1[5];
                float v1798_data = ir4[5];
                ir4[5] = (v1798_data + (v1764_data * (sycl::group_broadcast(item.get_sub_group(), v1795_data, 10))));
                float v1801_data = r1[6];
                float v1804_data = ir4[6];
                ir4[6] = (v1804_data + (v1764_data * (sycl::group_broadcast(item.get_sub_group(), v1801_data, 10))));
                float v1807_data = r1[7];
                float v1810_data = ir4[7];
                ir4[7] = (v1810_data + (v1764_data * (sycl::group_broadcast(item.get_sub_group(), v1807_data, 10))));
                float v1813_data = r1[8];
                float v1816_data = ir4[8];
                ir4[8] = (v1816_data + (v1764_data * (sycl::group_broadcast(item.get_sub_group(), v1813_data, 10))));
                float v1819_data = r1[9];
                float v1822_data = ir4[9];
                ir4[9] = (v1822_data + (v1764_data * (sycl::group_broadcast(item.get_sub_group(), v1819_data, 10))));
                float v1825_data = r1[10];
                float v1828_data = ir4[10];
                ir4[10] = (v1828_data + (v1764_data * (sycl::group_broadcast(item.get_sub_group(), v1825_data, 10))));
                float v1831_data = r1[11];
                float v1834_data = ir4[11];
                ir4[11] = (v1834_data + (v1764_data * (sycl::group_broadcast(item.get_sub_group(), v1831_data, 10))));
              }
              if (v10_lead < 6) {
                float v1840_data = r3[11];
                float v1841_data = r1[0];
                float v1844_data = ir4[0];
                ir4[0] = (v1844_data + (v1840_data * (sycl::group_broadcast(item.get_sub_group(), v1841_data, 11))));
                float v1847_data = r1[1];
                float v1850_data = ir4[1];
                ir4[1] = (v1850_data + (v1840_data * (sycl::group_broadcast(item.get_sub_group(), v1847_data, 11))));
                float v1853_data = r1[2];
                float v1856_data = ir4[2];
                ir4[2] = (v1856_data + (v1840_data * (sycl::group_broadcast(item.get_sub_group(), v1853_data, 11))));
                float v1859_data = r1[3];
                float v1862_data = ir4[3];
                ir4[3] = (v1862_data + (v1840_data * (sycl::group_broadcast(item.get_sub_group(), v1859_data, 11))));
                float v1865_data = r1[4];
                float v1868_data = ir4[4];
                ir4[4] = (v1868_data + (v1840_data * (sycl::group_broadcast(item.get_sub_group(), v1865_data, 11))));
                float v1871_data = r1[5];
                float v1874_data = ir4[5];
                ir4[5] = (v1874_data + (v1840_data * (sycl::group_broadcast(item.get_sub_group(), v1871_data, 11))));
                float v1877_data = r1[6];
                float v1880_data = ir4[6];
                ir4[6] = (v1880_data + (v1840_data * (sycl::group_broadcast(item.get_sub_group(), v1877_data, 11))));
                float v1883_data = r1[7];
                float v1886_data = ir4[7];
                ir4[7] = (v1886_data + (v1840_data * (sycl::group_broadcast(item.get_sub_group(), v1883_data, 11))));
                float v1889_data = r1[8];
                float v1892_data = ir4[8];
                ir4[8] = (v1892_data + (v1840_data * (sycl::group_broadcast(item.get_sub_group(), v1889_data, 11))));
                float v1895_data = r1[9];
                float v1898_data = ir4[9];
                ir4[9] = (v1898_data + (v1840_data * (sycl::group_broadcast(item.get_sub_group(), v1895_data, 11))));
                float v1901_data = r1[10];
                float v1904_data = ir4[10];
                ir4[10] = (v1904_data + (v1840_data * (sycl::group_broadcast(item.get_sub_group(), v1901_data, 11))));
                float v1907_data = r1[11];
                float v1910_data = ir4[11];
                ir4[11] = (v1910_data + (v1840_data * (sycl::group_broadcast(item.get_sub_group(), v1907_data, 11))));
              }
              if (v10_lead < 6) {
                #pragma unroll
                for (int32_t v1916_n1 = 0; v1916_n1 < 12; ++v1916_n1) {
                  float v1918_data = ir4[v1916_n1];
                  r4[v1916_n1] = v1918_data;
                }
              }
              // s0 = store{r>s}(localShrMem0, r4);
              if (v10_lead < 6) {
                int32_t v1932_off = v10_lead + 6;
                #pragma unroll
                for (int32_t v1924_i1 = 0; v1924_i1 < 12; ++v1924_i1) {
                  float v1926_data = r4[v1924_i1];
                  int32_t v1934_a = v1932_off + (v1924_i1 * 12);
                  s0[(v1934_a ^ ((v1934_a >> 4) & 15))] = v1926_data;
                }
              }
              // wait(r5 = load{g>r}(glb_m4););
              float r6[12]{};
              sycl::group_barrier(item.get_sub_group());
              // r6 = +(r5 * s0) + None
              // [(0, 12), (0, 12)] [(0, 12)]
              float ir6[12]{};
              if (v10_lead < 12) {
                float v1944_data = r5[0];
                float v1945_data = s0[0];
                float v1947_data = ir6[0];
                ir6[0] = (v1947_data + (v1944_data * v1945_data));
                float v1950_data = s0[12];
                float v1952_data = ir6[1];
                ir6[1] = (v1952_data + (v1944_data * v1950_data));
                float v1955_data = s0[25];
                float v1957_data = ir6[2];
                ir6[2] = (v1957_data + (v1944_data * v1955_data));
                float v1960_data = s0[38];
                float v1962_data = ir6[3];
                ir6[3] = (v1962_data + (v1944_data * v1960_data));
                float v1965_data = s0[51];
                float v1967_data = ir6[4];
                ir6[4] = (v1967_data + (v1944_data * v1965_data));
                float v1970_data = s0[63];
                float v1972_data = ir6[5];
                ir6[5] = (v1972_data + (v1944_data * v1970_data));
                float v1975_data = s0[76];
                float v1977_data = ir6[6];
                ir6[6] = (v1977_data + (v1944_data * v1975_data));
                float v1980_data = s0[81];
                float v1982_data = ir6[7];
                ir6[7] = (v1982_data + (v1944_data * v1980_data));
                float v1985_data = s0[102];
                float v1987_data = ir6[8];
                ir6[8] = (v1987_data + (v1944_data * v1985_data));
                float v1990_data = s0[106];
                float v1992_data = ir6[9];
                ir6[9] = (v1992_data + (v1944_data * v1990_data));
                float v1995_data = s0[127];
                float v1997_data = ir6[10];
                ir6[10] = (v1997_data + (v1944_data * v1995_data));
                float v2000_data = s0[140];
                float v2002_data = ir6[11];
                ir6[11] = (v2002_data + (v1944_data * v2000_data));
              }
              if (v10_lead < 12) {
                float v2008_data = r5[1];
                float v2009_data = s0[1];
                float v2011_data = ir6[0];
                ir6[0] = (v2011_data + (v2008_data * v2009_data));
                float v2014_data = s0[13];
                float v2016_data = ir6[1];
                ir6[1] = (v2016_data + (v2008_data * v2014_data));
                float v2019_data = s0[24];
                float v2021_data = ir6[2];
                ir6[2] = (v2021_data + (v2008_data * v2019_data));
                float v2024_data = s0[39];
                float v2026_data = ir6[3];
                ir6[3] = (v2026_data + (v2008_data * v2024_data));
                float v2029_data = s0[50];
                float v2031_data = ir6[4];
                ir6[4] = (v2031_data + (v2008_data * v2029_data));
                float v2034_data = s0[62];
                float v2036_data = ir6[5];
                ir6[5] = (v2036_data + (v2008_data * v2034_data));
                float v2039_data = s0[77];
                float v2041_data = ir6[6];
                ir6[6] = (v2041_data + (v2008_data * v2039_data));
                float v2044_data = s0[80];
                float v2046_data = ir6[7];
                ir6[7] = (v2046_data + (v2008_data * v2044_data));
                float v2049_data = s0[103];
                float v2051_data = ir6[8];
                ir6[8] = (v2051_data + (v2008_data * v2049_data));
                float v2054_data = s0[107];
                float v2056_data = ir6[9];
                ir6[9] = (v2056_data + (v2008_data * v2054_data));
                float v2059_data = s0[126];
                float v2061_data = ir6[10];
                ir6[10] = (v2061_data + (v2008_data * v2059_data));
                float v2064_data = s0[141];
                float v2066_data = ir6[11];
                ir6[11] = (v2066_data + (v2008_data * v2064_data));
              }
              if (v10_lead < 12) {
                float v2072_data = r5[2];
                float v2073_data = s0[2];
                float v2075_data = ir6[0];
                ir6[0] = (v2075_data + (v2072_data * v2073_data));
                float v2078_data = s0[14];
                float v2080_data = ir6[1];
                ir6[1] = (v2080_data + (v2072_data * v2078_data));
                float v2083_data = s0[27];
                float v2085_data = ir6[2];
                ir6[2] = (v2085_data + (v2072_data * v2083_data));
                float v2088_data = s0[36];
                float v2090_data = ir6[3];
                ir6[3] = (v2090_data + (v2072_data * v2088_data));
                float v2093_data = s0[49];
                float v2095_data = ir6[4];
                ir6[4] = (v2095_data + (v2072_data * v2093_data));
                float v2098_data = s0[61];
                float v2100_data = ir6[5];
                ir6[5] = (v2100_data + (v2072_data * v2098_data));
                float v2103_data = s0[78];
                float v2105_data = ir6[6];
                ir6[6] = (v2105_data + (v2072_data * v2103_data));
                float v2108_data = s0[83];
                float v2110_data = ir6[7];
                ir6[7] = (v2110_data + (v2072_data * v2108_data));
                float v2113_data = s0[100];
                float v2115_data = ir6[8];
                ir6[8] = (v2115_data + (v2072_data * v2113_data));
                float v2118_data = s0[104];
                float v2120_data = ir6[9];
                ir6[9] = (v2120_data + (v2072_data * v2118_data));
                float v2123_data = s0[125];
                float v2125_data = ir6[10];
                ir6[10] = (v2125_data + (v2072_data * v2123_data));
                float v2128_data = s0[142];
                float v2130_data = ir6[11];
                ir6[11] = (v2130_data + (v2072_data * v2128_data));
              }
              if (v10_lead < 12) {
                float v2136_data = r5[3];
                float v2137_data = s0[3];
                float v2139_data = ir6[0];
                ir6[0] = (v2139_data + (v2136_data * v2137_data));
                float v2142_data = s0[15];
                float v2144_data = ir6[1];
                ir6[1] = (v2144_data + (v2136_data * v2142_data));
                float v2147_data = s0[26];
                float v2149_data = ir6[2];
                ir6[2] = (v2149_data + (v2136_data * v2147_data));
                float v2152_data = s0[37];
                float v2154_data = ir6[3];
                ir6[3] = (v2154_data + (v2136_data * v2152_data));
                float v2157_data = s0[48];
                float v2159_data = ir6[4];
                ir6[4] = (v2159_data + (v2136_data * v2157_data));
                float v2162_data = s0[60];
                float v2164_data = ir6[5];
                ir6[5] = (v2164_data + (v2136_data * v2162_data));
                float v2167_data = s0[79];
                float v2169_data = ir6[6];
                ir6[6] = (v2169_data + (v2136_data * v2167_data));
                float v2172_data = s0[82];
                float v2174_data = ir6[7];
                ir6[7] = (v2174_data + (v2136_data * v2172_data));
                float v2177_data = s0[101];
                float v2179_data = ir6[8];
                ir6[8] = (v2179_data + (v2136_data * v2177_data));
                float v2182_data = s0[105];
                float v2184_data = ir6[9];
                ir6[9] = (v2184_data + (v2136_data * v2182_data));
                float v2187_data = s0[124];
                float v2189_data = ir6[10];
                ir6[10] = (v2189_data + (v2136_data * v2187_data));
                float v2192_data = s0[143];
                float v2194_data = ir6[11];
                ir6[11] = (v2194_data + (v2136_data * v2192_data));
              }
              if (v10_lead < 12) {
                float v2200_data = r5[4];
                float v2201_data = s0[4];
                float v2203_data = ir6[0];
                ir6[0] = (v2203_data + (v2200_data * v2201_data));
                float v2206_data = s0[17];
                float v2208_data = ir6[1];
                ir6[1] = (v2208_data + (v2200_data * v2206_data));
                float v2211_data = s0[29];
                float v2213_data = ir6[2];
                ir6[2] = (v2213_data + (v2200_data * v2211_data));
                float v2216_data = s0[42];
                float v2218_data = ir6[3];
                ir6[3] = (v2218_data + (v2200_data * v2216_data));
                float v2221_data = s0[55];
                float v2223_data = ir6[4];
                ir6[4] = (v2223_data + (v2200_data * v2221_data));
                float v2226_data = s0[68];
                float v2228_data = ir6[5];
                ir6[5] = (v2228_data + (v2200_data * v2226_data));
                float v2231_data = s0[72];
                float v2233_data = ir6[6];
                ir6[6] = (v2233_data + (v2200_data * v2231_data));
                float v2236_data = s0[93];
                float v2238_data = ir6[7];
                ir6[7] = (v2238_data + (v2200_data * v2236_data));
                float v2241_data = s0[98];
                float v2243_data = ir6[8];
                ir6[8] = (v2243_data + (v2200_data * v2241_data));
                float v2246_data = s0[119];
                float v2248_data = ir6[9];
                ir6[9] = (v2248_data + (v2200_data * v2246_data));
                float v2251_data = s0[123];
                float v2253_data = ir6[10];
                ir6[10] = (v2253_data + (v2200_data * v2251_data));
                float v2256_data = s0[128];
                float v2258_data = ir6[11];
                ir6[11] = (v2258_data + (v2200_data * v2256_data));
              }
              if (v10_lead < 12) {
                float v2264_data = r5[5];
                float v2265_data = s0[5];
                float v2267_data = ir6[0];
                ir6[0] = (v2267_data + (v2264_data * v2265_data));
                float v2270_data = s0[16];
                float v2272_data = ir6[1];
                ir6[1] = (v2272_data + (v2264_data * v2270_data));
                float v2275_data = s0[28];
                float v2277_data = ir6[2];
                ir6[2] = (v2277_data + (v2264_data * v2275_data));
                float v2280_data = s0[43];
                float v2282_data = ir6[3];
                ir6[3] = (v2282_data + (v2264_data * v2280_data));
                float v2285_data = s0[54];
                float v2287_data = ir6[4];
                ir6[4] = (v2287_data + (v2264_data * v2285_data));
                float v2290_data = s0[69];
                float v2292_data = ir6[5];
                ir6[5] = (v2292_data + (v2264_data * v2290_data));
                float v2295_data = s0[73];
                float v2297_data = ir6[6];
                ir6[6] = (v2297_data + (v2264_data * v2295_data));
                float v2300_data = s0[92];
                float v2302_data = ir6[7];
                ir6[7] = (v2302_data + (v2264_data * v2300_data));
                float v2305_data = s0[99];
                float v2307_data = ir6[8];
                ir6[8] = (v2307_data + (v2264_data * v2305_data));
                float v2310_data = s0[118];
                float v2312_data = ir6[9];
                ir6[9] = (v2312_data + (v2264_data * v2310_data));
                float v2315_data = s0[122];
                float v2317_data = ir6[10];
                ir6[10] = (v2317_data + (v2264_data * v2315_data));
                float v2320_data = s0[129];
                float v2322_data = ir6[11];
                ir6[11] = (v2322_data + (v2264_data * v2320_data));
              }
              if (v10_lead < 12) {
                float v2328_data = r5[6];
                float v2329_data = s0[6];
                float v2331_data = ir6[0];
                ir6[0] = (v2331_data + (v2328_data * v2329_data));
                float v2334_data = s0[19];
                float v2336_data = ir6[1];
                ir6[1] = (v2336_data + (v2328_data * v2334_data));
                float v2339_data = s0[31];
                float v2341_data = ir6[2];
                ir6[2] = (v2341_data + (v2328_data * v2339_data));
                float v2344_data = s0[40];
                float v2346_data = ir6[3];
                ir6[3] = (v2346_data + (v2328_data * v2344_data));
                float v2349_data = s0[53];
                float v2351_data = ir6[4];
                ir6[4] = (v2351_data + (v2328_data * v2349_data));
                float v2354_data = s0[70];
                float v2356_data = ir6[5];
                ir6[5] = (v2356_data + (v2328_data * v2354_data));
                float v2359_data = s0[74];
                float v2361_data = ir6[6];
                ir6[6] = (v2361_data + (v2328_data * v2359_data));
                float v2364_data = s0[95];
                float v2366_data = ir6[7];
                ir6[7] = (v2366_data + (v2328_data * v2364_data));
                float v2369_data = s0[96];
                float v2371_data = ir6[8];
                ir6[8] = (v2371_data + (v2328_data * v2369_data));
                float v2374_data = s0[117];
                float v2376_data = ir6[9];
                ir6[9] = (v2376_data + (v2328_data * v2374_data));
                float v2379_data = s0[121];
                float v2381_data = ir6[10];
                ir6[10] = (v2381_data + (v2328_data * v2379_data));
                float v2384_data = s0[130];
                float v2386_data = ir6[11];
                ir6[11] = (v2386_data + (v2328_data * v2384_data));
              }
              if (v10_lead < 12) {
                float v2392_data = r5[7];
                float v2393_data = s0[7];
                float v2395_data = ir6[0];
                ir6[0] = (v2395_data + (v2392_data * v2393_data));
                float v2398_data = s0[18];
                float v2400_data = ir6[1];
                ir6[1] = (v2400_data + (v2392_data * v2398_data));
                float v2403_data = s0[30];
                float v2405_data = ir6[2];
                ir6[2] = (v2405_data + (v2392_data * v2403_data));
                float v2408_data = s0[41];
                float v2410_data = ir6[3];
                ir6[3] = (v2410_data + (v2392_data * v2408_data));
                float v2413_data = s0[52];
                float v2415_data = ir6[4];
                ir6[4] = (v2415_data + (v2392_data * v2413_data));
                float v2418_data = s0[71];
                float v2420_data = ir6[5];
                ir6[5] = (v2420_data + (v2392_data * v2418_data));
                float v2423_data = s0[75];
                float v2425_data = ir6[6];
                ir6[6] = (v2425_data + (v2392_data * v2423_data));
                float v2428_data = s0[94];
                float v2430_data = ir6[7];
                ir6[7] = (v2430_data + (v2392_data * v2428_data));
                float v2433_data = s0[97];
                float v2435_data = ir6[8];
                ir6[8] = (v2435_data + (v2392_data * v2433_data));
                float v2438_data = s0[116];
                float v2440_data = ir6[9];
                ir6[9] = (v2440_data + (v2392_data * v2438_data));
                float v2443_data = s0[120];
                float v2445_data = ir6[10];
                ir6[10] = (v2445_data + (v2392_data * v2443_data));
                float v2448_data = s0[131];
                float v2450_data = ir6[11];
                ir6[11] = (v2450_data + (v2392_data * v2448_data));
              }
              if (v10_lead < 12) {
                float v2456_data = r5[8];
                float v2457_data = s0[8];
                float v2459_data = ir6[0];
                ir6[0] = (v2459_data + (v2456_data * v2457_data));
                float v2462_data = s0[21];
                float v2464_data = ir6[1];
                ir6[1] = (v2464_data + (v2456_data * v2462_data));
                float v2467_data = s0[34];
                float v2469_data = ir6[2];
                ir6[2] = (v2469_data + (v2456_data * v2467_data));
                float v2472_data = s0[46];
                float v2474_data = ir6[3];
                ir6[3] = (v2474_data + (v2456_data * v2472_data));
                float v2477_data = s0[59];
                float v2479_data = ir6[4];
                ir6[4] = (v2479_data + (v2456_data * v2477_data));
                float v2482_data = s0[64];
                float v2484_data = ir6[5];
                ir6[5] = (v2484_data + (v2456_data * v2482_data));
                float v2487_data = s0[85];
                float v2489_data = ir6[6];
                ir6[6] = (v2489_data + (v2456_data * v2487_data));
                float v2492_data = s0[89];
                float v2494_data = ir6[7];
                ir6[7] = (v2494_data + (v2456_data * v2492_data));
                float v2497_data = s0[110];
                float v2499_data = ir6[8];
                ir6[8] = (v2499_data + (v2456_data * v2497_data));
                float v2502_data = s0[115];
                float v2504_data = ir6[9];
                ir6[9] = (v2504_data + (v2456_data * v2502_data));
                float v2507_data = s0[136];
                float v2509_data = ir6[10];
                ir6[10] = (v2509_data + (v2456_data * v2507_data));
                float v2512_data = s0[132];
                float v2514_data = ir6[11];
                ir6[11] = (v2514_data + (v2456_data * v2512_data));
              }
              if (v10_lead < 12) {
                float v2520_data = r5[9];
                float v2521_data = s0[9];
                float v2523_data = ir6[0];
                ir6[0] = (v2523_data + (v2520_data * v2521_data));
                float v2526_data = s0[20];
                float v2528_data = ir6[1];
                ir6[1] = (v2528_data + (v2520_data * v2526_data));
                float v2531_data = s0[35];
                float v2533_data = ir6[2];
                ir6[2] = (v2533_data + (v2520_data * v2531_data));
                float v2536_data = s0[47];
                float v2538_data = ir6[3];
                ir6[3] = (v2538_data + (v2520_data * v2536_data));
                float v2541_data = s0[58];
                float v2543_data = ir6[4];
                ir6[4] = (v2543_data + (v2520_data * v2541_data));
                float v2546_data = s0[65];
                float v2548_data = ir6[5];
                ir6[5] = (v2548_data + (v2520_data * v2546_data));
                float v2551_data = s0[84];
                float v2553_data = ir6[6];
                ir6[6] = (v2553_data + (v2520_data * v2551_data));
                float v2556_data = s0[88];
                float v2558_data = ir6[7];
                ir6[7] = (v2558_data + (v2520_data * v2556_data));
                float v2561_data = s0[111];
                float v2563_data = ir6[8];
                ir6[8] = (v2563_data + (v2520_data * v2561_data));
                float v2566_data = s0[114];
                float v2568_data = ir6[9];
                ir6[9] = (v2568_data + (v2520_data * v2566_data));
                float v2571_data = s0[137];
                float v2573_data = ir6[10];
                ir6[10] = (v2573_data + (v2520_data * v2571_data));
                float v2576_data = s0[133];
                float v2578_data = ir6[11];
                ir6[11] = (v2578_data + (v2520_data * v2576_data));
              }
              if (v10_lead < 12) {
                float v2584_data = r5[10];
                float v2585_data = s0[10];
                float v2587_data = ir6[0];
                ir6[0] = (v2587_data + (v2584_data * v2585_data));
                float v2590_data = s0[23];
                float v2592_data = ir6[1];
                ir6[1] = (v2592_data + (v2584_data * v2590_data));
                float v2595_data = s0[32];
                float v2597_data = ir6[2];
                ir6[2] = (v2597_data + (v2584_data * v2595_data));
                float v2600_data = s0[44];
                float v2602_data = ir6[3];
                ir6[3] = (v2602_data + (v2584_data * v2600_data));
                float v2605_data = s0[57];
                float v2607_data = ir6[4];
                ir6[4] = (v2607_data + (v2584_data * v2605_data));
                float v2610_data = s0[66];
                float v2612_data = ir6[5];
                ir6[5] = (v2612_data + (v2584_data * v2610_data));
                float v2615_data = s0[87];
                float v2617_data = ir6[6];
                ir6[6] = (v2617_data + (v2584_data * v2615_data));
                float v2620_data = s0[91];
                float v2622_data = ir6[7];
                ir6[7] = (v2622_data + (v2584_data * v2620_data));
                float v2625_data = s0[108];
                float v2627_data = ir6[8];
                ir6[8] = (v2627_data + (v2584_data * v2625_data));
                float v2630_data = s0[113];
                float v2632_data = ir6[9];
                ir6[9] = (v2632_data + (v2584_data * v2630_data));
                float v2635_data = s0[138];
                float v2637_data = ir6[10];
                ir6[10] = (v2637_data + (v2584_data * v2635_data));
                float v2640_data = s0[134];
                float v2642_data = ir6[11];
                ir6[11] = (v2642_data + (v2584_data * v2640_data));
              }
              if (v10_lead < 12) {
                float v2648_data = r5[11];
                float v2649_data = s0[11];
                float v2651_data = ir6[0];
                ir6[0] = (v2651_data + (v2648_data * v2649_data));
                float v2654_data = s0[22];
                float v2656_data = ir6[1];
                ir6[1] = (v2656_data + (v2648_data * v2654_data));
                float v2659_data = s0[33];
                float v2661_data = ir6[2];
                ir6[2] = (v2661_data + (v2648_data * v2659_data));
                float v2664_data = s0[45];
                float v2666_data = ir6[3];
                ir6[3] = (v2666_data + (v2648_data * v2664_data));
                float v2669_data = s0[56];
                float v2671_data = ir6[4];
                ir6[4] = (v2671_data + (v2648_data * v2669_data));
                float v2674_data = s0[67];
                float v2676_data = ir6[5];
                ir6[5] = (v2676_data + (v2648_data * v2674_data));
                float v2679_data = s0[86];
                float v2681_data = ir6[6];
                ir6[6] = (v2681_data + (v2648_data * v2679_data));
                float v2684_data = s0[90];
                float v2686_data = ir6[7];
                ir6[7] = (v2686_data + (v2648_data * v2684_data));
                float v2689_data = s0[109];
                float v2691_data = ir6[8];
                ir6[8] = (v2691_data + (v2648_data * v2689_data));
                float v2694_data = s0[112];
                float v2696_data = ir6[9];
                ir6[9] = (v2696_data + (v2648_data * v2694_data));
                float v2699_data = s0[139];
                float v2701_data = ir6[10];
                ir6[10] = (v2701_data + (v2648_data * v2699_data));
                float v2704_data = s0[135];
                float v2706_data = ir6[11];
                ir6[11] = (v2706_data + (v2648_data * v2704_data));
              }
              if (v10_lead < 12) {
                #pragma unroll
                for (int32_t v2712_n1 = 0; v2712_n1 < 12; ++v2712_n1) {
                  float v2714_data = ir6[v2712_n1];
                  r6[v2712_n1] = v2714_data;
                }
              }
              // glb_m3 = store{r>g}(r6);
              if (v10_lead < 12) {
                #pragma unroll
                for (int32_t v2720_i1 = 0; v2720_i1 < 12; ++v2720_i1) {
                  float v2722_data = r6[v2720_i1];
                  glb_m3[(v10_lead + (v2720_i1 * 12))] = v2722_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

