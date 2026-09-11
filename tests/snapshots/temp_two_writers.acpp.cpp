// === base name ===
kernel_1a1c5b41a34696e6

// === header ===
void launcher_kernel_1a1c5b41a34696e6(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_1a1c5b41a34696e6(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_1a1c5b41a34696e6(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_1a1c5b41a34696e6(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (2560, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
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
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[160 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[144];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 72 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v3_batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v3_batchId0 * 144 + 0 + m4_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v19_lead = item.get_local_id(0) % 16;
              if (v19_lead < 6) {
                #pragma unroll
                for (int32_t v21_i1 = 0; v21_i1 < 12; ++v21_i1) {
                  float v29_data = glb_m0[(v19_lead + (v21_i1 * 6))];
                  r0[v21_i1] = v29_data;
                }
              }
              float r1[12]{};
              // r1 = load{g>r}(glb_m1);
              if (v19_lead < 12) {
                #pragma unroll
                for (int32_t v36_i1 = 0; v36_i1 < 12; ++v36_i1) {
                  float v44_data = glb_m1[(v19_lead + (v36_i1 * 12))];
                  r1[v36_i1] = v44_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[12]{};
              // r3 = load{g>r}(glb_m2);
              if (v19_lead < 6) {
                #pragma unroll
                for (int32_t v51_i1 = 0; v51_i1 < 12; ++v51_i1) {
                  float v59_data = glb_m2[(v19_lead + (v51_i1 * 6))];
                  r3[v51_i1] = v59_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[12]{};
              // r2 = +(r0 * r1) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              if (v19_lead < 6) {
                float v66_data = r0[0];
                float v67_data = r1[0];
                float v70_data = r2[0];
                r2[0] = (v70_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 0))));
                float v73_data = r1[1];
                float v76_data = r2[1];
                r2[1] = (v76_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 0))));
                float v79_data = r1[2];
                float v82_data = r2[2];
                r2[2] = (v82_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 0))));
                float v85_data = r1[3];
                float v88_data = r2[3];
                r2[3] = (v88_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 0))));
                float v91_data = r1[4];
                float v94_data = r2[4];
                r2[4] = (v94_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 0))));
                float v97_data = r1[5];
                float v100_data = r2[5];
                r2[5] = (v100_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 0))));
                float v103_data = r1[6];
                float v106_data = r2[6];
                r2[6] = (v106_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 0))));
                float v109_data = r1[7];
                float v112_data = r2[7];
                r2[7] = (v112_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 0))));
                float v115_data = r1[8];
                float v118_data = r2[8];
                r2[8] = (v118_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v115_data, 0))));
                float v121_data = r1[9];
                float v124_data = r2[9];
                r2[9] = (v124_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v121_data, 0))));
                float v127_data = r1[10];
                float v130_data = r2[10];
                r2[10] = (v130_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 0))));
                float v133_data = r1[11];
                float v136_data = r2[11];
                r2[11] = (v136_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 0))));
              }
              if (v19_lead < 6) {
                float v142_data = r0[1];
                float v143_data = r1[0];
                float v146_data = r2[0];
                r2[0] = (v146_data + (v142_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 1))));
                float v149_data = r1[1];
                float v152_data = r2[1];
                r2[1] = (v152_data + (v142_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 1))));
                float v155_data = r1[2];
                float v158_data = r2[2];
                r2[2] = (v158_data + (v142_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 1))));
                float v161_data = r1[3];
                float v164_data = r2[3];
                r2[3] = (v164_data + (v142_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 1))));
                float v167_data = r1[4];
                float v170_data = r2[4];
                r2[4] = (v170_data + (v142_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 1))));
                float v173_data = r1[5];
                float v176_data = r2[5];
                r2[5] = (v176_data + (v142_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 1))));
                float v179_data = r1[6];
                float v182_data = r2[6];
                r2[6] = (v182_data + (v142_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 1))));
                float v185_data = r1[7];
                float v188_data = r2[7];
                r2[7] = (v188_data + (v142_data * (sycl::group_broadcast(item.get_sub_group(), v185_data, 1))));
                float v191_data = r1[8];
                float v194_data = r2[8];
                r2[8] = (v194_data + (v142_data * (sycl::group_broadcast(item.get_sub_group(), v191_data, 1))));
                float v197_data = r1[9];
                float v200_data = r2[9];
                r2[9] = (v200_data + (v142_data * (sycl::group_broadcast(item.get_sub_group(), v197_data, 1))));
                float v203_data = r1[10];
                float v206_data = r2[10];
                r2[10] = (v206_data + (v142_data * (sycl::group_broadcast(item.get_sub_group(), v203_data, 1))));
                float v209_data = r1[11];
                float v212_data = r2[11];
                r2[11] = (v212_data + (v142_data * (sycl::group_broadcast(item.get_sub_group(), v209_data, 1))));
              }
              if (v19_lead < 6) {
                float v218_data = r0[2];
                float v219_data = r1[0];
                float v222_data = r2[0];
                r2[0] = (v222_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v219_data, 2))));
                float v225_data = r1[1];
                float v228_data = r2[1];
                r2[1] = (v228_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v225_data, 2))));
                float v231_data = r1[2];
                float v234_data = r2[2];
                r2[2] = (v234_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v231_data, 2))));
                float v237_data = r1[3];
                float v240_data = r2[3];
                r2[3] = (v240_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v237_data, 2))));
                float v243_data = r1[4];
                float v246_data = r2[4];
                r2[4] = (v246_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v243_data, 2))));
                float v249_data = r1[5];
                float v252_data = r2[5];
                r2[5] = (v252_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v249_data, 2))));
                float v255_data = r1[6];
                float v258_data = r2[6];
                r2[6] = (v258_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v255_data, 2))));
                float v261_data = r1[7];
                float v264_data = r2[7];
                r2[7] = (v264_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v261_data, 2))));
                float v267_data = r1[8];
                float v270_data = r2[8];
                r2[8] = (v270_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v267_data, 2))));
                float v273_data = r1[9];
                float v276_data = r2[9];
                r2[9] = (v276_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v273_data, 2))));
                float v279_data = r1[10];
                float v282_data = r2[10];
                r2[10] = (v282_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v279_data, 2))));
                float v285_data = r1[11];
                float v288_data = r2[11];
                r2[11] = (v288_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v285_data, 2))));
              }
              if (v19_lead < 6) {
                float v294_data = r0[3];
                float v295_data = r1[0];
                float v298_data = r2[0];
                r2[0] = (v298_data + (v294_data * (sycl::group_broadcast(item.get_sub_group(), v295_data, 3))));
                float v301_data = r1[1];
                float v304_data = r2[1];
                r2[1] = (v304_data + (v294_data * (sycl::group_broadcast(item.get_sub_group(), v301_data, 3))));
                float v307_data = r1[2];
                float v310_data = r2[2];
                r2[2] = (v310_data + (v294_data * (sycl::group_broadcast(item.get_sub_group(), v307_data, 3))));
                float v313_data = r1[3];
                float v316_data = r2[3];
                r2[3] = (v316_data + (v294_data * (sycl::group_broadcast(item.get_sub_group(), v313_data, 3))));
                float v319_data = r1[4];
                float v322_data = r2[4];
                r2[4] = (v322_data + (v294_data * (sycl::group_broadcast(item.get_sub_group(), v319_data, 3))));
                float v325_data = r1[5];
                float v328_data = r2[5];
                r2[5] = (v328_data + (v294_data * (sycl::group_broadcast(item.get_sub_group(), v325_data, 3))));
                float v331_data = r1[6];
                float v334_data = r2[6];
                r2[6] = (v334_data + (v294_data * (sycl::group_broadcast(item.get_sub_group(), v331_data, 3))));
                float v337_data = r1[7];
                float v340_data = r2[7];
                r2[7] = (v340_data + (v294_data * (sycl::group_broadcast(item.get_sub_group(), v337_data, 3))));
                float v343_data = r1[8];
                float v346_data = r2[8];
                r2[8] = (v346_data + (v294_data * (sycl::group_broadcast(item.get_sub_group(), v343_data, 3))));
                float v349_data = r1[9];
                float v352_data = r2[9];
                r2[9] = (v352_data + (v294_data * (sycl::group_broadcast(item.get_sub_group(), v349_data, 3))));
                float v355_data = r1[10];
                float v358_data = r2[10];
                r2[10] = (v358_data + (v294_data * (sycl::group_broadcast(item.get_sub_group(), v355_data, 3))));
                float v361_data = r1[11];
                float v364_data = r2[11];
                r2[11] = (v364_data + (v294_data * (sycl::group_broadcast(item.get_sub_group(), v361_data, 3))));
              }
              if (v19_lead < 6) {
                float v370_data = r0[4];
                float v371_data = r1[0];
                float v374_data = r2[0];
                r2[0] = (v374_data + (v370_data * (sycl::group_broadcast(item.get_sub_group(), v371_data, 4))));
                float v377_data = r1[1];
                float v380_data = r2[1];
                r2[1] = (v380_data + (v370_data * (sycl::group_broadcast(item.get_sub_group(), v377_data, 4))));
                float v383_data = r1[2];
                float v386_data = r2[2];
                r2[2] = (v386_data + (v370_data * (sycl::group_broadcast(item.get_sub_group(), v383_data, 4))));
                float v389_data = r1[3];
                float v392_data = r2[3];
                r2[3] = (v392_data + (v370_data * (sycl::group_broadcast(item.get_sub_group(), v389_data, 4))));
                float v395_data = r1[4];
                float v398_data = r2[4];
                r2[4] = (v398_data + (v370_data * (sycl::group_broadcast(item.get_sub_group(), v395_data, 4))));
                float v401_data = r1[5];
                float v404_data = r2[5];
                r2[5] = (v404_data + (v370_data * (sycl::group_broadcast(item.get_sub_group(), v401_data, 4))));
                float v407_data = r1[6];
                float v410_data = r2[6];
                r2[6] = (v410_data + (v370_data * (sycl::group_broadcast(item.get_sub_group(), v407_data, 4))));
                float v413_data = r1[7];
                float v416_data = r2[7];
                r2[7] = (v416_data + (v370_data * (sycl::group_broadcast(item.get_sub_group(), v413_data, 4))));
                float v419_data = r1[8];
                float v422_data = r2[8];
                r2[8] = (v422_data + (v370_data * (sycl::group_broadcast(item.get_sub_group(), v419_data, 4))));
                float v425_data = r1[9];
                float v428_data = r2[9];
                r2[9] = (v428_data + (v370_data * (sycl::group_broadcast(item.get_sub_group(), v425_data, 4))));
                float v431_data = r1[10];
                float v434_data = r2[10];
                r2[10] = (v434_data + (v370_data * (sycl::group_broadcast(item.get_sub_group(), v431_data, 4))));
                float v437_data = r1[11];
                float v440_data = r2[11];
                r2[11] = (v440_data + (v370_data * (sycl::group_broadcast(item.get_sub_group(), v437_data, 4))));
              }
              if (v19_lead < 6) {
                float v446_data = r0[5];
                float v447_data = r1[0];
                float v450_data = r2[0];
                r2[0] = (v450_data + (v446_data * (sycl::group_broadcast(item.get_sub_group(), v447_data, 5))));
                float v453_data = r1[1];
                float v456_data = r2[1];
                r2[1] = (v456_data + (v446_data * (sycl::group_broadcast(item.get_sub_group(), v453_data, 5))));
                float v459_data = r1[2];
                float v462_data = r2[2];
                r2[2] = (v462_data + (v446_data * (sycl::group_broadcast(item.get_sub_group(), v459_data, 5))));
                float v465_data = r1[3];
                float v468_data = r2[3];
                r2[3] = (v468_data + (v446_data * (sycl::group_broadcast(item.get_sub_group(), v465_data, 5))));
                float v471_data = r1[4];
                float v474_data = r2[4];
                r2[4] = (v474_data + (v446_data * (sycl::group_broadcast(item.get_sub_group(), v471_data, 5))));
                float v477_data = r1[5];
                float v480_data = r2[5];
                r2[5] = (v480_data + (v446_data * (sycl::group_broadcast(item.get_sub_group(), v477_data, 5))));
                float v483_data = r1[6];
                float v486_data = r2[6];
                r2[6] = (v486_data + (v446_data * (sycl::group_broadcast(item.get_sub_group(), v483_data, 5))));
                float v489_data = r1[7];
                float v492_data = r2[7];
                r2[7] = (v492_data + (v446_data * (sycl::group_broadcast(item.get_sub_group(), v489_data, 5))));
                float v495_data = r1[8];
                float v498_data = r2[8];
                r2[8] = (v498_data + (v446_data * (sycl::group_broadcast(item.get_sub_group(), v495_data, 5))));
                float v501_data = r1[9];
                float v504_data = r2[9];
                r2[9] = (v504_data + (v446_data * (sycl::group_broadcast(item.get_sub_group(), v501_data, 5))));
                float v507_data = r1[10];
                float v510_data = r2[10];
                r2[10] = (v510_data + (v446_data * (sycl::group_broadcast(item.get_sub_group(), v507_data, 5))));
                float v513_data = r1[11];
                float v516_data = r2[11];
                r2[11] = (v516_data + (v446_data * (sycl::group_broadcast(item.get_sub_group(), v513_data, 5))));
              }
              if (v19_lead < 6) {
                float v522_data = r0[6];
                float v523_data = r1[0];
                float v526_data = r2[0];
                r2[0] = (v526_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v523_data, 6))));
                float v529_data = r1[1];
                float v532_data = r2[1];
                r2[1] = (v532_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v529_data, 6))));
                float v535_data = r1[2];
                float v538_data = r2[2];
                r2[2] = (v538_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v535_data, 6))));
                float v541_data = r1[3];
                float v544_data = r2[3];
                r2[3] = (v544_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v541_data, 6))));
                float v547_data = r1[4];
                float v550_data = r2[4];
                r2[4] = (v550_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v547_data, 6))));
                float v553_data = r1[5];
                float v556_data = r2[5];
                r2[5] = (v556_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v553_data, 6))));
                float v559_data = r1[6];
                float v562_data = r2[6];
                r2[6] = (v562_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v559_data, 6))));
                float v565_data = r1[7];
                float v568_data = r2[7];
                r2[7] = (v568_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v565_data, 6))));
                float v571_data = r1[8];
                float v574_data = r2[8];
                r2[8] = (v574_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v571_data, 6))));
                float v577_data = r1[9];
                float v580_data = r2[9];
                r2[9] = (v580_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v577_data, 6))));
                float v583_data = r1[10];
                float v586_data = r2[10];
                r2[10] = (v586_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v583_data, 6))));
                float v589_data = r1[11];
                float v592_data = r2[11];
                r2[11] = (v592_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v589_data, 6))));
              }
              if (v19_lead < 6) {
                float v598_data = r0[7];
                float v599_data = r1[0];
                float v602_data = r2[0];
                r2[0] = (v602_data + (v598_data * (sycl::group_broadcast(item.get_sub_group(), v599_data, 7))));
                float v605_data = r1[1];
                float v608_data = r2[1];
                r2[1] = (v608_data + (v598_data * (sycl::group_broadcast(item.get_sub_group(), v605_data, 7))));
                float v611_data = r1[2];
                float v614_data = r2[2];
                r2[2] = (v614_data + (v598_data * (sycl::group_broadcast(item.get_sub_group(), v611_data, 7))));
                float v617_data = r1[3];
                float v620_data = r2[3];
                r2[3] = (v620_data + (v598_data * (sycl::group_broadcast(item.get_sub_group(), v617_data, 7))));
                float v623_data = r1[4];
                float v626_data = r2[4];
                r2[4] = (v626_data + (v598_data * (sycl::group_broadcast(item.get_sub_group(), v623_data, 7))));
                float v629_data = r1[5];
                float v632_data = r2[5];
                r2[5] = (v632_data + (v598_data * (sycl::group_broadcast(item.get_sub_group(), v629_data, 7))));
                float v635_data = r1[6];
                float v638_data = r2[6];
                r2[6] = (v638_data + (v598_data * (sycl::group_broadcast(item.get_sub_group(), v635_data, 7))));
                float v641_data = r1[7];
                float v644_data = r2[7];
                r2[7] = (v644_data + (v598_data * (sycl::group_broadcast(item.get_sub_group(), v641_data, 7))));
                float v647_data = r1[8];
                float v650_data = r2[8];
                r2[8] = (v650_data + (v598_data * (sycl::group_broadcast(item.get_sub_group(), v647_data, 7))));
                float v653_data = r1[9];
                float v656_data = r2[9];
                r2[9] = (v656_data + (v598_data * (sycl::group_broadcast(item.get_sub_group(), v653_data, 7))));
                float v659_data = r1[10];
                float v662_data = r2[10];
                r2[10] = (v662_data + (v598_data * (sycl::group_broadcast(item.get_sub_group(), v659_data, 7))));
                float v665_data = r1[11];
                float v668_data = r2[11];
                r2[11] = (v668_data + (v598_data * (sycl::group_broadcast(item.get_sub_group(), v665_data, 7))));
              }
              if (v19_lead < 6) {
                float v674_data = r0[8];
                float v675_data = r1[0];
                float v678_data = r2[0];
                r2[0] = (v678_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v675_data, 8))));
                float v681_data = r1[1];
                float v684_data = r2[1];
                r2[1] = (v684_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v681_data, 8))));
                float v687_data = r1[2];
                float v690_data = r2[2];
                r2[2] = (v690_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v687_data, 8))));
                float v693_data = r1[3];
                float v696_data = r2[3];
                r2[3] = (v696_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v693_data, 8))));
                float v699_data = r1[4];
                float v702_data = r2[4];
                r2[4] = (v702_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v699_data, 8))));
                float v705_data = r1[5];
                float v708_data = r2[5];
                r2[5] = (v708_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v705_data, 8))));
                float v711_data = r1[6];
                float v714_data = r2[6];
                r2[6] = (v714_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v711_data, 8))));
                float v717_data = r1[7];
                float v720_data = r2[7];
                r2[7] = (v720_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v717_data, 8))));
                float v723_data = r1[8];
                float v726_data = r2[8];
                r2[8] = (v726_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v723_data, 8))));
                float v729_data = r1[9];
                float v732_data = r2[9];
                r2[9] = (v732_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v729_data, 8))));
                float v735_data = r1[10];
                float v738_data = r2[10];
                r2[10] = (v738_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v735_data, 8))));
                float v741_data = r1[11];
                float v744_data = r2[11];
                r2[11] = (v744_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v741_data, 8))));
              }
              if (v19_lead < 6) {
                float v750_data = r0[9];
                float v751_data = r1[0];
                float v754_data = r2[0];
                r2[0] = (v754_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v751_data, 9))));
                float v757_data = r1[1];
                float v760_data = r2[1];
                r2[1] = (v760_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v757_data, 9))));
                float v763_data = r1[2];
                float v766_data = r2[2];
                r2[2] = (v766_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v763_data, 9))));
                float v769_data = r1[3];
                float v772_data = r2[3];
                r2[3] = (v772_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v769_data, 9))));
                float v775_data = r1[4];
                float v778_data = r2[4];
                r2[4] = (v778_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v775_data, 9))));
                float v781_data = r1[5];
                float v784_data = r2[5];
                r2[5] = (v784_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v781_data, 9))));
                float v787_data = r1[6];
                float v790_data = r2[6];
                r2[6] = (v790_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v787_data, 9))));
                float v793_data = r1[7];
                float v796_data = r2[7];
                r2[7] = (v796_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v793_data, 9))));
                float v799_data = r1[8];
                float v802_data = r2[8];
                r2[8] = (v802_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v799_data, 9))));
                float v805_data = r1[9];
                float v808_data = r2[9];
                r2[9] = (v808_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v805_data, 9))));
                float v811_data = r1[10];
                float v814_data = r2[10];
                r2[10] = (v814_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v811_data, 9))));
                float v817_data = r1[11];
                float v820_data = r2[11];
                r2[11] = (v820_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v817_data, 9))));
              }
              if (v19_lead < 6) {
                float v826_data = r0[10];
                float v827_data = r1[0];
                float v830_data = r2[0];
                r2[0] = (v830_data + (v826_data * (sycl::group_broadcast(item.get_sub_group(), v827_data, 10))));
                float v833_data = r1[1];
                float v836_data = r2[1];
                r2[1] = (v836_data + (v826_data * (sycl::group_broadcast(item.get_sub_group(), v833_data, 10))));
                float v839_data = r1[2];
                float v842_data = r2[2];
                r2[2] = (v842_data + (v826_data * (sycl::group_broadcast(item.get_sub_group(), v839_data, 10))));
                float v845_data = r1[3];
                float v848_data = r2[3];
                r2[3] = (v848_data + (v826_data * (sycl::group_broadcast(item.get_sub_group(), v845_data, 10))));
                float v851_data = r1[4];
                float v854_data = r2[4];
                r2[4] = (v854_data + (v826_data * (sycl::group_broadcast(item.get_sub_group(), v851_data, 10))));
                float v857_data = r1[5];
                float v860_data = r2[5];
                r2[5] = (v860_data + (v826_data * (sycl::group_broadcast(item.get_sub_group(), v857_data, 10))));
                float v863_data = r1[6];
                float v866_data = r2[6];
                r2[6] = (v866_data + (v826_data * (sycl::group_broadcast(item.get_sub_group(), v863_data, 10))));
                float v869_data = r1[7];
                float v872_data = r2[7];
                r2[7] = (v872_data + (v826_data * (sycl::group_broadcast(item.get_sub_group(), v869_data, 10))));
                float v875_data = r1[8];
                float v878_data = r2[8];
                r2[8] = (v878_data + (v826_data * (sycl::group_broadcast(item.get_sub_group(), v875_data, 10))));
                float v881_data = r1[9];
                float v884_data = r2[9];
                r2[9] = (v884_data + (v826_data * (sycl::group_broadcast(item.get_sub_group(), v881_data, 10))));
                float v887_data = r1[10];
                float v890_data = r2[10];
                r2[10] = (v890_data + (v826_data * (sycl::group_broadcast(item.get_sub_group(), v887_data, 10))));
                float v893_data = r1[11];
                float v896_data = r2[11];
                r2[11] = (v896_data + (v826_data * (sycl::group_broadcast(item.get_sub_group(), v893_data, 10))));
              }
              if (v19_lead < 6) {
                float v902_data = r0[11];
                float v903_data = r1[0];
                float v906_data = r2[0];
                r2[0] = (v906_data + (v902_data * (sycl::group_broadcast(item.get_sub_group(), v903_data, 11))));
                float v909_data = r1[1];
                float v912_data = r2[1];
                r2[1] = (v912_data + (v902_data * (sycl::group_broadcast(item.get_sub_group(), v909_data, 11))));
                float v915_data = r1[2];
                float v918_data = r2[2];
                r2[2] = (v918_data + (v902_data * (sycl::group_broadcast(item.get_sub_group(), v915_data, 11))));
                float v921_data = r1[3];
                float v924_data = r2[3];
                r2[3] = (v924_data + (v902_data * (sycl::group_broadcast(item.get_sub_group(), v921_data, 11))));
                float v927_data = r1[4];
                float v930_data = r2[4];
                r2[4] = (v930_data + (v902_data * (sycl::group_broadcast(item.get_sub_group(), v927_data, 11))));
                float v933_data = r1[5];
                float v936_data = r2[5];
                r2[5] = (v936_data + (v902_data * (sycl::group_broadcast(item.get_sub_group(), v933_data, 11))));
                float v939_data = r1[6];
                float v942_data = r2[6];
                r2[6] = (v942_data + (v902_data * (sycl::group_broadcast(item.get_sub_group(), v939_data, 11))));
                float v945_data = r1[7];
                float v948_data = r2[7];
                r2[7] = (v948_data + (v902_data * (sycl::group_broadcast(item.get_sub_group(), v945_data, 11))));
                float v951_data = r1[8];
                float v954_data = r2[8];
                r2[8] = (v954_data + (v902_data * (sycl::group_broadcast(item.get_sub_group(), v951_data, 11))));
                float v957_data = r1[9];
                float v960_data = r2[9];
                r2[9] = (v960_data + (v902_data * (sycl::group_broadcast(item.get_sub_group(), v957_data, 11))));
                float v963_data = r1[10];
                float v966_data = r2[10];
                r2[10] = (v966_data + (v902_data * (sycl::group_broadcast(item.get_sub_group(), v963_data, 11))));
                float v969_data = r1[11];
                float v972_data = r2[11];
                r2[11] = (v972_data + (v902_data * (sycl::group_broadcast(item.get_sub_group(), v969_data, 11))));
              }
              // s0 = store{r>s}(localShrMem0, r2);
              if (v19_lead < 6) {
                #pragma unroll
                for (int32_t v978_i1 = 0; v978_i1 < 12; ++v978_i1) {
                  float v980_data = r2[v978_i1];
                  int32_t v987_a = v19_lead + (v978_i1 * 12);
                  s0[(v987_a ^ ((v987_a >> 4) & 15))] = v980_data;
                }
              }
              float r5[12]{};
              // r5 = load{g>r}(glb_m4);
              if (v19_lead < 12) {
                #pragma unroll
                for (int32_t v996_i1 = 0; v996_i1 < 12; ++v996_i1) {
                  float v1004_data = glb_m4[(v19_lead + (v996_i1 * 12))];
                  r5[v996_i1] = v1004_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m2););
              float r4[12]{};
              // r4 = +(r3 * r1) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              float ir4[12]{};
              if (v19_lead < 6) {
                float v1012_data = r3[0];
                float v1013_data = r1[0];
                float v1016_data = ir4[0];
                ir4[0] = (v1016_data + (v1012_data * (sycl::group_broadcast(item.get_sub_group(), v1013_data, 0))));
                float v1019_data = r1[1];
                float v1022_data = ir4[1];
                ir4[1] = (v1022_data + (v1012_data * (sycl::group_broadcast(item.get_sub_group(), v1019_data, 0))));
                float v1025_data = r1[2];
                float v1028_data = ir4[2];
                ir4[2] = (v1028_data + (v1012_data * (sycl::group_broadcast(item.get_sub_group(), v1025_data, 0))));
                float v1031_data = r1[3];
                float v1034_data = ir4[3];
                ir4[3] = (v1034_data + (v1012_data * (sycl::group_broadcast(item.get_sub_group(), v1031_data, 0))));
                float v1037_data = r1[4];
                float v1040_data = ir4[4];
                ir4[4] = (v1040_data + (v1012_data * (sycl::group_broadcast(item.get_sub_group(), v1037_data, 0))));
                float v1043_data = r1[5];
                float v1046_data = ir4[5];
                ir4[5] = (v1046_data + (v1012_data * (sycl::group_broadcast(item.get_sub_group(), v1043_data, 0))));
                float v1049_data = r1[6];
                float v1052_data = ir4[6];
                ir4[6] = (v1052_data + (v1012_data * (sycl::group_broadcast(item.get_sub_group(), v1049_data, 0))));
                float v1055_data = r1[7];
                float v1058_data = ir4[7];
                ir4[7] = (v1058_data + (v1012_data * (sycl::group_broadcast(item.get_sub_group(), v1055_data, 0))));
                float v1061_data = r1[8];
                float v1064_data = ir4[8];
                ir4[8] = (v1064_data + (v1012_data * (sycl::group_broadcast(item.get_sub_group(), v1061_data, 0))));
                float v1067_data = r1[9];
                float v1070_data = ir4[9];
                ir4[9] = (v1070_data + (v1012_data * (sycl::group_broadcast(item.get_sub_group(), v1067_data, 0))));
                float v1073_data = r1[10];
                float v1076_data = ir4[10];
                ir4[10] = (v1076_data + (v1012_data * (sycl::group_broadcast(item.get_sub_group(), v1073_data, 0))));
                float v1079_data = r1[11];
                float v1082_data = ir4[11];
                ir4[11] = (v1082_data + (v1012_data * (sycl::group_broadcast(item.get_sub_group(), v1079_data, 0))));
              }
              if (v19_lead < 6) {
                float v1088_data = r3[1];
                float v1089_data = r1[0];
                float v1092_data = ir4[0];
                ir4[0] = (v1092_data + (v1088_data * (sycl::group_broadcast(item.get_sub_group(), v1089_data, 1))));
                float v1095_data = r1[1];
                float v1098_data = ir4[1];
                ir4[1] = (v1098_data + (v1088_data * (sycl::group_broadcast(item.get_sub_group(), v1095_data, 1))));
                float v1101_data = r1[2];
                float v1104_data = ir4[2];
                ir4[2] = (v1104_data + (v1088_data * (sycl::group_broadcast(item.get_sub_group(), v1101_data, 1))));
                float v1107_data = r1[3];
                float v1110_data = ir4[3];
                ir4[3] = (v1110_data + (v1088_data * (sycl::group_broadcast(item.get_sub_group(), v1107_data, 1))));
                float v1113_data = r1[4];
                float v1116_data = ir4[4];
                ir4[4] = (v1116_data + (v1088_data * (sycl::group_broadcast(item.get_sub_group(), v1113_data, 1))));
                float v1119_data = r1[5];
                float v1122_data = ir4[5];
                ir4[5] = (v1122_data + (v1088_data * (sycl::group_broadcast(item.get_sub_group(), v1119_data, 1))));
                float v1125_data = r1[6];
                float v1128_data = ir4[6];
                ir4[6] = (v1128_data + (v1088_data * (sycl::group_broadcast(item.get_sub_group(), v1125_data, 1))));
                float v1131_data = r1[7];
                float v1134_data = ir4[7];
                ir4[7] = (v1134_data + (v1088_data * (sycl::group_broadcast(item.get_sub_group(), v1131_data, 1))));
                float v1137_data = r1[8];
                float v1140_data = ir4[8];
                ir4[8] = (v1140_data + (v1088_data * (sycl::group_broadcast(item.get_sub_group(), v1137_data, 1))));
                float v1143_data = r1[9];
                float v1146_data = ir4[9];
                ir4[9] = (v1146_data + (v1088_data * (sycl::group_broadcast(item.get_sub_group(), v1143_data, 1))));
                float v1149_data = r1[10];
                float v1152_data = ir4[10];
                ir4[10] = (v1152_data + (v1088_data * (sycl::group_broadcast(item.get_sub_group(), v1149_data, 1))));
                float v1155_data = r1[11];
                float v1158_data = ir4[11];
                ir4[11] = (v1158_data + (v1088_data * (sycl::group_broadcast(item.get_sub_group(), v1155_data, 1))));
              }
              if (v19_lead < 6) {
                float v1164_data = r3[2];
                float v1165_data = r1[0];
                float v1168_data = ir4[0];
                ir4[0] = (v1168_data + (v1164_data * (sycl::group_broadcast(item.get_sub_group(), v1165_data, 2))));
                float v1171_data = r1[1];
                float v1174_data = ir4[1];
                ir4[1] = (v1174_data + (v1164_data * (sycl::group_broadcast(item.get_sub_group(), v1171_data, 2))));
                float v1177_data = r1[2];
                float v1180_data = ir4[2];
                ir4[2] = (v1180_data + (v1164_data * (sycl::group_broadcast(item.get_sub_group(), v1177_data, 2))));
                float v1183_data = r1[3];
                float v1186_data = ir4[3];
                ir4[3] = (v1186_data + (v1164_data * (sycl::group_broadcast(item.get_sub_group(), v1183_data, 2))));
                float v1189_data = r1[4];
                float v1192_data = ir4[4];
                ir4[4] = (v1192_data + (v1164_data * (sycl::group_broadcast(item.get_sub_group(), v1189_data, 2))));
                float v1195_data = r1[5];
                float v1198_data = ir4[5];
                ir4[5] = (v1198_data + (v1164_data * (sycl::group_broadcast(item.get_sub_group(), v1195_data, 2))));
                float v1201_data = r1[6];
                float v1204_data = ir4[6];
                ir4[6] = (v1204_data + (v1164_data * (sycl::group_broadcast(item.get_sub_group(), v1201_data, 2))));
                float v1207_data = r1[7];
                float v1210_data = ir4[7];
                ir4[7] = (v1210_data + (v1164_data * (sycl::group_broadcast(item.get_sub_group(), v1207_data, 2))));
                float v1213_data = r1[8];
                float v1216_data = ir4[8];
                ir4[8] = (v1216_data + (v1164_data * (sycl::group_broadcast(item.get_sub_group(), v1213_data, 2))));
                float v1219_data = r1[9];
                float v1222_data = ir4[9];
                ir4[9] = (v1222_data + (v1164_data * (sycl::group_broadcast(item.get_sub_group(), v1219_data, 2))));
                float v1225_data = r1[10];
                float v1228_data = ir4[10];
                ir4[10] = (v1228_data + (v1164_data * (sycl::group_broadcast(item.get_sub_group(), v1225_data, 2))));
                float v1231_data = r1[11];
                float v1234_data = ir4[11];
                ir4[11] = (v1234_data + (v1164_data * (sycl::group_broadcast(item.get_sub_group(), v1231_data, 2))));
              }
              if (v19_lead < 6) {
                float v1240_data = r3[3];
                float v1241_data = r1[0];
                float v1244_data = ir4[0];
                ir4[0] = (v1244_data + (v1240_data * (sycl::group_broadcast(item.get_sub_group(), v1241_data, 3))));
                float v1247_data = r1[1];
                float v1250_data = ir4[1];
                ir4[1] = (v1250_data + (v1240_data * (sycl::group_broadcast(item.get_sub_group(), v1247_data, 3))));
                float v1253_data = r1[2];
                float v1256_data = ir4[2];
                ir4[2] = (v1256_data + (v1240_data * (sycl::group_broadcast(item.get_sub_group(), v1253_data, 3))));
                float v1259_data = r1[3];
                float v1262_data = ir4[3];
                ir4[3] = (v1262_data + (v1240_data * (sycl::group_broadcast(item.get_sub_group(), v1259_data, 3))));
                float v1265_data = r1[4];
                float v1268_data = ir4[4];
                ir4[4] = (v1268_data + (v1240_data * (sycl::group_broadcast(item.get_sub_group(), v1265_data, 3))));
                float v1271_data = r1[5];
                float v1274_data = ir4[5];
                ir4[5] = (v1274_data + (v1240_data * (sycl::group_broadcast(item.get_sub_group(), v1271_data, 3))));
                float v1277_data = r1[6];
                float v1280_data = ir4[6];
                ir4[6] = (v1280_data + (v1240_data * (sycl::group_broadcast(item.get_sub_group(), v1277_data, 3))));
                float v1283_data = r1[7];
                float v1286_data = ir4[7];
                ir4[7] = (v1286_data + (v1240_data * (sycl::group_broadcast(item.get_sub_group(), v1283_data, 3))));
                float v1289_data = r1[8];
                float v1292_data = ir4[8];
                ir4[8] = (v1292_data + (v1240_data * (sycl::group_broadcast(item.get_sub_group(), v1289_data, 3))));
                float v1295_data = r1[9];
                float v1298_data = ir4[9];
                ir4[9] = (v1298_data + (v1240_data * (sycl::group_broadcast(item.get_sub_group(), v1295_data, 3))));
                float v1301_data = r1[10];
                float v1304_data = ir4[10];
                ir4[10] = (v1304_data + (v1240_data * (sycl::group_broadcast(item.get_sub_group(), v1301_data, 3))));
                float v1307_data = r1[11];
                float v1310_data = ir4[11];
                ir4[11] = (v1310_data + (v1240_data * (sycl::group_broadcast(item.get_sub_group(), v1307_data, 3))));
              }
              if (v19_lead < 6) {
                float v1316_data = r3[4];
                float v1317_data = r1[0];
                float v1320_data = ir4[0];
                ir4[0] = (v1320_data + (v1316_data * (sycl::group_broadcast(item.get_sub_group(), v1317_data, 4))));
                float v1323_data = r1[1];
                float v1326_data = ir4[1];
                ir4[1] = (v1326_data + (v1316_data * (sycl::group_broadcast(item.get_sub_group(), v1323_data, 4))));
                float v1329_data = r1[2];
                float v1332_data = ir4[2];
                ir4[2] = (v1332_data + (v1316_data * (sycl::group_broadcast(item.get_sub_group(), v1329_data, 4))));
                float v1335_data = r1[3];
                float v1338_data = ir4[3];
                ir4[3] = (v1338_data + (v1316_data * (sycl::group_broadcast(item.get_sub_group(), v1335_data, 4))));
                float v1341_data = r1[4];
                float v1344_data = ir4[4];
                ir4[4] = (v1344_data + (v1316_data * (sycl::group_broadcast(item.get_sub_group(), v1341_data, 4))));
                float v1347_data = r1[5];
                float v1350_data = ir4[5];
                ir4[5] = (v1350_data + (v1316_data * (sycl::group_broadcast(item.get_sub_group(), v1347_data, 4))));
                float v1353_data = r1[6];
                float v1356_data = ir4[6];
                ir4[6] = (v1356_data + (v1316_data * (sycl::group_broadcast(item.get_sub_group(), v1353_data, 4))));
                float v1359_data = r1[7];
                float v1362_data = ir4[7];
                ir4[7] = (v1362_data + (v1316_data * (sycl::group_broadcast(item.get_sub_group(), v1359_data, 4))));
                float v1365_data = r1[8];
                float v1368_data = ir4[8];
                ir4[8] = (v1368_data + (v1316_data * (sycl::group_broadcast(item.get_sub_group(), v1365_data, 4))));
                float v1371_data = r1[9];
                float v1374_data = ir4[9];
                ir4[9] = (v1374_data + (v1316_data * (sycl::group_broadcast(item.get_sub_group(), v1371_data, 4))));
                float v1377_data = r1[10];
                float v1380_data = ir4[10];
                ir4[10] = (v1380_data + (v1316_data * (sycl::group_broadcast(item.get_sub_group(), v1377_data, 4))));
                float v1383_data = r1[11];
                float v1386_data = ir4[11];
                ir4[11] = (v1386_data + (v1316_data * (sycl::group_broadcast(item.get_sub_group(), v1383_data, 4))));
              }
              if (v19_lead < 6) {
                float v1392_data = r3[5];
                float v1393_data = r1[0];
                float v1396_data = ir4[0];
                ir4[0] = (v1396_data + (v1392_data * (sycl::group_broadcast(item.get_sub_group(), v1393_data, 5))));
                float v1399_data = r1[1];
                float v1402_data = ir4[1];
                ir4[1] = (v1402_data + (v1392_data * (sycl::group_broadcast(item.get_sub_group(), v1399_data, 5))));
                float v1405_data = r1[2];
                float v1408_data = ir4[2];
                ir4[2] = (v1408_data + (v1392_data * (sycl::group_broadcast(item.get_sub_group(), v1405_data, 5))));
                float v1411_data = r1[3];
                float v1414_data = ir4[3];
                ir4[3] = (v1414_data + (v1392_data * (sycl::group_broadcast(item.get_sub_group(), v1411_data, 5))));
                float v1417_data = r1[4];
                float v1420_data = ir4[4];
                ir4[4] = (v1420_data + (v1392_data * (sycl::group_broadcast(item.get_sub_group(), v1417_data, 5))));
                float v1423_data = r1[5];
                float v1426_data = ir4[5];
                ir4[5] = (v1426_data + (v1392_data * (sycl::group_broadcast(item.get_sub_group(), v1423_data, 5))));
                float v1429_data = r1[6];
                float v1432_data = ir4[6];
                ir4[6] = (v1432_data + (v1392_data * (sycl::group_broadcast(item.get_sub_group(), v1429_data, 5))));
                float v1435_data = r1[7];
                float v1438_data = ir4[7];
                ir4[7] = (v1438_data + (v1392_data * (sycl::group_broadcast(item.get_sub_group(), v1435_data, 5))));
                float v1441_data = r1[8];
                float v1444_data = ir4[8];
                ir4[8] = (v1444_data + (v1392_data * (sycl::group_broadcast(item.get_sub_group(), v1441_data, 5))));
                float v1447_data = r1[9];
                float v1450_data = ir4[9];
                ir4[9] = (v1450_data + (v1392_data * (sycl::group_broadcast(item.get_sub_group(), v1447_data, 5))));
                float v1453_data = r1[10];
                float v1456_data = ir4[10];
                ir4[10] = (v1456_data + (v1392_data * (sycl::group_broadcast(item.get_sub_group(), v1453_data, 5))));
                float v1459_data = r1[11];
                float v1462_data = ir4[11];
                ir4[11] = (v1462_data + (v1392_data * (sycl::group_broadcast(item.get_sub_group(), v1459_data, 5))));
              }
              if (v19_lead < 6) {
                float v1468_data = r3[6];
                float v1469_data = r1[0];
                float v1472_data = ir4[0];
                ir4[0] = (v1472_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1469_data, 6))));
                float v1475_data = r1[1];
                float v1478_data = ir4[1];
                ir4[1] = (v1478_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1475_data, 6))));
                float v1481_data = r1[2];
                float v1484_data = ir4[2];
                ir4[2] = (v1484_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1481_data, 6))));
                float v1487_data = r1[3];
                float v1490_data = ir4[3];
                ir4[3] = (v1490_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1487_data, 6))));
                float v1493_data = r1[4];
                float v1496_data = ir4[4];
                ir4[4] = (v1496_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1493_data, 6))));
                float v1499_data = r1[5];
                float v1502_data = ir4[5];
                ir4[5] = (v1502_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1499_data, 6))));
                float v1505_data = r1[6];
                float v1508_data = ir4[6];
                ir4[6] = (v1508_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1505_data, 6))));
                float v1511_data = r1[7];
                float v1514_data = ir4[7];
                ir4[7] = (v1514_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1511_data, 6))));
                float v1517_data = r1[8];
                float v1520_data = ir4[8];
                ir4[8] = (v1520_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1517_data, 6))));
                float v1523_data = r1[9];
                float v1526_data = ir4[9];
                ir4[9] = (v1526_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1523_data, 6))));
                float v1529_data = r1[10];
                float v1532_data = ir4[10];
                ir4[10] = (v1532_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1529_data, 6))));
                float v1535_data = r1[11];
                float v1538_data = ir4[11];
                ir4[11] = (v1538_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1535_data, 6))));
              }
              if (v19_lead < 6) {
                float v1544_data = r3[7];
                float v1545_data = r1[0];
                float v1548_data = ir4[0];
                ir4[0] = (v1548_data + (v1544_data * (sycl::group_broadcast(item.get_sub_group(), v1545_data, 7))));
                float v1551_data = r1[1];
                float v1554_data = ir4[1];
                ir4[1] = (v1554_data + (v1544_data * (sycl::group_broadcast(item.get_sub_group(), v1551_data, 7))));
                float v1557_data = r1[2];
                float v1560_data = ir4[2];
                ir4[2] = (v1560_data + (v1544_data * (sycl::group_broadcast(item.get_sub_group(), v1557_data, 7))));
                float v1563_data = r1[3];
                float v1566_data = ir4[3];
                ir4[3] = (v1566_data + (v1544_data * (sycl::group_broadcast(item.get_sub_group(), v1563_data, 7))));
                float v1569_data = r1[4];
                float v1572_data = ir4[4];
                ir4[4] = (v1572_data + (v1544_data * (sycl::group_broadcast(item.get_sub_group(), v1569_data, 7))));
                float v1575_data = r1[5];
                float v1578_data = ir4[5];
                ir4[5] = (v1578_data + (v1544_data * (sycl::group_broadcast(item.get_sub_group(), v1575_data, 7))));
                float v1581_data = r1[6];
                float v1584_data = ir4[6];
                ir4[6] = (v1584_data + (v1544_data * (sycl::group_broadcast(item.get_sub_group(), v1581_data, 7))));
                float v1587_data = r1[7];
                float v1590_data = ir4[7];
                ir4[7] = (v1590_data + (v1544_data * (sycl::group_broadcast(item.get_sub_group(), v1587_data, 7))));
                float v1593_data = r1[8];
                float v1596_data = ir4[8];
                ir4[8] = (v1596_data + (v1544_data * (sycl::group_broadcast(item.get_sub_group(), v1593_data, 7))));
                float v1599_data = r1[9];
                float v1602_data = ir4[9];
                ir4[9] = (v1602_data + (v1544_data * (sycl::group_broadcast(item.get_sub_group(), v1599_data, 7))));
                float v1605_data = r1[10];
                float v1608_data = ir4[10];
                ir4[10] = (v1608_data + (v1544_data * (sycl::group_broadcast(item.get_sub_group(), v1605_data, 7))));
                float v1611_data = r1[11];
                float v1614_data = ir4[11];
                ir4[11] = (v1614_data + (v1544_data * (sycl::group_broadcast(item.get_sub_group(), v1611_data, 7))));
              }
              if (v19_lead < 6) {
                float v1620_data = r3[8];
                float v1621_data = r1[0];
                float v1624_data = ir4[0];
                ir4[0] = (v1624_data + (v1620_data * (sycl::group_broadcast(item.get_sub_group(), v1621_data, 8))));
                float v1627_data = r1[1];
                float v1630_data = ir4[1];
                ir4[1] = (v1630_data + (v1620_data * (sycl::group_broadcast(item.get_sub_group(), v1627_data, 8))));
                float v1633_data = r1[2];
                float v1636_data = ir4[2];
                ir4[2] = (v1636_data + (v1620_data * (sycl::group_broadcast(item.get_sub_group(), v1633_data, 8))));
                float v1639_data = r1[3];
                float v1642_data = ir4[3];
                ir4[3] = (v1642_data + (v1620_data * (sycl::group_broadcast(item.get_sub_group(), v1639_data, 8))));
                float v1645_data = r1[4];
                float v1648_data = ir4[4];
                ir4[4] = (v1648_data + (v1620_data * (sycl::group_broadcast(item.get_sub_group(), v1645_data, 8))));
                float v1651_data = r1[5];
                float v1654_data = ir4[5];
                ir4[5] = (v1654_data + (v1620_data * (sycl::group_broadcast(item.get_sub_group(), v1651_data, 8))));
                float v1657_data = r1[6];
                float v1660_data = ir4[6];
                ir4[6] = (v1660_data + (v1620_data * (sycl::group_broadcast(item.get_sub_group(), v1657_data, 8))));
                float v1663_data = r1[7];
                float v1666_data = ir4[7];
                ir4[7] = (v1666_data + (v1620_data * (sycl::group_broadcast(item.get_sub_group(), v1663_data, 8))));
                float v1669_data = r1[8];
                float v1672_data = ir4[8];
                ir4[8] = (v1672_data + (v1620_data * (sycl::group_broadcast(item.get_sub_group(), v1669_data, 8))));
                float v1675_data = r1[9];
                float v1678_data = ir4[9];
                ir4[9] = (v1678_data + (v1620_data * (sycl::group_broadcast(item.get_sub_group(), v1675_data, 8))));
                float v1681_data = r1[10];
                float v1684_data = ir4[10];
                ir4[10] = (v1684_data + (v1620_data * (sycl::group_broadcast(item.get_sub_group(), v1681_data, 8))));
                float v1687_data = r1[11];
                float v1690_data = ir4[11];
                ir4[11] = (v1690_data + (v1620_data * (sycl::group_broadcast(item.get_sub_group(), v1687_data, 8))));
              }
              if (v19_lead < 6) {
                float v1696_data = r3[9];
                float v1697_data = r1[0];
                float v1700_data = ir4[0];
                ir4[0] = (v1700_data + (v1696_data * (sycl::group_broadcast(item.get_sub_group(), v1697_data, 9))));
                float v1703_data = r1[1];
                float v1706_data = ir4[1];
                ir4[1] = (v1706_data + (v1696_data * (sycl::group_broadcast(item.get_sub_group(), v1703_data, 9))));
                float v1709_data = r1[2];
                float v1712_data = ir4[2];
                ir4[2] = (v1712_data + (v1696_data * (sycl::group_broadcast(item.get_sub_group(), v1709_data, 9))));
                float v1715_data = r1[3];
                float v1718_data = ir4[3];
                ir4[3] = (v1718_data + (v1696_data * (sycl::group_broadcast(item.get_sub_group(), v1715_data, 9))));
                float v1721_data = r1[4];
                float v1724_data = ir4[4];
                ir4[4] = (v1724_data + (v1696_data * (sycl::group_broadcast(item.get_sub_group(), v1721_data, 9))));
                float v1727_data = r1[5];
                float v1730_data = ir4[5];
                ir4[5] = (v1730_data + (v1696_data * (sycl::group_broadcast(item.get_sub_group(), v1727_data, 9))));
                float v1733_data = r1[6];
                float v1736_data = ir4[6];
                ir4[6] = (v1736_data + (v1696_data * (sycl::group_broadcast(item.get_sub_group(), v1733_data, 9))));
                float v1739_data = r1[7];
                float v1742_data = ir4[7];
                ir4[7] = (v1742_data + (v1696_data * (sycl::group_broadcast(item.get_sub_group(), v1739_data, 9))));
                float v1745_data = r1[8];
                float v1748_data = ir4[8];
                ir4[8] = (v1748_data + (v1696_data * (sycl::group_broadcast(item.get_sub_group(), v1745_data, 9))));
                float v1751_data = r1[9];
                float v1754_data = ir4[9];
                ir4[9] = (v1754_data + (v1696_data * (sycl::group_broadcast(item.get_sub_group(), v1751_data, 9))));
                float v1757_data = r1[10];
                float v1760_data = ir4[10];
                ir4[10] = (v1760_data + (v1696_data * (sycl::group_broadcast(item.get_sub_group(), v1757_data, 9))));
                float v1763_data = r1[11];
                float v1766_data = ir4[11];
                ir4[11] = (v1766_data + (v1696_data * (sycl::group_broadcast(item.get_sub_group(), v1763_data, 9))));
              }
              if (v19_lead < 6) {
                float v1772_data = r3[10];
                float v1773_data = r1[0];
                float v1776_data = ir4[0];
                ir4[0] = (v1776_data + (v1772_data * (sycl::group_broadcast(item.get_sub_group(), v1773_data, 10))));
                float v1779_data = r1[1];
                float v1782_data = ir4[1];
                ir4[1] = (v1782_data + (v1772_data * (sycl::group_broadcast(item.get_sub_group(), v1779_data, 10))));
                float v1785_data = r1[2];
                float v1788_data = ir4[2];
                ir4[2] = (v1788_data + (v1772_data * (sycl::group_broadcast(item.get_sub_group(), v1785_data, 10))));
                float v1791_data = r1[3];
                float v1794_data = ir4[3];
                ir4[3] = (v1794_data + (v1772_data * (sycl::group_broadcast(item.get_sub_group(), v1791_data, 10))));
                float v1797_data = r1[4];
                float v1800_data = ir4[4];
                ir4[4] = (v1800_data + (v1772_data * (sycl::group_broadcast(item.get_sub_group(), v1797_data, 10))));
                float v1803_data = r1[5];
                float v1806_data = ir4[5];
                ir4[5] = (v1806_data + (v1772_data * (sycl::group_broadcast(item.get_sub_group(), v1803_data, 10))));
                float v1809_data = r1[6];
                float v1812_data = ir4[6];
                ir4[6] = (v1812_data + (v1772_data * (sycl::group_broadcast(item.get_sub_group(), v1809_data, 10))));
                float v1815_data = r1[7];
                float v1818_data = ir4[7];
                ir4[7] = (v1818_data + (v1772_data * (sycl::group_broadcast(item.get_sub_group(), v1815_data, 10))));
                float v1821_data = r1[8];
                float v1824_data = ir4[8];
                ir4[8] = (v1824_data + (v1772_data * (sycl::group_broadcast(item.get_sub_group(), v1821_data, 10))));
                float v1827_data = r1[9];
                float v1830_data = ir4[9];
                ir4[9] = (v1830_data + (v1772_data * (sycl::group_broadcast(item.get_sub_group(), v1827_data, 10))));
                float v1833_data = r1[10];
                float v1836_data = ir4[10];
                ir4[10] = (v1836_data + (v1772_data * (sycl::group_broadcast(item.get_sub_group(), v1833_data, 10))));
                float v1839_data = r1[11];
                float v1842_data = ir4[11];
                ir4[11] = (v1842_data + (v1772_data * (sycl::group_broadcast(item.get_sub_group(), v1839_data, 10))));
              }
              if (v19_lead < 6) {
                float v1848_data = r3[11];
                float v1849_data = r1[0];
                float v1852_data = ir4[0];
                ir4[0] = (v1852_data + (v1848_data * (sycl::group_broadcast(item.get_sub_group(), v1849_data, 11))));
                float v1855_data = r1[1];
                float v1858_data = ir4[1];
                ir4[1] = (v1858_data + (v1848_data * (sycl::group_broadcast(item.get_sub_group(), v1855_data, 11))));
                float v1861_data = r1[2];
                float v1864_data = ir4[2];
                ir4[2] = (v1864_data + (v1848_data * (sycl::group_broadcast(item.get_sub_group(), v1861_data, 11))));
                float v1867_data = r1[3];
                float v1870_data = ir4[3];
                ir4[3] = (v1870_data + (v1848_data * (sycl::group_broadcast(item.get_sub_group(), v1867_data, 11))));
                float v1873_data = r1[4];
                float v1876_data = ir4[4];
                ir4[4] = (v1876_data + (v1848_data * (sycl::group_broadcast(item.get_sub_group(), v1873_data, 11))));
                float v1879_data = r1[5];
                float v1882_data = ir4[5];
                ir4[5] = (v1882_data + (v1848_data * (sycl::group_broadcast(item.get_sub_group(), v1879_data, 11))));
                float v1885_data = r1[6];
                float v1888_data = ir4[6];
                ir4[6] = (v1888_data + (v1848_data * (sycl::group_broadcast(item.get_sub_group(), v1885_data, 11))));
                float v1891_data = r1[7];
                float v1894_data = ir4[7];
                ir4[7] = (v1894_data + (v1848_data * (sycl::group_broadcast(item.get_sub_group(), v1891_data, 11))));
                float v1897_data = r1[8];
                float v1900_data = ir4[8];
                ir4[8] = (v1900_data + (v1848_data * (sycl::group_broadcast(item.get_sub_group(), v1897_data, 11))));
                float v1903_data = r1[9];
                float v1906_data = ir4[9];
                ir4[9] = (v1906_data + (v1848_data * (sycl::group_broadcast(item.get_sub_group(), v1903_data, 11))));
                float v1909_data = r1[10];
                float v1912_data = ir4[10];
                ir4[10] = (v1912_data + (v1848_data * (sycl::group_broadcast(item.get_sub_group(), v1909_data, 11))));
                float v1915_data = r1[11];
                float v1918_data = ir4[11];
                ir4[11] = (v1918_data + (v1848_data * (sycl::group_broadcast(item.get_sub_group(), v1915_data, 11))));
              }
              if (v19_lead < 6) {
                #pragma unroll
                for (int32_t v1924_n1 = 0; v1924_n1 < 12; ++v1924_n1) {
                  float v1926_data = ir4[v1924_n1];
                  r4[v1924_n1] = v1926_data;
                }
              }
              // s0 = store{r>s}(localShrMem0, r4);
              if (v19_lead < 6) {
                int32_t v1940_off = v19_lead + 6;
                #pragma unroll
                for (int32_t v1932_i1 = 0; v1932_i1 < 12; ++v1932_i1) {
                  float v1934_data = r4[v1932_i1];
                  int32_t v1942_a = v1940_off + (v1932_i1 * 12);
                  s0[(v1942_a ^ ((v1942_a >> 4) & 15))] = v1934_data;
                }
              }
              // wait(r5 = load{g>r}(glb_m4););
              float r6[12]{};
              sycl::group_barrier(item.get_sub_group());
              // r6 = +(r5 * s0) + None
              // [(0, 12), (0, 12)] [(0, 12)]
              float ir6[12]{};
              if (v19_lead < 12) {
                float v1952_data = r5[0];
                float v1953_data = s0[0];
                float v1955_data = ir6[0];
                ir6[0] = (v1955_data + (v1952_data * v1953_data));
                float v1958_data = s0[12];
                float v1960_data = ir6[1];
                ir6[1] = (v1960_data + (v1952_data * v1958_data));
                float v1963_data = s0[25];
                float v1965_data = ir6[2];
                ir6[2] = (v1965_data + (v1952_data * v1963_data));
                float v1968_data = s0[38];
                float v1970_data = ir6[3];
                ir6[3] = (v1970_data + (v1952_data * v1968_data));
                float v1973_data = s0[51];
                float v1975_data = ir6[4];
                ir6[4] = (v1975_data + (v1952_data * v1973_data));
                float v1978_data = s0[63];
                float v1980_data = ir6[5];
                ir6[5] = (v1980_data + (v1952_data * v1978_data));
                float v1983_data = s0[76];
                float v1985_data = ir6[6];
                ir6[6] = (v1985_data + (v1952_data * v1983_data));
                float v1988_data = s0[81];
                float v1990_data = ir6[7];
                ir6[7] = (v1990_data + (v1952_data * v1988_data));
                float v1993_data = s0[102];
                float v1995_data = ir6[8];
                ir6[8] = (v1995_data + (v1952_data * v1993_data));
                float v1998_data = s0[106];
                float v2000_data = ir6[9];
                ir6[9] = (v2000_data + (v1952_data * v1998_data));
                float v2003_data = s0[127];
                float v2005_data = ir6[10];
                ir6[10] = (v2005_data + (v1952_data * v2003_data));
                float v2008_data = s0[140];
                float v2010_data = ir6[11];
                ir6[11] = (v2010_data + (v1952_data * v2008_data));
              }
              if (v19_lead < 12) {
                float v2016_data = r5[1];
                float v2017_data = s0[1];
                float v2019_data = ir6[0];
                ir6[0] = (v2019_data + (v2016_data * v2017_data));
                float v2022_data = s0[13];
                float v2024_data = ir6[1];
                ir6[1] = (v2024_data + (v2016_data * v2022_data));
                float v2027_data = s0[24];
                float v2029_data = ir6[2];
                ir6[2] = (v2029_data + (v2016_data * v2027_data));
                float v2032_data = s0[39];
                float v2034_data = ir6[3];
                ir6[3] = (v2034_data + (v2016_data * v2032_data));
                float v2037_data = s0[50];
                float v2039_data = ir6[4];
                ir6[4] = (v2039_data + (v2016_data * v2037_data));
                float v2042_data = s0[62];
                float v2044_data = ir6[5];
                ir6[5] = (v2044_data + (v2016_data * v2042_data));
                float v2047_data = s0[77];
                float v2049_data = ir6[6];
                ir6[6] = (v2049_data + (v2016_data * v2047_data));
                float v2052_data = s0[80];
                float v2054_data = ir6[7];
                ir6[7] = (v2054_data + (v2016_data * v2052_data));
                float v2057_data = s0[103];
                float v2059_data = ir6[8];
                ir6[8] = (v2059_data + (v2016_data * v2057_data));
                float v2062_data = s0[107];
                float v2064_data = ir6[9];
                ir6[9] = (v2064_data + (v2016_data * v2062_data));
                float v2067_data = s0[126];
                float v2069_data = ir6[10];
                ir6[10] = (v2069_data + (v2016_data * v2067_data));
                float v2072_data = s0[141];
                float v2074_data = ir6[11];
                ir6[11] = (v2074_data + (v2016_data * v2072_data));
              }
              if (v19_lead < 12) {
                float v2080_data = r5[2];
                float v2081_data = s0[2];
                float v2083_data = ir6[0];
                ir6[0] = (v2083_data + (v2080_data * v2081_data));
                float v2086_data = s0[14];
                float v2088_data = ir6[1];
                ir6[1] = (v2088_data + (v2080_data * v2086_data));
                float v2091_data = s0[27];
                float v2093_data = ir6[2];
                ir6[2] = (v2093_data + (v2080_data * v2091_data));
                float v2096_data = s0[36];
                float v2098_data = ir6[3];
                ir6[3] = (v2098_data + (v2080_data * v2096_data));
                float v2101_data = s0[49];
                float v2103_data = ir6[4];
                ir6[4] = (v2103_data + (v2080_data * v2101_data));
                float v2106_data = s0[61];
                float v2108_data = ir6[5];
                ir6[5] = (v2108_data + (v2080_data * v2106_data));
                float v2111_data = s0[78];
                float v2113_data = ir6[6];
                ir6[6] = (v2113_data + (v2080_data * v2111_data));
                float v2116_data = s0[83];
                float v2118_data = ir6[7];
                ir6[7] = (v2118_data + (v2080_data * v2116_data));
                float v2121_data = s0[100];
                float v2123_data = ir6[8];
                ir6[8] = (v2123_data + (v2080_data * v2121_data));
                float v2126_data = s0[104];
                float v2128_data = ir6[9];
                ir6[9] = (v2128_data + (v2080_data * v2126_data));
                float v2131_data = s0[125];
                float v2133_data = ir6[10];
                ir6[10] = (v2133_data + (v2080_data * v2131_data));
                float v2136_data = s0[142];
                float v2138_data = ir6[11];
                ir6[11] = (v2138_data + (v2080_data * v2136_data));
              }
              if (v19_lead < 12) {
                float v2144_data = r5[3];
                float v2145_data = s0[3];
                float v2147_data = ir6[0];
                ir6[0] = (v2147_data + (v2144_data * v2145_data));
                float v2150_data = s0[15];
                float v2152_data = ir6[1];
                ir6[1] = (v2152_data + (v2144_data * v2150_data));
                float v2155_data = s0[26];
                float v2157_data = ir6[2];
                ir6[2] = (v2157_data + (v2144_data * v2155_data));
                float v2160_data = s0[37];
                float v2162_data = ir6[3];
                ir6[3] = (v2162_data + (v2144_data * v2160_data));
                float v2165_data = s0[48];
                float v2167_data = ir6[4];
                ir6[4] = (v2167_data + (v2144_data * v2165_data));
                float v2170_data = s0[60];
                float v2172_data = ir6[5];
                ir6[5] = (v2172_data + (v2144_data * v2170_data));
                float v2175_data = s0[79];
                float v2177_data = ir6[6];
                ir6[6] = (v2177_data + (v2144_data * v2175_data));
                float v2180_data = s0[82];
                float v2182_data = ir6[7];
                ir6[7] = (v2182_data + (v2144_data * v2180_data));
                float v2185_data = s0[101];
                float v2187_data = ir6[8];
                ir6[8] = (v2187_data + (v2144_data * v2185_data));
                float v2190_data = s0[105];
                float v2192_data = ir6[9];
                ir6[9] = (v2192_data + (v2144_data * v2190_data));
                float v2195_data = s0[124];
                float v2197_data = ir6[10];
                ir6[10] = (v2197_data + (v2144_data * v2195_data));
                float v2200_data = s0[143];
                float v2202_data = ir6[11];
                ir6[11] = (v2202_data + (v2144_data * v2200_data));
              }
              if (v19_lead < 12) {
                float v2208_data = r5[4];
                float v2209_data = s0[4];
                float v2211_data = ir6[0];
                ir6[0] = (v2211_data + (v2208_data * v2209_data));
                float v2214_data = s0[17];
                float v2216_data = ir6[1];
                ir6[1] = (v2216_data + (v2208_data * v2214_data));
                float v2219_data = s0[29];
                float v2221_data = ir6[2];
                ir6[2] = (v2221_data + (v2208_data * v2219_data));
                float v2224_data = s0[42];
                float v2226_data = ir6[3];
                ir6[3] = (v2226_data + (v2208_data * v2224_data));
                float v2229_data = s0[55];
                float v2231_data = ir6[4];
                ir6[4] = (v2231_data + (v2208_data * v2229_data));
                float v2234_data = s0[68];
                float v2236_data = ir6[5];
                ir6[5] = (v2236_data + (v2208_data * v2234_data));
                float v2239_data = s0[72];
                float v2241_data = ir6[6];
                ir6[6] = (v2241_data + (v2208_data * v2239_data));
                float v2244_data = s0[93];
                float v2246_data = ir6[7];
                ir6[7] = (v2246_data + (v2208_data * v2244_data));
                float v2249_data = s0[98];
                float v2251_data = ir6[8];
                ir6[8] = (v2251_data + (v2208_data * v2249_data));
                float v2254_data = s0[119];
                float v2256_data = ir6[9];
                ir6[9] = (v2256_data + (v2208_data * v2254_data));
                float v2259_data = s0[123];
                float v2261_data = ir6[10];
                ir6[10] = (v2261_data + (v2208_data * v2259_data));
                float v2264_data = s0[128];
                float v2266_data = ir6[11];
                ir6[11] = (v2266_data + (v2208_data * v2264_data));
              }
              if (v19_lead < 12) {
                float v2272_data = r5[5];
                float v2273_data = s0[5];
                float v2275_data = ir6[0];
                ir6[0] = (v2275_data + (v2272_data * v2273_data));
                float v2278_data = s0[16];
                float v2280_data = ir6[1];
                ir6[1] = (v2280_data + (v2272_data * v2278_data));
                float v2283_data = s0[28];
                float v2285_data = ir6[2];
                ir6[2] = (v2285_data + (v2272_data * v2283_data));
                float v2288_data = s0[43];
                float v2290_data = ir6[3];
                ir6[3] = (v2290_data + (v2272_data * v2288_data));
                float v2293_data = s0[54];
                float v2295_data = ir6[4];
                ir6[4] = (v2295_data + (v2272_data * v2293_data));
                float v2298_data = s0[69];
                float v2300_data = ir6[5];
                ir6[5] = (v2300_data + (v2272_data * v2298_data));
                float v2303_data = s0[73];
                float v2305_data = ir6[6];
                ir6[6] = (v2305_data + (v2272_data * v2303_data));
                float v2308_data = s0[92];
                float v2310_data = ir6[7];
                ir6[7] = (v2310_data + (v2272_data * v2308_data));
                float v2313_data = s0[99];
                float v2315_data = ir6[8];
                ir6[8] = (v2315_data + (v2272_data * v2313_data));
                float v2318_data = s0[118];
                float v2320_data = ir6[9];
                ir6[9] = (v2320_data + (v2272_data * v2318_data));
                float v2323_data = s0[122];
                float v2325_data = ir6[10];
                ir6[10] = (v2325_data + (v2272_data * v2323_data));
                float v2328_data = s0[129];
                float v2330_data = ir6[11];
                ir6[11] = (v2330_data + (v2272_data * v2328_data));
              }
              if (v19_lead < 12) {
                float v2336_data = r5[6];
                float v2337_data = s0[6];
                float v2339_data = ir6[0];
                ir6[0] = (v2339_data + (v2336_data * v2337_data));
                float v2342_data = s0[19];
                float v2344_data = ir6[1];
                ir6[1] = (v2344_data + (v2336_data * v2342_data));
                float v2347_data = s0[31];
                float v2349_data = ir6[2];
                ir6[2] = (v2349_data + (v2336_data * v2347_data));
                float v2352_data = s0[40];
                float v2354_data = ir6[3];
                ir6[3] = (v2354_data + (v2336_data * v2352_data));
                float v2357_data = s0[53];
                float v2359_data = ir6[4];
                ir6[4] = (v2359_data + (v2336_data * v2357_data));
                float v2362_data = s0[70];
                float v2364_data = ir6[5];
                ir6[5] = (v2364_data + (v2336_data * v2362_data));
                float v2367_data = s0[74];
                float v2369_data = ir6[6];
                ir6[6] = (v2369_data + (v2336_data * v2367_data));
                float v2372_data = s0[95];
                float v2374_data = ir6[7];
                ir6[7] = (v2374_data + (v2336_data * v2372_data));
                float v2377_data = s0[96];
                float v2379_data = ir6[8];
                ir6[8] = (v2379_data + (v2336_data * v2377_data));
                float v2382_data = s0[117];
                float v2384_data = ir6[9];
                ir6[9] = (v2384_data + (v2336_data * v2382_data));
                float v2387_data = s0[121];
                float v2389_data = ir6[10];
                ir6[10] = (v2389_data + (v2336_data * v2387_data));
                float v2392_data = s0[130];
                float v2394_data = ir6[11];
                ir6[11] = (v2394_data + (v2336_data * v2392_data));
              }
              if (v19_lead < 12) {
                float v2400_data = r5[7];
                float v2401_data = s0[7];
                float v2403_data = ir6[0];
                ir6[0] = (v2403_data + (v2400_data * v2401_data));
                float v2406_data = s0[18];
                float v2408_data = ir6[1];
                ir6[1] = (v2408_data + (v2400_data * v2406_data));
                float v2411_data = s0[30];
                float v2413_data = ir6[2];
                ir6[2] = (v2413_data + (v2400_data * v2411_data));
                float v2416_data = s0[41];
                float v2418_data = ir6[3];
                ir6[3] = (v2418_data + (v2400_data * v2416_data));
                float v2421_data = s0[52];
                float v2423_data = ir6[4];
                ir6[4] = (v2423_data + (v2400_data * v2421_data));
                float v2426_data = s0[71];
                float v2428_data = ir6[5];
                ir6[5] = (v2428_data + (v2400_data * v2426_data));
                float v2431_data = s0[75];
                float v2433_data = ir6[6];
                ir6[6] = (v2433_data + (v2400_data * v2431_data));
                float v2436_data = s0[94];
                float v2438_data = ir6[7];
                ir6[7] = (v2438_data + (v2400_data * v2436_data));
                float v2441_data = s0[97];
                float v2443_data = ir6[8];
                ir6[8] = (v2443_data + (v2400_data * v2441_data));
                float v2446_data = s0[116];
                float v2448_data = ir6[9];
                ir6[9] = (v2448_data + (v2400_data * v2446_data));
                float v2451_data = s0[120];
                float v2453_data = ir6[10];
                ir6[10] = (v2453_data + (v2400_data * v2451_data));
                float v2456_data = s0[131];
                float v2458_data = ir6[11];
                ir6[11] = (v2458_data + (v2400_data * v2456_data));
              }
              if (v19_lead < 12) {
                float v2464_data = r5[8];
                float v2465_data = s0[8];
                float v2467_data = ir6[0];
                ir6[0] = (v2467_data + (v2464_data * v2465_data));
                float v2470_data = s0[21];
                float v2472_data = ir6[1];
                ir6[1] = (v2472_data + (v2464_data * v2470_data));
                float v2475_data = s0[34];
                float v2477_data = ir6[2];
                ir6[2] = (v2477_data + (v2464_data * v2475_data));
                float v2480_data = s0[46];
                float v2482_data = ir6[3];
                ir6[3] = (v2482_data + (v2464_data * v2480_data));
                float v2485_data = s0[59];
                float v2487_data = ir6[4];
                ir6[4] = (v2487_data + (v2464_data * v2485_data));
                float v2490_data = s0[64];
                float v2492_data = ir6[5];
                ir6[5] = (v2492_data + (v2464_data * v2490_data));
                float v2495_data = s0[85];
                float v2497_data = ir6[6];
                ir6[6] = (v2497_data + (v2464_data * v2495_data));
                float v2500_data = s0[89];
                float v2502_data = ir6[7];
                ir6[7] = (v2502_data + (v2464_data * v2500_data));
                float v2505_data = s0[110];
                float v2507_data = ir6[8];
                ir6[8] = (v2507_data + (v2464_data * v2505_data));
                float v2510_data = s0[115];
                float v2512_data = ir6[9];
                ir6[9] = (v2512_data + (v2464_data * v2510_data));
                float v2515_data = s0[136];
                float v2517_data = ir6[10];
                ir6[10] = (v2517_data + (v2464_data * v2515_data));
                float v2520_data = s0[132];
                float v2522_data = ir6[11];
                ir6[11] = (v2522_data + (v2464_data * v2520_data));
              }
              if (v19_lead < 12) {
                float v2528_data = r5[9];
                float v2529_data = s0[9];
                float v2531_data = ir6[0];
                ir6[0] = (v2531_data + (v2528_data * v2529_data));
                float v2534_data = s0[20];
                float v2536_data = ir6[1];
                ir6[1] = (v2536_data + (v2528_data * v2534_data));
                float v2539_data = s0[35];
                float v2541_data = ir6[2];
                ir6[2] = (v2541_data + (v2528_data * v2539_data));
                float v2544_data = s0[47];
                float v2546_data = ir6[3];
                ir6[3] = (v2546_data + (v2528_data * v2544_data));
                float v2549_data = s0[58];
                float v2551_data = ir6[4];
                ir6[4] = (v2551_data + (v2528_data * v2549_data));
                float v2554_data = s0[65];
                float v2556_data = ir6[5];
                ir6[5] = (v2556_data + (v2528_data * v2554_data));
                float v2559_data = s0[84];
                float v2561_data = ir6[6];
                ir6[6] = (v2561_data + (v2528_data * v2559_data));
                float v2564_data = s0[88];
                float v2566_data = ir6[7];
                ir6[7] = (v2566_data + (v2528_data * v2564_data));
                float v2569_data = s0[111];
                float v2571_data = ir6[8];
                ir6[8] = (v2571_data + (v2528_data * v2569_data));
                float v2574_data = s0[114];
                float v2576_data = ir6[9];
                ir6[9] = (v2576_data + (v2528_data * v2574_data));
                float v2579_data = s0[137];
                float v2581_data = ir6[10];
                ir6[10] = (v2581_data + (v2528_data * v2579_data));
                float v2584_data = s0[133];
                float v2586_data = ir6[11];
                ir6[11] = (v2586_data + (v2528_data * v2584_data));
              }
              if (v19_lead < 12) {
                float v2592_data = r5[10];
                float v2593_data = s0[10];
                float v2595_data = ir6[0];
                ir6[0] = (v2595_data + (v2592_data * v2593_data));
                float v2598_data = s0[23];
                float v2600_data = ir6[1];
                ir6[1] = (v2600_data + (v2592_data * v2598_data));
                float v2603_data = s0[32];
                float v2605_data = ir6[2];
                ir6[2] = (v2605_data + (v2592_data * v2603_data));
                float v2608_data = s0[44];
                float v2610_data = ir6[3];
                ir6[3] = (v2610_data + (v2592_data * v2608_data));
                float v2613_data = s0[57];
                float v2615_data = ir6[4];
                ir6[4] = (v2615_data + (v2592_data * v2613_data));
                float v2618_data = s0[66];
                float v2620_data = ir6[5];
                ir6[5] = (v2620_data + (v2592_data * v2618_data));
                float v2623_data = s0[87];
                float v2625_data = ir6[6];
                ir6[6] = (v2625_data + (v2592_data * v2623_data));
                float v2628_data = s0[91];
                float v2630_data = ir6[7];
                ir6[7] = (v2630_data + (v2592_data * v2628_data));
                float v2633_data = s0[108];
                float v2635_data = ir6[8];
                ir6[8] = (v2635_data + (v2592_data * v2633_data));
                float v2638_data = s0[113];
                float v2640_data = ir6[9];
                ir6[9] = (v2640_data + (v2592_data * v2638_data));
                float v2643_data = s0[138];
                float v2645_data = ir6[10];
                ir6[10] = (v2645_data + (v2592_data * v2643_data));
                float v2648_data = s0[134];
                float v2650_data = ir6[11];
                ir6[11] = (v2650_data + (v2592_data * v2648_data));
              }
              if (v19_lead < 12) {
                float v2656_data = r5[11];
                float v2657_data = s0[11];
                float v2659_data = ir6[0];
                ir6[0] = (v2659_data + (v2656_data * v2657_data));
                float v2662_data = s0[22];
                float v2664_data = ir6[1];
                ir6[1] = (v2664_data + (v2656_data * v2662_data));
                float v2667_data = s0[33];
                float v2669_data = ir6[2];
                ir6[2] = (v2669_data + (v2656_data * v2667_data));
                float v2672_data = s0[45];
                float v2674_data = ir6[3];
                ir6[3] = (v2674_data + (v2656_data * v2672_data));
                float v2677_data = s0[56];
                float v2679_data = ir6[4];
                ir6[4] = (v2679_data + (v2656_data * v2677_data));
                float v2682_data = s0[67];
                float v2684_data = ir6[5];
                ir6[5] = (v2684_data + (v2656_data * v2682_data));
                float v2687_data = s0[86];
                float v2689_data = ir6[6];
                ir6[6] = (v2689_data + (v2656_data * v2687_data));
                float v2692_data = s0[90];
                float v2694_data = ir6[7];
                ir6[7] = (v2694_data + (v2656_data * v2692_data));
                float v2697_data = s0[109];
                float v2699_data = ir6[8];
                ir6[8] = (v2699_data + (v2656_data * v2697_data));
                float v2702_data = s0[112];
                float v2704_data = ir6[9];
                ir6[9] = (v2704_data + (v2656_data * v2702_data));
                float v2707_data = s0[139];
                float v2709_data = ir6[10];
                ir6[10] = (v2709_data + (v2656_data * v2707_data));
                float v2712_data = s0[135];
                float v2714_data = ir6[11];
                ir6[11] = (v2714_data + (v2656_data * v2712_data));
              }
              if (v19_lead < 12) {
                #pragma unroll
                for (int32_t v2720_n1 = 0; v2720_n1 < 12; ++v2720_n1) {
                  float v2722_data = ir6[v2720_n1];
                  r6[v2720_n1] = v2722_data;
                }
              }
              // glb_m3 = store{r>g}(r6);
              if (v19_lead < 12) {
                #pragma unroll
                for (int32_t v2728_i1 = 0; v2728_i1 < 12; ++v2728_i1) {
                  float v2730_data = r6[v2728_i1];
                  glb_m3[(v19_lead + (v2728_i1 * 12))] = v2730_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}

