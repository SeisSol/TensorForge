// === base name ===
kernel_f07cbe02c10596a1

// === header ===
void launcher_kernel_f07cbe02c10596a1(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f07cbe02c10596a1(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_f07cbe02c10596a1(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  m7,  m7_extraOffset,  m8,  m8_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f07cbe02c10596a1(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 12×12(12×12) {0..12}×{0..12} strided
        // m2 12×8(12×8) {0..12}×{0..8} strided
        // m3 12×12(12×12) {0..12}×{0..12} strided
        // m4 12×8(12×8) {0..12}×{0..8} strided
        // m5 12×12(12×12) {0..12}×{0..12} strided
        // m6 12×8(12×8) {0..12}×{0..8} strided
        // m7 12×12(12×12) {0..12}×{0..12} strided
        // m8 12×8(12×8) {0..12}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m2 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m3 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m4 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m5 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m6 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m7 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m8 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v2_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v2_batchId0 < numElements0; v2_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v3_ahead1 = v2_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v5_batchId1 = (v3_ahead1 < numElements0) ? v3_ahead1 : v2_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v2_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v2_batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v2_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v2_batchId0 * 96 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v2_batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v2_batchId0 * 96 + 0 + m4_extraOffset];
              const float *const __restrict__ glb_m5 = &m5[v2_batchId0 * 144 + 0 + m5_extraOffset];
              const float *const __restrict__ glb_m6 = &m6[v2_batchId0 * 96 + 0 + m6_extraOffset];
              const float *const __restrict__ glb_m7 = &m7[v2_batchId0 * 144 + 0 + m7_extraOffset];
              const float *const __restrict__ glb_m8 = &m8[v2_batchId0 * 96 + 0 + m8_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v22_lead = item.get_local_id(0) % 16;
              if (v22_lead < 12) {
                #pragma unroll
                for (int32_t v24_i1 = 0; v24_i1 < 12; ++v24_i1) {
                  float v32_data = glb_m1[(v22_lead + (v24_i1 * 12))];
                  r0[v24_i1] = v32_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              if (v22_lead < 12) {
                #pragma unroll
                for (int32_t v39_i1 = 0; v39_i1 < 8; ++v39_i1) {
                  float v47_data = glb_m2[(v22_lead + (v39_i1 * 12))];
                  r1[v39_i1] = v47_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              if (v22_lead < 12) {
                #pragma unroll
                for (int32_t v54_i1 = 0; v54_i1 < 12; ++v54_i1) {
                  float v62_data = glb_m3[(v22_lead + (v54_i1 * 12))];
                  r3[v54_i1] = v62_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir2[8]{};
              if (v22_lead < 12) {
                float v70_data = r0[0];
                float v71_data = r1[0];
                float v74_data = ir2[0];
                ir2[0] = (v74_data + (v70_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 0))));
                float v77_data = r1[1];
                float v80_data = ir2[1];
                ir2[1] = (v80_data + (v70_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 0))));
                float v83_data = r1[2];
                float v86_data = ir2[2];
                ir2[2] = (v86_data + (v70_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 0))));
                float v89_data = r1[3];
                float v92_data = ir2[3];
                ir2[3] = (v92_data + (v70_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 0))));
                float v95_data = r1[4];
                float v98_data = ir2[4];
                ir2[4] = (v98_data + (v70_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 0))));
                float v101_data = r1[5];
                float v104_data = ir2[5];
                ir2[5] = (v104_data + (v70_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 0))));
                float v107_data = r1[6];
                float v110_data = ir2[6];
                ir2[6] = (v110_data + (v70_data * (sycl::group_broadcast(item.get_sub_group(), v107_data, 0))));
                float v113_data = r1[7];
                float v116_data = ir2[7];
                ir2[7] = (v116_data + (v70_data * (sycl::group_broadcast(item.get_sub_group(), v113_data, 0))));
              }
              if (v22_lead < 12) {
                float v122_data = r0[1];
                float v123_data = r1[0];
                float v126_data = ir2[0];
                ir2[0] = (v126_data + (v122_data * (sycl::group_broadcast(item.get_sub_group(), v123_data, 1))));
                float v129_data = r1[1];
                float v132_data = ir2[1];
                ir2[1] = (v132_data + (v122_data * (sycl::group_broadcast(item.get_sub_group(), v129_data, 1))));
                float v135_data = r1[2];
                float v138_data = ir2[2];
                ir2[2] = (v138_data + (v122_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 1))));
                float v141_data = r1[3];
                float v144_data = ir2[3];
                ir2[3] = (v144_data + (v122_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 1))));
                float v147_data = r1[4];
                float v150_data = ir2[4];
                ir2[4] = (v150_data + (v122_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 1))));
                float v153_data = r1[5];
                float v156_data = ir2[5];
                ir2[5] = (v156_data + (v122_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 1))));
                float v159_data = r1[6];
                float v162_data = ir2[6];
                ir2[6] = (v162_data + (v122_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 1))));
                float v165_data = r1[7];
                float v168_data = ir2[7];
                ir2[7] = (v168_data + (v122_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 1))));
              }
              if (v22_lead < 12) {
                float v174_data = r0[2];
                float v175_data = r1[0];
                float v178_data = ir2[0];
                ir2[0] = (v178_data + (v174_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 2))));
                float v181_data = r1[1];
                float v184_data = ir2[1];
                ir2[1] = (v184_data + (v174_data * (sycl::group_broadcast(item.get_sub_group(), v181_data, 2))));
                float v187_data = r1[2];
                float v190_data = ir2[2];
                ir2[2] = (v190_data + (v174_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 2))));
                float v193_data = r1[3];
                float v196_data = ir2[3];
                ir2[3] = (v196_data + (v174_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 2))));
                float v199_data = r1[4];
                float v202_data = ir2[4];
                ir2[4] = (v202_data + (v174_data * (sycl::group_broadcast(item.get_sub_group(), v199_data, 2))));
                float v205_data = r1[5];
                float v208_data = ir2[5];
                ir2[5] = (v208_data + (v174_data * (sycl::group_broadcast(item.get_sub_group(), v205_data, 2))));
                float v211_data = r1[6];
                float v214_data = ir2[6];
                ir2[6] = (v214_data + (v174_data * (sycl::group_broadcast(item.get_sub_group(), v211_data, 2))));
                float v217_data = r1[7];
                float v220_data = ir2[7];
                ir2[7] = (v220_data + (v174_data * (sycl::group_broadcast(item.get_sub_group(), v217_data, 2))));
              }
              if (v22_lead < 12) {
                float v226_data = r0[3];
                float v227_data = r1[0];
                float v230_data = ir2[0];
                ir2[0] = (v230_data + (v226_data * (sycl::group_broadcast(item.get_sub_group(), v227_data, 3))));
                float v233_data = r1[1];
                float v236_data = ir2[1];
                ir2[1] = (v236_data + (v226_data * (sycl::group_broadcast(item.get_sub_group(), v233_data, 3))));
                float v239_data = r1[2];
                float v242_data = ir2[2];
                ir2[2] = (v242_data + (v226_data * (sycl::group_broadcast(item.get_sub_group(), v239_data, 3))));
                float v245_data = r1[3];
                float v248_data = ir2[3];
                ir2[3] = (v248_data + (v226_data * (sycl::group_broadcast(item.get_sub_group(), v245_data, 3))));
                float v251_data = r1[4];
                float v254_data = ir2[4];
                ir2[4] = (v254_data + (v226_data * (sycl::group_broadcast(item.get_sub_group(), v251_data, 3))));
                float v257_data = r1[5];
                float v260_data = ir2[5];
                ir2[5] = (v260_data + (v226_data * (sycl::group_broadcast(item.get_sub_group(), v257_data, 3))));
                float v263_data = r1[6];
                float v266_data = ir2[6];
                ir2[6] = (v266_data + (v226_data * (sycl::group_broadcast(item.get_sub_group(), v263_data, 3))));
                float v269_data = r1[7];
                float v272_data = ir2[7];
                ir2[7] = (v272_data + (v226_data * (sycl::group_broadcast(item.get_sub_group(), v269_data, 3))));
              }
              if (v22_lead < 12) {
                float v278_data = r0[4];
                float v279_data = r1[0];
                float v282_data = ir2[0];
                ir2[0] = (v282_data + (v278_data * (sycl::group_broadcast(item.get_sub_group(), v279_data, 4))));
                float v285_data = r1[1];
                float v288_data = ir2[1];
                ir2[1] = (v288_data + (v278_data * (sycl::group_broadcast(item.get_sub_group(), v285_data, 4))));
                float v291_data = r1[2];
                float v294_data = ir2[2];
                ir2[2] = (v294_data + (v278_data * (sycl::group_broadcast(item.get_sub_group(), v291_data, 4))));
                float v297_data = r1[3];
                float v300_data = ir2[3];
                ir2[3] = (v300_data + (v278_data * (sycl::group_broadcast(item.get_sub_group(), v297_data, 4))));
                float v303_data = r1[4];
                float v306_data = ir2[4];
                ir2[4] = (v306_data + (v278_data * (sycl::group_broadcast(item.get_sub_group(), v303_data, 4))));
                float v309_data = r1[5];
                float v312_data = ir2[5];
                ir2[5] = (v312_data + (v278_data * (sycl::group_broadcast(item.get_sub_group(), v309_data, 4))));
                float v315_data = r1[6];
                float v318_data = ir2[6];
                ir2[6] = (v318_data + (v278_data * (sycl::group_broadcast(item.get_sub_group(), v315_data, 4))));
                float v321_data = r1[7];
                float v324_data = ir2[7];
                ir2[7] = (v324_data + (v278_data * (sycl::group_broadcast(item.get_sub_group(), v321_data, 4))));
              }
              if (v22_lead < 12) {
                float v330_data = r0[5];
                float v331_data = r1[0];
                float v334_data = ir2[0];
                ir2[0] = (v334_data + (v330_data * (sycl::group_broadcast(item.get_sub_group(), v331_data, 5))));
                float v337_data = r1[1];
                float v340_data = ir2[1];
                ir2[1] = (v340_data + (v330_data * (sycl::group_broadcast(item.get_sub_group(), v337_data, 5))));
                float v343_data = r1[2];
                float v346_data = ir2[2];
                ir2[2] = (v346_data + (v330_data * (sycl::group_broadcast(item.get_sub_group(), v343_data, 5))));
                float v349_data = r1[3];
                float v352_data = ir2[3];
                ir2[3] = (v352_data + (v330_data * (sycl::group_broadcast(item.get_sub_group(), v349_data, 5))));
                float v355_data = r1[4];
                float v358_data = ir2[4];
                ir2[4] = (v358_data + (v330_data * (sycl::group_broadcast(item.get_sub_group(), v355_data, 5))));
                float v361_data = r1[5];
                float v364_data = ir2[5];
                ir2[5] = (v364_data + (v330_data * (sycl::group_broadcast(item.get_sub_group(), v361_data, 5))));
                float v367_data = r1[6];
                float v370_data = ir2[6];
                ir2[6] = (v370_data + (v330_data * (sycl::group_broadcast(item.get_sub_group(), v367_data, 5))));
                float v373_data = r1[7];
                float v376_data = ir2[7];
                ir2[7] = (v376_data + (v330_data * (sycl::group_broadcast(item.get_sub_group(), v373_data, 5))));
              }
              if (v22_lead < 12) {
                float v382_data = r0[6];
                float v383_data = r1[0];
                float v386_data = ir2[0];
                ir2[0] = (v386_data + (v382_data * (sycl::group_broadcast(item.get_sub_group(), v383_data, 6))));
                float v389_data = r1[1];
                float v392_data = ir2[1];
                ir2[1] = (v392_data + (v382_data * (sycl::group_broadcast(item.get_sub_group(), v389_data, 6))));
                float v395_data = r1[2];
                float v398_data = ir2[2];
                ir2[2] = (v398_data + (v382_data * (sycl::group_broadcast(item.get_sub_group(), v395_data, 6))));
                float v401_data = r1[3];
                float v404_data = ir2[3];
                ir2[3] = (v404_data + (v382_data * (sycl::group_broadcast(item.get_sub_group(), v401_data, 6))));
                float v407_data = r1[4];
                float v410_data = ir2[4];
                ir2[4] = (v410_data + (v382_data * (sycl::group_broadcast(item.get_sub_group(), v407_data, 6))));
                float v413_data = r1[5];
                float v416_data = ir2[5];
                ir2[5] = (v416_data + (v382_data * (sycl::group_broadcast(item.get_sub_group(), v413_data, 6))));
                float v419_data = r1[6];
                float v422_data = ir2[6];
                ir2[6] = (v422_data + (v382_data * (sycl::group_broadcast(item.get_sub_group(), v419_data, 6))));
                float v425_data = r1[7];
                float v428_data = ir2[7];
                ir2[7] = (v428_data + (v382_data * (sycl::group_broadcast(item.get_sub_group(), v425_data, 6))));
              }
              if (v22_lead < 12) {
                float v434_data = r0[7];
                float v435_data = r1[0];
                float v438_data = ir2[0];
                ir2[0] = (v438_data + (v434_data * (sycl::group_broadcast(item.get_sub_group(), v435_data, 7))));
                float v441_data = r1[1];
                float v444_data = ir2[1];
                ir2[1] = (v444_data + (v434_data * (sycl::group_broadcast(item.get_sub_group(), v441_data, 7))));
                float v447_data = r1[2];
                float v450_data = ir2[2];
                ir2[2] = (v450_data + (v434_data * (sycl::group_broadcast(item.get_sub_group(), v447_data, 7))));
                float v453_data = r1[3];
                float v456_data = ir2[3];
                ir2[3] = (v456_data + (v434_data * (sycl::group_broadcast(item.get_sub_group(), v453_data, 7))));
                float v459_data = r1[4];
                float v462_data = ir2[4];
                ir2[4] = (v462_data + (v434_data * (sycl::group_broadcast(item.get_sub_group(), v459_data, 7))));
                float v465_data = r1[5];
                float v468_data = ir2[5];
                ir2[5] = (v468_data + (v434_data * (sycl::group_broadcast(item.get_sub_group(), v465_data, 7))));
                float v471_data = r1[6];
                float v474_data = ir2[6];
                ir2[6] = (v474_data + (v434_data * (sycl::group_broadcast(item.get_sub_group(), v471_data, 7))));
                float v477_data = r1[7];
                float v480_data = ir2[7];
                ir2[7] = (v480_data + (v434_data * (sycl::group_broadcast(item.get_sub_group(), v477_data, 7))));
              }
              if (v22_lead < 12) {
                float v486_data = r0[8];
                float v487_data = r1[0];
                float v490_data = ir2[0];
                ir2[0] = (v490_data + (v486_data * (sycl::group_broadcast(item.get_sub_group(), v487_data, 8))));
                float v493_data = r1[1];
                float v496_data = ir2[1];
                ir2[1] = (v496_data + (v486_data * (sycl::group_broadcast(item.get_sub_group(), v493_data, 8))));
                float v499_data = r1[2];
                float v502_data = ir2[2];
                ir2[2] = (v502_data + (v486_data * (sycl::group_broadcast(item.get_sub_group(), v499_data, 8))));
                float v505_data = r1[3];
                float v508_data = ir2[3];
                ir2[3] = (v508_data + (v486_data * (sycl::group_broadcast(item.get_sub_group(), v505_data, 8))));
                float v511_data = r1[4];
                float v514_data = ir2[4];
                ir2[4] = (v514_data + (v486_data * (sycl::group_broadcast(item.get_sub_group(), v511_data, 8))));
                float v517_data = r1[5];
                float v520_data = ir2[5];
                ir2[5] = (v520_data + (v486_data * (sycl::group_broadcast(item.get_sub_group(), v517_data, 8))));
                float v523_data = r1[6];
                float v526_data = ir2[6];
                ir2[6] = (v526_data + (v486_data * (sycl::group_broadcast(item.get_sub_group(), v523_data, 8))));
                float v529_data = r1[7];
                float v532_data = ir2[7];
                ir2[7] = (v532_data + (v486_data * (sycl::group_broadcast(item.get_sub_group(), v529_data, 8))));
              }
              if (v22_lead < 12) {
                float v538_data = r0[9];
                float v539_data = r1[0];
                float v542_data = ir2[0];
                ir2[0] = (v542_data + (v538_data * (sycl::group_broadcast(item.get_sub_group(), v539_data, 9))));
                float v545_data = r1[1];
                float v548_data = ir2[1];
                ir2[1] = (v548_data + (v538_data * (sycl::group_broadcast(item.get_sub_group(), v545_data, 9))));
                float v551_data = r1[2];
                float v554_data = ir2[2];
                ir2[2] = (v554_data + (v538_data * (sycl::group_broadcast(item.get_sub_group(), v551_data, 9))));
                float v557_data = r1[3];
                float v560_data = ir2[3];
                ir2[3] = (v560_data + (v538_data * (sycl::group_broadcast(item.get_sub_group(), v557_data, 9))));
                float v563_data = r1[4];
                float v566_data = ir2[4];
                ir2[4] = (v566_data + (v538_data * (sycl::group_broadcast(item.get_sub_group(), v563_data, 9))));
                float v569_data = r1[5];
                float v572_data = ir2[5];
                ir2[5] = (v572_data + (v538_data * (sycl::group_broadcast(item.get_sub_group(), v569_data, 9))));
                float v575_data = r1[6];
                float v578_data = ir2[6];
                ir2[6] = (v578_data + (v538_data * (sycl::group_broadcast(item.get_sub_group(), v575_data, 9))));
                float v581_data = r1[7];
                float v584_data = ir2[7];
                ir2[7] = (v584_data + (v538_data * (sycl::group_broadcast(item.get_sub_group(), v581_data, 9))));
              }
              if (v22_lead < 12) {
                float v590_data = r0[10];
                float v591_data = r1[0];
                float v594_data = ir2[0];
                ir2[0] = (v594_data + (v590_data * (sycl::group_broadcast(item.get_sub_group(), v591_data, 10))));
                float v597_data = r1[1];
                float v600_data = ir2[1];
                ir2[1] = (v600_data + (v590_data * (sycl::group_broadcast(item.get_sub_group(), v597_data, 10))));
                float v603_data = r1[2];
                float v606_data = ir2[2];
                ir2[2] = (v606_data + (v590_data * (sycl::group_broadcast(item.get_sub_group(), v603_data, 10))));
                float v609_data = r1[3];
                float v612_data = ir2[3];
                ir2[3] = (v612_data + (v590_data * (sycl::group_broadcast(item.get_sub_group(), v609_data, 10))));
                float v615_data = r1[4];
                float v618_data = ir2[4];
                ir2[4] = (v618_data + (v590_data * (sycl::group_broadcast(item.get_sub_group(), v615_data, 10))));
                float v621_data = r1[5];
                float v624_data = ir2[5];
                ir2[5] = (v624_data + (v590_data * (sycl::group_broadcast(item.get_sub_group(), v621_data, 10))));
                float v627_data = r1[6];
                float v630_data = ir2[6];
                ir2[6] = (v630_data + (v590_data * (sycl::group_broadcast(item.get_sub_group(), v627_data, 10))));
                float v633_data = r1[7];
                float v636_data = ir2[7];
                ir2[7] = (v636_data + (v590_data * (sycl::group_broadcast(item.get_sub_group(), v633_data, 10))));
              }
              if (v22_lead < 12) {
                float v642_data = r0[11];
                float v643_data = r1[0];
                float v646_data = ir2[0];
                ir2[0] = (v646_data + (v642_data * (sycl::group_broadcast(item.get_sub_group(), v643_data, 11))));
                float v649_data = r1[1];
                float v652_data = ir2[1];
                ir2[1] = (v652_data + (v642_data * (sycl::group_broadcast(item.get_sub_group(), v649_data, 11))));
                float v655_data = r1[2];
                float v658_data = ir2[2];
                ir2[2] = (v658_data + (v642_data * (sycl::group_broadcast(item.get_sub_group(), v655_data, 11))));
                float v661_data = r1[3];
                float v664_data = ir2[3];
                ir2[3] = (v664_data + (v642_data * (sycl::group_broadcast(item.get_sub_group(), v661_data, 11))));
                float v667_data = r1[4];
                float v670_data = ir2[4];
                ir2[4] = (v670_data + (v642_data * (sycl::group_broadcast(item.get_sub_group(), v667_data, 11))));
                float v673_data = r1[5];
                float v676_data = ir2[5];
                ir2[5] = (v676_data + (v642_data * (sycl::group_broadcast(item.get_sub_group(), v673_data, 11))));
                float v679_data = r1[6];
                float v682_data = ir2[6];
                ir2[6] = (v682_data + (v642_data * (sycl::group_broadcast(item.get_sub_group(), v679_data, 11))));
                float v685_data = r1[7];
                float v688_data = ir2[7];
                ir2[7] = (v688_data + (v642_data * (sycl::group_broadcast(item.get_sub_group(), v685_data, 11))));
              }
              if (v22_lead < 12) {
                #pragma unroll
                for (int32_t v694_n1 = 0; v694_n1 < 8; ++v694_n1) {
                  float v696_data = ir2[v694_n1];
                  r2[v694_n1] = v696_data;
                }
              }
              float r4[8]{};
              // r4 = load{g>r}(glb_m4);
              if (v22_lead < 12) {
                #pragma unroll
                for (int32_t v703_i1 = 0; v703_i1 < 8; ++v703_i1) {
                  float v711_data = glb_m4[(v22_lead + (v703_i1 * 12))];
                  r4[v703_i1] = v711_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m3););
              float r6[12]{};
              // r6 = load{g>r}(glb_m5);
              if (v22_lead < 12) {
                #pragma unroll
                for (int32_t v718_i1 = 0; v718_i1 < 12; ++v718_i1) {
                  float v726_data = glb_m5[(v22_lead + (v718_i1 * 12))];
                  r6[v718_i1] = v726_data;
                }
              }
              // wait(r4 = load{g>r}(glb_m4););
              float r5[8]{};
              // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir5[8]{};
              if (v22_lead < 12) {
                float v734_data = r3[0];
                float v735_data = r4[0];
                float v738_data = ir5[0];
                ir5[0] = (v738_data + (v734_data * (sycl::group_broadcast(item.get_sub_group(), v735_data, 0))));
                float v741_data = r4[1];
                float v744_data = ir5[1];
                ir5[1] = (v744_data + (v734_data * (sycl::group_broadcast(item.get_sub_group(), v741_data, 0))));
                float v747_data = r4[2];
                float v750_data = ir5[2];
                ir5[2] = (v750_data + (v734_data * (sycl::group_broadcast(item.get_sub_group(), v747_data, 0))));
                float v753_data = r4[3];
                float v756_data = ir5[3];
                ir5[3] = (v756_data + (v734_data * (sycl::group_broadcast(item.get_sub_group(), v753_data, 0))));
                float v759_data = r4[4];
                float v762_data = ir5[4];
                ir5[4] = (v762_data + (v734_data * (sycl::group_broadcast(item.get_sub_group(), v759_data, 0))));
                float v765_data = r4[5];
                float v768_data = ir5[5];
                ir5[5] = (v768_data + (v734_data * (sycl::group_broadcast(item.get_sub_group(), v765_data, 0))));
                float v771_data = r4[6];
                float v774_data = ir5[6];
                ir5[6] = (v774_data + (v734_data * (sycl::group_broadcast(item.get_sub_group(), v771_data, 0))));
                float v777_data = r4[7];
                float v780_data = ir5[7];
                ir5[7] = (v780_data + (v734_data * (sycl::group_broadcast(item.get_sub_group(), v777_data, 0))));
              }
              if (v22_lead < 12) {
                float v786_data = r3[1];
                float v787_data = r4[0];
                float v790_data = ir5[0];
                ir5[0] = (v790_data + (v786_data * (sycl::group_broadcast(item.get_sub_group(), v787_data, 1))));
                float v793_data = r4[1];
                float v796_data = ir5[1];
                ir5[1] = (v796_data + (v786_data * (sycl::group_broadcast(item.get_sub_group(), v793_data, 1))));
                float v799_data = r4[2];
                float v802_data = ir5[2];
                ir5[2] = (v802_data + (v786_data * (sycl::group_broadcast(item.get_sub_group(), v799_data, 1))));
                float v805_data = r4[3];
                float v808_data = ir5[3];
                ir5[3] = (v808_data + (v786_data * (sycl::group_broadcast(item.get_sub_group(), v805_data, 1))));
                float v811_data = r4[4];
                float v814_data = ir5[4];
                ir5[4] = (v814_data + (v786_data * (sycl::group_broadcast(item.get_sub_group(), v811_data, 1))));
                float v817_data = r4[5];
                float v820_data = ir5[5];
                ir5[5] = (v820_data + (v786_data * (sycl::group_broadcast(item.get_sub_group(), v817_data, 1))));
                float v823_data = r4[6];
                float v826_data = ir5[6];
                ir5[6] = (v826_data + (v786_data * (sycl::group_broadcast(item.get_sub_group(), v823_data, 1))));
                float v829_data = r4[7];
                float v832_data = ir5[7];
                ir5[7] = (v832_data + (v786_data * (sycl::group_broadcast(item.get_sub_group(), v829_data, 1))));
              }
              if (v22_lead < 12) {
                float v838_data = r3[2];
                float v839_data = r4[0];
                float v842_data = ir5[0];
                ir5[0] = (v842_data + (v838_data * (sycl::group_broadcast(item.get_sub_group(), v839_data, 2))));
                float v845_data = r4[1];
                float v848_data = ir5[1];
                ir5[1] = (v848_data + (v838_data * (sycl::group_broadcast(item.get_sub_group(), v845_data, 2))));
                float v851_data = r4[2];
                float v854_data = ir5[2];
                ir5[2] = (v854_data + (v838_data * (sycl::group_broadcast(item.get_sub_group(), v851_data, 2))));
                float v857_data = r4[3];
                float v860_data = ir5[3];
                ir5[3] = (v860_data + (v838_data * (sycl::group_broadcast(item.get_sub_group(), v857_data, 2))));
                float v863_data = r4[4];
                float v866_data = ir5[4];
                ir5[4] = (v866_data + (v838_data * (sycl::group_broadcast(item.get_sub_group(), v863_data, 2))));
                float v869_data = r4[5];
                float v872_data = ir5[5];
                ir5[5] = (v872_data + (v838_data * (sycl::group_broadcast(item.get_sub_group(), v869_data, 2))));
                float v875_data = r4[6];
                float v878_data = ir5[6];
                ir5[6] = (v878_data + (v838_data * (sycl::group_broadcast(item.get_sub_group(), v875_data, 2))));
                float v881_data = r4[7];
                float v884_data = ir5[7];
                ir5[7] = (v884_data + (v838_data * (sycl::group_broadcast(item.get_sub_group(), v881_data, 2))));
              }
              if (v22_lead < 12) {
                float v890_data = r3[3];
                float v891_data = r4[0];
                float v894_data = ir5[0];
                ir5[0] = (v894_data + (v890_data * (sycl::group_broadcast(item.get_sub_group(), v891_data, 3))));
                float v897_data = r4[1];
                float v900_data = ir5[1];
                ir5[1] = (v900_data + (v890_data * (sycl::group_broadcast(item.get_sub_group(), v897_data, 3))));
                float v903_data = r4[2];
                float v906_data = ir5[2];
                ir5[2] = (v906_data + (v890_data * (sycl::group_broadcast(item.get_sub_group(), v903_data, 3))));
                float v909_data = r4[3];
                float v912_data = ir5[3];
                ir5[3] = (v912_data + (v890_data * (sycl::group_broadcast(item.get_sub_group(), v909_data, 3))));
                float v915_data = r4[4];
                float v918_data = ir5[4];
                ir5[4] = (v918_data + (v890_data * (sycl::group_broadcast(item.get_sub_group(), v915_data, 3))));
                float v921_data = r4[5];
                float v924_data = ir5[5];
                ir5[5] = (v924_data + (v890_data * (sycl::group_broadcast(item.get_sub_group(), v921_data, 3))));
                float v927_data = r4[6];
                float v930_data = ir5[6];
                ir5[6] = (v930_data + (v890_data * (sycl::group_broadcast(item.get_sub_group(), v927_data, 3))));
                float v933_data = r4[7];
                float v936_data = ir5[7];
                ir5[7] = (v936_data + (v890_data * (sycl::group_broadcast(item.get_sub_group(), v933_data, 3))));
              }
              if (v22_lead < 12) {
                float v942_data = r3[4];
                float v943_data = r4[0];
                float v946_data = ir5[0];
                ir5[0] = (v946_data + (v942_data * (sycl::group_broadcast(item.get_sub_group(), v943_data, 4))));
                float v949_data = r4[1];
                float v952_data = ir5[1];
                ir5[1] = (v952_data + (v942_data * (sycl::group_broadcast(item.get_sub_group(), v949_data, 4))));
                float v955_data = r4[2];
                float v958_data = ir5[2];
                ir5[2] = (v958_data + (v942_data * (sycl::group_broadcast(item.get_sub_group(), v955_data, 4))));
                float v961_data = r4[3];
                float v964_data = ir5[3];
                ir5[3] = (v964_data + (v942_data * (sycl::group_broadcast(item.get_sub_group(), v961_data, 4))));
                float v967_data = r4[4];
                float v970_data = ir5[4];
                ir5[4] = (v970_data + (v942_data * (sycl::group_broadcast(item.get_sub_group(), v967_data, 4))));
                float v973_data = r4[5];
                float v976_data = ir5[5];
                ir5[5] = (v976_data + (v942_data * (sycl::group_broadcast(item.get_sub_group(), v973_data, 4))));
                float v979_data = r4[6];
                float v982_data = ir5[6];
                ir5[6] = (v982_data + (v942_data * (sycl::group_broadcast(item.get_sub_group(), v979_data, 4))));
                float v985_data = r4[7];
                float v988_data = ir5[7];
                ir5[7] = (v988_data + (v942_data * (sycl::group_broadcast(item.get_sub_group(), v985_data, 4))));
              }
              if (v22_lead < 12) {
                float v994_data = r3[5];
                float v995_data = r4[0];
                float v998_data = ir5[0];
                ir5[0] = (v998_data + (v994_data * (sycl::group_broadcast(item.get_sub_group(), v995_data, 5))));
                float v1001_data = r4[1];
                float v1004_data = ir5[1];
                ir5[1] = (v1004_data + (v994_data * (sycl::group_broadcast(item.get_sub_group(), v1001_data, 5))));
                float v1007_data = r4[2];
                float v1010_data = ir5[2];
                ir5[2] = (v1010_data + (v994_data * (sycl::group_broadcast(item.get_sub_group(), v1007_data, 5))));
                float v1013_data = r4[3];
                float v1016_data = ir5[3];
                ir5[3] = (v1016_data + (v994_data * (sycl::group_broadcast(item.get_sub_group(), v1013_data, 5))));
                float v1019_data = r4[4];
                float v1022_data = ir5[4];
                ir5[4] = (v1022_data + (v994_data * (sycl::group_broadcast(item.get_sub_group(), v1019_data, 5))));
                float v1025_data = r4[5];
                float v1028_data = ir5[5];
                ir5[5] = (v1028_data + (v994_data * (sycl::group_broadcast(item.get_sub_group(), v1025_data, 5))));
                float v1031_data = r4[6];
                float v1034_data = ir5[6];
                ir5[6] = (v1034_data + (v994_data * (sycl::group_broadcast(item.get_sub_group(), v1031_data, 5))));
                float v1037_data = r4[7];
                float v1040_data = ir5[7];
                ir5[7] = (v1040_data + (v994_data * (sycl::group_broadcast(item.get_sub_group(), v1037_data, 5))));
              }
              if (v22_lead < 12) {
                float v1046_data = r3[6];
                float v1047_data = r4[0];
                float v1050_data = ir5[0];
                ir5[0] = (v1050_data + (v1046_data * (sycl::group_broadcast(item.get_sub_group(), v1047_data, 6))));
                float v1053_data = r4[1];
                float v1056_data = ir5[1];
                ir5[1] = (v1056_data + (v1046_data * (sycl::group_broadcast(item.get_sub_group(), v1053_data, 6))));
                float v1059_data = r4[2];
                float v1062_data = ir5[2];
                ir5[2] = (v1062_data + (v1046_data * (sycl::group_broadcast(item.get_sub_group(), v1059_data, 6))));
                float v1065_data = r4[3];
                float v1068_data = ir5[3];
                ir5[3] = (v1068_data + (v1046_data * (sycl::group_broadcast(item.get_sub_group(), v1065_data, 6))));
                float v1071_data = r4[4];
                float v1074_data = ir5[4];
                ir5[4] = (v1074_data + (v1046_data * (sycl::group_broadcast(item.get_sub_group(), v1071_data, 6))));
                float v1077_data = r4[5];
                float v1080_data = ir5[5];
                ir5[5] = (v1080_data + (v1046_data * (sycl::group_broadcast(item.get_sub_group(), v1077_data, 6))));
                float v1083_data = r4[6];
                float v1086_data = ir5[6];
                ir5[6] = (v1086_data + (v1046_data * (sycl::group_broadcast(item.get_sub_group(), v1083_data, 6))));
                float v1089_data = r4[7];
                float v1092_data = ir5[7];
                ir5[7] = (v1092_data + (v1046_data * (sycl::group_broadcast(item.get_sub_group(), v1089_data, 6))));
              }
              if (v22_lead < 12) {
                float v1098_data = r3[7];
                float v1099_data = r4[0];
                float v1102_data = ir5[0];
                ir5[0] = (v1102_data + (v1098_data * (sycl::group_broadcast(item.get_sub_group(), v1099_data, 7))));
                float v1105_data = r4[1];
                float v1108_data = ir5[1];
                ir5[1] = (v1108_data + (v1098_data * (sycl::group_broadcast(item.get_sub_group(), v1105_data, 7))));
                float v1111_data = r4[2];
                float v1114_data = ir5[2];
                ir5[2] = (v1114_data + (v1098_data * (sycl::group_broadcast(item.get_sub_group(), v1111_data, 7))));
                float v1117_data = r4[3];
                float v1120_data = ir5[3];
                ir5[3] = (v1120_data + (v1098_data * (sycl::group_broadcast(item.get_sub_group(), v1117_data, 7))));
                float v1123_data = r4[4];
                float v1126_data = ir5[4];
                ir5[4] = (v1126_data + (v1098_data * (sycl::group_broadcast(item.get_sub_group(), v1123_data, 7))));
                float v1129_data = r4[5];
                float v1132_data = ir5[5];
                ir5[5] = (v1132_data + (v1098_data * (sycl::group_broadcast(item.get_sub_group(), v1129_data, 7))));
                float v1135_data = r4[6];
                float v1138_data = ir5[6];
                ir5[6] = (v1138_data + (v1098_data * (sycl::group_broadcast(item.get_sub_group(), v1135_data, 7))));
                float v1141_data = r4[7];
                float v1144_data = ir5[7];
                ir5[7] = (v1144_data + (v1098_data * (sycl::group_broadcast(item.get_sub_group(), v1141_data, 7))));
              }
              if (v22_lead < 12) {
                float v1150_data = r3[8];
                float v1151_data = r4[0];
                float v1154_data = ir5[0];
                ir5[0] = (v1154_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v1151_data, 8))));
                float v1157_data = r4[1];
                float v1160_data = ir5[1];
                ir5[1] = (v1160_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v1157_data, 8))));
                float v1163_data = r4[2];
                float v1166_data = ir5[2];
                ir5[2] = (v1166_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v1163_data, 8))));
                float v1169_data = r4[3];
                float v1172_data = ir5[3];
                ir5[3] = (v1172_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v1169_data, 8))));
                float v1175_data = r4[4];
                float v1178_data = ir5[4];
                ir5[4] = (v1178_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v1175_data, 8))));
                float v1181_data = r4[5];
                float v1184_data = ir5[5];
                ir5[5] = (v1184_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v1181_data, 8))));
                float v1187_data = r4[6];
                float v1190_data = ir5[6];
                ir5[6] = (v1190_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v1187_data, 8))));
                float v1193_data = r4[7];
                float v1196_data = ir5[7];
                ir5[7] = (v1196_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v1193_data, 8))));
              }
              if (v22_lead < 12) {
                float v1202_data = r3[9];
                float v1203_data = r4[0];
                float v1206_data = ir5[0];
                ir5[0] = (v1206_data + (v1202_data * (sycl::group_broadcast(item.get_sub_group(), v1203_data, 9))));
                float v1209_data = r4[1];
                float v1212_data = ir5[1];
                ir5[1] = (v1212_data + (v1202_data * (sycl::group_broadcast(item.get_sub_group(), v1209_data, 9))));
                float v1215_data = r4[2];
                float v1218_data = ir5[2];
                ir5[2] = (v1218_data + (v1202_data * (sycl::group_broadcast(item.get_sub_group(), v1215_data, 9))));
                float v1221_data = r4[3];
                float v1224_data = ir5[3];
                ir5[3] = (v1224_data + (v1202_data * (sycl::group_broadcast(item.get_sub_group(), v1221_data, 9))));
                float v1227_data = r4[4];
                float v1230_data = ir5[4];
                ir5[4] = (v1230_data + (v1202_data * (sycl::group_broadcast(item.get_sub_group(), v1227_data, 9))));
                float v1233_data = r4[5];
                float v1236_data = ir5[5];
                ir5[5] = (v1236_data + (v1202_data * (sycl::group_broadcast(item.get_sub_group(), v1233_data, 9))));
                float v1239_data = r4[6];
                float v1242_data = ir5[6];
                ir5[6] = (v1242_data + (v1202_data * (sycl::group_broadcast(item.get_sub_group(), v1239_data, 9))));
                float v1245_data = r4[7];
                float v1248_data = ir5[7];
                ir5[7] = (v1248_data + (v1202_data * (sycl::group_broadcast(item.get_sub_group(), v1245_data, 9))));
              }
              if (v22_lead < 12) {
                float v1254_data = r3[10];
                float v1255_data = r4[0];
                float v1258_data = ir5[0];
                ir5[0] = (v1258_data + (v1254_data * (sycl::group_broadcast(item.get_sub_group(), v1255_data, 10))));
                float v1261_data = r4[1];
                float v1264_data = ir5[1];
                ir5[1] = (v1264_data + (v1254_data * (sycl::group_broadcast(item.get_sub_group(), v1261_data, 10))));
                float v1267_data = r4[2];
                float v1270_data = ir5[2];
                ir5[2] = (v1270_data + (v1254_data * (sycl::group_broadcast(item.get_sub_group(), v1267_data, 10))));
                float v1273_data = r4[3];
                float v1276_data = ir5[3];
                ir5[3] = (v1276_data + (v1254_data * (sycl::group_broadcast(item.get_sub_group(), v1273_data, 10))));
                float v1279_data = r4[4];
                float v1282_data = ir5[4];
                ir5[4] = (v1282_data + (v1254_data * (sycl::group_broadcast(item.get_sub_group(), v1279_data, 10))));
                float v1285_data = r4[5];
                float v1288_data = ir5[5];
                ir5[5] = (v1288_data + (v1254_data * (sycl::group_broadcast(item.get_sub_group(), v1285_data, 10))));
                float v1291_data = r4[6];
                float v1294_data = ir5[6];
                ir5[6] = (v1294_data + (v1254_data * (sycl::group_broadcast(item.get_sub_group(), v1291_data, 10))));
                float v1297_data = r4[7];
                float v1300_data = ir5[7];
                ir5[7] = (v1300_data + (v1254_data * (sycl::group_broadcast(item.get_sub_group(), v1297_data, 10))));
              }
              if (v22_lead < 12) {
                float v1306_data = r3[11];
                float v1307_data = r4[0];
                float v1310_data = ir5[0];
                ir5[0] = (v1310_data + (v1306_data * (sycl::group_broadcast(item.get_sub_group(), v1307_data, 11))));
                float v1313_data = r4[1];
                float v1316_data = ir5[1];
                ir5[1] = (v1316_data + (v1306_data * (sycl::group_broadcast(item.get_sub_group(), v1313_data, 11))));
                float v1319_data = r4[2];
                float v1322_data = ir5[2];
                ir5[2] = (v1322_data + (v1306_data * (sycl::group_broadcast(item.get_sub_group(), v1319_data, 11))));
                float v1325_data = r4[3];
                float v1328_data = ir5[3];
                ir5[3] = (v1328_data + (v1306_data * (sycl::group_broadcast(item.get_sub_group(), v1325_data, 11))));
                float v1331_data = r4[4];
                float v1334_data = ir5[4];
                ir5[4] = (v1334_data + (v1306_data * (sycl::group_broadcast(item.get_sub_group(), v1331_data, 11))));
                float v1337_data = r4[5];
                float v1340_data = ir5[5];
                ir5[5] = (v1340_data + (v1306_data * (sycl::group_broadcast(item.get_sub_group(), v1337_data, 11))));
                float v1343_data = r4[6];
                float v1346_data = ir5[6];
                ir5[6] = (v1346_data + (v1306_data * (sycl::group_broadcast(item.get_sub_group(), v1343_data, 11))));
                float v1349_data = r4[7];
                float v1352_data = ir5[7];
                ir5[7] = (v1352_data + (v1306_data * (sycl::group_broadcast(item.get_sub_group(), v1349_data, 11))));
              }
              if (v22_lead < 12) {
                #pragma unroll
                for (int32_t v1358_n1 = 0; v1358_n1 < 8; ++v1358_n1) {
                  float v1360_data = ir5[v1358_n1];
                  float v1362_data = r2[v1358_n1];
                  r5[v1358_n1] = (v1362_data + v1360_data);
                }
              }
              float r7[8]{};
              // r7 = load{g>r}(glb_m6);
              if (v22_lead < 12) {
                #pragma unroll
                for (int32_t v1370_i1 = 0; v1370_i1 < 8; ++v1370_i1) {
                  float v1378_data = glb_m6[(v22_lead + (v1370_i1 * 12))];
                  r7[v1370_i1] = v1378_data;
                }
              }
              // wait(r6 = load{g>r}(glb_m5););
              float r9[12]{};
              // r9 = load{g>r}(glb_m7);
              if (v22_lead < 12) {
                #pragma unroll
                for (int32_t v1385_i1 = 0; v1385_i1 < 12; ++v1385_i1) {
                  float v1393_data = glb_m7[(v22_lead + (v1385_i1 * 12))];
                  r9[v1385_i1] = v1393_data;
                }
              }
              // wait(r7 = load{g>r}(glb_m6););
              float r8[8]{};
              // r8 = +(r6 * r7) + name: r5, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir8[8]{};
              if (v22_lead < 12) {
                float v1401_data = r6[0];
                float v1402_data = r7[0];
                float v1405_data = ir8[0];
                ir8[0] = (v1405_data + (v1401_data * (sycl::group_broadcast(item.get_sub_group(), v1402_data, 0))));
                float v1408_data = r7[1];
                float v1411_data = ir8[1];
                ir8[1] = (v1411_data + (v1401_data * (sycl::group_broadcast(item.get_sub_group(), v1408_data, 0))));
                float v1414_data = r7[2];
                float v1417_data = ir8[2];
                ir8[2] = (v1417_data + (v1401_data * (sycl::group_broadcast(item.get_sub_group(), v1414_data, 0))));
                float v1420_data = r7[3];
                float v1423_data = ir8[3];
                ir8[3] = (v1423_data + (v1401_data * (sycl::group_broadcast(item.get_sub_group(), v1420_data, 0))));
                float v1426_data = r7[4];
                float v1429_data = ir8[4];
                ir8[4] = (v1429_data + (v1401_data * (sycl::group_broadcast(item.get_sub_group(), v1426_data, 0))));
                float v1432_data = r7[5];
                float v1435_data = ir8[5];
                ir8[5] = (v1435_data + (v1401_data * (sycl::group_broadcast(item.get_sub_group(), v1432_data, 0))));
                float v1438_data = r7[6];
                float v1441_data = ir8[6];
                ir8[6] = (v1441_data + (v1401_data * (sycl::group_broadcast(item.get_sub_group(), v1438_data, 0))));
                float v1444_data = r7[7];
                float v1447_data = ir8[7];
                ir8[7] = (v1447_data + (v1401_data * (sycl::group_broadcast(item.get_sub_group(), v1444_data, 0))));
              }
              if (v22_lead < 12) {
                float v1453_data = r6[1];
                float v1454_data = r7[0];
                float v1457_data = ir8[0];
                ir8[0] = (v1457_data + (v1453_data * (sycl::group_broadcast(item.get_sub_group(), v1454_data, 1))));
                float v1460_data = r7[1];
                float v1463_data = ir8[1];
                ir8[1] = (v1463_data + (v1453_data * (sycl::group_broadcast(item.get_sub_group(), v1460_data, 1))));
                float v1466_data = r7[2];
                float v1469_data = ir8[2];
                ir8[2] = (v1469_data + (v1453_data * (sycl::group_broadcast(item.get_sub_group(), v1466_data, 1))));
                float v1472_data = r7[3];
                float v1475_data = ir8[3];
                ir8[3] = (v1475_data + (v1453_data * (sycl::group_broadcast(item.get_sub_group(), v1472_data, 1))));
                float v1478_data = r7[4];
                float v1481_data = ir8[4];
                ir8[4] = (v1481_data + (v1453_data * (sycl::group_broadcast(item.get_sub_group(), v1478_data, 1))));
                float v1484_data = r7[5];
                float v1487_data = ir8[5];
                ir8[5] = (v1487_data + (v1453_data * (sycl::group_broadcast(item.get_sub_group(), v1484_data, 1))));
                float v1490_data = r7[6];
                float v1493_data = ir8[6];
                ir8[6] = (v1493_data + (v1453_data * (sycl::group_broadcast(item.get_sub_group(), v1490_data, 1))));
                float v1496_data = r7[7];
                float v1499_data = ir8[7];
                ir8[7] = (v1499_data + (v1453_data * (sycl::group_broadcast(item.get_sub_group(), v1496_data, 1))));
              }
              if (v22_lead < 12) {
                float v1505_data = r6[2];
                float v1506_data = r7[0];
                float v1509_data = ir8[0];
                ir8[0] = (v1509_data + (v1505_data * (sycl::group_broadcast(item.get_sub_group(), v1506_data, 2))));
                float v1512_data = r7[1];
                float v1515_data = ir8[1];
                ir8[1] = (v1515_data + (v1505_data * (sycl::group_broadcast(item.get_sub_group(), v1512_data, 2))));
                float v1518_data = r7[2];
                float v1521_data = ir8[2];
                ir8[2] = (v1521_data + (v1505_data * (sycl::group_broadcast(item.get_sub_group(), v1518_data, 2))));
                float v1524_data = r7[3];
                float v1527_data = ir8[3];
                ir8[3] = (v1527_data + (v1505_data * (sycl::group_broadcast(item.get_sub_group(), v1524_data, 2))));
                float v1530_data = r7[4];
                float v1533_data = ir8[4];
                ir8[4] = (v1533_data + (v1505_data * (sycl::group_broadcast(item.get_sub_group(), v1530_data, 2))));
                float v1536_data = r7[5];
                float v1539_data = ir8[5];
                ir8[5] = (v1539_data + (v1505_data * (sycl::group_broadcast(item.get_sub_group(), v1536_data, 2))));
                float v1542_data = r7[6];
                float v1545_data = ir8[6];
                ir8[6] = (v1545_data + (v1505_data * (sycl::group_broadcast(item.get_sub_group(), v1542_data, 2))));
                float v1548_data = r7[7];
                float v1551_data = ir8[7];
                ir8[7] = (v1551_data + (v1505_data * (sycl::group_broadcast(item.get_sub_group(), v1548_data, 2))));
              }
              if (v22_lead < 12) {
                float v1557_data = r6[3];
                float v1558_data = r7[0];
                float v1561_data = ir8[0];
                ir8[0] = (v1561_data + (v1557_data * (sycl::group_broadcast(item.get_sub_group(), v1558_data, 3))));
                float v1564_data = r7[1];
                float v1567_data = ir8[1];
                ir8[1] = (v1567_data + (v1557_data * (sycl::group_broadcast(item.get_sub_group(), v1564_data, 3))));
                float v1570_data = r7[2];
                float v1573_data = ir8[2];
                ir8[2] = (v1573_data + (v1557_data * (sycl::group_broadcast(item.get_sub_group(), v1570_data, 3))));
                float v1576_data = r7[3];
                float v1579_data = ir8[3];
                ir8[3] = (v1579_data + (v1557_data * (sycl::group_broadcast(item.get_sub_group(), v1576_data, 3))));
                float v1582_data = r7[4];
                float v1585_data = ir8[4];
                ir8[4] = (v1585_data + (v1557_data * (sycl::group_broadcast(item.get_sub_group(), v1582_data, 3))));
                float v1588_data = r7[5];
                float v1591_data = ir8[5];
                ir8[5] = (v1591_data + (v1557_data * (sycl::group_broadcast(item.get_sub_group(), v1588_data, 3))));
                float v1594_data = r7[6];
                float v1597_data = ir8[6];
                ir8[6] = (v1597_data + (v1557_data * (sycl::group_broadcast(item.get_sub_group(), v1594_data, 3))));
                float v1600_data = r7[7];
                float v1603_data = ir8[7];
                ir8[7] = (v1603_data + (v1557_data * (sycl::group_broadcast(item.get_sub_group(), v1600_data, 3))));
              }
              if (v22_lead < 12) {
                float v1609_data = r6[4];
                float v1610_data = r7[0];
                float v1613_data = ir8[0];
                ir8[0] = (v1613_data + (v1609_data * (sycl::group_broadcast(item.get_sub_group(), v1610_data, 4))));
                float v1616_data = r7[1];
                float v1619_data = ir8[1];
                ir8[1] = (v1619_data + (v1609_data * (sycl::group_broadcast(item.get_sub_group(), v1616_data, 4))));
                float v1622_data = r7[2];
                float v1625_data = ir8[2];
                ir8[2] = (v1625_data + (v1609_data * (sycl::group_broadcast(item.get_sub_group(), v1622_data, 4))));
                float v1628_data = r7[3];
                float v1631_data = ir8[3];
                ir8[3] = (v1631_data + (v1609_data * (sycl::group_broadcast(item.get_sub_group(), v1628_data, 4))));
                float v1634_data = r7[4];
                float v1637_data = ir8[4];
                ir8[4] = (v1637_data + (v1609_data * (sycl::group_broadcast(item.get_sub_group(), v1634_data, 4))));
                float v1640_data = r7[5];
                float v1643_data = ir8[5];
                ir8[5] = (v1643_data + (v1609_data * (sycl::group_broadcast(item.get_sub_group(), v1640_data, 4))));
                float v1646_data = r7[6];
                float v1649_data = ir8[6];
                ir8[6] = (v1649_data + (v1609_data * (sycl::group_broadcast(item.get_sub_group(), v1646_data, 4))));
                float v1652_data = r7[7];
                float v1655_data = ir8[7];
                ir8[7] = (v1655_data + (v1609_data * (sycl::group_broadcast(item.get_sub_group(), v1652_data, 4))));
              }
              if (v22_lead < 12) {
                float v1661_data = r6[5];
                float v1662_data = r7[0];
                float v1665_data = ir8[0];
                ir8[0] = (v1665_data + (v1661_data * (sycl::group_broadcast(item.get_sub_group(), v1662_data, 5))));
                float v1668_data = r7[1];
                float v1671_data = ir8[1];
                ir8[1] = (v1671_data + (v1661_data * (sycl::group_broadcast(item.get_sub_group(), v1668_data, 5))));
                float v1674_data = r7[2];
                float v1677_data = ir8[2];
                ir8[2] = (v1677_data + (v1661_data * (sycl::group_broadcast(item.get_sub_group(), v1674_data, 5))));
                float v1680_data = r7[3];
                float v1683_data = ir8[3];
                ir8[3] = (v1683_data + (v1661_data * (sycl::group_broadcast(item.get_sub_group(), v1680_data, 5))));
                float v1686_data = r7[4];
                float v1689_data = ir8[4];
                ir8[4] = (v1689_data + (v1661_data * (sycl::group_broadcast(item.get_sub_group(), v1686_data, 5))));
                float v1692_data = r7[5];
                float v1695_data = ir8[5];
                ir8[5] = (v1695_data + (v1661_data * (sycl::group_broadcast(item.get_sub_group(), v1692_data, 5))));
                float v1698_data = r7[6];
                float v1701_data = ir8[6];
                ir8[6] = (v1701_data + (v1661_data * (sycl::group_broadcast(item.get_sub_group(), v1698_data, 5))));
                float v1704_data = r7[7];
                float v1707_data = ir8[7];
                ir8[7] = (v1707_data + (v1661_data * (sycl::group_broadcast(item.get_sub_group(), v1704_data, 5))));
              }
              if (v22_lead < 12) {
                float v1713_data = r6[6];
                float v1714_data = r7[0];
                float v1717_data = ir8[0];
                ir8[0] = (v1717_data + (v1713_data * (sycl::group_broadcast(item.get_sub_group(), v1714_data, 6))));
                float v1720_data = r7[1];
                float v1723_data = ir8[1];
                ir8[1] = (v1723_data + (v1713_data * (sycl::group_broadcast(item.get_sub_group(), v1720_data, 6))));
                float v1726_data = r7[2];
                float v1729_data = ir8[2];
                ir8[2] = (v1729_data + (v1713_data * (sycl::group_broadcast(item.get_sub_group(), v1726_data, 6))));
                float v1732_data = r7[3];
                float v1735_data = ir8[3];
                ir8[3] = (v1735_data + (v1713_data * (sycl::group_broadcast(item.get_sub_group(), v1732_data, 6))));
                float v1738_data = r7[4];
                float v1741_data = ir8[4];
                ir8[4] = (v1741_data + (v1713_data * (sycl::group_broadcast(item.get_sub_group(), v1738_data, 6))));
                float v1744_data = r7[5];
                float v1747_data = ir8[5];
                ir8[5] = (v1747_data + (v1713_data * (sycl::group_broadcast(item.get_sub_group(), v1744_data, 6))));
                float v1750_data = r7[6];
                float v1753_data = ir8[6];
                ir8[6] = (v1753_data + (v1713_data * (sycl::group_broadcast(item.get_sub_group(), v1750_data, 6))));
                float v1756_data = r7[7];
                float v1759_data = ir8[7];
                ir8[7] = (v1759_data + (v1713_data * (sycl::group_broadcast(item.get_sub_group(), v1756_data, 6))));
              }
              if (v22_lead < 12) {
                float v1765_data = r6[7];
                float v1766_data = r7[0];
                float v1769_data = ir8[0];
                ir8[0] = (v1769_data + (v1765_data * (sycl::group_broadcast(item.get_sub_group(), v1766_data, 7))));
                float v1772_data = r7[1];
                float v1775_data = ir8[1];
                ir8[1] = (v1775_data + (v1765_data * (sycl::group_broadcast(item.get_sub_group(), v1772_data, 7))));
                float v1778_data = r7[2];
                float v1781_data = ir8[2];
                ir8[2] = (v1781_data + (v1765_data * (sycl::group_broadcast(item.get_sub_group(), v1778_data, 7))));
                float v1784_data = r7[3];
                float v1787_data = ir8[3];
                ir8[3] = (v1787_data + (v1765_data * (sycl::group_broadcast(item.get_sub_group(), v1784_data, 7))));
                float v1790_data = r7[4];
                float v1793_data = ir8[4];
                ir8[4] = (v1793_data + (v1765_data * (sycl::group_broadcast(item.get_sub_group(), v1790_data, 7))));
                float v1796_data = r7[5];
                float v1799_data = ir8[5];
                ir8[5] = (v1799_data + (v1765_data * (sycl::group_broadcast(item.get_sub_group(), v1796_data, 7))));
                float v1802_data = r7[6];
                float v1805_data = ir8[6];
                ir8[6] = (v1805_data + (v1765_data * (sycl::group_broadcast(item.get_sub_group(), v1802_data, 7))));
                float v1808_data = r7[7];
                float v1811_data = ir8[7];
                ir8[7] = (v1811_data + (v1765_data * (sycl::group_broadcast(item.get_sub_group(), v1808_data, 7))));
              }
              if (v22_lead < 12) {
                float v1817_data = r6[8];
                float v1818_data = r7[0];
                float v1821_data = ir8[0];
                ir8[0] = (v1821_data + (v1817_data * (sycl::group_broadcast(item.get_sub_group(), v1818_data, 8))));
                float v1824_data = r7[1];
                float v1827_data = ir8[1];
                ir8[1] = (v1827_data + (v1817_data * (sycl::group_broadcast(item.get_sub_group(), v1824_data, 8))));
                float v1830_data = r7[2];
                float v1833_data = ir8[2];
                ir8[2] = (v1833_data + (v1817_data * (sycl::group_broadcast(item.get_sub_group(), v1830_data, 8))));
                float v1836_data = r7[3];
                float v1839_data = ir8[3];
                ir8[3] = (v1839_data + (v1817_data * (sycl::group_broadcast(item.get_sub_group(), v1836_data, 8))));
                float v1842_data = r7[4];
                float v1845_data = ir8[4];
                ir8[4] = (v1845_data + (v1817_data * (sycl::group_broadcast(item.get_sub_group(), v1842_data, 8))));
                float v1848_data = r7[5];
                float v1851_data = ir8[5];
                ir8[5] = (v1851_data + (v1817_data * (sycl::group_broadcast(item.get_sub_group(), v1848_data, 8))));
                float v1854_data = r7[6];
                float v1857_data = ir8[6];
                ir8[6] = (v1857_data + (v1817_data * (sycl::group_broadcast(item.get_sub_group(), v1854_data, 8))));
                float v1860_data = r7[7];
                float v1863_data = ir8[7];
                ir8[7] = (v1863_data + (v1817_data * (sycl::group_broadcast(item.get_sub_group(), v1860_data, 8))));
              }
              if (v22_lead < 12) {
                float v1869_data = r6[9];
                float v1870_data = r7[0];
                float v1873_data = ir8[0];
                ir8[0] = (v1873_data + (v1869_data * (sycl::group_broadcast(item.get_sub_group(), v1870_data, 9))));
                float v1876_data = r7[1];
                float v1879_data = ir8[1];
                ir8[1] = (v1879_data + (v1869_data * (sycl::group_broadcast(item.get_sub_group(), v1876_data, 9))));
                float v1882_data = r7[2];
                float v1885_data = ir8[2];
                ir8[2] = (v1885_data + (v1869_data * (sycl::group_broadcast(item.get_sub_group(), v1882_data, 9))));
                float v1888_data = r7[3];
                float v1891_data = ir8[3];
                ir8[3] = (v1891_data + (v1869_data * (sycl::group_broadcast(item.get_sub_group(), v1888_data, 9))));
                float v1894_data = r7[4];
                float v1897_data = ir8[4];
                ir8[4] = (v1897_data + (v1869_data * (sycl::group_broadcast(item.get_sub_group(), v1894_data, 9))));
                float v1900_data = r7[5];
                float v1903_data = ir8[5];
                ir8[5] = (v1903_data + (v1869_data * (sycl::group_broadcast(item.get_sub_group(), v1900_data, 9))));
                float v1906_data = r7[6];
                float v1909_data = ir8[6];
                ir8[6] = (v1909_data + (v1869_data * (sycl::group_broadcast(item.get_sub_group(), v1906_data, 9))));
                float v1912_data = r7[7];
                float v1915_data = ir8[7];
                ir8[7] = (v1915_data + (v1869_data * (sycl::group_broadcast(item.get_sub_group(), v1912_data, 9))));
              }
              if (v22_lead < 12) {
                float v1921_data = r6[10];
                float v1922_data = r7[0];
                float v1925_data = ir8[0];
                ir8[0] = (v1925_data + (v1921_data * (sycl::group_broadcast(item.get_sub_group(), v1922_data, 10))));
                float v1928_data = r7[1];
                float v1931_data = ir8[1];
                ir8[1] = (v1931_data + (v1921_data * (sycl::group_broadcast(item.get_sub_group(), v1928_data, 10))));
                float v1934_data = r7[2];
                float v1937_data = ir8[2];
                ir8[2] = (v1937_data + (v1921_data * (sycl::group_broadcast(item.get_sub_group(), v1934_data, 10))));
                float v1940_data = r7[3];
                float v1943_data = ir8[3];
                ir8[3] = (v1943_data + (v1921_data * (sycl::group_broadcast(item.get_sub_group(), v1940_data, 10))));
                float v1946_data = r7[4];
                float v1949_data = ir8[4];
                ir8[4] = (v1949_data + (v1921_data * (sycl::group_broadcast(item.get_sub_group(), v1946_data, 10))));
                float v1952_data = r7[5];
                float v1955_data = ir8[5];
                ir8[5] = (v1955_data + (v1921_data * (sycl::group_broadcast(item.get_sub_group(), v1952_data, 10))));
                float v1958_data = r7[6];
                float v1961_data = ir8[6];
                ir8[6] = (v1961_data + (v1921_data * (sycl::group_broadcast(item.get_sub_group(), v1958_data, 10))));
                float v1964_data = r7[7];
                float v1967_data = ir8[7];
                ir8[7] = (v1967_data + (v1921_data * (sycl::group_broadcast(item.get_sub_group(), v1964_data, 10))));
              }
              if (v22_lead < 12) {
                float v1973_data = r6[11];
                float v1974_data = r7[0];
                float v1977_data = ir8[0];
                ir8[0] = (v1977_data + (v1973_data * (sycl::group_broadcast(item.get_sub_group(), v1974_data, 11))));
                float v1980_data = r7[1];
                float v1983_data = ir8[1];
                ir8[1] = (v1983_data + (v1973_data * (sycl::group_broadcast(item.get_sub_group(), v1980_data, 11))));
                float v1986_data = r7[2];
                float v1989_data = ir8[2];
                ir8[2] = (v1989_data + (v1973_data * (sycl::group_broadcast(item.get_sub_group(), v1986_data, 11))));
                float v1992_data = r7[3];
                float v1995_data = ir8[3];
                ir8[3] = (v1995_data + (v1973_data * (sycl::group_broadcast(item.get_sub_group(), v1992_data, 11))));
                float v1998_data = r7[4];
                float v2001_data = ir8[4];
                ir8[4] = (v2001_data + (v1973_data * (sycl::group_broadcast(item.get_sub_group(), v1998_data, 11))));
                float v2004_data = r7[5];
                float v2007_data = ir8[5];
                ir8[5] = (v2007_data + (v1973_data * (sycl::group_broadcast(item.get_sub_group(), v2004_data, 11))));
                float v2010_data = r7[6];
                float v2013_data = ir8[6];
                ir8[6] = (v2013_data + (v1973_data * (sycl::group_broadcast(item.get_sub_group(), v2010_data, 11))));
                float v2016_data = r7[7];
                float v2019_data = ir8[7];
                ir8[7] = (v2019_data + (v1973_data * (sycl::group_broadcast(item.get_sub_group(), v2016_data, 11))));
              }
              if (v22_lead < 12) {
                #pragma unroll
                for (int32_t v2025_n1 = 0; v2025_n1 < 8; ++v2025_n1) {
                  float v2027_data = ir8[v2025_n1];
                  float v2029_data = r5[v2025_n1];
                  r8[v2025_n1] = (v2029_data + v2027_data);
                }
              }
              float r10[8]{};
              // r10 = load{g>r}(glb_m8);
              if (v22_lead < 12) {
                #pragma unroll
                for (int32_t v2037_i1 = 0; v2037_i1 < 8; ++v2037_i1) {
                  float v2045_data = glb_m8[(v22_lead + (v2037_i1 * 12))];
                  r10[v2037_i1] = v2045_data;
                }
              }
              // wait(r9 = load{g>r}(glb_m7););
              // wait(r10 = load{g>r}(glb_m8););
              float r11[8]{};
              // r11 = +(r9 * r10) + name: r8, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir11[8]{};
              if (v22_lead < 12) {
                float v2053_data = r9[0];
                float v2054_data = r10[0];
                float v2057_data = ir11[0];
                ir11[0] = (v2057_data + (v2053_data * (sycl::group_broadcast(item.get_sub_group(), v2054_data, 0))));
                float v2060_data = r10[1];
                float v2063_data = ir11[1];
                ir11[1] = (v2063_data + (v2053_data * (sycl::group_broadcast(item.get_sub_group(), v2060_data, 0))));
                float v2066_data = r10[2];
                float v2069_data = ir11[2];
                ir11[2] = (v2069_data + (v2053_data * (sycl::group_broadcast(item.get_sub_group(), v2066_data, 0))));
                float v2072_data = r10[3];
                float v2075_data = ir11[3];
                ir11[3] = (v2075_data + (v2053_data * (sycl::group_broadcast(item.get_sub_group(), v2072_data, 0))));
                float v2078_data = r10[4];
                float v2081_data = ir11[4];
                ir11[4] = (v2081_data + (v2053_data * (sycl::group_broadcast(item.get_sub_group(), v2078_data, 0))));
                float v2084_data = r10[5];
                float v2087_data = ir11[5];
                ir11[5] = (v2087_data + (v2053_data * (sycl::group_broadcast(item.get_sub_group(), v2084_data, 0))));
                float v2090_data = r10[6];
                float v2093_data = ir11[6];
                ir11[6] = (v2093_data + (v2053_data * (sycl::group_broadcast(item.get_sub_group(), v2090_data, 0))));
                float v2096_data = r10[7];
                float v2099_data = ir11[7];
                ir11[7] = (v2099_data + (v2053_data * (sycl::group_broadcast(item.get_sub_group(), v2096_data, 0))));
              }
              if (v22_lead < 12) {
                float v2105_data = r9[1];
                float v2106_data = r10[0];
                float v2109_data = ir11[0];
                ir11[0] = (v2109_data + (v2105_data * (sycl::group_broadcast(item.get_sub_group(), v2106_data, 1))));
                float v2112_data = r10[1];
                float v2115_data = ir11[1];
                ir11[1] = (v2115_data + (v2105_data * (sycl::group_broadcast(item.get_sub_group(), v2112_data, 1))));
                float v2118_data = r10[2];
                float v2121_data = ir11[2];
                ir11[2] = (v2121_data + (v2105_data * (sycl::group_broadcast(item.get_sub_group(), v2118_data, 1))));
                float v2124_data = r10[3];
                float v2127_data = ir11[3];
                ir11[3] = (v2127_data + (v2105_data * (sycl::group_broadcast(item.get_sub_group(), v2124_data, 1))));
                float v2130_data = r10[4];
                float v2133_data = ir11[4];
                ir11[4] = (v2133_data + (v2105_data * (sycl::group_broadcast(item.get_sub_group(), v2130_data, 1))));
                float v2136_data = r10[5];
                float v2139_data = ir11[5];
                ir11[5] = (v2139_data + (v2105_data * (sycl::group_broadcast(item.get_sub_group(), v2136_data, 1))));
                float v2142_data = r10[6];
                float v2145_data = ir11[6];
                ir11[6] = (v2145_data + (v2105_data * (sycl::group_broadcast(item.get_sub_group(), v2142_data, 1))));
                float v2148_data = r10[7];
                float v2151_data = ir11[7];
                ir11[7] = (v2151_data + (v2105_data * (sycl::group_broadcast(item.get_sub_group(), v2148_data, 1))));
              }
              if (v22_lead < 12) {
                float v2157_data = r9[2];
                float v2158_data = r10[0];
                float v2161_data = ir11[0];
                ir11[0] = (v2161_data + (v2157_data * (sycl::group_broadcast(item.get_sub_group(), v2158_data, 2))));
                float v2164_data = r10[1];
                float v2167_data = ir11[1];
                ir11[1] = (v2167_data + (v2157_data * (sycl::group_broadcast(item.get_sub_group(), v2164_data, 2))));
                float v2170_data = r10[2];
                float v2173_data = ir11[2];
                ir11[2] = (v2173_data + (v2157_data * (sycl::group_broadcast(item.get_sub_group(), v2170_data, 2))));
                float v2176_data = r10[3];
                float v2179_data = ir11[3];
                ir11[3] = (v2179_data + (v2157_data * (sycl::group_broadcast(item.get_sub_group(), v2176_data, 2))));
                float v2182_data = r10[4];
                float v2185_data = ir11[4];
                ir11[4] = (v2185_data + (v2157_data * (sycl::group_broadcast(item.get_sub_group(), v2182_data, 2))));
                float v2188_data = r10[5];
                float v2191_data = ir11[5];
                ir11[5] = (v2191_data + (v2157_data * (sycl::group_broadcast(item.get_sub_group(), v2188_data, 2))));
                float v2194_data = r10[6];
                float v2197_data = ir11[6];
                ir11[6] = (v2197_data + (v2157_data * (sycl::group_broadcast(item.get_sub_group(), v2194_data, 2))));
                float v2200_data = r10[7];
                float v2203_data = ir11[7];
                ir11[7] = (v2203_data + (v2157_data * (sycl::group_broadcast(item.get_sub_group(), v2200_data, 2))));
              }
              if (v22_lead < 12) {
                float v2209_data = r9[3];
                float v2210_data = r10[0];
                float v2213_data = ir11[0];
                ir11[0] = (v2213_data + (v2209_data * (sycl::group_broadcast(item.get_sub_group(), v2210_data, 3))));
                float v2216_data = r10[1];
                float v2219_data = ir11[1];
                ir11[1] = (v2219_data + (v2209_data * (sycl::group_broadcast(item.get_sub_group(), v2216_data, 3))));
                float v2222_data = r10[2];
                float v2225_data = ir11[2];
                ir11[2] = (v2225_data + (v2209_data * (sycl::group_broadcast(item.get_sub_group(), v2222_data, 3))));
                float v2228_data = r10[3];
                float v2231_data = ir11[3];
                ir11[3] = (v2231_data + (v2209_data * (sycl::group_broadcast(item.get_sub_group(), v2228_data, 3))));
                float v2234_data = r10[4];
                float v2237_data = ir11[4];
                ir11[4] = (v2237_data + (v2209_data * (sycl::group_broadcast(item.get_sub_group(), v2234_data, 3))));
                float v2240_data = r10[5];
                float v2243_data = ir11[5];
                ir11[5] = (v2243_data + (v2209_data * (sycl::group_broadcast(item.get_sub_group(), v2240_data, 3))));
                float v2246_data = r10[6];
                float v2249_data = ir11[6];
                ir11[6] = (v2249_data + (v2209_data * (sycl::group_broadcast(item.get_sub_group(), v2246_data, 3))));
                float v2252_data = r10[7];
                float v2255_data = ir11[7];
                ir11[7] = (v2255_data + (v2209_data * (sycl::group_broadcast(item.get_sub_group(), v2252_data, 3))));
              }
              if (v22_lead < 12) {
                float v2261_data = r9[4];
                float v2262_data = r10[0];
                float v2265_data = ir11[0];
                ir11[0] = (v2265_data + (v2261_data * (sycl::group_broadcast(item.get_sub_group(), v2262_data, 4))));
                float v2268_data = r10[1];
                float v2271_data = ir11[1];
                ir11[1] = (v2271_data + (v2261_data * (sycl::group_broadcast(item.get_sub_group(), v2268_data, 4))));
                float v2274_data = r10[2];
                float v2277_data = ir11[2];
                ir11[2] = (v2277_data + (v2261_data * (sycl::group_broadcast(item.get_sub_group(), v2274_data, 4))));
                float v2280_data = r10[3];
                float v2283_data = ir11[3];
                ir11[3] = (v2283_data + (v2261_data * (sycl::group_broadcast(item.get_sub_group(), v2280_data, 4))));
                float v2286_data = r10[4];
                float v2289_data = ir11[4];
                ir11[4] = (v2289_data + (v2261_data * (sycl::group_broadcast(item.get_sub_group(), v2286_data, 4))));
                float v2292_data = r10[5];
                float v2295_data = ir11[5];
                ir11[5] = (v2295_data + (v2261_data * (sycl::group_broadcast(item.get_sub_group(), v2292_data, 4))));
                float v2298_data = r10[6];
                float v2301_data = ir11[6];
                ir11[6] = (v2301_data + (v2261_data * (sycl::group_broadcast(item.get_sub_group(), v2298_data, 4))));
                float v2304_data = r10[7];
                float v2307_data = ir11[7];
                ir11[7] = (v2307_data + (v2261_data * (sycl::group_broadcast(item.get_sub_group(), v2304_data, 4))));
              }
              if (v22_lead < 12) {
                float v2313_data = r9[5];
                float v2314_data = r10[0];
                float v2317_data = ir11[0];
                ir11[0] = (v2317_data + (v2313_data * (sycl::group_broadcast(item.get_sub_group(), v2314_data, 5))));
                float v2320_data = r10[1];
                float v2323_data = ir11[1];
                ir11[1] = (v2323_data + (v2313_data * (sycl::group_broadcast(item.get_sub_group(), v2320_data, 5))));
                float v2326_data = r10[2];
                float v2329_data = ir11[2];
                ir11[2] = (v2329_data + (v2313_data * (sycl::group_broadcast(item.get_sub_group(), v2326_data, 5))));
                float v2332_data = r10[3];
                float v2335_data = ir11[3];
                ir11[3] = (v2335_data + (v2313_data * (sycl::group_broadcast(item.get_sub_group(), v2332_data, 5))));
                float v2338_data = r10[4];
                float v2341_data = ir11[4];
                ir11[4] = (v2341_data + (v2313_data * (sycl::group_broadcast(item.get_sub_group(), v2338_data, 5))));
                float v2344_data = r10[5];
                float v2347_data = ir11[5];
                ir11[5] = (v2347_data + (v2313_data * (sycl::group_broadcast(item.get_sub_group(), v2344_data, 5))));
                float v2350_data = r10[6];
                float v2353_data = ir11[6];
                ir11[6] = (v2353_data + (v2313_data * (sycl::group_broadcast(item.get_sub_group(), v2350_data, 5))));
                float v2356_data = r10[7];
                float v2359_data = ir11[7];
                ir11[7] = (v2359_data + (v2313_data * (sycl::group_broadcast(item.get_sub_group(), v2356_data, 5))));
              }
              if (v22_lead < 12) {
                float v2365_data = r9[6];
                float v2366_data = r10[0];
                float v2369_data = ir11[0];
                ir11[0] = (v2369_data + (v2365_data * (sycl::group_broadcast(item.get_sub_group(), v2366_data, 6))));
                float v2372_data = r10[1];
                float v2375_data = ir11[1];
                ir11[1] = (v2375_data + (v2365_data * (sycl::group_broadcast(item.get_sub_group(), v2372_data, 6))));
                float v2378_data = r10[2];
                float v2381_data = ir11[2];
                ir11[2] = (v2381_data + (v2365_data * (sycl::group_broadcast(item.get_sub_group(), v2378_data, 6))));
                float v2384_data = r10[3];
                float v2387_data = ir11[3];
                ir11[3] = (v2387_data + (v2365_data * (sycl::group_broadcast(item.get_sub_group(), v2384_data, 6))));
                float v2390_data = r10[4];
                float v2393_data = ir11[4];
                ir11[4] = (v2393_data + (v2365_data * (sycl::group_broadcast(item.get_sub_group(), v2390_data, 6))));
                float v2396_data = r10[5];
                float v2399_data = ir11[5];
                ir11[5] = (v2399_data + (v2365_data * (sycl::group_broadcast(item.get_sub_group(), v2396_data, 6))));
                float v2402_data = r10[6];
                float v2405_data = ir11[6];
                ir11[6] = (v2405_data + (v2365_data * (sycl::group_broadcast(item.get_sub_group(), v2402_data, 6))));
                float v2408_data = r10[7];
                float v2411_data = ir11[7];
                ir11[7] = (v2411_data + (v2365_data * (sycl::group_broadcast(item.get_sub_group(), v2408_data, 6))));
              }
              if (v22_lead < 12) {
                float v2417_data = r9[7];
                float v2418_data = r10[0];
                float v2421_data = ir11[0];
                ir11[0] = (v2421_data + (v2417_data * (sycl::group_broadcast(item.get_sub_group(), v2418_data, 7))));
                float v2424_data = r10[1];
                float v2427_data = ir11[1];
                ir11[1] = (v2427_data + (v2417_data * (sycl::group_broadcast(item.get_sub_group(), v2424_data, 7))));
                float v2430_data = r10[2];
                float v2433_data = ir11[2];
                ir11[2] = (v2433_data + (v2417_data * (sycl::group_broadcast(item.get_sub_group(), v2430_data, 7))));
                float v2436_data = r10[3];
                float v2439_data = ir11[3];
                ir11[3] = (v2439_data + (v2417_data * (sycl::group_broadcast(item.get_sub_group(), v2436_data, 7))));
                float v2442_data = r10[4];
                float v2445_data = ir11[4];
                ir11[4] = (v2445_data + (v2417_data * (sycl::group_broadcast(item.get_sub_group(), v2442_data, 7))));
                float v2448_data = r10[5];
                float v2451_data = ir11[5];
                ir11[5] = (v2451_data + (v2417_data * (sycl::group_broadcast(item.get_sub_group(), v2448_data, 7))));
                float v2454_data = r10[6];
                float v2457_data = ir11[6];
                ir11[6] = (v2457_data + (v2417_data * (sycl::group_broadcast(item.get_sub_group(), v2454_data, 7))));
                float v2460_data = r10[7];
                float v2463_data = ir11[7];
                ir11[7] = (v2463_data + (v2417_data * (sycl::group_broadcast(item.get_sub_group(), v2460_data, 7))));
              }
              if (v22_lead < 12) {
                float v2469_data = r9[8];
                float v2470_data = r10[0];
                float v2473_data = ir11[0];
                ir11[0] = (v2473_data + (v2469_data * (sycl::group_broadcast(item.get_sub_group(), v2470_data, 8))));
                float v2476_data = r10[1];
                float v2479_data = ir11[1];
                ir11[1] = (v2479_data + (v2469_data * (sycl::group_broadcast(item.get_sub_group(), v2476_data, 8))));
                float v2482_data = r10[2];
                float v2485_data = ir11[2];
                ir11[2] = (v2485_data + (v2469_data * (sycl::group_broadcast(item.get_sub_group(), v2482_data, 8))));
                float v2488_data = r10[3];
                float v2491_data = ir11[3];
                ir11[3] = (v2491_data + (v2469_data * (sycl::group_broadcast(item.get_sub_group(), v2488_data, 8))));
                float v2494_data = r10[4];
                float v2497_data = ir11[4];
                ir11[4] = (v2497_data + (v2469_data * (sycl::group_broadcast(item.get_sub_group(), v2494_data, 8))));
                float v2500_data = r10[5];
                float v2503_data = ir11[5];
                ir11[5] = (v2503_data + (v2469_data * (sycl::group_broadcast(item.get_sub_group(), v2500_data, 8))));
                float v2506_data = r10[6];
                float v2509_data = ir11[6];
                ir11[6] = (v2509_data + (v2469_data * (sycl::group_broadcast(item.get_sub_group(), v2506_data, 8))));
                float v2512_data = r10[7];
                float v2515_data = ir11[7];
                ir11[7] = (v2515_data + (v2469_data * (sycl::group_broadcast(item.get_sub_group(), v2512_data, 8))));
              }
              if (v22_lead < 12) {
                float v2521_data = r9[9];
                float v2522_data = r10[0];
                float v2525_data = ir11[0];
                ir11[0] = (v2525_data + (v2521_data * (sycl::group_broadcast(item.get_sub_group(), v2522_data, 9))));
                float v2528_data = r10[1];
                float v2531_data = ir11[1];
                ir11[1] = (v2531_data + (v2521_data * (sycl::group_broadcast(item.get_sub_group(), v2528_data, 9))));
                float v2534_data = r10[2];
                float v2537_data = ir11[2];
                ir11[2] = (v2537_data + (v2521_data * (sycl::group_broadcast(item.get_sub_group(), v2534_data, 9))));
                float v2540_data = r10[3];
                float v2543_data = ir11[3];
                ir11[3] = (v2543_data + (v2521_data * (sycl::group_broadcast(item.get_sub_group(), v2540_data, 9))));
                float v2546_data = r10[4];
                float v2549_data = ir11[4];
                ir11[4] = (v2549_data + (v2521_data * (sycl::group_broadcast(item.get_sub_group(), v2546_data, 9))));
                float v2552_data = r10[5];
                float v2555_data = ir11[5];
                ir11[5] = (v2555_data + (v2521_data * (sycl::group_broadcast(item.get_sub_group(), v2552_data, 9))));
                float v2558_data = r10[6];
                float v2561_data = ir11[6];
                ir11[6] = (v2561_data + (v2521_data * (sycl::group_broadcast(item.get_sub_group(), v2558_data, 9))));
                float v2564_data = r10[7];
                float v2567_data = ir11[7];
                ir11[7] = (v2567_data + (v2521_data * (sycl::group_broadcast(item.get_sub_group(), v2564_data, 9))));
              }
              if (v22_lead < 12) {
                float v2573_data = r9[10];
                float v2574_data = r10[0];
                float v2577_data = ir11[0];
                ir11[0] = (v2577_data + (v2573_data * (sycl::group_broadcast(item.get_sub_group(), v2574_data, 10))));
                float v2580_data = r10[1];
                float v2583_data = ir11[1];
                ir11[1] = (v2583_data + (v2573_data * (sycl::group_broadcast(item.get_sub_group(), v2580_data, 10))));
                float v2586_data = r10[2];
                float v2589_data = ir11[2];
                ir11[2] = (v2589_data + (v2573_data * (sycl::group_broadcast(item.get_sub_group(), v2586_data, 10))));
                float v2592_data = r10[3];
                float v2595_data = ir11[3];
                ir11[3] = (v2595_data + (v2573_data * (sycl::group_broadcast(item.get_sub_group(), v2592_data, 10))));
                float v2598_data = r10[4];
                float v2601_data = ir11[4];
                ir11[4] = (v2601_data + (v2573_data * (sycl::group_broadcast(item.get_sub_group(), v2598_data, 10))));
                float v2604_data = r10[5];
                float v2607_data = ir11[5];
                ir11[5] = (v2607_data + (v2573_data * (sycl::group_broadcast(item.get_sub_group(), v2604_data, 10))));
                float v2610_data = r10[6];
                float v2613_data = ir11[6];
                ir11[6] = (v2613_data + (v2573_data * (sycl::group_broadcast(item.get_sub_group(), v2610_data, 10))));
                float v2616_data = r10[7];
                float v2619_data = ir11[7];
                ir11[7] = (v2619_data + (v2573_data * (sycl::group_broadcast(item.get_sub_group(), v2616_data, 10))));
              }
              if (v22_lead < 12) {
                float v2625_data = r9[11];
                float v2626_data = r10[0];
                float v2629_data = ir11[0];
                ir11[0] = (v2629_data + (v2625_data * (sycl::group_broadcast(item.get_sub_group(), v2626_data, 11))));
                float v2632_data = r10[1];
                float v2635_data = ir11[1];
                ir11[1] = (v2635_data + (v2625_data * (sycl::group_broadcast(item.get_sub_group(), v2632_data, 11))));
                float v2638_data = r10[2];
                float v2641_data = ir11[2];
                ir11[2] = (v2641_data + (v2625_data * (sycl::group_broadcast(item.get_sub_group(), v2638_data, 11))));
                float v2644_data = r10[3];
                float v2647_data = ir11[3];
                ir11[3] = (v2647_data + (v2625_data * (sycl::group_broadcast(item.get_sub_group(), v2644_data, 11))));
                float v2650_data = r10[4];
                float v2653_data = ir11[4];
                ir11[4] = (v2653_data + (v2625_data * (sycl::group_broadcast(item.get_sub_group(), v2650_data, 11))));
                float v2656_data = r10[5];
                float v2659_data = ir11[5];
                ir11[5] = (v2659_data + (v2625_data * (sycl::group_broadcast(item.get_sub_group(), v2656_data, 11))));
                float v2662_data = r10[6];
                float v2665_data = ir11[6];
                ir11[6] = (v2665_data + (v2625_data * (sycl::group_broadcast(item.get_sub_group(), v2662_data, 11))));
                float v2668_data = r10[7];
                float v2671_data = ir11[7];
                ir11[7] = (v2671_data + (v2625_data * (sycl::group_broadcast(item.get_sub_group(), v2668_data, 11))));
              }
              if (v22_lead < 12) {
                #pragma unroll
                for (int32_t v2677_n1 = 0; v2677_n1 < 8; ++v2677_n1) {
                  float v2679_data = ir11[v2677_n1];
                  float v2681_data = r8[v2677_n1];
                  r11[v2677_n1] = (v2681_data + v2679_data);
                }
              }
              // glb_m0 = store{r>g}(r11);
              if (v22_lead < 12) {
                #pragma unroll
                for (int32_t v2688_i1 = 0; v2688_i1 < 8; ++v2688_i1) {
                  float v2690_data = r11[v2688_i1];
                  glb_m0[(v22_lead + (v2688_i1 * 12))] = v2690_data;
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

