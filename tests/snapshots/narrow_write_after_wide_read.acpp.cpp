// === base name ===
kernel_154580c330

// === header ===
void launcher_kernel_154580c330(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_154580c330(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid ((numElements0 + 8 - 1) / 8, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_154580c330(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_154580c330(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 32×13(32×13) {0..32}×{0..13} strided
        // m1 32×12(32×12) {0..32}×{0..12} strided
        // m2 12×13(12×13) {0..12}×{0..13} strided
        // m3 32×13(32×13) {0..32}×{0..13} strided
        // m4 13×13(13×13) {0..13}×{0..13} strided
        // t0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] = m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1]
        // t0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] += m1 32×12(32×12) {0..32}×{0..12} strided({0..32}×{0..12})[0, -1]×m2 12×13(12×13) {0..12}×{0..13} strided({0..12}×{0..13})[-1, 1]
        // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..1})[0, 1] = t0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..1})[0, 1]
        // m3 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] = m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, -1]×m4 13×13(13×13) {0..13}×{0..13} strided({0..13}×{0..13})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 416 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 384 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 156 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[batchId0 * 416 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[batchId0 * 169 + 0 + m4_extraOffset];
              float r0[13]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v8_lead = item.get_local_id(0) % 32;
              #pragma unroll
              for (int32_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
                int32_t v15_lead = v8_lead + (v9_i0 * 32);
                #pragma unroll
                for (int32_t v10_i1 = 0; v10_i1 < 13; ++v10_i1) {
                  float v18_data = glb_m0[(v15_lead + (v10_i1 * 32))];
                  r0[(v9_i0 + v10_i1)] = v18_data;
                }
              }
              float r2[12]{};
              // r2 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
                int32_t v30_lead = v8_lead + (v24_i0 * 32);
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 12; ++v25_i1) {
                  float v33_data = glb_m1[(v30_lead + (v25_i1 * 32))];
                  r2[(v24_i0 + v25_i1)] = v33_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[13]{};
              // r1 = +(r0) + None
              // [(0, 32), (0, 13)] []
              float v39_data = r0[0];
              float v40_data = r1[0];
              r1[0] = (v40_data + v39_data);
              float v42_data = r0[1];
              float v43_data = r1[1];
              r1[1] = (v43_data + v42_data);
              float v45_data = r0[2];
              float v46_data = r1[2];
              r1[2] = (v46_data + v45_data);
              float v48_data = r0[3];
              float v49_data = r1[3];
              r1[3] = (v49_data + v48_data);
              float v51_data = r0[4];
              float v52_data = r1[4];
              r1[4] = (v52_data + v51_data);
              float v54_data = r0[5];
              float v55_data = r1[5];
              r1[5] = (v55_data + v54_data);
              float v57_data = r0[6];
              float v58_data = r1[6];
              r1[6] = (v58_data + v57_data);
              float v60_data = r0[7];
              float v61_data = r1[7];
              r1[7] = (v61_data + v60_data);
              float v63_data = r0[8];
              float v64_data = r1[8];
              r1[8] = (v64_data + v63_data);
              float v66_data = r0[9];
              float v67_data = r1[9];
              r1[9] = (v67_data + v66_data);
              float v69_data = r0[10];
              float v70_data = r1[10];
              r1[10] = (v70_data + v69_data);
              float v72_data = r0[11];
              float v73_data = r1[11];
              r1[11] = (v73_data + v72_data);
              float v75_data = r0[12];
              float v76_data = r1[12];
              r1[12] = (v76_data + v75_data);
              float r3[13]{};
              // r3 = load{g>r}(glb_m2);
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v83_i1 = 0; v83_i1 < 13; ++v83_i1) {
                  float v91_data = glb_m2[(v8_lead + (v83_i1 * 12))];
                  r3[v83_i1] = v91_data;
                }
              }
              // wait(r2 = load{g>r}(glb_m1););
              // wait(r3 = load{g>r}(glb_m2););
              float r4[13]{};
              // r4 = +(r2 * r3) + name: r1, type: SymbolType.Register, lead: [0]
              // [(0, 32), (0, 13)] [(0, 12)]
              float ir4[13]{};
              float v98_data = r2[0];
              float v99_data = r3[0];
              float v102_data = ir4[0];
              ir4[0] = (v102_data + (v98_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 0))));
              float v105_data = r3[1];
              float v108_data = ir4[1];
              ir4[1] = (v108_data + (v98_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 0))));
              float v111_data = r3[2];
              float v114_data = ir4[2];
              ir4[2] = (v114_data + (v98_data * (sycl::group_broadcast(item.get_sub_group(), v111_data, 0))));
              float v117_data = r3[3];
              float v120_data = ir4[3];
              ir4[3] = (v120_data + (v98_data * (sycl::group_broadcast(item.get_sub_group(), v117_data, 0))));
              float v123_data = r3[4];
              float v126_data = ir4[4];
              ir4[4] = (v126_data + (v98_data * (sycl::group_broadcast(item.get_sub_group(), v123_data, 0))));
              float v129_data = r3[5];
              float v132_data = ir4[5];
              ir4[5] = (v132_data + (v98_data * (sycl::group_broadcast(item.get_sub_group(), v129_data, 0))));
              float v135_data = r3[6];
              float v138_data = ir4[6];
              ir4[6] = (v138_data + (v98_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 0))));
              float v141_data = r3[7];
              float v144_data = ir4[7];
              ir4[7] = (v144_data + (v98_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 0))));
              float v147_data = r3[8];
              float v150_data = ir4[8];
              ir4[8] = (v150_data + (v98_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 0))));
              float v153_data = r3[9];
              float v156_data = ir4[9];
              ir4[9] = (v156_data + (v98_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 0))));
              float v159_data = r3[10];
              float v162_data = ir4[10];
              ir4[10] = (v162_data + (v98_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 0))));
              float v165_data = r3[11];
              float v168_data = ir4[11];
              ir4[11] = (v168_data + (v98_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 0))));
              float v171_data = r3[12];
              float v174_data = ir4[12];
              ir4[12] = (v174_data + (v98_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 0))));
              float v179_data = r2[1];
              float v183_data = ir4[0];
              ir4[0] = (v183_data + (v179_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 1))));
              float v189_data = ir4[1];
              ir4[1] = (v189_data + (v179_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 1))));
              float v195_data = ir4[2];
              ir4[2] = (v195_data + (v179_data * (sycl::group_broadcast(item.get_sub_group(), v111_data, 1))));
              float v201_data = ir4[3];
              ir4[3] = (v201_data + (v179_data * (sycl::group_broadcast(item.get_sub_group(), v117_data, 1))));
              float v207_data = ir4[4];
              ir4[4] = (v207_data + (v179_data * (sycl::group_broadcast(item.get_sub_group(), v123_data, 1))));
              float v213_data = ir4[5];
              ir4[5] = (v213_data + (v179_data * (sycl::group_broadcast(item.get_sub_group(), v129_data, 1))));
              float v219_data = ir4[6];
              ir4[6] = (v219_data + (v179_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 1))));
              float v225_data = ir4[7];
              ir4[7] = (v225_data + (v179_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 1))));
              float v231_data = ir4[8];
              ir4[8] = (v231_data + (v179_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 1))));
              float v237_data = ir4[9];
              ir4[9] = (v237_data + (v179_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 1))));
              float v243_data = ir4[10];
              ir4[10] = (v243_data + (v179_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 1))));
              float v249_data = ir4[11];
              ir4[11] = (v249_data + (v179_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 1))));
              float v255_data = ir4[12];
              ir4[12] = (v255_data + (v179_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 1))));
              float v260_data = r2[2];
              float v264_data = ir4[0];
              ir4[0] = (v264_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 2))));
              float v270_data = ir4[1];
              ir4[1] = (v270_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 2))));
              float v276_data = ir4[2];
              ir4[2] = (v276_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v111_data, 2))));
              float v282_data = ir4[3];
              ir4[3] = (v282_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v117_data, 2))));
              float v288_data = ir4[4];
              ir4[4] = (v288_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v123_data, 2))));
              float v294_data = ir4[5];
              ir4[5] = (v294_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v129_data, 2))));
              float v300_data = ir4[6];
              ir4[6] = (v300_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 2))));
              float v306_data = ir4[7];
              ir4[7] = (v306_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 2))));
              float v312_data = ir4[8];
              ir4[8] = (v312_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 2))));
              float v318_data = ir4[9];
              ir4[9] = (v318_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 2))));
              float v324_data = ir4[10];
              ir4[10] = (v324_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 2))));
              float v330_data = ir4[11];
              ir4[11] = (v330_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 2))));
              float v336_data = ir4[12];
              ir4[12] = (v336_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 2))));
              float v341_data = r2[3];
              float v345_data = ir4[0];
              ir4[0] = (v345_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 3))));
              float v351_data = ir4[1];
              ir4[1] = (v351_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 3))));
              float v357_data = ir4[2];
              ir4[2] = (v357_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v111_data, 3))));
              float v363_data = ir4[3];
              ir4[3] = (v363_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v117_data, 3))));
              float v369_data = ir4[4];
              ir4[4] = (v369_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v123_data, 3))));
              float v375_data = ir4[5];
              ir4[5] = (v375_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v129_data, 3))));
              float v381_data = ir4[6];
              ir4[6] = (v381_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 3))));
              float v387_data = ir4[7];
              ir4[7] = (v387_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 3))));
              float v393_data = ir4[8];
              ir4[8] = (v393_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 3))));
              float v399_data = ir4[9];
              ir4[9] = (v399_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 3))));
              float v405_data = ir4[10];
              ir4[10] = (v405_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 3))));
              float v411_data = ir4[11];
              ir4[11] = (v411_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 3))));
              float v417_data = ir4[12];
              ir4[12] = (v417_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 3))));
              float v422_data = r2[4];
              float v426_data = ir4[0];
              ir4[0] = (v426_data + (v422_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 4))));
              float v432_data = ir4[1];
              ir4[1] = (v432_data + (v422_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 4))));
              float v438_data = ir4[2];
              ir4[2] = (v438_data + (v422_data * (sycl::group_broadcast(item.get_sub_group(), v111_data, 4))));
              float v444_data = ir4[3];
              ir4[3] = (v444_data + (v422_data * (sycl::group_broadcast(item.get_sub_group(), v117_data, 4))));
              float v450_data = ir4[4];
              ir4[4] = (v450_data + (v422_data * (sycl::group_broadcast(item.get_sub_group(), v123_data, 4))));
              float v456_data = ir4[5];
              ir4[5] = (v456_data + (v422_data * (sycl::group_broadcast(item.get_sub_group(), v129_data, 4))));
              float v462_data = ir4[6];
              ir4[6] = (v462_data + (v422_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 4))));
              float v468_data = ir4[7];
              ir4[7] = (v468_data + (v422_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 4))));
              float v474_data = ir4[8];
              ir4[8] = (v474_data + (v422_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 4))));
              float v480_data = ir4[9];
              ir4[9] = (v480_data + (v422_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 4))));
              float v486_data = ir4[10];
              ir4[10] = (v486_data + (v422_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 4))));
              float v492_data = ir4[11];
              ir4[11] = (v492_data + (v422_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 4))));
              float v498_data = ir4[12];
              ir4[12] = (v498_data + (v422_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 4))));
              float v503_data = r2[5];
              float v507_data = ir4[0];
              ir4[0] = (v507_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 5))));
              float v513_data = ir4[1];
              ir4[1] = (v513_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 5))));
              float v519_data = ir4[2];
              ir4[2] = (v519_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v111_data, 5))));
              float v525_data = ir4[3];
              ir4[3] = (v525_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v117_data, 5))));
              float v531_data = ir4[4];
              ir4[4] = (v531_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v123_data, 5))));
              float v537_data = ir4[5];
              ir4[5] = (v537_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v129_data, 5))));
              float v543_data = ir4[6];
              ir4[6] = (v543_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 5))));
              float v549_data = ir4[7];
              ir4[7] = (v549_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 5))));
              float v555_data = ir4[8];
              ir4[8] = (v555_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 5))));
              float v561_data = ir4[9];
              ir4[9] = (v561_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 5))));
              float v567_data = ir4[10];
              ir4[10] = (v567_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 5))));
              float v573_data = ir4[11];
              ir4[11] = (v573_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 5))));
              float v579_data = ir4[12];
              ir4[12] = (v579_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 5))));
              float v584_data = r2[6];
              float v588_data = ir4[0];
              ir4[0] = (v588_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 6))));
              float v594_data = ir4[1];
              ir4[1] = (v594_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 6))));
              float v600_data = ir4[2];
              ir4[2] = (v600_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v111_data, 6))));
              float v606_data = ir4[3];
              ir4[3] = (v606_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v117_data, 6))));
              float v612_data = ir4[4];
              ir4[4] = (v612_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v123_data, 6))));
              float v618_data = ir4[5];
              ir4[5] = (v618_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v129_data, 6))));
              float v624_data = ir4[6];
              ir4[6] = (v624_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 6))));
              float v630_data = ir4[7];
              ir4[7] = (v630_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 6))));
              float v636_data = ir4[8];
              ir4[8] = (v636_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 6))));
              float v642_data = ir4[9];
              ir4[9] = (v642_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 6))));
              float v648_data = ir4[10];
              ir4[10] = (v648_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 6))));
              float v654_data = ir4[11];
              ir4[11] = (v654_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 6))));
              float v660_data = ir4[12];
              ir4[12] = (v660_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 6))));
              float v665_data = r2[7];
              float v669_data = ir4[0];
              ir4[0] = (v669_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 7))));
              float v675_data = ir4[1];
              ir4[1] = (v675_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 7))));
              float v681_data = ir4[2];
              ir4[2] = (v681_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v111_data, 7))));
              float v687_data = ir4[3];
              ir4[3] = (v687_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v117_data, 7))));
              float v693_data = ir4[4];
              ir4[4] = (v693_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v123_data, 7))));
              float v699_data = ir4[5];
              ir4[5] = (v699_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v129_data, 7))));
              float v705_data = ir4[6];
              ir4[6] = (v705_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 7))));
              float v711_data = ir4[7];
              ir4[7] = (v711_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 7))));
              float v717_data = ir4[8];
              ir4[8] = (v717_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 7))));
              float v723_data = ir4[9];
              ir4[9] = (v723_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 7))));
              float v729_data = ir4[10];
              ir4[10] = (v729_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 7))));
              float v735_data = ir4[11];
              ir4[11] = (v735_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 7))));
              float v741_data = ir4[12];
              ir4[12] = (v741_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 7))));
              float v746_data = r2[8];
              float v750_data = ir4[0];
              ir4[0] = (v750_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 8))));
              float v756_data = ir4[1];
              ir4[1] = (v756_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 8))));
              float v762_data = ir4[2];
              ir4[2] = (v762_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v111_data, 8))));
              float v768_data = ir4[3];
              ir4[3] = (v768_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v117_data, 8))));
              float v774_data = ir4[4];
              ir4[4] = (v774_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v123_data, 8))));
              float v780_data = ir4[5];
              ir4[5] = (v780_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v129_data, 8))));
              float v786_data = ir4[6];
              ir4[6] = (v786_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 8))));
              float v792_data = ir4[7];
              ir4[7] = (v792_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 8))));
              float v798_data = ir4[8];
              ir4[8] = (v798_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 8))));
              float v804_data = ir4[9];
              ir4[9] = (v804_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 8))));
              float v810_data = ir4[10];
              ir4[10] = (v810_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 8))));
              float v816_data = ir4[11];
              ir4[11] = (v816_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 8))));
              float v822_data = ir4[12];
              ir4[12] = (v822_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 8))));
              float v827_data = r2[9];
              float v831_data = ir4[0];
              ir4[0] = (v831_data + (v827_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 9))));
              float v837_data = ir4[1];
              ir4[1] = (v837_data + (v827_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 9))));
              float v843_data = ir4[2];
              ir4[2] = (v843_data + (v827_data * (sycl::group_broadcast(item.get_sub_group(), v111_data, 9))));
              float v849_data = ir4[3];
              ir4[3] = (v849_data + (v827_data * (sycl::group_broadcast(item.get_sub_group(), v117_data, 9))));
              float v855_data = ir4[4];
              ir4[4] = (v855_data + (v827_data * (sycl::group_broadcast(item.get_sub_group(), v123_data, 9))));
              float v861_data = ir4[5];
              ir4[5] = (v861_data + (v827_data * (sycl::group_broadcast(item.get_sub_group(), v129_data, 9))));
              float v867_data = ir4[6];
              ir4[6] = (v867_data + (v827_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 9))));
              float v873_data = ir4[7];
              ir4[7] = (v873_data + (v827_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 9))));
              float v879_data = ir4[8];
              ir4[8] = (v879_data + (v827_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 9))));
              float v885_data = ir4[9];
              ir4[9] = (v885_data + (v827_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 9))));
              float v891_data = ir4[10];
              ir4[10] = (v891_data + (v827_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 9))));
              float v897_data = ir4[11];
              ir4[11] = (v897_data + (v827_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 9))));
              float v903_data = ir4[12];
              ir4[12] = (v903_data + (v827_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 9))));
              float v908_data = r2[10];
              float v912_data = ir4[0];
              ir4[0] = (v912_data + (v908_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 10))));
              float v918_data = ir4[1];
              ir4[1] = (v918_data + (v908_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 10))));
              float v924_data = ir4[2];
              ir4[2] = (v924_data + (v908_data * (sycl::group_broadcast(item.get_sub_group(), v111_data, 10))));
              float v930_data = ir4[3];
              ir4[3] = (v930_data + (v908_data * (sycl::group_broadcast(item.get_sub_group(), v117_data, 10))));
              float v936_data = ir4[4];
              ir4[4] = (v936_data + (v908_data * (sycl::group_broadcast(item.get_sub_group(), v123_data, 10))));
              float v942_data = ir4[5];
              ir4[5] = (v942_data + (v908_data * (sycl::group_broadcast(item.get_sub_group(), v129_data, 10))));
              float v948_data = ir4[6];
              ir4[6] = (v948_data + (v908_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 10))));
              float v954_data = ir4[7];
              ir4[7] = (v954_data + (v908_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 10))));
              float v960_data = ir4[8];
              ir4[8] = (v960_data + (v908_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 10))));
              float v966_data = ir4[9];
              ir4[9] = (v966_data + (v908_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 10))));
              float v972_data = ir4[10];
              ir4[10] = (v972_data + (v908_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 10))));
              float v978_data = ir4[11];
              ir4[11] = (v978_data + (v908_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 10))));
              float v984_data = ir4[12];
              ir4[12] = (v984_data + (v908_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 10))));
              float v989_data = r2[11];
              float v993_data = ir4[0];
              ir4[0] = (v993_data + (v989_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 11))));
              float v999_data = ir4[1];
              ir4[1] = (v999_data + (v989_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 11))));
              float v1005_data = ir4[2];
              ir4[2] = (v1005_data + (v989_data * (sycl::group_broadcast(item.get_sub_group(), v111_data, 11))));
              float v1011_data = ir4[3];
              ir4[3] = (v1011_data + (v989_data * (sycl::group_broadcast(item.get_sub_group(), v117_data, 11))));
              float v1017_data = ir4[4];
              ir4[4] = (v1017_data + (v989_data * (sycl::group_broadcast(item.get_sub_group(), v123_data, 11))));
              float v1023_data = ir4[5];
              ir4[5] = (v1023_data + (v989_data * (sycl::group_broadcast(item.get_sub_group(), v129_data, 11))));
              float v1029_data = ir4[6];
              ir4[6] = (v1029_data + (v989_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 11))));
              float v1035_data = ir4[7];
              ir4[7] = (v1035_data + (v989_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 11))));
              float v1041_data = ir4[8];
              ir4[8] = (v1041_data + (v989_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 11))));
              float v1047_data = ir4[9];
              ir4[9] = (v1047_data + (v989_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 11))));
              float v1053_data = ir4[10];
              ir4[10] = (v1053_data + (v989_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 11))));
              float v1059_data = ir4[11];
              ir4[11] = (v1059_data + (v989_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 11))));
              float v1065_data = ir4[12];
              ir4[12] = (v1065_data + (v989_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 11))));
              #pragma unroll
              for (int32_t v1070_n0 = 0; v1070_n0 < 1; ++v1070_n0) {
                #pragma unroll
                for (int32_t v1071_n1 = 0; v1071_n1 < 13; ++v1071_n1) {
                  int32_t v1072_a = v1070_n0 + v1071_n1;
                  float v1073_data = ir4[v1072_a];
                  float v1075_data = r1[v1072_a];
                  r4[v1072_a] = (v1075_data + v1073_data);
                }
              }
              float r5[1]{};
              // r5 = +(r4) + None
              // [(0, 32), (0, 1)] []
              float ir5[1]{};
              float v1083_data = r4[4];
              float v1084_data = ir5[0];
              ir5[0] = (v1084_data + v1083_data);
              #pragma unroll
              for (int32_t v1089_n0 = 0; v1089_n0 < 1; ++v1089_n0) {
                #pragma unroll
                for (int32_t v1090_n1 = 0; v1090_n1 < 1; ++v1090_n1) {
                  int32_t v1091_a = v1089_n0 + v1090_n1;
                  float v1092_data = ir5[v1091_a];
                  r5[v1091_a] = v1092_data;
                }
              }
              // glb_m0 = store{r>g}(r5);
              #pragma unroll
              for (int32_t v1097_i0 = 0; v1097_i0 < 1; ++v1097_i0) {
                int32_t v1105_lead = v8_lead + (v1097_i0 * 32);
                #pragma unroll
                for (int32_t v1098_i1 = 0; v1098_i1 < 1; ++v1098_i1) {
                  float v1100_data = r5[(v1097_i0 + v1098_i1)];
                  glb_m0[(v1105_lead + ((v1098_i1 + 4) * 32))] = v1100_data;
                }
              }
              float r6[13]{};
              // r6 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v1113_i0 = 0; v1113_i0 < 1; ++v1113_i0) {
                int32_t v1119_lead = v8_lead + (v1113_i0 * 32);
                #pragma unroll
                for (int32_t v1114_i1 = 0; v1114_i1 < 13; ++v1114_i1) {
                  float v1122_data = glb_m0[(v1119_lead + (v1114_i1 * 32))];
                  r6[(v1113_i0 + v1114_i1)] = v1122_data;
                }
              }
              float r7[13]{};
              // r7 = load{g>r}(glb_m4);
              if (v8_lead < 13) {
                #pragma unroll
                for (int32_t v1129_i1 = 0; v1129_i1 < 13; ++v1129_i1) {
                  float v1137_data = glb_m4[(v8_lead + (v1129_i1 * 13))];
                  r7[v1129_i1] = v1137_data;
                }
              }
              // wait(r6 = load{g>r}(glb_m0););
              // wait(r7 = load{g>r}(glb_m4););
              float r8[13]{};
              // r8 = +(r6 * r7) + None
              // [(0, 32), (0, 13)] [(0, 13)]
              float ir8[13]{};
              float v1144_data = r6[0];
              float v1145_data = r7[0];
              float v1148_data = ir8[0];
              ir8[0] = (v1148_data + (v1144_data * (sycl::group_broadcast(item.get_sub_group(), v1145_data, 0))));
              float v1151_data = r7[1];
              float v1154_data = ir8[1];
              ir8[1] = (v1154_data + (v1144_data * (sycl::group_broadcast(item.get_sub_group(), v1151_data, 0))));
              float v1157_data = r7[2];
              float v1160_data = ir8[2];
              ir8[2] = (v1160_data + (v1144_data * (sycl::group_broadcast(item.get_sub_group(), v1157_data, 0))));
              float v1163_data = r7[3];
              float v1166_data = ir8[3];
              ir8[3] = (v1166_data + (v1144_data * (sycl::group_broadcast(item.get_sub_group(), v1163_data, 0))));
              float v1169_data = r7[4];
              float v1172_data = ir8[4];
              ir8[4] = (v1172_data + (v1144_data * (sycl::group_broadcast(item.get_sub_group(), v1169_data, 0))));
              float v1175_data = r7[5];
              float v1178_data = ir8[5];
              ir8[5] = (v1178_data + (v1144_data * (sycl::group_broadcast(item.get_sub_group(), v1175_data, 0))));
              float v1181_data = r7[6];
              float v1184_data = ir8[6];
              ir8[6] = (v1184_data + (v1144_data * (sycl::group_broadcast(item.get_sub_group(), v1181_data, 0))));
              float v1187_data = r7[7];
              float v1190_data = ir8[7];
              ir8[7] = (v1190_data + (v1144_data * (sycl::group_broadcast(item.get_sub_group(), v1187_data, 0))));
              float v1193_data = r7[8];
              float v1196_data = ir8[8];
              ir8[8] = (v1196_data + (v1144_data * (sycl::group_broadcast(item.get_sub_group(), v1193_data, 0))));
              float v1199_data = r7[9];
              float v1202_data = ir8[9];
              ir8[9] = (v1202_data + (v1144_data * (sycl::group_broadcast(item.get_sub_group(), v1199_data, 0))));
              float v1205_data = r7[10];
              float v1208_data = ir8[10];
              ir8[10] = (v1208_data + (v1144_data * (sycl::group_broadcast(item.get_sub_group(), v1205_data, 0))));
              float v1211_data = r7[11];
              float v1214_data = ir8[11];
              ir8[11] = (v1214_data + (v1144_data * (sycl::group_broadcast(item.get_sub_group(), v1211_data, 0))));
              float v1217_data = r7[12];
              float v1220_data = ir8[12];
              ir8[12] = (v1220_data + (v1144_data * (sycl::group_broadcast(item.get_sub_group(), v1217_data, 0))));
              float v1225_data = r6[1];
              float v1229_data = ir8[0];
              ir8[0] = (v1229_data + (v1225_data * (sycl::group_broadcast(item.get_sub_group(), v1145_data, 1))));
              float v1235_data = ir8[1];
              ir8[1] = (v1235_data + (v1225_data * (sycl::group_broadcast(item.get_sub_group(), v1151_data, 1))));
              float v1241_data = ir8[2];
              ir8[2] = (v1241_data + (v1225_data * (sycl::group_broadcast(item.get_sub_group(), v1157_data, 1))));
              float v1247_data = ir8[3];
              ir8[3] = (v1247_data + (v1225_data * (sycl::group_broadcast(item.get_sub_group(), v1163_data, 1))));
              float v1253_data = ir8[4];
              ir8[4] = (v1253_data + (v1225_data * (sycl::group_broadcast(item.get_sub_group(), v1169_data, 1))));
              float v1259_data = ir8[5];
              ir8[5] = (v1259_data + (v1225_data * (sycl::group_broadcast(item.get_sub_group(), v1175_data, 1))));
              float v1265_data = ir8[6];
              ir8[6] = (v1265_data + (v1225_data * (sycl::group_broadcast(item.get_sub_group(), v1181_data, 1))));
              float v1271_data = ir8[7];
              ir8[7] = (v1271_data + (v1225_data * (sycl::group_broadcast(item.get_sub_group(), v1187_data, 1))));
              float v1277_data = ir8[8];
              ir8[8] = (v1277_data + (v1225_data * (sycl::group_broadcast(item.get_sub_group(), v1193_data, 1))));
              float v1283_data = ir8[9];
              ir8[9] = (v1283_data + (v1225_data * (sycl::group_broadcast(item.get_sub_group(), v1199_data, 1))));
              float v1289_data = ir8[10];
              ir8[10] = (v1289_data + (v1225_data * (sycl::group_broadcast(item.get_sub_group(), v1205_data, 1))));
              float v1295_data = ir8[11];
              ir8[11] = (v1295_data + (v1225_data * (sycl::group_broadcast(item.get_sub_group(), v1211_data, 1))));
              float v1301_data = ir8[12];
              ir8[12] = (v1301_data + (v1225_data * (sycl::group_broadcast(item.get_sub_group(), v1217_data, 1))));
              float v1306_data = r6[2];
              float v1310_data = ir8[0];
              ir8[0] = (v1310_data + (v1306_data * (sycl::group_broadcast(item.get_sub_group(), v1145_data, 2))));
              float v1316_data = ir8[1];
              ir8[1] = (v1316_data + (v1306_data * (sycl::group_broadcast(item.get_sub_group(), v1151_data, 2))));
              float v1322_data = ir8[2];
              ir8[2] = (v1322_data + (v1306_data * (sycl::group_broadcast(item.get_sub_group(), v1157_data, 2))));
              float v1328_data = ir8[3];
              ir8[3] = (v1328_data + (v1306_data * (sycl::group_broadcast(item.get_sub_group(), v1163_data, 2))));
              float v1334_data = ir8[4];
              ir8[4] = (v1334_data + (v1306_data * (sycl::group_broadcast(item.get_sub_group(), v1169_data, 2))));
              float v1340_data = ir8[5];
              ir8[5] = (v1340_data + (v1306_data * (sycl::group_broadcast(item.get_sub_group(), v1175_data, 2))));
              float v1346_data = ir8[6];
              ir8[6] = (v1346_data + (v1306_data * (sycl::group_broadcast(item.get_sub_group(), v1181_data, 2))));
              float v1352_data = ir8[7];
              ir8[7] = (v1352_data + (v1306_data * (sycl::group_broadcast(item.get_sub_group(), v1187_data, 2))));
              float v1358_data = ir8[8];
              ir8[8] = (v1358_data + (v1306_data * (sycl::group_broadcast(item.get_sub_group(), v1193_data, 2))));
              float v1364_data = ir8[9];
              ir8[9] = (v1364_data + (v1306_data * (sycl::group_broadcast(item.get_sub_group(), v1199_data, 2))));
              float v1370_data = ir8[10];
              ir8[10] = (v1370_data + (v1306_data * (sycl::group_broadcast(item.get_sub_group(), v1205_data, 2))));
              float v1376_data = ir8[11];
              ir8[11] = (v1376_data + (v1306_data * (sycl::group_broadcast(item.get_sub_group(), v1211_data, 2))));
              float v1382_data = ir8[12];
              ir8[12] = (v1382_data + (v1306_data * (sycl::group_broadcast(item.get_sub_group(), v1217_data, 2))));
              float v1387_data = r6[3];
              float v1391_data = ir8[0];
              ir8[0] = (v1391_data + (v1387_data * (sycl::group_broadcast(item.get_sub_group(), v1145_data, 3))));
              float v1397_data = ir8[1];
              ir8[1] = (v1397_data + (v1387_data * (sycl::group_broadcast(item.get_sub_group(), v1151_data, 3))));
              float v1403_data = ir8[2];
              ir8[2] = (v1403_data + (v1387_data * (sycl::group_broadcast(item.get_sub_group(), v1157_data, 3))));
              float v1409_data = ir8[3];
              ir8[3] = (v1409_data + (v1387_data * (sycl::group_broadcast(item.get_sub_group(), v1163_data, 3))));
              float v1415_data = ir8[4];
              ir8[4] = (v1415_data + (v1387_data * (sycl::group_broadcast(item.get_sub_group(), v1169_data, 3))));
              float v1421_data = ir8[5];
              ir8[5] = (v1421_data + (v1387_data * (sycl::group_broadcast(item.get_sub_group(), v1175_data, 3))));
              float v1427_data = ir8[6];
              ir8[6] = (v1427_data + (v1387_data * (sycl::group_broadcast(item.get_sub_group(), v1181_data, 3))));
              float v1433_data = ir8[7];
              ir8[7] = (v1433_data + (v1387_data * (sycl::group_broadcast(item.get_sub_group(), v1187_data, 3))));
              float v1439_data = ir8[8];
              ir8[8] = (v1439_data + (v1387_data * (sycl::group_broadcast(item.get_sub_group(), v1193_data, 3))));
              float v1445_data = ir8[9];
              ir8[9] = (v1445_data + (v1387_data * (sycl::group_broadcast(item.get_sub_group(), v1199_data, 3))));
              float v1451_data = ir8[10];
              ir8[10] = (v1451_data + (v1387_data * (sycl::group_broadcast(item.get_sub_group(), v1205_data, 3))));
              float v1457_data = ir8[11];
              ir8[11] = (v1457_data + (v1387_data * (sycl::group_broadcast(item.get_sub_group(), v1211_data, 3))));
              float v1463_data = ir8[12];
              ir8[12] = (v1463_data + (v1387_data * (sycl::group_broadcast(item.get_sub_group(), v1217_data, 3))));
              float v1468_data = r6[4];
              float v1472_data = ir8[0];
              ir8[0] = (v1472_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1145_data, 4))));
              float v1478_data = ir8[1];
              ir8[1] = (v1478_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1151_data, 4))));
              float v1484_data = ir8[2];
              ir8[2] = (v1484_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1157_data, 4))));
              float v1490_data = ir8[3];
              ir8[3] = (v1490_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1163_data, 4))));
              float v1496_data = ir8[4];
              ir8[4] = (v1496_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1169_data, 4))));
              float v1502_data = ir8[5];
              ir8[5] = (v1502_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1175_data, 4))));
              float v1508_data = ir8[6];
              ir8[6] = (v1508_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1181_data, 4))));
              float v1514_data = ir8[7];
              ir8[7] = (v1514_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1187_data, 4))));
              float v1520_data = ir8[8];
              ir8[8] = (v1520_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1193_data, 4))));
              float v1526_data = ir8[9];
              ir8[9] = (v1526_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1199_data, 4))));
              float v1532_data = ir8[10];
              ir8[10] = (v1532_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1205_data, 4))));
              float v1538_data = ir8[11];
              ir8[11] = (v1538_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1211_data, 4))));
              float v1544_data = ir8[12];
              ir8[12] = (v1544_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v1217_data, 4))));
              float v1549_data = r6[5];
              float v1553_data = ir8[0];
              ir8[0] = (v1553_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1145_data, 5))));
              float v1559_data = ir8[1];
              ir8[1] = (v1559_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1151_data, 5))));
              float v1565_data = ir8[2];
              ir8[2] = (v1565_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1157_data, 5))));
              float v1571_data = ir8[3];
              ir8[3] = (v1571_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1163_data, 5))));
              float v1577_data = ir8[4];
              ir8[4] = (v1577_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1169_data, 5))));
              float v1583_data = ir8[5];
              ir8[5] = (v1583_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1175_data, 5))));
              float v1589_data = ir8[6];
              ir8[6] = (v1589_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1181_data, 5))));
              float v1595_data = ir8[7];
              ir8[7] = (v1595_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1187_data, 5))));
              float v1601_data = ir8[8];
              ir8[8] = (v1601_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1193_data, 5))));
              float v1607_data = ir8[9];
              ir8[9] = (v1607_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1199_data, 5))));
              float v1613_data = ir8[10];
              ir8[10] = (v1613_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1205_data, 5))));
              float v1619_data = ir8[11];
              ir8[11] = (v1619_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1211_data, 5))));
              float v1625_data = ir8[12];
              ir8[12] = (v1625_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1217_data, 5))));
              float v1630_data = r6[6];
              float v1634_data = ir8[0];
              ir8[0] = (v1634_data + (v1630_data * (sycl::group_broadcast(item.get_sub_group(), v1145_data, 6))));
              float v1640_data = ir8[1];
              ir8[1] = (v1640_data + (v1630_data * (sycl::group_broadcast(item.get_sub_group(), v1151_data, 6))));
              float v1646_data = ir8[2];
              ir8[2] = (v1646_data + (v1630_data * (sycl::group_broadcast(item.get_sub_group(), v1157_data, 6))));
              float v1652_data = ir8[3];
              ir8[3] = (v1652_data + (v1630_data * (sycl::group_broadcast(item.get_sub_group(), v1163_data, 6))));
              float v1658_data = ir8[4];
              ir8[4] = (v1658_data + (v1630_data * (sycl::group_broadcast(item.get_sub_group(), v1169_data, 6))));
              float v1664_data = ir8[5];
              ir8[5] = (v1664_data + (v1630_data * (sycl::group_broadcast(item.get_sub_group(), v1175_data, 6))));
              float v1670_data = ir8[6];
              ir8[6] = (v1670_data + (v1630_data * (sycl::group_broadcast(item.get_sub_group(), v1181_data, 6))));
              float v1676_data = ir8[7];
              ir8[7] = (v1676_data + (v1630_data * (sycl::group_broadcast(item.get_sub_group(), v1187_data, 6))));
              float v1682_data = ir8[8];
              ir8[8] = (v1682_data + (v1630_data * (sycl::group_broadcast(item.get_sub_group(), v1193_data, 6))));
              float v1688_data = ir8[9];
              ir8[9] = (v1688_data + (v1630_data * (sycl::group_broadcast(item.get_sub_group(), v1199_data, 6))));
              float v1694_data = ir8[10];
              ir8[10] = (v1694_data + (v1630_data * (sycl::group_broadcast(item.get_sub_group(), v1205_data, 6))));
              float v1700_data = ir8[11];
              ir8[11] = (v1700_data + (v1630_data * (sycl::group_broadcast(item.get_sub_group(), v1211_data, 6))));
              float v1706_data = ir8[12];
              ir8[12] = (v1706_data + (v1630_data * (sycl::group_broadcast(item.get_sub_group(), v1217_data, 6))));
              float v1711_data = r6[7];
              float v1715_data = ir8[0];
              ir8[0] = (v1715_data + (v1711_data * (sycl::group_broadcast(item.get_sub_group(), v1145_data, 7))));
              float v1721_data = ir8[1];
              ir8[1] = (v1721_data + (v1711_data * (sycl::group_broadcast(item.get_sub_group(), v1151_data, 7))));
              float v1727_data = ir8[2];
              ir8[2] = (v1727_data + (v1711_data * (sycl::group_broadcast(item.get_sub_group(), v1157_data, 7))));
              float v1733_data = ir8[3];
              ir8[3] = (v1733_data + (v1711_data * (sycl::group_broadcast(item.get_sub_group(), v1163_data, 7))));
              float v1739_data = ir8[4];
              ir8[4] = (v1739_data + (v1711_data * (sycl::group_broadcast(item.get_sub_group(), v1169_data, 7))));
              float v1745_data = ir8[5];
              ir8[5] = (v1745_data + (v1711_data * (sycl::group_broadcast(item.get_sub_group(), v1175_data, 7))));
              float v1751_data = ir8[6];
              ir8[6] = (v1751_data + (v1711_data * (sycl::group_broadcast(item.get_sub_group(), v1181_data, 7))));
              float v1757_data = ir8[7];
              ir8[7] = (v1757_data + (v1711_data * (sycl::group_broadcast(item.get_sub_group(), v1187_data, 7))));
              float v1763_data = ir8[8];
              ir8[8] = (v1763_data + (v1711_data * (sycl::group_broadcast(item.get_sub_group(), v1193_data, 7))));
              float v1769_data = ir8[9];
              ir8[9] = (v1769_data + (v1711_data * (sycl::group_broadcast(item.get_sub_group(), v1199_data, 7))));
              float v1775_data = ir8[10];
              ir8[10] = (v1775_data + (v1711_data * (sycl::group_broadcast(item.get_sub_group(), v1205_data, 7))));
              float v1781_data = ir8[11];
              ir8[11] = (v1781_data + (v1711_data * (sycl::group_broadcast(item.get_sub_group(), v1211_data, 7))));
              float v1787_data = ir8[12];
              ir8[12] = (v1787_data + (v1711_data * (sycl::group_broadcast(item.get_sub_group(), v1217_data, 7))));
              float v1792_data = r6[8];
              float v1796_data = ir8[0];
              ir8[0] = (v1796_data + (v1792_data * (sycl::group_broadcast(item.get_sub_group(), v1145_data, 8))));
              float v1802_data = ir8[1];
              ir8[1] = (v1802_data + (v1792_data * (sycl::group_broadcast(item.get_sub_group(), v1151_data, 8))));
              float v1808_data = ir8[2];
              ir8[2] = (v1808_data + (v1792_data * (sycl::group_broadcast(item.get_sub_group(), v1157_data, 8))));
              float v1814_data = ir8[3];
              ir8[3] = (v1814_data + (v1792_data * (sycl::group_broadcast(item.get_sub_group(), v1163_data, 8))));
              float v1820_data = ir8[4];
              ir8[4] = (v1820_data + (v1792_data * (sycl::group_broadcast(item.get_sub_group(), v1169_data, 8))));
              float v1826_data = ir8[5];
              ir8[5] = (v1826_data + (v1792_data * (sycl::group_broadcast(item.get_sub_group(), v1175_data, 8))));
              float v1832_data = ir8[6];
              ir8[6] = (v1832_data + (v1792_data * (sycl::group_broadcast(item.get_sub_group(), v1181_data, 8))));
              float v1838_data = ir8[7];
              ir8[7] = (v1838_data + (v1792_data * (sycl::group_broadcast(item.get_sub_group(), v1187_data, 8))));
              float v1844_data = ir8[8];
              ir8[8] = (v1844_data + (v1792_data * (sycl::group_broadcast(item.get_sub_group(), v1193_data, 8))));
              float v1850_data = ir8[9];
              ir8[9] = (v1850_data + (v1792_data * (sycl::group_broadcast(item.get_sub_group(), v1199_data, 8))));
              float v1856_data = ir8[10];
              ir8[10] = (v1856_data + (v1792_data * (sycl::group_broadcast(item.get_sub_group(), v1205_data, 8))));
              float v1862_data = ir8[11];
              ir8[11] = (v1862_data + (v1792_data * (sycl::group_broadcast(item.get_sub_group(), v1211_data, 8))));
              float v1868_data = ir8[12];
              ir8[12] = (v1868_data + (v1792_data * (sycl::group_broadcast(item.get_sub_group(), v1217_data, 8))));
              float v1873_data = r6[9];
              float v1877_data = ir8[0];
              ir8[0] = (v1877_data + (v1873_data * (sycl::group_broadcast(item.get_sub_group(), v1145_data, 9))));
              float v1883_data = ir8[1];
              ir8[1] = (v1883_data + (v1873_data * (sycl::group_broadcast(item.get_sub_group(), v1151_data, 9))));
              float v1889_data = ir8[2];
              ir8[2] = (v1889_data + (v1873_data * (sycl::group_broadcast(item.get_sub_group(), v1157_data, 9))));
              float v1895_data = ir8[3];
              ir8[3] = (v1895_data + (v1873_data * (sycl::group_broadcast(item.get_sub_group(), v1163_data, 9))));
              float v1901_data = ir8[4];
              ir8[4] = (v1901_data + (v1873_data * (sycl::group_broadcast(item.get_sub_group(), v1169_data, 9))));
              float v1907_data = ir8[5];
              ir8[5] = (v1907_data + (v1873_data * (sycl::group_broadcast(item.get_sub_group(), v1175_data, 9))));
              float v1913_data = ir8[6];
              ir8[6] = (v1913_data + (v1873_data * (sycl::group_broadcast(item.get_sub_group(), v1181_data, 9))));
              float v1919_data = ir8[7];
              ir8[7] = (v1919_data + (v1873_data * (sycl::group_broadcast(item.get_sub_group(), v1187_data, 9))));
              float v1925_data = ir8[8];
              ir8[8] = (v1925_data + (v1873_data * (sycl::group_broadcast(item.get_sub_group(), v1193_data, 9))));
              float v1931_data = ir8[9];
              ir8[9] = (v1931_data + (v1873_data * (sycl::group_broadcast(item.get_sub_group(), v1199_data, 9))));
              float v1937_data = ir8[10];
              ir8[10] = (v1937_data + (v1873_data * (sycl::group_broadcast(item.get_sub_group(), v1205_data, 9))));
              float v1943_data = ir8[11];
              ir8[11] = (v1943_data + (v1873_data * (sycl::group_broadcast(item.get_sub_group(), v1211_data, 9))));
              float v1949_data = ir8[12];
              ir8[12] = (v1949_data + (v1873_data * (sycl::group_broadcast(item.get_sub_group(), v1217_data, 9))));
              float v1954_data = r6[10];
              float v1958_data = ir8[0];
              ir8[0] = (v1958_data + (v1954_data * (sycl::group_broadcast(item.get_sub_group(), v1145_data, 10))));
              float v1964_data = ir8[1];
              ir8[1] = (v1964_data + (v1954_data * (sycl::group_broadcast(item.get_sub_group(), v1151_data, 10))));
              float v1970_data = ir8[2];
              ir8[2] = (v1970_data + (v1954_data * (sycl::group_broadcast(item.get_sub_group(), v1157_data, 10))));
              float v1976_data = ir8[3];
              ir8[3] = (v1976_data + (v1954_data * (sycl::group_broadcast(item.get_sub_group(), v1163_data, 10))));
              float v1982_data = ir8[4];
              ir8[4] = (v1982_data + (v1954_data * (sycl::group_broadcast(item.get_sub_group(), v1169_data, 10))));
              float v1988_data = ir8[5];
              ir8[5] = (v1988_data + (v1954_data * (sycl::group_broadcast(item.get_sub_group(), v1175_data, 10))));
              float v1994_data = ir8[6];
              ir8[6] = (v1994_data + (v1954_data * (sycl::group_broadcast(item.get_sub_group(), v1181_data, 10))));
              float v2000_data = ir8[7];
              ir8[7] = (v2000_data + (v1954_data * (sycl::group_broadcast(item.get_sub_group(), v1187_data, 10))));
              float v2006_data = ir8[8];
              ir8[8] = (v2006_data + (v1954_data * (sycl::group_broadcast(item.get_sub_group(), v1193_data, 10))));
              float v2012_data = ir8[9];
              ir8[9] = (v2012_data + (v1954_data * (sycl::group_broadcast(item.get_sub_group(), v1199_data, 10))));
              float v2018_data = ir8[10];
              ir8[10] = (v2018_data + (v1954_data * (sycl::group_broadcast(item.get_sub_group(), v1205_data, 10))));
              float v2024_data = ir8[11];
              ir8[11] = (v2024_data + (v1954_data * (sycl::group_broadcast(item.get_sub_group(), v1211_data, 10))));
              float v2030_data = ir8[12];
              ir8[12] = (v2030_data + (v1954_data * (sycl::group_broadcast(item.get_sub_group(), v1217_data, 10))));
              float v2035_data = r6[11];
              float v2039_data = ir8[0];
              ir8[0] = (v2039_data + (v2035_data * (sycl::group_broadcast(item.get_sub_group(), v1145_data, 11))));
              float v2045_data = ir8[1];
              ir8[1] = (v2045_data + (v2035_data * (sycl::group_broadcast(item.get_sub_group(), v1151_data, 11))));
              float v2051_data = ir8[2];
              ir8[2] = (v2051_data + (v2035_data * (sycl::group_broadcast(item.get_sub_group(), v1157_data, 11))));
              float v2057_data = ir8[3];
              ir8[3] = (v2057_data + (v2035_data * (sycl::group_broadcast(item.get_sub_group(), v1163_data, 11))));
              float v2063_data = ir8[4];
              ir8[4] = (v2063_data + (v2035_data * (sycl::group_broadcast(item.get_sub_group(), v1169_data, 11))));
              float v2069_data = ir8[5];
              ir8[5] = (v2069_data + (v2035_data * (sycl::group_broadcast(item.get_sub_group(), v1175_data, 11))));
              float v2075_data = ir8[6];
              ir8[6] = (v2075_data + (v2035_data * (sycl::group_broadcast(item.get_sub_group(), v1181_data, 11))));
              float v2081_data = ir8[7];
              ir8[7] = (v2081_data + (v2035_data * (sycl::group_broadcast(item.get_sub_group(), v1187_data, 11))));
              float v2087_data = ir8[8];
              ir8[8] = (v2087_data + (v2035_data * (sycl::group_broadcast(item.get_sub_group(), v1193_data, 11))));
              float v2093_data = ir8[9];
              ir8[9] = (v2093_data + (v2035_data * (sycl::group_broadcast(item.get_sub_group(), v1199_data, 11))));
              float v2099_data = ir8[10];
              ir8[10] = (v2099_data + (v2035_data * (sycl::group_broadcast(item.get_sub_group(), v1205_data, 11))));
              float v2105_data = ir8[11];
              ir8[11] = (v2105_data + (v2035_data * (sycl::group_broadcast(item.get_sub_group(), v1211_data, 11))));
              float v2111_data = ir8[12];
              ir8[12] = (v2111_data + (v2035_data * (sycl::group_broadcast(item.get_sub_group(), v1217_data, 11))));
              float v2116_data = r6[12];
              float v2120_data = ir8[0];
              ir8[0] = (v2120_data + (v2116_data * (sycl::group_broadcast(item.get_sub_group(), v1145_data, 12))));
              float v2126_data = ir8[1];
              ir8[1] = (v2126_data + (v2116_data * (sycl::group_broadcast(item.get_sub_group(), v1151_data, 12))));
              float v2132_data = ir8[2];
              ir8[2] = (v2132_data + (v2116_data * (sycl::group_broadcast(item.get_sub_group(), v1157_data, 12))));
              float v2138_data = ir8[3];
              ir8[3] = (v2138_data + (v2116_data * (sycl::group_broadcast(item.get_sub_group(), v1163_data, 12))));
              float v2144_data = ir8[4];
              ir8[4] = (v2144_data + (v2116_data * (sycl::group_broadcast(item.get_sub_group(), v1169_data, 12))));
              float v2150_data = ir8[5];
              ir8[5] = (v2150_data + (v2116_data * (sycl::group_broadcast(item.get_sub_group(), v1175_data, 12))));
              float v2156_data = ir8[6];
              ir8[6] = (v2156_data + (v2116_data * (sycl::group_broadcast(item.get_sub_group(), v1181_data, 12))));
              float v2162_data = ir8[7];
              ir8[7] = (v2162_data + (v2116_data * (sycl::group_broadcast(item.get_sub_group(), v1187_data, 12))));
              float v2168_data = ir8[8];
              ir8[8] = (v2168_data + (v2116_data * (sycl::group_broadcast(item.get_sub_group(), v1193_data, 12))));
              float v2174_data = ir8[9];
              ir8[9] = (v2174_data + (v2116_data * (sycl::group_broadcast(item.get_sub_group(), v1199_data, 12))));
              float v2180_data = ir8[10];
              ir8[10] = (v2180_data + (v2116_data * (sycl::group_broadcast(item.get_sub_group(), v1205_data, 12))));
              float v2186_data = ir8[11];
              ir8[11] = (v2186_data + (v2116_data * (sycl::group_broadcast(item.get_sub_group(), v1211_data, 12))));
              float v2192_data = ir8[12];
              ir8[12] = (v2192_data + (v2116_data * (sycl::group_broadcast(item.get_sub_group(), v1217_data, 12))));
              #pragma unroll
              for (int32_t v2197_n0 = 0; v2197_n0 < 1; ++v2197_n0) {
                #pragma unroll
                for (int32_t v2198_n1 = 0; v2198_n1 < 13; ++v2198_n1) {
                  int32_t v2199_a = v2197_n0 + v2198_n1;
                  float v2200_data = ir8[v2199_a];
                  r8[v2199_a] = v2200_data;
                }
              }
              // glb_m3 = store{r>g}(r8);
              #pragma unroll
              for (int32_t v2205_i0 = 0; v2205_i0 < 1; ++v2205_i0) {
                int32_t v2213_lead = v8_lead + (v2205_i0 * 32);
                #pragma unroll
                for (int32_t v2206_i1 = 0; v2206_i1 < 13; ++v2206_i1) {
                  float v2208_data = r8[(v2205_i0 + v2206_i1)];
                  glb_m3[(v2213_lead + (v2206_i1 * 32))] = v2208_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

