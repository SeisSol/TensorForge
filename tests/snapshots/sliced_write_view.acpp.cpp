// === base name ===
kernel_09b7925bd997bc7d

// === header ===
void launcher_kernel_09b7925bd997bc7d(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_09b7925bd997bc7d(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 1, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_09b7925bd997bc7d(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_09b7925bd997bc7d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 32×13(32×13) {0..32}×{0..13} strided
        // m1 32×13(32×13) {0..32}×{0..13} strided
        // m2 13×13(13×13) {0..13}×{0..13} strided
        // m3 32×13(32×13) {0..32}×{0..13} strided
        // m4 13×13(13×13) {0..13}×{0..13} strided
        // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..1})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{10..13})[0, -1]×m2 13×13(13×13) {0..13}×{0..13} strided({10..13}×{0..1})[-1, 1]
        // m3 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] = m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, -1]×m4 13×13(13×13) {0..13}×{0..13} strided({0..13}×{0..13})[-1, 1]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          for (size_t v0_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v0_batchId0 < numElements0; v0_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v1_ahead1 = v0_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v3_batchId1 = (v1_ahead1 < numElements0) ? v1_ahead1 : v0_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v0_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v0_batchId0 * 416 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v0_batchId0 * 416 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v0_batchId0 * 169 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v0_batchId0 * 416 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v0_batchId0 * 169 + 0 + m4_extraOffset];
              float r0[3]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v16_lead = item.get_local_id(0) % 32;
              #pragma unroll
              for (int32_t v17_i0 = 0; v17_i0 < 1; ++v17_i0) {
                int32_t v23_lead = v16_lead + (v17_i0 * 32);
                #pragma unroll
                for (int32_t v18_i1 = 10; v18_i1 < 13; ++v18_i1) {
                  float v26_data = glb_m1[(v23_lead + (v18_i1 * 32))];
                  r0[(v17_i0 + (v18_i1 - 10))] = v26_data;
                }
              }
              float r1[1]{};
              // r1 = load{g>r}(glb_m2);
              if ((v16_lead >= 10) && (v16_lead < 13)) {
                #pragma unroll
                for (int32_t v36_i1 = 8; v36_i1 < 9; ++v36_i1) {
                  float v44_data = glb_m2[(v16_lead + (v36_i1 * 13))];
                  r1[(v36_i1 - 8)] = v44_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[1]{};
              // r2 = +(r0 * r1) + None
              // [(0, 32), (0, 1)] [(10, 13)]
              float ir2[1]{};
              float v52_data = r0[0];
              float v53_data = r1[0];
              float v56_data = ir2[0];
              ir2[0] = (v56_data + (v52_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 10))));
              float v61_data = r0[1];
              float v65_data = ir2[0];
              ir2[0] = (v65_data + (v61_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 11))));
              float v70_data = r0[2];
              float v74_data = ir2[0];
              ir2[0] = (v74_data + (v70_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 12))));
              #pragma unroll
              for (int32_t v79_n0 = 0; v79_n0 < 1; ++v79_n0) {
                #pragma unroll
                for (int32_t v80_n1 = 0; v80_n1 < 1; ++v80_n1) {
                  int32_t v81_a = v79_n0 + v80_n1;
                  float v82_data = ir2[v81_a];
                  r2[v81_a] = v82_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v87_i0 = 0; v87_i0 < 1; ++v87_i0) {
                int32_t v95_lead = v16_lead + (v87_i0 * 32);
                #pragma unroll
                for (int32_t v88_i1 = 0; v88_i1 < 1; ++v88_i1) {
                  float v90_data = r2[(v87_i0 + v88_i1)];
                  glb_m0[(v95_lead + ((v88_i1 + 8) * 32))] = v90_data;
                }
              }
              float r3[13]{};
              // r3 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v103_i0 = 0; v103_i0 < 1; ++v103_i0) {
                int32_t v109_lead = v16_lead + (v103_i0 * 32);
                #pragma unroll
                for (int32_t v104_i1 = 0; v104_i1 < 13; ++v104_i1) {
                  float v112_data = glb_m0[(v109_lead + (v104_i1 * 32))];
                  r3[(v103_i0 + v104_i1)] = v112_data;
                }
              }
              float r4[13]{};
              // r4 = load{g>r}(glb_m4);
              if (v16_lead < 13) {
                #pragma unroll
                for (int32_t v119_i1 = 0; v119_i1 < 13; ++v119_i1) {
                  float v127_data = glb_m4[(v16_lead + (v119_i1 * 13))];
                  r4[v119_i1] = v127_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m0););
              // wait(r4 = load{g>r}(glb_m4););
              float r5[13]{};
              // r5 = +(r3 * r4) + None
              // [(0, 32), (0, 13)] [(0, 13)]
              float ir5[13]{};
              float v134_data = r3[0];
              float v135_data = r4[0];
              float v138_data = ir5[0];
              ir5[0] = (v138_data + (v134_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 0))));
              float v141_data = r4[1];
              float v144_data = ir5[1];
              ir5[1] = (v144_data + (v134_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 0))));
              float v147_data = r4[2];
              float v150_data = ir5[2];
              ir5[2] = (v150_data + (v134_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 0))));
              float v153_data = r4[3];
              float v156_data = ir5[3];
              ir5[3] = (v156_data + (v134_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 0))));
              float v159_data = r4[4];
              float v162_data = ir5[4];
              ir5[4] = (v162_data + (v134_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 0))));
              float v165_data = r4[5];
              float v168_data = ir5[5];
              ir5[5] = (v168_data + (v134_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 0))));
              float v171_data = r4[6];
              float v174_data = ir5[6];
              ir5[6] = (v174_data + (v134_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 0))));
              float v177_data = r4[7];
              float v180_data = ir5[7];
              ir5[7] = (v180_data + (v134_data * (sycl::group_broadcast(item.get_sub_group(), v177_data, 0))));
              float v183_data = r4[8];
              float v186_data = ir5[8];
              ir5[8] = (v186_data + (v134_data * (sycl::group_broadcast(item.get_sub_group(), v183_data, 0))));
              float v189_data = r4[9];
              float v192_data = ir5[9];
              ir5[9] = (v192_data + (v134_data * (sycl::group_broadcast(item.get_sub_group(), v189_data, 0))));
              float v195_data = r4[10];
              float v198_data = ir5[10];
              ir5[10] = (v198_data + (v134_data * (sycl::group_broadcast(item.get_sub_group(), v195_data, 0))));
              float v201_data = r4[11];
              float v204_data = ir5[11];
              ir5[11] = (v204_data + (v134_data * (sycl::group_broadcast(item.get_sub_group(), v201_data, 0))));
              float v207_data = r4[12];
              float v210_data = ir5[12];
              ir5[12] = (v210_data + (v134_data * (sycl::group_broadcast(item.get_sub_group(), v207_data, 0))));
              float v215_data = r3[1];
              float v219_data = ir5[0];
              ir5[0] = (v219_data + (v215_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 1))));
              float v225_data = ir5[1];
              ir5[1] = (v225_data + (v215_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 1))));
              float v231_data = ir5[2];
              ir5[2] = (v231_data + (v215_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 1))));
              float v237_data = ir5[3];
              ir5[3] = (v237_data + (v215_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 1))));
              float v243_data = ir5[4];
              ir5[4] = (v243_data + (v215_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 1))));
              float v249_data = ir5[5];
              ir5[5] = (v249_data + (v215_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 1))));
              float v255_data = ir5[6];
              ir5[6] = (v255_data + (v215_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 1))));
              float v261_data = ir5[7];
              ir5[7] = (v261_data + (v215_data * (sycl::group_broadcast(item.get_sub_group(), v177_data, 1))));
              float v267_data = ir5[8];
              ir5[8] = (v267_data + (v215_data * (sycl::group_broadcast(item.get_sub_group(), v183_data, 1))));
              float v273_data = ir5[9];
              ir5[9] = (v273_data + (v215_data * (sycl::group_broadcast(item.get_sub_group(), v189_data, 1))));
              float v279_data = ir5[10];
              ir5[10] = (v279_data + (v215_data * (sycl::group_broadcast(item.get_sub_group(), v195_data, 1))));
              float v285_data = ir5[11];
              ir5[11] = (v285_data + (v215_data * (sycl::group_broadcast(item.get_sub_group(), v201_data, 1))));
              float v291_data = ir5[12];
              ir5[12] = (v291_data + (v215_data * (sycl::group_broadcast(item.get_sub_group(), v207_data, 1))));
              float v296_data = r3[2];
              float v300_data = ir5[0];
              ir5[0] = (v300_data + (v296_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 2))));
              float v306_data = ir5[1];
              ir5[1] = (v306_data + (v296_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 2))));
              float v312_data = ir5[2];
              ir5[2] = (v312_data + (v296_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 2))));
              float v318_data = ir5[3];
              ir5[3] = (v318_data + (v296_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 2))));
              float v324_data = ir5[4];
              ir5[4] = (v324_data + (v296_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 2))));
              float v330_data = ir5[5];
              ir5[5] = (v330_data + (v296_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 2))));
              float v336_data = ir5[6];
              ir5[6] = (v336_data + (v296_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 2))));
              float v342_data = ir5[7];
              ir5[7] = (v342_data + (v296_data * (sycl::group_broadcast(item.get_sub_group(), v177_data, 2))));
              float v348_data = ir5[8];
              ir5[8] = (v348_data + (v296_data * (sycl::group_broadcast(item.get_sub_group(), v183_data, 2))));
              float v354_data = ir5[9];
              ir5[9] = (v354_data + (v296_data * (sycl::group_broadcast(item.get_sub_group(), v189_data, 2))));
              float v360_data = ir5[10];
              ir5[10] = (v360_data + (v296_data * (sycl::group_broadcast(item.get_sub_group(), v195_data, 2))));
              float v366_data = ir5[11];
              ir5[11] = (v366_data + (v296_data * (sycl::group_broadcast(item.get_sub_group(), v201_data, 2))));
              float v372_data = ir5[12];
              ir5[12] = (v372_data + (v296_data * (sycl::group_broadcast(item.get_sub_group(), v207_data, 2))));
              float v377_data = r3[3];
              float v381_data = ir5[0];
              ir5[0] = (v381_data + (v377_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 3))));
              float v387_data = ir5[1];
              ir5[1] = (v387_data + (v377_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 3))));
              float v393_data = ir5[2];
              ir5[2] = (v393_data + (v377_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 3))));
              float v399_data = ir5[3];
              ir5[3] = (v399_data + (v377_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 3))));
              float v405_data = ir5[4];
              ir5[4] = (v405_data + (v377_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 3))));
              float v411_data = ir5[5];
              ir5[5] = (v411_data + (v377_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 3))));
              float v417_data = ir5[6];
              ir5[6] = (v417_data + (v377_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 3))));
              float v423_data = ir5[7];
              ir5[7] = (v423_data + (v377_data * (sycl::group_broadcast(item.get_sub_group(), v177_data, 3))));
              float v429_data = ir5[8];
              ir5[8] = (v429_data + (v377_data * (sycl::group_broadcast(item.get_sub_group(), v183_data, 3))));
              float v435_data = ir5[9];
              ir5[9] = (v435_data + (v377_data * (sycl::group_broadcast(item.get_sub_group(), v189_data, 3))));
              float v441_data = ir5[10];
              ir5[10] = (v441_data + (v377_data * (sycl::group_broadcast(item.get_sub_group(), v195_data, 3))));
              float v447_data = ir5[11];
              ir5[11] = (v447_data + (v377_data * (sycl::group_broadcast(item.get_sub_group(), v201_data, 3))));
              float v453_data = ir5[12];
              ir5[12] = (v453_data + (v377_data * (sycl::group_broadcast(item.get_sub_group(), v207_data, 3))));
              float v458_data = r3[4];
              float v462_data = ir5[0];
              ir5[0] = (v462_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 4))));
              float v468_data = ir5[1];
              ir5[1] = (v468_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 4))));
              float v474_data = ir5[2];
              ir5[2] = (v474_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 4))));
              float v480_data = ir5[3];
              ir5[3] = (v480_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 4))));
              float v486_data = ir5[4];
              ir5[4] = (v486_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 4))));
              float v492_data = ir5[5];
              ir5[5] = (v492_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 4))));
              float v498_data = ir5[6];
              ir5[6] = (v498_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 4))));
              float v504_data = ir5[7];
              ir5[7] = (v504_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v177_data, 4))));
              float v510_data = ir5[8];
              ir5[8] = (v510_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v183_data, 4))));
              float v516_data = ir5[9];
              ir5[9] = (v516_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v189_data, 4))));
              float v522_data = ir5[10];
              ir5[10] = (v522_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v195_data, 4))));
              float v528_data = ir5[11];
              ir5[11] = (v528_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v201_data, 4))));
              float v534_data = ir5[12];
              ir5[12] = (v534_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v207_data, 4))));
              float v539_data = r3[5];
              float v543_data = ir5[0];
              ir5[0] = (v543_data + (v539_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 5))));
              float v549_data = ir5[1];
              ir5[1] = (v549_data + (v539_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 5))));
              float v555_data = ir5[2];
              ir5[2] = (v555_data + (v539_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 5))));
              float v561_data = ir5[3];
              ir5[3] = (v561_data + (v539_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 5))));
              float v567_data = ir5[4];
              ir5[4] = (v567_data + (v539_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 5))));
              float v573_data = ir5[5];
              ir5[5] = (v573_data + (v539_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 5))));
              float v579_data = ir5[6];
              ir5[6] = (v579_data + (v539_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 5))));
              float v585_data = ir5[7];
              ir5[7] = (v585_data + (v539_data * (sycl::group_broadcast(item.get_sub_group(), v177_data, 5))));
              float v591_data = ir5[8];
              ir5[8] = (v591_data + (v539_data * (sycl::group_broadcast(item.get_sub_group(), v183_data, 5))));
              float v597_data = ir5[9];
              ir5[9] = (v597_data + (v539_data * (sycl::group_broadcast(item.get_sub_group(), v189_data, 5))));
              float v603_data = ir5[10];
              ir5[10] = (v603_data + (v539_data * (sycl::group_broadcast(item.get_sub_group(), v195_data, 5))));
              float v609_data = ir5[11];
              ir5[11] = (v609_data + (v539_data * (sycl::group_broadcast(item.get_sub_group(), v201_data, 5))));
              float v615_data = ir5[12];
              ir5[12] = (v615_data + (v539_data * (sycl::group_broadcast(item.get_sub_group(), v207_data, 5))));
              float v620_data = r3[6];
              float v624_data = ir5[0];
              ir5[0] = (v624_data + (v620_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 6))));
              float v630_data = ir5[1];
              ir5[1] = (v630_data + (v620_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 6))));
              float v636_data = ir5[2];
              ir5[2] = (v636_data + (v620_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 6))));
              float v642_data = ir5[3];
              ir5[3] = (v642_data + (v620_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 6))));
              float v648_data = ir5[4];
              ir5[4] = (v648_data + (v620_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 6))));
              float v654_data = ir5[5];
              ir5[5] = (v654_data + (v620_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 6))));
              float v660_data = ir5[6];
              ir5[6] = (v660_data + (v620_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 6))));
              float v666_data = ir5[7];
              ir5[7] = (v666_data + (v620_data * (sycl::group_broadcast(item.get_sub_group(), v177_data, 6))));
              float v672_data = ir5[8];
              ir5[8] = (v672_data + (v620_data * (sycl::group_broadcast(item.get_sub_group(), v183_data, 6))));
              float v678_data = ir5[9];
              ir5[9] = (v678_data + (v620_data * (sycl::group_broadcast(item.get_sub_group(), v189_data, 6))));
              float v684_data = ir5[10];
              ir5[10] = (v684_data + (v620_data * (sycl::group_broadcast(item.get_sub_group(), v195_data, 6))));
              float v690_data = ir5[11];
              ir5[11] = (v690_data + (v620_data * (sycl::group_broadcast(item.get_sub_group(), v201_data, 6))));
              float v696_data = ir5[12];
              ir5[12] = (v696_data + (v620_data * (sycl::group_broadcast(item.get_sub_group(), v207_data, 6))));
              float v701_data = r3[7];
              float v705_data = ir5[0];
              ir5[0] = (v705_data + (v701_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 7))));
              float v711_data = ir5[1];
              ir5[1] = (v711_data + (v701_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 7))));
              float v717_data = ir5[2];
              ir5[2] = (v717_data + (v701_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 7))));
              float v723_data = ir5[3];
              ir5[3] = (v723_data + (v701_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 7))));
              float v729_data = ir5[4];
              ir5[4] = (v729_data + (v701_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 7))));
              float v735_data = ir5[5];
              ir5[5] = (v735_data + (v701_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 7))));
              float v741_data = ir5[6];
              ir5[6] = (v741_data + (v701_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 7))));
              float v747_data = ir5[7];
              ir5[7] = (v747_data + (v701_data * (sycl::group_broadcast(item.get_sub_group(), v177_data, 7))));
              float v753_data = ir5[8];
              ir5[8] = (v753_data + (v701_data * (sycl::group_broadcast(item.get_sub_group(), v183_data, 7))));
              float v759_data = ir5[9];
              ir5[9] = (v759_data + (v701_data * (sycl::group_broadcast(item.get_sub_group(), v189_data, 7))));
              float v765_data = ir5[10];
              ir5[10] = (v765_data + (v701_data * (sycl::group_broadcast(item.get_sub_group(), v195_data, 7))));
              float v771_data = ir5[11];
              ir5[11] = (v771_data + (v701_data * (sycl::group_broadcast(item.get_sub_group(), v201_data, 7))));
              float v777_data = ir5[12];
              ir5[12] = (v777_data + (v701_data * (sycl::group_broadcast(item.get_sub_group(), v207_data, 7))));
              float v782_data = r3[8];
              float v786_data = ir5[0];
              ir5[0] = (v786_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 8))));
              float v792_data = ir5[1];
              ir5[1] = (v792_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 8))));
              float v798_data = ir5[2];
              ir5[2] = (v798_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 8))));
              float v804_data = ir5[3];
              ir5[3] = (v804_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 8))));
              float v810_data = ir5[4];
              ir5[4] = (v810_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 8))));
              float v816_data = ir5[5];
              ir5[5] = (v816_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 8))));
              float v822_data = ir5[6];
              ir5[6] = (v822_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 8))));
              float v828_data = ir5[7];
              ir5[7] = (v828_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v177_data, 8))));
              float v834_data = ir5[8];
              ir5[8] = (v834_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v183_data, 8))));
              float v840_data = ir5[9];
              ir5[9] = (v840_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v189_data, 8))));
              float v846_data = ir5[10];
              ir5[10] = (v846_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v195_data, 8))));
              float v852_data = ir5[11];
              ir5[11] = (v852_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v201_data, 8))));
              float v858_data = ir5[12];
              ir5[12] = (v858_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v207_data, 8))));
              float v863_data = r3[9];
              float v867_data = ir5[0];
              ir5[0] = (v867_data + (v863_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 9))));
              float v873_data = ir5[1];
              ir5[1] = (v873_data + (v863_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 9))));
              float v879_data = ir5[2];
              ir5[2] = (v879_data + (v863_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 9))));
              float v885_data = ir5[3];
              ir5[3] = (v885_data + (v863_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 9))));
              float v891_data = ir5[4];
              ir5[4] = (v891_data + (v863_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 9))));
              float v897_data = ir5[5];
              ir5[5] = (v897_data + (v863_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 9))));
              float v903_data = ir5[6];
              ir5[6] = (v903_data + (v863_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 9))));
              float v909_data = ir5[7];
              ir5[7] = (v909_data + (v863_data * (sycl::group_broadcast(item.get_sub_group(), v177_data, 9))));
              float v915_data = ir5[8];
              ir5[8] = (v915_data + (v863_data * (sycl::group_broadcast(item.get_sub_group(), v183_data, 9))));
              float v921_data = ir5[9];
              ir5[9] = (v921_data + (v863_data * (sycl::group_broadcast(item.get_sub_group(), v189_data, 9))));
              float v927_data = ir5[10];
              ir5[10] = (v927_data + (v863_data * (sycl::group_broadcast(item.get_sub_group(), v195_data, 9))));
              float v933_data = ir5[11];
              ir5[11] = (v933_data + (v863_data * (sycl::group_broadcast(item.get_sub_group(), v201_data, 9))));
              float v939_data = ir5[12];
              ir5[12] = (v939_data + (v863_data * (sycl::group_broadcast(item.get_sub_group(), v207_data, 9))));
              float v944_data = r3[10];
              float v948_data = ir5[0];
              ir5[0] = (v948_data + (v944_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 10))));
              float v954_data = ir5[1];
              ir5[1] = (v954_data + (v944_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 10))));
              float v960_data = ir5[2];
              ir5[2] = (v960_data + (v944_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 10))));
              float v966_data = ir5[3];
              ir5[3] = (v966_data + (v944_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 10))));
              float v972_data = ir5[4];
              ir5[4] = (v972_data + (v944_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 10))));
              float v978_data = ir5[5];
              ir5[5] = (v978_data + (v944_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 10))));
              float v984_data = ir5[6];
              ir5[6] = (v984_data + (v944_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 10))));
              float v990_data = ir5[7];
              ir5[7] = (v990_data + (v944_data * (sycl::group_broadcast(item.get_sub_group(), v177_data, 10))));
              float v996_data = ir5[8];
              ir5[8] = (v996_data + (v944_data * (sycl::group_broadcast(item.get_sub_group(), v183_data, 10))));
              float v1002_data = ir5[9];
              ir5[9] = (v1002_data + (v944_data * (sycl::group_broadcast(item.get_sub_group(), v189_data, 10))));
              float v1008_data = ir5[10];
              ir5[10] = (v1008_data + (v944_data * (sycl::group_broadcast(item.get_sub_group(), v195_data, 10))));
              float v1014_data = ir5[11];
              ir5[11] = (v1014_data + (v944_data * (sycl::group_broadcast(item.get_sub_group(), v201_data, 10))));
              float v1020_data = ir5[12];
              ir5[12] = (v1020_data + (v944_data * (sycl::group_broadcast(item.get_sub_group(), v207_data, 10))));
              float v1025_data = r3[11];
              float v1029_data = ir5[0];
              ir5[0] = (v1029_data + (v1025_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 11))));
              float v1035_data = ir5[1];
              ir5[1] = (v1035_data + (v1025_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 11))));
              float v1041_data = ir5[2];
              ir5[2] = (v1041_data + (v1025_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 11))));
              float v1047_data = ir5[3];
              ir5[3] = (v1047_data + (v1025_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 11))));
              float v1053_data = ir5[4];
              ir5[4] = (v1053_data + (v1025_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 11))));
              float v1059_data = ir5[5];
              ir5[5] = (v1059_data + (v1025_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 11))));
              float v1065_data = ir5[6];
              ir5[6] = (v1065_data + (v1025_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 11))));
              float v1071_data = ir5[7];
              ir5[7] = (v1071_data + (v1025_data * (sycl::group_broadcast(item.get_sub_group(), v177_data, 11))));
              float v1077_data = ir5[8];
              ir5[8] = (v1077_data + (v1025_data * (sycl::group_broadcast(item.get_sub_group(), v183_data, 11))));
              float v1083_data = ir5[9];
              ir5[9] = (v1083_data + (v1025_data * (sycl::group_broadcast(item.get_sub_group(), v189_data, 11))));
              float v1089_data = ir5[10];
              ir5[10] = (v1089_data + (v1025_data * (sycl::group_broadcast(item.get_sub_group(), v195_data, 11))));
              float v1095_data = ir5[11];
              ir5[11] = (v1095_data + (v1025_data * (sycl::group_broadcast(item.get_sub_group(), v201_data, 11))));
              float v1101_data = ir5[12];
              ir5[12] = (v1101_data + (v1025_data * (sycl::group_broadcast(item.get_sub_group(), v207_data, 11))));
              float v1106_data = r3[12];
              float v1110_data = ir5[0];
              ir5[0] = (v1110_data + (v1106_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 12))));
              float v1116_data = ir5[1];
              ir5[1] = (v1116_data + (v1106_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 12))));
              float v1122_data = ir5[2];
              ir5[2] = (v1122_data + (v1106_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 12))));
              float v1128_data = ir5[3];
              ir5[3] = (v1128_data + (v1106_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 12))));
              float v1134_data = ir5[4];
              ir5[4] = (v1134_data + (v1106_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 12))));
              float v1140_data = ir5[5];
              ir5[5] = (v1140_data + (v1106_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 12))));
              float v1146_data = ir5[6];
              ir5[6] = (v1146_data + (v1106_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 12))));
              float v1152_data = ir5[7];
              ir5[7] = (v1152_data + (v1106_data * (sycl::group_broadcast(item.get_sub_group(), v177_data, 12))));
              float v1158_data = ir5[8];
              ir5[8] = (v1158_data + (v1106_data * (sycl::group_broadcast(item.get_sub_group(), v183_data, 12))));
              float v1164_data = ir5[9];
              ir5[9] = (v1164_data + (v1106_data * (sycl::group_broadcast(item.get_sub_group(), v189_data, 12))));
              float v1170_data = ir5[10];
              ir5[10] = (v1170_data + (v1106_data * (sycl::group_broadcast(item.get_sub_group(), v195_data, 12))));
              float v1176_data = ir5[11];
              ir5[11] = (v1176_data + (v1106_data * (sycl::group_broadcast(item.get_sub_group(), v201_data, 12))));
              float v1182_data = ir5[12];
              ir5[12] = (v1182_data + (v1106_data * (sycl::group_broadcast(item.get_sub_group(), v207_data, 12))));
              #pragma unroll
              for (int32_t v1187_n0 = 0; v1187_n0 < 1; ++v1187_n0) {
                #pragma unroll
                for (int32_t v1188_n1 = 0; v1188_n1 < 13; ++v1188_n1) {
                  int32_t v1189_a = v1187_n0 + v1188_n1;
                  float v1190_data = ir5[v1189_a];
                  r5[v1189_a] = v1190_data;
                }
              }
              // glb_m3 = store{r>g}(r5);
              #pragma unroll
              for (int32_t v1195_i0 = 0; v1195_i0 < 1; ++v1195_i0) {
                int32_t v1203_lead = v16_lead + (v1195_i0 * 32);
                #pragma unroll
                for (int32_t v1196_i1 = 0; v1196_i1 < 13; ++v1196_i1) {
                  float v1198_data = r5[(v1195_i0 + v1196_i1)];
                  glb_m3[(v1203_lead + (v1196_i1 * 32))] = v1198_data;
                }
              }
              item.barrier();
            }
          }
        }
      });
    }
  });
}

