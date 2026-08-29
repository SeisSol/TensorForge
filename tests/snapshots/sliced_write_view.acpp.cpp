// === base name ===
kernel_939857c66e

// === header ===
void launcher_kernel_939857c66e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_939857c66e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid ((numElements0 + 8 - 1) / 8, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_939857c66e(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_939857c66e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 32×13(32×13) {0..32}×{0..13} strided
        // m1 32×13(32×13) {0..32}×{0..13} strided
        // m2 13×13(13×13) {0..13}×{0..13} strided
        // m3 32×13(32×13) {0..32}×{0..13} strided
        // m4 13×13(13×13) {0..13}×{0..13} strided
        // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..1})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{10..13})[0, -1]×m2 13×13(13×13) {0..13}×{0..13} strided({10..13}×{0..1})[-1, 1]
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
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 416 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 169 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[batchId0 * 416 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[batchId0 * 169 + 0 + m4_extraOffset];
              float r0[3]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v8_lead = item.get_local_id(0) % 32;
              #pragma unroll
              for (int32_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
                int32_t v15_lead = v8_lead + (v9_i0 * 32);
                #pragma unroll
                for (int32_t v10_i1 = 10; v10_i1 < 13; ++v10_i1) {
                  float v18_data = glb_m1[(v15_lead + (v10_i1 * 32))];
                  r0[(v9_i0 + (v10_i1 - 10))] = v18_data;
                }
              }
              float r1[13]{};
              // r1 = load{g>r}(glb_m2);
              float v22_lin = glb_m2[0 + item.get_local_id(0) * 1];
              r1[0] = v22_lin;
              float v23_lin = glb_m2[32 + item.get_local_id(0) * 1];
              r1[1] = v23_lin;
              float v24_lin = glb_m2[64 + item.get_local_id(0) * 1];
              r1[2] = v24_lin;
              float v25_lin = glb_m2[96 + item.get_local_id(0) * 1];
              r1[3] = v25_lin;
              float v26_lin = glb_m2[128 + item.get_local_id(0) * 1];
              r1[4] = v26_lin;
              float v27_lin = glb_m2[160 + item.get_local_id(0) * 1];
              r1[5] = v27_lin;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[1]{};
              // r2 = +(r0 * r1) + None
              // [(0, 32), (0, 1)] [(10, 13)]
              float ir2[1]{};
              float v33_data = r0[0];
              float v34_data = r1[8];
              float v37_data = ir2[0];
              ir2[0] = (v37_data + (v33_data * (sycl::group_broadcast(item.get_sub_group(), v34_data, 10))));
              float v42_data = r0[1];
              float v46_data = ir2[0];
              ir2[0] = (v46_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v34_data, 11))));
              float v51_data = r0[2];
              float v55_data = ir2[0];
              ir2[0] = (v55_data + (v51_data * (sycl::group_broadcast(item.get_sub_group(), v34_data, 12))));
              #pragma unroll
              for (int32_t v60_n0 = 0; v60_n0 < 1; ++v60_n0) {
                #pragma unroll
                for (int32_t v61_n1 = 0; v61_n1 < 1; ++v61_n1) {
                  int32_t v62_a = v60_n0 + v61_n1;
                  float v63_data = ir2[v62_a];
                  r2[v62_a] = v63_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v68_i0 = 0; v68_i0 < 1; ++v68_i0) {
                int32_t v76_lead = v8_lead + (v68_i0 * 32);
                #pragma unroll
                for (int32_t v69_i1 = 0; v69_i1 < 1; ++v69_i1) {
                  float v71_data = r2[(v68_i0 + v69_i1)];
                  glb_m0[(v76_lead + ((v69_i1 + 8) * 32))] = v71_data;
                }
              }
              float r3[13]{};
              // r3 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v84_i0 = 0; v84_i0 < 1; ++v84_i0) {
                int32_t v90_lead = v8_lead + (v84_i0 * 32);
                #pragma unroll
                for (int32_t v85_i1 = 0; v85_i1 < 13; ++v85_i1) {
                  float v93_data = glb_m0[(v90_lead + (v85_i1 * 32))];
                  r3[(v84_i0 + v85_i1)] = v93_data;
                }
              }
              float r4[13]{};
              // r4 = load{g>r}(glb_m4);
              float v96_lin = glb_m4[0 + item.get_local_id(0) * 1];
              r4[0] = v96_lin;
              float v97_lin = glb_m4[32 + item.get_local_id(0) * 1];
              r4[1] = v97_lin;
              float v98_lin = glb_m4[64 + item.get_local_id(0) * 1];
              r4[2] = v98_lin;
              float v99_lin = glb_m4[96 + item.get_local_id(0) * 1];
              r4[3] = v99_lin;
              float v100_lin = glb_m4[128 + item.get_local_id(0) * 1];
              r4[4] = v100_lin;
              float v101_lin = glb_m4[160 + item.get_local_id(0) * 1];
              r4[5] = v101_lin;
              // wait(r3 = load{g>r}(glb_m0););
              // wait(r4 = load{g>r}(glb_m4););
              float r5[13]{};
              // r5 = +(r3 * r4) + None
              // [(0, 32), (0, 13)] [(0, 13)]
              float ir5[13]{};
              float v107_data = r3[0];
              float v108_data = r4[0];
              float v111_data = ir5[0];
              ir5[0] = (v111_data + (v107_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 0))));
              float v114_data = r4[1];
              float v117_data = ir5[1];
              ir5[1] = (v117_data + (v107_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 0))));
              float v120_data = r4[2];
              float v123_data = ir5[2];
              ir5[2] = (v123_data + (v107_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 0))));
              float v126_data = r4[3];
              float v129_data = ir5[3];
              ir5[3] = (v129_data + (v107_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 0))));
              float v132_data = r4[4];
              float v135_data = ir5[4];
              ir5[4] = (v135_data + (v107_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 0))));
              float v138_data = r4[5];
              float v141_data = ir5[5];
              ir5[5] = (v141_data + (v107_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 0))));
              float v144_data = r4[6];
              float v147_data = ir5[6];
              ir5[6] = (v147_data + (v107_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 0))));
              float v150_data = r4[7];
              float v153_data = ir5[7];
              ir5[7] = (v153_data + (v107_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 0))));
              float v156_data = r4[8];
              float v159_data = ir5[8];
              ir5[8] = (v159_data + (v107_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 0))));
              float v162_data = r4[9];
              float v165_data = ir5[9];
              ir5[9] = (v165_data + (v107_data * (sycl::group_broadcast(item.get_sub_group(), v162_data, 0))));
              float v168_data = r4[10];
              float v171_data = ir5[10];
              ir5[10] = (v171_data + (v107_data * (sycl::group_broadcast(item.get_sub_group(), v168_data, 0))));
              float v174_data = r4[11];
              float v177_data = ir5[11];
              ir5[11] = (v177_data + (v107_data * (sycl::group_broadcast(item.get_sub_group(), v174_data, 0))));
              float v180_data = r4[12];
              float v183_data = ir5[12];
              ir5[12] = (v183_data + (v107_data * (sycl::group_broadcast(item.get_sub_group(), v180_data, 0))));
              float v188_data = r3[1];
              float v192_data = ir5[0];
              ir5[0] = (v192_data + (v188_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 1))));
              float v198_data = ir5[1];
              ir5[1] = (v198_data + (v188_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 1))));
              float v204_data = ir5[2];
              ir5[2] = (v204_data + (v188_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 1))));
              float v210_data = ir5[3];
              ir5[3] = (v210_data + (v188_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 1))));
              float v216_data = ir5[4];
              ir5[4] = (v216_data + (v188_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 1))));
              float v222_data = ir5[5];
              ir5[5] = (v222_data + (v188_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 1))));
              float v228_data = ir5[6];
              ir5[6] = (v228_data + (v188_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 1))));
              float v234_data = ir5[7];
              ir5[7] = (v234_data + (v188_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 1))));
              float v240_data = ir5[8];
              ir5[8] = (v240_data + (v188_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 1))));
              float v246_data = ir5[9];
              ir5[9] = (v246_data + (v188_data * (sycl::group_broadcast(item.get_sub_group(), v162_data, 1))));
              float v252_data = ir5[10];
              ir5[10] = (v252_data + (v188_data * (sycl::group_broadcast(item.get_sub_group(), v168_data, 1))));
              float v258_data = ir5[11];
              ir5[11] = (v258_data + (v188_data * (sycl::group_broadcast(item.get_sub_group(), v174_data, 1))));
              float v264_data = ir5[12];
              ir5[12] = (v264_data + (v188_data * (sycl::group_broadcast(item.get_sub_group(), v180_data, 1))));
              float v269_data = r3[2];
              float v273_data = ir5[0];
              ir5[0] = (v273_data + (v269_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 2))));
              float v279_data = ir5[1];
              ir5[1] = (v279_data + (v269_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 2))));
              float v285_data = ir5[2];
              ir5[2] = (v285_data + (v269_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 2))));
              float v291_data = ir5[3];
              ir5[3] = (v291_data + (v269_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 2))));
              float v297_data = ir5[4];
              ir5[4] = (v297_data + (v269_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 2))));
              float v303_data = ir5[5];
              ir5[5] = (v303_data + (v269_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 2))));
              float v309_data = ir5[6];
              ir5[6] = (v309_data + (v269_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 2))));
              float v315_data = ir5[7];
              ir5[7] = (v315_data + (v269_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 2))));
              float v321_data = ir5[8];
              ir5[8] = (v321_data + (v269_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 2))));
              float v327_data = ir5[9];
              ir5[9] = (v327_data + (v269_data * (sycl::group_broadcast(item.get_sub_group(), v162_data, 2))));
              float v333_data = ir5[10];
              ir5[10] = (v333_data + (v269_data * (sycl::group_broadcast(item.get_sub_group(), v168_data, 2))));
              float v339_data = ir5[11];
              ir5[11] = (v339_data + (v269_data * (sycl::group_broadcast(item.get_sub_group(), v174_data, 2))));
              float v345_data = ir5[12];
              ir5[12] = (v345_data + (v269_data * (sycl::group_broadcast(item.get_sub_group(), v180_data, 2))));
              float v350_data = r3[3];
              float v354_data = ir5[0];
              ir5[0] = (v354_data + (v350_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 3))));
              float v360_data = ir5[1];
              ir5[1] = (v360_data + (v350_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 3))));
              float v366_data = ir5[2];
              ir5[2] = (v366_data + (v350_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 3))));
              float v372_data = ir5[3];
              ir5[3] = (v372_data + (v350_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 3))));
              float v378_data = ir5[4];
              ir5[4] = (v378_data + (v350_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 3))));
              float v384_data = ir5[5];
              ir5[5] = (v384_data + (v350_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 3))));
              float v390_data = ir5[6];
              ir5[6] = (v390_data + (v350_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 3))));
              float v396_data = ir5[7];
              ir5[7] = (v396_data + (v350_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 3))));
              float v402_data = ir5[8];
              ir5[8] = (v402_data + (v350_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 3))));
              float v408_data = ir5[9];
              ir5[9] = (v408_data + (v350_data * (sycl::group_broadcast(item.get_sub_group(), v162_data, 3))));
              float v414_data = ir5[10];
              ir5[10] = (v414_data + (v350_data * (sycl::group_broadcast(item.get_sub_group(), v168_data, 3))));
              float v420_data = ir5[11];
              ir5[11] = (v420_data + (v350_data * (sycl::group_broadcast(item.get_sub_group(), v174_data, 3))));
              float v426_data = ir5[12];
              ir5[12] = (v426_data + (v350_data * (sycl::group_broadcast(item.get_sub_group(), v180_data, 3))));
              float v431_data = r3[4];
              float v435_data = ir5[0];
              ir5[0] = (v435_data + (v431_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 4))));
              float v441_data = ir5[1];
              ir5[1] = (v441_data + (v431_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 4))));
              float v447_data = ir5[2];
              ir5[2] = (v447_data + (v431_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 4))));
              float v453_data = ir5[3];
              ir5[3] = (v453_data + (v431_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 4))));
              float v459_data = ir5[4];
              ir5[4] = (v459_data + (v431_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 4))));
              float v465_data = ir5[5];
              ir5[5] = (v465_data + (v431_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 4))));
              float v471_data = ir5[6];
              ir5[6] = (v471_data + (v431_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 4))));
              float v477_data = ir5[7];
              ir5[7] = (v477_data + (v431_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 4))));
              float v483_data = ir5[8];
              ir5[8] = (v483_data + (v431_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 4))));
              float v489_data = ir5[9];
              ir5[9] = (v489_data + (v431_data * (sycl::group_broadcast(item.get_sub_group(), v162_data, 4))));
              float v495_data = ir5[10];
              ir5[10] = (v495_data + (v431_data * (sycl::group_broadcast(item.get_sub_group(), v168_data, 4))));
              float v501_data = ir5[11];
              ir5[11] = (v501_data + (v431_data * (sycl::group_broadcast(item.get_sub_group(), v174_data, 4))));
              float v507_data = ir5[12];
              ir5[12] = (v507_data + (v431_data * (sycl::group_broadcast(item.get_sub_group(), v180_data, 4))));
              float v512_data = r3[5];
              float v516_data = ir5[0];
              ir5[0] = (v516_data + (v512_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 5))));
              float v522_data = ir5[1];
              ir5[1] = (v522_data + (v512_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 5))));
              float v528_data = ir5[2];
              ir5[2] = (v528_data + (v512_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 5))));
              float v534_data = ir5[3];
              ir5[3] = (v534_data + (v512_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 5))));
              float v540_data = ir5[4];
              ir5[4] = (v540_data + (v512_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 5))));
              float v546_data = ir5[5];
              ir5[5] = (v546_data + (v512_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 5))));
              float v552_data = ir5[6];
              ir5[6] = (v552_data + (v512_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 5))));
              float v558_data = ir5[7];
              ir5[7] = (v558_data + (v512_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 5))));
              float v564_data = ir5[8];
              ir5[8] = (v564_data + (v512_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 5))));
              float v570_data = ir5[9];
              ir5[9] = (v570_data + (v512_data * (sycl::group_broadcast(item.get_sub_group(), v162_data, 5))));
              float v576_data = ir5[10];
              ir5[10] = (v576_data + (v512_data * (sycl::group_broadcast(item.get_sub_group(), v168_data, 5))));
              float v582_data = ir5[11];
              ir5[11] = (v582_data + (v512_data * (sycl::group_broadcast(item.get_sub_group(), v174_data, 5))));
              float v588_data = ir5[12];
              ir5[12] = (v588_data + (v512_data * (sycl::group_broadcast(item.get_sub_group(), v180_data, 5))));
              float v593_data = r3[6];
              float v597_data = ir5[0];
              ir5[0] = (v597_data + (v593_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 6))));
              float v603_data = ir5[1];
              ir5[1] = (v603_data + (v593_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 6))));
              float v609_data = ir5[2];
              ir5[2] = (v609_data + (v593_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 6))));
              float v615_data = ir5[3];
              ir5[3] = (v615_data + (v593_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 6))));
              float v621_data = ir5[4];
              ir5[4] = (v621_data + (v593_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 6))));
              float v627_data = ir5[5];
              ir5[5] = (v627_data + (v593_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 6))));
              float v633_data = ir5[6];
              ir5[6] = (v633_data + (v593_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 6))));
              float v639_data = ir5[7];
              ir5[7] = (v639_data + (v593_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 6))));
              float v645_data = ir5[8];
              ir5[8] = (v645_data + (v593_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 6))));
              float v651_data = ir5[9];
              ir5[9] = (v651_data + (v593_data * (sycl::group_broadcast(item.get_sub_group(), v162_data, 6))));
              float v657_data = ir5[10];
              ir5[10] = (v657_data + (v593_data * (sycl::group_broadcast(item.get_sub_group(), v168_data, 6))));
              float v663_data = ir5[11];
              ir5[11] = (v663_data + (v593_data * (sycl::group_broadcast(item.get_sub_group(), v174_data, 6))));
              float v669_data = ir5[12];
              ir5[12] = (v669_data + (v593_data * (sycl::group_broadcast(item.get_sub_group(), v180_data, 6))));
              float v674_data = r3[7];
              float v678_data = ir5[0];
              ir5[0] = (v678_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 7))));
              float v684_data = ir5[1];
              ir5[1] = (v684_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 7))));
              float v690_data = ir5[2];
              ir5[2] = (v690_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 7))));
              float v696_data = ir5[3];
              ir5[3] = (v696_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 7))));
              float v702_data = ir5[4];
              ir5[4] = (v702_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 7))));
              float v708_data = ir5[5];
              ir5[5] = (v708_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 7))));
              float v714_data = ir5[6];
              ir5[6] = (v714_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 7))));
              float v720_data = ir5[7];
              ir5[7] = (v720_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 7))));
              float v726_data = ir5[8];
              ir5[8] = (v726_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 7))));
              float v732_data = ir5[9];
              ir5[9] = (v732_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v162_data, 7))));
              float v738_data = ir5[10];
              ir5[10] = (v738_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v168_data, 7))));
              float v744_data = ir5[11];
              ir5[11] = (v744_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v174_data, 7))));
              float v750_data = ir5[12];
              ir5[12] = (v750_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v180_data, 7))));
              float v755_data = r3[8];
              float v759_data = ir5[0];
              ir5[0] = (v759_data + (v755_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 8))));
              float v765_data = ir5[1];
              ir5[1] = (v765_data + (v755_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 8))));
              float v771_data = ir5[2];
              ir5[2] = (v771_data + (v755_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 8))));
              float v777_data = ir5[3];
              ir5[3] = (v777_data + (v755_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 8))));
              float v783_data = ir5[4];
              ir5[4] = (v783_data + (v755_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 8))));
              float v789_data = ir5[5];
              ir5[5] = (v789_data + (v755_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 8))));
              float v795_data = ir5[6];
              ir5[6] = (v795_data + (v755_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 8))));
              float v801_data = ir5[7];
              ir5[7] = (v801_data + (v755_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 8))));
              float v807_data = ir5[8];
              ir5[8] = (v807_data + (v755_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 8))));
              float v813_data = ir5[9];
              ir5[9] = (v813_data + (v755_data * (sycl::group_broadcast(item.get_sub_group(), v162_data, 8))));
              float v819_data = ir5[10];
              ir5[10] = (v819_data + (v755_data * (sycl::group_broadcast(item.get_sub_group(), v168_data, 8))));
              float v825_data = ir5[11];
              ir5[11] = (v825_data + (v755_data * (sycl::group_broadcast(item.get_sub_group(), v174_data, 8))));
              float v831_data = ir5[12];
              ir5[12] = (v831_data + (v755_data * (sycl::group_broadcast(item.get_sub_group(), v180_data, 8))));
              float v836_data = r3[9];
              float v840_data = ir5[0];
              ir5[0] = (v840_data + (v836_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 9))));
              float v846_data = ir5[1];
              ir5[1] = (v846_data + (v836_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 9))));
              float v852_data = ir5[2];
              ir5[2] = (v852_data + (v836_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 9))));
              float v858_data = ir5[3];
              ir5[3] = (v858_data + (v836_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 9))));
              float v864_data = ir5[4];
              ir5[4] = (v864_data + (v836_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 9))));
              float v870_data = ir5[5];
              ir5[5] = (v870_data + (v836_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 9))));
              float v876_data = ir5[6];
              ir5[6] = (v876_data + (v836_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 9))));
              float v882_data = ir5[7];
              ir5[7] = (v882_data + (v836_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 9))));
              float v888_data = ir5[8];
              ir5[8] = (v888_data + (v836_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 9))));
              float v894_data = ir5[9];
              ir5[9] = (v894_data + (v836_data * (sycl::group_broadcast(item.get_sub_group(), v162_data, 9))));
              float v900_data = ir5[10];
              ir5[10] = (v900_data + (v836_data * (sycl::group_broadcast(item.get_sub_group(), v168_data, 9))));
              float v906_data = ir5[11];
              ir5[11] = (v906_data + (v836_data * (sycl::group_broadcast(item.get_sub_group(), v174_data, 9))));
              float v912_data = ir5[12];
              ir5[12] = (v912_data + (v836_data * (sycl::group_broadcast(item.get_sub_group(), v180_data, 9))));
              float v917_data = r3[10];
              float v921_data = ir5[0];
              ir5[0] = (v921_data + (v917_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 10))));
              float v927_data = ir5[1];
              ir5[1] = (v927_data + (v917_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 10))));
              float v933_data = ir5[2];
              ir5[2] = (v933_data + (v917_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 10))));
              float v939_data = ir5[3];
              ir5[3] = (v939_data + (v917_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 10))));
              float v945_data = ir5[4];
              ir5[4] = (v945_data + (v917_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 10))));
              float v951_data = ir5[5];
              ir5[5] = (v951_data + (v917_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 10))));
              float v957_data = ir5[6];
              ir5[6] = (v957_data + (v917_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 10))));
              float v963_data = ir5[7];
              ir5[7] = (v963_data + (v917_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 10))));
              float v969_data = ir5[8];
              ir5[8] = (v969_data + (v917_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 10))));
              float v975_data = ir5[9];
              ir5[9] = (v975_data + (v917_data * (sycl::group_broadcast(item.get_sub_group(), v162_data, 10))));
              float v981_data = ir5[10];
              ir5[10] = (v981_data + (v917_data * (sycl::group_broadcast(item.get_sub_group(), v168_data, 10))));
              float v987_data = ir5[11];
              ir5[11] = (v987_data + (v917_data * (sycl::group_broadcast(item.get_sub_group(), v174_data, 10))));
              float v993_data = ir5[12];
              ir5[12] = (v993_data + (v917_data * (sycl::group_broadcast(item.get_sub_group(), v180_data, 10))));
              float v998_data = r3[11];
              float v1002_data = ir5[0];
              ir5[0] = (v1002_data + (v998_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 11))));
              float v1008_data = ir5[1];
              ir5[1] = (v1008_data + (v998_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 11))));
              float v1014_data = ir5[2];
              ir5[2] = (v1014_data + (v998_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 11))));
              float v1020_data = ir5[3];
              ir5[3] = (v1020_data + (v998_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 11))));
              float v1026_data = ir5[4];
              ir5[4] = (v1026_data + (v998_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 11))));
              float v1032_data = ir5[5];
              ir5[5] = (v1032_data + (v998_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 11))));
              float v1038_data = ir5[6];
              ir5[6] = (v1038_data + (v998_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 11))));
              float v1044_data = ir5[7];
              ir5[7] = (v1044_data + (v998_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 11))));
              float v1050_data = ir5[8];
              ir5[8] = (v1050_data + (v998_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 11))));
              float v1056_data = ir5[9];
              ir5[9] = (v1056_data + (v998_data * (sycl::group_broadcast(item.get_sub_group(), v162_data, 11))));
              float v1062_data = ir5[10];
              ir5[10] = (v1062_data + (v998_data * (sycl::group_broadcast(item.get_sub_group(), v168_data, 11))));
              float v1068_data = ir5[11];
              ir5[11] = (v1068_data + (v998_data * (sycl::group_broadcast(item.get_sub_group(), v174_data, 11))));
              float v1074_data = ir5[12];
              ir5[12] = (v1074_data + (v998_data * (sycl::group_broadcast(item.get_sub_group(), v180_data, 11))));
              float v1079_data = r3[12];
              float v1083_data = ir5[0];
              ir5[0] = (v1083_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 12))));
              float v1089_data = ir5[1];
              ir5[1] = (v1089_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 12))));
              float v1095_data = ir5[2];
              ir5[2] = (v1095_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 12))));
              float v1101_data = ir5[3];
              ir5[3] = (v1101_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 12))));
              float v1107_data = ir5[4];
              ir5[4] = (v1107_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 12))));
              float v1113_data = ir5[5];
              ir5[5] = (v1113_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 12))));
              float v1119_data = ir5[6];
              ir5[6] = (v1119_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 12))));
              float v1125_data = ir5[7];
              ir5[7] = (v1125_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 12))));
              float v1131_data = ir5[8];
              ir5[8] = (v1131_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 12))));
              float v1137_data = ir5[9];
              ir5[9] = (v1137_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v162_data, 12))));
              float v1143_data = ir5[10];
              ir5[10] = (v1143_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v168_data, 12))));
              float v1149_data = ir5[11];
              ir5[11] = (v1149_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v174_data, 12))));
              float v1155_data = ir5[12];
              ir5[12] = (v1155_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v180_data, 12))));
              #pragma unroll
              for (int32_t v1160_n0 = 0; v1160_n0 < 1; ++v1160_n0) {
                #pragma unroll
                for (int32_t v1161_n1 = 0; v1161_n1 < 13; ++v1161_n1) {
                  int32_t v1162_a = v1160_n0 + v1161_n1;
                  float v1163_data = ir5[v1162_a];
                  r5[v1162_a] = v1163_data;
                }
              }
              // glb_m3 = store{r>g}(r5);
              #pragma unroll
              for (int32_t v1168_i0 = 0; v1168_i0 < 1; ++v1168_i0) {
                int32_t v1176_lead = v8_lead + (v1168_i0 * 32);
                #pragma unroll
                for (int32_t v1169_i1 = 0; v1169_i1 < 13; ++v1169_i1) {
                  float v1171_data = r5[(v1168_i0 + v1169_i1)];
                  glb_m3[(v1176_lead + (v1169_i1 * 32))] = v1171_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

