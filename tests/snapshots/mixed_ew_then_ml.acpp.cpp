// === base name ===
kernel_a587425bdd

// === header ===
void launcher_kernel_a587425bdd(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_a587425bdd(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_a587425bdd(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_a587425bdd(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8×8(8×8) {0..8}×{0..8} strided
        // TMP = abs(A)
        // m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, 1] = t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, -1]×m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              int32_t v8_lead = item.get_local_id(0) % 16;
              bool v9_g = v8_lead < 8;
              if (v9_g) {
                #pragma unroll
                for (int32_t v10_i1 = 0; v10_i1 < 8; ++v10_i1) {
                  float v18_data = glb_m2[(v8_lead + (v10_i1 * 8))];
                  r1[v10_i1] = v18_data;
                }
              }
              float r0[8]{};
              // r0 = abs(glb_m0)
              if (v9_g) {
                #pragma unroll
                for (int32_t v25_k1 = 0; v25_k1 < 8; ++v25_k1) {
                  float v33_data = glb_m0[(v8_lead + (v25_k1 * 8))];
                  r0[v25_k1] = (sycl::fabs(v33_data));
                }
              }
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              float ir2[8]{};
              if (v9_g) {
                float v42_data = r0[0];
                float v43_data = r1[0];
                float v46_data = ir2[0];
                ir2[0] = (v46_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 0))));
                float v49_data = r1[1];
                float v52_data = ir2[1];
                ir2[1] = (v52_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 0))));
                float v55_data = r1[2];
                float v58_data = ir2[2];
                ir2[2] = (v58_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 0))));
                float v61_data = r1[3];
                float v64_data = ir2[3];
                ir2[3] = (v64_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 0))));
                float v67_data = r1[4];
                float v70_data = ir2[4];
                ir2[4] = (v70_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 0))));
                float v73_data = r1[5];
                float v76_data = ir2[5];
                ir2[5] = (v76_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 0))));
                float v79_data = r1[6];
                float v82_data = ir2[6];
                ir2[6] = (v82_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 0))));
                float v85_data = r1[7];
                float v88_data = ir2[7];
                ir2[7] = (v88_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 0))));
              }
              if (v9_g) {
                float v94_data = r0[1];
                float v95_data = r1[0];
                float v98_data = ir2[0];
                ir2[0] = (v98_data + (v94_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 1))));
                float v101_data = r1[1];
                float v104_data = ir2[1];
                ir2[1] = (v104_data + (v94_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 1))));
                float v107_data = r1[2];
                float v110_data = ir2[2];
                ir2[2] = (v110_data + (v94_data * (sycl::group_broadcast(item.get_sub_group(), v107_data, 1))));
                float v113_data = r1[3];
                float v116_data = ir2[3];
                ir2[3] = (v116_data + (v94_data * (sycl::group_broadcast(item.get_sub_group(), v113_data, 1))));
                float v119_data = r1[4];
                float v122_data = ir2[4];
                ir2[4] = (v122_data + (v94_data * (sycl::group_broadcast(item.get_sub_group(), v119_data, 1))));
                float v125_data = r1[5];
                float v128_data = ir2[5];
                ir2[5] = (v128_data + (v94_data * (sycl::group_broadcast(item.get_sub_group(), v125_data, 1))));
                float v131_data = r1[6];
                float v134_data = ir2[6];
                ir2[6] = (v134_data + (v94_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 1))));
                float v137_data = r1[7];
                float v140_data = ir2[7];
                ir2[7] = (v140_data + (v94_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 1))));
              }
              if (v9_g) {
                float v146_data = r0[2];
                float v147_data = r1[0];
                float v150_data = ir2[0];
                ir2[0] = (v150_data + (v146_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 2))));
                float v153_data = r1[1];
                float v156_data = ir2[1];
                ir2[1] = (v156_data + (v146_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 2))));
                float v159_data = r1[2];
                float v162_data = ir2[2];
                ir2[2] = (v162_data + (v146_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 2))));
                float v165_data = r1[3];
                float v168_data = ir2[3];
                ir2[3] = (v168_data + (v146_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 2))));
                float v171_data = r1[4];
                float v174_data = ir2[4];
                ir2[4] = (v174_data + (v146_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 2))));
                float v177_data = r1[5];
                float v180_data = ir2[5];
                ir2[5] = (v180_data + (v146_data * (sycl::group_broadcast(item.get_sub_group(), v177_data, 2))));
                float v183_data = r1[6];
                float v186_data = ir2[6];
                ir2[6] = (v186_data + (v146_data * (sycl::group_broadcast(item.get_sub_group(), v183_data, 2))));
                float v189_data = r1[7];
                float v192_data = ir2[7];
                ir2[7] = (v192_data + (v146_data * (sycl::group_broadcast(item.get_sub_group(), v189_data, 2))));
              }
              if (v9_g) {
                float v198_data = r0[3];
                float v199_data = r1[0];
                float v202_data = ir2[0];
                ir2[0] = (v202_data + (v198_data * (sycl::group_broadcast(item.get_sub_group(), v199_data, 3))));
                float v205_data = r1[1];
                float v208_data = ir2[1];
                ir2[1] = (v208_data + (v198_data * (sycl::group_broadcast(item.get_sub_group(), v205_data, 3))));
                float v211_data = r1[2];
                float v214_data = ir2[2];
                ir2[2] = (v214_data + (v198_data * (sycl::group_broadcast(item.get_sub_group(), v211_data, 3))));
                float v217_data = r1[3];
                float v220_data = ir2[3];
                ir2[3] = (v220_data + (v198_data * (sycl::group_broadcast(item.get_sub_group(), v217_data, 3))));
                float v223_data = r1[4];
                float v226_data = ir2[4];
                ir2[4] = (v226_data + (v198_data * (sycl::group_broadcast(item.get_sub_group(), v223_data, 3))));
                float v229_data = r1[5];
                float v232_data = ir2[5];
                ir2[5] = (v232_data + (v198_data * (sycl::group_broadcast(item.get_sub_group(), v229_data, 3))));
                float v235_data = r1[6];
                float v238_data = ir2[6];
                ir2[6] = (v238_data + (v198_data * (sycl::group_broadcast(item.get_sub_group(), v235_data, 3))));
                float v241_data = r1[7];
                float v244_data = ir2[7];
                ir2[7] = (v244_data + (v198_data * (sycl::group_broadcast(item.get_sub_group(), v241_data, 3))));
              }
              if (v9_g) {
                float v250_data = r0[4];
                float v251_data = r1[0];
                float v254_data = ir2[0];
                ir2[0] = (v254_data + (v250_data * (sycl::group_broadcast(item.get_sub_group(), v251_data, 4))));
                float v257_data = r1[1];
                float v260_data = ir2[1];
                ir2[1] = (v260_data + (v250_data * (sycl::group_broadcast(item.get_sub_group(), v257_data, 4))));
                float v263_data = r1[2];
                float v266_data = ir2[2];
                ir2[2] = (v266_data + (v250_data * (sycl::group_broadcast(item.get_sub_group(), v263_data, 4))));
                float v269_data = r1[3];
                float v272_data = ir2[3];
                ir2[3] = (v272_data + (v250_data * (sycl::group_broadcast(item.get_sub_group(), v269_data, 4))));
                float v275_data = r1[4];
                float v278_data = ir2[4];
                ir2[4] = (v278_data + (v250_data * (sycl::group_broadcast(item.get_sub_group(), v275_data, 4))));
                float v281_data = r1[5];
                float v284_data = ir2[5];
                ir2[5] = (v284_data + (v250_data * (sycl::group_broadcast(item.get_sub_group(), v281_data, 4))));
                float v287_data = r1[6];
                float v290_data = ir2[6];
                ir2[6] = (v290_data + (v250_data * (sycl::group_broadcast(item.get_sub_group(), v287_data, 4))));
                float v293_data = r1[7];
                float v296_data = ir2[7];
                ir2[7] = (v296_data + (v250_data * (sycl::group_broadcast(item.get_sub_group(), v293_data, 4))));
              }
              if (v9_g) {
                float v302_data = r0[5];
                float v303_data = r1[0];
                float v306_data = ir2[0];
                ir2[0] = (v306_data + (v302_data * (sycl::group_broadcast(item.get_sub_group(), v303_data, 5))));
                float v309_data = r1[1];
                float v312_data = ir2[1];
                ir2[1] = (v312_data + (v302_data * (sycl::group_broadcast(item.get_sub_group(), v309_data, 5))));
                float v315_data = r1[2];
                float v318_data = ir2[2];
                ir2[2] = (v318_data + (v302_data * (sycl::group_broadcast(item.get_sub_group(), v315_data, 5))));
                float v321_data = r1[3];
                float v324_data = ir2[3];
                ir2[3] = (v324_data + (v302_data * (sycl::group_broadcast(item.get_sub_group(), v321_data, 5))));
                float v327_data = r1[4];
                float v330_data = ir2[4];
                ir2[4] = (v330_data + (v302_data * (sycl::group_broadcast(item.get_sub_group(), v327_data, 5))));
                float v333_data = r1[5];
                float v336_data = ir2[5];
                ir2[5] = (v336_data + (v302_data * (sycl::group_broadcast(item.get_sub_group(), v333_data, 5))));
                float v339_data = r1[6];
                float v342_data = ir2[6];
                ir2[6] = (v342_data + (v302_data * (sycl::group_broadcast(item.get_sub_group(), v339_data, 5))));
                float v345_data = r1[7];
                float v348_data = ir2[7];
                ir2[7] = (v348_data + (v302_data * (sycl::group_broadcast(item.get_sub_group(), v345_data, 5))));
              }
              if (v9_g) {
                float v354_data = r0[6];
                float v355_data = r1[0];
                float v358_data = ir2[0];
                ir2[0] = (v358_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v355_data, 6))));
                float v361_data = r1[1];
                float v364_data = ir2[1];
                ir2[1] = (v364_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v361_data, 6))));
                float v367_data = r1[2];
                float v370_data = ir2[2];
                ir2[2] = (v370_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v367_data, 6))));
                float v373_data = r1[3];
                float v376_data = ir2[3];
                ir2[3] = (v376_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v373_data, 6))));
                float v379_data = r1[4];
                float v382_data = ir2[4];
                ir2[4] = (v382_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v379_data, 6))));
                float v385_data = r1[5];
                float v388_data = ir2[5];
                ir2[5] = (v388_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v385_data, 6))));
                float v391_data = r1[6];
                float v394_data = ir2[6];
                ir2[6] = (v394_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v391_data, 6))));
                float v397_data = r1[7];
                float v400_data = ir2[7];
                ir2[7] = (v400_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v397_data, 6))));
              }
              if (v9_g) {
                float v406_data = r0[7];
                float v407_data = r1[0];
                float v410_data = ir2[0];
                ir2[0] = (v410_data + (v406_data * (sycl::group_broadcast(item.get_sub_group(), v407_data, 7))));
                float v413_data = r1[1];
                float v416_data = ir2[1];
                ir2[1] = (v416_data + (v406_data * (sycl::group_broadcast(item.get_sub_group(), v413_data, 7))));
                float v419_data = r1[2];
                float v422_data = ir2[2];
                ir2[2] = (v422_data + (v406_data * (sycl::group_broadcast(item.get_sub_group(), v419_data, 7))));
                float v425_data = r1[3];
                float v428_data = ir2[3];
                ir2[3] = (v428_data + (v406_data * (sycl::group_broadcast(item.get_sub_group(), v425_data, 7))));
                float v431_data = r1[4];
                float v434_data = ir2[4];
                ir2[4] = (v434_data + (v406_data * (sycl::group_broadcast(item.get_sub_group(), v431_data, 7))));
                float v437_data = r1[5];
                float v440_data = ir2[5];
                ir2[5] = (v440_data + (v406_data * (sycl::group_broadcast(item.get_sub_group(), v437_data, 7))));
                float v443_data = r1[6];
                float v446_data = ir2[6];
                ir2[6] = (v446_data + (v406_data * (sycl::group_broadcast(item.get_sub_group(), v443_data, 7))));
                float v449_data = r1[7];
                float v452_data = ir2[7];
                ir2[7] = (v452_data + (v406_data * (sycl::group_broadcast(item.get_sub_group(), v449_data, 7))));
              }
              if (v9_g) {
                #pragma unroll
                for (int32_t v458_n1 = 0; v458_n1 < 8; ++v458_n1) {
                  float v460_data = ir2[v458_n1];
                  r2[v458_n1] = v460_data;
                }
              }
              // glb_m1 = store{r>g}(r2);
              if (v9_g) {
                #pragma unroll
                for (int32_t v466_i1 = 0; v466_i1 < 8; ++v466_i1) {
                  float v468_data = r2[v466_i1];
                  glb_m1[(v8_lead + (v466_i1 * 8))] = v468_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

