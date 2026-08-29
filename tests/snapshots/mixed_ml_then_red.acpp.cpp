// === base name ===
kernel_4b748443ff

// === header ===
void launcher_kernel_4b748443ff(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_4b748443ff(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_4b748443ff(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_4b748443ff(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8(8) {0..8} strided
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        // OUT = +(TMP, dims=[1])
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[80 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[64];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0 * 8 + 0 + m2_extraOffset];
              float r0[8]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v8_lead = item.get_local_id(0) % 16;
              if (v8_lead < 8) {
                #pragma unroll
                for (int32_t v10_i1 = 0; v10_i1 < 8; ++v10_i1) {
                  float v18_data = glb_m0[(v8_lead + (v10_i1 * 8))];
                  r0[v10_i1] = v18_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m1);
              float v21_lin = glb_m1[0 + item.get_local_id(0) * 1];
              r1[0] = v21_lin;
              float v22_lin = glb_m1[16 + item.get_local_id(0) * 1];
              r1[1] = v22_lin;
              float v23_lin = glb_m1[32 + item.get_local_id(0) * 1];
              r1[2] = v23_lin;
              float v24_lin = glb_m1[48 + item.get_local_id(0) * 1];
              r1[3] = v24_lin;
              // wait(r0 = load{g>r}(glb_m0););
              // wait(r1 = load{g>r}(glb_m1););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              if (v8_lead < 8) {
                float v30_data = r0[0];
                float v31_data = r1[0];
                float v34_data = r2[0];
                r2[0] = (v34_data + (v30_data * (sycl::group_broadcast(item.get_sub_group(), v31_data, 0))));
                float v37_data = r1[1];
                float v40_data = r2[1];
                r2[1] = (v40_data + (v30_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 0))));
                float v43_data = r1[2];
                float v46_data = r2[2];
                r2[2] = (v46_data + (v30_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 0))));
                float v49_data = r1[3];
                float v52_data = r2[3];
                r2[3] = (v52_data + (v30_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 0))));
                float v55_data = r1[4];
                float v58_data = r2[4];
                r2[4] = (v58_data + (v30_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 0))));
                float v61_data = r1[5];
                float v64_data = r2[5];
                r2[5] = (v64_data + (v30_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 0))));
                float v67_data = r1[6];
                float v70_data = r2[6];
                r2[6] = (v70_data + (v30_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 0))));
                float v73_data = r1[7];
                float v76_data = r2[7];
                r2[7] = (v76_data + (v30_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 0))));
              }
              if (v8_lead < 8) {
                float v82_data = r0[1];
                float v83_data = r1[0];
                float v86_data = r2[0];
                r2[0] = (v86_data + (v82_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 1))));
                float v89_data = r1[1];
                float v92_data = r2[1];
                r2[1] = (v92_data + (v82_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 1))));
                float v95_data = r1[2];
                float v98_data = r2[2];
                r2[2] = (v98_data + (v82_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 1))));
                float v101_data = r1[3];
                float v104_data = r2[3];
                r2[3] = (v104_data + (v82_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 1))));
                float v107_data = r1[4];
                float v110_data = r2[4];
                r2[4] = (v110_data + (v82_data * (sycl::group_broadcast(item.get_sub_group(), v107_data, 1))));
                float v113_data = r1[5];
                float v116_data = r2[5];
                r2[5] = (v116_data + (v82_data * (sycl::group_broadcast(item.get_sub_group(), v113_data, 1))));
                float v119_data = r1[6];
                float v122_data = r2[6];
                r2[6] = (v122_data + (v82_data * (sycl::group_broadcast(item.get_sub_group(), v119_data, 1))));
                float v125_data = r1[7];
                float v128_data = r2[7];
                r2[7] = (v128_data + (v82_data * (sycl::group_broadcast(item.get_sub_group(), v125_data, 1))));
              }
              if (v8_lead < 8) {
                float v134_data = r0[2];
                float v135_data = r1[0];
                float v138_data = r2[0];
                r2[0] = (v138_data + (v134_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 2))));
                float v141_data = r1[1];
                float v144_data = r2[1];
                r2[1] = (v144_data + (v134_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 2))));
                float v147_data = r1[2];
                float v150_data = r2[2];
                r2[2] = (v150_data + (v134_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 2))));
                float v153_data = r1[3];
                float v156_data = r2[3];
                r2[3] = (v156_data + (v134_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 2))));
                float v159_data = r1[4];
                float v162_data = r2[4];
                r2[4] = (v162_data + (v134_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 2))));
                float v165_data = r1[5];
                float v168_data = r2[5];
                r2[5] = (v168_data + (v134_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 2))));
                float v171_data = r1[6];
                float v174_data = r2[6];
                r2[6] = (v174_data + (v134_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 2))));
                float v177_data = r1[7];
                float v180_data = r2[7];
                r2[7] = (v180_data + (v134_data * (sycl::group_broadcast(item.get_sub_group(), v177_data, 2))));
              }
              if (v8_lead < 8) {
                float v186_data = r0[3];
                float v187_data = r1[0];
                float v190_data = r2[0];
                r2[0] = (v190_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 3))));
                float v193_data = r1[1];
                float v196_data = r2[1];
                r2[1] = (v196_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 3))));
                float v199_data = r1[2];
                float v202_data = r2[2];
                r2[2] = (v202_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v199_data, 3))));
                float v205_data = r1[3];
                float v208_data = r2[3];
                r2[3] = (v208_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v205_data, 3))));
                float v211_data = r1[4];
                float v214_data = r2[4];
                r2[4] = (v214_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v211_data, 3))));
                float v217_data = r1[5];
                float v220_data = r2[5];
                r2[5] = (v220_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v217_data, 3))));
                float v223_data = r1[6];
                float v226_data = r2[6];
                r2[6] = (v226_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v223_data, 3))));
                float v229_data = r1[7];
                float v232_data = r2[7];
                r2[7] = (v232_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v229_data, 3))));
              }
              if (v8_lead < 8) {
                float v238_data = r0[4];
                float v239_data = r1[0];
                float v242_data = r2[0];
                r2[0] = (v242_data + (v238_data * (sycl::group_broadcast(item.get_sub_group(), v239_data, 4))));
                float v245_data = r1[1];
                float v248_data = r2[1];
                r2[1] = (v248_data + (v238_data * (sycl::group_broadcast(item.get_sub_group(), v245_data, 4))));
                float v251_data = r1[2];
                float v254_data = r2[2];
                r2[2] = (v254_data + (v238_data * (sycl::group_broadcast(item.get_sub_group(), v251_data, 4))));
                float v257_data = r1[3];
                float v260_data = r2[3];
                r2[3] = (v260_data + (v238_data * (sycl::group_broadcast(item.get_sub_group(), v257_data, 4))));
                float v263_data = r1[4];
                float v266_data = r2[4];
                r2[4] = (v266_data + (v238_data * (sycl::group_broadcast(item.get_sub_group(), v263_data, 4))));
                float v269_data = r1[5];
                float v272_data = r2[5];
                r2[5] = (v272_data + (v238_data * (sycl::group_broadcast(item.get_sub_group(), v269_data, 4))));
                float v275_data = r1[6];
                float v278_data = r2[6];
                r2[6] = (v278_data + (v238_data * (sycl::group_broadcast(item.get_sub_group(), v275_data, 4))));
                float v281_data = r1[7];
                float v284_data = r2[7];
                r2[7] = (v284_data + (v238_data * (sycl::group_broadcast(item.get_sub_group(), v281_data, 4))));
              }
              if (v8_lead < 8) {
                float v290_data = r0[5];
                float v291_data = r1[0];
                float v294_data = r2[0];
                r2[0] = (v294_data + (v290_data * (sycl::group_broadcast(item.get_sub_group(), v291_data, 5))));
                float v297_data = r1[1];
                float v300_data = r2[1];
                r2[1] = (v300_data + (v290_data * (sycl::group_broadcast(item.get_sub_group(), v297_data, 5))));
                float v303_data = r1[2];
                float v306_data = r2[2];
                r2[2] = (v306_data + (v290_data * (sycl::group_broadcast(item.get_sub_group(), v303_data, 5))));
                float v309_data = r1[3];
                float v312_data = r2[3];
                r2[3] = (v312_data + (v290_data * (sycl::group_broadcast(item.get_sub_group(), v309_data, 5))));
                float v315_data = r1[4];
                float v318_data = r2[4];
                r2[4] = (v318_data + (v290_data * (sycl::group_broadcast(item.get_sub_group(), v315_data, 5))));
                float v321_data = r1[5];
                float v324_data = r2[5];
                r2[5] = (v324_data + (v290_data * (sycl::group_broadcast(item.get_sub_group(), v321_data, 5))));
                float v327_data = r1[6];
                float v330_data = r2[6];
                r2[6] = (v330_data + (v290_data * (sycl::group_broadcast(item.get_sub_group(), v327_data, 5))));
                float v333_data = r1[7];
                float v336_data = r2[7];
                r2[7] = (v336_data + (v290_data * (sycl::group_broadcast(item.get_sub_group(), v333_data, 5))));
              }
              if (v8_lead < 8) {
                float v342_data = r0[6];
                float v343_data = r1[0];
                float v346_data = r2[0];
                r2[0] = (v346_data + (v342_data * (sycl::group_broadcast(item.get_sub_group(), v343_data, 6))));
                float v349_data = r1[1];
                float v352_data = r2[1];
                r2[1] = (v352_data + (v342_data * (sycl::group_broadcast(item.get_sub_group(), v349_data, 6))));
                float v355_data = r1[2];
                float v358_data = r2[2];
                r2[2] = (v358_data + (v342_data * (sycl::group_broadcast(item.get_sub_group(), v355_data, 6))));
                float v361_data = r1[3];
                float v364_data = r2[3];
                r2[3] = (v364_data + (v342_data * (sycl::group_broadcast(item.get_sub_group(), v361_data, 6))));
                float v367_data = r1[4];
                float v370_data = r2[4];
                r2[4] = (v370_data + (v342_data * (sycl::group_broadcast(item.get_sub_group(), v367_data, 6))));
                float v373_data = r1[5];
                float v376_data = r2[5];
                r2[5] = (v376_data + (v342_data * (sycl::group_broadcast(item.get_sub_group(), v373_data, 6))));
                float v379_data = r1[6];
                float v382_data = r2[6];
                r2[6] = (v382_data + (v342_data * (sycl::group_broadcast(item.get_sub_group(), v379_data, 6))));
                float v385_data = r1[7];
                float v388_data = r2[7];
                r2[7] = (v388_data + (v342_data * (sycl::group_broadcast(item.get_sub_group(), v385_data, 6))));
              }
              if (v8_lead < 8) {
                float v394_data = r0[7];
                float v395_data = r1[0];
                float v398_data = r2[0];
                r2[0] = (v398_data + (v394_data * (sycl::group_broadcast(item.get_sub_group(), v395_data, 7))));
                float v401_data = r1[1];
                float v404_data = r2[1];
                r2[1] = (v404_data + (v394_data * (sycl::group_broadcast(item.get_sub_group(), v401_data, 7))));
                float v407_data = r1[2];
                float v410_data = r2[2];
                r2[2] = (v410_data + (v394_data * (sycl::group_broadcast(item.get_sub_group(), v407_data, 7))));
                float v413_data = r1[3];
                float v416_data = r2[3];
                r2[3] = (v416_data + (v394_data * (sycl::group_broadcast(item.get_sub_group(), v413_data, 7))));
                float v419_data = r1[4];
                float v422_data = r2[4];
                r2[4] = (v422_data + (v394_data * (sycl::group_broadcast(item.get_sub_group(), v419_data, 7))));
                float v425_data = r1[5];
                float v428_data = r2[5];
                r2[5] = (v428_data + (v394_data * (sycl::group_broadcast(item.get_sub_group(), v425_data, 7))));
                float v431_data = r1[6];
                float v434_data = r2[6];
                r2[6] = (v434_data + (v394_data * (sycl::group_broadcast(item.get_sub_group(), v431_data, 7))));
                float v437_data = r1[7];
                float v440_data = r2[7];
                r2[7] = (v440_data + (v394_data * (sycl::group_broadcast(item.get_sub_group(), v437_data, 7))));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = store{r>s}(localShrMem0, r2);
              if (v8_lead < 8) {
                #pragma unroll
                for (int32_t v447_i1 = 0; v447_i1 < 8; ++v447_i1) {
                  float v449_data = r2[v447_i1];
                  int32_t v456_a = v8_lead + (v447_i1 * 8);
                  s0[(v456_a ^ ((v456_a >> 5) & 31))] = v449_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              // glb_m2 = +(s0, dims=[1])
              if (v8_lead < 8) {
                float v465_acc0 = 0.0f;
                #pragma unroll
                for (int32_t v464_r1 = 0; v464_r1 < 8; ++v464_r1) {
                  int32_t v472_a = v8_lead + (v464_r1 * 8);
                  float v476_data = s0[(v472_a ^ ((v472_a >> 5) & 31))];
                  v465_acc0 = (v465_acc0 + v476_data);
                }
                glb_m2[v8_lead] = v465_acc0;
              }
            }
          }
        }
      });
    }
  });
}

