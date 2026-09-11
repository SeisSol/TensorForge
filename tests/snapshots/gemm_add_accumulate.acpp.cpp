// === base name ===
kernel_f67f9578dd166178

// === header ===
void launcher_kernel_f67f9578dd166178(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f67f9578dd166178(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_f67f9578dd166178(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f67f9578dd166178(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 12×16(12×16) {0..12}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m1 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
              const float *const __restrict__ glb_m1 = &m1[v2_batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v2_batchId0 * 128 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v16_lead = item.get_local_id(0) % 16;
              if (v16_lead < 12) {
                #pragma unroll
                for (int32_t v18_i1 = 0; v18_i1 < 16; ++v18_i1) {
                  float v26_data = glb_m1[(v16_lead + (v18_i1 * 12))];
                  r0[v18_i1] = v26_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v32_i0 = 0; v32_i0 < 1; ++v32_i0) {
                int32_t v38_lead = v16_lead + (v32_i0 * 16);
                #pragma unroll
                for (int32_t v33_i1 = 0; v33_i1 < 8; ++v33_i1) {
                  float v41_data = glb_m2[(v38_lead + (v33_i1 * 16))];
                  r1[(v32_i0 + v33_i1)] = v41_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r2[8]{};
              // r2 = load{g>r}(glb_m0);
              if (v16_lead < 12) {
                #pragma unroll
                for (int32_t v48_i1 = 0; v48_i1 < 8; ++v48_i1) {
                  float v56_data = glb_m0[(v16_lead + (v48_i1 * 12))];
                  r2[v48_i1] = v56_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m2););
              // wait(r2 = load{g>r}(glb_m0););
              float r3[8]{};
              // r3 = +(r0 * r1) + name: r2, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir3[8]{};
              if (v16_lead < 12) {
                float v64_data = r0[0];
                float v65_data = r1[0];
                float v68_data = ir3[0];
                ir3[0] = (v68_data + (v64_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 0))));
                float v71_data = r1[1];
                float v74_data = ir3[1];
                ir3[1] = (v74_data + (v64_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 0))));
                float v77_data = r1[2];
                float v80_data = ir3[2];
                ir3[2] = (v80_data + (v64_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 0))));
                float v83_data = r1[3];
                float v86_data = ir3[3];
                ir3[3] = (v86_data + (v64_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 0))));
                float v89_data = r1[4];
                float v92_data = ir3[4];
                ir3[4] = (v92_data + (v64_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 0))));
                float v95_data = r1[5];
                float v98_data = ir3[5];
                ir3[5] = (v98_data + (v64_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 0))));
                float v101_data = r1[6];
                float v104_data = ir3[6];
                ir3[6] = (v104_data + (v64_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 0))));
                float v107_data = r1[7];
                float v110_data = ir3[7];
                ir3[7] = (v110_data + (v64_data * (sycl::group_broadcast(item.get_sub_group(), v107_data, 0))));
              }
              if (v16_lead < 12) {
                float v116_data = r0[1];
                float v117_data = r1[0];
                float v120_data = ir3[0];
                ir3[0] = (v120_data + (v116_data * (sycl::group_broadcast(item.get_sub_group(), v117_data, 1))));
                float v123_data = r1[1];
                float v126_data = ir3[1];
                ir3[1] = (v126_data + (v116_data * (sycl::group_broadcast(item.get_sub_group(), v123_data, 1))));
                float v129_data = r1[2];
                float v132_data = ir3[2];
                ir3[2] = (v132_data + (v116_data * (sycl::group_broadcast(item.get_sub_group(), v129_data, 1))));
                float v135_data = r1[3];
                float v138_data = ir3[3];
                ir3[3] = (v138_data + (v116_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 1))));
                float v141_data = r1[4];
                float v144_data = ir3[4];
                ir3[4] = (v144_data + (v116_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 1))));
                float v147_data = r1[5];
                float v150_data = ir3[5];
                ir3[5] = (v150_data + (v116_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 1))));
                float v153_data = r1[6];
                float v156_data = ir3[6];
                ir3[6] = (v156_data + (v116_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 1))));
                float v159_data = r1[7];
                float v162_data = ir3[7];
                ir3[7] = (v162_data + (v116_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 1))));
              }
              if (v16_lead < 12) {
                float v168_data = r0[2];
                float v169_data = r1[0];
                float v172_data = ir3[0];
                ir3[0] = (v172_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 2))));
                float v175_data = r1[1];
                float v178_data = ir3[1];
                ir3[1] = (v178_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 2))));
                float v181_data = r1[2];
                float v184_data = ir3[2];
                ir3[2] = (v184_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v181_data, 2))));
                float v187_data = r1[3];
                float v190_data = ir3[3];
                ir3[3] = (v190_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 2))));
                float v193_data = r1[4];
                float v196_data = ir3[4];
                ir3[4] = (v196_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 2))));
                float v199_data = r1[5];
                float v202_data = ir3[5];
                ir3[5] = (v202_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v199_data, 2))));
                float v205_data = r1[6];
                float v208_data = ir3[6];
                ir3[6] = (v208_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v205_data, 2))));
                float v211_data = r1[7];
                float v214_data = ir3[7];
                ir3[7] = (v214_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v211_data, 2))));
              }
              if (v16_lead < 12) {
                float v220_data = r0[3];
                float v221_data = r1[0];
                float v224_data = ir3[0];
                ir3[0] = (v224_data + (v220_data * (sycl::group_broadcast(item.get_sub_group(), v221_data, 3))));
                float v227_data = r1[1];
                float v230_data = ir3[1];
                ir3[1] = (v230_data + (v220_data * (sycl::group_broadcast(item.get_sub_group(), v227_data, 3))));
                float v233_data = r1[2];
                float v236_data = ir3[2];
                ir3[2] = (v236_data + (v220_data * (sycl::group_broadcast(item.get_sub_group(), v233_data, 3))));
                float v239_data = r1[3];
                float v242_data = ir3[3];
                ir3[3] = (v242_data + (v220_data * (sycl::group_broadcast(item.get_sub_group(), v239_data, 3))));
                float v245_data = r1[4];
                float v248_data = ir3[4];
                ir3[4] = (v248_data + (v220_data * (sycl::group_broadcast(item.get_sub_group(), v245_data, 3))));
                float v251_data = r1[5];
                float v254_data = ir3[5];
                ir3[5] = (v254_data + (v220_data * (sycl::group_broadcast(item.get_sub_group(), v251_data, 3))));
                float v257_data = r1[6];
                float v260_data = ir3[6];
                ir3[6] = (v260_data + (v220_data * (sycl::group_broadcast(item.get_sub_group(), v257_data, 3))));
                float v263_data = r1[7];
                float v266_data = ir3[7];
                ir3[7] = (v266_data + (v220_data * (sycl::group_broadcast(item.get_sub_group(), v263_data, 3))));
              }
              if (v16_lead < 12) {
                float v272_data = r0[4];
                float v273_data = r1[0];
                float v276_data = ir3[0];
                ir3[0] = (v276_data + (v272_data * (sycl::group_broadcast(item.get_sub_group(), v273_data, 4))));
                float v279_data = r1[1];
                float v282_data = ir3[1];
                ir3[1] = (v282_data + (v272_data * (sycl::group_broadcast(item.get_sub_group(), v279_data, 4))));
                float v285_data = r1[2];
                float v288_data = ir3[2];
                ir3[2] = (v288_data + (v272_data * (sycl::group_broadcast(item.get_sub_group(), v285_data, 4))));
                float v291_data = r1[3];
                float v294_data = ir3[3];
                ir3[3] = (v294_data + (v272_data * (sycl::group_broadcast(item.get_sub_group(), v291_data, 4))));
                float v297_data = r1[4];
                float v300_data = ir3[4];
                ir3[4] = (v300_data + (v272_data * (sycl::group_broadcast(item.get_sub_group(), v297_data, 4))));
                float v303_data = r1[5];
                float v306_data = ir3[5];
                ir3[5] = (v306_data + (v272_data * (sycl::group_broadcast(item.get_sub_group(), v303_data, 4))));
                float v309_data = r1[6];
                float v312_data = ir3[6];
                ir3[6] = (v312_data + (v272_data * (sycl::group_broadcast(item.get_sub_group(), v309_data, 4))));
                float v315_data = r1[7];
                float v318_data = ir3[7];
                ir3[7] = (v318_data + (v272_data * (sycl::group_broadcast(item.get_sub_group(), v315_data, 4))));
              }
              if (v16_lead < 12) {
                float v324_data = r0[5];
                float v325_data = r1[0];
                float v328_data = ir3[0];
                ir3[0] = (v328_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v325_data, 5))));
                float v331_data = r1[1];
                float v334_data = ir3[1];
                ir3[1] = (v334_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v331_data, 5))));
                float v337_data = r1[2];
                float v340_data = ir3[2];
                ir3[2] = (v340_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v337_data, 5))));
                float v343_data = r1[3];
                float v346_data = ir3[3];
                ir3[3] = (v346_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v343_data, 5))));
                float v349_data = r1[4];
                float v352_data = ir3[4];
                ir3[4] = (v352_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v349_data, 5))));
                float v355_data = r1[5];
                float v358_data = ir3[5];
                ir3[5] = (v358_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v355_data, 5))));
                float v361_data = r1[6];
                float v364_data = ir3[6];
                ir3[6] = (v364_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v361_data, 5))));
                float v367_data = r1[7];
                float v370_data = ir3[7];
                ir3[7] = (v370_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v367_data, 5))));
              }
              if (v16_lead < 12) {
                float v376_data = r0[6];
                float v377_data = r1[0];
                float v380_data = ir3[0];
                ir3[0] = (v380_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v377_data, 6))));
                float v383_data = r1[1];
                float v386_data = ir3[1];
                ir3[1] = (v386_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v383_data, 6))));
                float v389_data = r1[2];
                float v392_data = ir3[2];
                ir3[2] = (v392_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v389_data, 6))));
                float v395_data = r1[3];
                float v398_data = ir3[3];
                ir3[3] = (v398_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v395_data, 6))));
                float v401_data = r1[4];
                float v404_data = ir3[4];
                ir3[4] = (v404_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v401_data, 6))));
                float v407_data = r1[5];
                float v410_data = ir3[5];
                ir3[5] = (v410_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v407_data, 6))));
                float v413_data = r1[6];
                float v416_data = ir3[6];
                ir3[6] = (v416_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v413_data, 6))));
                float v419_data = r1[7];
                float v422_data = ir3[7];
                ir3[7] = (v422_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v419_data, 6))));
              }
              if (v16_lead < 12) {
                float v428_data = r0[7];
                float v429_data = r1[0];
                float v432_data = ir3[0];
                ir3[0] = (v432_data + (v428_data * (sycl::group_broadcast(item.get_sub_group(), v429_data, 7))));
                float v435_data = r1[1];
                float v438_data = ir3[1];
                ir3[1] = (v438_data + (v428_data * (sycl::group_broadcast(item.get_sub_group(), v435_data, 7))));
                float v441_data = r1[2];
                float v444_data = ir3[2];
                ir3[2] = (v444_data + (v428_data * (sycl::group_broadcast(item.get_sub_group(), v441_data, 7))));
                float v447_data = r1[3];
                float v450_data = ir3[3];
                ir3[3] = (v450_data + (v428_data * (sycl::group_broadcast(item.get_sub_group(), v447_data, 7))));
                float v453_data = r1[4];
                float v456_data = ir3[4];
                ir3[4] = (v456_data + (v428_data * (sycl::group_broadcast(item.get_sub_group(), v453_data, 7))));
                float v459_data = r1[5];
                float v462_data = ir3[5];
                ir3[5] = (v462_data + (v428_data * (sycl::group_broadcast(item.get_sub_group(), v459_data, 7))));
                float v465_data = r1[6];
                float v468_data = ir3[6];
                ir3[6] = (v468_data + (v428_data * (sycl::group_broadcast(item.get_sub_group(), v465_data, 7))));
                float v471_data = r1[7];
                float v474_data = ir3[7];
                ir3[7] = (v474_data + (v428_data * (sycl::group_broadcast(item.get_sub_group(), v471_data, 7))));
              }
              if (v16_lead < 12) {
                float v480_data = r0[8];
                float v481_data = r1[0];
                float v484_data = ir3[0];
                ir3[0] = (v484_data + (v480_data * (sycl::group_broadcast(item.get_sub_group(), v481_data, 8))));
                float v487_data = r1[1];
                float v490_data = ir3[1];
                ir3[1] = (v490_data + (v480_data * (sycl::group_broadcast(item.get_sub_group(), v487_data, 8))));
                float v493_data = r1[2];
                float v496_data = ir3[2];
                ir3[2] = (v496_data + (v480_data * (sycl::group_broadcast(item.get_sub_group(), v493_data, 8))));
                float v499_data = r1[3];
                float v502_data = ir3[3];
                ir3[3] = (v502_data + (v480_data * (sycl::group_broadcast(item.get_sub_group(), v499_data, 8))));
                float v505_data = r1[4];
                float v508_data = ir3[4];
                ir3[4] = (v508_data + (v480_data * (sycl::group_broadcast(item.get_sub_group(), v505_data, 8))));
                float v511_data = r1[5];
                float v514_data = ir3[5];
                ir3[5] = (v514_data + (v480_data * (sycl::group_broadcast(item.get_sub_group(), v511_data, 8))));
                float v517_data = r1[6];
                float v520_data = ir3[6];
                ir3[6] = (v520_data + (v480_data * (sycl::group_broadcast(item.get_sub_group(), v517_data, 8))));
                float v523_data = r1[7];
                float v526_data = ir3[7];
                ir3[7] = (v526_data + (v480_data * (sycl::group_broadcast(item.get_sub_group(), v523_data, 8))));
              }
              if (v16_lead < 12) {
                float v532_data = r0[9];
                float v533_data = r1[0];
                float v536_data = ir3[0];
                ir3[0] = (v536_data + (v532_data * (sycl::group_broadcast(item.get_sub_group(), v533_data, 9))));
                float v539_data = r1[1];
                float v542_data = ir3[1];
                ir3[1] = (v542_data + (v532_data * (sycl::group_broadcast(item.get_sub_group(), v539_data, 9))));
                float v545_data = r1[2];
                float v548_data = ir3[2];
                ir3[2] = (v548_data + (v532_data * (sycl::group_broadcast(item.get_sub_group(), v545_data, 9))));
                float v551_data = r1[3];
                float v554_data = ir3[3];
                ir3[3] = (v554_data + (v532_data * (sycl::group_broadcast(item.get_sub_group(), v551_data, 9))));
                float v557_data = r1[4];
                float v560_data = ir3[4];
                ir3[4] = (v560_data + (v532_data * (sycl::group_broadcast(item.get_sub_group(), v557_data, 9))));
                float v563_data = r1[5];
                float v566_data = ir3[5];
                ir3[5] = (v566_data + (v532_data * (sycl::group_broadcast(item.get_sub_group(), v563_data, 9))));
                float v569_data = r1[6];
                float v572_data = ir3[6];
                ir3[6] = (v572_data + (v532_data * (sycl::group_broadcast(item.get_sub_group(), v569_data, 9))));
                float v575_data = r1[7];
                float v578_data = ir3[7];
                ir3[7] = (v578_data + (v532_data * (sycl::group_broadcast(item.get_sub_group(), v575_data, 9))));
              }
              if (v16_lead < 12) {
                float v584_data = r0[10];
                float v585_data = r1[0];
                float v588_data = ir3[0];
                ir3[0] = (v588_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v585_data, 10))));
                float v591_data = r1[1];
                float v594_data = ir3[1];
                ir3[1] = (v594_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v591_data, 10))));
                float v597_data = r1[2];
                float v600_data = ir3[2];
                ir3[2] = (v600_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v597_data, 10))));
                float v603_data = r1[3];
                float v606_data = ir3[3];
                ir3[3] = (v606_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v603_data, 10))));
                float v609_data = r1[4];
                float v612_data = ir3[4];
                ir3[4] = (v612_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v609_data, 10))));
                float v615_data = r1[5];
                float v618_data = ir3[5];
                ir3[5] = (v618_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v615_data, 10))));
                float v621_data = r1[6];
                float v624_data = ir3[6];
                ir3[6] = (v624_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v621_data, 10))));
                float v627_data = r1[7];
                float v630_data = ir3[7];
                ir3[7] = (v630_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v627_data, 10))));
              }
              if (v16_lead < 12) {
                float v636_data = r0[11];
                float v637_data = r1[0];
                float v640_data = ir3[0];
                ir3[0] = (v640_data + (v636_data * (sycl::group_broadcast(item.get_sub_group(), v637_data, 11))));
                float v643_data = r1[1];
                float v646_data = ir3[1];
                ir3[1] = (v646_data + (v636_data * (sycl::group_broadcast(item.get_sub_group(), v643_data, 11))));
                float v649_data = r1[2];
                float v652_data = ir3[2];
                ir3[2] = (v652_data + (v636_data * (sycl::group_broadcast(item.get_sub_group(), v649_data, 11))));
                float v655_data = r1[3];
                float v658_data = ir3[3];
                ir3[3] = (v658_data + (v636_data * (sycl::group_broadcast(item.get_sub_group(), v655_data, 11))));
                float v661_data = r1[4];
                float v664_data = ir3[4];
                ir3[4] = (v664_data + (v636_data * (sycl::group_broadcast(item.get_sub_group(), v661_data, 11))));
                float v667_data = r1[5];
                float v670_data = ir3[5];
                ir3[5] = (v670_data + (v636_data * (sycl::group_broadcast(item.get_sub_group(), v667_data, 11))));
                float v673_data = r1[6];
                float v676_data = ir3[6];
                ir3[6] = (v676_data + (v636_data * (sycl::group_broadcast(item.get_sub_group(), v673_data, 11))));
                float v679_data = r1[7];
                float v682_data = ir3[7];
                ir3[7] = (v682_data + (v636_data * (sycl::group_broadcast(item.get_sub_group(), v679_data, 11))));
              }
              if (v16_lead < 12) {
                float v688_data = r0[12];
                float v689_data = r1[0];
                float v692_data = ir3[0];
                ir3[0] = (v692_data + (v688_data * (sycl::group_broadcast(item.get_sub_group(), v689_data, 12))));
                float v695_data = r1[1];
                float v698_data = ir3[1];
                ir3[1] = (v698_data + (v688_data * (sycl::group_broadcast(item.get_sub_group(), v695_data, 12))));
                float v701_data = r1[2];
                float v704_data = ir3[2];
                ir3[2] = (v704_data + (v688_data * (sycl::group_broadcast(item.get_sub_group(), v701_data, 12))));
                float v707_data = r1[3];
                float v710_data = ir3[3];
                ir3[3] = (v710_data + (v688_data * (sycl::group_broadcast(item.get_sub_group(), v707_data, 12))));
                float v713_data = r1[4];
                float v716_data = ir3[4];
                ir3[4] = (v716_data + (v688_data * (sycl::group_broadcast(item.get_sub_group(), v713_data, 12))));
                float v719_data = r1[5];
                float v722_data = ir3[5];
                ir3[5] = (v722_data + (v688_data * (sycl::group_broadcast(item.get_sub_group(), v719_data, 12))));
                float v725_data = r1[6];
                float v728_data = ir3[6];
                ir3[6] = (v728_data + (v688_data * (sycl::group_broadcast(item.get_sub_group(), v725_data, 12))));
                float v731_data = r1[7];
                float v734_data = ir3[7];
                ir3[7] = (v734_data + (v688_data * (sycl::group_broadcast(item.get_sub_group(), v731_data, 12))));
              }
              if (v16_lead < 12) {
                float v740_data = r0[13];
                float v741_data = r1[0];
                float v744_data = ir3[0];
                ir3[0] = (v744_data + (v740_data * (sycl::group_broadcast(item.get_sub_group(), v741_data, 13))));
                float v747_data = r1[1];
                float v750_data = ir3[1];
                ir3[1] = (v750_data + (v740_data * (sycl::group_broadcast(item.get_sub_group(), v747_data, 13))));
                float v753_data = r1[2];
                float v756_data = ir3[2];
                ir3[2] = (v756_data + (v740_data * (sycl::group_broadcast(item.get_sub_group(), v753_data, 13))));
                float v759_data = r1[3];
                float v762_data = ir3[3];
                ir3[3] = (v762_data + (v740_data * (sycl::group_broadcast(item.get_sub_group(), v759_data, 13))));
                float v765_data = r1[4];
                float v768_data = ir3[4];
                ir3[4] = (v768_data + (v740_data * (sycl::group_broadcast(item.get_sub_group(), v765_data, 13))));
                float v771_data = r1[5];
                float v774_data = ir3[5];
                ir3[5] = (v774_data + (v740_data * (sycl::group_broadcast(item.get_sub_group(), v771_data, 13))));
                float v777_data = r1[6];
                float v780_data = ir3[6];
                ir3[6] = (v780_data + (v740_data * (sycl::group_broadcast(item.get_sub_group(), v777_data, 13))));
                float v783_data = r1[7];
                float v786_data = ir3[7];
                ir3[7] = (v786_data + (v740_data * (sycl::group_broadcast(item.get_sub_group(), v783_data, 13))));
              }
              if (v16_lead < 12) {
                float v792_data = r0[14];
                float v793_data = r1[0];
                float v796_data = ir3[0];
                ir3[0] = (v796_data + (v792_data * (sycl::group_broadcast(item.get_sub_group(), v793_data, 14))));
                float v799_data = r1[1];
                float v802_data = ir3[1];
                ir3[1] = (v802_data + (v792_data * (sycl::group_broadcast(item.get_sub_group(), v799_data, 14))));
                float v805_data = r1[2];
                float v808_data = ir3[2];
                ir3[2] = (v808_data + (v792_data * (sycl::group_broadcast(item.get_sub_group(), v805_data, 14))));
                float v811_data = r1[3];
                float v814_data = ir3[3];
                ir3[3] = (v814_data + (v792_data * (sycl::group_broadcast(item.get_sub_group(), v811_data, 14))));
                float v817_data = r1[4];
                float v820_data = ir3[4];
                ir3[4] = (v820_data + (v792_data * (sycl::group_broadcast(item.get_sub_group(), v817_data, 14))));
                float v823_data = r1[5];
                float v826_data = ir3[5];
                ir3[5] = (v826_data + (v792_data * (sycl::group_broadcast(item.get_sub_group(), v823_data, 14))));
                float v829_data = r1[6];
                float v832_data = ir3[6];
                ir3[6] = (v832_data + (v792_data * (sycl::group_broadcast(item.get_sub_group(), v829_data, 14))));
                float v835_data = r1[7];
                float v838_data = ir3[7];
                ir3[7] = (v838_data + (v792_data * (sycl::group_broadcast(item.get_sub_group(), v835_data, 14))));
              }
              if (v16_lead < 12) {
                float v844_data = r0[15];
                float v845_data = r1[0];
                float v848_data = ir3[0];
                ir3[0] = (v848_data + (v844_data * (sycl::group_broadcast(item.get_sub_group(), v845_data, 15))));
                float v851_data = r1[1];
                float v854_data = ir3[1];
                ir3[1] = (v854_data + (v844_data * (sycl::group_broadcast(item.get_sub_group(), v851_data, 15))));
                float v857_data = r1[2];
                float v860_data = ir3[2];
                ir3[2] = (v860_data + (v844_data * (sycl::group_broadcast(item.get_sub_group(), v857_data, 15))));
                float v863_data = r1[3];
                float v866_data = ir3[3];
                ir3[3] = (v866_data + (v844_data * (sycl::group_broadcast(item.get_sub_group(), v863_data, 15))));
                float v869_data = r1[4];
                float v872_data = ir3[4];
                ir3[4] = (v872_data + (v844_data * (sycl::group_broadcast(item.get_sub_group(), v869_data, 15))));
                float v875_data = r1[5];
                float v878_data = ir3[5];
                ir3[5] = (v878_data + (v844_data * (sycl::group_broadcast(item.get_sub_group(), v875_data, 15))));
                float v881_data = r1[6];
                float v884_data = ir3[6];
                ir3[6] = (v884_data + (v844_data * (sycl::group_broadcast(item.get_sub_group(), v881_data, 15))));
                float v887_data = r1[7];
                float v890_data = ir3[7];
                ir3[7] = (v890_data + (v844_data * (sycl::group_broadcast(item.get_sub_group(), v887_data, 15))));
              }
              if (v16_lead < 12) {
                #pragma unroll
                for (int32_t v896_n1 = 0; v896_n1 < 8; ++v896_n1) {
                  float v898_data = ir3[v896_n1];
                  float v900_data = r2[v896_n1];
                  r3[v896_n1] = (v900_data + v898_data);
                }
              }
              // glb_m0 = store{r>g}(r3);
              if (v16_lead < 12) {
                #pragma unroll
                for (int32_t v907_i1 = 0; v907_i1 < 8; ++v907_i1) {
                  float v909_data = r3[v907_i1];
                  glb_m0[(v16_lead + (v907_i1 * 12))] = v909_data;
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

