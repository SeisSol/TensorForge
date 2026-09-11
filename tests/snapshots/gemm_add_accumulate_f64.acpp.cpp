// === base name ===
kernel_bc65b1c832dc7d23

// === header ===
void launcher_kernel_bc65b1c832dc7d23(double* m0, size_t m0_extraOffset, const double* m1, size_t m1_extraOffset, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_bc65b1c832dc7d23(double* m0, size_t m0_extraOffset, const double* m1, size_t m1_extraOffset, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_bc65b1c832dc7d23(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_bc65b1c832dc7d23(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double* m0, size_t m0_extraOffset, const double* m1, size_t m1_extraOffset, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
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
          double* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          double* tempShrMem = &localShrMem0[0];
          for (size_t v2_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v2_batchId0 < numElements0; v2_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v3_ahead1 = v2_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v5_batchId1 = (v3_ahead1 < numElements0) ? v3_ahead1 : v2_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v2_batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v2_batchId0 * 96 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[v2_batchId0 * 192 + 0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v2_batchId0 * 128 + 0 + m2_extraOffset];
              double r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v16_lead = item.get_local_id(0) % 16;
              if (v16_lead < 12) {
                #pragma unroll
                for (int32_t v18_i1 = 0; v18_i1 < 16; ++v18_i1) {
                  double v26_data = glb_m1[(v16_lead + (v18_i1 * 12))];
                  r0[v18_i1] = v26_data;
                }
              }
              double r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v32_i0 = 0; v32_i0 < 1; ++v32_i0) {
                int32_t v38_lead = v16_lead + (v32_i0 * 16);
                #pragma unroll
                for (int32_t v33_i1 = 0; v33_i1 < 8; ++v33_i1) {
                  double v41_data = glb_m2[(v38_lead + (v33_i1 * 16))];
                  r1[(v32_i0 + v33_i1)] = v41_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              double r2[8]{};
              // r2 = load{g>r}(glb_m0);
              if (v16_lead < 12) {
                #pragma unroll
                for (int32_t v48_i1 = 0; v48_i1 < 8; ++v48_i1) {
                  double v56_data = glb_m0[(v16_lead + (v48_i1 * 12))];
                  r2[v48_i1] = v56_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m2););
              // wait(r2 = load{g>r}(glb_m0););
              double r3[8]{};
              // r3 = +(r0 * r1) + name: r2, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 16)]
              double ir3[8]{};
              if (v16_lead < 12) {
                double v64_data = r0[0];
                double v65_data = r1[0];
                double v68_data = ir3[0];
                ir3[0] = (v68_data + (v64_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 0))));
                double v71_data = r1[1];
                double v74_data = ir3[1];
                ir3[1] = (v74_data + (v64_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 0))));
                double v77_data = r1[2];
                double v80_data = ir3[2];
                ir3[2] = (v80_data + (v64_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 0))));
                double v83_data = r1[3];
                double v86_data = ir3[3];
                ir3[3] = (v86_data + (v64_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 0))));
                double v89_data = r1[4];
                double v92_data = ir3[4];
                ir3[4] = (v92_data + (v64_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 0))));
                double v95_data = r1[5];
                double v98_data = ir3[5];
                ir3[5] = (v98_data + (v64_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 0))));
                double v101_data = r1[6];
                double v104_data = ir3[6];
                ir3[6] = (v104_data + (v64_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 0))));
                double v107_data = r1[7];
                double v110_data = ir3[7];
                ir3[7] = (v110_data + (v64_data * (sycl::group_broadcast(item.get_sub_group(), v107_data, 0))));
              }
              if (v16_lead < 12) {
                double v116_data = r0[1];
                double v117_data = r1[0];
                double v120_data = ir3[0];
                ir3[0] = (v120_data + (v116_data * (sycl::group_broadcast(item.get_sub_group(), v117_data, 1))));
                double v123_data = r1[1];
                double v126_data = ir3[1];
                ir3[1] = (v126_data + (v116_data * (sycl::group_broadcast(item.get_sub_group(), v123_data, 1))));
                double v129_data = r1[2];
                double v132_data = ir3[2];
                ir3[2] = (v132_data + (v116_data * (sycl::group_broadcast(item.get_sub_group(), v129_data, 1))));
                double v135_data = r1[3];
                double v138_data = ir3[3];
                ir3[3] = (v138_data + (v116_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 1))));
                double v141_data = r1[4];
                double v144_data = ir3[4];
                ir3[4] = (v144_data + (v116_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 1))));
                double v147_data = r1[5];
                double v150_data = ir3[5];
                ir3[5] = (v150_data + (v116_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 1))));
                double v153_data = r1[6];
                double v156_data = ir3[6];
                ir3[6] = (v156_data + (v116_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 1))));
                double v159_data = r1[7];
                double v162_data = ir3[7];
                ir3[7] = (v162_data + (v116_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 1))));
              }
              if (v16_lead < 12) {
                double v168_data = r0[2];
                double v169_data = r1[0];
                double v172_data = ir3[0];
                ir3[0] = (v172_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 2))));
                double v175_data = r1[1];
                double v178_data = ir3[1];
                ir3[1] = (v178_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 2))));
                double v181_data = r1[2];
                double v184_data = ir3[2];
                ir3[2] = (v184_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v181_data, 2))));
                double v187_data = r1[3];
                double v190_data = ir3[3];
                ir3[3] = (v190_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 2))));
                double v193_data = r1[4];
                double v196_data = ir3[4];
                ir3[4] = (v196_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 2))));
                double v199_data = r1[5];
                double v202_data = ir3[5];
                ir3[5] = (v202_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v199_data, 2))));
                double v205_data = r1[6];
                double v208_data = ir3[6];
                ir3[6] = (v208_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v205_data, 2))));
                double v211_data = r1[7];
                double v214_data = ir3[7];
                ir3[7] = (v214_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v211_data, 2))));
              }
              if (v16_lead < 12) {
                double v220_data = r0[3];
                double v221_data = r1[0];
                double v224_data = ir3[0];
                ir3[0] = (v224_data + (v220_data * (sycl::group_broadcast(item.get_sub_group(), v221_data, 3))));
                double v227_data = r1[1];
                double v230_data = ir3[1];
                ir3[1] = (v230_data + (v220_data * (sycl::group_broadcast(item.get_sub_group(), v227_data, 3))));
                double v233_data = r1[2];
                double v236_data = ir3[2];
                ir3[2] = (v236_data + (v220_data * (sycl::group_broadcast(item.get_sub_group(), v233_data, 3))));
                double v239_data = r1[3];
                double v242_data = ir3[3];
                ir3[3] = (v242_data + (v220_data * (sycl::group_broadcast(item.get_sub_group(), v239_data, 3))));
                double v245_data = r1[4];
                double v248_data = ir3[4];
                ir3[4] = (v248_data + (v220_data * (sycl::group_broadcast(item.get_sub_group(), v245_data, 3))));
                double v251_data = r1[5];
                double v254_data = ir3[5];
                ir3[5] = (v254_data + (v220_data * (sycl::group_broadcast(item.get_sub_group(), v251_data, 3))));
                double v257_data = r1[6];
                double v260_data = ir3[6];
                ir3[6] = (v260_data + (v220_data * (sycl::group_broadcast(item.get_sub_group(), v257_data, 3))));
                double v263_data = r1[7];
                double v266_data = ir3[7];
                ir3[7] = (v266_data + (v220_data * (sycl::group_broadcast(item.get_sub_group(), v263_data, 3))));
              }
              if (v16_lead < 12) {
                double v272_data = r0[4];
                double v273_data = r1[0];
                double v276_data = ir3[0];
                ir3[0] = (v276_data + (v272_data * (sycl::group_broadcast(item.get_sub_group(), v273_data, 4))));
                double v279_data = r1[1];
                double v282_data = ir3[1];
                ir3[1] = (v282_data + (v272_data * (sycl::group_broadcast(item.get_sub_group(), v279_data, 4))));
                double v285_data = r1[2];
                double v288_data = ir3[2];
                ir3[2] = (v288_data + (v272_data * (sycl::group_broadcast(item.get_sub_group(), v285_data, 4))));
                double v291_data = r1[3];
                double v294_data = ir3[3];
                ir3[3] = (v294_data + (v272_data * (sycl::group_broadcast(item.get_sub_group(), v291_data, 4))));
                double v297_data = r1[4];
                double v300_data = ir3[4];
                ir3[4] = (v300_data + (v272_data * (sycl::group_broadcast(item.get_sub_group(), v297_data, 4))));
                double v303_data = r1[5];
                double v306_data = ir3[5];
                ir3[5] = (v306_data + (v272_data * (sycl::group_broadcast(item.get_sub_group(), v303_data, 4))));
                double v309_data = r1[6];
                double v312_data = ir3[6];
                ir3[6] = (v312_data + (v272_data * (sycl::group_broadcast(item.get_sub_group(), v309_data, 4))));
                double v315_data = r1[7];
                double v318_data = ir3[7];
                ir3[7] = (v318_data + (v272_data * (sycl::group_broadcast(item.get_sub_group(), v315_data, 4))));
              }
              if (v16_lead < 12) {
                double v324_data = r0[5];
                double v325_data = r1[0];
                double v328_data = ir3[0];
                ir3[0] = (v328_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v325_data, 5))));
                double v331_data = r1[1];
                double v334_data = ir3[1];
                ir3[1] = (v334_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v331_data, 5))));
                double v337_data = r1[2];
                double v340_data = ir3[2];
                ir3[2] = (v340_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v337_data, 5))));
                double v343_data = r1[3];
                double v346_data = ir3[3];
                ir3[3] = (v346_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v343_data, 5))));
                double v349_data = r1[4];
                double v352_data = ir3[4];
                ir3[4] = (v352_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v349_data, 5))));
                double v355_data = r1[5];
                double v358_data = ir3[5];
                ir3[5] = (v358_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v355_data, 5))));
                double v361_data = r1[6];
                double v364_data = ir3[6];
                ir3[6] = (v364_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v361_data, 5))));
                double v367_data = r1[7];
                double v370_data = ir3[7];
                ir3[7] = (v370_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v367_data, 5))));
              }
              if (v16_lead < 12) {
                double v376_data = r0[6];
                double v377_data = r1[0];
                double v380_data = ir3[0];
                ir3[0] = (v380_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v377_data, 6))));
                double v383_data = r1[1];
                double v386_data = ir3[1];
                ir3[1] = (v386_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v383_data, 6))));
                double v389_data = r1[2];
                double v392_data = ir3[2];
                ir3[2] = (v392_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v389_data, 6))));
                double v395_data = r1[3];
                double v398_data = ir3[3];
                ir3[3] = (v398_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v395_data, 6))));
                double v401_data = r1[4];
                double v404_data = ir3[4];
                ir3[4] = (v404_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v401_data, 6))));
                double v407_data = r1[5];
                double v410_data = ir3[5];
                ir3[5] = (v410_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v407_data, 6))));
                double v413_data = r1[6];
                double v416_data = ir3[6];
                ir3[6] = (v416_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v413_data, 6))));
                double v419_data = r1[7];
                double v422_data = ir3[7];
                ir3[7] = (v422_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v419_data, 6))));
              }
              if (v16_lead < 12) {
                double v428_data = r0[7];
                double v429_data = r1[0];
                double v432_data = ir3[0];
                ir3[0] = (v432_data + (v428_data * (sycl::group_broadcast(item.get_sub_group(), v429_data, 7))));
                double v435_data = r1[1];
                double v438_data = ir3[1];
                ir3[1] = (v438_data + (v428_data * (sycl::group_broadcast(item.get_sub_group(), v435_data, 7))));
                double v441_data = r1[2];
                double v444_data = ir3[2];
                ir3[2] = (v444_data + (v428_data * (sycl::group_broadcast(item.get_sub_group(), v441_data, 7))));
                double v447_data = r1[3];
                double v450_data = ir3[3];
                ir3[3] = (v450_data + (v428_data * (sycl::group_broadcast(item.get_sub_group(), v447_data, 7))));
                double v453_data = r1[4];
                double v456_data = ir3[4];
                ir3[4] = (v456_data + (v428_data * (sycl::group_broadcast(item.get_sub_group(), v453_data, 7))));
                double v459_data = r1[5];
                double v462_data = ir3[5];
                ir3[5] = (v462_data + (v428_data * (sycl::group_broadcast(item.get_sub_group(), v459_data, 7))));
                double v465_data = r1[6];
                double v468_data = ir3[6];
                ir3[6] = (v468_data + (v428_data * (sycl::group_broadcast(item.get_sub_group(), v465_data, 7))));
                double v471_data = r1[7];
                double v474_data = ir3[7];
                ir3[7] = (v474_data + (v428_data * (sycl::group_broadcast(item.get_sub_group(), v471_data, 7))));
              }
              if (v16_lead < 12) {
                double v480_data = r0[8];
                double v481_data = r1[0];
                double v484_data = ir3[0];
                ir3[0] = (v484_data + (v480_data * (sycl::group_broadcast(item.get_sub_group(), v481_data, 8))));
                double v487_data = r1[1];
                double v490_data = ir3[1];
                ir3[1] = (v490_data + (v480_data * (sycl::group_broadcast(item.get_sub_group(), v487_data, 8))));
                double v493_data = r1[2];
                double v496_data = ir3[2];
                ir3[2] = (v496_data + (v480_data * (sycl::group_broadcast(item.get_sub_group(), v493_data, 8))));
                double v499_data = r1[3];
                double v502_data = ir3[3];
                ir3[3] = (v502_data + (v480_data * (sycl::group_broadcast(item.get_sub_group(), v499_data, 8))));
                double v505_data = r1[4];
                double v508_data = ir3[4];
                ir3[4] = (v508_data + (v480_data * (sycl::group_broadcast(item.get_sub_group(), v505_data, 8))));
                double v511_data = r1[5];
                double v514_data = ir3[5];
                ir3[5] = (v514_data + (v480_data * (sycl::group_broadcast(item.get_sub_group(), v511_data, 8))));
                double v517_data = r1[6];
                double v520_data = ir3[6];
                ir3[6] = (v520_data + (v480_data * (sycl::group_broadcast(item.get_sub_group(), v517_data, 8))));
                double v523_data = r1[7];
                double v526_data = ir3[7];
                ir3[7] = (v526_data + (v480_data * (sycl::group_broadcast(item.get_sub_group(), v523_data, 8))));
              }
              if (v16_lead < 12) {
                double v532_data = r0[9];
                double v533_data = r1[0];
                double v536_data = ir3[0];
                ir3[0] = (v536_data + (v532_data * (sycl::group_broadcast(item.get_sub_group(), v533_data, 9))));
                double v539_data = r1[1];
                double v542_data = ir3[1];
                ir3[1] = (v542_data + (v532_data * (sycl::group_broadcast(item.get_sub_group(), v539_data, 9))));
                double v545_data = r1[2];
                double v548_data = ir3[2];
                ir3[2] = (v548_data + (v532_data * (sycl::group_broadcast(item.get_sub_group(), v545_data, 9))));
                double v551_data = r1[3];
                double v554_data = ir3[3];
                ir3[3] = (v554_data + (v532_data * (sycl::group_broadcast(item.get_sub_group(), v551_data, 9))));
                double v557_data = r1[4];
                double v560_data = ir3[4];
                ir3[4] = (v560_data + (v532_data * (sycl::group_broadcast(item.get_sub_group(), v557_data, 9))));
                double v563_data = r1[5];
                double v566_data = ir3[5];
                ir3[5] = (v566_data + (v532_data * (sycl::group_broadcast(item.get_sub_group(), v563_data, 9))));
                double v569_data = r1[6];
                double v572_data = ir3[6];
                ir3[6] = (v572_data + (v532_data * (sycl::group_broadcast(item.get_sub_group(), v569_data, 9))));
                double v575_data = r1[7];
                double v578_data = ir3[7];
                ir3[7] = (v578_data + (v532_data * (sycl::group_broadcast(item.get_sub_group(), v575_data, 9))));
              }
              if (v16_lead < 12) {
                double v584_data = r0[10];
                double v585_data = r1[0];
                double v588_data = ir3[0];
                ir3[0] = (v588_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v585_data, 10))));
                double v591_data = r1[1];
                double v594_data = ir3[1];
                ir3[1] = (v594_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v591_data, 10))));
                double v597_data = r1[2];
                double v600_data = ir3[2];
                ir3[2] = (v600_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v597_data, 10))));
                double v603_data = r1[3];
                double v606_data = ir3[3];
                ir3[3] = (v606_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v603_data, 10))));
                double v609_data = r1[4];
                double v612_data = ir3[4];
                ir3[4] = (v612_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v609_data, 10))));
                double v615_data = r1[5];
                double v618_data = ir3[5];
                ir3[5] = (v618_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v615_data, 10))));
                double v621_data = r1[6];
                double v624_data = ir3[6];
                ir3[6] = (v624_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v621_data, 10))));
                double v627_data = r1[7];
                double v630_data = ir3[7];
                ir3[7] = (v630_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v627_data, 10))));
              }
              if (v16_lead < 12) {
                double v636_data = r0[11];
                double v637_data = r1[0];
                double v640_data = ir3[0];
                ir3[0] = (v640_data + (v636_data * (sycl::group_broadcast(item.get_sub_group(), v637_data, 11))));
                double v643_data = r1[1];
                double v646_data = ir3[1];
                ir3[1] = (v646_data + (v636_data * (sycl::group_broadcast(item.get_sub_group(), v643_data, 11))));
                double v649_data = r1[2];
                double v652_data = ir3[2];
                ir3[2] = (v652_data + (v636_data * (sycl::group_broadcast(item.get_sub_group(), v649_data, 11))));
                double v655_data = r1[3];
                double v658_data = ir3[3];
                ir3[3] = (v658_data + (v636_data * (sycl::group_broadcast(item.get_sub_group(), v655_data, 11))));
                double v661_data = r1[4];
                double v664_data = ir3[4];
                ir3[4] = (v664_data + (v636_data * (sycl::group_broadcast(item.get_sub_group(), v661_data, 11))));
                double v667_data = r1[5];
                double v670_data = ir3[5];
                ir3[5] = (v670_data + (v636_data * (sycl::group_broadcast(item.get_sub_group(), v667_data, 11))));
                double v673_data = r1[6];
                double v676_data = ir3[6];
                ir3[6] = (v676_data + (v636_data * (sycl::group_broadcast(item.get_sub_group(), v673_data, 11))));
                double v679_data = r1[7];
                double v682_data = ir3[7];
                ir3[7] = (v682_data + (v636_data * (sycl::group_broadcast(item.get_sub_group(), v679_data, 11))));
              }
              if (v16_lead < 12) {
                double v688_data = r0[12];
                double v689_data = r1[0];
                double v692_data = ir3[0];
                ir3[0] = (v692_data + (v688_data * (sycl::group_broadcast(item.get_sub_group(), v689_data, 12))));
                double v695_data = r1[1];
                double v698_data = ir3[1];
                ir3[1] = (v698_data + (v688_data * (sycl::group_broadcast(item.get_sub_group(), v695_data, 12))));
                double v701_data = r1[2];
                double v704_data = ir3[2];
                ir3[2] = (v704_data + (v688_data * (sycl::group_broadcast(item.get_sub_group(), v701_data, 12))));
                double v707_data = r1[3];
                double v710_data = ir3[3];
                ir3[3] = (v710_data + (v688_data * (sycl::group_broadcast(item.get_sub_group(), v707_data, 12))));
                double v713_data = r1[4];
                double v716_data = ir3[4];
                ir3[4] = (v716_data + (v688_data * (sycl::group_broadcast(item.get_sub_group(), v713_data, 12))));
                double v719_data = r1[5];
                double v722_data = ir3[5];
                ir3[5] = (v722_data + (v688_data * (sycl::group_broadcast(item.get_sub_group(), v719_data, 12))));
                double v725_data = r1[6];
                double v728_data = ir3[6];
                ir3[6] = (v728_data + (v688_data * (sycl::group_broadcast(item.get_sub_group(), v725_data, 12))));
                double v731_data = r1[7];
                double v734_data = ir3[7];
                ir3[7] = (v734_data + (v688_data * (sycl::group_broadcast(item.get_sub_group(), v731_data, 12))));
              }
              if (v16_lead < 12) {
                double v740_data = r0[13];
                double v741_data = r1[0];
                double v744_data = ir3[0];
                ir3[0] = (v744_data + (v740_data * (sycl::group_broadcast(item.get_sub_group(), v741_data, 13))));
                double v747_data = r1[1];
                double v750_data = ir3[1];
                ir3[1] = (v750_data + (v740_data * (sycl::group_broadcast(item.get_sub_group(), v747_data, 13))));
                double v753_data = r1[2];
                double v756_data = ir3[2];
                ir3[2] = (v756_data + (v740_data * (sycl::group_broadcast(item.get_sub_group(), v753_data, 13))));
                double v759_data = r1[3];
                double v762_data = ir3[3];
                ir3[3] = (v762_data + (v740_data * (sycl::group_broadcast(item.get_sub_group(), v759_data, 13))));
                double v765_data = r1[4];
                double v768_data = ir3[4];
                ir3[4] = (v768_data + (v740_data * (sycl::group_broadcast(item.get_sub_group(), v765_data, 13))));
                double v771_data = r1[5];
                double v774_data = ir3[5];
                ir3[5] = (v774_data + (v740_data * (sycl::group_broadcast(item.get_sub_group(), v771_data, 13))));
                double v777_data = r1[6];
                double v780_data = ir3[6];
                ir3[6] = (v780_data + (v740_data * (sycl::group_broadcast(item.get_sub_group(), v777_data, 13))));
                double v783_data = r1[7];
                double v786_data = ir3[7];
                ir3[7] = (v786_data + (v740_data * (sycl::group_broadcast(item.get_sub_group(), v783_data, 13))));
              }
              if (v16_lead < 12) {
                double v792_data = r0[14];
                double v793_data = r1[0];
                double v796_data = ir3[0];
                ir3[0] = (v796_data + (v792_data * (sycl::group_broadcast(item.get_sub_group(), v793_data, 14))));
                double v799_data = r1[1];
                double v802_data = ir3[1];
                ir3[1] = (v802_data + (v792_data * (sycl::group_broadcast(item.get_sub_group(), v799_data, 14))));
                double v805_data = r1[2];
                double v808_data = ir3[2];
                ir3[2] = (v808_data + (v792_data * (sycl::group_broadcast(item.get_sub_group(), v805_data, 14))));
                double v811_data = r1[3];
                double v814_data = ir3[3];
                ir3[3] = (v814_data + (v792_data * (sycl::group_broadcast(item.get_sub_group(), v811_data, 14))));
                double v817_data = r1[4];
                double v820_data = ir3[4];
                ir3[4] = (v820_data + (v792_data * (sycl::group_broadcast(item.get_sub_group(), v817_data, 14))));
                double v823_data = r1[5];
                double v826_data = ir3[5];
                ir3[5] = (v826_data + (v792_data * (sycl::group_broadcast(item.get_sub_group(), v823_data, 14))));
                double v829_data = r1[6];
                double v832_data = ir3[6];
                ir3[6] = (v832_data + (v792_data * (sycl::group_broadcast(item.get_sub_group(), v829_data, 14))));
                double v835_data = r1[7];
                double v838_data = ir3[7];
                ir3[7] = (v838_data + (v792_data * (sycl::group_broadcast(item.get_sub_group(), v835_data, 14))));
              }
              if (v16_lead < 12) {
                double v844_data = r0[15];
                double v845_data = r1[0];
                double v848_data = ir3[0];
                ir3[0] = (v848_data + (v844_data * (sycl::group_broadcast(item.get_sub_group(), v845_data, 15))));
                double v851_data = r1[1];
                double v854_data = ir3[1];
                ir3[1] = (v854_data + (v844_data * (sycl::group_broadcast(item.get_sub_group(), v851_data, 15))));
                double v857_data = r1[2];
                double v860_data = ir3[2];
                ir3[2] = (v860_data + (v844_data * (sycl::group_broadcast(item.get_sub_group(), v857_data, 15))));
                double v863_data = r1[3];
                double v866_data = ir3[3];
                ir3[3] = (v866_data + (v844_data * (sycl::group_broadcast(item.get_sub_group(), v863_data, 15))));
                double v869_data = r1[4];
                double v872_data = ir3[4];
                ir3[4] = (v872_data + (v844_data * (sycl::group_broadcast(item.get_sub_group(), v869_data, 15))));
                double v875_data = r1[5];
                double v878_data = ir3[5];
                ir3[5] = (v878_data + (v844_data * (sycl::group_broadcast(item.get_sub_group(), v875_data, 15))));
                double v881_data = r1[6];
                double v884_data = ir3[6];
                ir3[6] = (v884_data + (v844_data * (sycl::group_broadcast(item.get_sub_group(), v881_data, 15))));
                double v887_data = r1[7];
                double v890_data = ir3[7];
                ir3[7] = (v890_data + (v844_data * (sycl::group_broadcast(item.get_sub_group(), v887_data, 15))));
              }
              if (v16_lead < 12) {
                #pragma unroll
                for (int32_t v896_n1 = 0; v896_n1 < 8; ++v896_n1) {
                  double v898_data = ir3[v896_n1];
                  double v900_data = r2[v896_n1];
                  r3[v896_n1] = (v900_data + v898_data);
                }
              }
              // glb_m0 = store{r>g}(r3);
              if (v16_lead < 12) {
                #pragma unroll
                for (int32_t v907_i1 = 0; v907_i1 < 8; ++v907_i1) {
                  double v909_data = r3[v907_i1];
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

