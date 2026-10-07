// === base name ===
kernel_eab73fd1b460ea77

// === header ===
#ifndef TENSORFORGE_LAUNCH_TYPES
#define TENSORFORGE_LAUNCH_TYPES
#include <cstddef>
namespace tensorforge {
// Fixed when the kernel is generated: `launch_info_<kernel>`.
struct LaunchInfo {
  unsigned block[3];
  unsigned threadsPerMult;
  unsigned activeThreads;
  unsigned leadWidth;
  unsigned multsPerBlock;
  std::size_t sharedMemBytes;
  bool cooperative;
  bool persistent;
  unsigned sections;
};
// What one launch uses, the grid included: `launch_config_<kernel>`.
struct LaunchConfig {
  std::size_t grid[3];
  std::size_t block[3];
  std::size_t sharedMemBytes;
  bool cooperative;
};
} // namespace tensorforge
#endif
inline constexpr tensorforge::LaunchInfo launch_info_kernel_eab73fd1b460ea77 = {{16, 16, 1}, 16, 16, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_eab73fd1b460ea77(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_eab73fd1b460ea77(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
#ifndef TENSORFORGE_LAUNCH_TYPES
#define TENSORFORGE_LAUNCH_TYPES
#include <cstddef>
namespace tensorforge {
// Fixed when the kernel is generated: `launch_info_<kernel>`.
struct LaunchInfo {
  unsigned block[3];
  unsigned threadsPerMult;
  unsigned activeThreads;
  unsigned leadWidth;
  unsigned multsPerBlock;
  std::size_t sharedMemBytes;
  bool cooperative;
  bool persistent;
  unsigned sections;
};
// What one launch uses, the grid included: `launch_config_<kernel>`.
struct LaunchConfig {
  std::size_t grid[3];
  std::size_t block[3];
  std::size_t sharedMemBytes;
  bool cooperative;
};
} // namespace tensorforge
#endif
tensorforge::LaunchConfig launch_config_kernel_eab73fd1b460ea77(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (16, 16, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 16 - 1) / 16;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 16;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 256 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_eab73fd1b460ea77(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_eab73fd1b460ea77(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_eab73fd1b460ea77(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_eab73fd1b460ea77(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 2048 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} strided
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"double","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":2048,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          double* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v7_batchId0 * 256 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[v7_batchId0 * 256 + 0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v7_batchId0 * 46 + 0 + m2_extraOffset];
              double r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v21_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v22_i0 = 0; v22_i0 < 1; ++v22_i0) {
                int32_t v25_lead = v21_lead + (v22_i0 * 16);
                #pragma unroll
                for (int32_t v23_i1 = 0; v23_i1 < 16; ++v23_i1) {
                  double v28_data = glb_m1[(v25_lead + (v23_i1 * 16))];
                  r0[(v22_i0 + v23_i1)] = v28_data;
                }
              }
              double r1[16]{};
              // r1 = load{g>r}(glb_m2);
              double v31_lin = glb_m2[0 + item.get_local_id(2) * 1];
              r1[0] = v31_lin;
              double v32_lin = glb_m2[16 + item.get_local_id(2) * 1];
              r1[1] = v32_lin;
              double v33_lin = glb_m2[32 + item.get_local_id(2) * 1];
              r1[2] = v33_lin;
              double r2[16]{};
              // ir2 = +(r0 * r1)
              // [(0, 16), (0, 16)] [(0, 16)]
              double ir2[16]{};
              double v36_data = r0[0];
              double v37_data = r1[0];
              double v40_data = ir2[0];
              ir2[0] = (v40_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v46_data = ir2[1];
              ir2[1] = (v46_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v62_data = r0[1];
              double v66_data = ir2[0];
              ir2[0] = (v66_data + (v62_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v72_data = ir2[1];
              ir2[1] = (v72_data + (v62_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v78_data = ir2[2];
              ir2[2] = (v78_data + (v62_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v93_data = r0[2];
              double v98_data = ir2[1];
              ir2[1] = (v98_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v104_data = ir2[2];
              ir2[2] = (v104_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v110_data = ir2[3];
              ir2[3] = (v110_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v124_data = r0[3];
              double v130_data = ir2[2];
              ir2[2] = (v130_data + (v124_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v136_data = ir2[3];
              ir2[3] = (v136_data + (v124_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v142_data = ir2[4];
              ir2[4] = (v142_data + (v124_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v155_data = r0[4];
              double v162_data = ir2[3];
              ir2[3] = (v162_data + (v155_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v168_data = ir2[4];
              ir2[4] = (v168_data + (v155_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v174_data = ir2[5];
              ir2[5] = (v174_data + (v155_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v186_data = r0[5];
              double v194_data = ir2[4];
              ir2[4] = (v194_data + (v186_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v200_data = ir2[5];
              ir2[5] = (v200_data + (v186_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v203_data = r1[1];
              double v206_data = ir2[6];
              ir2[6] = (v206_data + (v186_data * (sycl::select_from_group(item.get_sub_group(), v203_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v217_data = r0[6];
              double v226_data = ir2[5];
              ir2[5] = (v226_data + (v217_data * (sycl::select_from_group(item.get_sub_group(), v203_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v232_data = ir2[6];
              ir2[6] = (v232_data + (v217_data * (sycl::select_from_group(item.get_sub_group(), v203_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v238_data = ir2[7];
              ir2[7] = (v238_data + (v217_data * (sycl::select_from_group(item.get_sub_group(), v203_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v248_data = r0[7];
              double v258_data = ir2[6];
              ir2[6] = (v258_data + (v248_data * (sycl::select_from_group(item.get_sub_group(), v203_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v264_data = ir2[7];
              ir2[7] = (v264_data + (v248_data * (sycl::select_from_group(item.get_sub_group(), v203_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v270_data = ir2[8];
              ir2[8] = (v270_data + (v248_data * (sycl::select_from_group(item.get_sub_group(), v203_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v279_data = r0[8];
              double v290_data = ir2[7];
              ir2[7] = (v290_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v203_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v296_data = ir2[8];
              ir2[8] = (v296_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v203_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v302_data = ir2[9];
              ir2[9] = (v302_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v203_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v310_data = r0[9];
              double v322_data = ir2[8];
              ir2[8] = (v322_data + (v310_data * (sycl::select_from_group(item.get_sub_group(), v203_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v328_data = ir2[9];
              ir2[9] = (v328_data + (v310_data * (sycl::select_from_group(item.get_sub_group(), v203_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v334_data = ir2[10];
              ir2[10] = (v334_data + (v310_data * (sycl::select_from_group(item.get_sub_group(), v203_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v341_data = r0[10];
              double v354_data = ir2[9];
              ir2[9] = (v354_data + (v341_data * (sycl::select_from_group(item.get_sub_group(), v203_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v360_data = ir2[10];
              ir2[10] = (v360_data + (v341_data * (sycl::select_from_group(item.get_sub_group(), v203_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v363_data = r1[2];
              double v366_data = ir2[11];
              ir2[11] = (v366_data + (v341_data * (sycl::select_from_group(item.get_sub_group(), v363_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v372_data = r0[11];
              double v386_data = ir2[10];
              ir2[10] = (v386_data + (v372_data * (sycl::select_from_group(item.get_sub_group(), v203_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v392_data = ir2[11];
              ir2[11] = (v392_data + (v372_data * (sycl::select_from_group(item.get_sub_group(), v363_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v398_data = ir2[12];
              ir2[12] = (v398_data + (v372_data * (sycl::select_from_group(item.get_sub_group(), v363_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v403_data = r0[12];
              double v418_data = ir2[11];
              ir2[11] = (v418_data + (v403_data * (sycl::select_from_group(item.get_sub_group(), v363_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v424_data = ir2[12];
              ir2[12] = (v424_data + (v403_data * (sycl::select_from_group(item.get_sub_group(), v363_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v430_data = ir2[13];
              ir2[13] = (v430_data + (v403_data * (sycl::select_from_group(item.get_sub_group(), v363_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v434_data = r0[13];
              double v450_data = ir2[12];
              ir2[12] = (v450_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v363_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v456_data = ir2[13];
              ir2[13] = (v456_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v363_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v462_data = ir2[14];
              ir2[14] = (v462_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v363_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v465_data = r0[14];
              double v482_data = ir2[13];
              ir2[13] = (v482_data + (v465_data * (sycl::select_from_group(item.get_sub_group(), v363_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v488_data = ir2[14];
              ir2[14] = (v488_data + (v465_data * (sycl::select_from_group(item.get_sub_group(), v363_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v494_data = ir2[15];
              ir2[15] = (v494_data + (v465_data * (sycl::select_from_group(item.get_sub_group(), v363_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v496_data = r0[15];
              double v514_data = ir2[14];
              ir2[14] = (v514_data + (v496_data * (sycl::select_from_group(item.get_sub_group(), v363_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v520_data = ir2[15];
              ir2[15] = (v520_data + (v496_data * (sycl::select_from_group(item.get_sub_group(), v363_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              // r2 = ir2
              #pragma unroll
              for (int32_t v522_n0 = 0; v522_n0 < 1; ++v522_n0) {
                #pragma unroll
                for (int32_t v523_n1 = 0; v523_n1 < 16; ++v523_n1) {
                  int32_t v524_a = v522_n0 + v523_n1;
                  double v525_data = ir2[v524_a];
                  r2[v524_a] = v525_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v526_i0 = 0; v526_i0 < 1; ++v526_i0) {
                int32_t v531_lead = v21_lead + (v526_i0 * 16);
                #pragma unroll
                for (int32_t v527_i1 = 0; v527_i1 < 16; ++v527_i1) {
                  double v529_data = r2[(v526_i0 + v527_i1)];
                  glb_m0[(v531_lead + (v527_i1 * 16))] = v529_data;
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

