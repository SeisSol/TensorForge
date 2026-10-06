// === base name ===
kernel_a53f27688724a83e

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_a53f27688724a83e = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_a53f27688724a83e(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_a53f27688724a83e(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_a53f27688724a83e(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_a53f27688724a83e(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_a53f27688724a83e(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_a53f27688724a83e(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_a53f27688724a83e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} strided
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v9_batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 46 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v23_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
                int32_t v27_lead = v23_lead + (v24_i0 * 16);
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 16; ++v25_i1) {
                  float v30_data = glb_m1[(v27_lead + (v25_i1 * 16))];
                  r0[(v24_i0 + v25_i1)] = v30_data;
                }
              }
              float r1[16]{};
              // r1 = load{g>r}(glb_m2);
              float v33_lin = glb_m2[0 + item.get_local_id(2) * 1];
              r1[0] = v33_lin;
              float v34_lin = glb_m2[16 + item.get_local_id(2) * 1];
              r1[1] = v34_lin;
              float v35_lin = glb_m2[32 + item.get_local_id(2) * 1];
              r1[2] = v35_lin;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[16]{};
              // ir2 = +(r0 * r1)
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir2[16]{};
              float v38_data = r0[0];
              float v39_data = r1[0];
              float v42_data = ir2[0];
              ir2[0] = (v42_data + (v38_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v48_data = ir2[1];
              ir2[1] = (v48_data + (v38_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v64_data = r0[1];
              float v68_data = ir2[0];
              ir2[0] = (v68_data + (v64_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v74_data = ir2[1];
              ir2[1] = (v74_data + (v64_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v80_data = ir2[2];
              ir2[2] = (v80_data + (v64_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v95_data = r0[2];
              float v100_data = ir2[1];
              ir2[1] = (v100_data + (v95_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v106_data = ir2[2];
              ir2[2] = (v106_data + (v95_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v112_data = ir2[3];
              ir2[3] = (v112_data + (v95_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v126_data = r0[3];
              float v132_data = ir2[2];
              ir2[2] = (v132_data + (v126_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v138_data = ir2[3];
              ir2[3] = (v138_data + (v126_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v144_data = ir2[4];
              ir2[4] = (v144_data + (v126_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v157_data = r0[4];
              float v164_data = ir2[3];
              ir2[3] = (v164_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v170_data = ir2[4];
              ir2[4] = (v170_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v176_data = ir2[5];
              ir2[5] = (v176_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v188_data = r0[5];
              float v196_data = ir2[4];
              ir2[4] = (v196_data + (v188_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v202_data = ir2[5];
              ir2[5] = (v202_data + (v188_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v205_data = r1[1];
              float v208_data = ir2[6];
              ir2[6] = (v208_data + (v188_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v219_data = r0[6];
              float v228_data = ir2[5];
              ir2[5] = (v228_data + (v219_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v234_data = ir2[6];
              ir2[6] = (v234_data + (v219_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v240_data = ir2[7];
              ir2[7] = (v240_data + (v219_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v250_data = r0[7];
              float v260_data = ir2[6];
              ir2[6] = (v260_data + (v250_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v266_data = ir2[7];
              ir2[7] = (v266_data + (v250_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v272_data = ir2[8];
              ir2[8] = (v272_data + (v250_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v281_data = r0[8];
              float v292_data = ir2[7];
              ir2[7] = (v292_data + (v281_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v298_data = ir2[8];
              ir2[8] = (v298_data + (v281_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v304_data = ir2[9];
              ir2[9] = (v304_data + (v281_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v312_data = r0[9];
              float v324_data = ir2[8];
              ir2[8] = (v324_data + (v312_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v330_data = ir2[9];
              ir2[9] = (v330_data + (v312_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v336_data = ir2[10];
              ir2[10] = (v336_data + (v312_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v343_data = r0[10];
              float v356_data = ir2[9];
              ir2[9] = (v356_data + (v343_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v362_data = ir2[10];
              ir2[10] = (v362_data + (v343_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v365_data = r1[2];
              float v368_data = ir2[11];
              ir2[11] = (v368_data + (v343_data * (sycl::select_from_group(item.get_sub_group(), v365_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v374_data = r0[11];
              float v388_data = ir2[10];
              ir2[10] = (v388_data + (v374_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v394_data = ir2[11];
              ir2[11] = (v394_data + (v374_data * (sycl::select_from_group(item.get_sub_group(), v365_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v400_data = ir2[12];
              ir2[12] = (v400_data + (v374_data * (sycl::select_from_group(item.get_sub_group(), v365_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v405_data = r0[12];
              float v420_data = ir2[11];
              ir2[11] = (v420_data + (v405_data * (sycl::select_from_group(item.get_sub_group(), v365_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v426_data = ir2[12];
              ir2[12] = (v426_data + (v405_data * (sycl::select_from_group(item.get_sub_group(), v365_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v432_data = ir2[13];
              ir2[13] = (v432_data + (v405_data * (sycl::select_from_group(item.get_sub_group(), v365_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v436_data = r0[13];
              float v452_data = ir2[12];
              ir2[12] = (v452_data + (v436_data * (sycl::select_from_group(item.get_sub_group(), v365_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v458_data = ir2[13];
              ir2[13] = (v458_data + (v436_data * (sycl::select_from_group(item.get_sub_group(), v365_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v464_data = ir2[14];
              ir2[14] = (v464_data + (v436_data * (sycl::select_from_group(item.get_sub_group(), v365_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v467_data = r0[14];
              float v484_data = ir2[13];
              ir2[13] = (v484_data + (v467_data * (sycl::select_from_group(item.get_sub_group(), v365_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v490_data = ir2[14];
              ir2[14] = (v490_data + (v467_data * (sycl::select_from_group(item.get_sub_group(), v365_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v496_data = ir2[15];
              ir2[15] = (v496_data + (v467_data * (sycl::select_from_group(item.get_sub_group(), v365_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v498_data = r0[15];
              float v516_data = ir2[14];
              ir2[14] = (v516_data + (v498_data * (sycl::select_from_group(item.get_sub_group(), v365_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v522_data = ir2[15];
              ir2[15] = (v522_data + (v498_data * (sycl::select_from_group(item.get_sub_group(), v365_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              // r2 = ir2
              #pragma unroll
              for (int32_t v524_n0 = 0; v524_n0 < 1; ++v524_n0) {
                #pragma unroll
                for (int32_t v525_n1 = 0; v525_n1 < 16; ++v525_n1) {
                  int32_t v526_a = v524_n0 + v525_n1;
                  float v527_data = ir2[v526_a];
                  r2[v526_a] = v527_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v528_i0 = 0; v528_i0 < 1; ++v528_i0) {
                int32_t v533_lead = v23_lead + (v528_i0 * 16);
                #pragma unroll
                for (int32_t v529_i1 = 0; v529_i1 < 16; ++v529_i1) {
                  float v531_data = r2[(v528_i0 + v529_i1)];
                  glb_m0[(v533_lead + (v529_i1 * 16))] = v531_data;
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

