FAILED: GenerationError: section 0 cannot be emitted:
verify: 5 error(s), 0 note(s)
  [error] @17: group barrier inside a construct whose trip count is only simd-uniform. Threads that execute a different number of iterations never arrive at the barrier, so the kernel deadlocks rather than producing a wrong answer.
  [error] @19: group barrier inside a construct whose trip count is only simd-uniform. Threads that execute a different number of iterations never arrive at the barrier, so the kernel deadlocks rather than producing a wrong answer.
  [error] @25: group barrier inside a construct whose trip count is only simd-uniform. Threads that execute a different number of iterations never arrive at the barrier, so the kernel deadlocks rather than producing a wrong answer.
  [error] @27: group barrier inside a construct whose trip count is only simd-uniform. Threads that execute a different number of iterations never arrive at the barrier, so the kernel deadlocks rather than producing a wrong answer.
  [error] @31: group barrier inside a construct whose trip count is only simd-uniform. Threads that execute a different number of iterations never arrive at the barrier, so the kernel deadlocks rather than producing a wrong answer.
