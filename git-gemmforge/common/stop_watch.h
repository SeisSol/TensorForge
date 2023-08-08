/**
 * @class utils::StopWatch
 *
 * @brief Measures time intervals
 * */

#ifndef LBM_STOP_WATCH_H
#define LBM_STOP_WATCH_H

#include <chrono>

namespace utils {
  template <class D>
  class StopWatch {
  public:

    /**
     * @brief Sets up a beginning of a new time interval.
     * */
    void start() { m_TimePoint = std::chrono::steady_clock::now(); };

    /**
     * @brief Interrupts counting from the last start() call and accumulate time.
     * */
    void stop() { m_Duration += (std::chrono::steady_clock::now() - m_TimePoint); };

    /**
     * @brief Sets duration time buffer to zero.
     * */
    void reset() { m_Duration = std::chrono::steady_clock::duration::zero(); }

    /**
     * @brief Returns accumulated time duration from the last call to reset().
     * */
    typename D::rep getTime() { return std::chrono::duration_cast<D>(m_Duration).count(); }

  private:
    std::chrono::time_point<std::chrono::steady_clock> m_TimePoint;
    std::chrono::steady_clock::duration m_Duration = std::chrono::steady_clock::duration::zero();
  };
}

#endif //LBM_STOP_WATCH_H
