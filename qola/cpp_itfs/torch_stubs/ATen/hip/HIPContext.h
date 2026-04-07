// SPDX-License-Identifier: MIT
// QoLA stub <ATen/hip/HIPContext.h>
// Provides at::hip::getCurrentHIPStream() backed by a thread-local hipStream_t.
#pragma once

#include <hip/hip_runtime.h>

namespace at { namespace hip {

// Thin wrapper so StreamConfig{at::hip::getCurrentHIPStream()} compiles.
struct HIPStream {
    hipStream_t stream_;
    explicit HIPStream(hipStream_t s = nullptr) : stream_(s) {}
    operator hipStream_t() const { return stream_; }
};

namespace detail {
// Thread-local stream set by QoLA's entry point before calling AITER code.
inline hipStream_t& qola_current_stream() {
    static thread_local hipStream_t s = nullptr;
    return s;
}
} // namespace detail

inline HIPStream getCurrentHIPStream(int /*device_index*/ = -1) {
    return HIPStream{detail::qola_current_stream()};
}

// Called by QoLA's cpp_itfs entry point to inject the caller's stream.
inline void qola_set_stream(hipStream_t s) {
    detail::qola_current_stream() = s;
}

}} // namespace at::hip
