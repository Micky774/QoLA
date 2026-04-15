// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// QoLA stub <ATen/ATen.h>
//
// Provides the minimal torch/ATen surface needed to compile AITER's
// CK kernel instance code without linking libtorch.  Only the narrow
// API tier is supported: Tensor metadata access (size, stride, data_ptr,
// dtype, dim, sizes), ScalarType enum, TORCH_CHECK, and size_to_dim_.
//
// The stub Tensor carries pre-injected metadata — it does NOT manage
// storage, autograd, or device guards.
#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

// ---- ScalarType enum (matches c10::ScalarType values) ----

namespace c10 {

enum class ScalarType : int8_t {
    Byte = 0, Char = 1, Short = 2, Int = 3, Long = 4,
    Half = 5, Float = 6, Double = 7,
    ComplexHalf = 8, ComplexFloat = 9, ComplexDouble = 10,
    Bool = 11, QInt8 = 12, QUInt8 = 13, QInt32 = 14,
    BFloat16 = 15,
    // FP8/FP4 types added by ROCm/AITER
    Float8_e5m2 = 23, Float8_e4m3fn = 24,
    Float8_e4m3fnuz = 25, Float8_e5m2fnuz = 26,
    Undefined = 46,
    NumOptions = 47,
};

} // namespace c10

namespace at {
using ScalarType = c10::ScalarType;
} // namespace at

// ---- IntArrayRef (minimal span-like for sizes/strides) ----

namespace c10 {

class IntArrayRef {
    const int64_t* data_;
    size_t size_;
public:
    IntArrayRef() : data_(nullptr), size_(0) {}
    IntArrayRef(const int64_t* d, size_t n) : data_(d), size_(n) {}
    IntArrayRef(std::initializer_list<int64_t> il)
        : data_(il.begin()), size_(il.size()) {}
    size_t size() const { return size_; }
    const int64_t& operator[](size_t i) const { return data_[i]; }
    const int64_t* begin() const { return data_; }
    const int64_t* end() const { return data_ + size_; }
};

} // namespace c10

namespace at {
using IntArrayRef = c10::IntArrayRef;
} // namespace at

// ---- TORCH_CHECK macro ----

#define TORCH_CHECK(cond, ...)                                        \
    do {                                                              \
        if (!(cond)) {                                                \
            throw std::runtime_error("TORCH_CHECK failed: " #cond);  \
        }                                                             \
    } while (0)

// ---- Stub Tensor ----

// Forward-declare the stream accessor so zero_() can use it.
namespace at { namespace hip { namespace detail {
    hipStream_t& qola_current_stream();
}}}

namespace at {

class Tensor {
    void* ptr_;
    int64_t sizes_[8];
    int64_t strides_[8];
    int ndim_;
    ScalarType dtype_;

public:
    Tensor() : ptr_(nullptr), ndim_(0), dtype_(ScalarType::Undefined) {
        for (auto& s : sizes_) s = 0;
        for (auto& s : strides_) s = 0;
    }

    Tensor(void* p, std::initializer_list<int64_t> sz,
           std::initializer_list<int64_t> st, ScalarType dt)
        : ptr_(p), ndim_(static_cast<int>(sz.size())), dtype_(dt) {
        int i = 0;
        for (auto v : sz) sizes_[i++] = v;
        for (; i < 8; ++i) sizes_[i] = 0;
        i = 0;
        for (auto v : st) strides_[i++] = v;
        for (; i < 8; ++i) strides_[i] = 0;
    }

    void* data_ptr() const { return ptr_; }

    template <typename T>
    T* data_ptr() const { return static_cast<T*>(ptr_); }

    int64_t size(int64_t dim) const {
        if (dim < 0) dim += ndim_;
        return sizes_[dim];
    }

    int64_t stride(int64_t dim) const {
        if (dim < 0) dim += ndim_;
        return strides_[dim];
    }

    int dim() const { return ndim_; }

    c10::IntArrayRef sizes() const {
        return c10::IntArrayRef(sizes_, static_cast<size_t>(ndim_));
    }

    ScalarType dtype() const { return dtype_; }
    ScalarType scalar_type() const { return dtype_; }

    int64_t numel() const {
        int64_t n = 1;
        for (int i = 0; i < ndim_; ++i) n *= sizes_[i];
        return n;
    }

    int64_t element_size() const {
        switch (dtype_) {
            case ScalarType::Half:
            case ScalarType::BFloat16: return 2;
            case ScalarType::Float: return 4;
            case ScalarType::Double: return 8;
            case ScalarType::Byte:
            case ScalarType::Char:
            case ScalarType::Bool:
            case ScalarType::QInt8:
            case ScalarType::QUInt8: return 1;
            case ScalarType::Short: return 2;
            case ScalarType::Int:
            case ScalarType::QInt32: return 4;
            case ScalarType::Long: return 8;
            default: return 1;
        }
    }

    // In-place zero fill via hipMemsetAsync on the QoLA-injected stream.
    void zero_() const {
        (void)hipMemsetAsync(ptr_, 0,
                             static_cast<size_t>(numel()) * element_size(),
                             at::hip::detail::qola_current_stream());
    }

    bool has_value() const { return ptr_ != nullptr; }
};

} // namespace at

namespace torch {
using Tensor = at::Tensor;
} // namespace torch

// ---- torch::ScalarType constants (torch::kBFloat16, etc.) ----

namespace torch {
constexpr auto kByte      = at::ScalarType::Byte;
constexpr auto kChar      = at::ScalarType::Char;
constexpr auto kShort     = at::ScalarType::Short;
constexpr auto kInt       = at::ScalarType::Int;
constexpr auto kInt32     = at::ScalarType::Int;
constexpr auto kLong      = at::ScalarType::Long;
constexpr auto kHalf      = at::ScalarType::Half;
constexpr auto kFloat     = at::ScalarType::Float;
constexpr auto kFloat32   = at::ScalarType::Float;
constexpr auto kDouble    = at::ScalarType::Double;
constexpr auto kBFloat16  = at::ScalarType::BFloat16;
constexpr auto kBool      = at::ScalarType::Bool;
constexpr auto kUInt8     = at::ScalarType::Byte;
constexpr auto kInt8      = at::ScalarType::Char;
constexpr auto kFloat8_e4m3fn    = at::ScalarType::Float8_e4m3fn;
constexpr auto kFloat8_e4m3fnuz  = at::ScalarType::Float8_e4m3fnuz;
constexpr auto kFloat8_e5m2      = at::ScalarType::Float8_e5m2;
constexpr auto kFloat8_e5m2fnuz  = at::ScalarType::Float8_e5m2fnuz;
// FP4 type — may or may not exist in the torch build.
#ifdef TORCH_Float4_e2m1fn_x2
constexpr auto kFloat4_e2m1fn_x2 = at::ScalarType::Byte; // packed fp4x2 stored as uint8
#endif

using ScalarType = at::ScalarType;
} // namespace torch

// ---- device_of / OptionalHIPGuard (no-ops — caller owns device) ----

namespace at {

inline int device_of(const Tensor&) { return 0; }

namespace hip {
struct OptionalHIPGuardMasqueradingAsCUDA {
    explicit OptionalHIPGuardMasqueradingAsCUDA(int /*device*/) {}
};
} // namespace hip
} // namespace at

// ---- c10::Half / c10::BFloat16 (tag types for t2ck<> specializations) ----

namespace c10 {
struct Half {};
struct BFloat16 {};
} // namespace c10

// ---- caffe2::TypeMeta (minimal stub for torchDTypeToStr) ----

namespace caffe2 {
struct TypeMeta {
    c10::ScalarType st_;
    TypeMeta() : st_(c10::ScalarType::Undefined) {}
    TypeMeta(c10::ScalarType s) : st_(s) {}
    c10::ScalarType toScalarType() const { return st_; }
    bool operator==(c10::ScalarType s) const { return st_ == s; }
    friend bool operator==(c10::ScalarType s, const TypeMeta& t) { return t.st_ == s; }
};
} // namespace caffe2

// ---- size_to_dim_ (from c10/core/TensorImpl.h) ----

inline int64_t size_to_dim_(int k, c10::IntArrayRef dims) {
    int64_t r = 1;
    for (int i = 0; i < k; ++i) r *= dims[i];
    return r;
}
