// SPDX-License-Identifier: MIT
// QoLA stub <torch/all.h>
// The real <torch/all.h> transitively pulls in most of the C++ stdlib.
// AITER code relies on these transitive includes.
#pragma once

#include <ATen/ATen.h>

// Standard library headers that AITER code expects from <torch/all.h>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
