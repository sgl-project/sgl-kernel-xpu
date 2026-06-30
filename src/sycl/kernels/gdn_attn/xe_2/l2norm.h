#pragma once

#include <torch/all.h>

#include <sycl/sycl.hpp>

void l2norm(sycl::queue& queue, const torch::Tensor& q, const torch::Tensor& k);
