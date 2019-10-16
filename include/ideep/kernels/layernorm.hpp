#ifndef IDEEP_KERNELS_LAYERNORM_HPP
#define IDEEP_KERNELS_LAYERNORM_HPP

#include "common.hpp"

namespace ideep {

struct layer_normalization_forward : public dnnl::layer_normalization_forward {
  static inline void compute() {
  }
};

struct layer_normalization_backward: public dnnl::layer_normalization_backward {
  static inline void compute() {
  }
};

}  // namespace ideep

#endif