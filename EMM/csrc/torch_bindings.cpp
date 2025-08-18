#include <pybind11/pybind11.h>
#include <string>
#include <torch/extension.h>
#include <vector>

#include "allocator.hpp"
#include "constants.hpp"
#include "torch_utils.hpp"

namespace emm {

std::vector<torch::Tensor> create_kv_tensors(size_t size, size_t dtype_size,
                                             const std::string &dev_str,
                                             int64_t num_layers) {
  auto allocator = FTensorAllocator::global_allocator();
  auto dtype_ = torch_dtype_from_size(dtype_size);
  return allocator->create_kv_tensors(size, dtype_, dev_str, num_layers);
}

bool map_to_kv_tensors(const std::vector<offset_t> &offsets) {
  auto allocator = FTensorAllocator::global_allocator();
  return allocator->map_to_kv_tensors(offsets);
}

bool unmap_from_kv_tensors(const std::vector<offset_t> &offsets) {
  auto allocator = FTensorAllocator::global_allocator();
  return allocator->unmap_from_kv_tensors(offsets);
}

void free_kv_tensors() {
  auto allocator = FTensorAllocator::global_allocator();
  allocator->free_kv_tensors();
}

} // namespace emm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "emm VMM plugin";

  m.def("init_emm", &emm::FTensorAllocator::init,
        "Initialize emm");
  m.def("shutdown_emm", &emm::FTensorAllocator::shutdown,
        "Shutdown emm");
  // m.def("create_ktensors", &emm::create_ktensors, "create_ktensors");
  // m.def("create_vtensors", &emm::create_vtensors, "create_vtensors");
  m.def("create_kv_tensors", &emm::create_kv_tensors, "create_kv_tensors");
  m.def("free_kv_tensors", &emm::free_kv_tensors, "free_kv_tensors");
  m.def("map_to_kv_tensors", &emm::map_to_kv_tensors, "map_to_kv_tensors");
  m.def("unmap_from_kv_tensors", &emm::unmap_from_kv_tensors,
        "unmap_from_kv_tensors");
}
