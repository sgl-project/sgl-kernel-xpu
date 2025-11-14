Welcome to sgl-kernel-xpu! We are building high performance kernels for intel GPUs.
Appreciate your interest in contributing.This documentation provides a general guidelines for how to open a Pull Request.


We encourage you to open Pull Request from the following 4 perspectives:
* **Feature Enabling**: Enable a new feature for XPU device. For example, enable `fused_add_rmsnorm` for XPU.
* **Performance Optimization**: Improve the performance on XPU for critical operators, data types or quantization recipes. For example, support FP8 Flash Attention.
* **Module Refactoring**: Enhance the maintainability and readability of the code. For example, refactor the scratchpad memory allocation in kernels.
* **Bug Fixing**: Enhance test case coverage and code robustness.

### Guidelines for Pull Requests:
* Make sure update test cases when enabling a new feature and comprehesively cover all possible combinaitons in the test suite.
* Make sure update benchmarks cases when enabling a new feature or optimizing for new scenario especially for critical operators.
* Provide benchmark data to prove affectiveness of performance optimizations.
* Use main branch for device-agnostic features and optimizations.
* DO NOT expose hardware or software features for Intel(R) Protect IP on main branch.

### Review Process for Merging Pull Requests:
* Mark Pull Requests as `draft` and ask review from maintainers (mingfeima, airMeng, chunyuan-w, sunjiweiswift)
* At least one approval is needed before landing for simple Pull Requests
* At least two approvals are needed before landing for complexed Pull Requests.
* Mark Pull Requests as `ready for review` once comments are addressed.
* Pass all CI tests.

### Extra Requirements:
* For public Pull Requests on sglang repo, conduct internal review first before requesting reviews from community.
