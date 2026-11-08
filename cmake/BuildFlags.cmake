# Setup building flags for SYCL device and host codes.

# Import device IP version detection utility.
include(${CMAKE_CURRENT_LIST_DIR}/DeviceDetection.cmake)

function(CHECK_SYCL_FLAG FLAG VARIABLE_NAME)
  set(TEMP_DIR "${CMAKE_BINARY_DIR}/temp")
  file(MAKE_DIRECTORY ${TEMP_DIR})
  set(TEST_SRC_FILE "${TEMP_DIR}/check_options.cpp")
  set(TEST_EXE_FILE "${TEMP_DIR}/check_options.out")
  file(WRITE ${TEST_SRC_FILE} "#include <iostream>\nint main() { std::cout << \"Checking compiler options ...\" << std::endl; return 0; }\n")
  execute_process(
      COMMAND ${SYCL_COMPILER} -fsycl -ftemplate-backtrace-limit=0 ${TEST_SRC_FILE} -o ${TEST_EXE_FILE} ${FLAG}
      WORKING_DIRECTORY ${TEMP_DIR}
      OUTPUT_VARIABLE output
      ERROR_VARIABLE output
      RESULT_VARIABLE result
      TIMEOUT 60
  )
  if(result EQUAL 0)
      set(${VARIABLE_NAME} TRUE PARENT_SCOPE)
  else()
      set(${VARIABLE_NAME} FALSE PARENT_SCOPE)
  endif()
  file(REMOVE_RECURSE ${TEMP_DIR})
endfunction()

# Support GCC on Linux.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(STATUS "sgl-kernel-xpu build type: ${CMAKE_BUILD_TYPE}")
  set(SYCL_HOST_FLAGS)
  set(SYCL_KERNEL_OPTIONS)
  set(SYCL_COMPILE_FLAGS ${SYCL_FLAGS})
  set(SYCL_DEVICE_LINK_FLAGS ${SYCL_LINK_FLAGS})
  set(SYCL_OFFLINE_COMPILER_AOT_OPTIONS)
  set(SYCL_OFFLINE_COMPILER_CG_OPTIONS)
  set(SYCL_OFFLINE_COMPILER_FLAGS)

  # # -- Host flags (SYCL_CXX_FLAGS)
  list(APPEND SYCL_HOST_FLAGS -fPIC)
  list(APPEND SYCL_HOST_FLAGS -std=c++20)
  # Security-baseline diagnostics. Must stay ahead of the -Wno-* block below:
  # last flag wins, so the SYCL/torch header suppressions have to sit to the
  # right of -Wall/-Wextra. -Wconversion is deliberately excluded -- CuTe/sycl-tla index
  # arithmetic emits it by the thousand from headers we do not own.
  set(SGL_SDL_WARNING_FLAGS
    -Wall
    -Wextra
    -Wimplicit-fallthrough
    -Wformat
    -Wformat-security
    -Werror=format-security)
  # Opt-in regression gate. Requires the SYSTEM include dirs (CMakeLists.txt,
  # src/BuildOnLinux.cmake) so only first-party diagnostics are fatal.
  option(SGL_WERROR "Treat compiler warnings in first-party sources as errors" OFF)
  if(SGL_WERROR)
    list(APPEND SGL_SDL_WARNING_FLAGS -Werror)
  endif()
  list(APPEND SYCL_HOST_FLAGS ${SGL_SDL_WARNING_FLAGS})
  # SYCL headers warnings
  list(APPEND SYCL_HOST_FLAGS -Wno-deprecated-declarations)
  list(APPEND SYCL_HOST_FLAGS -Wno-deprecated)
  list(APPEND SYCL_HOST_FLAGS -Wno-attributes)
  list(APPEND SYCL_HOST_FLAGS -Wno-sign-compare)
  # Kernel sources are compiled twice: once by icpx for the device and once by
  # the -fsycl-host-compiler (gcc) for the host. `#pragma unroll` and
  # `#pragma clang diagnostic` are consumed by the device pass, but gcc cannot
  # act on either (its unroll spelling is `#pragma GCC unroll N`) and warns on
  # every occurrence -- ~34k diagnostics, none actionable. The device pass still
  # honours the pragmas, so nothing is lost by silencing the host pass.
  list(APPEND SYCL_HOST_FLAGS -Wno-unknown-pragmas)
  list(APPEND SYCL_HOST_FLAGS -fvisibility=hidden)
  list(APPEND SYCL_HOST_FLAGS -fvisibility-inlines-hidden)

  # Host-code hardening. x86-only: the driver warns and ignores
  # -fcf-protection for spir64, so these must never reach SYCL_KERNEL_OPTIONS.
  # Probed because the accepted set moves between oneAPI releases and the
  # runtime-JIT compile reuses this list verbatim.
  set(SGL_SDL_HARDENING_FLAGS)
  foreach(_sdl_flag -fcf-protection=full -fstack-protector-strong)
    string(MAKE_C_IDENTIFIER "SGL_SUPPORTS_${_sdl_flag}" _sdl_flag_var)
    CHECK_SYCL_FLAG("${_sdl_flag}" ${_sdl_flag_var})
    if(${_sdl_flag_var})
      list(APPEND SGL_SDL_HARDENING_FLAGS ${_sdl_flag})
    else()
      message(WARNING "Compiler rejects '${_sdl_flag}'; hardening is incomplete for this build.")
    endif()
  endforeach()
  # _FORTIFY_SOURCE is a no-op below -O1 and warns at -O0. -U first because the
  # oneAPI/conda toolchain may already define it.
  if(NOT CMAKE_BUILD_TYPE MATCHES Debug)
    list(APPEND SGL_SDL_HARDENING_FLAGS -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=2)
  endif()
  list(APPEND SYCL_HOST_FLAGS ${SGL_SDL_HARDENING_FLAGS})

  if(CMAKE_BUILD_TYPE MATCHES Debug)
    list(APPEND SYCL_HOST_FLAGS -g -fno-omit-frame-pointer -O0)
  elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    list(APPEND SYCL_HOST_FLAGS -g -O2)
  elseif(CMAKE_BUILD_TYPE MATCHES Release)
    list(APPEND SYCL_HOST_FLAGS -O3)
  endif()
  if(USE_PER_OPERATOR_HEADERS)
    list(APPEND SYCL_HOST_FLAGS -DAT_PER_OPERATOR_HEADERS)
  endif()
  if("${__INTEL_LLVM_COMPILER}" MATCHES "^[0-9]+$")
    list(APPEND SYCL_HOST_FLAGS -DSGL_INTEL_LLVM_COMPILER_VERSION=${__INTEL_LLVM_COMPILER})
  endif()
  # -- Kernel flags (SYCL_KERNEL_OPTIONS)
  # The fast-math will be enabled by default in SYCL compiler.
  # Refer to [https://clang.llvm.org/docs/UsersManual.html#cmdoption-fno-fast-math]
  # 1. We enable below flags here to be warn about NaN and Infinity,
  # which will be hidden by fast-math by default.
  # 2. The associative-math in fast-math allows floating point
  # operations to be reassociated, which will lead to non-deterministic
  # results compared with CUDA backend.
  # 3. The approx-func allows certain math function calls (such as log, sqrt, pow, etc)
  # to be replaced with an approximately equivalent set of instructions or
  # alternative math function calls, which have great errors.
  #
  # PSEUDO of separate compilation with DPCPP compiler.
  # 1. Kernel source compilation:
  # icpx -fsycl -fsycl-target=${SYCL_TARGETS_OPTION} ${SYCL_FLAGS} -fsycl-host-compiler=gcc -fsycl-host-compiler-options='${CMAKE_HOST_FLAGS}' kernel.cpp -o kernel.o
  # 2. Device code linkage:
  # icpx -fsycl -fsycl-target=${SYCL_TARGETS_OPTION} -fsycl-link ${SYCL_DEVICE_LINK_FLAGS} -Xs '${SYCL_OFFLINE_COMPILER_FLAGS}' kernel.o -o device-code.o
  # 3. Host only source compilation:
  # gcc ${CMAKE_HOST_FLAGS} host.cpp -o host.o
  # 4. Linkage:
  # gcc -shared host.o kernel.o device-code.o -o libxxx.so

  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -ftemplate-backtrace-limit=0)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fno-sycl-unnamed-lambda)
  # ATen/torch headers included in device-compiled FMHA runners require C++20.
  # -sycl-std sets the SYCL spec version, not the C++ language standard, so the
  # C++ standard must be set explicitly to match SYCL_HOST_FLAGS (-std=c++20).
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -std=c++20)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -sycl-std=2020)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fhonor-nans)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fhonor-infinities)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fno-associative-math)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fno-approx-func)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -Wno-absolute-value)
  # Mirrors the host-pass suppression above. nd_item::barrier(fence_space) is
  # __SYCL2020_DEPRECATED in favour of sycl::group_barrier(), but group_barrier
  # also fences CrossWorkgroupMemory, so the swap is a perf change and is
  # deferred until it can be benchmarked.
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -Wno-deprecated-declarations)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -no-ftz)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fno-sycl-instrument-device-code)
  # The device pass only carries clang's default-on diagnostics, not -Wall/-Wextra;
  # the gate keeps those (e.g. -Wmacro-redefined) from regressing.
  if(SGL_WERROR)
    set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -Werror)
  endif()

  if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -g -O0 -Rno-debug-disables-optimization)
  elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -g -fdebug-info-for-profiling -O2)
  endif()

  if("${__INTEL_LLVM_COMPILER}" MATCHES "^[0-9]+$")
    set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -DSGL_INTEL_LLVM_COMPILER_VERSION=${__INTEL_LLVM_COMPILER})
  endif()

  CHECK_SYCL_FLAG("-fsycl-fp64-conv-emu" SUPPORTS_FP64_CONV_EMU)
  if(SUPPORTS_FP64_CONV_EMU)
    set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fsycl-fp64-conv-emu)
  else()
    message(WARNING "The compiler does not support the '-fsycl-fp64-conv-emu' flag, \
    will disable it. On some platforms that don't support FP64, \
    running operations with the FP64 datatype will raise a Runtime error: Required aspect fp64 is not supported on the device \
    or a Native API failed error.")
  endif()

  set(TORCH_XPU_OPS_FLAGS ${SYCL_HOST_FLAGS})

  # -- Link hardening, applied per target in src/BuildOnLinux.cmake.
  # -z,now is required on top of -z,relro for *full* RELRO; -z,relro alone leaves
  # the lazily-bound GOT writable. No -pie: every target here is a shared library.
  set(SGL_SDL_LINK_FLAGS
    -Wl,-z,noexecstack
    -Wl,-z,relro
    -Wl,-z,now)

  # -- SYCL device object linkage flags
  include(ProcessorCount)
  ProcessorCount(proc_cnt)
  if((DEFINED ENV{MAX_JOBS}) AND ("$ENV{MAX_JOBS}" LESS_EQUAL ${proc_cnt}))
    set(SYCL_MAX_PARALLEL_LINK_JOBS $ENV{MAX_JOBS})
  else()
    set(SYCL_MAX_PARALLEL_LINK_JOBS ${proc_cnt})
  endif()
  set(SYCL_DEVICE_LINK_FLAGS ${SYCL_DEVICE_LINK_FLAGS} -fsycl-max-parallel-link-jobs=${SYCL_MAX_PARALLEL_LINK_JOBS})
  set(SYCL_DEVICE_LINK_FLAGS ${SYCL_DEVICE_LINK_FLAGS} --offload-compress)

  set(SYCL_OFFLINE_COMPILER_CG_OPTIONS "${SYCL_OFFLINE_COMPILER_CG_OPTIONS} -options -cl-poison-unsupported-fp64-kernels")
  set(SYCL_OFFLINE_COMPILER_CG_OPTIONS "${SYCL_OFFLINE_COMPILER_CG_OPTIONS} -options -cl-intel-enable-auto-large-GRF-mode")
  set(SYCL_OFFLINE_COMPILER_CG_OPTIONS "${SYCL_OFFLINE_COMPILER_CG_OPTIONS} -options -cl-fp32-correctly-rounded-divide-sqrt")
  set(SYCL_OFFLINE_COMPILER_CG_OPTIONS "${SYCL_OFFLINE_COMPILER_CG_OPTIONS} -options -cl-intel-greater-than-4GB-buffer-required")

  set(AOT_TARGETS)

  # Resolve DPCPP_SYCL_TARGET: user-provided takes priority, otherwise auto-detect.
  # get_device_ip_version returns a device-name string ('bmg' | 'cri') — the same
  # vocabulary consumed by every MATCHES site downstream.
  if(DPCPP_SYCL_TARGET)
    message(STATUS "Using user-provided DPCPP_SYCL_TARGET: ${DPCPP_SYCL_TARGET}")
  else()
    get_device_ip_version(DETECTED_TARGET)
    message(STATUS "Detected device target: ${DETECTED_TARGET}")
    if(DETECTED_TARGET MATCHES "^(bmg|cri)$")
      set(DPCPP_SYCL_TARGET "${DETECTED_TARGET}")
    else()
      message(WARNING
        "Unrecognized detected target '${DETECTED_TARGET}'. "
        "Cannot auto-detect DPCPP_SYCL_TARGET; must be one of: bmg, cri.")
    endif()
  endif()

  message(STATUS "DPCPP_SYCL_TARGET set to: ${DPCPP_SYCL_TARGET}")

  # Map DPCPP_SYCL_TARGET to AOT_TARGETS and compile definitions
  if(DPCPP_SYCL_TARGET MATCHES "bmg")
    list(APPEND AOT_TARGETS "bmg")
    set(SYCL_TARGETS_OPTION -fsycl-targets=spir64_gen)
  elseif(DPCPP_SYCL_TARGET MATCHES "cri")
    list(APPEND AOT_TARGETS "cri")
    set(SYCL_TARGETS_OPTION -fsycl-targets=spir64_gen)
    add_compile_definitions(SGL_PRE_SILICON)
  else()
    message(WARNING "Unknown DPCPP_SYCL_TARGET: ${DPCPP_SYCL_TARGET}. No AOT target set.")
  endif()

  list(REMOVE_DUPLICATES AOT_TARGETS)
  string(JOIN "," AOT_TARGETS_STR ${AOT_TARGETS})
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} ${SYCL_TARGETS_OPTION})
  set(SYCL_DEVICE_LINK_FLAGS ${SYCL_DEVICE_LINK_FLAGS} ${SYCL_TARGETS_OPTION})
  set(SYCL_OFFLINE_COMPILER_AOT_OPTIONS "-device ${AOT_TARGETS_STR}")
  message(STATUS "Compile Intel GPU AOT Targets for ${AOT_TARGETS}")
  # SYCL compiler in basekit after 2025.2 needs more spirv arguments. Single
  # source of truth for the spirv-translator extensions; reused by the AOT
  # device link here and by the runtime-JIT compile flags in BuildOnLinux.cmake.
  set(SGL_SYCL_SPIRV_EXTS +SPV_INTEL_split_barrier)
  if(SYCL_COMPILER_VERSION GREATER_EQUAL 20250806)
    list(APPEND SGL_SYCL_SPIRV_EXTS +SPV_INTEL_2d_block_io +SPV_INTEL_subgroup_matrix_multiply_accumulate)
  endif()
  string(JOIN "," SGL_SYCL_SPIRV_EXT_LIST ${SGL_SYCL_SPIRV_EXTS})
  set(SGL_SYCL_SPIRV_EXT_FLAGS -Xspirv-translator -spirv-ext=${SGL_SYCL_SPIRV_EXT_LIST})
  set(SYCL_DEVICE_LINK_FLAGS ${SYCL_DEVICE_LINK_FLAGS} ${SGL_SYCL_SPIRV_EXT_FLAGS})

  set(SYCL_COMPILE_FLAGS ${SYCL_COMPILE_FLAGS} ${SYCL_KERNEL_OPTIONS})

  set(SYCL_OFFLINE_COMPILER_FLAGS "${SYCL_OFFLINE_COMPILER_AOT_OPTIONS}${SYCL_OFFLINE_COMPILER_CG_OPTIONS}")
else()
  message("Not compiling with XPU. Currently only support GCC compiler on Linux as CXX compiler.")
  return()
endif()
