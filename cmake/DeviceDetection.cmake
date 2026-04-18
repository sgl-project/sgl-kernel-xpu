# Device IP version detection via Level Zero API.
#
# Provides:
#   get_device_ip_version(<variable>)
#     Compiles and runs a small Level Zero program that queries the GPU
#     device IP version, extracts the architecture bits, and stores the
#     result in <variable>.  Falls back to the BUILD_TARGET_DEVICE
#     environment variable when compilation or execution fails.

function(get_device_ip_version VARIABLE_NAME)
  # Define a C++ source file to be compiled and run
  set(TEMP_DIR "${CMAKE_BINARY_DIR}/temp")
  file(MAKE_DIRECTORY ${TEMP_DIR})
  set(TEST_SRC_FILE "${TEMP_DIR}/get_device_ip_version.cpp")
  set(TEST_EXE_FILE "${TEMP_DIR}/get_device_ip_version.out")

  file(WRITE "${TEST_SRC_FILE}"
    "
      #include <level_zero/ze_api.h>
      #include <iostream>
      #include <limits>
      #include <cassert>

      int main() {

        zeInit(0);
        ze_driver_handle_t driver;
        uint32_t driverCount = 1;
        zeDriverGet(&driverCount, &driver);

        ze_device_handle_t device;
        uint32_t deviceCount = 1;
        zeDeviceGet(driver, &deviceCount, &device);

        ze_device_properties_t props = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};

        ze_device_ip_version_ext_t zeDeviceIpVersion = {ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT};

        zeDeviceIpVersion.ipVersion = std::numeric_limits<uint32_t>::max();
        props.pNext = &zeDeviceIpVersion;
        zeDeviceGetProperties(device, &props);

        uint32_t gpu_arch = (zeDeviceIpVersion.ipVersion >> 22) & 0x3FF;

        std::cout << gpu_arch;

        return 0;
      }
    "
  )

  # Compile the C++ program first using try_compile
  try_compile(
    COMPILE_RESULT_VAR        # Variable to store the compile result (true for success)
    "${TEMP_DIR}"             # Binary directory for the compiled output
    "${TEST_SRC_FILE}"        # Source file to compile
    LINK_LIBRARIES ze_loader  # pass required libraries to create executable
    COPY_FILE "${TEST_EXE_FILE}" # Copy the compiled executable to our desired path
  )

  # Run the compiled executable with a 60-second timeout to avoid hanging
  # indefinitely if the GPU is unavailable or the driver is unresponsive
  set(RUN_RESULT_VAR 1)
  set(OUTPUT_STDOUT "")
  set(OUTPUT_STDERR "")
  if(COMPILE_RESULT_VAR)
    execute_process(
      COMMAND "${TEST_EXE_FILE}"
      OUTPUT_VARIABLE OUTPUT_STDOUT
      ERROR_VARIABLE OUTPUT_STDERR
      RESULT_VARIABLE RUN_RESULT_VAR
      TIMEOUT 60
    )
    if(RUN_RESULT_VAR STREQUAL "Process terminated due to timeout")
      message(FATAL_ERROR
        "get_device_ip_version: Timed out while querying the GPU device IP version. "
        "Ensure a Level Zero compatible GPU or simulator is available and the driver is functioning correctly, "
        "or set the DPCPP_SYCL_TARGET variable manually to skip auto-detection.")
    endif()
  endif()

  file(REMOVE_RECURSE ${TEMP_DIR})
  # Check if compilation and execution were successful
  if(COMPILE_RESULT_VAR AND RUN_RESULT_VAR EQUAL 0)
    # Extract only the last line from stdout to ignore runtime/driver noise
    string(STRIP "${OUTPUT_STDOUT}" OUTPUT_STRIPPED)
    string(REGEX REPLACE "\n" ";" OUTPUT_LINES "${OUTPUT_STRIPPED}")
    list(GET OUTPUT_LINES -1 LAST_LINE)
    string(STRIP "${LAST_LINE}" LAST_LINE)
    set(${VARIABLE_NAME} "${LAST_LINE}" PARENT_SCOPE)
  else()
    if(NOT COMPILE_RESULT_VAR)
        message(WARNING "Compilation failed.")
    endif()
    if(NOT RUN_RESULT_VAR EQUAL 0)
        message(WARNING "Program returned non-zero exit code: ${RUN_RESULT_VAR}")
    endif()
    message(WARNING "Failed to compile or run the get_device_ip_version C++ program.")
    set(BUILD_TARGET_DEVICE $ENV{BUILD_TARGET_DEVICE})
    # no default device
    if(NOT BUILD_TARGET_DEVICE)
      message(FATAL_ERROR "BUILD_TARGET_DEVICE environment variable is not set")
    endif()
    string(TOLOWER "${BUILD_TARGET_DEVICE}" BUILD_TARGET_DEVICE)
    set(${VARIABLE_NAME} ${BUILD_TARGET_DEVICE} PARENT_SCOPE)
  endif()
endfunction()
