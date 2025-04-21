# =============================================================================
# FindSVE.cmake
# This module finds the SVE (Scalable Vector Extension) support in the compiler.
# =============================================================================

INCLUDE(CheckCXXSourceRuns)

SET(SVE_CODE
    "
  #include <arm_sve.h>

  int main()
  {
    svfloat32_t a = svdup_f32(0);
    return 0;
  }
"
)

# Check for SVE support
message("Checking for SVE support")
SET(CMAKE_REQUIRED_FLAGS "-march=armv9-a+sve")
CHECK_CXX_SOURCE_RUNS("${SVE_CODE}" CXX_HAS_SVE)

IF(CXX_HAS_SVE)
  SET(CXX_SVE_FOUND TRUE CACHE BOOL "SVE support found")
  SET(CXX_SVE_FLAGS "-march=armv9-a+sve" CACHE STRING "Flags for SVE support")
ELSE()
  SET(CXX_SVE_FOUND FALSE CACHE BOOL "SVE support not found")
  SET(CXX_SVE_FLAGS "" CACHE STRING "Flags for SVE support")
ENDIF()

MARK_AS_ADVANCED(CXX_SVE_FOUND CXX_SVE_FLAGS)