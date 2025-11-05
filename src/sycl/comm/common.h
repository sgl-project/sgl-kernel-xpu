#pragma once

namespace {

// dispatch bool
#define AT_DISPATCH_BOOL(BOOL_V, BOOL_NAME, ...) \
  [&] {                                          \
    if (BOOL_V) {                                \
      constexpr bool BOOL_NAME = true;           \
      return __VA_ARGS__();                      \
    } else {                                     \
      constexpr bool BOOL_NAME = false;          \
      return __VA_ARGS__();                      \
    }                                            \
  }()

// dispatch bool
#define AT_DISPATCH_BOOL_NO_RETURN(BOOL_V, BOOL_NAME, ...) \
  if (BOOL_V) {                                            \
    constexpr bool BOOL_NAME = true;                       \
    __VA_ARGS__;                                           \
  } else {                                                 \
    constexpr bool BOOL_NAME = false;                      \
    __VA_ARGS__;                                           \
  }

}  // namespace
