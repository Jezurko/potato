#pragma once

#define POTATO_COMMON_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic ignored \"-Wsign-conversion\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wconversion\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wold-style-cast\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wunused-parameter\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wcast-align\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Woverloaded-virtual\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wctad-maybe-unsupported\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wdouble-promotion\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wshadow\"") \
  _Pragma( "GCC diagnostic ignored \"-Wunused-function\"")

#define POTATO_CLANG_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic ignored \"-Wambiguous-reversed-operator\"" )

#define POTATO_GCC_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic ignored \"-Wuseless-cast\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wnull-dereference\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wmaybe-uninitialized\"" )

#ifdef __clang__
#define POTATO_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic push" ) \
  POTATO_COMMON_RELAX_WARNINGS \
  POTATO_CLANG_RELAX_WARNINGS
#elif __GNUC__
#define POTATO_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic push" ) \
  POTATO_COMMON_RELAX_WARNINGS \
  POTATO_GCC_RELAX_WARNINGS
#else
#define POTATO_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic push" ) \
  POTATO_COMMON_RELAX_WARNINGS
#endif

#define POTATO_UNRELAX_WARNINGS \
  _Pragma( "GCC diagnostic pop" )

