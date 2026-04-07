#include <hip/hip_runtime.h>

// Allow consumers to wrap exports in a unique namespace to prevent symbol
// collisions when multiple QoLA-built libraries coexist in one process.
//   -DQOLA_NAMESPACE=te  ->  namespace qola { namespace te { ... } }
//   (unset)              ->  namespace qola { ... }
#ifdef QOLA_NAMESPACE
#define QOLA_NS_BEGIN namespace qola { namespace QOLA_NAMESPACE {
#define QOLA_NS_END   } }
#define QOLA_NS(sym)  qola::QOLA_NAMESPACE::sym
#else
#define QOLA_NS_BEGIN namespace qola {
#define QOLA_NS_END   }
#define QOLA_NS(sym)  qola::sym
#endif
