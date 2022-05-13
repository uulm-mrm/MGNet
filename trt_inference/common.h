#include "NvInfer.h"

#include <memory>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                 \
  {                                                                         \
    cudaError_t error_code = callstr;                                       \
    if (error_code != cudaSuccess) {                                        \
      std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" \
                << __LINE__;                                                \
    }                                                                       \
  }
#endif // CUDA_CHECK

#undef CHECK_FOR_NULLPTR
#define CHECK_FOR_NULLPTR(ptr, message)  \
  do {                                   \
    if (!(ptr)) {                        \
      std::cerr << message << std::endl; \
      abort();                           \
    }                                    \
  } while (0)

// destroy TensorRT objects if something goes wrong
struct TRTDestroy {
  template <typename T>
  void operator()(T* obj) const {
    delete obj;
  }
};

template <class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) throw() override {
    // remove this 'if' if you need more logged info
    if (severity <= Severity::kWARNING) {
      std::cout << msg << std::endl;
    }
  }
} gLogger;

static auto StreamDeleter = [](cudaStream_t* pStream) {
  if (pStream) {
    cudaStreamDestroy(*pStream);
    delete pStream;
  }
};

inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream() {
  std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(
      new cudaStream_t, StreamDeleter);
  if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) !=
      cudaSuccess) {
    pStream.reset(nullptr);
  }

  return pStream;
}