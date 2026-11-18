#pragma once

#include "ggml-backend.h"

// A stream interval over a half-open range in the submitted graph.
struct ggml_backend_timing_sample {
    int begin;
    int end;
    float elapsed_ms;
};

// Discover through the backend registry. Finish after normal graph completion.
// A null output aborts collection and releases resources.
typedef bool (*ggml_backend_timing_begin_t)(ggml_backend_t, const int *, size_t);
typedef size_t (*ggml_backend_timing_finish_t)(ggml_backend_t, ggml_backend_timing_sample *, size_t);
