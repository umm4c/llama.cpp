#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    6
#define RPC_PROTO_MINOR_VERSION    1
#define RPC_PROTO_PATCH_VERSION    0

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 101, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

#define GGML_RPC_PROFILE_MAX_RECORDS 4096
typedef struct ggml_rpc_profile_record {
    uint64_t sequence;
    uint64_t start_ns;
    uint64_t prepare_ns;
    uint64_t compute_ns;
    uint32_t device;
    uint32_t n_nodes;
    uint32_t reused;
    int32_t status;
} ggml_rpc_profile_record;

// Exclusive profiling window on the backend's shared endpoint connection.
// BEGIN is asynchronous. END waits for server completion and validates submission order.
// Returns false for unsupported servers, invalid state, overflow or mismatched records.
GGML_BACKEND_API bool ggml_backend_rpc_profile_begin(ggml_backend_t backend, uint64_t window, uint32_t capacity);
GGML_BACKEND_API bool ggml_backend_rpc_profile_end(ggml_backend_t backend, uint64_t window,
        ggml_rpc_profile_record * records, uint32_t capacity, uint32_t * count);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

#ifdef  __cplusplus
}
#endif
