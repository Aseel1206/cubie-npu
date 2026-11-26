// Thin C ABI around the public VIPLite API (vip_lite.h) so that Python can drive the Allwinner NPU through ctypes.
// This file is compiled on the user's machine against THEIR copy of the VIPLite SDK; nothing proprietary is shipped
// with cubie-npu. Keep it small: all the logic (dtype handling, quantisation, shapes) lives in Python.
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <chrono>
#include <fstream>
#include <iterator>
#include <mutex>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <vip_lite.h>

extern "C" {

struct CnpuTensor {
    char name[256];
    int data_format;      // vip_buffer_format_e
    int quant_format;     // vip_buffer_quantize_format_e
    int ndims;
    unsigned sizes[6];    // VIPLite order: sizes[0] is the innermost dimension
    int fixed_point_pos;  // DYNAMIC_FIXED_POINT
    float scale;          // TF_ASYMM
    int zero_point;       // TF_ASYMM
    unsigned bytes;       // size of the buffer
};

}  // extern "C"

namespace {
struct Io { vip_buffer buf = nullptr; CnpuTensor info; };
struct Model {
    vip_network net = nullptr;
    std::vector<Io> in, out;
    bool prepared = false;
};
// The vendor library prints a "VIPLite driver software version ..." banner to stdout on init/destroy. Hide it unless
// CUBIE_NPU_VERBOSE is set, so scripts that print results are not polluted.
struct QuietStdout {
    int saved = -1;
    QuietStdout() {
        if (getenv("CUBIE_NPU_VERBOSE")) return;
        fflush(stdout); saved = dup(1); int n = open("/dev/null", O_WRONLY); if (n >= 0) { dup2(n, 1); close(n); }
    }
    ~QuietStdout() { if (saved >= 0) { fflush(stdout); dup2(saved, 1); close(saved); } }
};
std::mutex g_mu;
int g_refs = 0;
char g_err[512] = "";
void set_err(const char* m, int code = 0) { snprintf(g_err, sizeof(g_err), code ? "%s (VIPLite status %d)" : "%s", m, code); }

bool make_buffers(vip_network net, bool is_input, int count, std::vector<Io>& v) {
    v.resize(count);
    for (int i = 0; i < count; ++i) {
        Io& io = v[i]; memset(&io.info, 0, sizeof(io.info));
        vip_buffer_create_params_t p; memset(&p, 0, sizeof(p)); p.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
        auto q = [&](vip_enum prop, void* o) { return is_input ? vip_query_input(net, i, prop, o) : vip_query_output(net, i, prop, o); };
        q(VIP_BUFFER_PROP_DATA_FORMAT, &p.data_format); q(VIP_BUFFER_PROP_NUM_OF_DIMENSION, &p.num_of_dims);
        q(VIP_BUFFER_PROP_SIZES_OF_DIMENSION, p.sizes); q(VIP_BUFFER_PROP_QUANT_FORMAT, &p.quant_format); q(VIP_BUFFER_PROP_NAME, io.info.name);
        if (p.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) q(VIP_BUFFER_PROP_FIXED_POINT_POS, &p.quant_data.dfp.fixed_point_pos);
        if (p.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) { q(VIP_BUFFER_PROP_TF_SCALE, &p.quant_data.affine.scale); q(VIP_BUFFER_PROP_TF_ZERO_POINT, &p.quant_data.affine.zeroPoint); }
        io.info.data_format = (int)p.data_format; io.info.quant_format = (int)p.quant_format; io.info.ndims = (int)p.num_of_dims;
        for (unsigned k = 0; k < p.num_of_dims && k < 6; ++k) io.info.sizes[k] = p.sizes[k];
        io.info.fixed_point_pos = p.quant_data.dfp.fixed_point_pos; io.info.scale = p.quant_data.affine.scale; io.info.zero_point = p.quant_data.affine.zeroPoint;
        if (p.quant_format != VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) io.info.fixed_point_pos = 0;
        if (p.quant_format != VIP_BUFFER_QUANTIZE_TF_ASYMM) { io.info.scale = 1.f; io.info.zero_point = 0; }
        vip_status_e s = vip_create_buffer(&p, sizeof(p), &io.buf);
        if (s != VIP_SUCCESS) { set_err("vip_create_buffer failed", s); return false; }
        io.info.bytes = vip_get_buffer_size(io.buf);
    }
    return true;
}
}  // namespace

extern "C" {

const char* cnpu_last_error() { return g_err; }
unsigned cnpu_driver_version() { return vip_get_version(); }

void* cnpu_load(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { set_err("cannot open model file"); return nullptr; }
    std::vector<char> nb((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    { std::lock_guard<std::mutex> l(g_mu);
      if (g_refs == 0) { QuietStdout q; vip_status_e s = vip_init(); if (s != VIP_SUCCESS) { set_err("vip_init failed - is the NPU driver loaded? (/dev/vipcore)", s); return nullptr; } }
      ++g_refs; }
    Model* m = new Model();
    vip_status_e s = vip_create_network(nb.data(), (vip_uint32_t)nb.size(), VIP_CREATE_NETWORK_FROM_MEMORY, &m->net);
    if (s != VIP_SUCCESS) { set_err("vip_create_network failed - not a valid .nb for this NPU?", s); m->net = nullptr; goto fail; }
    { vip_uint32_t nin = 0, nout = 0;
      vip_query_network(m->net, VIP_NETWORK_PROP_INPUT_COUNT, &nin); vip_query_network(m->net, VIP_NETWORK_PROP_OUTPUT_COUNT, &nout);
      if (!make_buffers(m->net, true, (int)nin, m->in) || !make_buffers(m->net, false, (int)nout, m->out)) goto fail; }
    s = vip_prepare_network(m->net);
    if (s != VIP_SUCCESS) { set_err("vip_prepare_network failed", s); goto fail; }
    m->prepared = true;
    return m;
fail:
    for (auto& io : m->in) if (io.buf) vip_destroy_buffer(io.buf);
    for (auto& io : m->out) if (io.buf) vip_destroy_buffer(io.buf);
    if (m->net) vip_destroy_network(m->net);
    delete m;
    { std::lock_guard<std::mutex> l(g_mu); if (--g_refs == 0) { QuietStdout q; vip_destroy(); } }
    return nullptr;
}

int cnpu_counts(void* h, int* nin, int* nout) { Model* m = (Model*)h; *nin = (int)m->in.size(); *nout = (int)m->out.size(); return 0; }

int cnpu_tensor_info(void* h, int is_input, int idx, CnpuTensor* out) {
    Model* m = (Model*)h; auto& v = is_input ? m->in : m->out;
    if (idx < 0 || idx >= (int)v.size()) { set_err("tensor index out of range"); return -1; }
    *out = v[idx].info; return 0;
}

int cnpu_set_input(void* h, int idx, const void* data, unsigned bytes) {
    Model* m = (Model*)h;
    if (idx < 0 || idx >= (int)m->in.size()) { set_err("input index out of range"); return -1; }
    Io& io = m->in[idx];
    if (bytes != io.info.bytes) { char b[160]; snprintf(b, sizeof(b), "input %d expects %u bytes, got %u", idx, io.info.bytes, bytes); set_err(b); return -2; }
    void* p = vip_map_buffer(io.buf); if (!p) { set_err("vip_map_buffer failed"); return -3; }
    memcpy(p, data, bytes); vip_unmap_buffer(io.buf); return 0;
}

// Runs the network on whatever is currently in the input buffers. Returns 0 on success; *ms = wall time of flush+run+invalidate.
int cnpu_run(void* h, double* ms) {
    Model* m = (Model*)h; auto t0 = std::chrono::steady_clock::now();
    for (auto& io : m->in) vip_flush_buffer(io.buf, VIP_BUFFER_OPER_TYPE_FLUSH);
    for (unsigned k = 0; k < m->in.size(); ++k) vip_set_input(m->net, k, m->in[k].buf);
    for (unsigned k = 0; k < m->out.size(); ++k) vip_set_output(m->net, k, m->out[k].buf);
    vip_status_e s = vip_run_network(m->net);
    for (auto& io : m->out) vip_flush_buffer(io.buf, VIP_BUFFER_OPER_TYPE_INVALIDATE);
    if (ms) *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (s != VIP_SUCCESS) { set_err("vip_run_network failed", s); return (int)s; }
    return 0;
}

int cnpu_get_output(void* h, int idx, void* dst, unsigned bytes) {
    Model* m = (Model*)h;
    if (idx < 0 || idx >= (int)m->out.size()) { set_err("output index out of range"); return -1; }
    Io& io = m->out[idx]; if (bytes != io.info.bytes) { set_err("output buffer size mismatch"); return -2; }
    void* p = vip_map_buffer(io.buf); if (!p) { set_err("vip_map_buffer failed"); return -3; }
    memcpy(dst, p, bytes); vip_unmap_buffer(io.buf); return 0;
}

void cnpu_free(void* h) {
    Model* m = (Model*)h; if (!m) return;
    if (m->net) vip_finish_network(m->net);
    for (auto& io : m->in) if (io.buf) vip_destroy_buffer(io.buf);
    for (auto& io : m->out) if (io.buf) vip_destroy_buffer(io.buf);
    if (m->net) vip_destroy_network(m->net);
    delete m;
    std::lock_guard<std::mutex> l(g_mu); if (--g_refs == 0) { QuietStdout q; vip_destroy(); }
}

}  // extern "C"
