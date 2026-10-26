// tests/ds4_attn_split_test.cu - the split-KV decode attention (DS4_ATTN_SPLIT=1, ds4_attn_split.cuh) against a
// double-precision host reference on decode-shaped inputs (AttnRows gen 0, T = 1..16, packed kKvFp4 rows), next to
// the stock dispatch on the same inputs.  The env switch is read once a process, so run it twice:
//   DS4_ATTN_SPLIT=0 ./ds4_attn_split_test --dump stock.bin
//   DS4_ATTN_SPLIT=1 ./ds4_attn_split_test --dump split.bin --cmp stock.bin
// Each run asserts its own outputs against the double reference; --cmp also reports split-vs-stock.
//   DS4_ATTN_SPLIT=0/1 ./ds4_attn_split_test --time     (times whichever path the env selects)
//   DS4_ATTN_SPLIT=1 ./ds4_attn_split_test --rope   (DS4_RW_SPLIT_ROPE: sparse_attn with AttnRope against finish(q),
//     sparse_attn, finish(o) on the same inputs, bit for bit, T = 1..6 at several positions; then both timed.  With
//     DS4_ATTN_SPLIT=0 it checks the stock fallback the same way)
// Build: cmake --build build --target ds4_attn_split_test (target next to ds4_gguf_test in CMakeLists.txt)
#include "../src/ds4/ds4_kernels.cu"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::ds4;

#define CK(x)                                                                                                   \
    do {                                                                                                        \
        cudaError_t e_ = (x);                                                                                   \
        if (e_ != cudaSuccess) {                                                                                \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));                    \
            std::exit(2);                                                                                       \
        }                                                                                                       \
    } while (0)

static uint16_t h_f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
static float h_bf2f(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// the compressed rows decoded to F32 by the engine's own unpack32 (the exact BF16 values both kernels see)
__global__ void decode_rows_kernel(const uint8_t* ckv, int n, float* out) {
    const int r = blockIdx.x, part = threadIdx.x;   // 16 threads: 32 values each
    if (part >= 16) return;
    __shared__ __align__(16) uint16_t row[512];
    unpack32(ckv + (size_t) r * kv_row_bytes(kKvFp4, 512), part, (uint4*) (row + part * 32));
    __syncthreads();
    for (int e = part; e < 512; e += 16) out[(size_t) r * 512 + e] = bf16f(row[e]);
}

struct Cfg {
    const char* name;
    int T;
    int nwin[16], nsel[16], win0[16];
};

int main(int argc, char** argv) {
    const int heads = 64, d = 512, kcap = 512, n_ckv = 4096, n_winrows = 1024;
    const float scale = 1.0f / std::sqrt((float) d);
    const char* dump = nullptr;
    const char* cmp = nullptr;
    bool time_only = false, rope_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) dump = argv[++i];
        else if (std::strcmp(argv[i], "--cmp") == 0 && i + 1 < argc) cmp = argv[++i];
        else if (std::strcmp(argv[i], "--time") == 0) time_only = true;
        else if (std::strcmp(argv[i], "--rope") == 0) rope_mode = true;
    }
    cudaDeviceProp p{};
    CK(cudaGetDeviceProperties(&p, 0));
    const char* se = std::getenv("DS4_ATTN_SPLIT");
    std::printf("device %s sm_%d%d  DS4_ATTN_SPLIT=%s\n", p.name, p.major, p.minor, se != nullptr ? se : "(unset)");
    std::mt19937 rng(20261025);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> byte(0, 255), scode(8 * 1, 8 * 6 + 7), rowpick(0, n_ckv - 1);

    const size_t crow = kv_row_bytes(kKvFp4, d);
    std::vector<uint8_t> h_ckv((size_t) n_ckv * crow, 0);
    for (int r = 0; r < n_ckv; ++r) {
        uint8_t* row = h_ckv.data() + (size_t) r * crow;
        for (int i = 0; i < 256; ++i) row[i] = (uint8_t) byte(rng);
        for (int i = 0; i < 32; ++i) row[256 + i] = (uint8_t) scode(rng);
    }
    uint8_t* d_ckv;
    CK(cudaMalloc(&d_ckv, h_ckv.size()));
    CK(cudaMemcpy(d_ckv, h_ckv.data(), h_ckv.size(), cudaMemcpyHostToDevice));
    float* d_dec;
    CK(cudaMalloc(&d_dec, (size_t) n_ckv * d * sizeof(float)));
    decode_rows_kernel<<<n_ckv, 16>>>(d_ckv, n_ckv, d_dec);
    CK(cudaDeviceSynchronize());
    std::vector<float> h_dec((size_t) n_ckv * d);
    CK(cudaMemcpy(h_dec.data(), d_dec, h_dec.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double rms = 0;
    for (float v : h_dec) rms += (double) v * v;
    rms = std::sqrt(rms / h_dec.size());
    if (!(rms > 1.0)) {
        std::fprintf(stderr, "the compressed rows decode to ~0: the test would not exercise them\n");
        return 2;
    }
    const float wmag = (float) rms, qmag = 3.0f / (scale * (float) rms * std::sqrt((float) d));

    std::vector<Cfg> cfgs = {
        {"T=1 deep (128+512)", 1, {128}, {512}, {0}},
        {"T=1 first token (1+0)", 1, {1}, {0}, {0}},
        {"T=1 ragged (37+41)", 1, {37}, {41}, {13}},
        {"T=2 verify (128+512, 129 window rows)", 2, {128, 128}, {512, 511}, {0, 1}},
        {"T=5 verify mixed", 5, {128, 64, 3, 128, 17}, {512, 300, 0, 40, 129}, {0, 5, 9, 2, 40}},
        {"T=5 all-window", 5, {128, 128, 128, 128, 128}, {0, 0, 0, 0, 0}, {0, 1, 2, 3, 4}},
        {"T=6 verify deep (5 drafts)", 6, {128, 128, 128, 128, 128, 128}, {512, 512, 512, 512, 512, 512}, {0, 1, 2, 3, 4, 5}},
        {"T=7 verify mixed", 7, {128, 64, 3, 128, 17, 90, 128}, {512, 300, 0, 40, 129, 7, 511}, {0, 5, 9, 2, 40, 3, 1}},
        {"T=16 max window deep", 16,
         {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
         {512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512},
         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
    };
    if (time_only) cfgs = {{"T=1 deep (timing)", 1, {128}, {512}, {0}},
                           {"T=5 deep (timing)", 5, {128, 128, 128, 128, 128}, {512, 512, 512, 512, 512}, {0, 1, 2, 3, 4}},
                           {"T=6 deep (timing)", 6, {128, 128, 128, 128, 128, 128}, {512, 512, 512, 512, 512, 512}, {0, 1, 2, 3, 4, 5}},
                           {"T=16 deep (timing)", 16,
                            {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
                            {512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512},
                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}};

    if (rope_mode) {
        std::vector<Cfg> rc = {
            {"T=1 deep", 1, {128}, {512}, {0}},
            {"T=1 first token", 1, {1}, {0}, {0}},
            {"T=2 verify", 2, {128, 128}, {512, 511}, {0, 1}},
            {"T=3 verify", 3, {128, 128, 128}, {512, 300, 40}, {0, 1, 2}},
            {"T=4 verify", 4, {128, 64, 3, 128}, {512, 300, 0, 40}, {0, 5, 9, 2}},
            {"T=5 verify mixed", 5, {128, 64, 3, 128, 17}, {512, 300, 0, 40, 129}, {0, 5, 9, 2, 40}},
            {"T=6 verify deep", 6, {128, 128, 128, 128, 128, 128}, {512, 512, 512, 512, 512, 512}, {0, 1, 2, 3, 4, 5}},
        };
        std::vector<float> h_fr(32);
        for (int i = 0; i < 32; ++i) h_fr[i] = (float) std::pow(10000.0, -2.0 * i / 64.0);
        float* d_fr;
        CK(cudaMalloc(&d_fr, 32 * 4));
        CK(cudaMemcpy(d_fr, h_fr.data(), 32 * 4, cudaMemcpyHostToDevice));
        const int poss[] = {0, 127, 8189, 40000, 65530};
        bool rok = true;
        for (const Cfg& c : rc) {
            const int T = c.T;
            AttnRows ar{};
            for (int t = 0; t < T; ++t) {
                ar.win0[t] = c.win0[t];
                ar.nwin[t] = c.nwin[t];
                ar.nsel[t] = c.nsel[t];
            }
            const size_t nq = (size_t) T * heads * d;
            std::vector<uint16_t> h_win((size_t) n_winrows * d);
            for (auto& v : h_win) v = h_f2bf(nd(rng) * wmag);
            std::vector<float> h_q(nq), h_sink(heads);
            for (auto& v : h_q) v = nd(rng) * qmag;
            for (auto& v : h_sink) v = nd(rng) * 2.0f;
            std::vector<int> h_sel((size_t) T * kcap);
            for (auto& v : h_sel) v = rowpick(rng);
            uint16_t* d_win;
            float *d_q0, *d_q, *d_sink, *d_o1, *d_o2;
            int* d_sel;
            CK(cudaMalloc(&d_win, h_win.size() * 2));
            CK(cudaMalloc(&d_q0, nq * 4));
            CK(cudaMalloc(&d_q, nq * 4));
            CK(cudaMalloc(&d_sink, heads * 4));
            CK(cudaMalloc(&d_sel, h_sel.size() * 4));
            CK(cudaMalloc(&d_o1, nq * 4));
            CK(cudaMalloc(&d_o2, nq * 4));
            CK(cudaMemcpy(d_win, h_win.data(), h_win.size() * 2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_q0, h_q.data(), nq * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_sink, h_sink.data(), heads * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_sel, h_sel.data(), h_sel.size() * 4, cudaMemcpyHostToDevice));
            for (int pos : poss) {
                AttnRope rope;
                rope.freqs = d_fr;
                rope.pos = pos;
                rope.rd = 64;
                Finish fq;   // the engine's unfused chain (ds4_model.cu decode layer)
                fq.rd = 64;
                fq.pos = pos;
                fq.vpp = heads;
                fq.freqs = d_fr;
                Finish fo = fq;
                fo.inverse = 1;
                fo.quant = 1;
                auto unfused = [&] {
                    CK(cudaMemcpyAsync(d_q, d_q0, nq * 4, cudaMemcpyDeviceToDevice, 0));
                    finish(d_q, T * heads, d, d, fq, 0);
                    sparse_attn(d_q, d_win, d_ckv, d_sel, kcap, ar, d_sink, heads, d, T, scale, d_o1, 0, kKvFp4, 0, nullptr);
                    finish(d_o1, T * heads, d, d, fo, 0);
                };
                auto fused = [&] {
                    CK(cudaMemcpyAsync(d_q, d_q0, nq * 4, cudaMemcpyDeviceToDevice, 0));
                    sparse_attn(d_q, d_win, d_ckv, d_sel, kcap, ar, d_sink, heads, d, T, scale, d_o2, 0, kKvFp4, 0, nullptr, &rope);
                };
                CK(cudaMemset(d_o1, 0xff, nq * 4));
                CK(cudaMemset(d_o2, 0xee, nq * 4));
                unfused();
                fused();
                CK(cudaGetLastError());
                CK(cudaDeviceSynchronize());
                std::vector<uint32_t> o1(nq), o2(nq);
                CK(cudaMemcpy(o1.data(), d_o1, nq * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(o2.data(), d_o2, nq * 4, cudaMemcpyDeviceToHost));
                size_t ndiff = 0, nrope = 0, shown = 0;
                for (size_t i = 0; i < nq; ++i)
                    if (o1[i] != o2[i]) {
                        ++ndiff;
                        nrope += (i % d) >= 448;
                        if (shown++ < 4) {
                            float a, b;
                            std::memcpy(&a, &o1[i], 4);
                            std::memcpy(&b, &o2[i], 4);
                            std::printf("    diff t=%zu h=%zu i=%zu unfused %.9g fused %.9g\n", i / d / heads, i / d % heads, i % d, a, b);
                        }
                    }
                rok = rok && ndiff == 0;
                std::printf("rope %-18s pos %5d: values %zu, differing %zu (%zu in the rope dims)  %s\n", c.name, pos, nq, ndiff,
                            nrope, ndiff == 0 ? "bitwise" : "FAIL");
                if (pos == 8189) {   // timing (the q copy in both)
                    cudaEvent_t e0, e1;
                    CK(cudaEventCreate(&e0));
                    CK(cudaEventCreate(&e1));
                    float ms[2];
                    for (int k = 0; k < 2; ++k) {
                        for (int i = 0; i < 5; ++i) k ? fused() : unfused();
                        CK(cudaEventRecord(e0));
                        for (int i = 0; i < 200; ++i) k ? fused() : unfused();
                        CK(cudaEventRecord(e1));
                        CK(cudaEventSynchronize(e1));
                        CK(cudaEventElapsedTime(&ms[k], e0, e1));
                    }
                    std::printf("     time: unfused %.1f us, fused %.1f us a call\n", ms[0] / 200 * 1000, ms[1] / 200 * 1000);
                    CK(cudaEventDestroy(e0));
                    CK(cudaEventDestroy(e1));
                }
            }
            CK(cudaFree(d_win));
            CK(cudaFree(d_q0));
            CK(cudaFree(d_q));
            CK(cudaFree(d_sink));
            CK(cudaFree(d_sel));
            CK(cudaFree(d_o1));
            CK(cudaFree(d_o2));
        }
        std::printf("%s\n", rok ? "ROPE ALL BITWISE" : "ROPE FAILED");
        return rok ? 0 : 1;
    }

    std::vector<float> all_out;
    bool ok = true;
    for (const Cfg& c : cfgs) {
        const int T = c.T;
        AttnRows ar{};
        ar.gen = 0;
        for (int t = 0; t < T; ++t) {
            ar.win0[t] = c.win0[t];
            ar.nwin[t] = c.nwin[t];
            ar.nsel[t] = c.nsel[t];
        }
        std::vector<uint16_t> h_win((size_t) n_winrows * d);
        for (auto& v : h_win) v = h_f2bf(nd(rng) * wmag);
        std::vector<float> h_q((size_t) T * heads * d), h_sink(heads);
        for (auto& v : h_q) v = nd(rng) * qmag;
        for (auto& v : h_sink) v = nd(rng) * 2.0f;
        std::vector<int> h_sel((size_t) T * kcap, 0);
        for (auto& v : h_sel) v = rowpick(rng);
        uint16_t* d_win;
        float *d_q, *d_sink, *d_out;
        int* d_sel;
        CK(cudaMalloc(&d_win, h_win.size() * 2));
        CK(cudaMalloc(&d_q, h_q.size() * 4));
        CK(cudaMalloc(&d_sink, heads * 4));
        CK(cudaMalloc(&d_sel, h_sel.size() * 4));
        CK(cudaMalloc(&d_out, h_q.size() * 4));
        CK(cudaMemcpy(d_win, h_win.data(), h_win.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_q, h_q.data(), h_q.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_sink, h_sink.data(), heads * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_sel, h_sel.data(), h_sel.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_out, 0xff, h_q.size() * 4));
        sparse_attn(d_q, d_win, d_ckv, d_sel, kcap, ar, d_sink, heads, d, T, scale, d_out, 0, kKvFp4, 0, nullptr);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        if (time_only) {
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0));
            CK(cudaEventCreate(&e1));
            for (int i = 0; i < 5; ++i) sparse_attn(d_q, d_win, d_ckv, d_sel, kcap, ar, d_sink, heads, d, T, scale, d_out, 0, kKvFp4, 0, nullptr);
            CK(cudaEventRecord(e0));
            const int iters = 100;
            for (int i = 0; i < iters; ++i) sparse_attn(d_q, d_win, d_ckv, d_sel, kcap, ar, d_sink, heads, d, T, scale, d_out, 0, kKvFp4, 0, nullptr);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            std::printf("%s: %.1f us a call (%d queries, %d rows)\n", c.name, ms / iters * 1000, T, c.nwin[0] + c.nsel[0]);
            CK(cudaEventDestroy(e0));
            CK(cudaEventDestroy(e1));
            CK(cudaFree(d_win));
            CK(cudaFree(d_q));
            CK(cudaFree(d_sink));
            CK(cudaFree(d_sel));
            CK(cudaFree(d_out));
            continue;
        }
        std::vector<float> h_out(h_q.size());
        CK(cudaMemcpy(h_out.data(), d_out, h_out.size() * 4, cudaMemcpyDeviceToHost));
        all_out.insert(all_out.end(), h_out.begin(), h_out.end());
        // the host reference in double
        std::vector<double> ref(h_q.size()), sc;
        for (int t = 0; t < T; ++t) {
            const int total = c.nwin[t] + c.nsel[t];
            for (int h = 0; h < heads; ++h) {
                const float* qh = h_q.data() + ((size_t) t * heads + h) * d;
                sc.assign(total, 0.0);
                double m = (double) h_sink[h];
                for (int j = 0; j < total; ++j) {
                    double s = 0;
                    if (j < c.nwin[t]) {
                        const uint16_t* row = h_win.data() + (size_t) (c.win0[t] + j) * d;
                        for (int e = 0; e < d; ++e) s += (double) qh[e] * h_bf2f(row[e]);
                    } else {
                        const float* row = h_dec.data() + (size_t) h_sel[(size_t) t * kcap + (j - c.nwin[t])] * d;
                        for (int e = 0; e < d; ++e) s += (double) qh[e] * row[e];
                    }
                    sc[j] = s * scale;
                    m = std::max(m, sc[j]);
                }
                double den = std::exp((double) h_sink[h] - m);
                double* o = ref.data() + ((size_t) t * heads + h) * d;
                for (int e = 0; e < d; ++e) o[e] = 0;
                for (int j = 0; j < total; ++j) {
                    const double pj = std::exp(sc[j] - m);
                    den += pj;
                    if (j < c.nwin[t]) {
                        const uint16_t* row = h_win.data() + (size_t) (c.win0[t] + j) * d;
                        for (int e = 0; e < d; ++e) o[e] += pj * h_bf2f(row[e]);
                    } else {
                        const float* row = h_dec.data() + (size_t) h_sel[(size_t) t * kcap + (j - c.nwin[t])] * d;
                        for (int e = 0; e < d; ++e) o[e] += pj * row[e];
                    }
                }
                for (int e = 0; e < d; ++e) o[e] /= den;
            }
        }
        double omax = 0, emax = 0;
        bool nan = false;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (!std::isfinite(h_out[i])) nan = true;
            omax = std::max(omax, std::fabs(ref[i]));
            emax = std::max(emax, std::fabs(h_out[i] - ref[i]));
        }
        const double rel = emax / omax;
        const bool pass = !nan && rel <= 1e-4;   // (observed ~3e-6; 1e-4 still catches a 10x regression)
        ok = ok && pass;
        std::printf("%-38s |ref|max %.3g  max_abs %.3g  rel %.2e  %s%s\n", c.name, omax, emax, rel, pass ? "ok" : "FAIL", nan ? " (NaN)" : "");
        CK(cudaFree(d_win));
        CK(cudaFree(d_q));
        CK(cudaFree(d_sink));
        CK(cudaFree(d_sel));
        CK(cudaFree(d_out));
    }
    if (time_only) return 0;
    if (dump != nullptr) {
        FILE* f = std::fopen(dump, "wb");
        if (f == nullptr || std::fwrite(all_out.data(), 4, all_out.size(), f) != all_out.size()) {
            std::fprintf(stderr, "cannot write %s\n", dump);
            return 2;
        }
        std::fclose(f);
        std::printf("dumped %zu floats to %s\n", all_out.size(), dump);
    }
    if (cmp != nullptr) {
        FILE* f = std::fopen(cmp, "rb");
        if (f == nullptr) {
            std::fprintf(stderr, "cannot read %s\n", cmp);
            return 2;
        }
        std::vector<float> other(all_out.size());
        const size_t got = std::fread(other.data(), 4, other.size(), f);
        std::fclose(f);
        if (got != other.size()) {
            std::fprintf(stderr, "%s holds %zu floats, this run %zu\n", cmp, got, other.size());
            return 2;
        }
        double emax = 0, omax = 0, rmax = 0;
        size_t ndiff = 0;
        for (size_t i = 0; i < other.size(); ++i) {
            const double e = std::fabs((double) all_out[i] - other[i]);
            if (e > 0) ++ndiff;
            emax = std::max(emax, e);
            omax = std::max(omax, std::fabs((double) other[i]));
            rmax = std::max(rmax, e / std::max(std::fabs((double) other[i]), 1e-12));
        }
        std::printf("vs %s: values %zu, differing %zu, max_abs %.9g, max_abs_over_max %.3e, max_rel %.3e\n", cmp,
                    other.size(), ndiff, emax, emax / omax, rmax);
    }
    std::printf("%s\n", ok ? "ALL OK" : "FAILED");
    return ok ? 0 : 1;
}
