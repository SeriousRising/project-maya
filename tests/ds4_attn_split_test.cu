// tests/ds4_attn_split_test.cu - the split-KV decode attention (DS4_ATTN_SPLIT=1, ds4_attn_split.cuh) against a
// double-precision host reference on decode-shaped inputs (AttnRows gen 0, T = 1..16, packed kKvFp4 rows), next to
// the stock dispatch on the same inputs.  The env switch is read once a process, so run it twice:
//   DS4_ATTN_SPLIT=0 ./ds4_attn_split_test --dump stock.bin
//   DS4_ATTN_SPLIT=1 ./ds4_attn_split_test --dump split.bin --cmp stock.bin
// Each run asserts its own outputs against the double reference; --cmp also reports split-vs-stock.
//   DS4_ATTN_SPLIT=0/1 ./ds4_attn_split_test --time     (times whichever path the env selects)
//   DS4_ATTN_SPLIT=1 ./ds4_attn_split_test --rope   (DS4_RW_SPLIT_ROPE: sparse_attn with AttnRope against finish(q),
//     sparse_attn, finish(o) on the same inputs, bit for bit, T = 1..6, 10, 16 at several positions; then both timed.  With
//     DS4_ATTN_SPLIT=0 it checks the stock fallback the same way)
//   DS4_ATTN_SPLIT=1 ./ds4_attn_split_test --tc   (DS4_RW_ATTN_TC: attn_part_mma (chunks of 16 / 32 rows) and the FP32 split
//     against the double reference and each other - packed rows at T = 1..16 at two score scales, the drafter's BF16
//     rows against the stock BF16 kernel - then the time a call (RoPE fold on) at T = 1, 2, 5, 6, 16, context 4K / 32K,
//     a ratio-4 layer (128 + 512 picked) and a ratio-128 layer (128 + every compressed row).  The dispatch itself:
//     DS4_ATTN_SPLIT=1 DS4_RW_ATTN_TC=1 with no flag (1e-4 vs double) and with --rope (the fold, bitwise))
//   DS4_ATTN_SPLIT=1 [DS4_RW_ATTN_TC=1] ./ds4_attn_split_test --tinv   (gate2's invariant: a query's output through the
//     dispatch, plain and with the RoPE fold, is the same bits alone (T=1) and inside every T=2..16 call that holds it)
//   ./ds4_attn_split_test --hc   (DS4_RW_HC_PRE / DS4_RW_HC_FUSE: hc_pre_norm_n_kernel against hc_pre_norm_kernel and
//     hc_post_pre_norm_kernel against hc_post_kernel + hc_pre_norm_kernel, bit for bit at dim 5120, hc 4, T = 1..16; timed)
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
    const int heads = 64, d = 512, kcap = 512, n_winrows = 1024;
    const float scale = 1.0f / std::sqrt((float) d);
    const char* dump = nullptr;
    const char* cmp = nullptr;
    bool time_only = false, rope_mode = false, hc_mode = false, tc_mode = false, tinv_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) dump = argv[++i];
        else if (std::strcmp(argv[i], "--cmp") == 0 && i + 1 < argc) cmp = argv[++i];
        else if (std::strcmp(argv[i], "--time") == 0) time_only = true;
        else if (std::strcmp(argv[i], "--rope") == 0) rope_mode = true;
        else if (std::strcmp(argv[i], "--hc") == 0) hc_mode = true;
        else if (std::strcmp(argv[i], "--tc") == 0) tc_mode = true;
        else if (std::strcmp(argv[i], "--tinv") == 0) tinv_mode = true;
    }
    const int n_ckv = tc_mode ? 8192 : 4096;   // (--tc: a 32K context's ratio-4 rows)
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
    // the double reference: query t's window rows win[win0 + j], then its selected rows srows[sel[t][i]] (F32, d each)
    auto reference = [&](const Cfg& c, const std::vector<float>& h_q, const std::vector<uint16_t>& h_win,
                         const std::vector<float>& h_sink, const std::vector<int>& h_sel, int sel_stride, const float* srows) {
        const int T = c.T;
        std::vector<double> ref((size_t) T * heads * d), sc;
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
                        const float* row = srows + (size_t) h_sel[(size_t) t * sel_stride + (j - c.nwin[t])] * d;
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
                        const float* row = srows + (size_t) h_sel[(size_t) t * sel_stride + (j - c.nwin[t])] * d;
                        for (int e = 0; e < d; ++e) o[e] += pj * row[e];
                    }
                }
                for (int e = 0; e < d; ++e) o[e] /= den;
            }
        }
        return ref;
    };

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

    if (hc_mode) {
        const int D = 5120, H = 4, TM = 16;   // (T up to kWin: v1.1.31's prompt-lookup windows)
        const float eps = 1e-6f;
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);
        std::vector<float> h_h((size_t) TM * H * D), h_y((size_t) TM * D), h_w(D), h_mix(TM * 8), h_post(TM * 8), h_comb(TM * 64);
        for (auto& v : h_h) v = nd(rng) * 0.7f;
        for (auto& v : h_y) v = nd(rng) * 0.3f;
        for (auto& v : h_w) v = 0.5f + u01(rng);
        for (auto& v : h_mix) v = u01(rng) + 1e-6f;
        for (auto& v : h_post) v = 2.0f * u01(rng);
        for (auto& v : h_comb) v = 0.5f * u01(rng);
        float *d_h, *d_y, *d_w, *d_mix, *d_post, *d_comb, *d_o1, *d_o2, *d_x1, *d_x2;
        auto up = [](float*& p, const std::vector<float>& v) {
            CK(cudaMalloc(&p, v.size() * 4));
            CK(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        };
        up(d_h, h_h), up(d_y, h_y), up(d_w, h_w), up(d_mix, h_mix), up(d_post, h_post), up(d_comb, h_comb);
        CK(cudaMalloc(&d_o1, h_h.size() * 4));
        CK(cudaMalloc(&d_o2, h_h.size() * 4));
        CK(cudaMalloc(&d_x1, h_y.size() * 4));
        CK(cudaMalloc(&d_x2, h_y.size() * 4));
        auto same = [](const float* a, const float* b, size_t n) {
            std::vector<uint32_t> x(n), y(n);
            CK(cudaMemcpy(x.data(), a, n * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(y.data(), b, n * 4, cudaMemcpyDeviceToHost));
            size_t nd_ = 0;
            for (size_t i = 0; i < n; ++i) nd_ += x[i] != y[i];
            return nd_;
        };
        bool hok = true;
        for (int T = 1; T <= TM; ++T)
            for (int wn = 0; wn < 2; ++wn) {
                const float* w = wn ? nullptr : d_w;
                CK(cudaMemset(d_x1, 0xff, h_y.size() * 4));
                CK(cudaMemset(d_x2, 0xee, h_y.size() * 4));
                hc_pre_norm_kernel<<<T, 1024>>>(d_h, d_mix, w, D, H, eps, d_x1);
                hc_pre_norm_n_kernel<5, 4><<<T, 1024>>>(d_h, d_mix, w, D, eps, d_x2);
                CK(cudaGetLastError());
                const size_t dp = same(d_x1, d_x2, (size_t) T * D);
                CK(cudaMemset(d_o1, 0xff, h_h.size() * 4));
                CK(cudaMemset(d_o2, 0xee, h_h.size() * 4));
                hc_post_kernel<<<dim3(cdiv(D * H, kThreads), T), kThreads>>>(d_y, d_h, d_post, d_comb, D, H, d_o1);
                hc_pre_norm_kernel<<<T, 1024>>>(d_o1, d_mix, w, D, H, eps, d_x1);
                hc_post_pre_norm_kernel<5, 4><<<T, 1024>>>(d_y, d_h, d_post, d_comb, d_mix, w, D, eps, d_o2, d_x2);
                CK(cudaGetLastError());
                const size_t dpo = same(d_o1, d_o2, (size_t) T * H * D), dpx = same(d_x1, d_x2, (size_t) T * D);
                hok = hok && dp == 0 && dpo == 0 && dpx == 0;
                std::printf("hc T=%d w %s: pre_norm differing %zu / %d; post+pre: h differing %zu / %d, x differing %zu  %s\n", T,
                            w ? "yes" : "null", dp, T * D, dpo, T * H * D, dpx, dp + dpo + dpx == 0 ? "bitwise" : "FAIL");
            }
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        for (int T : {1, 5}) {
            float ms[4];
            for (int k = 0; k < 4; ++k) {
                auto run = [&] {
                    if (k == 0) hc_pre_norm_kernel<<<T, 1024>>>(d_h, d_mix, d_w, D, H, eps, d_x1);
                    else if (k == 1) hc_pre_norm_n_kernel<5, 4><<<T, 1024>>>(d_h, d_mix, d_w, D, eps, d_x2);
                    else if (k == 2) {
                        hc_post_kernel<<<dim3(cdiv(D * H, kThreads), T), kThreads>>>(d_y, d_h, d_post, d_comb, D, H, d_o1);
                        hc_pre_norm_kernel<<<T, 1024>>>(d_o1, d_mix, d_w, D, H, eps, d_x1);
                    } else hc_post_pre_norm_kernel<5, 4><<<T, 1024>>>(d_y, d_h, d_post, d_comb, d_mix, d_w, D, eps, d_o2, d_x2);
                };
                for (int i = 0; i < 10; ++i) run();
                CK(cudaEventRecord(e0));
                for (int i = 0; i < 500; ++i) run();
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&ms[k], e0, e1));
            }
            std::printf("hc time T=%d: pre_norm %.2f -> %.2f us; post + pre_norm %.2f -> fused %.2f us\n", T, ms[0] * 2, ms[1] * 2,
                        ms[2] * 2, ms[3] * 2);
        }
        std::printf("%s\n", hok ? "HC ALL BITWISE" : "HC FAILED");
        return hok ? 0 : 1;
    }
    if (tc_mode) {
        // own scratch: 48 chunks (16-row chunks up to 768 rows; the engine's is kSplitCap = 24)
        attnsplit::AttnSplitScratch sc;
        sc.nsplit = 48;
        sc.max_t = 16;
        CK(cudaMalloc((void**) &sc.part, (size_t) 16 * heads * sc.nsplit * d * 4));
        CK(cudaMalloc((void**) &sc.ml, (size_t) 16 * heads * sc.nsplit * 2 * 4));
        struct Err {
            double abs = 0, rel = 0, erel = 0;   // max |a - b|, that over max |b|, max elementwise (|b| >= 1e-3 max |b|)
        };
        auto err = [](const std::vector<float>& a, const std::vector<double>& b) {
            double omax = 0;
            for (double v : b) omax = std::max(omax, std::fabs(v));
            Err e;
            for (size_t i = 0; i < a.size(); ++i) {
                const double x = std::isfinite(a[i]) ? std::fabs(a[i] - b[i]) : INFINITY;
                e.abs = std::max(e.abs, x);
                if (std::fabs(b[i]) >= 1e-3 * omax) e.erel = std::max(e.erel, x / std::fabs(b[i]));
            }
            e.rel = e.abs / omax;
            return e;
        };
        auto todouble = [](const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); };
        auto fetch = [](const float* p, size_t n) {
            std::vector<float> v(n);
            CK(cudaMemcpy(v.data(), p, n * 4, cudaMemcpyDeviceToHost));
            return v;
        };
        // the drafter's block rows (BF16, every query selects all T; as ds4_model.cu's l.mtp)
        const int n_blk = 16;
        std::vector<uint16_t> h_blk((size_t) n_blk * d);
        for (auto& v : h_blk) v = h_f2bf(nd(rng) * wmag);
        std::vector<float> h_blkf(h_blk.size());
        for (size_t i = 0; i < h_blk.size(); ++i) h_blkf[i] = h_bf2f(h_blk[i]);
        uint16_t* d_blk;
        CK(cudaMalloc(&d_blk, h_blk.size() * 2));
        CK(cudaMemcpy(d_blk, h_blk.data(), h_blk.size() * 2, cudaMemcpyHostToDevice));
        std::vector<Cfg> dc = {
            {"drafter T=2 (37+2)", 2, {37, 37}, {2, 2}, {0, 0}},
            {"drafter T=6 (128+6)", 6, {128, 128, 128, 128, 128, 128}, {6, 6, 6, 6, 6, 6}, {0, 0, 0, 0, 0, 0}},
            {"drafter T=16 (128+16)", 16,
             {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
             {16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16}, {0}},
        };
        bool tok = true;
        double worst = 0;
        for (int pass = 0; pass < 2; ++pass)
            for (const float qs : {1.0f, 3.0f}) {   // score spread ~3 and ~9 (a peaked softmax)
                const std::vector<Cfg>& cl = pass ? dc : cfgs;
                for (const Cfg& c : cl) {
                    const int T = c.T;
                    const bool bf = pass == 1;
                    AttnRows ar{};
                    int total_max = 0;
                    for (int t = 0; t < T; ++t) {
                        ar.win0[t] = c.win0[t];
                        ar.nwin[t] = c.nwin[t];
                        ar.nsel[t] = c.nsel[t];
                        total_max = std::max(total_max, c.nwin[t] + c.nsel[t]);
                    }
                    std::vector<uint16_t> h_win((size_t) n_winrows * d);
                    for (auto& v : h_win) v = h_f2bf(nd(rng) * wmag);
                    std::vector<float> h_q((size_t) T * heads * d), h_sink(heads);
                    for (auto& v : h_q) v = nd(rng) * qmag * qs;
                    for (auto& v : h_sink) v = nd(rng) * 2.0f;
                    std::vector<int> h_sel((size_t) T * kcap, 0);
                    for (int t = 0; t < T; ++t)
                        for (int i = 0; i < kcap; ++i) h_sel[(size_t) t * kcap + i] = bf ? i % n_blk : rowpick(rng);
                    uint16_t* d_win;
                    float *d_q, *d_sink, *d_out;
                    int* d_sel;
                    const size_t nq = h_q.size();
                    CK(cudaMalloc(&d_win, h_win.size() * 2));
                    CK(cudaMalloc(&d_q, nq * 4));
                    CK(cudaMalloc(&d_sink, heads * 4));
                    CK(cudaMalloc(&d_sel, h_sel.size() * 4));
                    CK(cudaMalloc(&d_out, nq * 4));
                    CK(cudaMemcpy(d_win, h_win.data(), h_win.size() * 2, cudaMemcpyHostToDevice));
                    CK(cudaMemcpy(d_q, h_q.data(), nq * 4, cudaMemcpyHostToDevice));
                    CK(cudaMemcpy(d_sink, h_sink.data(), heads * 4, cudaMemcpyHostToDevice));
                    CK(cudaMemcpy(d_sel, h_sel.data(), h_sel.size() * 4, cudaMemcpyHostToDevice));
                    const std::vector<double> ref = reference(c, h_q, h_win, h_sink, h_sel, kcap, bf ? h_blkf.data() : h_dec.data());
                    const uint8_t* ck = bf ? (const uint8_t*) d_blk : d_ckv;
                    // today's path: the FP32 split (packed rows) or the stock BF16 kernel (the drafter)
                    CK(cudaMemset(d_out, 0xff, nq * 4));
                    if (bf)
                        sparse_attn_kernel<false><<<dim3(heads, T), kThreads>>>(d_q, d_win, d_blk, d_sel, kcap, ar, d_sink, heads, d, scale, d_out);
                    else
                        attnsplit::sparse_attn_split<4, 40>(d_q, d_win, d_ckv, kv_row_bytes(kKvFp4, d), d_sel, kcap, ar, d_sink, heads, T,
                                                            scale, d_out, sc, total_max, 0);
                    CK(cudaGetLastError());
                    const std::vector<float> o_old = fetch(d_out, nq);
                    const Err e_old = err(o_old, ref);
                    std::printf("%-38s q x%.0f  %s vs double: max_abs %.3g rel %.2e erel %.2e\n", c.name, qs, bf ? "stock" : "split",
                                e_old.abs, e_old.rel, e_old.erel);
                    for (int NT = 1; NT <= 2; ++NT) {
                        if (total_max > sc.nsplit * 16 * NT) continue;
                        CK(cudaMemset(d_out, 0xff, nq * 4));
                        if (NT == 1) sparse_attn_split_tc<1>(d_q, d_win, ck, bf, d_sel, kcap, ar, d_sink, heads, T, scale, d_out, sc, total_max, 0);
                        else sparse_attn_split_tc<2>(d_q, d_win, ck, bf, d_sel, kcap, ar, d_sink, heads, T, scale, d_out, sc, total_max, 0);
                        CK(cudaGetLastError());
                        const std::vector<float> o = fetch(d_out, nq);
                        const Err e = err(o, ref), ev = err(o, todouble(o_old));
                        const bool pass_ = e.rel <= 1e-3;   // (bound for a hi/lo BF16 product: ~2^-16 a term)
                        tok = tok && pass_;
                        worst = std::max(worst, e.rel);
                        std::printf("    tc %2d-row chunks vs double: max_abs %.3g rel %.2e erel %.2e | vs %s: max_abs %.3g rel %.2e erel %.2e  %s\n",
                                    16 * NT, e.abs, e.rel, e.erel, bf ? "stock" : "split", ev.abs, ev.rel, ev.erel, pass_ ? "ok" : "FAIL");
                    }
                    CK(cudaFree(d_win));
                    CK(cudaFree(d_q));
                    CK(cudaFree(d_sink));
                    CK(cudaFree(d_sel));
                    CK(cudaFree(d_out));
                }
            }
        std::printf("%s (worst tc rel vs double %.2e)\n", tok ? "TC ACCURACY OK" : "TC ACCURACY FAILED", worst);
        // ---- time a call (the RoPE fold on, as served): context 4K / 32K, ratio-4 and ratio-128 layers, T = 1, 2, 5, 6, 16
        std::vector<float> h_fr(32);
        for (int i = 0; i < 32; ++i) h_fr[i] = (float) std::pow(10000.0, -2.0 * i / 64.0);
        float* d_fr;
        CK(cudaMalloc(&d_fr, 32 * 4));
        CK(cudaMemcpy(d_fr, h_fr.data(), 32 * 4, cudaMemcpyHostToDevice));
        std::vector<uint16_t> h_win((size_t) n_winrows * d);
        for (auto& v : h_win) v = h_f2bf(nd(rng) * wmag);
        std::vector<float> h_q((size_t) 16 * heads * d), h_sink(heads);
        for (auto& v : h_q) v = nd(rng) * qmag;
        for (auto& v : h_sink) v = nd(rng) * 2.0f;
        uint16_t* d_win;
        float *d_q, *d_sink, *d_out;
        int* d_sel;
        CK(cudaMalloc(&d_win, h_win.size() * 2));
        CK(cudaMalloc(&d_q, h_q.size() * 4));
        CK(cudaMalloc(&d_sink, heads * 4));
        CK(cudaMalloc(&d_sel, (size_t) 16 * kcap * 4));
        CK(cudaMalloc(&d_out, h_q.size() * 4));
        CK(cudaMemcpy(d_win, h_win.data(), h_win.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_q, h_q.data(), h_q.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_sink, h_sink.data(), heads * 4, cudaMemcpyHostToDevice));
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        attnsplit::SplitRope rp{d_fr, 8189};
        std::printf("time a call, us (RoPE fold on; the engine always takes 32-row chunks, see --tinv):\n");
        std::printf("%-7s %-26s %3s %9s %9s %9s\n", "ctx", "layer", "T", "split", "tc16", "tc32");
        for (const int ctx : {4096, 32768})
            for (const int ratio : {4, 128}) {
                const int ncomp = ctx / ratio, nsel = std::min(kcap, ncomp);
                std::vector<int> h_sel((size_t) 16 * kcap);
                for (int t = 0; t < 16; ++t)
                    for (int i = 0; i < kcap; ++i)
                        h_sel[(size_t) t * kcap + i] = ratio == 4 ? (int) (rng() % (unsigned) ncomp) : i % ncomp;
                CK(cudaMemcpy(d_sel, h_sel.data(), h_sel.size() * 4, cudaMemcpyHostToDevice));
                for (const int T : {1, 2, 5, 6, 16}) {
                    AttnRows ar{};
                    for (int t = 0; t < T; ++t) ar.win0[t] = t, ar.nwin[t] = 128, ar.nsel[t] = nsel;
                    const int total_max = 128 + nsel;
                    float us[3] = {-1, -1, -1};
                    for (int k = 0; k < 3; ++k) {
                        if (k == 1 && total_max > sc.nsplit * 16) continue;
                        auto run = [&] {
                            if (k == 0)
                                attnsplit::sparse_attn_split<4, 40>(d_q, d_win, d_ckv, kv_row_bytes(kKvFp4, d), d_sel, kcap, ar, d_sink,
                                                                    heads, T, scale, d_out, sc, total_max, 0, &rp);
                            else if (k == 1)
                                sparse_attn_split_tc<1>(d_q, d_win, d_ckv, false, d_sel, kcap, ar, d_sink, heads, T, scale, d_out, sc, total_max, 0, &rp);
                            else
                                sparse_attn_split_tc<2>(d_q, d_win, d_ckv, false, d_sel, kcap, ar, d_sink, heads, T, scale, d_out, sc, total_max, 0, &rp);
                        };
                        for (int i = 0; i < 10; ++i) run();
                        CK(cudaEventRecord(e0));
                        for (int i = 0; i < 200; ++i) run();
                        CK(cudaEventRecord(e1));
                        CK(cudaEventSynchronize(e1));
                        CK(cudaGetLastError());
                        float ms = 0;
                        CK(cudaEventElapsedTime(&ms, e0, e1));
                        us[k] = ms / 200 * 1000;
                    }
                    char lay[40];
                    std::snprintf(lay, sizeof lay, "ratio %d (128+%d rows)", ratio, nsel);
                    std::printf("%-7s %-26s %3d %9.1f %9.1f %9.1f\n", ctx == 4096 ? "4K" : "32K", lay, T, us[0], us[1], us[2]);
                }
            }
        return tok ? 0 : 1;
    }
    if (tinv_mode) {
        // 16 queries of 1..640 rows (both sides of 160 and of a chunk), each a fixed window / selection / q / position;
        // query s0 + t of the call (s0, T) must equal query s0 + t run alone
        const int NQ = 16;
        const int nw[NQ] = {1, 37, 128, 128, 128, 128, 128, 128, 128, 128, 128, 64, 3, 128, 17, 90};
        const int ns[NQ] = {0, 41, 5, 32, 33, 160, 255, 512, 0, 1, 2, 300, 0, 40, 129, 7};
        const size_t qn = (size_t) heads * d;
        std::vector<uint16_t> h_win((size_t) n_winrows * d);
        for (auto& v : h_win) v = h_f2bf(nd(rng) * wmag);
        std::vector<float> h_q(NQ * qn), h_sink(heads), h_fr(32);
        for (auto& v : h_q) v = nd(rng) * qmag;
        for (auto& v : h_sink) v = nd(rng) * 2.0f;
        for (int i = 0; i < 32; ++i) h_fr[i] = (float) std::pow(10000.0, -2.0 * i / 64.0);
        std::vector<int> h_sel((size_t) NQ * kcap);
        for (auto& v : h_sel) v = rowpick(rng);
        uint16_t* d_win;
        float *d_q0, *d_q, *d_sink, *d_out, *d_fr;
        int* d_sel;
        CK(cudaMalloc(&d_win, h_win.size() * 2));
        CK(cudaMalloc(&d_q0, h_q.size() * 4));
        CK(cudaMalloc(&d_q, h_q.size() * 4));
        CK(cudaMalloc(&d_sink, heads * 4));
        CK(cudaMalloc(&d_fr, 32 * 4));
        CK(cudaMalloc(&d_sel, h_sel.size() * 4));
        CK(cudaMalloc(&d_out, h_q.size() * 4));
        CK(cudaMemcpy(d_win, h_win.data(), h_win.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_q0, h_q.data(), h_q.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_sink, h_sink.data(), heads * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_fr, h_fr.data(), 32 * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_sel, h_sel.data(), h_sel.size() * 4, cudaMemcpyHostToDevice));
        bool iok = true;
        for (int rp = 0; rp < 2; ++rp) {
            std::vector<uint32_t> solo(NQ * qn), o(NQ * qn);
            std::printf("tinv %-5s differing values (of a call's), T=2..16:", rp ? "rope" : "plain");
            for (int T = 1; T <= NQ; ++T) {
                size_t ndiff = 0;
                for (int s0 = 0; s0 + T <= NQ; ++s0) {
                    AttnRows ar{};
                    for (int t = 0; t < T; ++t) {
                        ar.win0[t] = (s0 + t) * 13 % 256;
                        ar.nwin[t] = nw[s0 + t];
                        ar.nsel[t] = ns[s0 + t];
                    }
                    AttnRope rope;   // query t at pos + t, as in a window
                    rope.freqs = d_fr;
                    rope.pos = 8189 + s0;
                    rope.rd = 64;
                    CK(cudaMemcpy(d_q, d_q0, h_q.size() * 4, cudaMemcpyDeviceToDevice));   // (a fallback may rotate q in place)
                    CK(cudaMemset(d_out, 0xff, T * qn * 4));
                    sparse_attn(d_q + s0 * qn, d_win, d_ckv, d_sel + (size_t) s0 * kcap, kcap, ar, d_sink, heads, d, T, scale,
                                d_out, 0, kKvFp4, 0, nullptr, rp ? &rope : nullptr);
                    CK(cudaGetLastError());
                    CK(cudaMemcpy(o.data(), d_out, T * qn * 4, cudaMemcpyDeviceToHost));
                    if (T == 1) std::memcpy(solo.data() + s0 * qn, o.data(), qn * 4);
                    else
                        for (size_t i = 0; i < T * qn; ++i) ndiff += o[i] != solo[s0 * qn + i];
                }
                if (T > 1) std::printf(" %zu", ndiff);
                iok = iok && ndiff == 0;
            }
            std::printf("\n");
        }
        std::printf("%s\n", iok ? "TINV ALL BITWISE" : "TINV FAILED");
        return iok ? 0 : 1;
    }
    if (rope_mode) {
        std::vector<Cfg> rc = {
            {"T=1 deep", 1, {128}, {512}, {0}},
            {"T=1 first token", 1, {1}, {0}, {0}},
            {"T=2 verify", 2, {128, 128}, {512, 511}, {0, 1}},
            {"T=3 verify", 3, {128, 128, 128}, {512, 300, 40}, {0, 1, 2}},
            {"T=4 verify", 4, {128, 64, 3, 128}, {512, 300, 0, 40}, {0, 5, 9, 2}},
            {"T=5 verify mixed", 5, {128, 64, 3, 128, 17}, {512, 300, 0, 40, 129}, {0, 5, 9, 2, 40}},
            {"T=6 verify deep", 6, {128, 128, 128, 128, 128, 128}, {512, 512, 512, 512, 512, 512}, {0, 1, 2, 3, 4, 5}},
            // (v1.1.31 prompt lookup: verify windows up to kWin)
            {"T=10 lookup mixed", 10, {128, 64, 3, 128, 17, 128, 128, 1, 90, 128}, {512, 300, 0, 40, 129, 512, 7, 0, 260, 512},
             {0, 5, 9, 2, 40, 6, 7, 8, 30, 9}},
            {"T=16 lookup deep", 16, {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
             {512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512},
             {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
        };
        static const char* rnames[] = {"T=7", "T=8", "T=9", "T=11", "T=12", "T=13", "T=14", "T=15"};
        int ri = 0;
        for (int T : {7, 8, 9, 11, 12, 13, 14, 15}) {   // (every T to kWin: the T=16 rows, cut)
            Cfg c = rc[8];
            c.name = rnames[ri++];
            c.T = T;
            rc.push_back(c);
        }
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
        const std::vector<double> ref = reference(c, h_q, h_win, h_sink, h_sel, kcap, h_dec.data());
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
