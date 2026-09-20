// JOIN4 integrated executor: staged bounded expert cache + static pins +
// section timers + decode/prefill split + TTFT.
//
// Base: staged-q2k BOUNDED_C (v10-style async pread-into-slots, proven on
// this pin) + ROUTE_C. Delta vs staged (ONLY these):
//   - GGML_PHASE6_SLOTS=N: exact slot count (preferred over CACHE_BYTES)
//   - GGML_PHASE6_PINS=/path: static pins, one (layer<<8)|expert int/line;
//     pinned bundles never evicted; PRELOADED at init (pins resident t=0,
//     matching sim semantics); preload bytes/nanoseconds counted separately
//   - GGML_PHASE6_PROFILE=1: barrier-flushed per-node section timers
//     (thread 0; exact by construction: every node lands in one bucket):
//     sections ATTN/GDN/MOE/SHARED/LMHEAD/MISC via cb-name markers,
//     EXPERT = MUL_MAT_ID on _exps (fetch_prep subtracted for compute),
//     ROUTER_K2 = MOE section minus expert minus fetch
//   - decode/prefill split (MoE ids rows==1 -> decode), decode token count,
//     TTFT (first prefill start -> first decode graph end)
// Works with bounded ON or OFF (resident arm: same binary, env unset;
// timers independent of the cache).
//
// Injected into ggml/src/ggml-cpu/ggml-cpu.c at the g_state anchor (same
// pin 3057bb6 as staged). Timer hooks called from the graph thread loop
// (join4_node_start at node top, join4_graph_end at loop end, ith==0).
// K1K2 graph patch (llama-graph.cpp) + iqp redirect + loader/laizy-model
// patches are separate anchors (see kernel); this file is ggml-cpu.c only.

// Phase 7B pipeline explicit byte-bounded expert store.  The loader registers the
// original GGUF offsets, while this C backend owns only fixed anonymous slots.
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

struct phase6_record {
    const struct ggml_tensor * tensor;
    int fd;
    size_t base_offset;
    size_t slice_bytes;
    int layer;
    int kind; // gate=0, up=1, down=2
    int valid;
};

struct phase6_slot {
    int layer;
    int expert;
    uint64_t age;
    int valid;
    int ready_mask;
};

struct phase6_async_task {
    int layer;
    int expert;
    int slot;
    size_t read_bytes;
    uint64_t read_ns;
    int decode;
};

static struct phase6_record phase6_records[40][3];
static int phase6_bundle_slots[40][256];
static struct phase6_slot * phase6_slots = NULL;
static unsigned char * phase6_storage = NULL;
static size_t phase6_slot_bytes = 0;
static size_t phase6_slot_count = 0;
static uint64_t phase6_age = 0;
static uint64_t phase6_requests = 0;
static uint64_t phase6_hits = 0;
static uint64_t phase6_misses = 0;
static uint64_t phase6_evictions = 0;
static uint64_t phase6_read_calls = 0;
static uint64_t phase6_read_bytes = 0;
static uint64_t phase6_read_ns = 0;
static uint64_t phase6_async_tasks = 0;
static uint64_t phase6_async_wait_ns = 0;
static uint64_t phase6_ready_wait_ns = 0;
static uint64_t phase6_ready_wait_events = 0;
static int phase6_initialized = 0;
static int phase6_report_registered = 0;
// JOIN4: decode-only request counters + mode flag (ids rows==1 -> decode)
static uint64_t phase6_dec_requests = 0;
static uint64_t phase6_dec_hits = 0;
static uint64_t phase6_dec_misses = 0;
static uint64_t phase6_dec_read_bytes = 0;
static uint64_t phase6_dec_read_ns = 0;
static int phase6_decode_mode = 0;
// JOIN4: static pins (bitset over (layer<<8)|expert, universe 10240)
enum { PHASE6_PIN_WORDS = 160 };
static uint64_t phase6_pin_bits[PHASE6_PIN_WORDS];
static uint64_t phase6_pins_loaded = 0;
static uint64_t phase6_preload_bytes = 0;
static uint64_t phase6_preload_ns = 0;
static uint64_t phase6_pin_violations = 0;
enum { PHASE6_MAX_ASYNC_TASKS = 4096 };
static struct phase6_async_task phase6_async_task_records[PHASE6_MAX_ASYNC_TASKS];
static pthread_t phase6_async_threads[PHASE6_MAX_ASYNC_TASKS];
static int phase6_async_task_count = 0;
static int phase6_async_started_count = 0;
static int phase6_async_layer = -1;
static void phase6_reap_async(void);

static int phase6_enabled(void) {
    const char * value = getenv("GGML_PHASE6_BOUNDED_CACHE");
    return value != NULL && atoi(value) != 0;
}

static int phase6_async_enabled(void) {
    const char * value = getenv("GGML_PHASE6_ASYNC");
    return value != NULL && atoi(value) != 0;
}

static int join4_profile_enabled(void) {
    const char * value = getenv("GGML_PHASE6_PROFILE");
    return value != NULL && atoi(value) != 0;
}

// Cached flag for the per-node dispatch hook (avoids getenv per node).
static int join4_prof_cached = -1;
static int join4_prof_on(void) {
    if (join4_prof_cached < 0)
        join4_prof_cached = join4_profile_enabled() ? 1 : 0;
    return join4_prof_cached;
}

// Zero-copy challenger state: kernels consume expert bytes directly from a
// file-backed MAP_SHARED mapping of the GGUF.  Slots stay purely logical
// (same LRU policy/count as the control); eviction bounds RSS with
// MADV_DONTNEED on the evicted bundle's file ranges.  No pread, no slot
// copies, no reader threads.
// JOIN4: vehicle dead (phase10i verdict); code kept verbatim, never enabled.
static unsigned char * phase6_zc_file_map = NULL;
static size_t phase6_zc_file_size = 0;
static uint64_t phase6_zc_madvise_ns = 0;
static uint64_t phase6_zc_madvise_calls = 0;
static uint64_t phase6_zc_madvise_bytes = 0;
static uint64_t phase6_zc_madvise_errors = 0;
enum { PHASE6_ZC_EVICT_RING = 64 };
static int phase6_zc_evict_layer[PHASE6_ZC_EVICT_RING];
static int phase6_zc_evict_expert[PHASE6_ZC_EVICT_RING];
static int phase6_zc_evict_count = 0;

static int phase6_zc_enabled(void) {
    const char * value = getenv("GGML_PHASE6_ZERO_COPY");
    return phase6_enabled() && value != NULL && atoi(value) != 0;
}

static uint64_t phase6_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
}

// JOIN4 section timers: barrier-flushed per-node attribution (thread 0).
// Every node lands in exactly one bucket => sum == graph wall by design.
enum {
    JOIN4_SEC_MISC = 0, JOIN4_SEC_ATTN_TENT, JOIN4_SEC_MOE, JOIN4_SEC_SHARED,
    JOIN4_SEC_LMHEAD, JOIN4_NSECS
};
static uint64_t join4_dec_ns[JOIN4_NSECS];
static uint64_t join4_pre_ns[JOIN4_NSECS];
static uint64_t join4_dec_cnt[JOIN4_NSECS];
static uint64_t join4_pre_cnt[JOIN4_NSECS];
static uint64_t join4_dec_attn_ns = 0;   // full-attention section
static uint64_t join4_pre_attn_ns = 0;
static uint64_t join4_dec_gdn_ns = 0;    // gated-delta-net section
static uint64_t join4_pre_gdn_ns = 0;
static uint64_t join4_scratch_dec_ns = 0; // ATTN_TENT accumulates here...
static uint64_t join4_scratch_pre_ns = 0; // ...resolved to ATTN/GDN at close
static uint64_t join4_dec_expert_ns = 0; // MUL_MAT_ID on _exps (node wall)
static uint64_t join4_pre_expert_ns = 0;
static uint64_t join4_dec_expert_cnt = 0;
static uint64_t join4_pre_expert_cnt = 0;
static uint64_t join4_dec_fetchprep_ns = 0; // in-node fetch (subtracted)
static uint64_t join4_pre_fetchprep_ns = 0;
static int join4_section = JOIN4_SEC_MISC;
static int join4_decode_mode = 0; // init PREFILL: first graph is prefill
static int join4_pending_valid = 0;
static uint64_t join4_pending_t0 = 0;
static int join4_pending_bucket = JOIN4_SEC_MISC;
static int join4_pending_expert = 0;
static int join4_pending_resolve = 0; // 0=none, 1=attn, 2=gdn (closer nodes)
static int join4_pending_wb = -1; // weight bucket, -1 = not a matmul
static uint64_t join4_markers_seen = 0;
static uint64_t join4_force_closes = 0;
// JOIN4b: weight-bucket timers. EVERY MUL_MAT/MUL_MAT_ID attributed by its
// src0 (weight) tensor name from the loader: immutable, always present,
// independent of cb-name sections (v2 sections leaked unnamed bodies into
// MISC: shexp matmuls run before their "ffn_shexp" opener, etc.).
// Both attributions run (sections + weight buckets); residual cross-checks.
enum {
    JOIN4_WB_EXPS = 0, JOIN4_WB_ATTN, JOIN4_WB_GDN, JOIN4_WB_SHEXP,
    JOIN4_WB_ROUTER, JOIN4_WB_OUT, JOIN4_WB_OTHER, JOIN4_WB_N
};
static uint64_t join4_dec_wb[JOIN4_WB_N];
static uint64_t join4_pre_wb[JOIN4_WB_N];
static uint64_t join4_dec_wb_cnt[JOIN4_WB_N];
static uint64_t join4_pre_wb_cnt[JOIN4_WB_N];
static uint64_t join4_wb_unmatched_logged = 0;

static int join4_wbucket(const char * wname) {
    if (wname == NULL) return JOIN4_WB_OTHER;
    if (strstr(wname, "_exps")) return JOIN4_WB_EXPS;
    if (strstr(wname, "attn_") || strstr(wname, "qkv")) return JOIN4_WB_ATTN;
    if (strstr(wname, "ssm_") || strstr(wname, "conv") ||
        strstr(wname, "delta")) return JOIN4_WB_GDN;
    if (strstr(wname, "shexp")) return JOIN4_WB_SHEXP;
    if (strstr(wname, "ffn_gate_inp")) return JOIN4_WB_ROUTER;
    if (strstr(wname, "output")) return JOIN4_WB_OUT;
    return JOIN4_WB_OTHER;
}

// Nodelist ground truth: first graph's node sequence (env-gated, 1 run).
static FILE * join4_nodelist_fp = NULL;
static int join4_nodelist_done = 0;
static uint64_t join4_decode_graphs = 0;
static uint64_t join4_prefill_graphs = 0;
static uint64_t join4_graph_moe_nodes = 0;
static uint64_t join4_prefill_start_ns = 0;
static uint64_t join4_started = 0;
static uint64_t join4_ttft_ns = 0;

static void phase6_zc_evict_range(int layer, int expert) {
    if (phase6_zc_file_map == NULL) return;
    const uint64_t start = phase6_now_ns();
    long page = 4096;
#if defined(__linux__)
    const long configured = sysconf(_SC_PAGESIZE);
    if (configured > 0) page = configured;
#endif
    const size_t psize = (size_t) page;
    for (int kind = 0; kind < 3; ++kind) {
        const struct phase6_record * record = &phase6_records[layer][kind];
        const size_t begin = record->base_offset + (size_t) expert * record->slice_bytes;
        const size_t end = begin + record->slice_bytes;
        const size_t aligned_begin = (begin / psize) * psize;
        const size_t aligned_end = ((end + psize - 1) / psize) * psize;
        if (aligned_end > phase6_zc_file_size) continue;
#if defined(__linux__)
        if (madvise(phase6_zc_file_map + aligned_begin, aligned_end - aligned_begin, MADV_DONTNEED) != 0) {
            phase6_zc_madvise_errors++;
        }
#endif
        phase6_zc_madvise_bytes += aligned_end - aligned_begin;
    }
    phase6_zc_madvise_calls += 3;
    phase6_zc_madvise_ns += phase6_now_ns() - start;
}

static int phase6_kind(const char * name) {
    if (name == NULL || strstr(name, "ffn_") == NULL || strstr(name, "_exps") == NULL) return -1;
    if (strstr(name, "ffn_gate_exps")) return 0;
    if (strstr(name, "ffn_up_exps")) return 1;
    if (strstr(name, "ffn_down_exps")) return 2;
    return -1;
}

static int phase6_layer(const char * name) {
    int layer = -1;
    if (name != NULL) sscanf(name, "blk.%d.", &layer);
    return layer;
}

void ggml_cpu_phase6_register_lazy_tensor(const struct ggml_tensor * tensor,
        int fd, size_t base_offset, size_t nbytes, size_t expert_stride) {
    if (!phase6_enabled() || tensor == NULL || tensor->name == NULL) return;
    const int layer = phase6_layer(tensor->name);
    const int kind = phase6_kind(tensor->name);
    if (layer < 0 || layer >= 40 || kind < 0 || kind >= 3 || expert_stride == 0 ||
            expert_stride * 256 > nbytes) return;
    // llama_model_loader owns the original descriptor and may destroy it
    // after model construction.  The explicit cache owns this duplicate so
    // pread remains valid during generation.
    const int owned_fd = dup(fd);
    if (owned_fd < 0) {
        fprintf(stderr, "PHASE6_BOUNDED_CACHE_ERROR dup failed errno=%d\n", errno);
        abort();
    }
    phase6_records[layer][kind] = (struct phase6_record) {
        tensor, owned_fd, base_offset, expert_stride, layer, kind, 1
    };
}

static void phase6_fail(const char * message) {
    fprintf(stderr, "PHASE6_BOUNDED_CACHE_ERROR %s\n", message);
    abort();
}

// JOIN4: load static pins (one (layer<<8)|expert int per line).
static void phase6_load_pins(void) {
    const char * path = getenv("GGML_PHASE6_PINS");
    if (path == NULL || path[0] == '\0') return;
    FILE * fp = fopen(path, "r");
    if (fp == NULL) phase6_fail("pins file not readable");
    char line[256];
    while (fgets(line, sizeof(line), fp) != NULL) {
        char * end = NULL;
        const long key = strtol(line, &end, 10);
        if (end == line || key < 0 || key >= 10240) phase6_fail("bad pin key");
        phase6_pin_bits[((uint32_t) key >> 6) % PHASE6_PIN_WORDS] |= 1ULL << (key & 63);
        phase6_pins_loaded++;
    }
    fclose(fp);
}

static int phase6_pinned(int layer, int expert) {
    const uint32_t key = ((uint32_t) layer << 8) | (uint32_t) expert;
    return (int) ((phase6_pin_bits[(key >> 6) % PHASE6_PIN_WORDS] >> (key & 63)) & 1ULL);
}

static void phase6_copy_read(int fd, void * dst, size_t bytes, size_t offset) {
    size_t done = 0;
    while (done < bytes) {
        const ssize_t got = pread(fd, (char *) dst + done, bytes - done, offset + done);
        if (got < 0) {
            if (errno == EINTR) continue;
            phase6_fail("async pread failed");
        }
        if (got == 0) phase6_fail("async short pread");
        done += (size_t) got;
    }
}

static void phase6_report(void) {
    if (!phase6_enabled() && !join4_profile_enabled()) return;
    if (phase6_enabled() && phase6_async_enabled()) phase6_reap_async();
    if (phase6_enabled()) {
    fprintf(stderr,
        "PHASE6_BOUNDED_CACHE slot_bytes=%zu slots=%zu reserved_bytes=%zu "
        "requests=%llu hits=%llu misses=%llu evictions=%llu read_calls=%llu "
        "read_bytes=%llu read_ns=%llu async_tasks=%llu async_wait_ns=%llu "
        "ready_wait_ns=%llu ready_wait_events=%llu "
        "pins=%llu preload_bytes=%llu preload_ns=%llu pin_violations=%llu "
        "dec_requests=%llu dec_hits=%llu dec_misses=%llu "
        "dec_read_bytes=%llu dec_read_ns=%llu\n",
        phase6_slot_bytes, phase6_slot_count, phase6_slot_bytes * phase6_slot_count,
        (unsigned long long) phase6_requests, (unsigned long long) phase6_hits,
        (unsigned long long) phase6_misses, (unsigned long long) phase6_evictions,
        (unsigned long long) phase6_read_calls, (unsigned long long) phase6_read_bytes,
        (unsigned long long) phase6_read_ns, (unsigned long long) phase6_async_tasks,
        (unsigned long long) phase6_async_wait_ns,
        (unsigned long long) phase6_ready_wait_ns,
        (unsigned long long) phase6_ready_wait_events,
        (unsigned long long) phase6_pins_loaded,
        (unsigned long long) phase6_preload_bytes,
        (unsigned long long) phase6_preload_ns,
        (unsigned long long) phase6_pin_violations,
        (unsigned long long) phase6_dec_requests,
        (unsigned long long) phase6_dec_hits,
        (unsigned long long) phase6_dec_misses,
        (unsigned long long) phase6_dec_read_bytes,
        (unsigned long long) phase6_dec_read_ns);
    }
    if (join4_profile_enabled()) {
    fprintf(stderr,
        "PHASE6_PROFILE dec_graphs=%llu pre_graphs=%llu ttft_ns=%llu "
        "dec_attn_ns=%llu dec_gdn_ns=%llu dec_moe_rest_ns=%llu "
        "dec_expert_node_ns=%llu dec_expert_cnt=%llu dec_fetchprep_ns=%llu "
        "dec_shared_ns=%llu dec_lmhead_ns=%llu dec_misc_ns=%llu "
        "pre_attn_ns=%llu pre_gdn_ns=%llu pre_moe_rest_ns=%llu "
        "pre_expert_node_ns=%llu pre_expert_cnt=%llu pre_fetchprep_ns=%llu "
        "pre_shared_ns=%llu pre_lmhead_ns=%llu pre_misc_ns=%llu "
        "markers=%llu force_closes=%llu "
        "wb_dec_exps=%llu wb_dec_attn=%llu wb_dec_gdn=%llu "
        "wb_dec_shexp=%llu wb_dec_router=%llu wb_dec_out=%llu "
        "wb_dec_other=%llu wb_pre_exps=%llu wb_pre_attn=%llu "
        "wb_pre_gdn=%llu wb_pre_shexp=%llu wb_pre_router=%llu "
        "wb_pre_out=%llu wb_pre_other=%llu\n",
        (unsigned long long) join4_decode_graphs,
        (unsigned long long) join4_prefill_graphs,
        (unsigned long long) join4_ttft_ns,
        (unsigned long long) join4_dec_attn_ns,
        (unsigned long long) join4_dec_gdn_ns,
        (unsigned long long) join4_dec_ns[JOIN4_SEC_MOE],
        (unsigned long long) join4_dec_expert_ns,
        (unsigned long long) join4_dec_expert_cnt,
        (unsigned long long) join4_dec_fetchprep_ns,
        (unsigned long long) join4_dec_ns[JOIN4_SEC_SHARED],
        (unsigned long long) join4_dec_ns[JOIN4_SEC_LMHEAD],
        (unsigned long long) join4_dec_ns[JOIN4_SEC_MISC],
        (unsigned long long) join4_pre_attn_ns,
        (unsigned long long) join4_pre_gdn_ns,
        (unsigned long long) join4_pre_ns[JOIN4_SEC_MOE],
        (unsigned long long) join4_pre_expert_ns,
        (unsigned long long) join4_pre_expert_cnt,
        (unsigned long long) join4_pre_fetchprep_ns,
        (unsigned long long) join4_pre_ns[JOIN4_SEC_SHARED],
        (unsigned long long) join4_pre_ns[JOIN4_SEC_LMHEAD],
        (unsigned long long) join4_pre_ns[JOIN4_SEC_MISC],
        (unsigned long long) join4_markers_seen,
        (unsigned long long) join4_force_closes,
        (unsigned long long) join4_dec_wb[JOIN4_WB_EXPS],
        (unsigned long long) join4_dec_wb[JOIN4_WB_ATTN],
        (unsigned long long) join4_dec_wb[JOIN4_WB_GDN],
        (unsigned long long) join4_dec_wb[JOIN4_WB_SHEXP],
        (unsigned long long) join4_dec_wb[JOIN4_WB_ROUTER],
        (unsigned long long) join4_dec_wb[JOIN4_WB_OUT],
        (unsigned long long) join4_dec_wb[JOIN4_WB_OTHER],
        (unsigned long long) join4_pre_wb[JOIN4_WB_EXPS],
        (unsigned long long) join4_pre_wb[JOIN4_WB_ATTN],
        (unsigned long long) join4_pre_wb[JOIN4_WB_GDN],
        (unsigned long long) join4_pre_wb[JOIN4_WB_SHEXP],
        (unsigned long long) join4_pre_wb[JOIN4_WB_ROUTER],
        (unsigned long long) join4_pre_wb[JOIN4_WB_OUT],
        (unsigned long long) join4_pre_wb[JOIN4_WB_OTHER]);
    }
    if (phase6_enabled()) {
    if (phase6_zc_enabled()) {
        long admitted_pages = 0;
        long admitted_resident = 0;
        long evicted_pages = 0;
        long evicted_resident = 0;
#if defined(__linux__)
        long page = 4096;
        const long configured = sysconf(_SC_PAGESIZE);
        if (configured > 0) page = configured;
        const size_t psize = (size_t) page;
        unsigned char vec = 0;
        long sampled = 0;
        for (size_t i = 0; i < phase6_slot_count && sampled < 96; ++i) {
            if (!phase6_slots[i].valid) continue;
            const int layer = phase6_slots[i].layer;
            const int expert = phase6_slots[i].expert;
            const struct phase6_record * record = &phase6_records[layer][0];
            const size_t mid = record->base_offset + (size_t) expert * record->slice_bytes
                + record->slice_bytes / 2;
            const size_t aligned = (mid / psize) * psize;
            if (aligned + psize > phase6_zc_file_size) continue;
            vec = 0;
            if (mincore(phase6_zc_file_map + aligned, psize, &vec) == 0) {
                admitted_pages++;
                if (vec & 1) admitted_resident++;
            }
            sampled++;
        }
        const int ring = phase6_zc_evict_count < PHASE6_ZC_EVICT_RING
            ? phase6_zc_evict_count : PHASE6_ZC_EVICT_RING;
        for (int i = 0; i < ring; ++i) {
            const int layer = phase6_zc_evict_layer[i];
            const int expert = phase6_zc_evict_expert[i];
            if (layer < 0 || phase6_bundle_slots[layer][expert] >= 0) continue;
            const struct phase6_record * record = &phase6_records[layer][0];
            const size_t mid = record->base_offset + (size_t) expert * record->slice_bytes
                + record->slice_bytes / 2;
            const size_t aligned = (mid / psize) * psize;
            if (aligned + psize > phase6_zc_file_size) continue;
            vec = 0;
            if (mincore(phase6_zc_file_map + aligned, psize, &vec) == 0) {
                evicted_pages++;
                if (vec & 1) evicted_resident++;
            }
        }
#endif
        long smaps_rss_kb = -1, smaps_anon_kb = -1, smaps_file_kb = -1;
#if defined(__linux__)
        FILE * smaps = fopen("/proc/self/smaps_rollup", "r");
        if (smaps != NULL) {
            char line[256];
            while (fgets(line, sizeof(line), smaps) != NULL) {
                if (strncmp(line, "Rss:", 4) == 0) smaps_rss_kb = atol(line + 4);
                else if (strncmp(line, "Anonymous:", 10) == 0) smaps_anon_kb = atol(line + 10);
                else if (strncmp(line, "FilePmdMapped:", 14) == 0) { /* skip */ }
                else if (line[0] == 'F' && strncmp(line, "FileRSS:", 8) == 0) { /* older kernels */ }
            }
            fclose(smaps);
            // smaps_rollup reports Rss/Anonymous; derive file as Rss - Anonymous.
            if (smaps_rss_kb >= 0 && smaps_anon_kb >= 0) smaps_file_kb = smaps_rss_kb - smaps_anon_kb;
        }
#endif
        fprintf(stderr,
            "PHASE6_ZERO_COPY file_bytes=%zu madvise_calls=%llu madvise_bytes=%llu "
            "madvise_ns=%llu madvise_errors=%llu admitted_pages=%ld admitted_resident=%ld "
            "evicted_pages=%ld evicted_resident=%ld smaps_rss_kb=%ld smaps_anon_kb=%ld smaps_file_kb=%ld\n",
            phase6_zc_file_size,
            (unsigned long long) phase6_zc_madvise_calls,
            (unsigned long long) phase6_zc_madvise_bytes,
            (unsigned long long) phase6_zc_madvise_ns,
            (unsigned long long) phase6_zc_madvise_errors,
            admitted_pages, admitted_resident, evicted_pages, evicted_resident,
            smaps_rss_kb, smaps_anon_kb, smaps_file_kb);
    }
#if defined(__linux__)
    for (int layer = 0; layer < 40; ++layer) {
        for (int kind = 0; kind < 3; ++kind) {
            if (phase6_records[layer][kind].valid && phase6_records[layer][kind].fd >= 0) {
                close(phase6_records[layer][kind].fd);
                phase6_records[layer][kind].fd = -1;
            }
        }
    }
#endif
    }
}

static void phase6_init(void) {
    if (phase6_initialized) return;
    for (int layer = 0; layer < 40; ++layer) {
        for (int expert = 0; expert < 256; ++expert) phase6_bundle_slots[layer][expert] = -1;
    }
    for (int layer = 0; layer < 40; ++layer) {
        size_t total = 0;
        for (int kind = 0; kind < 3; ++kind) {
            if (!phase6_records[layer][kind].valid) phase6_fail("incomplete routed tensor registration");
            total += phase6_records[layer][kind].slice_bytes;
        }
        if (layer == 0) phase6_slot_bytes = total;
        if (total != phase6_slot_bytes) phase6_fail("nonuniform bundle size is not supported by v1");
    }
    // JOIN4: exact slot count preferred; byte capacity is the fallback.
    const char * slots_text = getenv("GGML_PHASE6_SLOTS");
    if (slots_text != NULL && slots_text[0] != '\0') {
        phase6_slot_count = (size_t) strtoull(slots_text, NULL, 10);
    } else {
        const char * capacity_text = getenv("GGML_PHASE6_CACHE_BYTES");
        const unsigned long long capacity = capacity_text ? strtoull(capacity_text, NULL, 10) : 0;
        phase6_slot_count = phase6_slot_bytes ? (size_t) (capacity / phase6_slot_bytes) : 0;
    }
    if (phase6_slot_count == 0) phase6_fail("cache capacity does not hold one routed bundle");
    phase6_load_pins();
    if ((size_t) phase6_pins_loaded > phase6_slot_count) phase6_fail("more pins than slots");
    phase6_slots = (struct phase6_slot *) calloc(phase6_slot_count, sizeof(*phase6_slots));
    if (phase6_zc_enabled()) {
        // Challenger: map the GGUF file-backed and consume expert bytes
        // directly.  Slots stay logical only; no anonymous storage.
        for (int i = 0; i < PHASE6_ZC_EVICT_RING; ++i) {
            phase6_zc_evict_layer[i] = -1;
            phase6_zc_evict_expert[i] = -1;
        }
        const int map_fd = phase6_records[0][0].fd;
        struct stat st;
        if (map_fd < 0 || fstat(map_fd, &st) != 0 || st.st_size <= 0) {
            phase6_fail("zero-copy fstat failed");
        }
        phase6_zc_file_size = (size_t) st.st_size;
#if defined(__linux__)
        phase6_zc_file_map = (unsigned char *) mmap(NULL, phase6_zc_file_size,
            PROT_READ, MAP_SHARED, map_fd, 0);
        if (phase6_zc_file_map == MAP_FAILED) phase6_zc_file_map = NULL;
        // v2 fix: MADV_RANDOM disables readahead AND fault-around.  v1 leaked
        // file RSS (4.02 GiB, still climbing) via fault-around zombie ptes:
        // the 64 KB speculative window re-mapped evicted pages adjacent to
        // admitted slices without re-admission.
        if (phase6_zc_file_map != NULL) {
            madvise(phase6_zc_file_map, phase6_zc_file_size, MADV_RANDOM);
        }
#endif
        if (phase6_zc_file_map == NULL || phase6_slots == NULL) {
            phase6_fail("zero-copy file mapping failed");
        }
    } else {
#if defined(__linux__)
        phase6_storage = (unsigned char *) mmap(NULL, phase6_slot_bytes * phase6_slot_count,
            PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (phase6_storage == MAP_FAILED) phase6_storage = NULL;
#endif
        if (phase6_storage == NULL || phase6_slots == NULL) phase6_fail("fixed cache allocation failed");
    }
    // JOIN4: preload pins (sync; one-time; counted separately from traffic).
    if (phase6_pins_loaded > 0 && !phase6_zc_enabled()) {
        const uint64_t t0 = phase6_now_ns();
        for (int layer = 0; layer < 40; ++layer) {
            for (int expert = 0; expert < 256; ++expert) {
                if (!phase6_pinned(layer, expert)) continue;
                int slot = -1;
                for (size_t i = 0; i < phase6_slot_count; ++i) {
                    if (!phase6_slots[i].valid) { slot = (int) i; break; }
                }
                if (slot < 0) phase6_fail("pin preload found no free slot");
                phase6_slots[slot] = (struct phase6_slot) {
                    layer, expert, ++phase6_age, 1, 7
                };
                phase6_bundle_slots[layer][expert] = slot;
                unsigned char * dst = phase6_storage + (size_t) slot * phase6_slot_bytes;
                size_t in_slot = 0;
                for (int kind = 0; kind < 3; ++kind) {
                    const struct phase6_record * record = &phase6_records[layer][kind];
                    phase6_copy_read(record->fd, dst + in_slot, record->slice_bytes,
                        record->base_offset + (size_t) expert * record->slice_bytes);
                    in_slot += record->slice_bytes;
                    phase6_preload_bytes += record->slice_bytes;
                }
            }
        }
        phase6_preload_ns = phase6_now_ns() - t0;
    }
    if (!phase6_report_registered) {
        phase6_report_registered = 1;
        atexit(phase6_report);
    }
    phase6_initialized = 1;
}

static int phase6_find_slot(int layer, int expert) {
    if (layer < 0 || layer >= 40 || expert < 0 || expert >= 256) return -1;
    return phase6_bundle_slots[layer][expert];
}

static void phase6_read_exact(int fd, void * dst, size_t bytes, size_t offset) {
    size_t done = 0;
    const uint64_t start = phase6_now_ns();
    while (done < bytes) {
        const ssize_t got = pread(fd, (char *) dst + done, bytes - done, offset + done);
        if (got < 0) {
            if (errno == EINTR) continue;
            phase6_fail("pread failed");
        }
        if (got == 0) phase6_fail("short pread");
        done += (size_t) got;
    }
    const uint64_t end = phase6_now_ns();
    phase6_read_ns += end - start;
    phase6_read_calls += 3;
    phase6_read_bytes += bytes;
    if (phase6_decode_mode) {
        phase6_dec_read_ns += end - start;
        phase6_dec_read_bytes += bytes;
    }
}

static int phase6_reserve_slot(int layer, int expert) {
    int slot = -1;
    for (size_t i = 0; i < phase6_slot_count; ++i) {
        if (!phase6_slots[i].valid) { slot = (int) i; break; }
    }
    if (slot < 0) {
        // JOIN4: victim = oldest NON-PINNED slot (pins never evicted).
        uint64_t oldest = UINT64_MAX;
        for (size_t i = 0; i < phase6_slot_count; ++i) {
            if (!phase6_slots[i].valid) continue;
            if (phase6_pinned(phase6_slots[i].layer, phase6_slots[i].expert)) continue;
            if (phase6_slots[i].age < oldest) { oldest = phase6_slots[i].age; slot = (int) i; }
        }
        if (slot < 0) {
            // Degenerate (all slots pinned): evict oldest anyway, counted.
            oldest = UINT64_MAX;
            for (size_t i = 0; i < phase6_slot_count; ++i) {
                if (phase6_slots[i].age < oldest) { oldest = phase6_slots[i].age; slot = (int) i; }
            }
            phase6_pin_violations++;
        }
        if (phase6_slots[slot].valid) {
            const int old_layer = phase6_slots[slot].layer;
            const int old_expert = phase6_slots[slot].expert;
            phase6_bundle_slots[old_layer][old_expert] = -1;
            phase6_evictions++;
            if (phase6_zc_enabled()) {
                const int ring = phase6_zc_evict_count % PHASE6_ZC_EVICT_RING;
                phase6_zc_evict_layer[ring] = old_layer;
                phase6_zc_evict_expert[ring] = old_expert;
                phase6_zc_evict_count++;
                phase6_zc_evict_range(old_layer, old_expert);
            }
        }
        /* S2 baseline (noDN): NO madvise(DONTNEED) on staged evict. The refill
         * pread overwrites the slot pages in place (~15x cheaper per miss,
         * exact-output; see probes/hw_agent3_memory/REPORT.md). The zero-copy
         * arm's file-range eviction (phase6_zc_evict_range above) is KEEP:
         * it bounds file-backed RSS, a different mechanism. */
    }
    phase6_slots[slot] = (struct phase6_slot) {
        layer, expert, ++phase6_age, 1,
        (phase6_async_enabled() && !phase6_zc_enabled()) ? 0 : 7
    };
    phase6_bundle_slots[layer][expert] = slot;
    return slot;
}

// JOIN4: per-expert accounting with decode split (mode set at prepare entry).
static void phase6_note(int layer, int expert, int hit) {
    (void) layer; (void) expert;
    phase6_requests++;
    if (hit) phase6_hits++; else phase6_misses++;
    if (phase6_decode_mode) {
        phase6_dec_requests++;
        if (hit) phase6_dec_hits++; else phase6_dec_misses++;
    }
}

static int phase6_load(int layer, int expert) {
    phase6_init();
    int slot = phase6_find_slot(layer, expert);
    if (slot >= 0) {
        phase6_note(layer, expert, 1);
        phase6_slots[slot].age = ++phase6_age;
        return slot;
    }
    phase6_note(layer, expert, 0);
    slot = phase6_reserve_slot(layer, expert);
    unsigned char * dst = phase6_storage + (size_t) slot * phase6_slot_bytes;
    size_t in_slot = 0;
    for (int kind = 0; kind < 3; ++kind) {
        const struct phase6_record * record = &phase6_records[layer][kind];
        phase6_read_exact(record->fd, dst + in_slot, record->slice_bytes,
                          record->base_offset + (size_t) expert * record->slice_bytes);
        in_slot += record->slice_bytes;
    }
    return slot;
}

static void * phase6_async_read_worker(void * opaque) {
    struct phase6_async_task * task = (struct phase6_async_task *) opaque;
    const uint64_t start = phase6_now_ns();
    unsigned char * dst = phase6_storage + (size_t) task->slot * phase6_slot_bytes;
    size_t in_slot = 0;
    for (int kind = 0; kind < 3; ++kind) {
        const struct phase6_record * record = &phase6_records[task->layer][kind];
        phase6_copy_read(record->fd, dst + in_slot, record->slice_bytes,
                         record->base_offset + (size_t) task->expert * record->slice_bytes);
        in_slot += record->slice_bytes;
        task->read_bytes += record->slice_bytes;
        __atomic_fetch_or(&phase6_slots[task->slot].ready_mask, 1 << kind, __ATOMIC_RELEASE);
    }
    task->read_ns = phase6_now_ns() - start;
    return NULL;
}

static void phase6_reap_async(void) {
    if (phase6_async_started_count == 0) return;
    const uint64_t start = phase6_now_ns();
    for (int i = 0; i < phase6_async_started_count; ++i) {
        if (pthread_join(phase6_async_threads[i], NULL) != 0) {
            phase6_fail("pipeline pthread_join failed");
        }
        phase6_read_ns += phase6_async_task_records[i].read_ns;
        phase6_read_calls += 3;
        phase6_read_bytes += phase6_async_task_records[i].read_bytes;
        if (phase6_async_task_records[i].decode) {
            phase6_dec_read_ns += phase6_async_task_records[i].read_ns;
            phase6_dec_read_bytes += phase6_async_task_records[i].read_bytes;
        }
    }
    phase6_async_wait_ns += phase6_now_ns() - start;
    phase6_async_task_count = 0;
    phase6_async_started_count = 0;
    phase6_async_layer = -1;
}

static void phase6_prepare_async(const struct ggml_tensor * tensor,
                                 const struct ggml_tensor * ids) {
    phase6_init();
    const int layer = phase6_layer(tensor->name);
    if (layer < 0 || layer >= 40 || ids == NULL || ids->type != GGML_TYPE_I32) {
        phase6_fail("invalid async routed node metadata");
    }
    // The previous async arm joined every task before compute.  This arm
    // retains the task records across graph nodes, publishes each plane's
    // readiness, and only reaps after the graph advances to another layer.
    if (phase6_async_layer >= 0 && phase6_async_layer != layer) phase6_reap_async();
    if (phase6_async_layer < 0) phase6_async_layer = layer;
    const int first_new = phase6_async_task_count;
    for (int64_t row = 0; row < ids->ne[1]; ++row) {
        for (int64_t i = 0; i < ids->ne[0]; ++i) {
            const int expert = *(const int32_t *) ((const char *) ids->data +
                row * ids->nb[1] + i * ids->nb[0]);
            if (expert < 0 || expert >= 256) phase6_fail("invalid native expert ID");
            int slot = phase6_find_slot(layer, expert);
            if (slot >= 0) {
                phase6_note(layer, expert, 1);
                phase6_slots[slot].age = ++phase6_age;
                continue;
            }
            phase6_note(layer, expert, 0);
            slot = phase6_reserve_slot(layer, expert);
            if (phase6_async_task_count >= PHASE6_MAX_ASYNC_TASKS) phase6_fail("too many pipeline tasks");
            phase6_async_task_records[phase6_async_task_count++] = (struct phase6_async_task) {
                layer, expert, slot, 0, 0, phase6_decode_mode
            };
        }
    }
    for (int i = first_new; i < phase6_async_task_count; ++i) {
        if (pthread_create(&phase6_async_threads[i], NULL, phase6_async_read_worker,
                           &phase6_async_task_records[i]) != 0) {
            phase6_fail("pipeline pthread_create failed");
        }
    }
    phase6_async_started_count = phase6_async_task_count;
    phase6_async_tasks += (uint64_t) (phase6_async_task_count - first_new);
}

static void phase6_prepare_zc(const struct ggml_tensor * tensor, const struct ggml_tensor * ids) {
    // Challenger: logical admission only.  No reads, no copies, no threads;
    // kernels fault the file-backed bytes directly on first touch.
    phase6_init();
    const int layer = phase6_layer(tensor->name);
    if (layer < 0 || layer >= 40 || ids == NULL || ids->type != GGML_TYPE_I32) {
        phase6_fail("invalid zero-copy routed node metadata");
    }
    for (int64_t row = 0; row < ids->ne[1]; ++row) {
        for (int64_t i = 0; i < ids->ne[0]; ++i) {
            const int expert = *(const int32_t *) ((const char *) ids->data + row * ids->nb[1] + i * ids->nb[0]);
            if (expert < 0 || expert >= 256) phase6_fail("invalid native expert ID");
            int slot = phase6_find_slot(layer, expert);
            if (slot >= 0) {
                phase6_note(layer, expert, 1);
                phase6_slots[slot].age = ++phase6_age;
                continue;
            }
            phase6_note(layer, expert, 0);
            phase6_reserve_slot(layer, expert);
        }
    }
}

static void phase6_prepare_inner(const struct ggml_tensor * tensor, const struct ggml_tensor * ids) {
    if (phase6_zc_enabled()) {
        phase6_prepare_zc(tensor, ids);
        return;
    }
    if (phase6_async_enabled()) {
        phase6_prepare_async(tensor, ids);
        return;
    }
    for (int64_t row = 0; row < ids->ne[1]; ++row) {
        for (int64_t i = 0; i < ids->ne[0]; ++i) {
            const int expert = *(const int32_t *) ((const char *) ids->data + row * ids->nb[1] + i * ids->nb[0]);
            if (expert < 0 || expert >= 256) phase6_fail("invalid native expert ID");
            phase6_load(phase6_layer(tensor->name), expert);
        }
    }
}

static void phase6_prepare(const struct ggml_tensor * tensor, const struct ggml_tensor * ids) {
    if (!phase6_enabled()) return;
    const int layer = phase6_layer(tensor->name);
    if (layer < 0 || layer >= 40 || ids == NULL || ids->type != GGML_TYPE_I32) {
        phase6_fail("invalid routed node metadata");
    }
    // JOIN4: mode + in-node fetch timing (subtracted from expert nodes).
    phase6_decode_mode = (ids->ne[1] == 1);
    const uint64_t t0 = phase6_now_ns();
    phase6_prepare_inner(tensor, ids);
    const uint64_t dt = phase6_now_ns() - t0;
    if (join4_profile_enabled()) {
        if (phase6_decode_mode) join4_dec_fetchprep_ns += dt;
        else join4_pre_fetchprep_ns += dt;
    }
}

static const char * phase6_tensor_ptr(const struct ggml_tensor * tensor, int expert) {
    const int layer = phase6_layer(tensor->name);
    const int kind = phase6_kind(tensor->name);
    if (layer < 0 || kind < 0 || !phase6_records[layer][kind].valid) return NULL;
    const int slot = phase6_find_slot(layer, expert);
    if (slot < 0) phase6_fail("selected expert was not prepared");
    if (phase6_zc_enabled()) {
        // Challenger: consume the exact registered file bytes directly.
        // Same bytes the control pread-copies; no slot indirection.
        const struct phase6_record * record = &phase6_records[layer][kind];
        return (const char *) phase6_zc_file_map + record->base_offset
            + (size_t) expert * record->slice_bytes;
    }
    if (phase6_async_enabled()) {
        const int want = 1 << kind;
        const uint64_t start = phase6_now_ns();
        while ((__atomic_load_n(&phase6_slots[slot].ready_mask, __ATOMIC_ACQUIRE) & want) == 0) {
#if defined(__linux__)
            sched_yield();
#endif
        }
        const uint64_t waited = phase6_now_ns() - start;
        if (waited != 0) {
            __atomic_fetch_add(&phase6_ready_wait_ns, waited, __ATOMIC_RELAXED);
            __atomic_fetch_add(&phase6_ready_wait_events, 1, __ATOMIC_RELAXED);
        }
    }
    size_t in_slot = 0;
    for (int k = 0; k < kind; ++k) in_slot += phase6_records[layer][k].slice_bytes;
    return (const char *) phase6_storage + (size_t) slot * phase6_slot_bytes + in_slot;
}

// Keep the native IQP selected-expert path enabled in bounded mode. The
// storage hook redirects only its source plane; decode and arithmetic remain
// the exact control implementation.
const char * ggml_cpu_phase6_iqp_source(const struct ggml_tensor * tensor, int64_t expert) {
    if (!phase6_enabled()) return NULL;
    return phase6_tensor_ptr(tensor, (int) expert);
}

// JOIN4 section machine: cb-name markers (substring; names are "cb-il").
enum {
    JOIN4_M_NONE = 0, JOIN4_M_OPEN_ATTN, JOIN4_M_CLOSE_ATTN, JOIN4_M_CLOSE_GDN,
    JOIN4_M_OPEN_MOE, JOIN4_M_CLOSE_MOE, JOIN4_M_OPEN_SHARED, JOIN4_M_CLOSE_SHARED,
    JOIN4_M_OPEN_LMHEAD, JOIN4_M_CLOSE_LMHEAD, JOIN4_M_TAIL
};

static int join4_marker(const char * name) {
    if (name == NULL) return JOIN4_M_NONE;
    if (strstr(name, "linear_attn_out")) return JOIN4_M_CLOSE_GDN;
    if (strstr(name, "attn_output")) return JOIN4_M_CLOSE_ATTN;
    if (strstr(name, "attn_post_norm")) return JOIN4_M_OPEN_MOE;
    if (strstr(name, "attn_norm")) return JOIN4_M_OPEN_ATTN;
    if (strstr(name, "ffn_moe_out")) return JOIN4_M_CLOSE_MOE;
    if (strstr(name, "ffn_shexp_gated")) return JOIN4_M_CLOSE_SHARED;
    if (strstr(name, "ffn_shexp")) return JOIN4_M_OPEN_SHARED;
    if (strstr(name, "result_output")) return JOIN4_M_CLOSE_LMHEAD;
    if (strstr(name, "result_norm")) return JOIN4_M_OPEN_LMHEAD;
    if (strstr(name, "ffn_out")) return JOIN4_M_TAIL;
    if (strstr(name, "l_out")) return JOIN4_M_TAIL;
    return JOIN4_M_NONE;
}

// Resolve the ATTN_TENT scratch into ATTN (to_gdn=0) or GDN (to_gdn=1).
static void join4_flush_attn_scratch(int to_gdn) {
    if (join4_decode_mode) {
        if (to_gdn) join4_dec_gdn_ns += join4_scratch_dec_ns;
        else join4_dec_attn_ns += join4_scratch_dec_ns;
        join4_scratch_dec_ns = 0;
    } else {
        if (to_gdn) join4_pre_gdn_ns += join4_scratch_pre_ns;
        else join4_pre_attn_ns += join4_scratch_pre_ns;
        join4_scratch_pre_ns = 0;
    }
}

// Attribute one node's wall time (decode/prefill by current mode).
// resolve: 0=none, 1=attn, 2=gdn (section closers bypass scratch).
// wb: weight bucket (-1 = not a matmul; matmuls counted in BOTH).
static void join4_accum(int bucket, int expert, int resolve, int wb,
                        uint64_t dt) {
    if (join4_decode_mode) {
        if (expert) { join4_dec_expert_ns += dt; join4_dec_expert_cnt++; }
        else if (resolve == 1) join4_dec_attn_ns += dt;
        else if (resolve == 2) join4_dec_gdn_ns += dt;
        else if (bucket == JOIN4_SEC_ATTN_TENT) join4_scratch_dec_ns += dt;
        else { join4_dec_ns[bucket] += dt; join4_dec_cnt[bucket]++; }
        if (wb >= 0) { join4_dec_wb[wb] += dt; join4_dec_wb_cnt[wb]++; }
    } else {
        if (expert) { join4_pre_expert_ns += dt; join4_pre_expert_cnt++; }
        else if (resolve == 1) join4_pre_attn_ns += dt;
        else if (resolve == 2) join4_pre_gdn_ns += dt;
        else if (bucket == JOIN4_SEC_ATTN_TENT) join4_scratch_pre_ns += dt;
        else { join4_pre_ns[bucket] += dt; join4_pre_cnt[bucket]++; }
        if (wb >= 0) { join4_pre_wb[wb] += dt; join4_pre_wb_cnt[wb]++; }
    }
}

// Force-close any open section (counts surprises; scratch defaults to ATTN).
static void join4_force_close(void) {
    if (join4_section == JOIN4_SEC_ATTN_TENT) join4_flush_attn_scratch(0);
    if (join4_section != JOIN4_SEC_MISC) join4_force_closes++;
    join4_section = JOIN4_SEC_MISC;
}

// Called on thread 0 at each graph node's top (PROFILE only). Flushes the
// previous pending node ([t0_prev, now) into its bucket: includes the
// barrier wait, which IS node wall on thread 0), processes markers, sets
// decode mode at expert nodes, and arms the pending record.
static void join4_node_start(const struct ggml_tensor * node, int node_n) {
    // Resident arm never runs phase6_init: ensure the atexit report here.
    if (!phase6_report_registered) {
        phase6_report_registered = 1;
        atexit(phase6_report);
    }
    const uint64_t now = phase6_now_ns();
    if (join4_pending_valid) {
        join4_accum(join4_pending_bucket, join4_pending_expert,
                    join4_pending_resolve, join4_pending_wb,
                    now - join4_pending_t0);
        join4_pending_valid = 0;
        join4_pending_resolve = 0;
        join4_pending_wb = -1;
    }
    // Nodelist ground truth (first graph only, env-gated).
    if (!join4_nodelist_done) {
        if (join4_nodelist_fp == NULL) {
            const char * nlp = getenv("GGML_PHASE6_NODELIST");
            if (nlp != NULL && nlp[0] != '\0') {
                join4_nodelist_fp = fopen(nlp, "w");
            } else {
                join4_nodelist_done = 1;
            }
        }
        if (join4_nodelist_fp != NULL && node != NULL) {
            const char * s0 = (node->src[0] != NULL) ? node->src[0]->name : NULL;
            fprintf(join4_nodelist_fp, "%d op=%d name=%s src0=%s\n", node_n,
                    (int) node->op, node->name ? node->name : "-",
                    (s0 && s0[0]) ? s0 : "-");
        }
    }
    if (node_n == 0) {
        join4_graph_moe_nodes = 0;
        if (!join4_started) {
            join4_started = 1;
            join4_prefill_start_ns = now;
        }
    }
    const char * name = (node != NULL) ? node->name : NULL;
    const int m = join4_marker(name);
    if (m != JOIN4_M_NONE) join4_markers_seen++;
    int expert = 0;
    // Expert nodes: MUL_MAT_ID on _exps weights (time -> EXPERT; the
    // in-node fetch_prep is subtracted at report; markers still apply).
    if (node != NULL && node->op == GGML_OP_MUL_MAT_ID && node->src[0] != NULL &&
        node->src[0]->name != NULL && strstr(node->src[0]->name, "_exps") != NULL) {
        expert = 1;
        const struct ggml_tensor * ids = node->src[2];
        if (ids != NULL && ids->type == GGML_TYPE_I32) {
            join4_decode_mode = (ids->ne[1] == 1);
            join4_graph_moe_nodes++;
        }
    }
    int bucket = join4_section;
    int resolve = 0;
    switch (m) {
        case JOIN4_M_OPEN_ATTN:
            join4_force_close();
            join4_section = JOIN4_SEC_ATTN_TENT;
            bucket = JOIN4_SEC_ATTN_TENT;
            break;
        case JOIN4_M_CLOSE_ATTN:
            // Closer attributed directly to ATTN; previous scratch moves too.
            join4_flush_attn_scratch(0);
            resolve = 1;
            join4_section = JOIN4_SEC_MISC;
            break;
        case JOIN4_M_CLOSE_GDN:
            join4_flush_attn_scratch(1);
            resolve = 2;
            join4_section = JOIN4_SEC_MISC;
            break;
        case JOIN4_M_OPEN_MOE:
            join4_force_close();
            join4_section = JOIN4_SEC_MOE;
            bucket = JOIN4_SEC_MOE;
            break;
        case JOIN4_M_CLOSE_MOE:
            if (join4_section != JOIN4_SEC_MOE && !expert) join4_force_closes++;
            bucket = JOIN4_SEC_MOE;
            join4_section = JOIN4_SEC_MISC;
            break;
        case JOIN4_M_OPEN_SHARED:
            join4_force_close();
            join4_section = JOIN4_SEC_SHARED;
            bucket = JOIN4_SEC_SHARED;
            break;
        case JOIN4_M_CLOSE_SHARED:
            bucket = JOIN4_SEC_SHARED;
            join4_section = JOIN4_SEC_MISC;
            break;
        case JOIN4_M_OPEN_LMHEAD:
            join4_force_close();
            join4_section = JOIN4_SEC_LMHEAD;
            bucket = JOIN4_SEC_LMHEAD;
            break;
        case JOIN4_M_CLOSE_LMHEAD:
            bucket = JOIN4_SEC_LMHEAD;
            join4_section = JOIN4_SEC_MISC;
            break;
        case JOIN4_M_TAIL:
            join4_force_close();
            bucket = JOIN4_SEC_MISC;
            break;
        default:
            break;
    }
    if (expert) {
        join4_pending_bucket = bucket; // unused for expert, kept for debug
        join4_pending_expert = 1;
    } else {
        join4_pending_bucket = bucket;
        join4_pending_expert = 0;
    }
    join4_pending_resolve = expert ? 0 : resolve;
    // Weight bucket for matmuls (both attributions run).
    join4_pending_wb = -1;
    if (node != NULL && (node->op == GGML_OP_MUL_MAT ||
                         node->op == GGML_OP_MUL_MAT_ID) &&
        node->src[0] != NULL) {
        const char * wname = node->src[0]->name;
        join4_pending_wb = join4_wbucket(wname);
        if (join4_pending_wb == JOIN4_WB_OTHER &&
            join4_wb_unmatched_logged < 20 && wname != NULL && wname[0]) {
            join4_wb_unmatched_logged++;
            fprintf(stderr, "PHASE6_WB_UNMATCHED %s\n", wname);
        }
    }
    join4_pending_t0 = now;
    join4_pending_valid = 1;
}

// Called on thread 0 at graph loop end (PROFILE only). Flushes the last
// node, counts decode/prefill graphs, stamps TTFT at first decode end.
static void join4_graph_end(void) {
    const uint64_t now = phase6_now_ns();
    if (join4_pending_valid) {
        join4_accum(join4_pending_bucket, join4_pending_expert,
                    join4_pending_resolve, join4_pending_wb,
                    now - join4_pending_t0);
        join4_pending_valid = 0;
        join4_pending_resolve = 0;
        join4_pending_wb = -1;
    }
    if (join4_nodelist_fp != NULL) {
        fclose(join4_nodelist_fp);
        join4_nodelist_fp = NULL;
        join4_nodelist_done = 1;
    }
    // Hygiene: no section may span graphs (counts surprises, no loss:
    // every node was already attributed; only scratch re-homes here).
    if (join4_section == JOIN4_SEC_ATTN_TENT) join4_flush_attn_scratch(0);
    if (join4_section != JOIN4_SEC_MISC) join4_force_closes++;
    join4_section = JOIN4_SEC_MISC;
    if (join4_graph_moe_nodes > 0) {
        if (join4_decode_mode) {
            join4_decode_graphs++;
            if (join4_decode_graphs == 1 && join4_started) {
                join4_ttft_ns = now - join4_prefill_start_ns;
            }
        } else {
            join4_prefill_graphs++;
        }
    }
}

static void ggml_phase6_route_trace(const struct ggml_compute_params * params,
                                    const struct ggml_tensor * tensor) {
    if (params->ith != 0 || tensor->op != GGML_OP_MUL_MAT_ID) return;
    const char * path = getenv("GGML_PHASE6_ROUTE_TRACE");
    if (path == NULL || path[0] == '\0') return;
    const struct ggml_tensor * weights = tensor->src[0];
    const struct ggml_tensor * ids = tensor->src[2];
    if (weights == NULL || ids == NULL || ids->type != GGML_TYPE_I32 ||
            strstr(weights->name, "ffn_") == NULL || strstr(weights->name, "_exps") == NULL) return;
    static FILE * fp = NULL;
    static uint64_t event_id = 0;
    if (fp == NULL) {
        fp = fopen(path, "a");
        if (fp == NULL) return;
        setvbuf(fp, NULL, _IOLBF, 0);
    }
    const int64_t n = ggml_nelements(ids);
    const int32_t * values = (const int32_t *) ids->data;
    fprintf(fp, "{\"event\":%" PRIu64 ",\"weight\":\"%s\",\"shape\":[%" PRId64 ",%" PRId64 "],\"ids\":[",
        event_id++, weights->name, ids->ne[0], ids->ne[1]);
    for (int64_t i = 0; i < n; ++i) fprintf(fp, "%s%d", i ? "," : "", values[i]);
    fprintf(fp, "]}\n");
}
