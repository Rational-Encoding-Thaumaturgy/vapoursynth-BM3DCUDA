/*
 * VapourSynth wrapper for BM3DCUDA
 * Copyright (c) 2021 WolframRhodium
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <VapourSynth4.h>
#include <VSHelper4.h>

using namespace std::string_literals;

extern cudaGraphExec_t get_graphexec(
    float * d_res, float * d_src, float * h_res,
    int width, int height, int stride,
    float sigma, int block_step, int bm_range,
    int radius, int ps_num, int ps_range,
    bool chroma, float sigma_u, float sigma_v,
    bool final_, float extractor
) noexcept;

#define checkError(expr) do {                                            \
    if (cudaError_t result = expr; result != cudaSuccess) [[unlikely]] { \
        const char * error_str = cudaGetErrorString(result);             \
        return set_error("'"s + # expr + "' failed: " + error_str);      \
    }                                                                    \
} while(0)

constexpr int kFast = 4;

static VSPlugin * myself = nullptr;

struct ticket_semaphore {
    std::atomic<intptr_t> ticket {};
    std::atomic<intptr_t> current {};

    void acquire() noexcept {
        intptr_t tk { ticket.fetch_add(1, std::memory_order::acquire) };
        while (true) {
            intptr_t curr { current.load(std::memory_order::acquire) };
            if (tk <= curr) {
                return;
            }
            current.wait(curr, std::memory_order::relaxed);
        }
    }

    void release() noexcept {
        current.fetch_add(1, std::memory_order::release);
        current.notify_all();
    }
};

template <typename T, auto deleter>
    requires
        std::default_initializable<T> &&
        std::is_trivially_copy_assignable_v<T> &&
        std::convertible_to<T, bool> &&
        std::invocable<decltype(deleter), T>
struct Resource {
    T data;

    [[nodiscard]] constexpr Resource() noexcept = default;

    [[nodiscard]] constexpr Resource(T x) noexcept : data(x) {}

    [[nodiscard]] constexpr Resource(Resource&& other) noexcept
        : data(std::exchange(other.data, T{}))
    { }

    constexpr Resource& operator=(Resource&& other) noexcept {
        if (this == &other) return *this;
        deleter_(data);
        data = std::exchange(other.data, T{});
        return *this;
    }

    Resource operator=(Resource other) = delete;

    Resource(const Resource& other) = delete;

    constexpr operator T() const noexcept {
        return data;
    }

    constexpr auto deleter_(T x) noexcept {
        if (x) {
            deleter(x);
        }
    }

    constexpr Resource& operator=(T x) noexcept {
        deleter_(data);
        data = x;
        return *this;
    }

    constexpr ~Resource() noexcept {
        deleter_(data);
    }
};

struct CUDA_Resource {
    Resource<float *, cudaFree> d_src;
    Resource<float *, cudaFree> d_res;
    Resource<float *, cudaFreeHost> h_res;
    Resource<cudaStream_t, cudaStreamDestroy> stream;
    std::array<Resource<cudaGraphExec_t, cudaGraphExecDestroy>, 3> graphexecs;
};

struct BM3DData {
    VSNode * node;
    VSNode * ref_node;
    const VSVideoInfo * vi;

    // stored in graphexec:
    // float sigma[3];
    // int block_step[3];
    // int bm_range[3];
    // int ps_num[3];
    // int ps_range[3];
    // float extractor;

    int radius;
    int num_copy_engines; // fast
    bool chroma;
    bool process[3]; // sigma != 0
    bool final_;
    bool zero_init;

    int d_pitch;
    int device_id;

    ticket_semaphore semaphore;
    std::vector<CUDA_Resource> resources;
    std::mutex resources_lock;
};

static inline void Aggregation(
    float * VS_RESTRICT dstp, int dst_stride,
    const float * VS_RESTRICT srcp, int src_stride,
    int width, int height
) noexcept {

    const float * wdst = srcp;
    const float * weight = &srcp[height * src_stride];

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            dstp[x] = wdst[x] / weight[x];
        }

        dstp += dst_stride;
        wdst += src_stride;
        weight += src_stride;
    }
}

static const VSFrame *VS_CC BM3DGetFrame(
    int n, int activationReason, void *instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d = static_cast<BM3DData *>(instanceData);

    if (activationReason == arInitial) {
        int start_frame = std::max(n - d->radius, 0);
        int end_frame = std::min(n + d->radius, d->vi->numFrames - 1);

        for (int i = start_frame; i <= end_frame; ++i) {
            vsapi->requestFrameFilter(i, d->node, frameCtx);
        }
        if (d->final_) {
            for (int i = start_frame; i <= end_frame; ++i) {
                vsapi->requestFrameFilter(i, d->ref_node, frameCtx);
            }
        }
    } else if (activationReason == arAllFramesReady) {
        if (auto error = cudaSetDevice(d->device_id); error != cudaSuccess) {
            vsapi->setFilterError(
                ("BM3D: "s + cudaGetErrorString(error)).c_str(),
                frameCtx
            );
            return nullptr;
        }

        int radius = d->radius;
        int temporal_width = 2 * radius + 1;
        bool final_ = d->final_;
        int num_input_frames = temporal_width * (final_ ? 2 : 1); // including ref

        using freeFrame_t = decltype(vsapi->freeFrame);
        const std::vector srcs = [&](){
            std::vector<std::unique_ptr<const VSFrame, const freeFrame_t &>> temp;

            temp.reserve(num_input_frames);

            if (final_) {
                for (int i = -radius; i <= radius; ++i) {
                    int clamped_n = std::clamp(n + i, 0, d->vi->numFrames - 1);
                    temp.emplace_back(
                        vsapi->getFrameFilter(clamped_n, d->ref_node, frameCtx),
                        vsapi->freeFrame
                    );
                }
            }

            for (int i = -radius; i <= radius; ++i) {
                int clamped_n = std::clamp(n + i, 0, d->vi->numFrames - 1);
                temp.emplace_back(
                    vsapi->getFrameFilter(clamped_n, d->node, frameCtx),
                    vsapi->freeFrame
                );
            }

            return temp;
        }();

        const VSFrame * src = srcs[radius + (final_ ? temporal_width : 0)].get();

        std::unique_ptr<VSFrame, const freeFrame_t &> dst { nullptr, vsapi->freeFrame };
        if (radius) {
            dst.reset(
                vsapi->newVideoFrame(
                    &d->vi->format, d->vi->width,
                    d->vi->height * 2 * temporal_width,
                    src, core)
            );
            for (int i = 0; i < d->vi->format.numPlanes; ++i) {
                if (d->zero_init && !d->process[i]) {
                    auto ptr = vsapi->getWritePtr(dst.get(), i);
                    auto height = vsapi->getFrameHeight(dst.get(), i);
                    auto pitch = vsapi->getStride(dst.get(), i);
                    memset(ptr, 0, height * pitch);
                }
            }
        } else {
            const VSFrame * fr[] = {
                d->process[0] ? nullptr : src,
                d->process[1] ? nullptr : src,
                d->process[2] ? nullptr : src
            };
            const int pl[] = { 0, 1, 2 };

            dst.reset(
                vsapi->newVideoFrame2(
                    &d->vi->format, d->vi->width,
                    d->vi->height, fr, pl, src, core)
            );
        }

        d->semaphore.acquire();
        d->resources_lock.lock();
        auto resource = std::move(d->resources.back());
        d->resources.pop_back();
        d->resources_lock.unlock();

        const auto set_error = [&](const std::string & error_message) {
            d->resources_lock.lock();
            d->resources.push_back(std::move(resource));
            d->resources_lock.unlock();
            d->semaphore.release();

            vsapi->setFilterError(("BM3D: " + error_message).c_str(), frameCtx);

            return nullptr;
        };

        float * const h_res = resource.h_res;
        cudaStream_t stream = resource.stream;
        int d_pitch = d->d_pitch;
        int d_stride = d_pitch / sizeof(float);

        if (d->chroma) {
            int width = vsapi->getFrameWidth(src, 0);
            int height = vsapi->getFrameHeight(src, 0);
            int s_pitch = vsapi->getStride(src, 0);
            int s_stride = s_pitch / sizeof(float);
            int width_bytes = width * sizeof(float);

            cudaGraphExec_t graphexec = resource.graphexecs[0];

            float * h_src = h_res;
            for (int outer = 0; outer < (final_ ? 2 : 1); ++outer) {
                for (int i = 0; i < std::ssize(d->process); ++i) {
                    for (int j = 0; j < temporal_width; ++j) {
                        if (i == 0 || d->process[i]) {
                            auto current_src = srcs[j + outer * temporal_width].get();

                            vsh::bitblt(
                                h_src, d_pitch,
                                vsapi->getReadPtr(current_src, i), s_pitch,
                                width_bytes, height
                            );
                        }
                        h_src += d_stride * height;
                    }
                }
            }

            checkError(cudaGraphLaunch(graphexec, stream));

            checkError(cudaStreamSynchronize(stream));

            float * h_dst = h_res;
            for (int plane = 0; plane < std::ssize(d->process); ++plane) {
                if (!d->process[plane]) {
                    h_dst += d_stride * height * 2 * temporal_width;
                    continue;
                }

                float * dstp = reinterpret_cast<float *>(
                    vsapi->getWritePtr(dst.get(), plane));

                if (radius) {
                    vsh::bitblt(
                        dstp, s_pitch, h_dst, d_pitch,
                        width_bytes, height * 2 * temporal_width
                    );
                } else {
                    Aggregation(
                        dstp, s_stride,
                        h_dst, d_stride,
                        width, height
                    );
                }

                h_dst += d_stride * height * 2 * temporal_width;
            }
        } else { // !d->chroma
            for (int plane = 0; plane < d->vi->format.numPlanes; plane++) {
                if (!d->process[plane]) {
                    continue;
                }

                int width = vsapi->getFrameWidth(src, plane);
                int height = vsapi->getFrameHeight(src, plane);
                int s_pitch = vsapi->getStride(src, plane);
                int s_stride = s_pitch / sizeof(float);
                int width_bytes = width * sizeof(float);

                cudaGraphExec_t graphexec = resource.graphexecs[plane];

                float * h_src = h_res;
                for (int i = 0; i < num_input_frames; ++i) {
                    vsh::bitblt(
                        h_src, d_pitch,
                        vsapi->getReadPtr(srcs[i].get(), plane), s_pitch,
                        width_bytes, height
                    );
                    h_src += d_stride * height;
                }

                checkError(cudaGraphLaunch(graphexec, stream));

                checkError(cudaStreamSynchronize(stream));

                float * dstp = reinterpret_cast<float *>(
                    vsapi->getWritePtr(dst.get(), plane));

                if (radius) {
                    vsh::bitblt(
                        dstp, s_pitch, h_res, d_pitch,
                        width_bytes, height * 2 * temporal_width
                    );
                } else {
                    Aggregation(
                        dstp, s_stride,
                        h_res, d_stride,
                        width, height
                    );
                }
            }
        }

        d->resources_lock.lock();
        d->resources.push_back(std::move(resource));
        d->resources_lock.unlock();
        d->semaphore.release();

        if (radius) {
            VSMap * dst_prop { vsapi->getFramePropertiesRW(dst.get()) };

            vsapi->mapSetInt(dst_prop, "BM3D_V_radius", d->radius, maReplace);

            int64_t process[3] { d->process[0], d->process[1], d->process[2] };
            vsapi->mapSetIntArray(dst_prop, "BM3D_V_process", process, 3);
        }

        return dst.release();
    }

    return nullptr;
}

static void VS_CC BM3DFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d = static_cast<BM3DData *>(instanceData);

    vsapi->freeNode(d->node);
    vsapi->freeNode(d->ref_node);

    cudaSetDevice(d->device_id);

    delete d;
}

static void VS_CC BM3DCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d { std::make_unique<BM3DData>() };

    const auto set_error = [&](const std::string & error_message) {
        vsapi->mapSetError(out, ("BM3D: " + error_message).c_str());
        vsapi->freeNode(d->node);
        vsapi->freeNode(d->ref_node);
    };

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);
    const int width = d->vi->width;
    const int height = d->vi->height;
    const int bits_per_sample = d->vi->format.bitsPerSample;

    if (
        !vsh::isConstantVideoFormat(d->vi) || d->vi->format.sampleType == stInteger ||
        (d->vi->format.sampleType == stFloat && bits_per_sample != 32)) {
        return set_error("only constant format 32bit float input supported");
    }

    int error;

    d->ref_node = vsapi->mapGetNode(in, "ref", 0, &error);
    bool final_;
    if (error) {
        d->ref_node = nullptr;
        final_ = false;
    } else {
        const auto * ref_vi = vsapi->getVideoInfo(d->ref_node);
        if (!vsh::isSameVideoFormat(&ref_vi->format, &d->vi->format)) {
            return set_error("\"ref\" must be of the same format as \"clip\"");
        } else if (ref_vi->width != width || ref_vi->height != height ) {
            return set_error("\"ref\" must be of the same dimensions as \"clip\"");
        } else if (ref_vi->numFrames != d->vi->numFrames) {
            return set_error("\"ref\" must be of the same number of frames as \"clip\"");
        }

        final_ = true;
    }
    d->final_ = final_;

    float sigma[3];
    for (int i = 0; i < std::ssize(sigma); ++i) {
        sigma[i] = static_cast<float>(
            vsapi->mapGetFloat(in, "sigma", i, &error));

        if (error) {
            sigma[i] = (i == 0) ? 3.0f : sigma[i - 1];
        } else if (sigma[i] < 0.0f) {
            return set_error("\"sigma\" must be non-negative");
        }

        if (sigma[i] < std::numeric_limits<float>::epsilon()) {
            d->process[i] = false;
        } else {
            d->process[i] = true;
        }
    }
    for (int i = 0; i < std::ssize(sigma); ++i) {
        // assumes grayscale input, hard_thr = 2.7
        sigma[i] *= (3.0f / 4.0f) / 255.0f * 64.0f * (final_ ? 1.0f : 2.7f);
    }

    int block_step[3];
    for (int i = 0; i < std::ssize(block_step); ++i) {
        block_step[i] = vsh::int64ToIntS(
            vsapi->mapGetInt(in, "block_step", i, &error));

        if (error) {
            block_step[i] = (i == 0) ? 8 : block_step[i - 1];
        } else if (block_step[i] <= 0 || block_step[i] > 8) {
            return set_error("\"block_step\" must be in range [1, 8]");
        }
    }

    int bm_range[3];
    for (int i = 0; i < std::ssize(bm_range); ++i) {
        bm_range[i] = vsh::int64ToIntS(
            vsapi->mapGetInt(in, "bm_range", i, &error));

        if (error) {
            bm_range[i] = (i == 0) ? 9 : bm_range[i - 1];
        } else if (bm_range[i] <= 0) {
            return set_error("\"bm_range\" must be positive");
        }
    }

    const int radius = [&](){
        int temp = vsh::int64ToIntS(vsapi->mapGetInt(in, "radius", 0, &error));
        if (error) {
            return 0;
        }
        return temp;
    }();
    if (radius < 0) {
        return set_error("\"radius\" must be non-negative");
    }
    d->radius = radius;

    int ps_num[3];
    for (int i = 0; i < std::ssize(ps_num); ++i) {
        ps_num[i] = vsh::int64ToIntS(
            vsapi->mapGetInt(in, "ps_num", i, &error));

        if (error) {
            ps_num[i] = (i == 0) ? 2 : ps_num[i - 1];
        } else if (ps_num[i] <= 0 || ps_num[i] > 8) {
            return set_error("\"ps_num\" must be in range [1, 8]");
        }
    }

    int ps_range[3];
    for (int i = 0; i < std::ssize(ps_range); ++i) {
        ps_range[i] = vsh::int64ToIntS(
            vsapi->mapGetInt(in, "ps_range", i, &error));

        if (error) {
            ps_range[i] = (i == 0) ? 4 : ps_range[i - 1];
        } else if (ps_range[i] <= 0) {
            return set_error("\"ps_range\" must be positive");
        }
    }

    const bool chroma = [&](){
        bool temp = !!vsapi->mapGetInt(in, "chroma", 0, &error);
        if (error) {
            return false;
        }
        return temp;
    }();
    if (chroma && !vsh::isSameVideoPresetFormat(pfYUV444PS, &d->vi->format, core, vsapi)) {
        return set_error("clip format must be YUV444 when \"chroma\" is true");
    }
    d->chroma = chroma;

    const int device_id = [&](){
        int temp = vsh::int64ToIntS(vsapi->mapGetInt(in, "device_id", 0, &error));
        if (error) {
            return 0;
        }
        return temp;
    }();
    int device_count;
    checkError(cudaGetDeviceCount(&device_count));
    if (0 <= device_id && device_id < device_count) {
        checkError(cudaSetDevice(device_id));
    } else {
        return set_error("invalid device ID (" + std::to_string(device_id) + ")");
    }
    d->device_id = device_id;

    const bool fast = [&](){
        bool temp = !!vsapi->mapGetInt(in, "fast", 0, &error);
        if (error) {
            return true;
        }
        return temp;
    }();
    const int num_copy_engines { fast ? kFast : 1 };
    d->num_copy_engines = num_copy_engines;

    const float extractor = [&](){
        int temp = vsh::int64ToIntS(vsapi->mapGetInt(in, "extractor_exp", 0, &error));
        if (error) {
            return 0.0f;
        }
        return (temp ? std::ldexp(1.0f, temp) : 0.0f);
    }();

    d->zero_init = !!vsapi->mapGetInt(in, "zero_init", 0, &error);
    if (error) {
        d->zero_init = true;
    }

    // GPU resource allocation
    {
        d->semaphore.current.store(num_copy_engines - 1, std::memory_order::relaxed);

        d->resources.reserve(num_copy_engines);

        int max_width { d->process[0] ? width : width >> d->vi->format.subSamplingW };
        int max_height { d->process[0] ? height : height >> d->vi->format.subSamplingH };

        int num_planes { chroma ? 3 : 1 };
        int temporal_width = 2 * radius + 1;
        size_t d_pitch;
        int d_stride;
        for (int i = 0; i < num_copy_engines; ++i) {
            Resource<float *, cudaFree> d_src {};
            if (i == 0) {
                checkError(cudaMallocPitch(
                    &d_src.data, &d_pitch, max_width * sizeof(float),
                    (final_ ? 2 : 1) * num_planes * temporal_width * max_height));
                d_stride = static_cast<int>(d_pitch / sizeof(float));
                d->d_pitch = static_cast<int>(d_pitch);
            } else {
                checkError(cudaMalloc(&d_src.data,
                    (final_ ? 2 : 1) * num_planes * temporal_width * max_height * d_pitch));
            }

            Resource<float *, cudaFree> d_res {};
            checkError(cudaMalloc(&d_res.data,
                num_planes * temporal_width * 2 * max_height * d_pitch));

            Resource<float *, cudaFreeHost> h_res {};
            checkError(cudaMallocHost(&h_res.data,
                num_planes * temporal_width * 2 * max_height * d_pitch));

            Resource<cudaStream_t, cudaStreamDestroy> stream {};
            checkError(cudaStreamCreateWithFlags(&stream.data,
                cudaStreamNonBlocking));

            std::array<Resource<cudaGraphExec_t, cudaGraphExecDestroy>, 3> graphexecs {};
            if (d->chroma) {
                graphexecs[0] = get_graphexec(
                    d_res, d_src, h_res,
                    width, height, d_stride,
                    sigma[0], block_step[0], bm_range[0],
                    radius, ps_num[0], ps_range[0],
                    true, sigma[1], sigma[2],
                    final_, extractor
                );
            } else {
                auto subsamplingW = d->vi->format.subSamplingW;
                auto subsamplingH = d->vi->format.subSamplingH;

                for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
                    if (d->process[plane]) {
                        int plane_width { plane == 0 ? width : width >> subsamplingW };
                        int plane_height { plane == 0 ? height : height >> subsamplingH };

                        graphexecs[plane] = get_graphexec(
                            d_res, d_src, h_res,
                            plane_width, plane_height, d_stride,
                            sigma[plane], block_step[plane], bm_range[plane],
                            radius, ps_num[plane], ps_range[plane],
                            false, 0.0f, 0.0f,
                            final_, extractor
                        );
                    }
                }
            }

            d->resources.push_back(CUDA_Resource{
                .d_src = std::move(d_src),
                .d_res = std::move(d_res),
                .h_res = std::move(h_res),
                .stream = std::move(stream),
                .graphexecs = std::move(graphexecs)
            });
        }
    }

    VSVideoInfo vi = *d->vi;
    
    if (radius)
        vi.height *= 2 * (2 * d->radius + 1);

    std::vector<VSFilterDependency> deps = {
        {d->node, rpGeneral},
    };

    if (d->ref_node)
        deps.push_back({d->ref_node, rpGeneral});

    vsapi->createVideoFilter(
        out, "BM3D", &vi, BM3DGetFrame, BM3DFree,
        fmParallel, deps.data(), deps.size(), d.release(), core);
}

struct VAggregateData {
    VSNode * node;

    VSNode * src_node;
    const VSVideoInfo * src_vi;

    std::array<bool, 3> process; // sigma != 0

    int radius;

    std::unordered_map<std::thread::id, float *> buffer;
    std::shared_mutex buffer_lock;
};

static const VSFrame *VS_CC VAggregateGetFrame(
    int n, int activationReason, void *instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) {

    auto * d = static_cast<VAggregateData *>(instanceData);

    if (activationReason == arInitial) {
        int start_frame = std::max(n - d->radius, 0);
        int end_frame = std::min(n + d->radius, d->src_vi->numFrames - 1);

        for (int i = start_frame; i <= end_frame; ++i) {
            vsapi->requestFrameFilter(i, d->node, frameCtx);
        }
        vsapi->requestFrameFilter(n, d->src_node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame * src_frame = vsapi->getFrameFilter(n, d->src_node, frameCtx);

        std::vector<const VSFrame *> vbm3d_frames;
        vbm3d_frames.reserve(2 * d->radius + 1);
        for (int i = n - d->radius; i <= n + d->radius; ++i) {
            auto frame_id = std::clamp(i, 0, d->src_vi->numFrames - 1);
            vbm3d_frames.emplace_back(vsapi->getFrameFilter(frame_id, d->node, frameCtx));
        }

        float * buffer {};
        {
            const auto thread_id = std::this_thread::get_id();
            bool init = true;

            d->buffer_lock.lock_shared();

            try {
                const auto & const_buffer = d->buffer;
                buffer = const_buffer.at(thread_id);
            } catch (const std::out_of_range &) {
                init = false;
            }

            d->buffer_lock.unlock_shared();

            if (!init) {
                assert(d->process[0] || d->src_vi->format.numPlanes > 1);

                const int max_width {
                    d->process[0] ?
                    vsapi->getFrameWidth(src_frame, 0) :
                    vsapi->getFrameWidth(src_frame, 1)
                };

                buffer = reinterpret_cast<float *>(std::malloc(2 * max_width * sizeof(float)));

                std::lock_guard _ { d->buffer_lock };
                d->buffer.emplace(thread_id, buffer);
            }
        }

        const VSFrame * fr[] {
            d->process[0] ? nullptr : src_frame,
            d->process[1] ? nullptr : src_frame,
            d->process[2] ? nullptr : src_frame
        };
        constexpr int pl[] { 0, 1, 2 };
        auto dst_frame = vsapi->newVideoFrame2(
            &d->src_vi->format,
            d->src_vi->width, d->src_vi->height,
            fr, pl, src_frame, core);

        for (int plane = 0; plane < d->src_vi->format.numPlanes; ++plane) {
            if (d->process[plane]) {
                int plane_width = vsapi->getFrameWidth(src_frame, plane);
                int plane_height = vsapi->getFrameHeight(src_frame, plane);
                int plane_stride = vsapi->getStride(src_frame, plane) / sizeof(float);

                std::vector<const float *> srcps;
                srcps.reserve(2 * d->radius + 1);
                for (int i = 0; i < 2 * d->radius + 1; ++i) {
                    srcps.emplace_back(reinterpret_cast<const float *>(vsapi->getReadPtr(vbm3d_frames[i], plane)));
                }

                auto dstp = reinterpret_cast<float *>(vsapi->getWritePtr(dst_frame, plane));

                for (int y = 0; y < plane_height; ++y) {
                    memset(buffer, 0, 2 * plane_width * sizeof(float));
                    for (int i = 0; i < 2 * d->radius + 1; ++i) {
                        const float * agg_src = srcps[i];
                        // bm3d.VAggregate implements zero padding in temporal dimension
                        // here we implements replication padding
                        agg_src += (
                            std::clamp(2 * d->radius - i, n - d->src_vi->numFrames + 1 + d->radius, n + d->radius)
                            * 2 * plane_height + y) * plane_stride;
                        for (int x = 0; x < plane_width; ++x) {
                            buffer[x] += agg_src[x];
                        }
                        agg_src += plane_height * plane_stride;
                        for (int x = 0; x < plane_width; ++x) {
                            buffer[plane_width + x] += agg_src[x];
                        }
                    }
                    for (int x = 0; x < plane_width; ++x) {
                        dstp[x] = buffer[x] / buffer[plane_width + x];
                    }
                    dstp += plane_stride;
                }
            }
        }

        for (const auto & frame : vbm3d_frames) {
            vsapi->freeFrame(frame);
        }
        vsapi->freeFrame(src_frame);

        return dst_frame;
    }

    return nullptr;
}

static void VS_CC VAggregateFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi
) noexcept {

    VAggregateData * d = static_cast<VAggregateData *>(instanceData);

    for (const auto & [_, ptr] : d->buffer) {
        std::free(ptr);
    }

    vsapi->freeNode(d->src_node);
    vsapi->freeNode(d->node);

    delete d;
}

static void VS_CC VAggregateCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) {

    auto d { std::make_unique<VAggregateData>() };

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    auto vi = vsapi->getVideoInfo(d->node);
    d->src_node = vsapi->mapGetNode(in, "src", 0, nullptr);
    d->src_vi = vsapi->getVideoInfo(d->src_node);

    d->radius = (vi->height / d->src_vi->height - 2) / 4;

    d->process.fill(false);
    int num_planes_args = vsapi->mapNumElements(in, "planes");
    for (int i = 0; i < num_planes_args; ++i) {
        int plane = vsapi->mapGetInt(in, "planes", i, nullptr);
        d->process[plane] = true;
    }

    VSCoreInfo core_info;
    vsapi->getCoreInfo(core, &core_info);
    d->buffer.reserve(core_info.numThreads);

    VSFilterDependency deps[] = {
        {d->node, rpGeneral},
        {d->src_node, rpGeneral},
    };

    vsapi->createVideoFilter(
        out, "VAggregate", d->src_vi, VAggregateGetFrame, VAggregateFree,
        fmParallel, deps, 2, d.release(), core);
}

static void VS_CC BM3Dv2Create(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) {

    std::array<bool, 3> process;
    process.fill(true);

    int num_sigma_args = vsapi->mapNumElements(in, "sigma");
    for (int i = 0; i < std::min(3, num_sigma_args); ++i) {
        auto sigma = vsapi->mapGetFloat(in, "sigma", i, nullptr);
        if (sigma < std::numeric_limits<float>::epsilon()) {
            process[i] = false;
        }
    }
    if (num_sigma_args > 0) { // num_sigma_args may be -1
        for (int i = num_sigma_args; i < 3; ++i) {
            process[i] = process[i - 1];
        }
    }

    bool skip = true;
    auto src = vsapi->mapGetNode(in, "clip", 0, nullptr);
    auto src_vi = vsapi->getVideoInfo(src);
    for (int i = 0; i < src_vi->format.numPlanes; ++i) {
        skip &= !process[i];
    }
    if (skip) {
        vsapi->mapSetNode(out, "clip", src, maReplace);
        vsapi->freeNode(src);
        return ;
    }

    auto map = vsapi->invoke(myself, "BM3D", in);
    if (auto error = vsapi->mapGetError(map); error) {
        vsapi->mapSetError(out, error);
        vsapi->freeMap(map);
        vsapi->freeNode(src);
        return ;
    }

    int error;
    int radius = vsapi->mapGetInt(in, "radius", 0, &error);
    if (error) {
        radius = 0;
    }
    if (radius == 0) {
        // spatial BM3D should handle everything itself
        auto node = vsapi->mapGetNode(map, "clip", 0, nullptr);
        vsapi->freeMap(map);
        vsapi->mapSetNode(out, "clip", node, maReplace);
        vsapi->freeNode(node);
        vsapi->freeNode(src);
        return ;
    }

    vsapi->mapSetNode(map, "src", src, maReplace);
    vsapi->freeNode(src);

    for (int i = 0; i < 3; ++i) {
        if (process[i]) {
            vsapi->mapSetInt(map, "planes", i, maAppend);
        }
    }

    auto map2 = vsapi->invoke(myself, "VAggregate", map);
    vsapi->freeMap(map);
    if (auto error = vsapi->mapGetError(map2); error) {
        vsapi->mapSetError(out, error);
        vsapi->freeMap(map2);
        return ;
    }

    auto node = vsapi->mapGetNode(map2, "clip", 0, nullptr);
    vsapi->freeMap(map2);
    vsapi->mapSetNode(out, "clip", node, maReplace);
    vsapi->freeNode(node);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {

    myself = plugin;

    vspapi->configPlugin(
        "com.wolframrhodium.bm3dcuda", "bm3dcuda",
        "BM3D algorithm implemented in CUDA",
        VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0, plugin);

    constexpr auto bm3d_args {
        "clip:vnode;"
        "ref:vnode:opt;"
        "sigma:float[]:opt;"
        "block_step:int[]:opt;"
        "bm_range:int[]:opt;"
        "radius:int:opt;"
        "ps_num:int[]:opt;"
        "ps_range:int[]:opt;"
        "chroma:int:opt;"
        "device_id:int:opt;"
        "fast:int:opt;"
        "extractor_exp:int:opt;"
        "zero_init:int:opt;"
    };

     vspapi->registerFunction("BM3D", bm3d_args, "clip:vnode;", BM3DCreate, nullptr, plugin);

     vspapi->registerFunction(
        "VAggregate",
        "clip:vnode;"
        "src:vnode;"
        "planes:int[];",
        "clip:vnode;",
        VAggregateCreate, nullptr, plugin);

     vspapi->registerFunction("BM3Dv2", bm3d_args, "clip:vnode;", BM3Dv2Create, nullptr, plugin);
}
