/*
 * Copyright (C) 2026 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// An OpenXR sample driving Filament's Vulkan backend through XR_KHR_vulkan_enable2.
//
// Filament renders both eyes in a single multiview pass straight into the OpenXR color and depth
// swapchain images: the XR runtime creates the images, a VulkanPlatform subclass hands them to the
// backend as the default render target, and the two-layer swapchain turns that into a multiview
// render pass.
//
// Intentionally free of filamentapp/SDL and of any windowing system so the same render path can be
// reused on Android.

#if defined(_WIN32)
#define XR_USE_PLATFORM_WIN32
#elif defined(__ANDROID__)
#define XR_USE_PLATFORM_ANDROID
#endif
#define XR_USE_GRAPHICS_API_VULKAN

// Brings in BlueVK, the OpenXR headers in the right order, and XRLOG.
#include "helloxr_features.h"

#include <backend/platforms/VulkanPlatform.h>

#include <filament/Camera.h>
#include <filament/Color.h>
#include <filament/Engine.h>
#include <filament/Fence.h>
#include <filament/IndirectLight.h>
#include <filament/LightManager.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/Skybox.h>
#include <filament/SwapChain.h>
#include <filament/TransformManager.h>
#include <filament/View.h>
#include <filament/Viewport.h>

#include <filameshio/MeshReader.h>

#include <ktxreader/Ktx1Reader.h>

#include <utils/Entity.h>
#include <utils/EntityManager.h>

#include <math/mat3.h>
#include <math/mat4.h>
#include <math/quat.h>
#include <math/vec3.h>

#if defined(__ANDROID__)
#include <android/asset_manager.h>
#include <android/log.h>
#include <android_native_app_glue.h>
#else
#include "generated/resources/monkey.h"
#include "generated/resources/resources.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace filament;
using namespace filament::backend;
using namespace filament::math;
using namespace bluevk;

namespace {

constexpr uint32_t kEyeCount = 2;
constexpr XrViewConfigurationType kViewConfigType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
constexpr uint32_t kQuadSize = 1024;

struct Config {
    uint32_t frames = 0;            // 0 means "no frame limit"
#if defined(__ANDROID__)
    // On Android the app is an Activity the user (or `adb shell am force-stop`) can close, so the
    // watchdog that keeps desktop runs bounded would only get in the way.
    double timeoutSeconds = 0.0;
    std::string ibl = "lightroom_14b";
#else
    double timeoutSeconds = 15.0;   // 0 means "no timeout"
    // Prefix of a pair of <prefix>_ibl.ktx / <prefix>_skybox.ktx files, as produced by cmgen.
    std::string ibl = "assets/ibl/lightroom_14b/lightroom_14b";
#endif
    double nearPlane = 0.05;
    double farPlane = 100.0;
    uint8_t msaa = 4;               // 1 disables multi-sampling; only 1 and 4 are meaningful
    uint8_t quadMsaa = 0;           // 0 means "same as msaa"
    bool validation = true;
    bool depthLayer = true;
    bool listExtensions = false;
    bool renderModels = true;
    bool handMeshes = true;
    bool quadLayer = true;
    bool dumpQuad = false;          // dump the quad layer instead of the projection layer
    uint32_t dumpFrame = 0;         // 0 means "never dump"
    std::string dumpPrefix = "helloxr";
};

char const* xrResultName(XrInstance instance, XrResult result) {    static char buffer[XR_MAX_RESULT_STRING_SIZE];
    if (instance != XR_NULL_HANDLE && XR_SUCCEEDED(xrResultToString(instance, result, buffer))) {
        return buffer;
    }
    snprintf(buffer, sizeof(buffer), "XrResult(%d)", static_cast<int>(result));
    return buffer;
}

mat4 toMat4(XrPosef const& pose) {
    quat const q{ pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z };
    return mat4{ mat3{ q }, double3{ pose.position.x, pose.position.y, pose.position.z } };
}

// OpenXR fov angles are signed half-angles from the view axis, so their tangents scaled by the near
// distance give the near-plane extents directly. A finite far plane is used deliberately: it keeps
// the depth range submitted to the compositor finite too.
mat4 projectionFromFov(XrFovf const& fov, double near, double far) {
    return mat4::frustum(std::tan(fov.angleLeft) * near, std::tan(fov.angleRight) * near,
            std::tan(fov.angleDown) * near, std::tan(fov.angleUp) * near, near, far);
}

} // anonymous namespace

// ------------------------------------------------------------------------------------------------
// Platform
// ------------------------------------------------------------------------------------------------

// Presents the OpenXR swapchains to Filament as if they were a regular Vulkan swapchain. Every
// override below runs on Filament's driver thread.
class XrVulkanPlatform final : public VulkanPlatform {
public:
    struct XrSwapChain : public Platform::SwapChain {
        XrSwapchain color = XR_NULL_HANDLE;
        XrSwapchain depth = XR_NULL_HANDLE;
        SwapChainBundle bundle;
    };

    // Asks for the next presented frame of one swapchain to be copied back to the host. Used to
    // verify from the command line that both multiview layers and the depth buffer really were
    // written.
    void requestFrameDump(XrSwapChain const* target, char const* prefix) noexcept {
        mDumpTarget = target;
        mDumpPrefix = prefix;
    }

    Customization getCustomization() const noexcept override {
        Customization customization;
        // OpenXR expects COLOR_ATTACHMENT_OPTIMAL at release, not PRESENT_SRC.
        customization.transitionSwapChainImageLayoutForPresent = false;
        return customization;
    }

    SwapChainBundle getSwapChainBundle(SwapChainPtr handle) override {
        return static_cast<XrSwapChain*>(handle)->bundle;
    }

    // The pointer handed to Engine::createSwapChain() is the XR swapchain to render into, which is
    // what lets a projection layer and a quad layer each have their own.
    SwapChainPtr createSwapChain(void* nativeWindow, uint64_t, VkExtent2D) override {
        return static_cast<XrSwapChain*>(nativeWindow);
    }

    void destroy(SwapChainPtr) override {} // the XR swapchains outlive the Engine

    bool hasResized(SwapChainPtr) override { return false; }

    bool isProtected(SwapChainPtr) override { return false; }

    VkResult recreate(SwapChainPtr) override { return VK_SUCCESS; }

    VkResult acquire(SwapChainPtr handle, ImageSyncData* outImageSyncData) override {
        auto* swapChain = static_cast<XrSwapChain*>(handle);
        uint32_t colorIndex = 0;
        if (!acquireAndWait(swapChain->color, &colorIndex)) {
            return VK_ERROR_UNKNOWN;
        }
        if (swapChain->depth != XR_NULL_HANDLE) {
            uint32_t depthIndex = 0;
            if (!acquireAndWait(swapChain->depth, &depthIndex)) {
                return VK_ERROR_UNKNOWN;
            }
            // Filament indexes color and depth with one image index, so the chains must stay in
            // lockstep. They do as long as we always acquire and release them together.
            if (depthIndex != colorIndex) {
                XRLOG("color/depth swapchain indices diverged (%u vs %u)", colorIndex, depthIndex);
                return VK_ERROR_UNKNOWN;
            }
        }
        outImageSyncData->imageIndex = colorIndex;
        outImageSyncData->imageReadySemaphore = VK_NULL_HANDLE;
        return VK_SUCCESS;
    }

    VkResult present(SwapChainPtr handle, uint32_t index, VkSemaphore finishedDrawing) override {
        auto* swapChain = static_cast<XrSwapChain*>(handle);

        // OpenXR has no way to consume Filament's completion semaphore, and Filament recycles it
        // once this image index comes around again. Drain it with an empty submit so it is never
        // reused while still signaled.
        if (finishedDrawing != VK_NULL_HANDLE) {
            VkPipelineStageFlags const waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            VkSubmitInfo const submitInfo = {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &finishedDrawing,
                .pWaitDstStageMask = &waitStage,
            };
            vkQueueSubmit(getGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
        }

        if (mDumpPrefix != nullptr && swapChain == mDumpTarget) {
            dumpFrame(*swapChain, index);
            mDumpPrefix = nullptr;
            mDumpTarget = nullptr;
        }

        XrSwapchainImageReleaseInfo const releaseInfo = { XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
        if (XR_FAILED(xrReleaseSwapchainImage(swapChain->color, &releaseInfo))) {
            return VK_ERROR_UNKNOWN;
        }
        if (swapChain->depth != XR_NULL_HANDLE &&
                XR_FAILED(xrReleaseSwapchainImage(swapChain->depth, &releaseInfo))) {
            return VK_ERROR_UNKNOWN;
        }
        return VK_SUCCESS;
    }

protected:
    ExtensionSet getSwapchainInstanceExtensions() const override { return {}; }

    SurfaceBundle createVkSurfaceKHR(void*, VkInstance, uint64_t) const noexcept override {
        return { VK_NULL_HANDLE, { 0, 0 } };
    }

private:
    static bool acquireAndWait(XrSwapchain swapChain, uint32_t* outIndex) {
        XrSwapchainImageAcquireInfo const acquireInfo = { XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
        if (XR_FAILED(xrAcquireSwapchainImage(swapChain, &acquireInfo, outIndex))) {
            return false;
        }
        XrSwapchainImageWaitInfo waitInfo = { XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
        waitInfo.timeout = XR_INFINITE_DURATION;
        return XR_SUCCEEDED(xrWaitSwapchainImage(swapChain, &waitInfo));
    }

    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties) const {
        VkPhysicalDeviceMemoryProperties memoryProperties = {};
        vkGetPhysicalDeviceMemoryProperties(getPhysicalDevice(), &memoryProperties);
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if ((typeBits & (1u << i)) &&
                    (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        return UINT32_MAX;
    }

    // Copies every array layer of one swapchain image into host memory. The image is put back into
    // the layout OpenXR expects at release time. A barrier must name every aspect of the format,
    // while a buffer copy must name exactly one, hence the two masks.
    std::vector<uint8_t> readBack(VkImage image, VkImageAspectFlags barrierAspect,
            VkImageAspectFlags copyAspect, VkImageLayout layout, VkExtent2D extent,
            uint32_t layers, uint32_t bytesPerPixel) const {
        VkDevice const device = getDevice();
        VkDeviceSize const size = VkDeviceSize(extent.width) * extent.height * bytesPerPixel * layers;

        VkBufferCreateInfo const bufferInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        VkBuffer buffer = VK_NULL_HANDLE;
        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            return {};
        }
        VkMemoryRequirements requirements = {};
        vkGetBufferMemoryRequirements(device, buffer, &requirements);
        VkMemoryAllocateInfo const allocateInfo = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = requirements.size,
            .memoryTypeIndex = findMemoryType(requirements.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
        };
        VkDeviceMemory memory = VK_NULL_HANDLE;
        vkAllocateMemory(device, &allocateInfo, nullptr, &memory);
        vkBindBufferMemory(device, buffer, memory, 0);

        VkCommandPoolCreateInfo const poolInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = getGraphicsQueueFamilyIndex(),
        };
        VkCommandPool pool = VK_NULL_HANDLE;
        vkCreateCommandPool(device, &poolInfo, nullptr, &pool);
        VkCommandBufferAllocateInfo const cmdInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(device, &cmdInfo, &cmd);

        VkCommandBufferBeginInfo const beginInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkBeginCommandBuffer(cmd, &beginInfo);

        VkImageMemoryBarrier barrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = layout,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = { barrierAspect, 0, 1, 0, layers },
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

        VkBufferImageCopy const region = {
            .imageSubresource = { copyAspect, 0, 0, layers },
            .imageExtent = { extent.width, extent.height, 1 },
        };
        vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1,
                &region);

        std::swap(barrier.oldLayout, barrier.newLayout);
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo const submitInfo = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd,
        };
        vkQueueSubmit(getGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(getGraphicsQueue());

        std::vector<uint8_t> pixels(size);
        void* mapped = nullptr;
        vkMapMemory(device, memory, 0, size, 0, &mapped);
        memcpy(pixels.data(), mapped, size_t(size));
        vkUnmapMemory(device, memory);

        vkDestroyCommandPool(device, pool, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        vkFreeMemory(device, memory, nullptr);
        return pixels;
    }

    void dumpFrame(XrSwapChain const& swapChain, uint32_t index) const {
        auto const& bundle = swapChain.bundle;
        VkExtent2D const extent = bundle.extent;
        uint32_t const layers = bundle.layerCount;
        size_t const pixelsPerLayer = size_t(extent.width) * extent.height;

        if (bundle.colorFormat != VK_FORMAT_R8G8B8A8_SRGB &&
                bundle.colorFormat != VK_FORMAT_R8G8B8A8_UNORM) {
            XRLOG("frame dump: unsupported color format %d", int(bundle.colorFormat));
            return;
        }
        std::vector<uint8_t> const color = readBack(bundle.colors[index],
                VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, extent, layers, 4);
        if (color.empty()) {
            XRLOG("frame dump: color read back failed");
            return;
        }

        size_t differingPixels = 0;
        std::vector<uint8_t> geometryMask(pixelsPerLayer * layers, 0);
        for (uint32_t layer = 0; layer < layers; ++layer) {
            uint8_t const* data = color.data() + layer * pixelsPerLayer * 4;
            // The most common color is the sky; a fixed reference pixel is no good because the
            // subject can end up anywhere once the headset moves.
            std::unordered_map<uint32_t, uint32_t> histogram;
            for (size_t i = 0; i < pixelsPerLayer; i += 32) {
                uint8_t const* pixel = data + i * 4;
                uint32_t const key = (uint32_t(pixel[0]) << 16) | (uint32_t(pixel[1]) << 8) |
                                     uint32_t(pixel[2]);
                histogram[key]++;
            }
            uint32_t background = 0;
            uint32_t backgroundCount = 0;
            for (auto const& entry: histogram) {
                if (entry.second > backgroundCount) {
                    backgroundCount = entry.second;
                    background = entry.first;
                }
            }
            int const bgR = int((background >> 16) & 0xFF);
            int const bgG = int((background >> 8) & 0xFF);
            int const bgB = int(background & 0xFF);

            double luma = 0.0;
            size_t geometry = 0;
            double centroidX = 0.0;
            for (size_t i = 0; i < pixelsPerLayer; ++i) {
                uint8_t const* pixel = data + i * 4;
                luma += (pixel[0] + pixel[1] + pixel[2]) / 3.0;
                int const delta = std::abs(int(pixel[0]) - bgR) + std::abs(int(pixel[1]) - bgG) +
                                  std::abs(int(pixel[2]) - bgB);
                if (delta > 12) {
                    geometryMask[layer * pixelsPerLayer + i] = 1;
                    geometry++;
                    centroidX += double(i % extent.width);
                }
                if (layer > 0 && memcmp(pixel, &color[i * 4], 3) != 0) {
                    differingPixels++;
                }
            }
            XRLOG("frame dump: eye %u mean luma %.1f, geometry %.1f%%, geometry centroid x %.1f",
                    layer, luma / double(pixelsPerLayer),
                    100.0 * geometry / double(pixelsPerLayer),
                    geometry ? centroidX / double(geometry) : 0.0);
            writePpm(std::string(mDumpPrefix) + "_eye" + std::to_string(layer) + ".ppm", data,
                    extent);
        }
        if (layers > 1) {
            XRLOG("frame dump: eyes differ on %.1f%% of pixels (stereo disparity)",
                    100.0 * differingPixels / double(pixelsPerLayer));
        }

        if (bundle.depths.empty()) {
            return;
        }
        if (bundle.depthFormat != VK_FORMAT_D32_SFLOAT &&
                bundle.depthFormat != VK_FORMAT_D32_SFLOAT_S8_UINT &&
                bundle.depthFormat != VK_FORMAT_D24_UNORM_S8_UINT) {
            XRLOG("frame dump: unsupported depth format %d", int(bundle.depthFormat));
            return;
        }
        VkImageAspectFlags depthAspect = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (bundle.depthFormat == VK_FORMAT_D32_SFLOAT_S8_UINT ||
                bundle.depthFormat == VK_FORMAT_D24_UNORM_S8_UINT) {
            depthAspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        std::vector<uint8_t> const depthBytes = readBack(bundle.depths[index], depthAspect,
                VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                extent, layers, 4);
        if (depthBytes.empty()) {
            XRLOG("frame dump: depth read back failed");
            return;
        }

        // The depth aspect of D24_UNORM_S8_UINT copies out as X8_D24_UNORM_PACK32, which holds the
        // depth in bits 8..31.
        bool const isPacked24 = bundle.depthFormat == VK_FORMAT_D24_UNORM_S8_UINT;
        uint32_t maxRaw = 0;
        std::vector<float> depthValues(pixelsPerLayer * layers);
        for (size_t i = 0; i < depthValues.size(); ++i) {
            uint32_t raw;
            memcpy(&raw, depthBytes.data() + i * 4, sizeof(raw));
            maxRaw = std::max(maxRaw, raw);
            if (isPacked24) {
                depthValues[i] = float(raw >> 8) / 16777215.0f;
            } else {
                memcpy(&depthValues[i], &raw, sizeof(float));
            }
        }
        XRLOG("frame dump: depth format %d, max raw word 0x%08x", int(bundle.depthFormat), maxRaw);
        if (maxRaw == 0) {
            // Distinguishes "the runtime would not let us read the image" from "nothing was drawn".
            // Some runtimes (Meta's on Quest) appear not to honour XR_SWAPCHAIN_USAGE_TRANSFER_SRC
            // on depth swapchains, which makes the copy yield nothing.
            XRLOG("frame dump: depth read back empty; the runtime likely disallows copying from "
                  "the depth swapchain, so this says nothing about what was rendered");
            return;
        }
        float const* depth = depthValues.data();
        for (uint32_t layer = 0; layer < layers; ++layer) {
            float const* data = depth + layer * pixelsPerLayer;
            uint8_t const* mask = geometryMask.data() + layer * pixelsPerLayer;
            float minDepth = 1.0f;
            float maxDepth = 0.0f;
            size_t written = 0;
            size_t agree = 0;
            size_t missing = 0;
            size_t spurious = 0;
            double centroidX = 0.0;
            for (size_t i = 0; i < pixelsPerLayer; ++i) {
                minDepth = std::min(minDepth, data[i]);
                maxDepth = std::max(maxDepth, data[i]);
                // Filament uses reversed-Z, so anything above 0 is closer than the far plane.
                bool const near = data[i] > 0.0f;
                written += near ? 1 : 0;
                if (near) {
                    centroidX += double(i % extent.width);
                }
                agree += (near && mask[i]) ? 1 : 0;
                missing += (!near && mask[i]) ? 1 : 0;
                spurious += (near && !mask[i]) ? 1 : 0;
            }
            XRLOG("frame dump: eye %u depth range [%.4f, %.4f], %.1f%% closer than far, "
                  "centroid x %.1f",
                    layer, minDepth, maxDepth, 100.0 * written / double(pixelsPerLayer),
                    written ? centroidX / double(written) : 0.0);
            XRLOG("frame dump: eye %u depth vs color: agree %.1f%%, missing %.1f%%, "
                  "spurious %.1f%%",
                    layer, 100.0 * agree / double(pixelsPerLayer),
                    100.0 * missing / double(pixelsPerLayer),
                    100.0 * spurious / double(pixelsPerLayer));
        }
    }

    static void writePpm(std::string const& path, uint8_t const* rgba, VkExtent2D extent) {
        FILE* file = fopen(path.c_str(), "wb");
        if (!file) {
            return;
        }
        fprintf(file, "P6\n%u %u\n255\n", extent.width, extent.height);
        std::vector<uint8_t> row(size_t(extent.width) * 3);
        for (uint32_t y = 0; y < extent.height; ++y) {
            for (uint32_t x = 0; x < extent.width; ++x) {
                uint8_t const* pixel = rgba + (size_t(y) * extent.width + x) * 4;
                row[x * 3 + 0] = pixel[0];
                row[x * 3 + 1] = pixel[1];
                row[x * 3 + 2] = pixel[2];
            }
            fwrite(row.data(), 1, row.size(), file);
        }
        fclose(file);
        XRLOG("frame dump: wrote %s", path.c_str());
    }

    XrSwapChain const* mDumpTarget = nullptr;
    char const* mDumpPrefix = nullptr;
};

// ------------------------------------------------------------------------------------------------
// App
// ------------------------------------------------------------------------------------------------

class HelloXr {
public:
#if defined(__ANDROID__)
    HelloXr(Config const& config, android_app* app) : mConfig(config), mApp(app) { addFeatures(); }
#else
    explicit HelloXr(Config const& config) : mConfig(config) { addFeatures(); }
#endif

    ~HelloXr() {
        for (auto& feature: mFeatures) {
            if (feature) {
                feature->terminate();
            }
        }
        mFeatures.clear();
        if (mEngine) {
            mEngine->flushAndWait();
            mEngine->destroy(mSkybox);
            mEngine->destroy(mIndirectLight);
            mEngine->destroy(mIblTexture);
            mEngine->destroy(mSkyboxTexture);
            mEngine->destroy(mMonkey.renderable);
            mEngine->destroy(mMonkey.vertexBuffer);
            mEngine->destroy(mMonkey.indexBuffer);
            mEngine->destroy(mMaterialInstance);
            mEngine->destroy(mMaterial);
            mEngine->destroy(mLight);
            mEngine->destroy(mView);
            mEngine->destroy(mScene);
            mEngine->destroy(mRenderer);
            mEngine->destroy(mFilamentSwapChain);
            if (mQuadView != nullptr) {
                mEngine->destroy(mQuadView);
                mEngine->destroy(mQuadRenderer);
                mEngine->destroy(mQuadSwapChain);
                mEngine->destroyCameraComponent(mQuadCameraEntity);
            }
            mEngine->destroyCameraComponent(mCameraEntity);
            auto& em = utils::EntityManager::get();
            em.destroy(mCameraEntity);
            em.destroy(mQuadCameraEntity);
            em.destroy(mLight);
            Engine::destroy(&mEngine);
        }
        if (mXrSwapChain.color != XR_NULL_HANDLE) {
            xrDestroySwapchain(mXrSwapChain.color);
        }
        if (mXrSwapChain.depth != XR_NULL_HANDLE) {
            xrDestroySwapchain(mXrSwapChain.depth);
        }
        if (mQuad.color != XR_NULL_HANDLE) {
            xrDestroySwapchain(mQuad.color);
        }
        if (mQuad.depth != XR_NULL_HANDLE) {
            xrDestroySwapchain(mQuad.depth);
        }
        if (mAppSpace != XR_NULL_HANDLE) {
            xrDestroySpace(mAppSpace);
        }
        if (mViewSpace != XR_NULL_HANDLE) {
            xrDestroySpace(mViewSpace);
        }
        if (mSession != XR_NULL_HANDLE) {
            xrDestroySession(mSession);
        }
        // The Vulkan instance and device came from the XR runtime and go away with it.
        if (mXrInstance != XR_NULL_HANDLE) {
            xrDestroyInstance(mXrInstance);
        }
    }

    bool initialize() {
        return createXrInstance() && createVulkanContext() && createSession() &&
               createSwapChains() && createEngine() && createScene() && initializeFeatures();
    }

    void run() {
        auto const start = std::chrono::steady_clock::now();
        auto lastReport = start;
        uint32_t framesSinceReport = 0;

        while (!mExitRequested) {
            pumpAndroidEvents();
            pollEvents();
            if (mExitRequested) {
                break;
            }

            auto const now = std::chrono::steady_clock::now();
            double const elapsed = std::chrono::duration<double>(now - start).count();
            if (mConfig.timeoutSeconds > 0.0 && elapsed >= mConfig.timeoutSeconds) {
                XRLOG("timeout after %.1fs (%u frames)", elapsed, mFrameCount);
                if (mExitPending) {
                    // The runtime never took us to EXITING; leave anyway.
                    break;
                }
                requestExit();
                continue;
            }

            if (!mSessionRunning) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }

            renderFrame();
            mFrameCount++;
            framesSinceReport++;

            if (mFrameCount == mConfig.dumpFrame) {
                if (!mConfig.ibl.empty() && mIndirectLight) {
                    XRLOG("note: the color/depth agreement below assumes a flat skybox; re-run "
                          "with --ibl= for a meaningful comparison");
                }
                mPlatform.requestFrameDump(mConfig.dumpQuad ? &mQuad : &mXrSwapChain,
                        mConfig.dumpPrefix.c_str());
            }

            double const sinceReport = std::chrono::duration<double>(now - lastReport).count();
            if (sinceReport >= 1.0) {
                XRLOG("%.1f fps | frame %u | head (% .2f,% .2f,% .2f) | ipd %.3f m",
                        framesSinceReport / sinceReport, mFrameCount, mLastHeadPosition.x,
                        mLastHeadPosition.y, mLastHeadPosition.z,
                        length(mLastEyeOffsets[1] - mLastEyeOffsets[0]));
                lastReport = now;
                framesSinceReport = 0;
            }

            if (!mExitPending && mConfig.frames != 0 && mFrameCount >= mConfig.frames) {
                XRLOG("reached the %u frame limit", mConfig.frames);
                requestExit();
            }
        }
        XRLOG("rendered %u frames", mFrameCount);
    }

private:
    void addFeatures() {
        if (mConfig.renderModels) {
            mFeatures.push_back(helloxr::createRenderModels());
        }
        if (mConfig.handMeshes) {
            mFeatures.push_back(helloxr::createHandMeshes());
        }
    }

    // Runs after the scene exists, and drops any feature that cannot set itself up.
    bool initializeFeatures() {
        helloxr::FeatureContext const context{ mXrInstance, mSession, mAppSpace, mEngine, mScene,
            mMaterial, mConfig.dumpFrame != 0 ? mConfig.dumpPrefix : std::string() };
        for (auto& feature: mFeatures) {
            if (!feature) {
                continue;
            }
            if (feature->initialize(context)) {
                XRLOG("%s enabled", feature->name());
            } else {
                XRLOG("%s disabled: initialization failed", feature->name());
                feature->terminate();
                feature.reset();
            }
        }
        return true;
    }

    bool xrCheck(XrResult result, char const* what) const {
        if (XR_SUCCEEDED(result)) {
            return true;
        }
        XRLOG("%s failed: %s", what, xrResultName(mXrInstance, result));
        return false;
    }

    template<typename Fn>
    bool loadXrFunction(char const* name, Fn* out) const {
        return xrCheck(xrGetInstanceProcAddr(mXrInstance, name,
                               reinterpret_cast<PFN_xrVoidFunction*>(out)),
                name);
    }

    bool createXrInstance() {
#if defined(__ANDROID__)
        // Must happen before any other OpenXR call: the Android loader needs the VM and Activity
        // to find the runtime through the system broker.
        PFN_xrInitializeLoaderKHR initializeLoader = nullptr;
        if (XR_FAILED(xrGetInstanceProcAddr(XR_NULL_HANDLE, "xrInitializeLoaderKHR",
                    reinterpret_cast<PFN_xrVoidFunction*>(&initializeLoader))) ||
                initializeLoader == nullptr) {
            XRLOG("xrInitializeLoaderKHR is unavailable");
            return false;
        }
        XrLoaderInitInfoAndroidKHR loaderInfo = { XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR };
        loaderInfo.applicationVM = mApp->activity->vm;
        loaderInfo.applicationContext = mApp->activity->clazz;
        if (!xrCheck(initializeLoader(
                    reinterpret_cast<XrLoaderInitInfoBaseHeaderKHR const*>(&loaderInfo)),
                "xrInitializeLoaderKHR")) {
            return false;
        }
#endif

        uint32_t extensionCount = 0;
        if (!xrCheck(xrEnumerateInstanceExtensionProperties(nullptr, 0, &extensionCount, nullptr),
                    "xrEnumerateInstanceExtensionProperties")) {
            return false;
        }
        std::vector<XrExtensionProperties> available(extensionCount,
                { XR_TYPE_EXTENSION_PROPERTIES });
        if (!xrCheck(xrEnumerateInstanceExtensionProperties(nullptr, extensionCount,
                            &extensionCount, available.data()),
                    "xrEnumerateInstanceExtensionProperties")) {
            return false;
        }
        auto const supports = [&available](char const* name) {
            return std::any_of(available.begin(), available.end(),
                    [name](XrExtensionProperties const& e) {
                        return strcmp(e.extensionName, name) == 0;
                    });
        };

        if (mConfig.listExtensions) {
            XRLOG("runtime exposes %u extensions:", extensionCount);
            for (auto const& e: available) {
                XRLOG("  %s (v%u)", e.extensionName, e.extensionVersion);
            }
        }

        if (!supports(XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME)) {
            XRLOG("runtime does not support %s", XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME);
            return false;
        }
        std::vector<char const*> extensions{ XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME };

        mDepthLayerSupported = supports(XR_KHR_COMPOSITION_LAYER_DEPTH_EXTENSION_NAME);
        if (!mConfig.depthLayer) {
            mDepthLayerSupported = false;
        } else if (mDepthLayerSupported) {
            extensions.push_back(XR_KHR_COMPOSITION_LAYER_DEPTH_EXTENSION_NAME);
        } else {
            XRLOG("warning: runtime does not support %s; submitting color only",
                    XR_KHR_COMPOSITION_LAYER_DEPTH_EXTENSION_NAME);
        }

        // A feature is only kept if the runtime has everything it asked for.
        for (auto& feature: mFeatures) {
            auto const needed = feature->requiredExtensions();
            bool const satisfied = std::all_of(needed.begin(), needed.end(), supports);
            if (satisfied) {
                extensions.insert(extensions.end(), needed.begin(), needed.end());
            } else {
                XRLOG("%s disabled: the runtime is missing one of its extensions",
                        feature->name());
                feature.reset();
            }
        }

        XrInstanceCreateInfo createInfo = { XR_TYPE_INSTANCE_CREATE_INFO };
#if defined(__ANDROID__)
        if (!supports(XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME)) {
            XRLOG("runtime does not support %s", XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME);
            return false;
        }
        extensions.push_back(XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME);
        XrInstanceCreateInfoAndroidKHR androidInfo = { XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR };
        androidInfo.applicationVM = mApp->activity->vm;
        androidInfo.applicationActivity = mApp->activity->clazz;
        createInfo.next = &androidInfo;
#endif
        snprintf(createInfo.applicationInfo.applicationName,
                sizeof(createInfo.applicationInfo.applicationName), "helloxr");
        snprintf(createInfo.applicationInfo.engineName,
                sizeof(createInfo.applicationInfo.engineName), "Filament");
        createInfo.applicationInfo.apiVersion = XR_API_VERSION_1_0;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.enabledExtensionNames = extensions.data();

        if (!xrCheck(xrCreateInstance(&createInfo, &mXrInstance), "xrCreateInstance")) {
            return false;
        }

        XrInstanceProperties props = { XR_TYPE_INSTANCE_PROPERTIES };
        if (xrCheck(xrGetInstanceProperties(mXrInstance, &props), "xrGetInstanceProperties")) {
            XRLOG("runtime: %s %d.%d.%d", props.runtimeName,
                    static_cast<int>(XR_VERSION_MAJOR(props.runtimeVersion)),
                    static_cast<int>(XR_VERSION_MINOR(props.runtimeVersion)),
                    static_cast<int>(XR_VERSION_PATCH(props.runtimeVersion)));
        }

        XrSystemGetInfo systemInfo = { XR_TYPE_SYSTEM_GET_INFO };
        systemInfo.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
        if (!xrCheck(xrGetSystem(mXrInstance, &systemInfo, &mSystemId), "xrGetSystem")) {
            return false;
        }

        XrSystemProperties systemProps = { XR_TYPE_SYSTEM_PROPERTIES };
        if (xrCheck(xrGetSystemProperties(mXrInstance, mSystemId, &systemProps),
                    "xrGetSystemProperties")) {
            XRLOG("system: %s", systemProps.systemName);
        }

        uint32_t viewCount = 0;
        if (!xrCheck(xrEnumerateViewConfigurationViews(mXrInstance, mSystemId, kViewConfigType, 0,
                            &viewCount, nullptr),
                    "xrEnumerateViewConfigurationViews")) {
            return false;
        }
        if (viewCount != kEyeCount) {
            XRLOG("expected %u views for primary stereo, got %u", kEyeCount, viewCount);
            return false;
        }
        for (auto& view: mViewConfigs) {
            view = { XR_TYPE_VIEW_CONFIGURATION_VIEW };
        }
        if (!xrCheck(xrEnumerateViewConfigurationViews(mXrInstance, mSystemId, kViewConfigType,
                            viewCount, &viewCount, mViewConfigs),
                    "xrEnumerateViewConfigurationViews")) {
            return false;
        }
        mEyeWidth = mViewConfigs[0].recommendedImageRectWidth;
        mEyeHeight = mViewConfigs[0].recommendedImageRectHeight;
        XRLOG("view configuration: %ux%u per eye", mEyeWidth, mEyeHeight);
        return true;
    }

    bool createVulkanContext() {
        if (!bluevk::initialize()) {
            XRLOG("BlueVK could not load the Vulkan loader");
            return false;
        }

        PFN_xrGetVulkanGraphicsRequirements2KHR getRequirements = nullptr;
        PFN_xrCreateVulkanInstanceKHR createVulkanInstance = nullptr;
        PFN_xrGetVulkanGraphicsDevice2KHR getGraphicsDevice = nullptr;
        PFN_xrCreateVulkanDeviceKHR createVulkanDevice = nullptr;
        if (!loadXrFunction("xrGetVulkanGraphicsRequirements2KHR", &getRequirements) ||
                !loadXrFunction("xrCreateVulkanInstanceKHR", &createVulkanInstance) ||
                !loadXrFunction("xrGetVulkanGraphicsDevice2KHR", &getGraphicsDevice) ||
                !loadXrFunction("xrCreateVulkanDeviceKHR", &createVulkanDevice)) {
            return false;
        }

        XrGraphicsRequirementsVulkanKHR requirements = {
            XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR
        };
        if (!xrCheck(getRequirements(mXrInstance, mSystemId, &requirements),
                    "xrGetVulkanGraphicsRequirements2KHR")) {
            return false;
        }

        // Filament skips all layer and extension discovery when given a shared context, so the
        // validation layer has to be requested here or not at all.
        std::vector<char const*> layers;
        std::vector<char const*> instanceExtensions;
        if (mConfig.validation && hasInstanceLayer("VK_LAYER_KHRONOS_validation")) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            mDebugUtilsEnabled = true;
            XRLOG("Vulkan validation layer enabled");
        }

        VkApplicationInfo const appInfo = {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "helloxr",
            .pEngineName = "Filament",
            .apiVersion = VK_API_VERSION_1_1,
        };
        VkInstanceCreateInfo const vkInstanceInfo = {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = static_cast<uint32_t>(layers.size()),
            .ppEnabledLayerNames = layers.data(),
            .enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size()),
            .ppEnabledExtensionNames = instanceExtensions.data(),
        };
        XrVulkanInstanceCreateInfoKHR xrInstanceInfo = { XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR };
        xrInstanceInfo.systemId = mSystemId;
        xrInstanceInfo.pfnGetInstanceProcAddr = bluevk::vkGetInstanceProcAddr;
        xrInstanceInfo.vulkanCreateInfo = &vkInstanceInfo;

        VkResult vkResult = VK_SUCCESS;
        if (!xrCheck(createVulkanInstance(mXrInstance, &xrInstanceInfo, &mVkInstance, &vkResult),
                    "xrCreateVulkanInstanceKHR")) {
            return false;
        }
        if (vkResult != VK_SUCCESS) {
            XRLOG("vkCreateInstance returned %d", static_cast<int>(vkResult));
            return false;
        }
        bluevk::bindInstance(mVkInstance);

        XrVulkanGraphicsDeviceGetInfoKHR deviceGetInfo = {
            XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR
        };
        deviceGetInfo.systemId = mSystemId;
        deviceGetInfo.vulkanInstance = mVkInstance;
        if (!xrCheck(getGraphicsDevice(mXrInstance, &deviceGetInfo, &mVkPhysicalDevice),
                    "xrGetVulkanGraphicsDevice2KHR")) {
            return false;
        }

        VkPhysicalDeviceProperties deviceProps = {};
        vkGetPhysicalDeviceProperties(mVkPhysicalDevice, &deviceProps);
        XRLOG("gpu: %s", deviceProps.deviceName);

        if (!findGraphicsQueueFamily()) {
            return false;
        }

        VkPhysicalDeviceMultiviewFeatures multiviewFeatures = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES,
        };
        VkPhysicalDeviceFeatures2 features = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &multiviewFeatures,
        };
        vkGetPhysicalDeviceFeatures2(mVkPhysicalDevice, &features);
        if (multiviewFeatures.multiview != VK_TRUE) {
            XRLOG("the selected GPU does not support Vulkan multiview");
            return false;
        }
        multiviewFeatures.multiviewGeometryShader = VK_FALSE;
        multiviewFeatures.multiviewTessellationShader = VK_FALSE;

        // The runtime renders its own compositor and debug views on the device we hand it, so it
        // needs features Filament never asks for. Enable the ones it actually uses.
        std::vector<char const*> deviceExtensions;
        std::vector<VkExtensionProperties> const supportedExtensions =
                enumerateDeviceExtensions(mVkPhysicalDevice);
        auto const supportsExtension = [&supportedExtensions](char const* name) {
            return std::any_of(supportedExtensions.begin(), supportedExtensions.end(),
                    [name](VkExtensionProperties const& e) {
                        return strcmp(e.extensionName, name) == 0;
                    });
        };

        VkPhysicalDeviceTimelineSemaphoreFeatures timelineFeatures = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
            .timelineSemaphore = VK_TRUE,
        };
        if (supportsExtension(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME)) {
            deviceExtensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
            timelineFeatures.pNext = features.pNext;
            features.pNext = &timelineFeatures;
        }

        VkPhysicalDevicePipelineCreationCacheControlFeatures cacheControlFeatures = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES,
            .pipelineCreationCacheControl = VK_TRUE,
        };
        if (supportsExtension(VK_EXT_PIPELINE_CREATION_CACHE_CONTROL_EXTENSION_NAME)) {
            deviceExtensions.push_back(VK_EXT_PIPELINE_CREATION_CACHE_CONTROL_EXTENSION_NAME);
            cacheControlFeatures.pNext = features.pNext;
            features.pNext = &cacheControlFeatures;
        }

        if (deviceProps.apiVersion < VK_API_VERSION_1_1) {
            deviceExtensions.push_back(VK_KHR_MULTIVIEW_EXTENSION_NAME);
        }

        // Filament resolves the multi-sampled depth buffer into the XR depth image through a
        // render pass resolve, which only the renderpass2 structures can describe. We create the
        // device, and Filament skips extension discovery when given a shared context, so these
        // have to be requested here.
        if (mConfig.msaa > 1) {
            for (char const* name: { VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME,
                         VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME }) {
                if (!supportsExtension(name)) {
                    XRLOG("the GPU does not support %s, which %ux MSAA requires", name,
                            uint32_t(mConfig.msaa));
                    return false;
                }
                deviceExtensions.push_back(name);
            }
        }

        // One queue, shared with Filament: the runtime synchronizes against the queue named in the
        // graphics binding, so every submission has to land on that same queue.
        float const queuePriority = 1.0f;
        VkDeviceQueueCreateInfo const queueInfo = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = mGraphicsQueueFamilyIndex,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority,
        };
        VkDeviceCreateInfo const vkDeviceInfo = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = &features,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queueInfo,
            .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
            .ppEnabledExtensionNames = deviceExtensions.data(),
        };
        XrVulkanDeviceCreateInfoKHR xrDeviceInfo = { XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR };
        xrDeviceInfo.systemId = mSystemId;
        xrDeviceInfo.pfnGetInstanceProcAddr = bluevk::vkGetInstanceProcAddr;
        xrDeviceInfo.vulkanPhysicalDevice = mVkPhysicalDevice;
        xrDeviceInfo.vulkanCreateInfo = &vkDeviceInfo;

        if (!xrCheck(createVulkanDevice(mXrInstance, &xrDeviceInfo, &mVkDevice, &vkResult),
                    "xrCreateVulkanDeviceKHR")) {
            return false;
        }
        if (vkResult != VK_SUCCESS) {
            XRLOG("vkCreateDevice returned %d", static_cast<int>(vkResult));
            return false;
        }
        return true;
    }

    static bool hasInstanceLayer(char const* name) {
        uint32_t count = 0;
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> layers(count);
        vkEnumerateInstanceLayerProperties(&count, layers.data());
        return std::any_of(layers.begin(), layers.end(), [name](VkLayerProperties const& l) {
            return strcmp(l.layerName, name) == 0;
        });
    }

    static std::vector<VkExtensionProperties> enumerateDeviceExtensions(VkPhysicalDevice device) {
        uint32_t count = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> extensions(count);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data());
        return extensions;
    }

    bool findGraphicsQueueFamily() {
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(mVkPhysicalDevice, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(mVkPhysicalDevice, &count, families.data());
        for (uint32_t i = 0; i < count; ++i) {
            if (families[i].queueCount > 0 && (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                mGraphicsQueueFamilyIndex = i;
                return true;
            }
        }
        XRLOG("no graphics queue family found");
        return false;
    }

    bool createSession() {
        XrGraphicsBindingVulkanKHR binding = { XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR };
        binding.instance = mVkInstance;
        binding.physicalDevice = mVkPhysicalDevice;
        binding.device = mVkDevice;
        binding.queueFamilyIndex = mGraphicsQueueFamilyIndex;
        binding.queueIndex = 0;

        XrSessionCreateInfo createInfo = { XR_TYPE_SESSION_CREATE_INFO };
        createInfo.next = &binding;
        createInfo.systemId = mSystemId;
        if (!xrCheck(xrCreateSession(mXrInstance, &createInfo, &mSession), "xrCreateSession")) {
            return false;
        }

        uint32_t spaceCount = 0;
        xrEnumerateReferenceSpaces(mSession, 0, &spaceCount, nullptr);
        std::vector<XrReferenceSpaceType> spaces(spaceCount);
        xrEnumerateReferenceSpaces(mSession, spaceCount, &spaceCount, spaces.data());
        bool const hasLocal = std::find(spaces.begin(), spaces.end(),
                                      XR_REFERENCE_SPACE_TYPE_LOCAL) != spaces.end();

        XrReferenceSpaceCreateInfo spaceInfo = { XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
        spaceInfo.poseInReferenceSpace.orientation.w = 1.0f;
        spaceInfo.referenceSpaceType = hasLocal ? XR_REFERENCE_SPACE_TYPE_LOCAL
                                                : XR_REFERENCE_SPACE_TYPE_STAGE;
        if (!xrCheck(xrCreateReferenceSpace(mSession, &spaceInfo, &mAppSpace),
                    "xrCreateReferenceSpace")) {
            return false;
        }
        XRLOG("reference space: %s", hasLocal ? "LOCAL" : "STAGE");

        spaceInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
        return xrCheck(xrCreateReferenceSpace(mSession, &spaceInfo, &mViewSpace),
                "xrCreateReferenceSpace(VIEW)");
    }

    static int64_t selectSwapChainFormat(std::vector<int64_t> const& supported,
            std::vector<VkFormat> const& preferred) {
        for (VkFormat const format: preferred) {
            if (std::find(supported.begin(), supported.end(), int64_t(format)) != supported.end()) {
                return int64_t(format);
            }
        }
        return 0;
    }

    bool createSwapChains() {
        uint32_t formatCount = 0;
        if (!xrCheck(xrEnumerateSwapchainFormats(mSession, 0, &formatCount, nullptr),
                    "xrEnumerateSwapchainFormats")) {
            return false;
        }
        std::vector<int64_t> formats(formatCount);
        if (!xrCheck(xrEnumerateSwapchainFormats(mSession, formatCount, &formatCount,
                            formats.data()),
                    "xrEnumerateSwapchainFormats")) {
            return false;
        }

        std::string formatList;
        for (int64_t const format: formats) {
            formatList += std::to_string(format) + " ";
        }
        XRLOG("runtime swapchain formats: %s", formatList.c_str());

        // Post-processing is disabled under multiview, so Filament writes linear values straight to
        // the attachment and an sRGB target gets the encoding done by the hardware.
        mColorFormat = selectSwapChainFormat(formats,
                { VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_R8G8B8A8_UNORM,
                  VK_FORMAT_B8G8R8A8_UNORM });
        if (mColorFormat == 0) {
            XRLOG("no supported color swapchain format");
            return false;
        }
        // A combined depth-stencil format is preferred: once the depth swapchain is submitted for
        // reprojection the runtime barriers the image itself, and the Meta runtime does so with a
        // DEPTH|STENCIL aspect mask, which is invalid on a depth-only image.
        mDepthFormat = selectSwapChainFormat(formats,
                { VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT,
                  VK_FORMAT_D16_UNORM });
        if (mDepthFormat == 0) {
            XRLOG("the runtime exposes no depth format; rendering without a depth buffer");
        }

        if (!createXrSwapChain(&mXrSwapChain, mEyeWidth, mEyeHeight, kEyeCount)) {
            return false;
        }
        mDepthLayerSupported = mDepthLayerSupported && mXrSwapChain.depth != XR_NULL_HANDLE;
        XRLOG("depth submission: %s", mDepthLayerSupported ? "enabled" : "disabled");

        // A flat panel the compositor samples at its own resolution, which is why it is worth a
        // layer of its own rather than a quad inside the scene.
        if (mConfig.quadLayer && !createXrSwapChain(&mQuad, kQuadSize, kQuadSize, 1)) {
            return false;
        }
        return true;
    }

    // Every one of these becomes its own Filament SwapChain, so another composition layer only
    // needs another call.
    bool createXrSwapChain(XrVulkanPlatform::XrSwapChain* out, uint32_t width, uint32_t height,
            uint32_t arraySize) {
        XrSwapchainCreateInfo createInfo = { XR_TYPE_SWAPCHAIN_CREATE_INFO };
        createInfo.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
        if (mConfig.dumpFrame != 0) {
            createInfo.usageFlags |= XR_SWAPCHAIN_USAGE_TRANSFER_SRC_BIT;
        }
        createInfo.format = mColorFormat;
        createInfo.sampleCount = 1;
        createInfo.width = width;
        createInfo.height = height;
        createInfo.faceCount = 1;
        createInfo.arraySize = arraySize;
        createInfo.mipCount = 1;
        if (!xrCheck(xrCreateSwapchain(mSession, &createInfo, &out->color),
                    "xrCreateSwapchain(color)")) {
            return false;
        }

        std::vector<VkImage> colorImages;
        if (!enumerateSwapChainImages(out->color, &colorImages)) {
            return false;
        }

        // Even a layer that never submits depth needs one, otherwise Filament has nothing to
        // depth test against.
        std::vector<VkImage> depthImages;
        if (mDepthFormat != 0) {
            createInfo.usageFlags = XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            if (mConfig.dumpFrame != 0) {
                createInfo.usageFlags |= XR_SWAPCHAIN_USAGE_TRANSFER_SRC_BIT;
            }
            createInfo.format = mDepthFormat;
            if (!xrCheck(xrCreateSwapchain(mSession, &createInfo, &out->depth),
                        "xrCreateSwapchain(depth)")) {
                return false;
            }
            if (!enumerateSwapChainImages(out->depth, &depthImages)) {
                return false;
            }
            if (depthImages.size() != colorImages.size()) {
                XRLOG("color and depth swapchains have different lengths (%zu vs %zu)",
                        colorImages.size(), depthImages.size());
                return false;
            }
        }

        auto& bundle = out->bundle;
        bundle.colors = utils::FixedCapacityVector<VkImage>::with_capacity(colorImages.size());
        for (VkImage const image: colorImages) {
            bundle.colors.push_back(image);
        }
        bundle.depths = utils::FixedCapacityVector<VkImage>::with_capacity(depthImages.size());
        for (VkImage const image: depthImages) {
            bundle.depths.push_back(image);
        }
        bundle.colorFormat = VkFormat(mColorFormat);
        bundle.depthFormat = VkFormat(mDepthFormat);
        bundle.extent = { width, height };
        // Anything above one is what makes Filament build a multiview render target.
        bundle.layerCount = arraySize;

        XRLOG("swapchain: %ux%u, %u images, %u layers, color format %d, depth format %d", width,
                height, uint32_t(colorImages.size()), arraySize, int(mColorFormat),
                int(mDepthFormat));
        return true;
    }

    bool enumerateSwapChainImages(XrSwapchain swapChain, std::vector<VkImage>* outImages) {
        uint32_t count = 0;
        if (!xrCheck(xrEnumerateSwapchainImages(swapChain, 0, &count, nullptr),
                    "xrEnumerateSwapchainImages")) {
            return false;
        }
        std::vector<XrSwapchainImageVulkanKHR> images(count,
                { XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR });
        if (!xrCheck(xrEnumerateSwapchainImages(swapChain, count, &count,
                            reinterpret_cast<XrSwapchainImageBaseHeader*>(images.data())),
                    "xrEnumerateSwapchainImages")) {
            return false;
        }
        outImages->clear();
        for (auto const& image: images) {
            outImages->push_back(image.image);
        }
        return true;
    }

    bool createEngine() {
        VulkanPlatform::VulkanSharedContext sharedContext = {};
        sharedContext.instance = mVkInstance;
        sharedContext.physicalDevice = mVkPhysicalDevice;
        sharedContext.logicalDevice = mVkDevice;
        sharedContext.graphicsQueueFamilyIndex = mGraphicsQueueFamilyIndex;
        sharedContext.graphicsQueueIndex = 0;
        sharedContext.debugUtilsEnabled = mDebugUtilsEnabled;
        sharedContext.multiviewSupported = true;

        Engine::Config config = {};
        config.stereoscopicType = Engine::StereoscopicType::MULTIVIEW;
        config.stereoscopicEyeCount = kEyeCount;

        mEngine = Engine::Builder()
                          .backend(Engine::Backend::VULKAN)
                          .platform(&mPlatform)
                          .sharedContext(&sharedContext)
                          .config(&config)
                          .build();
        if (!mEngine) {
            XRLOG("failed to create the Filament Engine");
            return false;
        }
        if (!mEngine->isStereoSupported(Engine::StereoscopicType::MULTIVIEW)) {
            XRLOG("the Filament backend reports multiview as unsupported");
            return false;
        }
        XRLOG("Filament engine created with multiview stereo");

        // Without this Filament discards the depth attachment at the end of the render pass and
        // the runtime would be handed undefined depth.
        uint64_t swapChainFlags = 0;
        if (mXrSwapChain.depth != XR_NULL_HANDLE) {
            swapChainFlags |= filament::SwapChain::CONFIG_PRESERVE_DEPTH_BUFFER;
        }
        if (mXrSwapChain.bundle.depthFormat == VK_FORMAT_D32_SFLOAT_S8_UINT ||
                mXrSwapChain.bundle.depthFormat == VK_FORMAT_D24_UNORM_S8_UINT) {
            swapChainFlags |= filament::SwapChain::CONFIG_HAS_STENCIL_BUFFER;
        }
        // The XR images stay single-sampled; the backend renders to a sidecar and resolves.
        if (mConfig.msaa > 1) {
            swapChainFlags |= filament::SwapChain::CONFIG_MSAA_4_SAMPLES;
        }
        mFilamentSwapChain = mEngine->createSwapChain(&mXrSwapChain, swapChainFlags);
        mRenderer = mEngine->createRenderer();
        if (mFilamentSwapChain == nullptr || mRenderer == nullptr) {
            return false;
        }

        if (mConfig.quadLayer) {
            // Sample count travels with the swapchain, so a layer that does not need smooth edges
            // can skip the cost the projection layer pays. The quad never hands its depth to the
            // compositor either, so it only needs whatever keeps depth testing working.
            uint64_t quadFlags = swapChainFlags &
                                 ~(filament::SwapChain::CONFIG_PRESERVE_DEPTH_BUFFER |
                                         filament::SwapChain::CONFIG_MSAA_4_SAMPLES);
            if (quadSamples() > 1) {
                quadFlags |= filament::SwapChain::CONFIG_MSAA_4_SAMPLES;
            }
            mQuadSwapChain = mEngine->createSwapChain(&mQuad, quadFlags);
            mQuadRenderer = mEngine->createRenderer();
            if (mQuadSwapChain == nullptr || mQuadRenderer == nullptr) {
                return false;
            }
            XRLOG("quad layer: %ux multi-sampling", uint32_t(quadSamples()));
        }
        return true;
    }

    uint8_t quadSamples() const {
        return mConfig.quadMsaa != 0 ? mConfig.quadMsaa : mConfig.msaa;
    }

    bool createScene() {
        auto& em = utils::EntityManager::get();
        mScene = mEngine->createScene();
        mView = mEngine->createView();

#if defined(__ANDROID__)
        if (!helloxr::readAsset("aiDefaultMat.filamat", &mMaterialPackage) ||
                !helloxr::readAsset("suzanne.filamesh", &mMeshData)) {
            XRLOG("failed to read the material or mesh from the APK assets");
            return false;
        }
        void const* materialData = mMaterialPackage.data();
        size_t const materialSize = mMaterialPackage.size();
        void const* meshData = mMeshData.data();
        size_t const meshSize = mMeshData.size();
#else
        void const* materialData = RESOURCES_AIDEFAULTMAT_DATA;
        size_t const materialSize = RESOURCES_AIDEFAULTMAT_SIZE;
        void const* meshData = MONKEY_SUZANNE_DATA;
        size_t const meshSize = MONKEY_SUZANNE_SIZE;
#endif

        mMaterial = Material::Builder().package(materialData, materialSize).build(*mEngine);
        mMaterialInstance = mMaterial->createInstance();
        mMaterialInstance->setParameter("baseColor", RgbType::LINEAR, float3{ 0.8f, 1.0f, 1.0f });
        mMaterialInstance->setParameter("metallic", 0.0f);
        mMaterialInstance->setParameter("roughness", 0.4f);
        mMaterialInstance->setParameter("reflectance", 0.5f);

        mMonkey = filamesh::MeshReader::loadMeshFromBuffer(mEngine, meshData, meshSize, nullptr,
                nullptr, mMaterialInstance);
        auto& rcm = mEngine->getRenderableManager();
        rcm.setCastShadows(rcm.getInstance(mMonkey.renderable), false);
        mScene->addEntity(mMonkey.renderable);

        mLight = em.create();
        LightManager::Builder(LightManager::Type::SUN)
                .color(Color::toLinear<ACCURATE>(sRGBColor(0.98f, 0.92f, 0.89f)))
                .intensity(110000)
                .direction({ 0.7f, -1.0f, -0.8f })
                .sunAngularRadius(1.9f)
                .castShadows(false)
                .build(*mEngine, mLight);
        mScene->addEntity(mLight);

        if (!loadIbl()) {
            mSkybox = Skybox::Builder().color({ 0.06f, 0.07f, 0.10f, 1.0f }).build(*mEngine);
        }
        mScene->setSkybox(mSkybox);

        mCameraEntity = em.create();
        mCamera = mEngine->createCamera(mCameraEntity);

        mView->setScene(mScene);
        mView->setCamera(mCamera);
        mView->setViewport({ 0, 0, mEyeWidth, mEyeHeight });
        mView->setPostProcessingEnabled(false);
        mView->setShadowingEnabled(false);
        mView->setStereoscopicOptions({ .enabled = true });

        if (mConfig.quadLayer) {
            // The same scene from a fixed viewpoint, so the panel shows something recognisable
            // without needing assets of its own. Stereo has to be off: the quad swapchain has a
            // single layer, so a two-eye pass would have nowhere to put the second view.
            mQuadCameraEntity = em.create();
            mQuadCamera = mEngine->createCamera(mQuadCameraEntity);
            mQuadCamera->setProjection(60.0, 1.0, 0.1, 20.0, Camera::Fov::VERTICAL);
            mQuadCamera->lookAt({ 1.5f, 0.8f, -1.0f }, { 0.0f, 0.0f, -2.0f }, { 0.0f, 1.0f, 0.0f });

            mQuadView = mEngine->createView();
            mQuadView->setScene(mScene);
            mQuadView->setCamera(mQuadCamera);
            mQuadView->setViewport({ 0, 0, kQuadSize, kQuadSize });
            mQuadView->setPostProcessingEnabled(false);
            mQuadView->setShadowingEnabled(false);
            mQuadView->setStereoscopicOptions({ .enabled = false });
        }
        return true;
    }

    // Falls back to a flat skybox when the files are absent, so the sample runs from anywhere.
    bool loadIbl() {
        if (mConfig.ibl.empty()) {
            return false;
        }
        std::vector<uint8_t> iblData;
        std::vector<uint8_t> skyData;
        if (!helloxr::readAsset(mConfig.ibl + "_ibl.ktx", &iblData) ||
                !helloxr::readAsset(mConfig.ibl + "_skybox.ktx", &skyData)) {
            XRLOG("no IBL at '%s'; using a flat skybox", mConfig.ibl.c_str());
            return false;
        }

        auto* iblBundle = new image::Ktx1Bundle(iblData.data(), uint32_t(iblData.size()));
        auto* skyBundle = new image::Ktx1Bundle(skyData.data(), uint32_t(skyData.size()));
        math::float3 sphericalHarmonics[9];
        bool const hasHarmonics = iblBundle->getSphericalHarmonics(sphericalHarmonics);

        mIblTexture = ktxreader::Ktx1Reader::createTexture(mEngine, iblBundle, false);
        mSkyboxTexture = ktxreader::Ktx1Reader::createTexture(mEngine, skyBundle, false);

        auto builder = IndirectLight::Builder().reflections(mIblTexture).intensity(30000.0f);
        if (hasHarmonics) {
            builder.irradiance(3, sphericalHarmonics);
        }
        mIndirectLight = builder.build(*mEngine);
        mScene->setIndirectLight(mIndirectLight);

        mSkybox = Skybox::Builder().environment(mSkyboxTexture).showSun(true).build(*mEngine);
        XRLOG("loaded IBL from '%s'", mConfig.ibl.c_str());
        return true;
    }

    // Keeps the Activity lifecycle moving; without draining the looper the app would ANR.
    void pumpAndroidEvents() {
#if defined(__ANDROID__)
        int events = 0;
        android_poll_source* source = nullptr;
        // Block only while there is nothing to draw, otherwise never stall the render loop.
        int const timeoutMs = mSessionRunning ? 0 : 100;
        while (ALooper_pollOnce(timeoutMs, nullptr, &events,
                       reinterpret_cast<void**>(&source)) >= 0) {
            if (source != nullptr) {
                source->process(mApp, source);
            }
            if (mApp->destroyRequested != 0) {
                mExitRequested = true;
                return;
            }
        }
#endif
    }

    void pollEvents() {
        while (true) {
            XrEventDataBuffer event = { XR_TYPE_EVENT_DATA_BUFFER };
            XrResult const result = xrPollEvent(mXrInstance, &event);
            if (result == XR_EVENT_UNAVAILABLE) {
                return;
            }
            if (XR_FAILED(result)) {
                XRLOG("xrPollEvent failed: %s", xrResultName(mXrInstance, result));
                mExitRequested = true;
                return;
            }
            switch (event.type) {
                case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING:
                    XRLOG("instance loss pending");
                    mExitRequested = true;
                    return;
                case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED:
                    handleSessionStateChange(
                            *reinterpret_cast<XrEventDataSessionStateChanged*>(&event));
                    break;
                default:
                    break;
            }
        }
    }

    void handleSessionStateChange(XrEventDataSessionStateChanged const& event) {
        mSessionState = event.state;
        switch (mSessionState) {
            case XR_SESSION_STATE_READY: {
                XrSessionBeginInfo beginInfo = { XR_TYPE_SESSION_BEGIN_INFO };
                beginInfo.primaryViewConfigurationType = kViewConfigType;
                if (xrCheck(xrBeginSession(mSession, &beginInfo), "xrBeginSession")) {
                    mSessionRunning = true;
                    XRLOG("session started");
                }
                break;
            }
            case XR_SESSION_STATE_STOPPING:
                mSessionRunning = false;
                xrCheck(xrEndSession(mSession), "xrEndSession");
                XRLOG("session stopped");
                break;
            case XR_SESSION_STATE_EXITING:
            case XR_SESSION_STATE_LOSS_PENDING:
                mSessionRunning = false;
                mExitRequested = true;
                break;
            default:
                break;
        }
    }

    void requestExit() {
        mExitPending = true;
        if (mSessionRunning) {
            xrCheck(xrRequestExitSession(mSession), "xrRequestExitSession");
        } else {
            mExitRequested = true;
        }
    }

    void renderFrame() {
        XrFrameWaitInfo const waitInfo = { XR_TYPE_FRAME_WAIT_INFO };
        XrFrameState frameState = { XR_TYPE_FRAME_STATE };
        if (!xrCheck(xrWaitFrame(mSession, &waitInfo, &frameState), "xrWaitFrame")) {
            mExitRequested = true;
            return;
        }

        XrFrameBeginInfo const beginInfo = { XR_TYPE_FRAME_BEGIN_INFO };
        if (!xrCheck(xrBeginFrame(mSession, &beginInfo), "xrBeginFrame")) {
            mExitRequested = true;
            return;
        }

        XrCompositionLayerProjectionView projectionViews[kEyeCount] = {};
        XrCompositionLayerDepthInfoKHR depthInfos[kEyeCount] = {};
        XrCompositionLayerProjection layer = { XR_TYPE_COMPOSITION_LAYER_PROJECTION };
        XrCompositionLayerQuad quad = { XR_TYPE_COMPOSITION_LAYER_QUAD };
        XrCompositionLayerBaseHeader const* layers[2] = {};
        uint32_t layerCount = 0;

        if (frameState.shouldRender) {
            bool const drewProjection =
                    renderLayer(frameState.predictedDisplayTime, projectionViews, depthInfos,
                            &layer);
            bool const drewQuad = mConfig.quadLayer && renderQuadLayer(&quad);
            if (drewProjection || drewQuad) {
                // Waits for the driver thread to reach the release, and no further: OpenXR only
                // asks that the work has been queued, not that it has finished. Leaving the driver
                // thread with nothing queued is also what keeps Filament off the shared Vulkan
                // queue while the runtime submits on it during xrEndFrame.
                Fence::waitAndDestroy(mEngine->createFence());
            }
            if (drewProjection) {
                layers[layerCount++] =
                        reinterpret_cast<XrCompositionLayerBaseHeader const*>(&layer);
            }
            if (drewQuad) {
                layers[layerCount++] =
                        reinterpret_cast<XrCompositionLayerBaseHeader const*>(&quad);
            }
        }

        XrFrameEndInfo endInfo = { XR_TYPE_FRAME_END_INFO };
        endInfo.displayTime = frameState.predictedDisplayTime;
        endInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
        endInfo.layerCount = layerCount;
        endInfo.layers = layers;
        xrCheck(xrEndFrame(mSession, &endInfo), "xrEndFrame");
    }

    bool renderLayer(XrTime displayTime, XrCompositionLayerProjectionView* projectionViews,
            XrCompositionLayerDepthInfoKHR* depthInfos, XrCompositionLayerProjection* layer) {
        XrViewLocateInfo locateInfo = { XR_TYPE_VIEW_LOCATE_INFO };
        locateInfo.viewConfigurationType = kViewConfigType;
        locateInfo.displayTime = displayTime;
        locateInfo.space = mAppSpace;

        XrViewState viewState = { XR_TYPE_VIEW_STATE };
        XrView views[kEyeCount] = { { XR_TYPE_VIEW }, { XR_TYPE_VIEW } };
        uint32_t viewCount = kEyeCount;
        if (!xrCheck(xrLocateViews(mSession, &locateInfo, &viewState, kEyeCount, &viewCount, views),
                    "xrLocateViews")) {
            return false;
        }
        constexpr XrViewStateFlags kValid =
                XR_VIEW_STATE_POSITION_VALID_BIT | XR_VIEW_STATE_ORIENTATION_VALID_BIT;
        if ((viewState.viewStateFlags & kValid) != kValid) {
            return false;
        }

        // Filament wants a head transform plus a per-eye offset, which maps onto the VIEW reference
        // space plus each located view pose.
        mat4 worldFromHead = toMat4(views[0].pose);
        XrSpaceLocation headLocation = { XR_TYPE_SPACE_LOCATION };
        constexpr XrSpaceLocationFlags kLocated =
                XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
        if (XR_SUCCEEDED(xrLocateSpace(mViewSpace, mAppSpace, displayTime, &headLocation)) &&
                (headLocation.locationFlags & kLocated) == kLocated) {
            worldFromHead = toMat4(headLocation.pose);
        }
        mLastHeadPosition = float3(worldFromHead[3].xyz);

        mat4 const headFromWorld = inverse(worldFromHead);
        mat4 projections[kEyeCount];
        XrFovf cullingFov = views[0].fov;
        for (uint32_t i = 0; i < kEyeCount; ++i) {
            projections[i] = projectionFromFov(views[i].fov, mConfig.nearPlane, mConfig.farPlane);
            mat4 const headFromEye = headFromWorld * toMat4(views[i].pose);
            mCamera->setEyeModelMatrix(uint8_t(i), headFromEye);
            mLastEyeOffsets[i] = float3(headFromEye[3].xyz);
            cullingFov.angleLeft = std::min(cullingFov.angleLeft, views[i].fov.angleLeft);
            cullingFov.angleRight = std::max(cullingFov.angleRight, views[i].fov.angleRight);
            cullingFov.angleDown = std::min(cullingFov.angleDown, views[i].fov.angleDown);
            cullingFov.angleUp = std::max(cullingFov.angleUp, views[i].fov.angleUp);
        }
        mCamera->setModelMatrix(worldFromHead);
        mCamera->setCustomEyeProjection(projections, kEyeCount,
                projectionFromFov(cullingFov, mConfig.nearPlane, mConfig.farPlane),
                mConfig.nearPlane, mConfig.farPlane);

        animate(displayTime);

        for (auto& feature: mFeatures) {
            if (feature) {
                feature->update(displayTime);
            }
        }

        if (!mRenderer->beginFrame(mFilamentSwapChain)) {
            return false;
        }
        mRenderer->render(mView);
        mRenderer->endFrame();

        for (uint32_t i = 0; i < kEyeCount; ++i) {
            projectionViews[i] = { XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW };
            projectionViews[i].pose = views[i].pose;
            projectionViews[i].fov = views[i].fov;
            projectionViews[i].subImage.swapchain = mXrSwapChain.color;
            projectionViews[i].subImage.imageRect = {
                { 0, 0 },
                { int32_t(mEyeWidth), int32_t(mEyeHeight) }
            };
            projectionViews[i].subImage.imageArrayIndex = i;

            if (mDepthLayerSupported && mXrSwapChain.depth != XR_NULL_HANDLE) {
                depthInfos[i] = { XR_TYPE_COMPOSITION_LAYER_DEPTH_INFO_KHR };
                depthInfos[i].subImage = projectionViews[i].subImage;
                depthInfos[i].subImage.swapchain = mXrSwapChain.depth;
                // Filament writes reversed-Z, so minDepth (0) is the far plane and maxDepth (1)
                // is the near plane. nearZ > farZ is how the spec expects that to be signalled.
                depthInfos[i].minDepth = 0.0f;
                depthInfos[i].maxDepth = 1.0f;
                depthInfos[i].nearZ = float(mConfig.farPlane);
                depthInfos[i].farZ = float(mConfig.nearPlane);
                projectionViews[i].next = &depthInfos[i];
            }
        }
        layer->space = mAppSpace;
        layer->viewCount = kEyeCount;
        layer->views = projectionViews;
        return true;
    }

    // A second XR swapchain driven through its own Filament SwapChain, which only works because
    // the platform hands back whichever one it was given.
    bool renderQuadLayer(XrCompositionLayerQuad* quad) {
        if (!mQuadRenderer->beginFrame(mQuadSwapChain)) {
            return false;
        }
        mQuadRenderer->render(mQuadView);
        mQuadRenderer->endFrame();

        quad->layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
        quad->space = mAppSpace;
        quad->eyeVisibility = XR_EYE_VISIBILITY_BOTH;
        quad->subImage.swapchain = mQuad.color;
        quad->subImage.imageRect = { { 0, 0 }, { int32_t(kQuadSize), int32_t(kQuadSize) } };
        quad->subImage.imageArrayIndex = 0;
        quad->pose.orientation = { 0.0f, 0.0f, 0.0f, 1.0f };
        quad->pose.position = { -0.7f, 0.0f, -1.5f };
        quad->size = { 0.5f, 0.5f };
        return true;
    }

    void animate(XrTime displayTime) {        float const seconds = float(double(displayTime) * 1e-9);
        auto& tcm = mEngine->getTransformManager();
        mat4f const transform = mat4f::translation(float3{ 0.0f, 0.0f, -2.0f }) *
                                mat4f::rotation(seconds, float3{ 0.0f, 1.0f, 0.0f });
        tcm.setTransform(tcm.getInstance(mMonkey.renderable), transform);
    }

    Config const mConfig;

    XrInstance mXrInstance = XR_NULL_HANDLE;
    XrSystemId mSystemId = XR_NULL_SYSTEM_ID;
    XrSession mSession = XR_NULL_HANDLE;
    XrSpace mAppSpace = XR_NULL_HANDLE;
    XrSpace mViewSpace = XR_NULL_HANDLE;
    XrSessionState mSessionState = XR_SESSION_STATE_UNKNOWN;
    XrViewConfigurationView mViewConfigs[kEyeCount] = {};
    XrVulkanPlatform::XrSwapChain mXrSwapChain;
    XrVulkanPlatform::XrSwapChain mQuad;
    int64_t mColorFormat = 0;
    int64_t mDepthFormat = 0;

    VkInstance mVkInstance = VK_NULL_HANDLE;
    VkPhysicalDevice mVkPhysicalDevice = VK_NULL_HANDLE;
    VkDevice mVkDevice = VK_NULL_HANDLE;
    uint32_t mGraphicsQueueFamilyIndex = 0;
    bool mDebugUtilsEnabled = false;
    bool mDepthLayerSupported = false;

    XrVulkanPlatform mPlatform;
    Engine* mEngine = nullptr;
    Renderer* mRenderer = nullptr;
    filament::SwapChain* mFilamentSwapChain = nullptr;
    filament::SwapChain* mQuadSwapChain = nullptr;
    Renderer* mQuadRenderer = nullptr;
    View* mQuadView = nullptr;
    Camera* mQuadCamera = nullptr;
    utils::Entity mQuadCameraEntity;
    Scene* mScene = nullptr;
    View* mView = nullptr;
    Camera* mCamera = nullptr;
    Skybox* mSkybox = nullptr;
    IndirectLight* mIndirectLight = nullptr;
    Texture* mIblTexture = nullptr;
    Texture* mSkyboxTexture = nullptr;
    Material* mMaterial = nullptr;
    MaterialInstance* mMaterialInstance = nullptr;
    filamesh::MeshReader::Mesh mMonkey;
    utils::Entity mCameraEntity;
    utils::Entity mLight;
#if defined(__ANDROID__)
    android_app* mApp = nullptr;
    std::vector<uint8_t> mMaterialPackage;
    std::vector<uint8_t> mMeshData;
#endif

    uint32_t mEyeWidth = 0;
    uint32_t mEyeHeight = 0;
    std::vector<std::unique_ptr<helloxr::Feature>> mFeatures;
    uint32_t mFrameCount = 0;
    bool mSessionRunning = false;
    bool mExitRequested = false;
    bool mExitPending = false;
    float3 mLastHeadPosition = {};
    float3 mLastEyeOffsets[kEyeCount] = {};
};

namespace {

void printUsage() {
    XRLOG("helloxr: renders a Filament scene through OpenXR using Vulkan multiview.\n"
          "\n"
          "Options:\n"
          "  --frames=N        stop after N frames (default: unlimited)\n"
          "  --timeout=S       stop after S seconds, 0 to disable\n"
          "  --near=D          near plane distance in meters (default: 0.05)\n"
          "  --far=D           far plane distance in meters (default: 100)\n"
          "  --msaa=N          multi-sample count, 1 to disable (default: 4)\n"
          "  --quad-msaa=N     multi-sample count for the quad layer (default: same as --msaa)\n"
          "  --dump-frame=N    read frame N back and report per-eye color and depth stats\n"
          "                    (the color/depth cross-check assumes --ibl= i.e. a flat skybox)\n"
          "  --dump-prefix=P   file name prefix for --dump-frame\n"
          "  --ibl=PREFIX      load PREFIX_ibl.ktx and PREFIX_skybox.ktx, empty to disable\n"
          "  --no-validation   do not request the Vulkan validation layer\n"
          "  --no-depth-layer  do not submit depth with the projection layer\n"
          "  --list-extensions log every extension the runtime exposes\n"
          "  --no-render-models  do not draw the runtime's controller models\n"
          "  --no-hand-meshes  do not draw tracked hand meshes\n"
          "  --no-quad-layer   do not submit the second composition layer\n"
          "  --dump-quad       dump the quad layer rather than the projection layer\n"
          "  --help            print this message");
}

bool parseArguments(std::vector<std::string> const& args, Config* config) {
    for (std::string const& arg: args) {
        auto const startsWith = [&arg](char const* prefix) { return arg.rfind(prefix, 0) == 0; };
        if (arg == "--help" || arg == "-h") {
            printUsage();
            return false;
        } else if (startsWith("--frames=")) {
            config->frames = uint32_t(std::strtoul(arg.c_str() + 9, nullptr, 10));
        } else if (startsWith("--timeout=")) {
            config->timeoutSeconds = std::strtod(arg.c_str() + 10, nullptr);
        } else if (startsWith("--near=")) {
            config->nearPlane = std::strtod(arg.c_str() + 7, nullptr);
        } else if (startsWith("--far=")) {
            config->farPlane = std::strtod(arg.c_str() + 6, nullptr);
        } else if (startsWith("--msaa=")) {
            config->msaa = uint8_t(std::strtoul(arg.c_str() + 7, nullptr, 10));
        } else if (startsWith("--quad-msaa=")) {
            config->quadMsaa = uint8_t(std::strtoul(arg.c_str() + 12, nullptr, 10));
        } else if (startsWith("--dump-frame=")) {
            config->dumpFrame = uint32_t(std::strtoul(arg.c_str() + 13, nullptr, 10));
        } else if (startsWith("--dump-prefix=")) {
            config->dumpPrefix = arg.substr(14);
        } else if (startsWith("--ibl=")) {
            config->ibl = arg.substr(6);
        } else if (arg == "--no-validation") {
            config->validation = false;
        } else if (arg == "--no-depth-layer") {
            config->depthLayer = false;
        } else if (arg == "--list-extensions") {
            config->listExtensions = true;
        } else if (arg == "--no-render-models") {
            config->renderModels = false;
        } else if (arg == "--no-hand-meshes") {
            config->handMeshes = false;
        } else if (arg == "--no-quad-layer") {
            config->quadLayer = false;
        } else if (arg == "--dump-quad") {
            config->dumpQuad = true;
        } else {
            XRLOG("unknown argument: %s", arg.c_str());
            printUsage();
            return false;
        }
    }
    return true;
}

} // anonymous namespace

#if defined(__ANDROID__)

// There is no argv on Android. An optional args.txt in the app's external files directory lets a
// run be configured over adb without any JNI plumbing:
//   adb shell "echo --frames=300 --ibl= > /sdcard/Android/data/<pkg>/files/args.txt"
void android_main(android_app* app) {
    helloxr::setAssetManager(app->activity->assetManager);

    Config config;
    std::string const dataDir = app->activity->externalDataPath != nullptr
                                        ? app->activity->externalDataPath
                                        : app->activity->internalDataPath;
    config.dumpPrefix = dataDir + "/helloxr";

    std::vector<uint8_t> argsFile;
    std::vector<std::string> args;
    if (helloxr::readFile(dataDir + "/args.txt", &argsFile)) {
        std::string const contents(argsFile.begin(), argsFile.end());
        std::istringstream stream(contents);
        for (std::string token; stream >> token;) {
            args.push_back(token);
        }
        XRLOG("read %zu argument(s) from args.txt", args.size());
    }
    if (!parseArguments(args, &config)) {
        ANativeActivity_finish(app->activity);
        return;
    }

    {
        HelloXr xr(config, app);
        if (xr.initialize()) {
            xr.run();
        }
    }
    ANativeActivity_finish(app->activity);
}

#else

int main(int argc, char** argv) {
    std::vector<std::string> const args(argv + 1, argv + argc);
    Config config;
    if (!parseArguments(args, &config)) {
        return EXIT_FAILURE;
    }

    HelloXr app(config);
    if (!app.initialize()) {
        return EXIT_FAILURE;
    }
    app.run();
    return EXIT_SUCCESS;
}

#endif
