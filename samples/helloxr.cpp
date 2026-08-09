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

// BlueVK must come first: it pulls in vulkan.h with VK_NO_PROTOTYPES and (on Windows) windows.h,
// both of which openxr_platform.h expects to already be present.
#include <bluevk/BlueVK.h>

#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <backend/platforms/VulkanPlatform.h>

#include <filament/Camera.h>
#include <filament/Color.h>
#include <filament/Engine.h>
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

#include <utils/Entity.h>
#include <utils/EntityManager.h>

#include <math/mat3.h>
#include <math/mat4.h>
#include <math/quat.h>
#include <math/vec3.h>

#include "generated/resources/monkey.h"
#include "generated/resources/resources.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace filament;
using namespace filament::backend;
using namespace filament::math;
using namespace bluevk;

namespace {

constexpr uint32_t kEyeCount = 2;
constexpr XrViewConfigurationType kViewConfigType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;

// Filament treats a swapchain with no native window as headless and then never calls present();
// the pointer itself is never dereferenced because XrVulkanPlatform owns the real swapchain.
void* const kNativeWindowSentinel = reinterpret_cast<void*>(uintptr_t(1));

#define XRLOG(...) do { printf("[helloxr] " __VA_ARGS__); printf("\n"); fflush(stdout); } while (0)

struct Config {
    uint32_t frames = 0;            // 0 means "no frame limit"
    double timeoutSeconds = 15.0;   // 0 means "no timeout"
    double nearPlane = 0.05;
    double farPlane = 100.0;
    bool validation = true;
    uint32_t dumpFrame = 0;         // 0 means "never dump"
    std::string dumpPrefix = "helloxr";
};

char const* xrResultName(XrInstance instance, XrResult result) {
    static char buffer[XR_MAX_RESULT_STRING_SIZE];
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

    void setSwapChain(XrSwapChain* swapChain) noexcept { mSwapChain = swapChain; }

    // Asks for the next presented frame to be copied back to the host. Used to verify from the
    // command line that both multiview layers and the depth buffer really were written.
    void requestFrameDump(char const* prefix) noexcept { mDumpPrefix = prefix; }

    Customization getCustomization() const noexcept override {
        Customization customization;
        // OpenXR expects COLOR_ATTACHMENT_OPTIMAL at release, not PRESENT_SRC.
        customization.transitionSwapChainImageLayoutForPresent = false;
        return customization;
    }

    SwapChainBundle getSwapChainBundle(SwapChainPtr handle) override {
        return static_cast<XrSwapChain*>(handle)->bundle;
    }

    SwapChainPtr createSwapChain(void*, uint64_t, VkExtent2D) override { return mSwapChain; }

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

        if (mDumpPrefix != nullptr) {
            dumpFrame(*swapChain, index);
            mDumpPrefix = nullptr;
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
    // the layout OpenXR expects at release time.
    std::vector<uint8_t> readBack(VkImage image, VkImageAspectFlags aspect, VkImageLayout layout,
            VkExtent2D extent, uint32_t layers, uint32_t bytesPerPixel) const {
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
            .subresourceRange = { aspect, 0, 1, 0, layers },
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

        VkBufferImageCopy const region = {
            .imageSubresource = { aspect, 0, 0, layers },
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
                VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, extent,
                layers, 4);
        if (color.empty()) {
            XRLOG("frame dump: color read back failed");
            return;
        }

        size_t differingPixels = 0;
        std::vector<uint8_t> geometryMask(pixelsPerLayer * layers, 0);
        for (uint32_t layer = 0; layer < layers; ++layer) {
            uint8_t const* data = color.data() + layer * pixelsPerLayer * 4;
            // The corner is always sky, so it doubles as the background reference.
            uint8_t const* background = data;
            double luma = 0.0;
            size_t geometry = 0;
            double centroidX = 0.0;
            for (size_t i = 0; i < pixelsPerLayer; ++i) {
                uint8_t const* pixel = data + i * 4;
                luma += (pixel[0] + pixel[1] + pixel[2]) / 3.0;
                int const delta = std::abs(int(pixel[0]) - background[0]) +
                                  std::abs(int(pixel[1]) - background[1]) +
                                  std::abs(int(pixel[2]) - background[2]);
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
        if (bundle.depthFormat != VK_FORMAT_D32_SFLOAT) {
            XRLOG("frame dump: unsupported depth format %d", int(bundle.depthFormat));
            return;
        }
        std::vector<uint8_t> const depthBytes = readBack(bundle.depths[index],
                VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                extent, layers, 4);
        if (depthBytes.empty()) {
            XRLOG("frame dump: depth read back failed");
            return;
        }
        auto const* depth = reinterpret_cast<float const*>(depthBytes.data());
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

    XrSwapChain* mSwapChain = nullptr;
    char const* mDumpPrefix = nullptr;
};

// ------------------------------------------------------------------------------------------------
// App
// ------------------------------------------------------------------------------------------------

class HelloXr {
public:
    explicit HelloXr(Config const& config) : mConfig(config) {}

    ~HelloXr() {
        if (mEngine) {
            mEngine->flushAndWait();
            mEngine->destroy(mSkybox);
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
            mEngine->destroyCameraComponent(mCameraEntity);
            auto& em = utils::EntityManager::get();
            em.destroy(mCameraEntity);
            em.destroy(mLight);
            Engine::destroy(&mEngine);
        }
        if (mXrSwapChain.color != XR_NULL_HANDLE) {
            xrDestroySwapchain(mXrSwapChain.color);
        }
        if (mXrSwapChain.depth != XR_NULL_HANDLE) {
            xrDestroySwapchain(mXrSwapChain.depth);
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
               createSwapChains() && createEngine() && createScene();
    }

    void run() {
        auto const start = std::chrono::steady_clock::now();
        auto lastReport = start;
        uint32_t framesSinceReport = 0;

        while (!mExitRequested) {
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
                mPlatform.requestFrameDump(mConfig.dumpPrefix.c_str());
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

        if (!supports(XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME)) {
            XRLOG("runtime does not support %s", XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME);
            return false;
        }
        std::vector<char const*> extensions{ XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME };

        XrInstanceCreateInfo createInfo = { XR_TYPE_INSTANCE_CREATE_INFO };
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

        // Post-processing is disabled under multiview, so Filament writes linear values straight to
        // the attachment and an sRGB target gets the encoding done by the hardware.
        int64_t const colorFormat = selectSwapChainFormat(formats,
                { VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_R8G8B8A8_UNORM,
                  VK_FORMAT_B8G8R8A8_UNORM });
        if (colorFormat == 0) {
            XRLOG("no supported color swapchain format");
            return false;
        }
        int64_t const depthFormat = selectSwapChainFormat(formats,
                { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT,
                  VK_FORMAT_D16_UNORM });

        XrSwapchainCreateInfo createInfo = { XR_TYPE_SWAPCHAIN_CREATE_INFO };
        createInfo.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
        if (mConfig.dumpFrame != 0) {
            createInfo.usageFlags |= XR_SWAPCHAIN_USAGE_TRANSFER_SRC_BIT;
        }
        createInfo.format = colorFormat;
        createInfo.sampleCount = 1;
        createInfo.width = mEyeWidth;
        createInfo.height = mEyeHeight;
        createInfo.faceCount = 1;
        createInfo.arraySize = kEyeCount;
        createInfo.mipCount = 1;
        if (!xrCheck(xrCreateSwapchain(mSession, &createInfo, &mXrSwapChain.color),
                    "xrCreateSwapchain(color)")) {
            return false;
        }

        std::vector<VkImage> colorImages;
        if (!enumerateSwapChainImages(mXrSwapChain.color, &colorImages)) {
            return false;
        }

        std::vector<VkImage> depthImages;
        if (depthFormat != 0) {
            createInfo.usageFlags = XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            if (mConfig.dumpFrame != 0) {
                createInfo.usageFlags |= XR_SWAPCHAIN_USAGE_TRANSFER_SRC_BIT;
            }
            createInfo.format = depthFormat;
            if (!xrCheck(xrCreateSwapchain(mSession, &createInfo, &mXrSwapChain.depth),
                        "xrCreateSwapchain(depth)")) {
                return false;
            }
            if (!enumerateSwapChainImages(mXrSwapChain.depth, &depthImages)) {
                return false;
            }
            if (depthImages.size() != colorImages.size()) {
                XRLOG("color and depth swapchains have different lengths (%zu vs %zu)",
                        colorImages.size(), depthImages.size());
                return false;
            }
        } else {
            XRLOG("the runtime exposes no depth format; rendering without a depth buffer");
        }

        auto& bundle = mXrSwapChain.bundle;
        bundle.colors = utils::FixedCapacityVector<VkImage>::with_capacity(colorImages.size());
        for (VkImage const image: colorImages) {
            bundle.colors.push_back(image);
        }
        bundle.depths = utils::FixedCapacityVector<VkImage>::with_capacity(depthImages.size());
        for (VkImage const image: depthImages) {
            bundle.depths.push_back(image);
        }
        bundle.colorFormat = VkFormat(colorFormat);
        bundle.depthFormat = VkFormat(depthFormat);
        bundle.extent = { mEyeWidth, mEyeHeight };
        // This is what makes Filament build a multiview default render target.
        bundle.layerCount = kEyeCount;

        XRLOG("swapchain: %u images, color format %d, depth format %d, %u layers",
                uint32_t(colorImages.size()), int(colorFormat), int(depthFormat),
                bundle.layerCount);
        mPlatform.setSwapChain(&mXrSwapChain);
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
        mFilamentSwapChain = mEngine->createSwapChain(kNativeWindowSentinel, swapChainFlags);
        mRenderer = mEngine->createRenderer();
        return mFilamentSwapChain != nullptr && mRenderer != nullptr;
    }

    bool createScene() {
        auto& em = utils::EntityManager::get();
        mScene = mEngine->createScene();
        mView = mEngine->createView();

        mMaterial = Material::Builder()
                            .package(RESOURCES_AIDEFAULTMAT_DATA, RESOURCES_AIDEFAULTMAT_SIZE)
                            .build(*mEngine);
        mMaterialInstance = mMaterial->createInstance();
        mMaterialInstance->setParameter("baseColor", RgbType::LINEAR, float3{ 0.8f, 1.0f, 1.0f });
        mMaterialInstance->setParameter("metallic", 0.0f);
        mMaterialInstance->setParameter("roughness", 0.4f);
        mMaterialInstance->setParameter("reflectance", 0.5f);

        mMonkey = filamesh::MeshReader::loadMeshFromBuffer(mEngine, MONKEY_SUZANNE_DATA,
                MONKEY_SUZANNE_SIZE, nullptr, nullptr, mMaterialInstance);
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

        mSkybox = Skybox::Builder().color({ 0.06f, 0.07f, 0.10f, 1.0f }).build(*mEngine);
        mScene->setSkybox(mSkybox);

        mCameraEntity = em.create();
        mCamera = mEngine->createCamera(mCameraEntity);

        mView->setScene(mScene);
        mView->setCamera(mCamera);
        mView->setViewport({ 0, 0, mEyeWidth, mEyeHeight });
        mView->setPostProcessingEnabled(false);
        mView->setShadowingEnabled(false);
        mView->setStereoscopicOptions({ .enabled = true });
        return true;
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
        XrCompositionLayerProjection layer = { XR_TYPE_COMPOSITION_LAYER_PROJECTION };
        XrCompositionLayerBaseHeader const* layers[1] = {};
        uint32_t layerCount = 0;

        if (frameState.shouldRender &&
                renderLayer(frameState.predictedDisplayTime, projectionViews, &layer)) {
            layers[0] = reinterpret_cast<XrCompositionLayerBaseHeader const*>(&layer);
            layerCount = 1;
        }

        XrFrameEndInfo endInfo = { XR_TYPE_FRAME_END_INFO };
        endInfo.displayTime = frameState.predictedDisplayTime;
        endInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
        endInfo.layerCount = layerCount;
        endInfo.layers = layers;
        xrCheck(xrEndFrame(mSession, &endInfo), "xrEndFrame");
    }

    bool renderLayer(XrTime displayTime, XrCompositionLayerProjectionView* projectionViews,
            XrCompositionLayerProjection* layer) {
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

        if (!mRenderer->beginFrame(mFilamentSwapChain)) {
            return false;
        }
        mRenderer->render(mView);
        mRenderer->endFrame();
        // The driver thread is what calls xrAcquire/Wait/ReleaseSwapchainImage, so it has to finish
        // before xrEndFrame.
        mEngine->flushAndWait();

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
        }
        layer->space = mAppSpace;
        layer->viewCount = kEyeCount;
        layer->views = projectionViews;
        return true;
    }

    void animate(XrTime displayTime) {
        float const seconds = float(double(displayTime) * 1e-9);
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

    VkInstance mVkInstance = VK_NULL_HANDLE;
    VkPhysicalDevice mVkPhysicalDevice = VK_NULL_HANDLE;
    VkDevice mVkDevice = VK_NULL_HANDLE;
    uint32_t mGraphicsQueueFamilyIndex = 0;
    bool mDebugUtilsEnabled = false;

    XrVulkanPlatform mPlatform;
    Engine* mEngine = nullptr;
    Renderer* mRenderer = nullptr;
    filament::SwapChain* mFilamentSwapChain = nullptr;
    Scene* mScene = nullptr;
    View* mView = nullptr;
    Camera* mCamera = nullptr;
    Skybox* mSkybox = nullptr;
    Material* mMaterial = nullptr;
    MaterialInstance* mMaterialInstance = nullptr;
    filamesh::MeshReader::Mesh mMonkey;
    utils::Entity mCameraEntity;
    utils::Entity mLight;

    uint32_t mEyeWidth = 0;
    uint32_t mEyeHeight = 0;
    uint32_t mFrameCount = 0;
    bool mSessionRunning = false;
    bool mExitRequested = false;
    bool mExitPending = false;
    float3 mLastHeadPosition = {};
    float3 mLastEyeOffsets[kEyeCount] = {};
};

namespace {

void printUsage() {
    printf("helloxr: renders a Filament scene through OpenXR using Vulkan multiview.\n"
           "\n"
           "Options:\n"
           "  --frames=N        stop after N frames (default: unlimited)\n"
           "  --timeout=S       stop after S seconds, 0 to disable (default: 15)\n"
           "  --near=D          near plane distance in meters (default: 0.05)\n"
           "  --far=D           far plane distance in meters (default: 100)\n"
           "  --dump-frame=N    read frame N back and report per-eye color and depth stats\n"
           "  --dump-prefix=P   file name prefix for --dump-frame (default: helloxr)\n"
           "  --no-validation   do not request the Vulkan validation layer\n"
           "  --help            print this message\n");
}

bool parseArguments(int argc, char** argv, Config* config) {
    for (int i = 1; i < argc; ++i) {
        std::string const arg = argv[i];
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
        } else if (startsWith("--dump-frame=")) {
            config->dumpFrame = uint32_t(std::strtoul(arg.c_str() + 13, nullptr, 10));
        } else if (startsWith("--dump-prefix=")) {
            config->dumpPrefix = arg.substr(14);
        } else if (arg == "--no-validation") {
            config->validation = false;
        } else {
            printf("unknown argument: %s\n", arg.c_str());
            printUsage();
            return false;
        }
    }
    return true;
}

} // anonymous namespace

int main(int argc, char** argv) {
    Config config;
    if (!parseArguments(argc, argv, &config)) {
        return EXIT_FAILURE;
    }

    HelloXr app(config);
    if (!app.initialize()) {
        return EXIT_FAILURE;
    }
    app.run();
    return EXIT_SUCCESS;
}
