/*
 * Copyright (C) 2022 The Android Open Source Project
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

#include "VulkanSwapChain.h"

#include "VulkanCommands.h"
#include "VulkanTexture.h"

#include <utils/debug.h>
#include <utils/FixedCapacityVector.h>
#include <utils/Panic.h>

using namespace bluevk;
using namespace utils;

namespace filament::backend {

namespace {

// Swapchain images are wrapped without a backend format (they are created from a VkFormat), but
// allocating a multi-sampled sidecar to match one needs it back. Only the formats a swapchain can
// actually be created with are worth handling.
bool toTextureFormat(VkFormat format, TextureFormat* out) {
    switch (format) {
        case VK_FORMAT_R8G8B8A8_SRGB: *out = TextureFormat::SRGB8_A8; return true;
        case VK_FORMAT_R8G8B8A8_UNORM: *out = TextureFormat::RGBA8; return true;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: *out = TextureFormat::RGB10_A2; return true;
        case VK_FORMAT_R16G16B16A16_SFLOAT: *out = TextureFormat::RGBA16F; return true;
        case VK_FORMAT_D16_UNORM: *out = TextureFormat::DEPTH16; return true;
        case VK_FORMAT_D32_SFLOAT: *out = TextureFormat::DEPTH32F; return true;
        case VK_FORMAT_D24_UNORM_S8_UINT: *out = TextureFormat::DEPTH24_STENCIL8; return true;
        case VK_FORMAT_D32_SFLOAT_S8_UINT: *out = TextureFormat::DEPTH32F_STENCIL8; return true;
        default: return false;
    }
}

} // anonymous namespace

VulkanSwapChain::VulkanSwapChain(VulkanPlatform* platform, VulkanContext const& context,
        fvkmemory::ResourceManager* resourceManager, VmaAllocator allocator,
        VulkanCommands* commands, VulkanStagePool& stagePool, void* nativeWindow, uint64_t flags,
        VkExtent2D extent)
    : mPlatform(platform),
      mContext(context),
      mResourceManager(resourceManager),
      mCommands(commands),
      mAllocator(allocator),
      mStagePool(stagePool),
      mHeadless(extent.width != 0 && extent.height != 0 && !nativeWindow),
      mFlushAndWaitOnResize(platform->getCustomization().flushAndWaitOnWindowResize),
      mTransitionSwapChainImageLayoutForPresent(
              platform->getCustomization().transitionSwapChainImageLayoutForPresent),
      mLayerCount(1),
      mSamples((flags & SWAP_CHAIN_CONFIG_MSAA_4_SAMPLES) ? 4 : 1),
      mPreserveDepth((flags & SWAP_CHAIN_CONFIG_PRESERVE_DEPTH_BUFFER) != 0),
      mCurrentSwapIndex(0),
      mAcquired(false),
      mIsFirstRenderPass(true) {
    swapChain = mPlatform->createSwapChain(nativeWindow, flags, extent);
    FILAMENT_CHECK_POSTCONDITION(swapChain) << "Unable to create swapchain";

    update();
}

VulkanSwapChain::~VulkanSwapChain() {
    // Must wait for the inflight command buffers to finish since they might contain the images
    // we're about to destroy.
    mCommands->flush();
    mCommands->wait();

    mColors = {};
    mDepths = {};
    mMsaaColor = {};
    mMsaaDepth = {};
    for (auto& semaphore : mFinishedDrawing) {
        semaphore = {};
    }
    mFinishedDrawing.clear();
    mPlatform->destroy(swapChain);
}

void VulkanSwapChain::update() {
    mColors.clear();
    mDepths.clear();

    auto const bundle = mPlatform->getSwapChainBundle(swapChain);
    size_t const swapChainCount = bundle.colors.size();
    mColors.reserve(bundle.colors.size());
    VkDevice const device = mPlatform->getDevice();

    mFinishedDrawing.clear();
    mFinishedDrawing.reserve(swapChainCount);
    mFinishedDrawing.resize(swapChainCount);
    for (size_t i = 0; i < swapChainCount; ++i) {
        mFinishedDrawing[i] = {};
    }

    TextureUsage depthUsage = TextureUsage::DEPTH_ATTACHMENT;
    TextureUsage colorUsage = TextureUsage::COLOR_ATTACHMENT;
    if (bundle.isProtected) {
        depthUsage |= TextureUsage::PROTECTED;
        colorUsage |= TextureUsage::PROTECTED;
    }

    for (auto const color: bundle.colors) {
        auto colorTexture = fvkmemory::resource_ptr<VulkanTexture>::construct(mResourceManager,
                mContext, device, mAllocator, mResourceManager, mCommands, color, VK_NULL_HANDLE,
                bundle.colorFormat, VK_NULL_HANDLE /*ycrcb */, VK_NULL_HANDLE, VK_NULL_HANDLE,
                Platform::ExternalImageHandle(),
                /*levels=*/1, /*samples=*/1, bundle.extent.width, bundle.extent.height,
                bundle.layerCount, colorUsage, mStagePool);
        mColors.push_back(colorTexture);
    }

    mDepths.reserve(bundle.depths.size());
    for (auto const depth: bundle.depths) {
        mDepths.push_back(fvkmemory::resource_ptr<VulkanTexture>::construct(mResourceManager,
                mContext, device, mAllocator, mResourceManager, mCommands, depth, VK_NULL_HANDLE,
                bundle.depthFormat, VK_NULL_HANDLE /*ycrcb */, VK_NULL_HANDLE, VK_NULL_HANDLE,
                Platform::ExternalImageHandle(), /*levels=*/1, /*samples=*/1, bundle.extent.width,
                bundle.extent.height, bundle.layerCount, depthUsage, mStagePool));
    }

    mExtent = bundle.extent;
    mLayerCount = bundle.layerCount;

    mMsaaColor = {};
    mMsaaDepth = {};
    if (mSamples > 1) {
        VkPhysicalDevice const physicalDevice = mPlatform->getPhysicalDevice();
        TextureFormat colorFormat;
        TextureFormat depthFormat;
        if (!toTextureFormat(bundle.colorFormat, &colorFormat)) {
            FVK_LOGW << "Swapchain format " << (int) bundle.colorFormat
                     << " cannot be multi-sampled; rendering without MSAA." << utils::io::endl;
        } else {
            mMsaaColor = fvkmemory::resource_ptr<VulkanTexture>::construct(mResourceManager, device,
                    physicalDevice, mContext, mAllocator, mResourceManager, mCommands,
                    SamplerType::SAMPLER_2D_ARRAY, /*levels=*/1, colorFormat, mSamples,
                    bundle.extent.width, bundle.extent.height, bundle.layerCount, colorUsage,
                    mStagePool);
            if (!mDepths.empty() && toTextureFormat(bundle.depthFormat, &depthFormat)) {
                mMsaaDepth = fvkmemory::resource_ptr<VulkanTexture>::construct(mResourceManager,
                        device, physicalDevice, mContext, mAllocator, mResourceManager, mCommands,
                        SamplerType::SAMPLER_2D_ARRAY, /*levels=*/1, depthFormat, mSamples,
                        bundle.extent.width, bundle.extent.height, bundle.layerCount, depthUsage,
                        mStagePool);
            }
        }
    }
}

void VulkanSwapChain::present(DriverBase& driver) {
    // The last acquire failed, so just skip presenting.
    if (!mAcquired) {
        return;
    }

    if (!mHeadless && mTransitionSwapChainImageLayoutForPresent) {
        VulkanCommandBuffer& commands = mCommands->get();
        VkImageSubresourceRange const subresources{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = mLayerCount,
        };
        mColors[mCurrentSwapIndex]->transitionLayout(&commands, subresources, VulkanLayout::PRESENT);
    }

    mCommands->flush();

    // We only present if it is not headless. No-op for headless.
    if (!mHeadless) {
        auto finishedDrawing = mCommands->acquireFinishedSignal();
        mFinishedDrawing[mCurrentSwapIndex] = finishedDrawing;
        VkResult const result =
                mPlatform->present(swapChain, mCurrentSwapIndex, finishedDrawing->getVkSemaphore());
        FILAMENT_CHECK_POSTCONDITION(result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR ||
                result == VK_ERROR_OUT_OF_DATE_KHR)
                << "Cannot present in swapchain. error=" << static_cast<int32_t>(result);
    }

    // We presented the last acquired buffer.
    mAcquired = false;
    mIsFirstRenderPass = true;

    if (mFrameScheduled.callback) {
        driver.scheduleCallback(mFrameScheduled.handler,
                [callback = mFrameScheduled.callback]() {
                    PresentCallable noop = PresentCallable(PresentCallable::noopPresent, nullptr);
                    callback->operator()(noop);
                });
    }
}

std::pair<bool, bool> VulkanSwapChain::acquire() {
    // Indicates whether the backing swapchain has changed (and might invalidate the associated
    // images that are tracked in the FBO cache).
    bool swapchainRecreated = false;

    // Final result of the call to acquire a swapchain image.
    VkResult result = VK_NOT_READY;

    // It's ok to call acquire multiple times due to it being linked to Driver::makeCurrent(). If a
    // valid swapchain has already been acquired, then this method is no-op.
    if (mAcquired) {
        return { mAcquired, swapchainRecreated };
    }

    // Check if the surface has resized; if so, we need to recreate a swapchain, which is done in
    // the while loop.
    if (mPlatform->hasResized(swapChain)) {
        // This indicates a surface size change and a need to recreate swapchain.
        result = VK_ERROR_OUT_OF_DATE_KHR;
    }

    VulkanPlatform::ImageSyncData imageSyncData;

    // Following is written as a loop to cover a few cases:
    //   - If resize is true from hasResized() above, then result == VK_ERROR_OUT_OF_DATE_KHR.
    //     And we will first recreate the swapchain before acquiring (on tryCount == 0).
    //   - If resize is not true, then just try to acquire (on tryCount == 0)
    //       - If the acquire succeeds, then result == VK_SUCCESS, break loop
    //       - if acquire fails and result = VK_SUBOPTIMAL_KHR or VK_ERROR_OUT_OF_DATE_KHR
    //         (on tryCount == 1), then
    //           - recreate swapchain and try to acquire again (on tryCount == 1).

    for (uint8_t tryCount = 0; result != VK_SUCCESS && tryCount < 2; tryCount++) {
        if (result == VK_SUBOPTIMAL_KHR || result == VK_ERROR_OUT_OF_DATE_KHR) {
            // Following recreates the swapchain

            // Calling flush multiptle times is ok, since it's no-op if not recording.
            if (mFlushAndWaitOnResize) {
                mCommands->flush();
                mCommands->wait();
            }
            mPlatform->recreate(swapChain);
            update();
            swapchainRecreated = true;
        }
        result = mPlatform->acquire(swapChain, &imageSyncData);
    }

    if (result != VK_SUCCESS) {
        // We just don't set mAcquired here so the next present will just skip.
        FVK_LOGD << "Failed to acquire next image in the swapchain result=" << (int) result;
        return { false, swapchainRecreated };
    }

    // At this point acquiring the next swapchain image has succeeded

    mCurrentSwapIndex = imageSyncData.imageIndex;
    assert_invariant(mCurrentSwapIndex < mFinishedDrawing.size());
    mFinishedDrawing[mCurrentSwapIndex] = {};

    if (imageSyncData.imageReadySemaphore != VK_NULL_HANDLE) {
        mCommands->injectDependency(imageSyncData.imageReadySemaphore,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    }
    mAcquired = true;

    return { true, swapchainRecreated };
}

}// namespace filament::backend
