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

#if defined(_WIN32)
#define XR_USE_PLATFORM_WIN32
#elif defined(__ANDROID__)
#define XR_USE_PLATFORM_ANDROID
#endif
#define XR_USE_GRAPHICS_API_VULKAN

#include <bluevk/BlueVK.h>

#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <cstdio>
#include <cstdlib>

namespace {

bool xrCheck(XrResult result, char const* what) {
    if (XR_SUCCEEDED(result)) {
        return true;
    }
    printf("[helloxr] %s failed: %d\n", what, static_cast<int>(result));
    return false;
}

} // namespace

int main(int, char**) {
    XrInstanceCreateInfo createInfo = { XR_TYPE_INSTANCE_CREATE_INFO };
    snprintf(createInfo.applicationInfo.applicationName,
            sizeof(createInfo.applicationInfo.applicationName), "helloxr");
    createInfo.applicationInfo.apiVersion = XR_API_VERSION_1_0;

    XrInstance instance = XR_NULL_HANDLE;
    if (!xrCheck(xrCreateInstance(&createInfo, &instance), "xrCreateInstance")) {
        return EXIT_FAILURE;
    }

    XrInstanceProperties props = { XR_TYPE_INSTANCE_PROPERTIES };
    if (xrCheck(xrGetInstanceProperties(instance, &props), "xrGetInstanceProperties")) {
        printf("[helloxr] runtime: %s (version %llu.%llu.%llu)\n", props.runtimeName,
                static_cast<unsigned long long>(XR_VERSION_MAJOR(props.runtimeVersion)),
                static_cast<unsigned long long>(XR_VERSION_MINOR(props.runtimeVersion)),
                static_cast<unsigned long long>(XR_VERSION_PATCH(props.runtimeVersion)));
    }

    xrDestroyInstance(instance);
    return EXIT_SUCCESS;
}
