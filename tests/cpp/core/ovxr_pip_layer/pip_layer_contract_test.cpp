// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Single translation unit so the application layer's private state is exercised
// against mocked OpenXR dispatch without a GPU or a second XR session.
// The package is built in Release; its contract checks must still execute.
#undef NDEBUG
#define ISAAC_TELEOP_PIP_TEST
#include "pip_layer.cpp"

#include <cassert>
#include <cstdio>
#include <vector>

namespace
{
int acquireCalls = 0;
int waitCalls = 0;
int releaseCalls = 0;
bool firstWaitTimeout = false;
bool releaseFails = false;
bool destroySwapchainFails = false;
bool destroySpaceFails = false;
int createSpaceCalls = 0;
int vkMapCalls = 0;
int vkBufferDestroys = 0;
int vkFenceWaits = 0;
bool vkFenceWaitFails = false;
bool vkIdleFails = false;
std::vector<int> destroyOrder;
std::vector<XrSwapchain> forwardedSwapchains;

XrResult XRAPI_CALL acquire(XrSwapchain, const XrSwapchainImageAcquireInfo*, uint32_t* index)
{
    ++acquireCalls;
    *index = 7; // Forces an exit after the swapchain state transition.
    return XR_SUCCESS;
}
XrResult XRAPI_CALL wait(XrSwapchain, const XrSwapchainImageWaitInfo*)
{
    ++waitCalls;
    return firstWaitTimeout && waitCalls == 1 ? XR_TIMEOUT_EXPIRED : XR_SUCCESS;
}
XrResult XRAPI_CALL release(XrSwapchain, const XrSwapchainImageReleaseInfo*)
{
    ++releaseCalls;
    return releaseFails ? XR_ERROR_RUNTIME_FAILURE : XR_SUCCESS;
}
XrResult XRAPI_CALL destroySwapchain(XrSwapchain)
{
    destroyOrder.push_back(1);
    return destroySwapchainFails ? XR_ERROR_RUNTIME_FAILURE : XR_SUCCESS;
}
XrResult XRAPI_CALL unavailableFormats(XrSession, uint32_t, uint32_t*, int64_t*)
{
    return XR_ERROR_RUNTIME_FAILURE;
}
int resizeCreates = 0;
bool imageEnumerationFails = true;
XrResult XRAPI_CALL availableFormats(XrSession, uint32_t capacity, uint32_t* count, int64_t* formats)
{
    *count = 1;
    if (capacity)
        formats[0] = VK_FORMAT_R8G8B8A8_UNORM;
    return XR_SUCCESS;
}
XrResult XRAPI_CALL createResizeSwapchain(XrSession, const XrSwapchainCreateInfo*, XrSwapchain* swapchain)
{
    *swapchain = reinterpret_cast<XrSwapchain>(uintptr_t{ 100 } + ++resizeCreates);
    return XR_SUCCESS;
}
XrResult XRAPI_CALL resizeImages(XrSwapchain, uint32_t capacity, uint32_t* count, XrSwapchainImageBaseHeader* images)
{
    *count = 1;
    if (capacity)
    {
        if (imageEnumerationFails)
            return XR_ERROR_RUNTIME_FAILURE;
        reinterpret_cast<XrSwapchainImageVulkanKHR*>(images)[0].image = VkImage(99);
    }
    return XR_SUCCESS;
}
XrResult XRAPI_CALL destroySpace(XrSpace)
{
    destroyOrder.push_back(2);
    return destroySpaceFails ? XR_ERROR_RUNTIME_FAILURE : XR_SUCCESS;
}
XrResult XRAPI_CALL createSpace(XrSession, const XrReferenceSpaceCreateInfo*, XrSpace* space)
{
    if (++createSpaceCalls == 2)
        return XR_ERROR_RUNTIME_FAILURE;
    *space = reinterpret_cast<XrSpace>(uintptr_t{ 30 });
    return XR_SUCCESS;
}
XrResult XRAPI_CALL testGipa(XrInstance, const char* name, PFN_xrVoidFunction* function)
{
    const std::pair<std::string_view, PFN_xrVoidFunction> functions[] = {
        { "xrCreateReferenceSpace", reinterpret_cast<PFN_xrVoidFunction>(createSpace) },
        { "xrDestroySpace", reinterpret_cast<PFN_xrVoidFunction>(destroySpace) },
        { "xrEnumerateSwapchainFormats", reinterpret_cast<PFN_xrVoidFunction>(availableFormats) },
        { "xrCreateSwapchain", reinterpret_cast<PFN_xrVoidFunction>(createResizeSwapchain) },
        { "xrDestroySwapchain", reinterpret_cast<PFN_xrVoidFunction>(destroySwapchain) },
        { "xrEnumerateSwapchainImages", reinterpret_cast<PFN_xrVoidFunction>(resizeImages) },
        { "xrAcquireSwapchainImage", reinterpret_cast<PFN_xrVoidFunction>(acquire) },
        { "xrWaitSwapchainImage", reinterpret_cast<PFN_xrVoidFunction>(wait) },
        { "xrReleaseSwapchainImage", reinterpret_cast<PFN_xrVoidFunction>(release) },
    };
    for (const auto& [entry, address] : functions)
    {
        if (entry == name)
        {
            *function = address;
            return XR_SUCCESS;
        }
    }
    return XR_ERROR_FUNCTION_UNSUPPORTED;
}
} // namespace

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(VkDevice,
                                                         const VkBufferCreateInfo*,
                                                         const VkAllocationCallbacks*,
                                                         VkBuffer* buffer)
{
    *buffer = VkBuffer(11);
    return VK_SUCCESS;
}
extern "C" VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements(VkDevice, VkBuffer, VkMemoryRequirements* req)
{
    req->size = 64;
    req->memoryTypeBits = 1;
}
extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice,
                                                                          VkPhysicalDeviceMemoryProperties* props)
{
    props->memoryTypeCount = 1;
    props->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
}
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkAllocateMemory(VkDevice,
                                                           const VkMemoryAllocateInfo*,
                                                           const VkAllocationCallbacks*,
                                                           VkDeviceMemory* memory)
{
    *memory = VkDeviceMemory(12);
    return VK_SUCCESS;
}
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize)
{
    return VK_SUCCESS;
}
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkMapMemory(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags, void** data)
{
    static uint32_t bytes = 0;
    ++vkMapCalls;
    if (vkMapCalls == 1)
    {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    *data = &bytes;
    return VK_SUCCESS;
}
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateCommandPool(VkDevice,
                                                              const VkCommandPoolCreateInfo*,
                                                              const VkAllocationCallbacks*,
                                                              VkCommandPool* pool)
{
    *pool = VkCommandPool(13);
    return VK_SUCCESS;
}
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkAllocateCommandBuffers(VkDevice,
                                                                   const VkCommandBufferAllocateInfo*,
                                                                   VkCommandBuffer* command)
{
    *command = VkCommandBuffer(14);
    return VK_SUCCESS;
}
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateFence(VkDevice,
                                                        const VkFenceCreateInfo*,
                                                        const VkAllocationCallbacks*,
                                                        VkFence* fence)
{
    *fence = VkFence(15);
    return VK_SUCCESS;
}
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t)
{
    ++vkFenceWaits;
    return vkFenceWaitFails ? VK_TIMEOUT : VK_SUCCESS;
}
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkDeviceWaitIdle(VkDevice)
{
    return vkIdleFails ? VK_TIMEOUT : VK_SUCCESS;
}
extern "C" VKAPI_ATTR void VKAPI_CALL vkDestroyFence(VkDevice, VkFence, const VkAllocationCallbacks*)
{
}
extern "C" VKAPI_ATTR void VKAPI_CALL vkDestroyCommandPool(VkDevice, VkCommandPool, const VkAllocationCallbacks*)
{
}
extern "C" VKAPI_ATTR void VKAPI_CALL vkUnmapMemory(VkDevice, VkDeviceMemory)
{
}
extern "C" VKAPI_ATTR void VKAPI_CALL vkDestroyBuffer(VkDevice, VkBuffer, const VkAllocationCallbacks*)
{
    ++vkBufferDestroys;
}
extern "C" VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*)
{
}

#if defined(ISAAC_TELEOP_PIP_CUDA) && defined(__linux__)
namespace
{
int sameDeviceCopies = 0;
int peerCopies = 0;
cudaError_t copyResult = cudaSuccess;
cudaError_t syncResult = cudaSuccess;
int cudaSyncCalls = 0;
cudaMemcpy3DPeerParms peerCopy{};
}
extern "C" cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync(
    void*, size_t pitch, cudaArray_const_t, size_t, size_t, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t)
{
    ++sameDeviceCopies;
    assert(pitch == 640 * 4 && width == pitch && height == 480 && kind == cudaMemcpyDeviceToDevice);
    return copyResult;
}
extern "C" cudaError_t CUDARTAPI cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms* parameters, cudaStream_t)
{
    ++peerCopies;
    peerCopy = *parameters;
    return copyResult;
}
extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t)
{
    ++cudaSyncCalls;
    return syncResult;
}
#endif

int main()
{
#if defined(ISAAC_TELEOP_PIP_CUDA) && defined(__linux__)
    CameraSession copyCamera{};
    copyCamera.cudaDevice = 0;
    copyCamera.cudaPixels = reinterpret_cast<void*>(uintptr_t{ 42 });
    const void* sourceArray = reinterpret_cast<void*>(uintptr_t{ 43 });
    assert(copyCameraArray(copyCamera, sourceArray, 640, 480, 0) == cudaSuccess);
    assert(sameDeviceCopies == 1 && peerCopies == 0);
    assert(copyCameraArray(copyCamera, sourceArray, 640, 480, 1) == cudaSuccess);
    assert(sameDeviceCopies == 1 && peerCopies == 1);
    assert(peerCopy.srcDevice == 1 && peerCopy.dstDevice == 0);
    assert(peerCopy.srcArray == sourceArray && peerCopy.dstPtr.ptr == copyCamera.cudaPixels);
    assert(peerCopy.extent.width == 640 && peerCopy.extent.height == 480 && peerCopy.extent.depth == 1);
    assert(peerCopy.dstPtr.pitch == 640 * 4 && peerCopy.dstPtr.xsize == 640 * 4);
    copyResult = cudaErrorInvalidDevice;
    assert(copyCameraArray(copyCamera, sourceArray, 640, 480, 1) == cudaErrorInvalidDevice);
    assert(copyCameraArrayAndWait(copyCamera, sourceArray, 640, 480, 1) == -6);
    assert(cudaSyncCalls == 1 && !copyCamera.cudaPending && !copyCamera.retired);
    syncResult = cudaErrorUnknown;
    assert(copyCameraArrayAndWait(copyCamera, sourceArray, 640, 480, 1) == -7);
    assert(cudaSyncCalls == 2 && copyCamera.cudaPending && copyCamera.retired);
    copyResult = syncResult = cudaSuccess;
#endif
    const auto instance = reinterpret_cast<XrInstance>(uintptr_t{ 1 });
    const auto session = reinterpret_cast<XrSession>(uintptr_t{ 2 });
    Dispatch dispatch{};
    dispatch.acquire = acquire;
    dispatch.wait = wait;
    dispatch.release = release;
    dispatch.destroySwapchain = destroySwapchain;
    dispatch.destroySpace = destroySpace;
    dispatch.formats = unavailableFormats;
    gInstances.emplace(instance, dispatch);
    CameraSession camera{};
    camera.instance = instance;
    camera.session = session;
    camera.view = reinterpret_cast<XrSpace>(uintptr_t{ 3 });
    camera.local = reinterpret_cast<XrSpace>(uintptr_t{ 4 });
    camera.swapchain = reinterpret_cast<XrSwapchain>(uintptr_t{ 5 });
    camera.width = camera.height = 1;
    camera.ready = true;
    // Resize cannot forget an old swapchain whose destruction failed. No new
    // GPU allocation is needed to exercise this native ownership boundary.
    CameraSession resizing = camera;
    destroySwapchainFails = true;
    const auto previousSwapchain = resizing.swapchain;
    assert(!createSwapchain(resizing, dispatch, 2, 2));
    assert(resizing.swapchain == previousSwapchain && resizing.retired && !resizing.ready);
    destroySwapchainFails = false;
    // A partially created resized swapchain must not advertise the old extent.
    // Return to the old size after failed enumeration: it still must recreate.
    resizing = camera;
    Dispatch resizeDispatch = dispatch;
    resizeDispatch.formats = availableFormats;
    resizeDispatch.createSwapchain = createResizeSwapchain;
    resizeDispatch.images = resizeImages;
    assert(!createSwapchain(resizing, resizeDispatch, 2, 2));
    assert(resizing.width == 0 && resizing.height == 0 && !resizing.ready);
    imageEnumerationFails = false;
    assert(createSwapchain(resizing, resizeDispatch, 1, 1));
    assert(resizeCreates == 2 && resizing.width == 1 && resizing.images[0].image == VkImage(99));
    assert(releaseSessionResources(resizing, resizeDispatch));
    destroyOrder.clear();
    CameraSlots cameras{};
    cameras[0] = camera;
    gSessions.emplace(session, cameras);

    assert(isaacteleop_pip_attach_v3(instance, session, nullptr, {}, {}, {}, 0) == -1);
    const auto* layer = isaacteleop_pip_layer_v3(0);
    assert(layer && layer->type == XR_TYPE_COMPOSITION_LAYER_QUAD && layer->layerFlags == 0);
    assert(isaacteleop_pip_layer_v3(1) == nullptr);
    assert(isaacteleop_pip_layer_v3(3) == nullptr);
    const auto* quad = reinterpret_cast<const XrCompositionLayerQuad*>(layer);
    assert(quad->subImage.swapchain == camera.swapchain && quad->eyeVisibility == XR_EYE_VISIBILITY_BOTH);

    // All three descriptors stay alive through the caller's submit, in caller order.
    for (size_t i = 1; i < kMaxCameras; ++i)
    {
        gSessions.at(session)[i] = camera;
        gSessions.at(session)[i].swapchain = reinterpret_cast<XrSwapchain>(uintptr_t{ 5 } + i);
        gSessions.at(session)[i].view = reinterpret_cast<XrSpace>(uintptr_t{ 30 } + i);
        gSessions.at(session)[i].local = XR_NULL_HANDLE;
    }
    const auto* second = isaacteleop_pip_layer_v3(1);
    const auto* third = isaacteleop_pip_layer_v3(2);
    assert(second && third && second != layer && second != third);
    assert(quad->subImage.swapchain == camera.swapchain);
    gPanels[1].enabled = false;
    assert(isaacteleop_pip_layer_v3(1) == nullptr);
    assert(isaacteleop_pip_layer_v3(0) == layer && isaacteleop_pip_layer_v3(2) == third);

    auto& owned = gSessions.at(session)[0];
    uint32_t rgba = 0;
    owned.pixels = &rgba;
#if defined(ISAAC_TELEOP_PIP_CUDA) && defined(__linux__)
    owned.device = VkDevice(24);
    owned.cudaDevice = 1;
    owned.queue = VkQueue(25);
    owned.width = owned.height = 1;
    assert(isaacteleop_pip_publish_rgba_v2(0, &rgba, 1, 1, 4) == -9);
    assert(owned.pixels == &rgba); // Reject without overwriting any resource.
    owned.cudaDevice = -1;
    owned.device = VK_NULL_HANDLE;
#endif
    firstWaitTimeout = true;
    assert(!upload(owned, dispatch, &rgba));
    assert(acquireCalls == 1 && waitCalls == 1 && releaseCalls == 0);
    assert(owned.acquired && !owned.waited && !owned.ready);
    assert(!upload(owned, dispatch, &rgba));
    assert(acquireCalls == 1 && waitCalls == 2 && releaseCalls == 1);
    assert(!owned.acquired && !owned.ready);

    // Model a fresh session for the independent release-failure cell.
    owned.retired = false;
    releaseFails = true;
    assert(!upload(owned, dispatch, &rgba));
    assert(acquireCalls == 2 && releaseCalls == 2);
    assert(owned.retired && !owned.ready);

    CameraSession partial{};
    partial.device = VkDevice(21);
    partial.physical = VkPhysicalDevice(22);
    partial.width = partial.height = 1;
    assert(!createUploadResources(partial));
    assert(!partial.staging && !partial.memory && !partial.pixels);
    assert(vkBufferDestroys == 1);
    assert(createUploadResources(partial));
    assert(partial.staging && partial.memory && partial.pixels && partial.pool && partial.fence);
    partial.fencePending = true;
    vkFenceWaitFails = true;
    vkIdleFails = true;
    assert(!releaseImageResources(partial));
    assert(partial.staging && partial.memory && partial.fence);
    assert(vkBufferDestroys == 1);
    vkIdleFails = false;
    assert(!releaseImageResources(partial)); // Another queue's idle is not our completion contract.
    assert(partial.staging && partial.memory && partial.fence);
    vkFenceWaitFails = false;
    assert(releaseImageResources(partial));
    assert(!partial.staging && !partial.memory && !partial.fence);
    assert(vkBufferDestroys == 2 && vkFenceWaits == 3);

    // This is application-owned storage in this test, not Vulkan storage.
    owned.pixels = nullptr;
    const int waitsBeforeClose = waitCalls;
    const int releasesBeforeClose = releaseCalls;
    destroySwapchainFails = true;
    assert(isaacteleop_pip_detach_v3() == -1);
    assert(gSessions.size() == 1); // Failed destruction still retains the owner.
    destroySwapchainFails = false;
    assert(isaacteleop_pip_detach_v3() == 0);
    // No application work remains. Destroying an acquired swapchain does not
    // require waiting for a runtime image that may never become available.
    assert(waitCalls == waitsBeforeClose && releaseCalls == releasesBeforeClose);
    assert(gSessions.empty() && gInstances.empty());
    assert((destroyOrder == std::vector<int>{ 1, 1, 2, 2, 1, 2, 1, 2 }));
    assert(isaacteleop_pip_layer_v3(0) == nullptr);
    assert(isaacteleop_pip_detach_v3() == 0);
    // Partial reference-space creation cannot lose the only cleanup owner.
    destroySpaceFails = true;
    assert(isaacteleop_pip_attach_v3(instance, session, testGipa, VkPhysicalDevice(1), VkDevice(2), VkQueue(3), 0) == -3);
    assert(gSessions.size() == 1 && gInstances.size() == 1);
    assert(gSessions.at(session)[0].view);
    assert(isaacteleop_pip_detach_v3() == -1);
    assert(gSessions.at(session)[0].view);
    destroySpaceFails = false;
    assert(isaacteleop_pip_detach_v3() == 0);
    assert(gSessions.empty() && gInstances.empty());
    std::puts("pip layer contract: PASS");
}
