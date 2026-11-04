// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

// Quick smoke test: does libargus see this board's SENSING camera at all, and can it
// pull one frame? Run directly on the Orin with the camera connected -- this has no
// dependency on anything else in the repo, just libargus + EGL.
//
// Usage: argus_sensing_probe [device_index]   (default device 0)
//
// Exit 0 + "PROBE: PASS" means Argus enumerated the device and captured a frame.
// Any other outcome prints exactly where it failed.

#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>
#include <EGL/egl.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace Argus;

namespace {

void printSensorMode(uint32_t index, SensorMode *mode) {
  ISensorMode *iMode = interface_cast<ISensorMode>(mode);
  if (!iMode) return;
  Size2D<uint32_t> res = iMode->getResolution();
  Range<uint64_t> dur = iMode->getFrameDurationRange();
  double maxFps = dur.min() ? 1e9 / static_cast<double>(dur.min()) : 0.0;
  std::printf("    mode %u: %ux%u, up to %.1f fps\n", index, res.width(), res.height(), maxFps);
}

bool fail(const char *where) {
  std::fprintf(stderr, "PROBE: FAIL at %s\n", where);
  return false;
}

}  // namespace

int main(int argc, char **argv) {
  uint32_t deviceIndex = 0;
  if (argc > 1) deviceIndex = static_cast<uint32_t>(std::atoi(argv[1]));

  // --- 1. Create the provider and enumerate devices -------------------------------
  UniqueObj<CameraProvider> provider(CameraProvider::create());
  ICameraProvider *iProvider = interface_cast<ICameraProvider>(provider);
  if (!iProvider) return fail("CameraProvider::create / ICameraProvider") , 1;

  std::vector<CameraDevice *> devices;
  if (iProvider->getCameraDevices(&devices) != STATUS_OK || devices.empty()) {
    fail("getCameraDevices (no devices visible to Argus at all)");
    return 1;
  }
  std::printf("PROBE: Argus sees %zu camera device(s)\n", devices.size());

  for (uint32_t i = 0; i < devices.size(); ++i) {
    ICameraProperties *iProps = interface_cast<ICameraProperties>(devices[i]);
    std::printf("  device %u%s\n", i, i == deviceIndex ? "  <-- probing this one" : "");
    if (!iProps) continue;
    std::vector<SensorMode *> modes;
    if (iProps->getAllSensorModes(&modes) == STATUS_OK) {
      for (uint32_t m = 0; m < modes.size(); ++m) printSensorMode(m, modes[m]);
    }
  }

  if (deviceIndex >= devices.size()) {
    std::fprintf(stderr, "PROBE: FAIL device index %u requested, only %zu available\n",
                 deviceIndex, devices.size());
    return 1;
  }

  ICameraProperties *iProps = interface_cast<ICameraProperties>(devices[deviceIndex]);
  std::vector<SensorMode *> modes;
  if (!iProps || iProps->getAllSensorModes(&modes) != STATUS_OK || modes.empty()) {
    return fail("getAllSensorModes on selected device"), 1;
  }
  SensorMode *sensorMode = modes[0];
  ISensorMode *iSensorMode = interface_cast<ISensorMode>(sensorMode);
  Size2D<uint32_t> resolution = iSensorMode->getResolution();
  std::printf("PROBE: using device %u, mode 0 (%ux%u)\n", deviceIndex, resolution.width(),
              resolution.height());

  // --- 2. Minimal headless EGL display (no window/X11 needed on the Orin) ---------
  EGLDisplay eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (eglDisplay == EGL_NO_DISPLAY || !eglInitialize(eglDisplay, nullptr, nullptr)) {
    return fail("eglGetDisplay/eglInitialize"), 1;
  }

  // --- 3. Capture session + one EGL output stream ----------------------------------
  UniqueObj<CaptureSession> session(iProvider->createCaptureSession(devices[deviceIndex]));
  ICaptureSession *iSession = interface_cast<ICaptureSession>(session);
  if (!iSession) return fail("createCaptureSession"), 1;

  UniqueObj<OutputStreamSettings> streamSettings(
      iSession->createOutputStreamSettings(STREAM_TYPE_EGL));
  IEGLOutputStreamSettings *iStreamSettings =
      interface_cast<IEGLOutputStreamSettings>(streamSettings);
  if (!iStreamSettings) return fail("createOutputStreamSettings"), 1;
  iStreamSettings->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
  iStreamSettings->setResolution(resolution);
  iStreamSettings->setEGLDisplay(eglDisplay);
  iStreamSettings->setMode(EGL_STREAM_MODE_FIFO);

  UniqueObj<OutputStream> outputStream(iSession->createOutputStream(streamSettings.get()));
  if (!outputStream) return fail("createOutputStream"), 1;

  UniqueObj<EGLStream::FrameConsumer> consumer(
      EGLStream::FrameConsumer::create(outputStream.get()));
  EGLStream::IFrameConsumer *iFrameConsumer =
      interface_cast<EGLStream::IFrameConsumer>(consumer);
  if (!iFrameConsumer) return fail("FrameConsumer::create"), 1;

  // --- 4. Build and submit a one-shot request ---------------------------------------
  UniqueObj<Request> request(iSession->createRequest());
  IRequest *iRequest = interface_cast<IRequest>(request);
  if (!iRequest) return fail("createRequest"), 1;
  if (iRequest->enableOutputStream(outputStream.get()) != STATUS_OK) {
    return fail("enableOutputStream"), 1;
  }
  ISourceSettings *iSourceSettings = interface_cast<ISourceSettings>(iRequest->getSourceSettings());
  if (iSourceSettings) iSourceSettings->setSensorMode(sensorMode);

  if (iSession->capture(request.get()) == 0) {
    return fail("capture() -- request was rejected"), 1;
  }

  // --- 5. Wait for the one frame back ------------------------------------------------
  Status status = STATUS_OK;
  UniqueObj<EGLStream::Frame> frame(iFrameConsumer->acquireFrame(5000000000ULL /* 5s */, &status));
  if (!frame || status != STATUS_OK) {
    std::fprintf(stderr, "PROBE: FAIL acquireFrame, status=%d\n", static_cast<int>(status));
    return 1;
  }
  EGLStream::IFrame *iFrame = interface_cast<EGLStream::IFrame>(frame);
  if (!iFrame || !iFrame->getImage()) {
    return fail("acquired frame has no image"), 1;
  }

  std::printf("PROBE: PASS -- captured one %ux%u frame from device %u\n", resolution.width(),
              resolution.height(), deviceIndex);

  eglTerminate(eglDisplay);
  return 0;
}
