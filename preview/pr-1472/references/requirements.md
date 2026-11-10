# System Requirements

Isaac Teleop can be used for teleoperation of robots, data collection, and human-centric data
collection. The hardware & software requirements are different for each use case.

## Teleoperation to Robots with Input Devices

![Hardware requirements for teleoperation with extra input devices](_static/hardware-req-02.svg)

When using extra input devices (such as Manus Gloves, Logitech Rudder Pedals, OAK-D Camera, Haptikos Exoskeletons etc.),
Isaac Teleop needs to be **run on a laptop or workstation**, that is connected to the robot via
network.  The minimum requirements for the laptop/workstation are:

| Component     | Requirement                                              |
|---------------|----------------------------------------------------------|
| CPU / GPU     | x86_64 <sup>[1](#jetson-nano)</sup>, NVIDIA GPU required |
| OS            | Ubuntu 22.04 or 24.04                                    |
| Python        | 3.11 / 3.12 / 3.13                                       |
| CUDA          | 12.8 or newer                                            |
| NVIDIA Driver | 580.95.05 or newer                                       |

## Teleoperation with Isaac Sim and Isaac Lab

![Hardware requirements for teleoperation with Isaac Sim and Isaac Lab](_static/hardware-req-03.svg)

For running simulation with Isaac Sim and Isaac Lab with RTX rendering, the CloudXR server and
Isaac Teleop sessions need to be **run on the same workstation with Isaac Sim and Isaac Lab**. The
workstation’s system requirements are driven by Isaac Sim and Isaac Lab <sup>[2](#isaacsim-req)</sup>.

The recommended workstation configuration for Sim-based Teleop is:

| Component     | Requirement                                                |
|---------------|------------------------------------------------------------|
| CPU           | AMD Ryzen Threadripper 7960x                               |
| GPU           | 1x RTX 6000 Pro (Blackwell) or 2x RTX 6000 (Ada)           |
| OS            | Ubuntu 22.04 or 24.04 <sup>[3](#isaaclab-req)</sup>        |
| Python        | 3.12 <sup>[3](#isaaclab-req)</sup>                         |
| CUDA          | 12.8 or newer                                              |
| NVIDIA Driver | Per Isaac Sim’s requirements <sup>[2](#isaacsim-req)</sup> |

If you are only using XR headsets for teleoperation, you can host the workstation in the cloud.
See [Isaac Lab Cloud Deployment](https://isaac-sim.github.io/IsaacLab/develop/source/features/docker_cloud.html#cloud-workstations)
for more details.

## Human-centric Data Collection

Isaac Teleop can also be used for human-centric data collection without a robot or simulator in the
loop. In this case, Isaac Teleop needs to be **run on a laptop or workstation**, that is connected
to device peripherals. The minimum requirements for the laptop/workstation are:

![Hardware requirements for human-centric data collection](_static/hardware-req-04.svg)

| Component     | Requirement                                              |
|---------------|----------------------------------------------------------|
| CPU / GPU     | x86_64 <sup>[1](#jetson-nano)</sup>, NVIDIA GPU required |
| OS            | Ubuntu 22.04 or 24.04                                    |
| Python        | 3.11 / 3.12 / 3.13                                       |
| CUDA          | 12.8 or newer                                            |
| NVIDIA Driver | 580.95.05 or newer                                       |

### Footnotes

* <a id='jetson-nano'>**[1]**</a> Jetson Nano support is being considered. See [#271](https://github.com/NVIDIA/IsaacTeleop/issues/271) for more details. Please up-vote and/or comment on the issue if you are interested in this feature.
* <a id='isaacsim-req'>**[2]**</a> Please refer to [Isaac Sim System Requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html) for more details.
* <a id='isaaclab-req'>**[3]**</a> Please refer to [Isaac Lab System Requirements](https://isaac-sim.github.io/IsaacLab/develop/source/setup/installation/index.html#system-requirements) for more details.
