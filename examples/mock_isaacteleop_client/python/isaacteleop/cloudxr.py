# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``CloudXRLauncher`` stand-in -- see ``GumanIsaacTeleopUsage.md`` §9: Guman only constructs it
with these kwargs and calls ``.stop()``; nothing else is read off it.
"""


class CloudXRLauncher:
    def __init__(
        self,
        install_dir: str,
        env_config: str | None = None,
        accept_eula: bool = False,
        setup_oob: bool = False,
        usb_local: bool = False,
        run_embedded: bool = True,
    ) -> None:
        self.install_dir = install_dir
        self.env_config = env_config
        self.accept_eula = accept_eula
        self.setup_oob = setup_oob
        self.usb_local = usb_local
        self.run_embedded = run_embedded

    def stop(self) -> None:
        pass
