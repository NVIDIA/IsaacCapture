<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mock camera images

The mock backend shows these animated GIFs on its simulated sensors. They are not part of the
Apache-2.0 licensed source: they are Animated Noto Emoji, © Google, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (`LICENSES/CC-BY-4.0.txt`).

| File | Emoji | Source |
| --- | --- | --- |
| `noto-robot.gif` | Robot, U+1F916 | `https://fonts.gstatic.com/s/e/notoemoji/latest/1f916/512.gif` |
| `noto-waving-hand.gif` | Waving hand, U+1F44B | `https://fonts.gstatic.com/s/e/notoemoji/latest/1f44b/512.gif` |

Licence and artwork: <https://googlefonts.github.io/noto-emoji-files/>. The files are the unmodified
512 px GIFs. The mock composites their transparent background onto a plain gray when it decodes
them; the files themselves are not changed.

The mock backend is a development tool. Keep these files out of anything shipped to users.
