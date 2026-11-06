# SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

add_executable(omniglove_hand_conversion_contract_test hand_conversion_contract_test.cpp)
target_link_libraries(omniglove_hand_conversion_contract_test PRIVATE isaac::omniglove_plugin)

add_test(NAME omniglove_hand_conversion_contract_test COMMAND omniglove_hand_conversion_contract_test)
