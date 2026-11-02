/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_TEST_HARNESS_H
#define ARGUS_SENDER_TEST_HARNESS_H

#include <cstdio>
#include <sstream>
#include <string>

class TestContext
{
public:
    explicit TestContext(const char* name) : m_name(name ? name : "test"), m_failures(0)
    {
    }

    void check(bool condition, const char* expression, const char* file, int line)
    {
        if (condition)
            return;
        ++m_failures;
        std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", file, line, expression);
    }

    template <typename Left, typename Right>
    void checkEqual(const Left& left,
                    const Right& right,
                    const char* leftExpression,
                    const char* rightExpression,
                    const char* file,
                    int line)
    {
        if (left == right)
            return;
        ++m_failures;
        std::ostringstream leftValue;
        std::ostringstream rightValue;
        leftValue << left;
        rightValue << right;
        std::fprintf(stderr, "%s:%d: CHECK_EQ failed: %s=%s, %s=%s\n", file, line, leftExpression,
                     leftValue.str().c_str(), rightExpression, rightValue.str().c_str());
    }

    int finish() const
    {
        if (m_failures == 0)
        {
            std::printf("PASS: %s\n", m_name.c_str());
            return 0;
        }
        std::fprintf(stderr, "FAIL: %s (%d assertion%s)\n", m_name.c_str(), m_failures, m_failures == 1 ? "" : "s");
        return 1;
    }

private:
    std::string m_name;
    int m_failures;
};

#define TEST_CHECK(context, expression) (context).check((expression), #expression, __FILE__, __LINE__)

#define TEST_CHECK_EQ(context, left, right) (context).checkEqual((left), (right), #left, #right, __FILE__, __LINE__)

#endif // ARGUS_SENDER_TEST_HARNESS_H
