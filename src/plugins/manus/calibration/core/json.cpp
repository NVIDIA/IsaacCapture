// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "json.hpp"

#include <algorithm>
#include <cstdio>

namespace plugins::manus::calibration::json
{

namespace
{

class Reader
{
public:
    explicit Reader(std::string_view text) : m_text(text)
    {
    }

    ParseResult run()
    {
        Members members;
        skip_space();
        if (!consume('{'))
        {
            return fail("expected a JSON object");
        }
        skip_space();
        if (!consume('}'))
        {
            while (true)
            {
                skip_space();
                std::string key;
                if (!read_string(key))
                {
                    return fail(m_error.empty() ? "expected a string key" : m_error);
                }
                if (std::any_of(members.begin(), members.end(), [&](const auto& m) { return m.first == key; }))
                {
                    return fail("duplicate key \"" + key + "\"");
                }
                skip_space();
                if (!consume(':'))
                {
                    return fail("expected ':' after key");
                }
                skip_space();
                Value value;
                if (!read_value(value))
                {
                    return fail(m_error.empty() ? "unsupported value" : m_error);
                }
                members.emplace_back(std::move(key), std::move(value));
                skip_space();
                if (consume('}'))
                {
                    break;
                }
                if (!consume(','))
                {
                    return fail("expected ',' or '}'");
                }
            }
        }
        skip_space();
        if (m_pos != m_text.size())
        {
            return fail("trailing bytes after the object");
        }
        return { std::move(members), {} };
    }

private:
    static ParseResult fail(std::string message)
    {
        return { std::nullopt, std::move(message) };
    }

    bool consume(char expected)
    {
        if (m_pos < m_text.size() && m_text[m_pos] == expected)
        {
            ++m_pos;
            return true;
        }
        return false;
    }

    void skip_space()
    {
        while (m_pos < m_text.size() &&
               (m_text[m_pos] == ' ' || m_text[m_pos] == '\t' || m_text[m_pos] == '\r' || m_text[m_pos] == '\n'))
        {
            ++m_pos;
        }
    }

    bool read_value(Value& value)
    {
        if (m_pos >= m_text.size())
        {
            return false;
        }
        const char c = m_text[m_pos];
        if (c == '"')
        {
            std::string s;
            if (!read_string(s))
            {
                return false;
            }
            value = std::move(s);
            return true;
        }
        if (c == '[')
        {
            return read_string_array(value);
        }
        if (c == '-' || (c >= '0' && c <= '9'))
        {
            return read_integer(value);
        }
        m_error = "unsupported value type";
        return false;
    }

    bool read_string_array(Value& value)
    {
        ++m_pos;
        std::vector<std::string> items;
        skip_space();
        if (consume(']'))
        {
            value = std::move(items);
            return true;
        }
        while (true)
        {
            skip_space();
            std::string item;
            if (!read_string(item))
            {
                m_error = "array elements must be strings";
                return false;
            }
            items.push_back(std::move(item));
            skip_space();
            if (consume(']'))
            {
                break;
            }
            if (!consume(','))
            {
                m_error = "expected ',' or ']'";
                return false;
            }
        }
        value = std::move(items);
        return true;
    }

    bool read_integer(Value& value)
    {
        const size_t start = m_pos;
        consume('-');
        const size_t digits_start = m_pos;
        while (m_pos < m_text.size() && m_text[m_pos] >= '0' && m_text[m_pos] <= '9')
        {
            ++m_pos;
        }
        const size_t digit_count = m_pos - digits_start;
        // No fraction, exponent or leading zero; at most 18 digits keeps the value inside int64_t.
        const bool leading_zero = digit_count > 1 && m_text[digits_start] == '0';
        const bool float_like = m_pos < m_text.size() && (m_text[m_pos] == '.' || m_text[m_pos] == 'e' ||
                                                          m_text[m_pos] == 'E');
        if (digit_count == 0 || digit_count > 18 || leading_zero || float_like)
        {
            m_error = "numbers must be plain integers";
            return false;
        }
        int64_t parsed = 0;
        for (size_t i = digits_start; i < m_pos; ++i)
        {
            parsed = parsed * 10 + (m_text[i] - '0');
        }
        value = (m_text[start] == '-') ? -parsed : parsed;
        return true;
    }

    bool read_string(std::string& out)
    {
        if (!consume('"'))
        {
            return false;
        }
        while (m_pos < m_text.size())
        {
            const unsigned char c = static_cast<unsigned char>(m_text[m_pos++]);
            if (c == '"')
            {
                return true;
            }
            if (c < 0x20)
            {
                m_error = "control character in string";
                return false;
            }
            if (c != '\\')
            {
                out.push_back(static_cast<char>(c));
                continue;
            }
            if (!read_escape(out))
            {
                return false;
            }
        }
        m_error = "unterminated string";
        return false;
    }

    bool read_escape(std::string& out)
    {
        if (m_pos >= m_text.size())
        {
            m_error = "unterminated escape";
            return false;
        }
        const char e = m_text[m_pos++];
        switch (e)
        {
        case '"':
        case '\\':
        case '/':
            out.push_back(e);
            return true;
        case 'b':
            out.push_back('\b');
            return true;
        case 'f':
            out.push_back('\f');
            return true;
        case 'n':
            out.push_back('\n');
            return true;
        case 'r':
            out.push_back('\r');
            return true;
        case 't':
            out.push_back('\t');
            return true;
        case 'u':
            return read_unicode_escape(out);
        default:
            m_error = "invalid escape";
            return false;
        }
    }

    bool read_hex4(uint32_t& value)
    {
        if (m_pos + 4 > m_text.size())
        {
            return false;
        }
        value = 0;
        for (int i = 0; i < 4; ++i)
        {
            const char h = m_text[m_pos++];
            value <<= 4;
            if (h >= '0' && h <= '9')
            {
                value |= uint32_t(h - '0');
            }
            else if (h >= 'a' && h <= 'f')
            {
                value |= uint32_t(h - 'a' + 10);
            }
            else if (h >= 'A' && h <= 'F')
            {
                value |= uint32_t(h - 'A' + 10);
            }
            else
            {
                return false;
            }
        }
        return true;
    }

    bool read_unicode_escape(std::string& out)
    {
        uint32_t code = 0;
        if (!read_hex4(code))
        {
            m_error = "invalid \\u escape";
            return false;
        }
        if (code >= 0xD800 && code <= 0xDBFF)
        {
            // A high surrogate must be followed by an escaped low surrogate.
            uint32_t low = 0;
            if (m_pos + 2 > m_text.size() || m_text[m_pos] != '\\' || m_text[m_pos + 1] != 'u')
            {
                m_error = "unpaired surrogate";
                return false;
            }
            m_pos += 2;
            if (!read_hex4(low) || low < 0xDC00 || low > 0xDFFF)
            {
                m_error = "unpaired surrogate";
                return false;
            }
            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
        }
        else if (code >= 0xDC00 && code <= 0xDFFF)
        {
            m_error = "unpaired surrogate";
            return false;
        }
        append_utf8(out, code);
        return true;
    }

    static void append_utf8(std::string& out, uint32_t code)
    {
        if (code < 0x80)
        {
            out.push_back(static_cast<char>(code));
        }
        else if (code < 0x800)
        {
            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        }
        else if (code < 0x10000)
        {
            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        }
        else
        {
            out.push_back(static_cast<char>(0xF0 | (code >> 18)));
            out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        }
    }

    std::string_view m_text;
    size_t m_pos = 0;
    std::string m_error;
};

// Length of the valid UTF-8 sequence starting at `text[pos]`, or 0 if the bytes there are not valid.
size_t utf8_sequence_length(std::string_view text, size_t pos)
{
    const auto byte = [&](size_t i) { return static_cast<unsigned char>(text[i]); };
    const unsigned char lead = byte(pos);
    size_t length = 0;
    uint32_t min_code = 0;
    uint32_t code = 0;
    if (lead < 0x80)
    {
        return 1;
    }
    if ((lead & 0xE0) == 0xC0)
    {
        length = 2;
        min_code = 0x80;
        code = lead & 0x1F;
    }
    else if ((lead & 0xF0) == 0xE0)
    {
        length = 3;
        min_code = 0x800;
        code = lead & 0x0F;
    }
    else if ((lead & 0xF8) == 0xF0)
    {
        length = 4;
        min_code = 0x10000;
        code = lead & 0x07;
    }
    else
    {
        return 0;
    }
    if (pos + length > text.size())
    {
        return 0;
    }
    for (size_t i = 1; i < length; ++i)
    {
        if ((byte(pos + i) & 0xC0) != 0x80)
        {
            return 0;
        }
        code = (code << 6) | (byte(pos + i) & 0x3F);
    }
    // Reject overlong forms, surrogates and values past U+10FFFF.
    if (code < min_code || code > 0x10FFFF || (code >= 0xD800 && code <= 0xDFFF))
    {
        return 0;
    }
    return length;
}

} // namespace

ParseResult parse_object(std::string_view text)
{
    return Reader(text).run();
}

void append_string(std::string& out, std::string_view text)
{
    out.push_back('"');
    size_t pos = 0;
    while (pos < text.size())
    {
        const size_t length = utf8_sequence_length(text, pos);
        if (length == 0)
        {
            out += "\xEF\xBF\xBD";
            ++pos;
            continue;
        }
        if (length > 1)
        {
            out.append(text.substr(pos, length));
            pos += length;
            continue;
        }
        const char c = text[pos++];
        switch (c)
        {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20)
            {
                char escaped[8];
                std::snprintf(escaped, sizeof(escaped), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
                out += escaped;
            }
            else
            {
                out.push_back(c);
            }
        }
    }
    out.push_back('"');
}

} // namespace plugins::manus::calibration::json
