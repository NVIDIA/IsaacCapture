# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional test certificates valid at both an older client clock and the host."""

from datetime import datetime, timedelta, timezone
import ipaddress
import os


def make_backdated_certificates(directory, address, not_before):
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    except ImportError as exc:
        raise ValueError(
            "Install the bridge [network] extra for --tls-not-before"
        ) from exc
    start = datetime.strptime(not_before, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    if start > now:
        raise ValueError("Certificate start date must not be in the future")
    # Expiration follows generation time, not the older start date.
    expires = now + timedelta(days=7)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "Remote Assistance Test CA")]
    )
    leaf_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, address)])

    def builder(subject, public_key, ca):
        return (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(ca_name)
            .public_key(public_key)
            .serial_number(x509.random_serial_number())
            .not_valid_before(start)
            .not_valid_after(expires)
            .add_extension(
                x509.BasicConstraints(ca=ca, path_length=0 if ca else None),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=not ca,
                    content_commitment=False,
                    key_encipherment=not ca,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=ca,
                    crl_sign=ca,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
        )

    ca = builder(ca_name, ca_key.public_key(), True).sign(ca_key, hashes.SHA256())
    leaf = (
        builder(leaf_name, leaf_key.public_key(), False)
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.IPv4Address(address))]
            ),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )
    files = {
        "ca.crt": ca.public_bytes(serialization.Encoding.PEM),
        "server.crt": leaf.public_bytes(serialization.Encoding.PEM),
    }
    for name, key in (("ca.key", ca_key), ("server.key", leaf_key)):
        files[name] = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    for name, data in files.items():
        with open(
            directory / name,
            "xb",
            opener=lambda path, flags: os.open(path, flags, 0o600),
        ) as stream:
            stream.write(data)
