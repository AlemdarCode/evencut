# Security policy

## Supported versions

Fixes go into the latest release only. That is 1.0.x today.

## Reporting a vulnerability

Please report it privately on GitHub: open the repository's **Security** tab and choose
**Report a vulnerability**. Do not open a public issue for a security problem.

Please include:

- the version, and which part is affected: the Python package, the native library (C ABI), the
  `evencut` CLI, the HTTP service or the Docker image;
- the smallest input or the steps that show the problem;
- what an attacker gains: a crash, wrong output, reading or writing data they should not.

Evencut is maintained by one person, so a reply can take some time. It comes in the private
report. When a fix is released, the advisory is published and you are credited, unless you would
rather not be.

## What counts

In scope:

- memory-safety bugs in the C++ engine or the C ABI that input can reach (crashes, out-of-bounds
  reads or writes), and integer overflows that silently give a wrong partition;
- the HTTP service accepting a request past its limits (`EVENCUT_MAX_ITEMS`,
  `EVENCUT_MAX_BODY_BYTES`) or being taken down by a single request;
- the CLI, `evencut-train` or the Python package running code, or reading or writing files, beyond
  what the command was asked to do.

Documented behaviour, not a vulnerability:

- the HTTP service has no authentication. Bind it to localhost, or put it behind your own gateway
  (see the README);
- `evencut-train probe` runs a model on the GPUs of the machine it is started on and reads
  `nvidia-smi` or `amd-smi`. It trusts that machine;
- time, cost or energy predictions that turn out wrong. Please open a normal issue for those.
