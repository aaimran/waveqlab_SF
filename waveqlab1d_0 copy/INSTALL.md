# waveqlab1d — Installation Guide

**Platform:** Linux x86_64 (tested on RHEL 9.2, `radsrvln00.utep.edu`)  
**Python:** 3.13.3 (built from source — system Python is too old)  
**Virtual env:** `wql1d` at `/work/aimran/wql1d`

---

## Prerequisites

### System packages required (check with `which`/`rpm -q`)

| Package | Purpose |
|---|---|
| `gcc`, `make` | C compiler and build tools |
| `tar`, `gzip` | Unpacking source tarballs |
| `openssl-devel` | Python `ssl` module |
| `zlib-devel` | Python `zlib` module |
| `bzip2-devel` | Python `bz2` module |
| `xz-devel` | Python `lzma` module |
| `readline-devel` | Python `readline` module |
| `ncurses-devel` | Python `curses` module |
| `git` | (optional) version control |

> **Note:** `libffi-devel` is **not** installed system-wide on this cluster, so it must be built from source (Step 1 below).

---

## Directory layout

```
/work/aimran/
├── libffi-3.4.6.tar.gz          # libffi source tarball
├── libffi-3.4.6/                # libffi build tree
├── libffi/                      # libffi install prefix
│   ├── include/                 #   ffi.h, ffitarget.h
│   └── lib64/                   #   libffi.so.8, libffi.so.8.1.4
├── Python-3.13.3.tgz            # CPython source tarball
├── Python-3.13.3/               # CPython build tree
├── python3.13/                  # CPython install prefix
│   ├── bin/python3.13
│   ├── lib/libpython3.13.so.1.0
│   └── lib/python3.13/
└── wql1d/                       # virtual environment
    ├── bin/python
    ├── bin/pip
    └── env.sh                   # environment activation script
```

---

## Step 1 — Build and install libffi 3.4.6

libffi is required by Python's `ctypes` module. The system does not provide it as a development package, so it is built from source.

```bash
cd /work/aimran

# Unpack (skip if libffi-3.4.6/ already exists)
tar xzf libffi-3.4.6.tar.gz

cd libffi-3.4.6

# Configure — install to /work/aimran/libffi
# --with-pic ensures position-independent code for shared library use
./configure \
  --prefix=/work/aimran/libffi \
  --with-pic

# Build and install
make -j$(nproc)
make install
```

**Verify:**
```bash
ls /work/aimran/libffi/lib64/libffi.so*
# Expected: libffi.so  libffi.so.8  libffi.so.8.1.4
```

---

## Step 2 — Build and install Python 3.13.3

Python must be built with `--enable-shared` (so its `.so` can be embedded later by Numba) and with the rpath to both libffi and its own lib directory baked in so no manual `LD_LIBRARY_PATH` is needed at runtime.

```bash
cd /work/aimran

# Unpack (skip if Python-3.13.3/ already exists)
tar xzf Python-3.13.3.tgz

cd Python-3.13.3

# Configure
LIBFFI=/work/aimran/libffi
PY=/work/aimran/python3.13

./configure \
  --prefix=$PY \
  --with-ensurepip=install \
  --enable-shared \
  LDFLAGS="-L${LIBFFI}/lib64 -Wl,-rpath,${LIBFFI}/lib64 -Wl,-rpath,${PY}/lib" \
  CPPFLAGS="-I${LIBFFI}/include" \
  PKG_CONFIG_PATH="${LIBFFI}/lib/pkgconfig:${LIBFFI}/lib64/pkgconfig"

# Build (uses all available cores)
make -j$(nproc)
```

### Known issue: `_sha2` module fails to link

The `_sha2` extension (SHA-256/512 via HACL*) fails to link on this system due to a missing `libHacl_Hash_SHA2.a` symbol. This is **non-critical** — it only affects Python's fallback SHA2 implementation (OpenSSL's `_hashlib` is used in practice). The `make install` target aborts because it cannot find the `.so` to install.

**Workaround — copy built extensions manually:**
```bash
DYNLOAD=/work/aimran/python3.13/lib/python3.13/lib-dynload

# Copy all successfully built extensions
for f in build/lib.linux-x86_64-3.13/*.so; do
  cp -f "$f" "$DYNLOAD/$(basename $f)"
done

# The critical one is _ctypes — verify:
ls $DYNLOAD/_ctypes*.so
```

**Verify Python works:**
```bash
LD_LIBRARY_PATH=/work/aimran/python3.13/lib:/work/aimran/libffi/lib64 \
  /work/aimran/python3.13/bin/python3.13 -c \
  "import _ctypes, ctypes; print('ctypes OK')"
```

> After the venv's `env.sh` is sourced (Step 4), `LD_LIBRARY_PATH` is set automatically.

---

## Step 3 — Create the `wql1d` virtual environment

```bash
/work/aimran/python3.13/bin/python3.13 -m venv /work/aimran/wql1d
```

Install all required packages (use the local pip cache to avoid repeated downloads):
```bash
/work/aimran/wql1d/bin/pip install \
  --cache-dir /scratch/aimran/pip-cache \
  numpy==2.4.4 \
  numba==0.65.0 \
  llvmlite==0.47.0 \
  matplotlib==3.10.8
```

**Full package list (pinned versions):**

| Package | Version | Purpose |
|---|---|---|
| `numpy` | 2.4.4 | Array operations, simulation state |
| `numba` | 0.65.0 | JIT compilation of kernels (`@njit`, `prange`) |
| `llvmlite` | 0.47.0 | LLVM backend for Numba |
| `matplotlib` | 3.10.8 | Plotting / visualisation |
| `pillow` | 12.2.0 | Image output (GIF animations) |
| `contourpy` | 1.3.3 | Matplotlib dependency |
| `cycler` | 0.12.1 | Matplotlib dependency |
| `fonttools` | 4.62.1 | Matplotlib dependency |
| `kiwisolver` | 1.5.0 | Matplotlib dependency |
| `python-dateutil` | 2.9.0.post0 | Matplotlib dependency |
| `pyparsing` | 3.3.2 | Matplotlib dependency |
| `packaging` | 26.0 | pip/build dependency |
| `six` | 1.17.0 | python-dateutil dependency |

---

## Step 4 — Activate the environment

Source the activation script before running any simulation:

```bash
source /work/aimran/wql1d/env.sh
```

This sets:

| Variable | Value |
|---|---|
| `PATH` | `wql1d/bin`, `python3.13/bin` prepended |
| `LD_LIBRARY_PATH` | `python3.13/lib`, `libffi/lib64` prepended |
| `NUMBA_CACHE_DIR` | `/work/aimran/wql1d/.numba_cache` |
| `NUMBA_NUM_THREADS` | Number of physical CPU cores (`nproc`) |
| `OMP_NUM_THREADS` | Same as above |

Expected output:
```
wql1d env active — Python 3.13.3, numba 0.65.0, threads=56
```

---

## Step 5 — Run the baseline test

```bash
source /work/aimran/wql1d/env.sh
cd /scratch/aimran/FNO/waveqlab_SF/waveqlab1d
python rupture_1d.py input/rupture_1d_SW.in
```

Expected output (abridged):
```
  step    577/577  t=4.9971s  slip=18.8898m  sliprate=4.0221m/s

total simulation time  = 0.24 s  (excl. 23.15 s JIT warmup)
spatial order          = 6
number of grid points  = 501
numba threads          = 1
Saved: output/rupture_SW_<run_id>.npz
```

> **First run:** Numba JIT warmup takes ~20 s. Subsequent runs with the same code use the on-disk cache and start in <1 s.

---

## Automation

All steps above can be run unattended using the provided `Makefile`:

```bash
cd /scratch/aimran/FNO/waveqlab_SF/waveqlab1d
make install        # Steps 1–3: libffi → Python → venv + packages
make verify         # Steps 4–5: source env and run baseline test
make clean-build    # Remove libffi and Python build trees (keeps installs)
make uninstall      # Remove installs and venv (destructive — asks for confirmation)
```

---

## Troubleshooting

### `No module named '_ctypes'`
The `_ctypes` extension was not copied into `lib-dynload`. Run:
```bash
cp /work/aimran/Python-3.13.3/build/lib.linux-x86_64-3.13/_ctypes*.so \
   /work/aimran/python3.13/lib/python3.13/lib-dynload/
```

### `libpython3.13.so.1.0: cannot open shared object file`
The rpath was not baked in or the library was not installed. Source `env.sh` which sets `LD_LIBRARY_PATH`, or verify the rpath with:
```bash
readelf -d /work/aimran/python3.13/bin/python3.13 | grep rpath
```

### `cannot cache function: no locator available`
Numba cannot cache functions compiled from a `-c` string. This only occurs in interactive one-liners — it is **not** an error when running `.py` files.

### Numba warnings about `NUMBA_NUM_THREADS`
If the environment variable was set before Numba was imported (e.g., from a previous shell), Numba may warn. Unset and re-source `env.sh`:
```bash
unset NUMBA_NUM_THREADS OMP_NUM_THREADS
source /work/aimran/wql1d/env.sh
```
