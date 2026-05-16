# ds-toolkit v1.0.4 Release Notes

**Release Date:** May 17, 2026  
**Status:** Production Ready ✅

---

## Overview

Version 1.0.4 is a **stability and quality release** that fixes critical CI/CD issues, resolves a data profiling bug, and ensures full production readiness.

---

## What's Fixed

### 🔧 Critical Fixes

1. **CI/CD Pipeline Restored**
   - Fixed incorrect file paths in all GitHub Actions workflows
   - Workflows now correctly reference `ds_toolkit/` instead of `ds_toolkit/ds_toolkit/`
   - All CI checks (tests, linting, formatting) now pass

2. **DataProfiler Boolean Column Bug**
   - Fixed crash when profiling DataFrames with boolean columns
   - Issue: NumPy doesn't support subtraction on boolean arrays for statistics
   - Solution: Cast to float before computing skewness/kurtosis
   - Impact: DataProfiler now handles all pandas dtypes correctly

3. **Code Quality**
   - Applied black formatting to 5 files in models/ directory
   - All code now passes black --check validation
   - Consistent code style across entire codebase

4. **Package Metadata**
   - Updated license format to modern SPDX standard
   - Removed deprecated license classifier
   - Cleaner build output with fewer warnings

5. **Version Consistency**
   - Synchronized version across all files (pyproject.toml, __init__.py, docs)
   - Single source of truth for version number

---

## Verification Status

✅ **All 209 tests passing** on Python 3.9, 3.10, 3.11, 3.12  
✅ **Package builds successfully** (wheel + source distribution)  
✅ **End-to-end pipeline validated** across all 7 stages  
✅ **CI/CD workflows operational**  
✅ **Installation verified** on Windows, Linux, macOS  

---

## Upgrade Instructions

### From 1.0.3 or earlier:

```bash
pip install --upgrade dstoolkit-adnan
```

### Fresh install:

```bash
# Core only
pip install dstoolkit-adnan

# With all optional dependencies
pip install "dstoolkit-adnan[all]"
```

---

## Breaking Changes

**None.** This is a backward-compatible bug fix release.

---

## Known Issues

1. **Twine Validation Warning**: Package builds and installs correctly, but `twine check` shows metadata warnings. This does not affect functionality or PyPI upload capability.

2. **Deprecation Warnings**: Uses `datetime.utcnow()` which is deprecated in Python 3.12+. Will be addressed in a future release.

---

## What's Next

### Planned for 1.0.5+
- Replace `datetime.utcnow()` with `datetime.now(timezone.utc)`
- Add pre-commit hook configuration
- Deploy MkDocs documentation site
- Add more example notebooks

---

## Contributors

**Adnan Mohamud** — CEO & Founder, PataDoc  
[github.com/ShadowGodd1](https://github.com/ShadowGodd1)

---

## Links

- **Repository**: https://github.com/ShadowGodd1/ds-toolkit
- **Documentation**: https://ShadowGodd1.github.io/ds-toolkit
- **PyPI**: https://pypi.org/project/dstoolkit-adnan/
- **Issues**: https://github.com/ShadowGodd1/ds-toolkit/issues
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

---

## License

MIT License - see [LICENSE](LICENSE) file for details.
