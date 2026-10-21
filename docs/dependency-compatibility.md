# OSWorld dependency compatibility

`desktop-env==1.0.2` supplies the OSWorld environment and evaluators. Its NumPy,
Pillow, and pandas requirements are runtime compatibility constraints:

| Package | Required version range | Resolved version |
| ------- | ---------------------- | ---------------- |
| NumPy   | `>=1.26,<2`            | `1.26.4`         |
| Pillow  | `~=11.0.0`             | `11.0.0`         |
| pandas  | `>=2.2,<2.3`           | `2.2.3`          |

Commit `f2d70b9` deliberately replaced NumPy 2 with NumPy 1 when introducing
`desktop-env`. The upgrade in `2c606c9` bypassed those evaluator constraints to
avoid local source builds. Restore the upstream bounds and address build issues
within them. A successful installation with newer dependencies does not verify
OSWorld evaluator compatibility. No project feature requires NumPy 2.

Keep these requirements active in dependency resolution. The exclusions for
unused pygame and matplotlib remain separate from the evaluator dependencies.
The matplotlib constraint applies to the version required by other project
dependencies. The borb pin addresses case-colliding wheel paths on macOS.

## Python and native builds

The project targets Python 3.14. The latest NumPy 1 release, 1.26.4, officially
[supports Python 3.9 through 3.12](https://numpy.org/doc/2.3/release/1.26.4-notes.html).
Its source build on Python 3.14 is outside that supported range. Older Pillow and
pandas releases also require source builds where compatible wheels are absent.
Retaining the constraints does not establish that those builds will succeed.

`uv lock` resolves the restored versions. A local `uv sync --locked` attempt on
macOS arm64 with Python 3.14.8 fails building Pillow 11.0.0 because JPEG headers
and libraries are missing. An isolated `uv pip install --no-deps numpy==1.26.4`
native build succeeds on the same interpreter, and its isolated import, array
sum, and matrix multiplication checks pass. This does not validate OSWorld
evaluators or the pandas build. Install the required native libraries or agree
on a supported runtime before claiming the full
environment works. Preserve the evaluator bounds while resolving these issues.
