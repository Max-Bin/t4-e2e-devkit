# Vendored audit

`uv run python tools/vendor.py check` is the source-drift check. Public
provenance is retained only for the two public reference sources in the vendor
manifest; generated headers for private or user-provided sources contain no
identity, revision or relative path.

The remaining upstream TODOs were reviewed as follows:

| area | decision |
|---|---|
| intersection collision-area branch | retained; belongs to the proposal scorer and needs map-layer parity before changing |
| drivable-area extraction | retained; the T4 reader supplies scene-local tensors, not an external map API |
| proposal re-planning conditions | retained; unrelated to the sensor-replay runner, which replans on its configured interval |
| comfort and IDM constants | retained; changing defaults would silently change the reference metric |
| route readability and parameter loading | retained; maintenance-only, no T4 behavior change |
| compatibility typing/state aliases | retained; unused compatibility surface, not part of the T4 contract |

These files must not be edited in place to remove a TODO. A behavioral change
belongs in a devkit adapter with a regression test, or in the public upstream
source followed by a vendor sync.
