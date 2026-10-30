# Draft: preserve prior raw evidence when a write fails

Actual filesystem fault injection interrupts `Path.write_text` after seven characters. Old `_save_raw` truncates the existing response JSON and parsing fails. The candidate keeps the previous complete JSON and cleans its temporary file.

Source-only baseline: 1 FAIL / 1 PASS. Candidate focused suite: 21 PASS, including write failure, replace failure, permissions and normal fetch/retry paths.

The same-directory exclusively created temporary file honors the process umask. Existing target permissions are preserved before atomic replacement. JSON serialization, output bytes and returned path are unchanged.

This guards runtime write failure. It does not claim fsync/power-loss durability or concurrent-write ordering.

Locked full suite: 337 PASS / 1 known saved-ensemble freshness FAIL. Merge remains blocked. No dependency, evaluation or uploader changes. Atomic publication adds filesystem operations; there is no performance-equivalence claim. Not merge-ready.

Current base: `27cb203`. Separate from #59 METAR and #58 comparison-date fixes.
