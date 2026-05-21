"""v13 — deterministic readers for previously-unsupported file extensions.

v10 had 6 tasks blocked by `needs_file_capability:unsupported_extension:*`
across mp3, zip, pdb. v13 adds deterministic readers:

- audio_read (mp3 / wav / m4a): Together's Whisper transcription endpoint
  (free with existing API key — no new key required).
- zip_read: recursively decompress and dispatch each contained file by its
  inner extension via the existing read_gaia_file machinery.
- pdb_read: extract header/title lines as plain text (PDB files are plain
  text with structured records — HEADER, TITLE, COMPND fields).

All emit standard `tool.called` / `tool.returned` / `source.opened` events
so the (claim, source) protocol consumes them like any other source.
"""
