# High-relevance session image links

Open the **Export** tab, choose **Minimum relevance score**, then select
**Export high-relevance CSV…**. The native save dialog lets you choose the folder
and filename and confirms replacement of an existing CSV. Export works while
monitoring is active and after Stop. It does not change the playback threshold.

The initial cutoff is **0.70** for metadata, context CLIP and learned-context
ranking, or **0.35** for the narrower raw CLIP score scale. These are editable
starting points, not calibrated definitions of correctness. Scores from different
ranking modes are not directly comparable. The CSV records the mode used for
each saved score. An image qualifies when its highest recorded session score is
**greater than or equal to** the chosen cutoff. There is no top-N fallback that
silently includes low-scoring images; the button is disabled when nothing matches.

## What the session contains

The ledger collects **accepted image batches**, including images waiting in the
queue, rather than every unfiltered search hit or only the last displayed images.
It starts when the application window opens. It is not a reconstruction of
previous sessions or the thumbnail history. One row is kept per exact direct
image URL; the best score and its associated credits/query/visual subjects stay
together. First and last seen times refer to accepted batches. Signed URL query
parameters are preserved, so different URL variants remain distinct.

Stop/resume, topic changes, queue replacement and clearing the image cache retain
the links. **New book / clear context**, **Delete session history**, and closing
the window clear the temporary ledger. Export before those actions. Explicitly
saved CSV files are not deleted, and existing story-retention settings do not
restore the temporary image ledger on relaunch.

## Columns and privacy

`session_id`, `first_seen_utc`, `last_seen_utc`, `relevance_score`,
`export_min_score`, `ranking_mode`, `image_url`, `source_url`, `title`, `provider`,
`creator`, `license`, `license_url`, `query`, `visual_subjects`.

Rows are sorted by highest score first, then image URL for deterministic ties.
The file is UTF-8 with a BOM for spreadsheet compatibility. CSV fields are quoted,
and formula-like text is prefixed with an apostrophe so provider-controlled titles
or credits cannot become spreadsheet formulas. Invalid or credential-bearing
HTTP(S) image URLs are excluded; unsafe source/license links are left blank.
Attribution and source licenses are preserved without implying permission to reuse.

The ledger uses a local temporary SQLite database with a bounded page cache,
not an unbounded list of image objects or pixel data. It stores no audio,
transcript, complete narrative, manual guidance, description, or local cache path.
Search queries and visual subjects can still reveal what you are listening to;
review the CSV before sharing it. Only an explicit export creates a retained CSV.
Temporary database records are deleted on normal close; abrupt process/OS failure
can leave temporary filesystem remnants. There are no new network requests.

Exports use a temporary file in the selected destination folder and atomically
replace the final CSV after writing succeeds. A write failure keeps any existing
CSV and recorded links intact. A recording failure leaves monitoring running and
shows an explicit incomplete-export warning rather than claiming a complete log.

## Implementation and checks

`session_image_export.py` contains the standard-library ledger and exporter.
`session_export_ui.py` extends the existing monitoring window without changing
its capture, ranking, fullscreen, context, or attribution logic. The normal
`__main__` entry point launches this export-enabled window, including installed
console scripts and the packaged launcher.

```sh
python -m pytest -q tests/test_session_image_export.py tests/test_session_export_ui.py
```

The UI tests use Qt's offscreen platform, local image fixtures and mocked device
discovery. They do not start audio capture, download models, or contact providers.
