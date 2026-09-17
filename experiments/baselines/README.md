# ADTC semifinal baseline

This is the immutable ADTC semifinal checkpoint. Future models are compared against this baseline.

The frozen artifact is `Fluxx08/jamii-afya-qwen3-0.6b` revision `db25331f85c8aab975a65efdc87af5e3ca14036d`, file `Qwen3-0.6B-Q4_0.gguf`, SHA-256 `c8519086e799fb60a4888ec51e87f78a35c27bad5554d74f68707f09590593ad`.

Use the revision-pinned URL in `semifinal.json`. Do not use a moving `/resolve/main/` URL when reproducing baseline measurements, and never replace the artifact under this baseline identifier.

The manifest deliberately distinguishes artifact identity from metric provenance. The committed Gate-1 profiler record was produced from repository commit `b0a976f61a195916a3264e7c52360b301d89257f`, before the hosted GGUF was replaced by the current v3 file. It does not contain a GGUF checksum. Therefore its throughput and memory numbers are retained as the semifinal headline measurements but are marked as not cryptographically linked to the current v3 artifact.

Likewise, accuracy values in `REPORT.md` are retained as reported measurements because raw per-example results and exact run manifests were not committed. The 300-example ARC Easy and MedMCQA values are consistent with the current Hugging Face v3 revision title, but the missing raw results prevent stronger provenance claims.

Any future baseline rerun must write a new result record containing the GGUF SHA-256, repository SHA, exact evaluator version, dataset revision, prompt format, and machine information. It must not edit `semifinal.json` to replace the historical record.
