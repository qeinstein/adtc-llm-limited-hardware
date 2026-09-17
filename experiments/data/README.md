# Training-data audit

Run:

\`\`\`bash
python scripts/audit_training_data.py \
  --manifest experiments/data/current_default.json \
  --output-dir experiments/data/reports/current_default
\`\`\`

The default manifest mirrors the default arguments of \`scripts/train_lora.py\`: MCQA once, the committed clinical chat set repeated three times, and the healthcare corpus once when present.

The auditor reports effective non-padding forward-pass tokens rather than row shares. For MCQA it counts the prompt once per candidate because \`train_lora.py\` constructs one sequence for every choice. For clinical SFT it renders the Qwen3 chat template with \`enable_thinking=False\`, appends EOS, and applies \`clinical_repeat\`. For healthcare causal-LM it reproduces the quarter-context/three-quarter-target split.

Source, category, language, and synthetic labels are read from row metadata when builders provide them. Otherwise the manifest default is used. Uncertain language classification is reported as \`other_or_uncertain\`; it is never silently treated as English.

Approximate duplicates use word-trigram SimHash with Hamming distance at most four. This is a triage signal, not proof that two examples are semantically equivalent.

Generated \`output/\` inputs are intentionally not committed. The JSON, CSV, and Markdown audit reports should be committed with each experiment manifest.

The historical semifinal v3 inputs (\`output/mcqa_v3.jsonl\` and \`output/sft_v3.json\`) are absent from the repository, so their token distribution cannot be reconstructed exactly. Row counts in scripts or reports are not a substitute for tokenizer-level measurements.
