# Kaggle execution

Heavy Phase 0/1 work runs on Kaggle. Local execution is limited to repository inspection, lightweight unit tests, and report/schema validation.

Push a notebook with:

\`\`\`bash
kaggle kernels push -p kaggle/phase01-data-audit
\`\`\`

Check status and download outputs with:

\`\`\`bash
kaggle kernels status toheebogunade/jamii-afya-phase-01-data-audit
kaggle kernels output toheebogunade/jamii-afya-phase-01-data-audit \
  -p output/kaggle-phase01-data-audit
\`\`\`

The notebook is evaluation-only. It does not start fine-tuning.
