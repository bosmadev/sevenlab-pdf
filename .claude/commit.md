# Commit

## Ready

chore: remove dead sandbox/ duplicate

backend/sandbox/ (sandbox_entry.py, run_sandbox.sh, Dockerfile.sandbox,
requirements.sandbox.txt) was an untracked, un-imported duplicate of
app/services/pdf_sandbox.py referring to non-existent sandbox_runner.py
and app/services/sandbox.py. Deleted and gitignored.

