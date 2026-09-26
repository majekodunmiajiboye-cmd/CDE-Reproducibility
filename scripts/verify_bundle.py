#!/usr/bin/env python3
from pathlib import Path
import hashlib
root=Path(__file__).resolve().parents[1]
manifest=root/'checksums/SHA256SUMS'
errors=[]
for line in manifest.read_text().splitlines():
    if not line.strip(): continue
    expected, rel=line.split('  ',1); p=root/rel
    if not p.exists(): errors.append(f"missing {rel}"); continue
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    if h!=expected: errors.append(f"hash mismatch {rel}")
if errors:
    raise SystemExit("\n".join(errors))
print(f"Verified {len(manifest.read_text().splitlines())} files against SHA-256 manifest.")
