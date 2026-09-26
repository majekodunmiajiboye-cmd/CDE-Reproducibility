PYTHON ?= python3
.PHONY: verify audit fetch
verify:
	$(PYTHON) scripts/verify_bundle.py
audit:
	$(PYTHON) scripts/verify_reported_values.py
fetch:
	$(PYTHON) scripts/fetch_public_inputs.py
