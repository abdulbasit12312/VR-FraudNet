# Convenience targets. Everything here is also a plain command; see README.md.

PY ?= python

.PHONY: help install test lint demo audit tables clean check

help:
	@echo "install   install runtime + dev dependencies"
	@echo "test      run the full test suite"
	@echo "lint      byte-compile every module and parse every config"
	@echo "demo      run the synthetic end-to-end demo (no data required)"
	@echo "audit     rerun every manuscript consistency check"
	@echo "tables    regenerate manuscript tables from results/"
	@echo "check     lint + test + audit (the release gate)"
	@echo "clean     remove caches and byte-code"

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install pytest

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m compileall -q src scripts tests examples
	$(PY) -c "import yaml,glob;[yaml.safe_load(open(f,encoding='utf-8')) for f in glob.glob('configs/*.yaml')+glob.glob('verifier/rules/*.yaml')];print('configs parse')"
	$(PY) -c "import json,glob;[json.load(open(f,encoding='utf-8')) for f in glob.glob('schemas/*.json')+glob.glob('examples/*.json')];print('schemas parse')"

demo:
	$(PY) examples/run_minimal_demo.py

audit:
	$(PY) scripts/audit_manuscript.py

tables:
	$(PY) scripts/make_tables.py --table all

check: lint test audit

clean:
	$(PY) -c "import shutil,pathlib;[shutil.rmtree(p,ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PY) -c "import shutil;shutil.rmtree('.pytest_cache',ignore_errors=True)"
