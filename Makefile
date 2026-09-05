# AI YouTube Editor — developer entry points.
#   make setup                 create .venv and install the package (editable)
#   make new NAME=x LANG=pl    create a project
#   make ingest NAME=x         normalize clips, build proxies/audio/peaks/thumbs
#   make status NAME=x         show clips, stages and spend
#   make validate NAME=x       validate plan/timeline.json
#   make transcribe NAME=x     ElevenLabs Scribe -> transcripts/
#   make analyze NAME=x        per-clip analysis + footage log
#   make plan NAME=x           edit plan + draft timeline (NOTES="..." optional)
#   make tidy NAME=x           pad cuts around speech on the existing timeline
#   make denoise NAME=x CLIP=c004 [ENGINE=elevenlabs|local]  voice isolation for a windy clip
#   make music NAME=x          generate music beds from the plan
#   make preview NAME=x        fast 720p render -> renders/preview.mp4
#   make master NAME=x         full master render -> exports/master_1080p.mp4
#   make qc NAME=x             playbook rule checker + loudness
#   make publish NAME=x        titles, description, chapters, thumbnails
#   make run NAME=x            ingest -> transcribe -> analyze -> plan (UNTIL=music)
#   make list                  list projects
#   make probe FILE=clip.mov   print probed media properties
#   make serve                 run the local web editor
#   make test                  generate fixtures and run pytest

UV      ?= $(HOME)/.local/bin/uv
VENV    ?= .venv
PY      := $(VENV)/bin/python
YTEDIT  := $(VENV)/bin/ytedit
PYTEST  := $(VENV)/bin/pytest
NAME    ?=
LANG    ?= pl
TITLE   ?=
HOST    ?= 127.0.0.1
PORT    ?= 8765

.DEFAULT_GOAL := help
.PHONY: help setup guard-name new ingest status validate transcribe analyze plan tidy denoise music preview master qc publish run list probe serve test fixtures clean-fixtures clean

help:
	@grep -E '^#   ' $(MAKEFILE_LIST) | sed 's/^#   //'

setup:
	$(UV) venv $(VENV)
	$(UV) pip install --python $(PY) -e ".[dev]"
	@$(YTEDIT) --help > /dev/null && echo "ytedit installed: $(YTEDIT)"

guard-name:
	@test -n "$(NAME)" || (echo "NAME is required, e.g. make ingest NAME=my-video"; exit 2)

new: guard-name
	$(YTEDIT) new $(NAME) --language $(LANG) $(if $(TITLE),--title "$(TITLE)",)

ingest: guard-name
	$(YTEDIT) ingest $(NAME) $(if $(FORCE),--force,)

status: guard-name
	$(YTEDIT) status $(NAME)

validate: guard-name
	$(YTEDIT) validate $(NAME)

transcribe: guard-name
	$(YTEDIT) transcribe $(NAME) $(if $(FORCE),--force,)

analyze: guard-name
	$(YTEDIT) analyze $(NAME) $(if $(FORCE),--force,)

plan: guard-name
	$(YTEDIT) plan $(NAME) $(if $(FORCE),--force,) $(if $(NOTES),--notes "$(NOTES)",)

tidy: guard-name
	$(YTEDIT) tidy $(NAME) $(if $(FORCE),--force,)

ENGINE ?= elevenlabs
denoise: guard-name
	@test -n "$(CLIP)" || (echo "CLIP is required, e.g. make denoise NAME=my-video CLIP=c004"; exit 2)
	$(YTEDIT) denoise $(NAME) --clip $(CLIP) --engine $(ENGINE) --preview $(if $(FORCE),--force,)

music: guard-name
	$(YTEDIT) music $(NAME) $(if $(FORCE),--force,)

preview: guard-name
	$(YTEDIT) render $(NAME) --preview

master: guard-name
	$(YTEDIT) render $(NAME) --master

qc: guard-name
	$(YTEDIT) qc $(NAME)

publish: guard-name
	$(YTEDIT) publish $(NAME)

UNTIL ?= plan
run: guard-name
	$(YTEDIT) run $(NAME) --until $(UNTIL) $(if $(FORCE),--force,)

list:
	$(YTEDIT) list

probe:
	@test -n "$(FILE)" || (echo "FILE is required, e.g. make probe FILE=clip.mov"; exit 2)
	$(YTEDIT) probe "$(FILE)"

serve:
	$(YTEDIT) serve --host $(HOST) --port $(PORT)

fixtures:
	$(PY) tests/fixtures/make_fixtures.py

clean-fixtures:
	rm -rf tests/fixtures/generated

clean: clean-fixtures
	rm -rf .pytest_cache **/__pycache__

test: fixtures
	$(PYTEST)
