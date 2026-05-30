.PHONY: all generate build build-dev build-release test test-unit lint lint-fix clean run verify setup help

PROJECT_NAME = GazeLock
SCHEME = GazeLock
CONFIG_DEBUG = Debug
CONFIG_RELEASE = Release
BUILD_DIR = build
DERIVED_DATA = $(BUILD_DIR)/DerivedData

XCODEBUILD = xcodebuild
XCODEBUILD_FLAGS = -project $(PROJECT_NAME).xcodeproj -scheme $(SCHEME) -derivedDataPath $(DERIVED_DATA)
# CI bypass: no signing, for fast test iterations. System extensions
# will NOT install with this — use `make build-dev` for local runs.
CODESIGN_FLAGS = CODE_SIGN_IDENTITY="-" CODE_SIGNING_REQUIRED=NO CODE_SIGNING_ALLOWED=NO ENABLE_HARDENED_RUNTIME=NO
# Dev-local signing: ad-hoc identity + entitlements attached + hardened
# runtime on. Required for system-extension installs in combination with
# `sudo systemextensionsctl developer on`.
DEVSIGN_FLAGS = CODE_SIGN_IDENTITY="-" CODE_SIGN_STYLE=Manual CODE_SIGNING_ALLOWED=YES CODE_SIGNING_REQUIRED=YES ENABLE_HARDENED_RUNTIME=YES
XCPRETTY ?= xcpretty

all: generate build

generate:
	@echo "==> Generating Xcode project from project.yml"
	@xcodegen generate
	@echo "    Done: $(PROJECT_NAME).xcodeproj"

build: generate
	@echo "==> Building $(PROJECT_NAME) (Debug, no-sign — fast iteration; system extension won't install)"
	$(XCODEBUILD) $(XCODEBUILD_FLAGS) -configuration $(CONFIG_DEBUG) $(CODESIGN_FLAGS) build

build-dev: generate
	@test -n "$(DEVELOPMENT_TEAM)" || ( \
		echo "" ; \
		echo "  ERROR: DEVELOPMENT_TEAM environment variable is not set." ; \
		echo "" ; \
		echo "  System extensions require the com.apple.developer.system-extension.install" ; \
		echo "  entitlement, which macOS only honours when the binary is signed with a real" ; \
		echo "  Apple Developer Team. Ad-hoc signing is not sufficient." ; \
		echo "" ; \
		echo "  Export your Team ID, then retry:" ; \
		echo "      export DEVELOPMENT_TEAM=<your-team-id>   # e.g. 45BHMK6XA9" ; \
		echo "      make build-dev" ; \
		echo "" ; \
		exit 1 )
	@echo "==> Building $(PROJECT_NAME) (Debug, team-signed as $(DEVELOPMENT_TEAM))"
	@echo "    If the system extension is new, also run ONCE:"
	@echo "        sudo systemextensionsctl developer on"
	$(XCODEBUILD) $(XCODEBUILD_FLAGS) -configuration $(CONFIG_DEBUG) \
		DEVELOPMENT_TEAM=$(DEVELOPMENT_TEAM) CODE_SIGN_STYLE=Automatic build

build-release: generate
	@echo "==> Building $(PROJECT_NAME) (Release)"
	$(XCODEBUILD) $(XCODEBUILD_FLAGS) -configuration $(CONFIG_RELEASE) build

test: generate
	@echo "==> Running tests"
	$(XCODEBUILD) $(XCODEBUILD_FLAGS) -configuration $(CONFIG_DEBUG) $(CODESIGN_FLAGS) test

test-unit: generate
	@echo "==> Running unit tests"
	$(XCODEBUILD) $(XCODEBUILD_FLAGS) -configuration $(CONFIG_DEBUG) $(CODESIGN_FLAGS) \
		-only-testing:GazeLockTests test

lint:
	@echo "==> Linting with SwiftLint"
	@swiftlint lint --config .swiftlint.yml --quiet

lint-fix:
	@echo "==> Auto-fixing lint issues"
	@swiftlint lint --fix --config .swiftlint.yml --quiet

clean:
	@echo "==> Cleaning build artifacts"
	@rm -rf $(BUILD_DIR)
	@rm -rf $(PROJECT_NAME).xcodeproj
	$(XCODEBUILD) $(XCODEBUILD_FLAGS) clean 2>/dev/null || true

LSREGISTER = /System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister

run: build-dev
	@echo "==> Copying to /Applications/$(PROJECT_NAME).app"
	@echo "    (macOS requires system-extension-hosting apps to live in /Applications)"
	@osascript -e 'quit app "$(PROJECT_NAME)"' 2>/dev/null ; killall $(PROJECT_NAME) 2>/dev/null ; true
	@rm -rf /Applications/$(PROJECT_NAME).app
	@cp -R "$(DERIVED_DATA)/Build/Products/$(CONFIG_DEBUG)/$(PROJECT_NAME).app" /Applications/
	@echo "==> Fixing Launch Services registration"
	@echo "    (otherwise OSSystemExtensionManager resolves the bundle from DerivedData"
	@echo "     and fails with 'Extension not found in App bundle')"
	@$(LSREGISTER) -u "$(DERIVED_DATA)/Build/Products/$(CONFIG_DEBUG)/$(PROJECT_NAME).app" 2>/dev/null || true
	@$(LSREGISTER) -f /Applications/$(PROJECT_NAME).app
	@echo "==> Launching from /Applications"
	@open /Applications/$(PROJECT_NAME).app

verify: lint test
	@echo "==> verify: lint + test passed"

setup:
	@echo "==> Installing dev tools"
	@which xcodegen > /dev/null || brew install xcodegen
	@which swiftlint > /dev/null || brew install swiftlint
	@which create-dmg > /dev/null || brew install create-dmg
	@echo "    Done. Remember to export DEVELOPMENT_TEAM."

help:
	@echo "GazeLock Makefile — available targets:"
	@echo "  make              -> generate + build (Debug)"
	@echo "  make generate     -> xcodegen generate"
	@echo "  make build        -> Debug build"
	@echo "  make build-release-> Release build"
	@echo "  make test         -> All tests"
	@echo "  make test-unit    -> Unit tests only"
	@echo "  make lint         -> SwiftLint"
	@echo "  make lint-fix     -> SwiftLint auto-fix"
	@echo "  make verify       -> lint + test (CI parity)"
	@echo "  make clean        -> Remove build artifacts + generated project"
	@echo "  make run          -> Build and open the app"
	@echo "  make setup        -> Install dev tools via Homebrew"

# -------- ML / Python pipeline --------

.PHONY: ml-setup ml-test ml-test-slow ml-lint ml-train ml-eval ml-export ml-verify

ml-setup:
	@echo "==> Installing Python dev dependencies via uv"
	@uv sync --extra dev

ml-test:
	@echo "==> Running ML test suite (fast tests only)"
	@uv run python -m pytest ML/gazelock_ml/tests -m "not slow" -v

ml-test-slow:
	@echo "==> Running ML test suite (including slow tests — Core ML export)"
	@uv run python -m pytest ML/gazelock_ml/tests -v

ml-lint:
	@echo "==> Linting ML package with ruff"
	@uv run ruff check ML/gazelock_ml

ml-train:
	@echo "==> Smoke training run (10 steps on fixtures)"
	@uv run gazelock-train --steps 10 --batch-size 4 --output-dir weights/debug

ml-eval:
	@echo "==> Evaluating weights/debug/final.pt"
	@uv run gazelock-eval --checkpoint weights/debug/final.pt

ml-export:
	@echo "==> Exporting to Core ML"
	@uv run gazelock-export --checkpoint weights/debug/final.pt --output weights/refiner.mlpackage

ml-verify: ml-lint ml-test
	@echo "==> ml-verify: lint + test passed"

# Phase 2c: Blender render pipeline
BLENDER_RENDER_OUT ?= weights/synthetic-eyes
BLENDER_RENDER_COUNT ?= 500000
BLENDER_RENDER_RES ?= 256
BLENDER_RENDER_SAMPLES ?= 64
BLENDER_SEED ?= 42
VIZ_PORT ?= 8765

.PHONY: ml-render
ml-render:
	@mkdir -p $(BLENDER_RENDER_OUT)
	blender --background --python \
		ML/gazelock_ml/synthesis/blender/render_eyes.py -- \
		--count $(BLENDER_RENDER_COUNT) \
		--output $(BLENDER_RENDER_OUT) \
		--resolution $(BLENDER_RENDER_RES) \
		--samples $(BLENDER_RENDER_SAMPLES) \
		--seed $(BLENDER_SEED) \
		--resume

.PHONY: ml-render-smoke
ml-render-smoke:
	$(MAKE) ml-render BLENDER_RENDER_COUNT=50 BLENDER_RENDER_RES=128 BLENDER_RENDER_SAMPLES=16

.PHONY: ml-fetch-assets
ml-fetch-assets:
	uv run --project ML python ML/gazelock_ml/synthesis/blender/fetch_assets.py

.PHONY: ml-render-viz
ml-render-viz:
	uv run --project ML python -m gazelock_ml.synthesis.blender.progress_viz \
		--root $(BLENDER_RENDER_OUT) \
		--total $(BLENDER_RENDER_COUNT) \
		--port $(VIZ_PORT)
