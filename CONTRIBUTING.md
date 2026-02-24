# Contributing to PneumoDetect AI

Thank you for considering contributing to PneumoDetect AI. This project welcomes contributions from engineers, researchers, and medical professionals. Please take a moment to review these guidelines to make the collaboration smooth.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Branch Strategy](#branch-strategy)
- [Commit Conventions](#commit-conventions)
- [Pull Request Process](#pull-request-process)
- [Code Standards](#code-standards)
- [Issue Templates](#issue-templates)

---

## Code of Conduct

Please read and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). We are committed to maintaining a welcoming and inclusive community.

---

## How to Contribute

### Types of Contributions Welcome

| Type | Examples |
|---|---|
| 🧠 **Model Improvements** | New backbone architectures, training optimizations |
| 📊 **Validation Expansion** | Multi-center, multi-age-group, multi-pathology testing |
| 🔌 **API Enhancements** | New endpoints, SDK wrappers, DICOM integration |
| 🎨 **UI/UX** | Streamlit improvements, accessibility |
| 📝 **Documentation** | Tutorials, notebooks, API docs |
| 🔒 **Security** | Dependency updates, vulnerability fixes |
| 🧪 **Tests** | Unit tests, integration tests, CI improvements |

### Reporting Issues

Before opening an issue:
1. Search existing issues to avoid duplicates
2. Use the appropriate issue template (Bug, Feature Request, Security)
3. Provide a minimal reproducible example for bugs

---

## Development Setup

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/Pneumonia-Detection-ai.git
cd Pneumonia-Detection-ai

# 3. Add upstream remote
git remote add upstream https://github.com/thriniiiiiiiiiiii/Pneumonia-Detection-ai.git

# 4. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

# 5. Install dependencies
pip install -r requirements.txt
pip install ruff flake8 pytest pytest-cov pip-audit

# 6. Verify setup
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## Branch Strategy

```
main          ← Production-ready code. PRs only. Protected.
develop       ← Integration branch for active development.
feat/*        ← New features (e.g., feat/subtype-classification)
fix/*         ← Bug fixes (e.g., fix/gradcam-shape-error)
docs/*        ← Documentation updates
ci/*          ← CI/CD pipeline changes
refactor/*    ← Code restructuring (no behavior change)
```

**Always branch from `develop`, not `main`.**

```bash
git checkout develop
git pull upstream develop
git checkout -b feat/your-feature-name
```

---

## Commit Conventions

This project uses **Conventional Commits** (enforced by CI):

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

### Types

| Type | When to Use |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change that neither fixes nor adds |
| `test` | Adding or fixing tests |
| `ci` | CI/CD pipeline changes |
| `chore` | Maintenance, dependency bumps |
| `perf` | Performance improvement |
| `security` | Security-related fix |

### Examples

```bash
# Good commits
git commit -m "feat(model): add EfficientNetB0 as alternative backbone"
git commit -m "fix(api): handle DICOM files larger than 5MB correctly"
git commit -m "docs(api): add JavaScript SDK example to API.md"
git commit -m "ci: add pip-audit security scan to CI pipeline"
git commit -m "test(inference): add unit tests for confidence level mapping"

# Bad commits (❌ will be flagged in PR review)
git commit -m "fixed stuff"
git commit -m "update"
git commit -m "WIP"
```

---

## Pull Request Process

1. **Keep PRs focused**: One feature or fix per PR
2. **Update documentation**: If your change modifies behavior, update the relevant docs in `docs/`
3. **Add tests**: Where applicable, add unit tests in `tests/`
4. **Pass all CI checks**: Lint, security audit, Docker build must pass
5. **Fill out the PR template**: Describe the change, motivation, and testing done
6. **Request review**: Tag at least one maintainer

### PR Checklist

```markdown
- [ ] Code follows project style (ruff passes)
- [ ] Added/updated tests for the change
- [ ] All CI checks pass
- [ ] Documentation updated if behavior changed
- [ ] Commit messages follow Conventional Commits
- [ ] No secrets, credentials, or dataset files committed
```

---

## Code Standards

### Python Style

- **Formatter**: `ruff format` (Black-compatible)
- **Linter**: `ruff check` + `flake8`
- **Line length**: 120 characters
- **Type hints**: Required for all public functions

```python
# Good
def predict_diagnosis(image_array: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Run pneumonia inference on a preprocessed image array.

    Args:
        image_array: Normalized float32 array of shape (1, 224, 224, 3).
        threshold: Classification threshold (default 0.5).

    Returns:
        dict with keys: diagnosis, confidence, raw_score.
    """
    score = float(model.predict(image_array)[0][0])
    return {
        "diagnosis": "PNEUMONIA" if score > threshold else "NORMAL",
        "confidence": round(score * 100, 2),
        "raw_score": score,
    }
```

### Run linting locally

```bash
ruff check .
ruff format --check .
flake8 scripts/ api/ --max-line-length=120
pip-audit -r requirements.txt
```

---

## Semantic Versioning

This project follows [Semantic Versioning](https://semver.org/):

| Version Bump | When |
|---|---|
| `MAJOR` (v2.0.0) | Breaking API changes |
| `MINOR` (v1.1.0) | New backward-compatible features |
| `PATCH` (v1.0.1) | Bug fixes, dependency updates |

---

## Questions?

Open a [GitHub Discussion](https://github.com/thriniiiiiiiiiiii/Pneumonia-Detection-ai/discussions) or contact the maintainer: [ayushirathour1804@gmail.com](mailto:ayushirathour1804@gmail.com)
