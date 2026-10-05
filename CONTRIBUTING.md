# Contributing

Thanks for your interest in improving this project! Issues, forks and pull requests are welcome.

## Workflow

1. **Fork** the repository and clone your fork.
2. Create a branch from `main` named after the change:
   - `feature/<short-description>` for new functionality
   - `fix/<short-description>` for bug fixes
   - `docs/<short-description>` for documentation
3. Make small, focused commits (see the conventions below).
4. Run the notebook from top to bottom in Google Colab to check it still works.
5. Open a **pull request** against `main` that describes what changed and why.

## Commit messages

This project follows [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short summary in the imperative mood>
```

| Type | Use for |
|---|---|
| `feat` | New functionality |
| `fix` | Bug fixes |
| `docs` | Documentation only |
| `refactor` | Code changes that don't change behaviour |
| `chore` | Maintenance (dependencies, project setup) |

Examples:

```
feat: add token-based chunking
fix: stop the last chunk from duplicating the overlap
docs: explain how MIN_SCORE was calibrated
```

## Versioning and releases

- Versions follow [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.
- Every release is recorded in [CHANGELOG.md](CHANGELOG.md) and tagged in git (for example `v1.0.0`).

## Reporting issues

Open an issue with:
- what you ran (command or notebook cell and settings),
- what you expected,
- what happened (error message or output).

## Development setup

The project is a Google Colab notebook. Upload `notebooks/semantic_search_colab.ipynb` and a text file to Colab and run all cells; the notebook installs its own dependencies (listed in `requirements.txt`).
