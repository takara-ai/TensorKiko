# TensorKiko Release Instructions

## Pushing a New Release

Follow these steps to push a new release of TensorKiko:

1. **Update Version Number**:

   - Open `setup.py` or `__init__.py` (wherever your version is defined).
   - Increment the version number according to [Semantic Versioning](https://semver.org/) principles.

2. **Update CHANGELOG.md**:

   - Add a new entry for the release version.
   - List all notable changes, new features, bug fixes, and breaking changes.

3. **Commit Changes**:

   ```bash
   git add .
   git commit -m "Prepare for release vX.Y.Z"
   ```

4. **Create and Push a New Tag**:

   ```bash
   git tag vX.Y.Z
   git push origin main vX.Y.Z
   ```

   Replace `X.Y.Z` with your new version number.

5. **Monitor GitHub Actions**:

   - Go to your GitHub repository's "Actions" tab.
   - Watch the "Release TensorKiko" workflow to ensure it completes successfully.

6. **Verify Release**:
   - Check the "Releases" page on your GitHub repository.
   - Ensure the new release is listed with the correct assets.
   - Verify that the package is available on PyPI after a few minutes.

## Important Information

### Version Numbering

- Use Semantic Versioning (SemVer): `MAJOR.MINOR.PATCH`
  - Increment MAJOR version for incompatible API changes.
  - Increment MINOR version for new functionality in a backwards compatible manner.
  - Increment PATCH version for backwards compatible bug fixes.

### GitHub Actions Workflow

- The workflow is triggered by pushing a tag starting with "v" (e.g., v1.0.0).
- It automatically builds the package, creates a GitHub release, and publishes to PyPI.

### PyPI Publishing

- Ensure your PyPI API token is up-to-date in GitHub Secrets (`PYPI_API_TOKEN`).
- The workflow uses this token to publish the package to PyPI automatically.

### Troubleshooting

- If the release fails, check the GitHub Actions logs for error messages.
- Common issues include:
  - Incorrect version number in `setup.py` or `__init__.py`.
  - Merge conflicts in `CHANGELOG.md`.
  - Expired PyPI API token.

### Best Practices

- Always test your changes thoroughly before creating a release.
- Keep your `CHANGELOG.md` up-to-date with each release.
- Consider creating a release checklist to ensure consistency.

## Useful Commands

- Check current tags: `git tag`
- Delete a local tag: `git tag -d vX.Y.Z`
- Delete a remote tag: `git push --delete origin vX.Y.Z`

Remember to replace `X.Y.Z` with the actual version numbers in all examples.
