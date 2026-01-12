# Setup Guide: Daily Releases and TestPyPI Publishing

This guide explains how to complete the setup for automated daily releases and publishing to TestPyPI.

## What Has Been Configured

### 1. Version Management with setuptools-scm

Both `pixie-sdk-py` and `pixie-examples` now use setuptools-scm for automatic version management based on git tags.

**How it works:**

- Version is automatically generated from git tags and commit history
- Format: `X.Y.Z.devN+gHASH` (e.g., `0.1.0.dev5+g1a2b3c4`)
- When you create a tag like `v0.1.0`, the version becomes `0.1.0`

### 2. GitHub Actions Workflows

Two workflows have been created:

- **pixie-sdk-py**: Daily release at 2:00 AM UTC (`.github/workflows/daily-release.yml`)
- **pixie-examples**: Daily release at 3:00 AM UTC (`.github/workflows/daily-release.yml`)

Each workflow:

- Runs on a daily schedule (can also be triggered manually)
- Generates a version using setuptools-scm
- Creates and pushes a git tag
- Builds the package
- Publishes to TestPyPI
- Creates a GitHub release

## Required Setup Steps

### Step 1: Create TestPyPI Account and API Token

1. **Create a TestPyPI account** (if you don't have one):

   - Go to https://test.pypi.org/account/register/
   - Complete the registration process
   - Verify your email

2. **Create API tokens** for each project:

   - Go to https://test.pypi.org/manage/account/token/
   - Click "Add API token"
   - Name: `pixie-sdk-py-github-actions`
   - Scope: "Entire account" (initially, then narrow it down after first upload)
   - Click "Add token"
   - **IMPORTANT**: Copy the token immediately (starts with `pypi-`)
   - You'll need to create separate tokens or use the same token for both repos

3. **After first successful upload**, create project-specific tokens:
   - Go back to https://test.pypi.org/manage/account/token/
   - Create new tokens with scope limited to specific projects
   - Delete the account-wide token for security

### Step 2: Add GitHub Secrets

For **pixie-sdk-py** repository:

1. Go to https://github.com/yiouli/pixie-sdk-py/settings/secrets/actions
2. Click "New repository secret"
3. Name: `TEST_PYPI_API_TOKEN`
4. Value: Paste the API token you copied from TestPyPI
5. Click "Add secret"

For **pixie-examples** repository:

1. Go to https://github.com/yiouli/pixie-examples/settings/secrets/actions
2. Click "New repository secret"
3. Name: `TEST_PYPI_API_TOKEN`
4. Value: Paste the API token (can be the same or different)
5. Click "Add secret"

### Step 3: Create Initial Git Tags

For setuptools-scm to work properly, you need at least one git tag in each repository.

For **pixie-sdk-py**:

```bash
cd /home/yiouli/repo/pixie-sdk-py
git add .
git commit -m "Setup setuptools-scm and daily releases"
git tag -a v0.1.0 -m "Initial release with setuptools-scm"
git push origin main
git push origin v0.1.0
```

For **pixie-examples**:

```bash
cd /home/yiouli/repo/pixie-examples
git add .
git commit -m "Setup setuptools-scm and daily releases"
git tag -a v0.1.0 -m "Initial release with setuptools-scm"
git push origin main
git push origin v0.1.0
```

### Step 4: Test the Workflow

You can manually trigger the workflow to test it before the scheduled run:

1. Go to the Actions tab in each repository:

   - https://github.com/yiouli/pixie-sdk-py/actions
   - https://github.com/yiouli/pixie-examples/actions

2. Click on "Daily Release and Publish" workflow

3. Click "Run workflow" button

4. Select the branch (main) and click "Run workflow"

5. Monitor the workflow execution for any errors

## Verification

After a successful run, verify:

1. **New tag created**: Check the tags page in each repo

   - https://github.com/yiouli/pixie-sdk-py/tags
   - https://github.com/yiouli/pixie-examples/tags

2. **Package published**: Check TestPyPI

   - https://test.pypi.org/project/pixie-sdk/
   - https://test.pypi.org/project/pixie-examples/

3. **GitHub Release created**: Check releases page
   - https://github.com/yiouli/pixie-sdk-py/releases
   - https://github.com/yiouli/pixie-examples/releases

## Testing Installation from TestPyPI

To test installing your packages from TestPyPI:

```bash
# Install pixie-sdk from TestPyPI
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pixie-sdk

# Install pixie-examples from TestPyPI
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pixie-examples
```

Note: The `--extra-index-url` is needed because dependencies will be pulled from the regular PyPI.

## Switching to Production PyPI

Once you've tested with TestPyPI and are ready to publish to production PyPI:

1. **Create PyPI account and tokens**:

   - Register at https://pypi.org/account/register/
   - Create API tokens at https://pypi.org/manage/account/token/

2. **Update GitHub secrets**:

   - Add a new secret named `PYPI_API_TOKEN` (keep `TEST_PYPI_API_TOKEN` for testing)

3. **Update the workflows**:

   - Change `TWINE_PASSWORD: ${{ secrets.TEST_PYPI_API_TOKEN }}`
   - To: `TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}`
   - Change `twine upload --repository testpypi dist/* --verbose`
   - To: `twine upload dist/* --verbose`

4. **Update release notes**:
   - Change TestPyPI URLs to PyPI URLs
   - Remove the `-i https://test.pypi.org/simple/` from install commands

## Troubleshooting

### Issue: "File already exists" error on TestPyPI

TestPyPI (and PyPI) doesn't allow uploading the same version twice. Each release must have a unique version number.

**Solution**: setuptools-scm automatically generates unique versions based on commits, so this shouldn't happen unless you're re-running the workflow on the same commit without new changes.

### Issue: Version contains "+unknown"

**Cause**: setuptools-scm couldn't determine the version from git history.

**Solution**:

- Ensure you have at least one git tag
- Ensure `fetch-depth: 0` is set in the checkout action (already configured)

### Issue: Authentication failed on TestPyPI

**Cause**: Invalid or missing API token.

**Solution**:

- Verify the token is correctly copied (no extra spaces)
- Verify the secret name matches exactly: `TEST_PYPI_API_TOKEN`
- Regenerate the token if necessary

## Schedule Information

- **pixie-sdk-py**: Releases daily at 2:00 AM UTC
- **pixie-examples**: Releases daily at 3:00 AM UTC (1 hour after SDK)

The staggered schedule ensures pixie-sdk is published before pixie-examples (in case of dependencies).

## Version Format

With setuptools-scm, versions follow this pattern:

- Tagged commit: `1.0.0`
- After tag: `1.0.1.dev1+g1234567` (1 commit after 1.0.0)
- Many commits: `1.0.1.dev42+g7654321` (42 commits after 1.0.0)

This ensures every build has a unique, meaningful version number.
