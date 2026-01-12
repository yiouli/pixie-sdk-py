# GitHub Pages Setup Instructions

This file contains instructions for setting up GitHub Pages to host the Pixie SDK documentation.

## Quick Setup

### 1. Enable GitHub Pages

1. Go to your repository on GitHub: https://github.com/yiouli/pixie-sdk-py
2. Click on **Settings** (in the repository menu)
3. In the left sidebar, click on **Pages** (under "Code and automation")
4. Under "Build and deployment":
   - **Source**: Select "GitHub Actions"
5. Save the settings

### 2. Verify Workflow Permissions

1. In repository Settings, go to **Actions** > **General**
2. Scroll to "Workflow permissions"
3. Ensure "Read and write permissions" is selected
4. Check "Allow GitHub Actions to create and approve pull requests"
5. Click "Save"

### 3. Trigger Deployment

The documentation will automatically deploy when:

- Changes are pushed to the `main` branch
- Changes are made to files in the `docs/` directory
- Changes are made to `.github/workflows/deploy-docs.yml`

To manually trigger deployment:

1. Go to **Actions** tab in GitHub
2. Select "Deploy Docs to GitHub Pages" workflow
3. Click "Run workflow"
4. Select the `main` branch
5. Click "Run workflow"

## After First Deployment

Once the first deployment completes successfully:

1. The documentation will be available at:

   ```
   https://yiouli.github.io/pixie-sdk-py/
   ```

2. GitHub will show the deployment status in the repository:
   - Go to repository home page
   - Look for "github-pages" in the "Environments" section (right sidebar)
   - Click to see deployment history

## Troubleshooting

### Workflow Fails

**Issue**: GitHub Actions workflow fails during build or deployment

**Solution**:

1. Check the Actions tab for error details
2. Common fixes:
   - Verify Node.js version in workflow matches package.json requirements
   - Check for syntax errors in documentation files
   - Ensure all dependencies are in package.json
   - Run `npm run build` locally to catch errors

### 404 After Deployment

**Issue**: Documentation site shows 404 error

**Solution**:

1. Verify `baseUrl` in `docs/docusaurus.config.ts`:
   ```typescript
   baseUrl: '/pixie-sdk-py/',
   ```
2. Ensure GitHub Pages source is set to "GitHub Actions"
3. Wait a few minutes for GitHub's CDN to update
4. Clear browser cache or try incognito mode

### Changes Not Reflecting

**Issue**: New documentation changes don't appear on the site

**Solution**:

1. Verify changes were pushed to `main` branch
2. Check Actions tab for successful workflow run
3. Wait for deployment to complete (usually 2-5 minutes)
4. Clear browser cache

### Build Works Locally But Fails in CI

**Issue**: `npm run build` works locally but fails in GitHub Actions

**Solution**:

1. Ensure package-lock.json is committed:
   ```bash
   cd docs
   npm install
   git add package-lock.json
   git commit -m "Add package-lock.json"
   ```
2. Check for environment-specific issues in build output
3. Verify all paths are relative, not absolute

## Custom Domain (Optional)

To use a custom domain for your documentation:

1. Add a `CNAME` file to `docs/static/`:

   ```
   docs.pixie-sdk.com
   ```

2. Configure DNS records with your domain provider:

   ```
   Type: CNAME
   Name: docs (or your subdomain)
   Value: yiouli.github.io
   ```

3. In GitHub repository Settings > Pages:
   - Enter your custom domain
   - Check "Enforce HTTPS"

## Maintenance

### Updating Documentation

1. Edit files in `docs/docs/` directory
2. Test locally: `cd docs && npm start`
3. Commit and push changes
4. GitHub Actions will automatically deploy

### Adding New Pages

1. Create new Markdown file in appropriate directory
2. Add frontmatter with `sidebar_position`
3. Update `docs/sidebars.ts` if creating new sections
4. Push changes to trigger deployment

### Monitoring

- Check deployment status: GitHub repository > Environments > github-pages
- View deployment logs: Actions tab > Deploy Docs to GitHub Pages
- Monitor site uptime: https://www.githubstatus.com/

## Additional Resources

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docusaurus Deployment Guide](https://docusaurus.io/docs/deployment)

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review GitHub Actions logs for detailed error messages
3. Search [Docusaurus GitHub Issues](https://github.com/facebook/docusaurus/issues)
4. Ask in [Pixie SDK Discord](https://discord.gg/YMNYu6Z3)
