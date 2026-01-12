# Quick Reference - Docusaurus Documentation

## Local Development

### Start Development Server

```bash
cd docs
npm start
```

Opens browser to http://localhost:3000

### Build for Production

```bash
cd docs
npm run build
```

Generates static files in `docs/build/`

### Preview Production Build

```bash
cd docs
npm run serve
```

Serves built site locally

### Clear Cache

```bash
cd docs
npm run clear
```

Clears Docusaurus cache if having issues

## Project Structure

```
docs/
├── docs/                    # Documentation content (Markdown)
│   ├── introduction.md
│   ├── quickstart.md
│   ├── tutorial/
│   ├── concepts/
│   └── api/
├── src/
│   ├── components/         # Custom React components
│   ├── css/                # Custom styles
│   └── pages/              # Custom pages (homepage, etc.)
├── static/                 # Static assets (images, files)
│   └── img/
├── docusaurus.config.ts   # Site configuration
├── sidebars.ts            # Sidebar navigation
└── package.json           # Dependencies
```

## Common Tasks

### Add a New Documentation Page

1. Create file in `docs/docs/` (e.g., `docs/docs/new-page.md`)
2. Add frontmatter:

   ```markdown
   ---
   sidebar_position: 5
   ---

   # Page Title

   Content here...
   ```

3. If in a subdirectory, update `sidebars.ts`

### Add an Image

1. Place image in `docs/static/img/`
2. Reference in Markdown:
   ```markdown
   ![Alt text](/img/my-image.png)
   ```

### Add Code Block with Syntax Highlighting

````markdown
```python
from pixie import app

@app
async def my_agent(query: str) -> str:
    return "Hello!"
```
````

### Link to Another Doc Page

```markdown
[Link text](./other-page.md)
[Link to tutorial](./tutorial/setup.md)
```

### Add a Callout/Admonition

```markdown
:::note
This is a note
:::

:::tip
Helpful tip here
:::

:::warning
Warning message
:::

:::danger
Danger/error message
:::
```

## Deployment

### Automatic Deployment

- Pushes to `main` branch automatically deploy
- Only triggers if files in `docs/` or workflow change

### Manual Deployment

1. Go to GitHub Actions tab
2. Select "Deploy Docs to GitHub Pages"
3. Click "Run workflow"
4. Select `main` branch
5. Click "Run workflow"

### Check Deployment Status

- GitHub repo → Environments → github-pages
- Or: Actions tab → latest workflow run

## Configuration

### Update Site Title/Tagline

Edit `docs/docusaurus.config.ts`:

```typescript
title: 'Pixie SDK',
tagline: 'Observability and control for AI agents',
```

### Update Navbar

Edit `docs/docusaurus.config.ts` → `themeConfig.navbar`

### Update Sidebar

Edit `docs/sidebars.ts`

### Update Homepage

Edit `docs/src/pages/index.tsx`

## Troubleshooting

### Build Fails

```bash
cd docs
npm run clear
rm -rf node_modules package-lock.json
npm install
npm run build
```

### Port Already in Use

```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Or use different port
npm start -- --port 3001
```

### Broken Links

- Check for correct relative paths
- Internal links: `./page.md` or `../category/page.md`
- External links: Use full URLs

### Styling Issues

- Edit `docs/src/css/custom.css`
- Clear browser cache
- Hard refresh (Ctrl+Shift+R or Cmd+Shift+R)

## Useful Commands

```bash
# Install dependencies
npm install

# Start dev server
npm start

# Build production
npm run build

# Serve production build
npm run serve

# Clear cache
npm run clear

# Check for broken links
npm run build -- --config docusaurus.config.ts

# Type check
npm run typecheck

# Format code (if configured)
npm run format

# Deploy to GitHub Pages (not recommended, use Actions)
npm run deploy
```

## Documentation Best Practices

### Writing Style

- Use clear, concise language
- Include code examples
- Add prerequisites when needed
- Use headings hierarchically (H1 → H2 → H3)
- Break up long sections

### Code Examples

- Test all code examples
- Include necessary imports
- Add comments for clarity
- Show expected output when relevant

### Navigation

- Use descriptive page titles
- Set appropriate `sidebar_position`
- Link related pages
- Use breadcrumbs

### Maintenance

- Keep examples up-to-date
- Test links regularly
- Update version numbers
- Review for accuracy

## Resources

- **Docusaurus Docs**: https://docusaurus.io/docs
- **Markdown Guide**: https://www.markdownguide.org/
- **Infima Styling**: https://infima.dev/
- **Prism Languages**: https://prismjs.com/#supported-languages

## Need Help?

1. Check this guide
2. Review [Docusaurus documentation](https://docusaurus.io/docs)
3. Search [GitHub issues](https://github.com/facebook/docusaurus/issues)
4. Ask in [Pixie Discord](https://discord.gg/YMNYu6Z3)

## Quick Checklist

Before deploying:

- [ ] All pages have proper frontmatter
- [ ] Code examples are tested
- [ ] Links are working
- [ ] Images are added to `static/img/`
- [ ] Sidebar is configured
- [ ] Build succeeds locally: `npm run build`
- [ ] Production build looks good: `npm run serve`
- [ ] GitHub Pages is enabled in repo settings
- [ ] Workflow permissions are configured

## File Locations

| What                    | Where                               |
| ----------------------- | ----------------------------------- |
| Documentation pages     | `docs/docs/`                        |
| Homepage                | `docs/src/pages/index.tsx`          |
| Custom components       | `docs/src/components/`              |
| Images                  | `docs/static/img/`                  |
| Site config             | `docs/docusaurus.config.ts`         |
| Sidebar config          | `docs/sidebars.ts`                  |
| Custom CSS              | `docs/src/css/custom.css`           |
| GitHub Actions workflow | `.github/workflows/deploy-docs.yml` |

## Live Site

After deployment, documentation is available at:
**https://yiouli.github.io/pixie-sdk-py/**

---

_Last updated: January 12, 2026_
