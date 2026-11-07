# Sheng Shi

Personal academic website: https://mikuase-shi.github.io/

An independently rebuilt, responsive page with a warm palette, typographic section index, and light / dark themes. The homepage uses local HTML, CSS, and JavaScript with no package dependencies, external fonts, or icon libraries.

## Edit and preview

- `index.html`: biography, education, research, learning resources, and projects.
- `assets/css/site.css`: shared layout and original warm color values.
- `assets/js/theme.js`: restores the saved theme before the page renders.
- `assets/js/site.js`: accessible theme control and system preference handling.
- `assets/img/`: profile photo, institution logos, favicons, and retained article figures.
- `assets/cv/mylist.pdf`: current downloadable CV.

Run `python3 -m http.server 8000` in this directory and open http://localhost:8000. Push to `main` to publish through the existing GitHub Pages branch deployment.

## Retained blog

The blog remains hidden by the `blog` and `_posts` exclusions in `_config.yml`. Its Markdown content and images are preserved. To restore it, remove those two exclusions and add a navigation link to `/blog/`. Jekyll uses `_layouts/` and `assets/css/blog.css` to render the index and articles; MathJax is loaded only for articles. Local article previews require Jekyll; the plain HTTP server previews the homepage only.

## Repository history

The earlier site used a third-party template. Its obsolete layouts, styles, placeholder images, icon dependencies, and setup documentation have been replaced. `LICENSE.md` preserves the original repository's license notice. Personal text, photos, CV, article content, and institutional marks retain their existing ownership.

## 中文维护说明

首页文案在 `index.html` 中修改，主题与响应式样式在 `assets/css/site.css`。保留原来的暖白 / 陶土橙及深色配色、所有个人资料与链接。无须安装前端依赖，推送 `main` 后由 GitHub Pages 发布。

博客目前仍隐藏，文章和配图均保留；恢复时只需移除 `_config.yml` 中 `blog`、`_posts` 两项排除并添加导航入口。原许可声明保留于仓库，不在页面展示模板署名。
