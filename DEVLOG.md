# Development Log

## 2026-08-17

### Changed

- Replaced the homepage avatar with a square crop of the new profile photo.
- Updated the displayed name from `Shengshi` to `Sheng Shi`.
- Revised `blog0.txt` to use a more restrained technical style and expanded its mathematical, psychoanalytic, and motor-learning perspectives on JEPA.
- Published the first blog post as an English translation of `blog0.txt`, with both English and Chinese titles.
- Enabled MathJax so GitHub Pages can render inline and display formulas.
- Inserted architecture, manifold, and Lacanian figures into the first blog post.

### Added

- Added a Jekyll-powered blog that preserves the existing Luka visual theme.
- Added an automatically generated post index at `/blog/`.
- Added reusable layouts for the blog index and article pages.
- Moved shared theme switching and section animation behavior into `assets/js/site.js`.
- Added a sample Markdown post that also documents the publishing format.

### Publishing a post

1. Add a Markdown file to `_posts/`.
2. Name it `YYYY-MM-DD-post-title.md`.
3. Include `title`, `date`, `description`, and `tags` in the front matter.
4. Push the file to GitHub; GitHub Pages will build and publish it automatically.
