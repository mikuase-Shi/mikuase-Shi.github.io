// Apply the existing preference before paint. Storage may be unavailable.
(() => {
  let savedTheme;
  try { savedTheme = localStorage.getItem("theme"); } catch {}
  const theme = savedTheme === "light" || savedTheme === "dark"
    ? savedTheme
    : window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  document.documentElement.dataset.theme = theme;
})();
