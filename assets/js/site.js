(() => {
  const button = document.getElementById("themeToggle");
  if (!button) return;
  const root = document.documentElement;
  const systemTheme = window.matchMedia("(prefers-color-scheme: dark)");
  let hasPreference = false;
  try { hasPreference = ["light", "dark"].includes(localStorage.getItem("theme")); } catch {}

  function applyTheme(theme) {
    root.dataset.theme = theme;
    const label = theme === "dark" ? "Switch to light theme" : "Switch to dark theme";
    button.setAttribute("aria-label", label);
    button.setAttribute("title", label);
  }

  applyTheme(root.dataset.theme || (systemTheme.matches ? "dark" : "light"));
  button.hidden = false;
  button.addEventListener("click", () => {
    const theme = root.dataset.theme === "dark" ? "light" : "dark";
    hasPreference = true;
    applyTheme(theme);
    try { localStorage.setItem("theme", theme); } catch {}
  });
  systemTheme.addEventListener("change", (event) => {
    if (!hasPreference) applyTheme(event.matches ? "dark" : "light");
  });
})();
