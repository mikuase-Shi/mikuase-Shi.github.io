(function() {
  const themeToggle = document.getElementById("themeToggle");
  const icon = themeToggle.querySelector("i");

  function setTheme(isDark) {
    document.documentElement.setAttribute("data-theme", isDark ? "dark" : "light");
    icon.classList.toggle("fa-sun", isDark);
    icon.classList.toggle("fa-moon", !isDark);
    localStorage.setItem("theme", isDark ? "dark" : "light");
  }

  const savedTheme = localStorage.getItem("theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  setTheme(savedTheme ? savedTheme === "dark" : prefersDark);

  themeToggle.addEventListener("click", function() {
    const isDark = document.documentElement.getAttribute("data-theme") === "dark";
    setTheme(!isDark);
  });

  const sections = document.querySelectorAll(".section");

  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1 });

    sections.forEach(function(section) {
      observer.observe(section);
    });
  } else {
    sections.forEach(function(section) {
      section.classList.add("visible");
    });
  }
})();
