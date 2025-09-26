document.addEventListener("DOMContentLoaded", () => {
    const toc = document.querySelector(".bd-toc-nav");
    if (!toc) return;

    // Get all TOC links
    const links = toc.querySelectorAll("a");

    const seen = new Set();
    links.forEach(link => {
      let text = link.textContent.trim();
      let norm = text.replace(/\(\)$/, ""); // strip trailing ()

      if (seen.has(norm)) {
        // hide duplicate
        link.parentElement.style.display = "none";
      } else {
        seen.add(norm);
      }
    });
  });
