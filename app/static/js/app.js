// Progressive enhancement for the stepper controls (workload / technicians).
// The underlying <input type="number"> works fine on its own (keyboard,
// screen readers, mobile numeric pad); this just wires the +/- buttons to it
// and keeps them in sync with min/max so they disable at the edges.
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".stepper").forEach((stepper) => {
    const input = stepper.querySelector("input");
    const decBtn = stepper.querySelector('[data-action="dec"]');
    const incBtn = stepper.querySelector('[data-action="inc"]');
    if (!input || !decBtn || !incBtn) return;

    const min = input.min !== "" ? Number(input.min) : -Infinity;
    const max = input.max !== "" ? Number(input.max) : Infinity;

    const sync = () => {
      const val = Number(input.value);
      decBtn.disabled = val <= min;
      incBtn.disabled = val >= max;
    };

    const step = (delta) => {
      const val = Math.min(max, Math.max(min, Number(input.value) + delta));
      input.value = val;
      sync();
    };

    decBtn.addEventListener("click", () => step(-1));
    incBtn.addEventListener("click", () => step(1));
    input.addEventListener("input", sync);
    sync();
  });

  // Dynamic reported_issue dropdown based on selected service_type
  const serviceSelect = document.getElementById("service_type");
  const issueSelect = document.getElementById("reported_issue");

  if (serviceSelect && issueSelect) {
    const issuesMap = window.REPORTED_ISSUES || {};

    const updateIssues = (selectedIssue) => {
      const service = serviceSelect.value;
      const issues = issuesMap[service] || [];
      const currentVal = selectedIssue || issueSelect.getAttribute("data-selected") || issueSelect.value || "";

      issueSelect.innerHTML = '<option value="">— Select issue —</option>';
      issues.forEach((issue) => {
        const opt = document.createElement("option");
        opt.value = issue;
        opt.textContent = issue;
        if (issue === currentVal) {
          opt.selected = true;
        }
        issueSelect.appendChild(opt);
      });
      issueSelect.removeAttribute("data-selected");
    };

    serviceSelect.addEventListener("change", () => updateIssues());

    // Populate on load if service_type already has a selection
    if (serviceSelect.value) {
      updateIssues();
    }
  }
});
