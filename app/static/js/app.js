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
});
