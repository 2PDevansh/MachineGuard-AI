/**
 * Renders an analog pressure-gauge-style dial into an <svg> element.
 * Pure DOM/SVG, no dependencies.
 */
(function () {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const START_ANGLE = -120; // degrees, 0 = pointing right
  const END_ANGLE = 120;
  const CX = 120, CY = 130, R = 96;

  function polar(cx, cy, r, angleDeg) {
    const rad = (angleDeg - 90) * (Math.PI / 180);
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
  }

  function arcPath(cx, cy, r, a0, a1) {
    const p0 = polar(cx, cy, r, a0);
    const p1 = polar(cx, cy, r, a1);
    const largeArc = a1 - a0 > 180 ? 1 : 0;
    return `M ${p0.x.toFixed(2)} ${p0.y.toFixed(2)} A ${r} ${r} 0 ${largeArc} 1 ${p1.x.toFixed(2)} ${p1.y.toFixed(2)}`;
  }

  function angleForValue(value) {
    const v = Math.max(0, Math.min(100, value));
    return START_ANGLE + (v / 100) * (END_ANGLE - START_ANGLE);
  }

  function el(tag, attrs) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    return node;
  }

  function build(svg) {
    svg.innerHTML = "";

    // background track
    svg.appendChild(el("path", {
      d: arcPath(CX, CY, R, START_ANGLE, END_ANGLE),
      fill: "none", stroke: "#232b32", "stroke-width": 14, "stroke-linecap": "round"
    }));

    // colored risk bands
    const bands = [
      { from: 0, to: 30, color: "#33c98f" },
      { from: 30, to: 65, color: "#f0b429" },
      { from: 65, to: 100, color: "#e5484d" }
    ];
    bands.forEach(b => {
      svg.appendChild(el("path", {
        d: arcPath(CX, CY, R, angleForValue(b.from), angleForValue(b.to)),
        fill: "none", stroke: b.color, "stroke-width": 5, "stroke-linecap": "butt", opacity: 0.55
      }));
    });

    // tick marks
    for (let v = 0; v <= 100; v += 10) {
      const a = angleForValue(v);
      const major = v % 20 === 0;
      const p0 = polar(CX, CY, R + 10, a);
      const p1 = polar(CX, CY, R + (major ? 20 : 15), a);
      svg.appendChild(el("line", {
        x1: p0.x.toFixed(2), y1: p0.y.toFixed(2), x2: p1.x.toFixed(2), y2: p1.y.toFixed(2),
        stroke: "#5b6770", "stroke-width": major ? 1.6 : 1
      }));
      if (major) {
        const lp = polar(CX, CY, R + 32, a);
        const label = el("text", {
          x: lp.x.toFixed(2), y: lp.y.toFixed(2), fill: "#5b6770",
          "font-size": "9", "font-family": "JetBrains Mono, monospace",
          "text-anchor": "middle", "dominant-baseline": "middle"
        });
        label.textContent = v;
        svg.appendChild(label);
      }
    }

    // needle group (rotated via CSS transform later)
    const needleGroup = el("g", { id: "gaugeNeedle", style: "transform-origin: 120px 130px; transition: transform 900ms cubic-bezier(.2,.9,.25,1);" });
    needleGroup.appendChild(el("line", {
      x1: CX, y1: CY, x2: CX, y2: CY - R + 22,
      stroke: "#f5a623", "stroke-width": 3, "stroke-linecap": "round"
    }));
    needleGroup.appendChild(el("circle", { cx: CX, cy: CY, r: 7, fill: "#f5a623" }));
    needleGroup.appendChild(el("circle", { cx: CX, cy: CY, r: 3, fill: "#10151a" }));
    svg.appendChild(needleGroup);

    // set initial rotation to 0-value position
    setValue(svg, 0, true);
  }

  function setValue(svg, value, instant) {
    const needle = svg.querySelector("#gaugeNeedle");
    if (!needle) return;
    const angle = angleForValue(value);
    if (instant) needle.style.transition = "none";
    needle.style.transform = `rotate(${angle}deg)`;
    if (instant) {
      // force reflow then restore transition
      // eslint-disable-next-line no-unused-expressions
      needle.getBoundingClientRect();
      needle.style.transition = "transform 900ms cubic-bezier(.2,.9,.25,1)";
    }
  }

  window.MGGauge = { build, setValue };
})();