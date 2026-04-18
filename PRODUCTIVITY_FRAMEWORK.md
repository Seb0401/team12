# The Productivity Framework
### **Layer 1 — The industry-standard equation (your anchor)**

In open-pit mining, shovel productivity is universally expressed as:

```
Productivity (tonnes/hour) = Payload × Cycles/hour × Availability × Utilization × Efficiency
```

Broken down:

| **Term**         | **Meaning**                                       | **Typical range**                          |
| ---------------- | ------------------------------------------------- | ------------------------------------------ |
| **Payload**      | Tonnes moved per bucket                           | EX-5600 nominal bucket ≈ 29 m³ ≈ 52 tonnes |
| **Cycles/hour**  | How many dig→dump cycles per hour                 | 60–120 depending on conditions             |
| **Availability** | % of shift the shovel is mechanically operational | 85–92% (out of your scope)                 |
| **Utilization**  | % of available time actually digging              | 75–88%                                     |
| **Efficiency**   | Actual output ÷ theoretical best                  | the hidden lever                           |

This is the **OEE-equivalent for mining** (Overall Equipment Effectiveness). Citing this framework signals to judges you know the domain.

### **Layer 2 — What YOU can actually measure in 15 min**

Be honest about scope. With stereo video + IMU and no weight sensor, you **cannot** directly measure payload or availability. You can measure:

| **Measurable**                           | **Source**                       | **Certainty** |
| ---------------------------------------- | -------------------------------- | ------------- |
| **Cycle count**                          | IMU yaw (quaternion)             | High          |
| **Cycle time** (total + per-phase)       | IMU yaw + accel                  | High          |
| **Swing angle**                          | Quaternion                       | High          |
| **Truck-exchange gap**                   | IMU idle + video truck detection | Medium        |
| **Dig energy proxy** (payload surrogate) | Accel magnitude during dig       | Medium        |
| **Operator consistency**                 | Cycle-time variance              | High          |
| **Bucket fill quality** (rough)          | Stereo video near bucket         | Low           |

**So your operational definition of productivity becomes:**

> _"Effective cycles per hour, weighted by dig-energy (payload proxy), discounted by avoidable idle time."_

That one sentence is what you put on slide 1.

### **Layer 3 — The 4 productivity levers (your narrative structure)**

Every productivity improvement in mining falls into one of four buckets. Frame your recommendations around these.

**1. Cycle speed** — How fast is each cycle?

* Measured as: median cycle time, per-phase time

* Improvement lever: operator training, swing angle reduction, dig technique

**2. Cycle consistency** — How much does cycle time vary?

* Measured as: coefficient of variation, P90/P50 ratio

* Improvement lever: standard operating procedures, real-time feedback

* _This is your most defensible insight — variance is almost always the hidden tax._

**3. Idle reduction** — How much of the shift is non-productive?

* Measured as: truck-exchange gap distribution, micro-stoppages

* Improvement lever: truck dispatch optimization, spotting protocols

**4. Per-cycle yield** — How much material per cycle?

* Measured as: dig-energy proxy, bucket-fill visual estimation

* Improvement lever: dig technique, face management, bucket teeth condition

### **Layer 4 — The diagnostic pyramid (how you present findings)**

```
              [ Productivity: tonnes/hour ]                        ▲              ┌─────────┴─────────┐         Cycles/hour          Tonnes/cycle              ▲                    ▲      ┌───────┴───────┐            │  Cycle time     Idle time    Dig quality      ▲              ▲              ▲  ┌───┴───┐      ┌───┴───┐      ┌───┴───┐  Dig  Swing   Truck gap  Micro  Fill  Energy       angle              stops
```

Every metric you compute should map to one node. If it doesn't, drop it.

### **Layer 5 — The measurement contract (what judges want to hear)**

For each metric, have a 3-part statement ready:

| **Piece**      | **Example**                                                                          |
| -------------- | ------------------------------------------------------------------------------------ |
| **Definition** | "Cycle time = duration from start of dig to start of next dig"                       |
| **Signal**     | "Detected as yaw-angle reversals >90° in the quaternion-derived heading"             |
| **Limit**      | "Accurate to ±1 sec; cannot distinguish operator hesitation from pile repositioning" |

The "Limit" part is what separates finalists from winners. **Admitting what you can't measure is a strength, not a weakness.** It shows data grounding.

### **Layer 6 — The value translation (what makes it actionable)**

Numbers don't win; **translated numbers do.** Every finding needs a line of the form:

> _"If the operator reduced top-quartile truck-exchange gaps to the median, we'd gain ~4 cycles/hour → at a nominal 52 t bucket, that's +208 tonnes/hour → at 20 hrs/day operation, +4,160 tonnes/day → at $X/tonne margin, $Y/day."_

You don't need the exact dollar figure (you don't have cost data), but the **tonnes/day** number is defensible and tangible. That's your pitch slide.

### **Layer 7 — The "what we're NOT claiming" boundary**

List this explicitly in your report. It's your shield in Q&A.

* We are not measuring actual payload (no weight sensor).

* We are not measuring fuel/energy consumption.

* We are not accounting for geotechnical constraints (pile stability, face conditions).

* We are not recommending equipment changes — only operator/dispatch decisions.

* Our 15-min window may not be representative of a full shift.

When a judge asks a "gotcha" question, you've already answered it.

## **The one-paragraph definition (memorize this)**

> _"We define shovel productivity as _**_effective loaded cycles per hour_**_, decomposed into dig, swing-loaded, dump, and swing-empty phases using IMU-derived heading and acceleration. We weight each cycle by a dig-energy proxy to approximate payload variation, and subtract avoidable idle time — primarily truck-exchange gaps exceeding the operator's own median. This definition is measurable from the provided sensors alone, maps directly to the industry-standard tonnes/hour equation, and yields recommendations the operator can act on without new equipment."_

That's ~70 seconds spoken. Deliver it verbatim and you've won the "clarity of thought" criterion before you've shown a single chart.

## **The insight hierarchy (what to hunt for, ranked by judge-impact)**

Aim up this ladder during Block 2. Stop when time runs out.

1. **Level 1 (baseline):** "We counted N cycles at median T seconds." _Every team has this._

2. **Level 2 (decomposition):** "Swing accounts for X% of cycle, dig Y%, dump Z%." _Most finalists have this._

3. **Level 3 (variance):** "Top-decile cycles are 22% faster than median — the operator is capable of this pace." _Differentiator._

4. **Level 4 (causal):** "Long cycles correlate with swing angles >120° — swing reduction is the highest-leverage intervention." _Winner territory._

5. **Level 5 (actionable):** "Restricting swing to <100° via truck positioning would add ~N cycles/hour = ~M tonnes/day." _Trophy._
