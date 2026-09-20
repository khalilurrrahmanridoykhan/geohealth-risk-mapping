# How to read the division dengue outlook -- and what it is not

*One page for a health officer or analyst who is given the ranked list from
[`notebooks/11_risk_scoring.ipynb`](../notebooks/11_risk_scoring.ipynb).*

## What you are looking at

For each of Bangladesh's 8 divisions: the number of **hospital admissions for dengue per
100,000 people per week, forecast two weeks ahead** of the latest DGHS weekly figures, with a
range around it and a 0-100 score (100 = the highest-forecast division).

It is built from DGHS's public weekly admission counts, recent weather (NASA POWER), and how
fast admissions have been changing. It was checked by re-running the forecast at every week
from week 20 onward using only data available at that time, and comparing to what happened.

## How to read it

- **Read the range, not the rank.** Where two divisions' ranges overlap, do not treat one as
  higher than the other. In the last run there were three tiers (a high group, a middle
  group, and two low divisions), not eight distinct ranks.
- **The ranges are too narrow.** They are labelled 80% but held only about 60% of the time
  in the back-test. Treat them as "likely, not certain".
- **It mostly repeats last week's ranking.** The models predicted *how much* admissions would
  grow better than "no change" did at 2-4 weeks ahead, but the *order* of divisions was
  no better than simply using last week's order. At one week ahead there was no gain at all.
- **The score is relative.** 100 means "highest of the eight", not "high risk" in any
  absolute sense.

## What it is not

- **Not a forecast of infections or of where people get bitten.** It forecasts *reported
  hospital admissions*, which depend on who seeks care, referral patterns (Dhaka's hospitals
  likely receive patients from elsewhere), and how fast hospitals report. The newest weeks are
  the most likely to be revised.
- **Not a ward, union, or upazila map.** Nothing finer than a division exists in the public
  data, and a division-wide average says nothing about any household or neighbourhood
  (the ecological fallacy).
- **Not evidence that weather or land features cause dengue.** With a single season, weather
  is tangled up with the time of year. Population density and surface-water layers added
  nothing to the forecast.
- **Not tested on a different year.** Validation was inside 2026, during a growing epidemic.
  Behaviour near or after the peak is unknown, and the ranges may not hold.
- **Not a basis for allocating resources by itself.** Use it alongside surveillance
  knowledge, local reports and clinical judgement, and treat it as one input.

## Data sources

DGHS HEOC dengue dashboard (division-by-week admissions, snapshot 2026-09-20); NASA POWER
(daily rainfall, temperature, humidity); WorldPop 2020 (population); JRC Global Surface Water.
Week boundaries in the DGHS series are undocumented; ISO weeks were assumed.
