# ThermoSense: an observation room

Design and verification notes, 16 October 2026.

## Decision and boundary

Make the existing dashboard feel like a considered local weather instrument rather than a generic admin template. Keep all six destinations, existing API contracts, observation submission and pipeline controls. No paid service, dependency, external font, backend change or new network request is needed.

The primary journey is: read the next forecast, understand its time and source, compare it with recent local/reference data, then inspect model evidence. Secondary journeys are observation entry, longer history, baseline comparisons and pipeline jobs.

In scope: hierarchy, spacing, typography, chart semantics, missing/error states, keyboard access and responsive layouts. Out of scope: model accuracy, collecting fresh weather data, inventing readings, a market-size estimate or changes to training jobs.

## Diagnosis and hypotheses

The previous interface made many cards equally important, buried trends under explanations and used tiny unlabelled mobile navigation. It mixed an average-temperature description with a 9 PM snapshot and treated some measured zeros as missing. The metrics radar transformed unlike units into arbitrary scores, obscuring their meaning.

Three testable hypotheses guided this revision:

1. A single forecast focal point and an ordered trend will make the main task clearer than four equal-weight status tiles.
2. Direct units and explicit missing states will make model evidence easier to interpret than a normalized radar.
3. Labelled mobile destinations, connected form labels and visible failure feedback will improve navigation without adding requests or dependencies.

These are design hypotheses, not measured usability results. No user study was performed.

## Plan, in order

| Priority | Change | Acceptance check |
| --- | --- | --- |
| 1 | Lead with tomorrow's temperature and 9 PM meaning, then station record and recent trend | Actual forecast source shown; zero retained; partial API failure cannot invent a forecast |
| 2 | Create a quiet visual system: mineral ground, evergreen rail, pale olive forecast, serif display headings and system sans/mono details | Desktop and narrow-screen visual inspection; no remote fonts |
| 3 | Replace the normalized radar with separate MAE/RMSE grouped bars and a complete table | Actual units, missing metrics stay missing, zero errors remain zero |
| 4 | Bring history, forecasts, leaderboard and pipeline into the same system | Six destinations, observation submission and job controls still work |
| 5 | Label navigation and controls, announce failures, add a skip link and keyboard-focusable table regions | Regression tests and automated accessibility checks |
| 6 | Check assets and rendering under the same workload | Production builds, unchanged request count, paired local browser measurements |

## Design system

- Typography: Georgia for large headings/forecast values; OS sans-serif for controls and prose; OS monospace for recorded numeric detail. These are a design choice, not a claim that a source prescribes these fonts.
- Spacing: a restrained 4/8-based rhythm with larger gaps between tasks, smaller gaps inside a record. Tables remain compact but scroll within labelled regions.
- Color: dark green primary text and navigation, a pale forecast field, green local/reference series, warm dashed API estimate, blue for other comparisons. Color is paired with names, line styles and units.
- Interaction: explicit current navigation destination, 44px standard controls, visible focus, reduced-motion support, retryable failures and inline job feedback instead of browser alerts.
- Charts: linear time series, no entrance animations, latest comparison dates sorted oldest-to-newest, no legend for sources absent from the plotted data. Null is not zero. Actual observations remain visible against the light background.
- Copy: distinguish forecasts from observations, recorded training scores from future accuracy and model artifacts from a verified active model. Keep method explanations in disclosures or below the data.

## Research and comparison lenses

The comparison is task-based, not a competitor ranking. Windy's station guide is useful for keeping the station/source context visible; the Met Office guide is useful for making forecast time meaning explicit. Neither proves this design improves accuracy or usability. Carbon supplies chart-selection and palette principles. NN/g supplies hierarchy/perception guidance. W3C supplies interaction-size guidance. Product pages, independent UX guidance and a primary accessibility standard provide different evidence lenses.

| Source | Evidence used | Coverage / confidence |
| --- | --- | --- |
| [NN/g: visual hierarchy](https://www.nngroup.com/articles/visual-hierarchy-ux-definition/) | Size, grouping and contrast establish an order of attention | General UX guidance; high confidence in principle, local treatment untested with users |
| [NN/g: dashboard perception](https://www.nngroup.com/articles/dashboards-preattentive/) | Use preattentive signals deliberately rather than competing highlights | Independent UX guidance; not ThermoSense-specific |
| [Carbon: chart types](https://carbondesignsystem.com/data-visualization/chart-types/) | Choose charts by comparison/trend purpose; separate unlike metrics | Primary design-system guidance, not a prohibition on every radar chart |
| [Carbon: color palettes](https://www.carbondesignsystem.com/building-blocks/data-visualization/color-palettes) | Keep categorical colors consistent and use accessible contrast | Primary design-system guidance; final colors checked locally |
| [W3C: target-size minimum](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html) | Avoid cramped interactive targets and preserve spacing | Primary accessibility guidance; 44px controls are our choice, not a claim that AA requires 44px |
| [Windy: weather stations](https://windy.app/guide/guide-to-weather-stations-in-windy-app-for-ios.html) | Station observations have a distinct source/context from forecasts | Comparable-product guidance; no visual copying or comparative performance claim |
| [Met Office: forecast meaning](https://weather.metoffice.gov.uk/guides/what-does-this-forecast-mean) | Explain what time a displayed forecast refers to | Official weather-product guide; does not establish ThermoSense's model calibration |

All linked pages were retrieved during this review. GOV.UK typography could not be retrieved and is not used as evidence. No gated or paywalled claims were inferred. Market-sizing and revenue methods do not apply to this interface decision; the decision, boundary, driver hypotheses, source register, customer journey, alternatives, scenarios and QA gate are retained instead.

## Alternatives and what would change the decision

A denser scientific console is better for daily model debugging but worse for a quick temperature check. A map-first weather app would need new geospatial data and would distract from this station's record. The observation-room direction best fits the existing single-location task.

If users mostly run jobs, promote the pipeline entry and job status. If they mostly compare models, put held-out evaluation ahead of forecast decoration. If forecasts are often missing, keep the unavailable state prominent rather than filling the hero with an inferred estimate. A short task-based user review should check these assumptions after deployment.

## Verification and limits

The screenshots use deterministic fixture data, not a claim about today's real temperature or model quality. Automated accessibility checks do not prove full accessibility. Production data/backend integration and assistive-technology testing remain separate checks. No live training or backfill job was launched to test appearance.

The regression tests preserve zero readings/errors, missing metrics, labelled navigation/forms, forecast day selection, observation submission, job-error feedback and retryable failures. Three accessibility/error tests fail on the original UI and pass after the revision. Python tests protect the unchanged backend. See the PR for exact run totals and performance measurements.

The frontend is JavaScript; there is no separate TypeScript gate. A production build runs the configured CRA lint/build checks. No lockfile or dependency changes are included. Revert the redesign commit to restore the previous interface without reverting unrelated backend fixes.

### Recorded gate results

- Frontend: 21 tests across three suites, passing. The three hard accessibility/error regressions fail on the baseline.
- Backend: 232 tests passing, five existing warnings.
- ESLint: zero errors/warnings under `eslint src --max-warnings 0`; production build succeeds. The existing Browserslist dataset emits its age notice; no dependency update is bundled.
- Automated WCAG A/AA checks: zero violations across six pages at 1440, 720, 390 and 320 CSS-pixel widths. No page-wide horizontal overflow in those fixtures. The 720px layout checks the reflow width corresponding to a 1440px display at 200%, not a claim of full browser-zoom/assistive-technology certification.
- Browser smoke: keyboard skip link focuses main content; labelled navigation; forecast error followed by refresh recovery; all six empty states; no uncaught page errors with valid empty API responses.
- Production gzip: main JS 196.35 to 187.60 kB; CSS 3.12 to 4.70 kB. Combined JS/CSS falls by about 7.17 kB. Dependency/lock files unchanged.
- Controlled render proxy: seven fresh-context runs per revision after one warmup pair, alternating order, Chromium on local HTTP, 1440x1000, reduced motion, same 30 history rows and three forecasts. Two animation frames after the populated heading: baseline median 330.3 ms (325.3-373.6), redesign 356.1 ms (342.9-376.8). Both make three API requests. This narrow local proxy is 25.8 ms slower, not a speedup; it does not measure production networks, lower-end devices or model inference. The redesign removes chart entrance animations, so it also avoids waiting for an animated data reveal.

Visually inspected settled before/after dashboard screenshots, all six revised pages on mobile/desktop, 320px navigation, and error/empty states. Tables deliberately retain horizontal scrolling within their own focusable region. Production deployment and real-data smoke remain the publication follow-up.
