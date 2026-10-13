# Third-party content in this directory

`who_kb.json` reproduces excerpts of a World Health Organization publication.
Those excerpts are **not** covered by the MIT licence of the rest of this
package. They are redistributed under the publication's own licence, with the
attribution it asks for:

> Cause of death assignment by physicians from verbal autopsy data: Manual for
> physician reviewers. Geneva: World Health Organization; 2026. License:
> CC BY-ND 4.0.

- Publication page:
  <https://www.who.int/publications/m/item/cause-of-death-assignment-by-physicians-from-verbal-autopsy-data---pcva-manual-for-physician-reviewers>
- Licence: Creative Commons Attribution-NoDerivatives 4.0 International,
  <https://creativecommons.org/licenses/by-nd/4.0/>
- © World Health Organization 2026. Some rights reserved.

## What is WHO text and what is not

Verbatim WHO text lives in exactly these fields of `who_kb.json`:
`criteria[<code>]`, `criteria_footnotes[<code>][*].who_quote`,
`group_notes[<group>].who_quote`, `shared_guidance[<name>].guidance`, and
`cross_cause_rules[*].who_quote` (plus `who_quote_2` / `who_quote_3` where a
summary needed a second span). The `_meta` block of the file records the source
pages and the excerpt conventions.

Everything else in `who_kb.json` — the cause keys, `analysis_va_codes`,
`icd10_scope`, `scope_note`, each rule's `summary` and `why_it_matters_here`,
and all `_meta` prose — and the whole of `cause_master.json` and
`who_target_list_adult.json` are authored for this package and are under its
MIT licence.

## How the excerpts were handled

The quoted text is reproduced unmodified except for two format conventions:
line and page wraps are flattened, and superscript footnote callouts are
removed from the quoted sentences (the footnote bodies are kept, under
`criteria_footnotes`). No word was changed, added or removed. The package's
tests check every quoted field against the source text.

The WHO logo is not reproduced. Nothing here suggests that WHO endorses this
package, its authors or any product or service. The condensed rule
`summary` fields are this package's own wording and are not WHO text; the
quotation each one is drawn from sits beside it.
