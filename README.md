# AgentForReverseKnowledge

Turn a past exam paper back into the teaching material that would have taught it.

Given a multiple-choice exam PDF, this project generates, for each question, an explanation of the
concept being tested and an analysis of why each option is right or wrong. The name is the idea:
work *backwards* from assessment to knowledge. A past paper tells you what gets tested; it doesn't
tell you why the answer is the answer.

I built this after my own entrance exams were already behind me, as a skills exercise rather than out of need. The starting point was a tooling gap: at the time, chat interfaces could not read a PDF's text layer, so getting a document into a model meant screenshotting each page and running OCR over the image. Removing that step is what the extraction half of this repository is.

Once extraction worked I needed something to point it at, and a past exam paper was a good target on two counts. The reverse-the-assessment idea was interesting on its own, and the document is typographically hostile in useful ways — Traditional Chinese, full-width punctuation, circled numerals, a superscript, and an embedded code block. Most of the findings below come from that hostility rather than from the model.

It is also a small comparison study. The same exam is run through two pipelines that differ in
prompt shape, in grounding rules, and in what text the model is given — though not as independently
as that sounds, because the second arm reuses the first arm's question parsing (see the coupling
note below the table).

Built on [DSPy](https://github.com/stanfordnlp/dspy) with a hand-written
[LiteLLM](https://github.com/BerriAI/litellm) language-model adapter, against
`gemini/gemini-1.5-flash-latest`.

## Source document

`計算機概論.pdf` — the Introduction to Computer Science entrance exam for the in-service master's
program in Information Science at National Taipei University of Education, ROC academic year 99
(2010). 40 multiple-choice questions over 7 pages, Traditional Chinese, covering operating systems,
data structures, digital logic, number systems, networking, and software engineering.

## The two pipelines

| | Experimental arm | Control arm |
|---|---|---|
| Script | `Agent(Experimental group).py` | `OCR(control_group).py` |
| Stem and options | parsed from the PDF text layer, via PyPDF2 | the same text, re-read from `generated_materials.txt` |
| Extra context | none — stem and options only | that question's segment of the image-OCR text |
| Prompt shape | one English textbook-style explanation per question | a separate explanation per option, plus an inferred answer |
| Grounding | model answers from parametric knowledge | model is told to reason from the supplied segment |
| Output | `generated_materials.txt` | `generated_explanations_output.txt` |

**The coupling.** The arms are not independent, and it is worth being precise about where they meet.
The control script reads its structured question list from `generated_materials.txt`, so both arms
see stems and options that came out of the PDF text layer; the image-OCR text reaches the control arm
only through the `context` field. The experimental arm therefore has to run first, and the extraction
comparison below is a comparison of what each source *makes available*, not of two fully separate
pipelines.

**Coverage.** The parsing code path covers all 40 questions. The outputs committed to this repository
predate that rewrite and cover 4 — see Known limitations.

## Findings

**Extraction method decides what is even answerable.** Question 2 asks the reader to put four
boot-sequence steps, labelled ①②③④, into order. The PDF's own text layer flattens those glyphs — the
four options come out as `1234`, `2431`, `2341`, `2314`, and the mapping from label to step is gone.
Image OCR keeps them (`②④③①`). Across the paper the text layer preserves 4 circled numerals against
the OCR pass's 20. The tradeoff runs the other way too: on question 16 the text layer correctly
reads `2ⁿ` and `ln n`, while OCR drops the exponent (`2`) and misreads `ln` as `In`. Neither source
dominates, which is the argument for keeping both extraction paths in the repo rather than picking
one and moving on. Given the coupling above, that argument has a concrete shape here: on question 2
*both* arms receive the flattened `1234` options, and the control arm's OCR context segment is the
only place the ①②③④ mapping still exists for it to recover from.

**A grounded pipeline makes upstream data defects visible; an ungrounded one hides them.** An
earlier revision split questions from options by line position, which truncated every multi-line
stem at the first newline. Question 4's stem reached the model as "…讓使用者退回前一" — cut
mid-sentence — with its continuation misfiled as an option and then discarded. The control arm
refused to answer and said why: the stem was incomplete. Question 3's stem was truncated the same
way, but its OCR context segment happened to carry the missing numbers, so the model recovered and
answered correctly. The experimental arm, working from parametric knowledge with no context to check
against, produced a fluent and correct-looking answer for question 4 regardless — the defect left no
trace in its output at all. The parser has since been rewritten to split on the `(A)`–`(D)` markers
(see below); the committed outputs predate that fix.

There is a deeper limit on the control arm worth stating plainly: its retrieval corpus *is the exam
paper*, which contains the questions but not the knowledge. Even where the segment is complete, it
restates the question rather than explaining it. Question 4's output shows the consequence — the
model names a stack as the answer and then declines to credit it, because nothing in the supplied
text connects a stack to a Back button. Retrieving over the questions you are trying to explain is
circular; a real version of this needs a textbook corpus, not the paper.

**A DSPy adapter incompatibility, diagnosed and fixed.** `failed_run_dspy_adapter.log` is a kept
record of a run where every question failed with 12 stacked tracebacks. DSPy's adapters invoke the
language model with keyword arguments only, passing `messages=[...]` and never a positional
`prompt`; the wrapper's `__call__` declared `prompt` as required and positional. `ChatAdapter`
swallows the resulting `TypeError` and retries through `JSONAdapter`, which fails the same way,
which is why one defect presented as three tracebacks per question and initially looked like a bug
in DSPy. The fix was to accept `**kwargs`, pull out whichever of `messages` or `prompt` is present,
return the raw completion text, and leave output parsing to the adapter.

## How the paper is parsed

Worth a section of its own, because it is where most of the failure modes lived.

Question boundaries are line-initial `N.` markers, accepted only when `N` is exactly the next
expected number. That rejects the in-body numbers that would otherwise look like new questions —
`3.2G` in question 10, `A8.C16` in question 12, `A[1..4，1..5]` in question 31. Option boundaries are
the `(A)`–`(D)` markers themselves rather than line positions, so everything before `(A)` is the
stem however many lines it spans; full-width `（甲）` and the parenthesised code in question 8 do not
collide with the half-width option labels. Page footers (`第 N 頁，共 7 頁`) and the
`※尚有試題…※` page-turn banners are stripped before splitting — they land *between* questions and,
in the earlier revision, ended the parse at the first one, which is why only questions 1–4 were ever
processed. PyPDF2 also inserts stray spaces between CJK characters, so extracted text is normalised
before use.

Verified offline against the real PDF: 40 questions, numbers 1–40 contiguous, exactly four
A–D options each, no banner or footer text in any stem, and the question list round-trips through
the control arm's own parser with a context segment for all 40.

## Running it

Requires Python 3.11+ and a Gemini API key.

```bash
pip install dspy-ai litellm PyPDF2 PyMuPDF python-dotenv
export GEMINI_API_KEY=...        # GOOGLE_API_KEY and a .env file also work
```

```bash
python "get_words.py"                      # PDF text layer -> textLayer.txt
python "Agent(Experimental group).py"      # -> generated_materials.txt
python "OCR(control_group).py"             # reads generated_materials.txt -> explanations
```

The image-OCR step that produced `control_group_litellm_image_ocr.txt` is not in this repository;
that file is committed as a fixed input so the control arm is reproducible without re-running it.
The LiteLLM wrapper still carries the `request_multimodal` path that step used.

## Known limitations

- **The committed outputs are stale.** The pipelines table above describes the code path; these two
  files do not match it yet. `generated_materials.txt` and `generated_explanations_output.txt` are
  from a run that predates the parser rewrite: 4 questions, with the truncated stems described above.
  Both arms need re-running to produce a 40-question result, which is also what would make the
  comparison between them worth anything statistically.
- **OCR scaffolding leaks into the last context segment on each page.** Segments are cut at question
  markers, so the final question on a page also picks up `第1頁,共7頁`, `--- Page Break ---` and
  `--- OCR Start ---`. Visible in question 4's segment. This is the job a stubbed-out cleaning
  function was originally a placeholder for; the stub was removed because it did nothing, but the
  need is real and still open.
- **No answer key, so no scoring.** The control arm reports an inferred answer but nothing checks
  it. The accuracy observations above come from reading the outputs, not from a metric.
- **Question 8's code block survives extraction poorly.** PyPDF2 returns `result` as `r e s u l t`
  inside that snippet. The parser normalises whitespace between CJK characters but does not try to
  reassemble letter-spaced Latin text, which would risk corrupting legitimate content elsewhere.
- **`textLayer.txt` is not consumed by either pipeline.** It exists as the extraction-quality
  comparison behind the first finding above.

## Files

| File | Role |
|---|---|
| `Agent(Experimental group).py` | Experimental arm: PDF text layer → English explanations |
| `OCR(control_group).py` | Control arm: OCR context → per-option explanations + inferred answer |
| `get_words.py` | PyMuPDF text-layer extraction |
| `計算機概論.pdf` | Source exam paper |
| `textLayer.txt` | PDF text layer, for extraction comparison |
| `control_group_litellm_image_ocr.txt` | Image-OCR text, fixed input to the control arm |
| `generated_materials.txt` | Experimental arm output |
| `generated_explanations_output.txt` | Control arm output |
| `failed_run_dspy_adapter.log` | Record of the DSPy adapter failure and its fix |

## Next steps

Re-run both arms over all 40 questions and commit the results. Strip the OCR page scaffolding out of
the context segments. Add an answer key so both arms can be scored rather than described. Then the
interesting experiment: merge the two extraction sources per question, preferring whichever
preserved the glyphs that question depends on, and give the control arm a real textbook corpus
instead of the exam paper it is trying to explain.
