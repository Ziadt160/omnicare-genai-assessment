"""What `ground` is allowed to remove from an answer.

`ground` is the last deterministic check before an answer reaches the
policyholder: it strips citations retrieval never returned and sentences whose
own numbers disprove them. Both are removals, and a removal that fires on a
correct answer is worse than the thing it was guarding against - the reader
gets a mangled sentence, or nothing at all, with no error and no way to tell.

An architecture review found three such false positives, all reproducible
end-to-end and all passing every gate. These cases pin both directions: what
must still be caught, and what must survive untouched. The second list is the
one that matters - it is the product's ordinary output.
"""

from __future__ import annotations

import pytest

from agent.app.graph.nodes import (
    _citation_spans,
    _contradicts_itself,
    _drop_false_comparisons,
)

CITATION_1 = "sample_policy.md § Section 1: Home Water Damage Coverage"
CITATION_2 = "sample_policy.md § Section 2: Personal Property Protection"
RETRIEVED = {CITATION_1, CITATION_2}


# ------------------------------------------------- false comparisons: caught

@pytest.mark.parametrize("sentence", [
    "An appraisal receipt is required since its value $1,500 exceeds $2,500.",
    "An appraisal receipt is needed because $1,500 exceeds $2,500.",
    "Your $4,000 ring is under the $2,500 threshold, so no appraisal is needed.",
])
def test_a_comparison_its_own_numbers_disprove_is_caught(sentence: str) -> None:
    """The failure this check exists for: a 7B told a policyholder their
    $1,500 television "exceeds $2,500" and invented a documentation
    requirement for someone who did not have one."""
    assert _contradicts_itself(sentence)


# ----------------------------------------------- true statements: must survive

@pytest.mark.parametrize("sentence", [
    # The payment split's own wording. "above the $25,000 limit" is not a claim
    # that 9,500 > 25,000, but it was read as one and the whole answer deleted.
    "OmniCare pays $25,000 and you pay $10,000, which is $9,500 above the $25,000 limit.",
    "You pay $10,000 ($500 deductible + $9,500 above the $25,000 limit).",
    # Three amounts, one comparison. Taking max() of the left and min() of the
    # right compared 1,500 against 500 and removed a correct answer.
    "Your $1,500 television is under the $2,500 appraisal threshold, so "
    "OmniCare would pay $1,000 after the $500 deductible.",
    # "under" as a preposition, not a comparison. This is how the assistant
    # names a section, so it appears in a large share of all answers.
    "Jewelry is covered up to $10,000 under the Personal Property Protection "
    "section, and single items over $2,500 need an appraisal.",
    # Correct comparisons.
    "A $4,000 ring exceeds $2,500, so it needs an individual appraisal receipt.",
    "$1,500 is under the $2,500 appraisal threshold, so no receipt is required.",
    # No comparison at all.
    "Sudden pipe bursts are covered up to $25,000 with a $500 deductible.",
])
def test_a_true_statement_is_left_alone(sentence: str) -> None:
    assert not _contradicts_itself(sentence)
    assert _drop_false_comparisons(sentence) == sentence


def test_an_answer_is_never_reduced_to_nothing_by_a_true_sentence() -> None:
    """The worst observed outcome: the worker only emits a token event when the
    text is non-empty, so the policyholder got a completed turn with no message
    and no error."""
    answer = (
        "OmniCare pays $25,000 and you pay $10,000, which is $9,500 above "
        "the $25,000 limit."
    )
    assert _drop_false_comparisons(answer).strip()


# ------------------------------------------------------------ citation spans

def test_a_real_citation_followed_by_prose_survives() -> None:
    """The tail used to be greedy to end-of-line, so a correct citation with a
    sentence after it swallowed the sentence, failed the comparison against the
    canonical form, was judged invented, and was removed - leaving the reader
    with "Under ,000." and a citation-precision score of 1.00."""
    text = (
        f"Under {CITATION_1} a burst pipe is covered up to $25,000."
    )
    spans = _citation_spans(text, RETRIEVED)

    assert len(spans) == 1
    start, end, real = spans[0]
    assert real is True
    assert text[start:end] == CITATION_1


def test_a_real_citation_in_brackets_survives() -> None:
    text = f"Sudden pipe bursts are covered up to $25,000 ({CITATION_1})."
    (start, end, real), = _citation_spans(text, RETRIEVED)
    assert real is True
    assert text[start:end] == CITATION_1


def test_an_abbreviated_citation_is_not_a_fabrication() -> None:
    """gpt-oss-120b writes "(sample_policy.md § Section 1)" for a section whose
    full heading is "Section 1: Home Water Damage Coverage".

    That matched nothing, was deleted as invented, and the confidence was then
    capped *because* something had been deleted - a correct citation punished
    twice for being terse. Accepted only when it identifies exactly one
    retrieved section, so nothing invented survives it."""
    text = "OmniCare will pay $25,000 (sample_policy.md § Section 1)."
    (_s, _e, real), = _citation_spans(text, RETRIEVED)
    assert real is True


def test_a_prefix_matching_several_sections_names_none_of_them() -> None:
    """"§ Section" is a prefix of both headings, so it identifies neither."""
    (_s, _e, real), = _citation_spans("See (sample_policy.md § Section).", RETRIEVED)
    assert real is False


def test_a_citation_to_a_section_that_does_not_exist_is_marked_invented() -> None:
    """EV-08: there is no Section 3 in this policy."""
    text = "There is no such cover (sample_policy.md § Section 3: Earthquake)."
    (start, end, real), = _citation_spans(text, RETRIEVED)
    assert real is False
    assert text[start:end] == "sample_policy.md § Section 3: Earthquake"


def test_an_invented_citation_does_not_eat_the_sentence_after_it() -> None:
    """Bounded removal. Leaving a few words of a fabricated section name is a
    much smaller harm than deleting the answer around it."""
    text = (
        "sample_policy.md § Section 3: Earthquake Coverage says a burst pipe "
        "is covered up to $25,000."
    )
    (start, end, real), = _citation_spans(text, RETRIEVED)

    assert real is False
    removed = text[start:end]
    assert "says a burst pipe" not in removed
    assert (text[:start] + text[end:]).strip()


def test_both_citations_are_found_in_one_answer() -> None:
    text = f"Water damage ({CITATION_1}) and property ({CITATION_2})."
    spans = _citation_spans(text, RETRIEVED)
    assert [real for _s, _e, real in spans] == [True, True]


def test_nothing_retrieved_means_every_citation_is_invented() -> None:
    text = f"Covered ({CITATION_1})."
    (_s, _e, real), = _citation_spans(text, set())
    assert real is False
