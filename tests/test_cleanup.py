from transcription_pipeline.cleanup import (bad_segment, clean_text, compile_replacements, hallucination_regex,
                                            is_loop, is_prompt_leak, low_diversity, plausible_fill)


def seg(text, start=0.0, end=1.0):
    return {"start": start, "end": end, "text": text, "words": []}


def test_loop_detection():
    assert is_loop("the " * 12)
    assert is_loop("we the people " * 6)
    assert not is_loop("blah, blah, blah, and so on")
    assert not is_loop("a perfectly normal sentence about the weather today")


def test_low_diversity():
    assert low_diversity("November November October November November October November November")
    assert not low_diversity("this sentence has quite a few different words in it")


def test_known_hallucinations_are_bad_segments():
    assert bad_segment(seg("Thanks for watching!"))
    assert bad_segment(seg("Subtitles by the community"))
    assert bad_segment(seg("..."))
    assert not bad_segment(seg("Thanks for joining us today, let's get started."))


def test_extra_hallucination_patterns():
    rx = hallucination_regex([r"see you next week"])
    assert bad_segment(seg("See you next week."), hallu=rx)
    assert not bad_segment(seg("See you next week."))


def test_prompt_leak():
    prompt = "Acme Radio. Hosts Jane Doe, John Roe. Topics: solar panels, heat pumps, grid storage."
    assert is_prompt_leak("Hosts Jane Doe, John Roe. Topics: solar panels", prompt)
    assert not is_prompt_leak("Jane Doe talks about heat pumps", prompt)
    assert not is_prompt_leak("anything at all", "")
    assert bad_segment(seg("Acme Radio. Hosts Jane Doe, John Roe."), prompt=prompt)


def test_plausible_fill():
    assert plausible_fill([seg("one two three four")], 5)
    assert not plausible_fill([seg("one two")], 5)                       # too few words
    assert not plausible_fill([seg("one two three four five")], 60)      # 5 words in 60 s = music
    assert plausible_fill([seg(" ".join(["w"] * 40))], 60)


def test_replacements():
    repl = compile_replacements([[r"\bJon Doe\b", "John Doe"], [r"\bum,?\s*", ""]])
    assert clean_text("um, Jon Doe  said hi", repl) == "John Doe said hi"
