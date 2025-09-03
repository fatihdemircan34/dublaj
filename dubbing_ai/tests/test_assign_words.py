from dubbing_ai.artifact_types import ASRSegment, Word, DiarSegment
from dubbing_ai.align.assign_words import assign_words

def test_assign_simple():
    asr = [ASRSegment(id="s1", start=0, end=2, text="", words=[Word(w="a", s=0.1, e=0.5, p=1.0), Word(w="b", s=1.0, e=1.5, p=1.0)])]
    diar = [DiarSegment(spk="S1", start=0.0, end=0.7), DiarSegment(spk="S2", start=0.8, end=2.0)]
    out = assign_words(asr, diar)
    assert out[0].words[0].spk == "S1"
    assert out[0].words[1].spk == "S2"
