from scripts.run_modern_llm_experiments import parse_generated_label


def test_parse_generated_label_accepts_indonesian_sarkastis_variants():
    assert parse_generated_label("sarkastis") == 1
    assert parse_generated_label("Label: sarkastis") == 1
    assert parse_generated_label("sarkas") == 1
    assert parse_generated_label("mengandung sarkasme") == 1
    assert parse_generated_label("tidak sarkastis") == 0
    assert parse_generated_label("bukan sarkastis") == 0
    assert parse_generated_label("tidak sarkas") == 0


def test_parse_generated_label_accepts_english_labels():
    assert parse_generated_label("sarcastic") == 1
    assert parse_generated_label("not sarcastic") == 0
    assert parse_generated_label("not_sarcastic") == 0
    assert parse_generated_label("non-sarcastic") == 0


def test_parse_generated_label_accepts_explicit_final_answer_and_thinking_blocks():
    assert parse_generated_label("<think>Need classify this.</think>\n\nFinal: sarcastic") == 1
    assert parse_generated_label("<think>Maybe sarcasm.</think>\n\nJawaban: tidak sarkastis") == 0
    assert parse_generated_label("Label: 1") == 1
    assert parse_generated_label("Label: 0") == 0
