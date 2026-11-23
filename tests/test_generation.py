from vi_rag_agent.generation import (
    MAX_PASSAGE_CHARS,
    SYSTEM_CLOSED,
    SYSTEM_RAG,
    build_messages,
    clean_answer,
)


def test_rag_prompt_numbers_passages_in_rank_order_and_truncates():
    msgs = build_messages("Thủ đô?", ["A" * 2000, "B"])
    assert msgs[0] == {"role": "system", "content": SYSTEM_RAG}
    user = msgs[1]["content"]
    assert user.index("[1] ") < user.index("[2] B")
    assert "A" * MAX_PASSAGE_CHARS in user and "A" * (MAX_PASSAGE_CHARS + 1) not in user
    assert user.endswith("Câu hỏi: Thủ đô?\nTrả lời:")


def test_closed_book_prompt_has_no_passages():
    msgs = build_messages("Thủ đô?", None)
    assert msgs[0]["content"] == SYSTEM_CLOSED
    assert "Đoạn văn" not in msgs[1]["content"]


def test_system_prompts_match_the_protocol_text():
    # Copied from docs/protocols/m5_generation.md; a change here needs a new protocol.
    assert SYSTEM_RAG == (
        "Bạn là trợ lý hỏi đáp tiếng Việt. Chỉ dùng thông tin trong các đoạn văn được cung cấp. "
        "Trả lời bằng một cụm từ ngắn, trích nguyên văn từ đoạn văn nếu có thể, không giải thích."
    )


def test_clean_answer():
    assert clean_answer('\n Trả lời: "Hà Nội"\nGiải thích thêm') == "Hà Nội"
    assert clean_answer("   ") == ""
    assert clean_answer("“Sông Hồng”") == "Sông Hồng"
