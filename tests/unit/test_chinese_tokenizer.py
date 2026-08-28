from src.core.chinese_tokenizer import tokenize_search_text
from src.core.query_processor import QueryProcessor


def test_tokenize_chinese_and_product_codes():
    tokens = tokenize_search_text("星云网关 ATLAS-X2 出现 E03 怎么重置？")

    assert "星云网关" in tokens
    assert "atlas-x2" in tokens
    assert "e03" in tokens
    assert "怎么" not in tokens


def test_query_processor_preserves_chinese_terms():
    processed = QueryProcessor().process_query("年假怎么申请？")

    assert "年假" in processed.tokens
    assert "申请" in processed.tokens
    assert "年假" in processed.normalized
