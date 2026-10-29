from backend.api.services.query_processing import filter_stopwords


def test_basic_filtering():
    assert (
        filter_stopwords("what is information retrieval")
        == "information retrieval"
    )


def test_question_filtering():
    assert (
        filter_stopwords("how does a search engine work")
        == "search engine work"
    )


def test_numeric_query():
    assert filter_stopwords("3/5 of 60") == "3 5 60"


def test_all_stopwords():
    assert filter_stopwords("the and") == ""


def test_case_normalization():
    assert filter_stopwords("WHAT IS INFORMATION") == "information"


def test_empty_query():
    assert filter_stopwords("") == ""
