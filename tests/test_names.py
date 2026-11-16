from src.data.names import performer_name, strip_company_name, strip_project_names


def test_performer_extracted_from_both_sentence_forms():
    assert performer_name("이 과제는 건국대학교에서 수행했으며, 물리 분야에 속합니다.") == "건국대학교"
    assert performer_name("주식회사락크리트에서 수행한 이 과제는 안전 분야에 속합니다.") == "주식회사락크리트"
    assert performer_name("이 과제는 콘크리트 균열 보수 재료를 개발합니다.") is None


def test_project_names_removed_everywhere():
    text = "(주)가나 테크에서 수행하는 이 과제는 센서를 개발합니다. 가나테크는 양산도 맡습니다."
    out = strip_project_names(text)
    assert "가나" not in out
    assert out.startswith("수행기관에서 수행하는 이 과제는")
    assert out.count("수행기관") == 2


def test_linked_company_names_removed_without_performer_sentence():
    text = "이 과제는 센서를 개발하며 다라전자(주)와 협력합니다."
    assert "다라전자" not in strip_project_names(text, ["(주)다라전자"])
    assert strip_project_names(text) == text


def test_company_own_name_replaced():
    out = strip_company_name("(주)택스세이버는 세무 소프트웨어를 개발합니다. 택스세이버의 제품은…", "(주)택스세이버")
    assert "택스세이버" not in out
    assert out.startswith("이 회사는 세무")
    assert strip_company_name("성원산업은 금속판을 만듭니다.", "성원산업") == "이 회사는 금속판을 만듭니다."
    assert strip_company_name("소프트웨어 기업입니다.", "가") == "소프트웨어 기업입니다."
