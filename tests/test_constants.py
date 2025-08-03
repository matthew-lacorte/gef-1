def test_constants_dict_no_duplicates():
    names = [c.name for c in CONSTANTS]
    assert len(names) == len(set(names)), "Duplicate constant names found"