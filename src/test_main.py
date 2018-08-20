from utils import parse_args

def test_parse_none():
    args = parse_args(None) 
    assert args != None
    assert len(args) == 0

def test_parse_empty():
    args = parse_args("")
    assert args != None
    assert len(args) == 0

def test_parse_whitespace():
    args = parse_args("   ")
    assert args != None
    assert len(args) == 0

def test_parse_normal():
    args = parse_args("--a --b --c")
    assert args == ["--a", "--b", "--c"]

def test_parse_long_with_whitespaces():
    args = parse_args('--a "a very long path" --b')
    assert args == ['--a', 'a very long path', '--b']