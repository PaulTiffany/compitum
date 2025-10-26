from compitum.capabilities import Capabilities


def test_capabilities_init():
    cap = Capabilities(regions={"us", "eu"}, tools_allowed={"none"}, deterministic=True)
    assert cap.regions == {"us", "eu"}
    assert cap.tools_allowed == {"none"}
    assert cap.deterministic is True


def test_supports_no_context():
    cap = Capabilities(regions={"us"}, tools_allowed={"none"})
    assert cap.supports(None) is True


def test_supports_with_region_context():
    cap = Capabilities(regions={"us"}, tools_allowed={"none"})
    assert cap.supports(None, context={"region": "us"}) is True
    assert cap.supports(None, context={"region": "eu"}) is False


def test_supports_with_empty_context():
    cap = Capabilities(regions={"us"}, tools_allowed={"none"})
    assert cap.supports(None, context={}) is True


def test_supports_with_irrelevant_context():
    cap = Capabilities(regions={"us"}, tools_allowed={"none"})
    assert cap.supports(None, context={"other_key": "value"}) is True
