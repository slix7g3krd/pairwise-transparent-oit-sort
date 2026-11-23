import pytest
import graphics.transparent_oit_sort as mod

CASES = [
    ("pool_overflow", (1,)),
    ("clear_head", (1,)),
    ("stable_key", (1,)),
    ("per_sample_list", (1,)),
    ("compose_fence", (1,)),
    ("determinism", (1,)),
    ("compose", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("list_cap", (1,)),
    ("metrics", (1,)),
    ("fragment_insert", (1,)),
    ("resolve", (1,)),
    ("depth_peel", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
