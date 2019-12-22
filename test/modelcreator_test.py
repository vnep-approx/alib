import pytest

from alib.modelcreator import construct_name, build_construct_name


def test_construct_name():
    assert construct_name("foo") == "foo"
    assert construct_name(
        name="node_mapping",
        req_name="req1",
        snode="u",
        vnode="i",
    ) == "node_mapping_req[req1]_vnode[i]_snode[u]"
    assert construct_name(
        "compute_edge_load",
        req_name="req1",
        sedge=("u", "v"),
    ) == "compute_edge_load_req[req1]_sedge[('u','v')]"
    assert construct_name(
        name="1",
        req_name=2,
        type=3,
        vnode=4,
        snode=5,
        vedge=6,
        sedge=7,
        other=8,
        sub_name=9,
        sol_name=10,
    ) == "1_req[2]_type[3]_vnode[4]_snode[5]_vedge[6]_sedge[7]_other[8]_substrate[9]_solution[10]"


def test_build_construct_name():
    # keywords == prefix
    cn = build_construct_name(["a", "b", "c"])
    assert cn("name", a=1, c=2) == "name_a[1]_c[2]"

    # prefix specified
    cn = build_construct_name([("a", "foo"), "b", ("c", "bar")])
    assert cn("name", a=1, b=2, c=3) == "name_foo[1]_b[2]_bar[3]"

    # converter specified
    cn = build_construct_name([("a", None, lambda v: "_".join(sorted(v)))])
    assert cn("name", a={"i", "j", "k"}) == "name_a[i_j_k]"

    # prefix and converter specified
    cn = build_construct_name([("a", "foo", lambda v: "_".join(sorted(v)))])
    assert cn("name", a={"i", "j", "k"}) == "name_foo[i_j_k]"

    # wrong argument
    with pytest.raises(TypeError) as e:
        cn("name", foo="bar")
    assert str(e.value).endswith("got an unexpected keyword argument 'foo'")
