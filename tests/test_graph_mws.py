import numpy as np
import pytest
from funlib.geometry import Roi

from volara.blockwise import GraphMWS
from volara.lut import LUT


def test_graph_mws_drop(sqlite_db_2d, tmp_path):
    """drop_artifacts() removes the saved LUT file."""
    lut = LUT(path=tmp_path / "lut.npz")
    lut.save(np.array([[1], [2]]))
    assert lut.file.exists()

    config = GraphMWS(
        roi=Roi((0, 0), (10, 10)),
        db=sqlite_db_2d,
        lut=lut,
        weights={"y_aff": (1, 0)},
    )
    config.drop_artifacts()
    assert not lut.file.exists()


@pytest.mark.parametrize("y_bias", [0.5, -0.5])
def test_graph_mws_merge_split(sqlite_db_2d, block_2d, tmp_path, y_bias):
    """Positive bias merges 2 nodes into 1 segment; negative bias keeps them separate."""
    # Seed DB with 2 nodes connected by an edge with y_aff=0
    db = sqlite_db_2d.open("r+")
    graph = db.read_graph()
    graph.add_node(1, position=(4, 2), size=600, raw_intensity=(0.1,))
    graph.add_node(2, position=(4, 7), size=400, raw_intensity=(0.1,))
    graph.add_edge(1, 2, y_aff=0)
    db.write_graph(graph)

    config = GraphMWS(
        roi=block_2d.read_roi,
        db=sqlite_db_2d,
        lut=LUT(path=tmp_path / "fragment_segment_lut.npz"),
        weights={"y_aff": (1, y_bias)},
    )

    with config.process_block_func() as process_block:
        process_block(block_2d)

    lut = config.lut.load()
    assert lut is not None
    fragments, segments = lut
    assert len(np.unique(fragments)) == 2
    # score = 1*0 + bias. Positive bias -> positive edge -> merge (1 seg).
    # Negative bias -> negative edge -> split (2 segs).
    assert len(np.unique(segments)) == 1 + (y_bias < 0)
