from graphviz import Digraph
from ..metadata import extensions

def add_start(dot):
    dot.node("start", "START", shape="ellipse", style="filled", fillcolor="black", fontcolor="white")
    return "start"


# ORIGINAL DATA

def add_lidar(dot, start, name=extensions["lidar"], current=False):
    dot.node(name, name, shape="box", style="rounded", fontcolor="red" if current else "black")
    dot.edge(start, name)
    return name

def add_multispectral(dot, start, name=extensions["multispectral"], current=False):
    dot.node(name, name, shape="box", style="rounded", fontcolor="red" if current else "black")
    dot.edge(start, name)
    return name

def add_narrow_swath(dot, start, name=extensions["narrow-swath"], current=False):
    dot.node(name, name, shape="box", style="rounded", fontcolor="red" if current else "black")
    dot.edge(start, name)
    return name

def add_geostationary(dot, start, name=extensions["geostationary"], current=False):
    dot.node(name, name, shape="box", style="rounded", fontcolor="red" if current else "black")
    dot.edge(start, name)
    return name

# PRE PROCESSING

def add_blending_workflow(dot, datacubes, name=extensions["blended_ancillary_data"], current=False):
    dot.node(name, name, shape="box", style="rounded", fontcolor="red" if current else "black")
    for type_doc in [extensions["lidar"], extensions["multispectral"]]:
        if type_doc in datacubes.keys():
            dot.edge(type_doc, name)
    return name


# SPATIO-TEMPORAL FUSION

def add_DTC_fitted(dot, geostationary, name=extensions["DTC-fitted"], current=False):
    dot.node(name, name, shape="box", style="rounded", fontcolor="red" if current else "black")
    if geostationary is not None:
        dot.edge(geostationary, name)
    return name

def add_fusioned(dot, narrow_swath, dtc_fitted, name=extensions["fusioned_LSTs"], current=False):
    dot.node(name, name, shape="box", style="rounded", fontcolor="red" if current else "black")
    if narrow_swath is not None:
        dot.edge(narrow_swath, name)
    if dtc_fitted is not None:
        dot.edge(dtc_fitted, name)
    return name

def add_blended(dot, fusioned, ancillary, name=extensions["blended_ancillary_and_LST"], current=False):
    dot.node(name, name, shape="box", style="rounded", fontcolor="red" if current else "black")
    if fusioned is not None:
        dot.edge(fusioned, name)
    if ancillary is not None:
        dot.edge(ancillary, name)
    return name


# SUPER-RESOLUTION

def add_enriched(dot, last, name=extensions["enriched_ancillary_data"], current=False):
    dot.node(name, name, shape="box", style="rounded", fontcolor="red" if current else "black")
    dot.edge(last, name)
    return name

def add_training_ready(dot, last, name=extensions["training_data"], current=False):
    dot.node(name, name, shape="box", style="rounded", fontcolor="red" if current else "black")
    dot.edge(last, name)
    return name

def add_DeThermIn(dot, last, name=extensions["end_product"], current=False):
    dot.node(name, name, shape="ellipse", fontcolor="red" if current else "black")
    dot.edge(last, name)
    return name

def get_workflow_graph(datacubes):

    dot = Digraph("DeThermIn workflow")

    start = add_start(dot)
    last = start

    # Original data

    if extensions["lidar"] in datacubes.keys():
        lidar = add_lidar(dot, start, current=datacubes[extensions["lidar"]]["current"])

    if extensions["multispectral"] in datacubes.keys():
        multispectral = add_multispectral(dot, start, current=datacubes[extensions["multispectral"]]["current"])

    if extensions["narrow-swath"] in datacubes.keys():
        narrow_swath = add_narrow_swath(dot, start, current=datacubes[extensions["narrow-swath"]]["current"])
    else:
        narrow_swath = None

    if extensions["geostationary"] in datacubes.keys():
        geostationary = add_geostationary(dot, start, current=datacubes[extensions["geostationary"]]["current"])
    else:
        geostationary = None

    if extensions["blended_ancillary_data"] in datacubes.keys():
        ancillary = add_blending_workflow(dot, datacubes, current=datacubes[extensions["blended_ancillary_data"]]["current"])
        last = ancillary
    else:
        ancillary = None


    # Spatio-temporal fusion

    if extensions["DTC-fitted"] in datacubes.keys():
        dtc_fitted = add_DTC_fitted(dot, geostationary, current=datacubes[extensions["DTC-fitted"]]["current"])
    else:
        dtc_fitted = None

    if extensions["fusioned_LSTs"] in datacubes.keys():
        fusioned = add_fusioned(dot, narrow_swath, dtc_fitted, current=datacubes[extensions["fusioned_LSTs"]]["current"])
    else:
        fusioned = None

    if extensions["blended_ancillary_and_LST"] in datacubes.keys():
        blended = add_blended(dot, fusioned, ancillary, current=datacubes[extensions["blended_ancillary_and_LST"]]["current"])
        last = blended

    if extensions["enriched_ancillary_data"] in datacubes.keys():
        enriched = add_enriched(dot, last, current=datacubes[extensions["enriched_ancillary_data"]]["current"])
        last = enriched

    if extensions["training_data"] in datacubes.keys():
        training_ready = add_training_ready(dot, last, current=datacubes[extensions["training_data"]]["current"])
        last = training_ready

    if extensions["end_product"] in datacubes.keys():
        DeThermIn = add_DeThermIn(dot, last, current=datacubes[extensions["end_product"]]["current"])

    return dot