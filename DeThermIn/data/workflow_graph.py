from graphviz import Digraph

def add_start(dot):
    dot.node("start", "START", shape="ellipse", style="filled", fillcolor="black", fontcolor="white")
    return "start"


# ORIGINAL DATA

def add_LSR_HTR(dot, start, current=False):
    dot.node("LSR-HTR", "LSR-HTR", shape="box", style="rounded", fontcolor="red" if current else "black")
    dot.edge(start, "LSR-HTR")
    return "LSR-HTR"

def add_VHSR(dot, start, current=False):
    dot.node("VHSR", "VHSR", shape="box", style="rounded", fontcolor="red" if current else "black")
    dot.edge(start, "VHSR")
    return "VHSR"

def add_HSR(dot, start, current=False):
    dot.node("HSR", "HSR", shape="box", style="rounded", fontcolor="red" if current else "black")
    dot.edge(start, "HSR")
    return "HSR"

def add_MSR_LTR(dot, start, current=False):
    dot.node("MSR-LTR", "MSR-LTR", shape="box", style="rounded", fontcolor="red" if current else "black")
    dot.edge(start, "MSR-LTR")
    return "MSR-LTR"

def add_VLSR_VHTR(dot, start, current=False):
    dot.node("VLSR-VHTR", "VLSR-VHTR", shape="box", style="rounded", fontcolor="red" if current else "black")
    dot.edge(start, "VLSR-VHTR")
    return "VLSR-VHTR"

# DATA BLENDING

def add_blending_workflow(dot, datacubes, current=False):
    dot.node("BLENDED", "BLENDED", shape="box", style="rounded", fontcolor="red" if current else "black")
    for type_doc in ["VHSR", "HSR"]:
        if type_doc in datacubes.keys():
            dot.edge(type_doc, "BLENDED")
    return "BLENDED"


# SPATIO-TEMPORAL FUSION

def add_DTC_fitted(dot, vlsr_vhtr, current=False):
    dot.node("VLSR-VHTR-DTC", "VLSR-VHTR-DTC", shape="box", style="rounded", fontcolor="red" if current else "black")
    if vlsr_vhtr is not None:
        dot.edge(vlsr_vhtr, "VLSR-VHTR-DTC")
    return "VLSR-VHTR-DTC"

def add_LSR_VHTR(dot, dtc_fitted, lsr_htr, current=False):
    dot.node("LSR-VHTR", "LSR-VHTR", shape="box", style="rounded", fontcolor="red" if current else "black")
    if dtc_fitted is not None:
        dot.edge(dtc_fitted, "LSR-VHTR")
    if lsr_htr is not None:
        dot.edge(lsr_htr, "LSR-VHTR")
    return "LSR-VHTR"

def add_MSR_VHTR(dot, msr_ltr, lsr_vhtr, current=False):
    dot.node("MSR-VHTR", "MSR-VHTR", shape="box", style="rounded", fontcolor="red" if current else "black")
    if msr_ltr is not None:
        dot.edge(msr_ltr, "MSR-VHTR")
    if lsr_vhtr is not None:
        dot.edge(lsr_vhtr, "MSR-VHTR")
    return "MSR-VHTR"

def add_fusioned(dot, msr_vhtr, blended, current=False):
    dot.node("fusioned", "FUSIONED", shape="box", style="rounded", fontcolor="red" if current else "black")
    if msr_vhtr is not None:
        dot.edge(msr_vhtr, "fusioned")
    if blended is not None:
        dot.edge(blended, "fusioned")
    return "fusioned"


# SUPER-RESOLUTION

def add_enriched(dot, last, current=False):
    dot.node("enriched", "ENRICHED", shape="box", style="rounded", fontcolor="red" if current else "black")
    dot.edge(last, "enriched")
    return "enriched"

def add_training_ready(dot, last, current=False):
    dot.node("TRAININGREADY", "TRAININGREADY", shape="box", style="rounded", fontcolor="red" if current else "black")
    dot.edge(last, "TRAININGREADY")
    return "TRAININGREADY"

def add_DeThermIn(dot, last, current=False):
    dot.node("DeThermIn", "DeThermIn", shape="ellipse", fontcolor="red" if current else "black")
    dot.edge(last, "DeThermIn")
    return "DeThermIn"

def get_workflow_graph(datacubes):

    dot = Digraph("DeThermIn workflow")

    start = add_start(dot)
    last = start

    # Original data

    if "VHSR" in datacubes.keys():
        vhsr = add_VHSR(dot, start, current=datacubes["VHSR"]["current"])

    if "HSR" in datacubes.keys():
        hsr = add_HSR(dot, start, current=datacubes["HSR"]["current"])

    if "MSR-LTR" in datacubes.keys():
        msr_ltr = add_MSR_LTR(dot, start, current=datacubes["MSR-LTR"]["current"])
    else:
        msr_ltr = None

    if "VLSR-VHTR" in datacubes.keys():
        vlsr_vhtr = add_VLSR_VHTR(dot, start, current=datacubes["VLSR-VHTR"]["current"])
    else:
        vlsr_vhtr = None

    if "LSR-HTR" in datacubes.keys():
        lsr_htr = add_LSR_HTR(dot, start, current=datacubes["LSR-HTR"]["current"])
    else:
        lsr_htr = None

    if "BLENDED" in datacubes.keys():
        blended = add_blending_workflow(dot, datacubes, current=datacubes["BLENDED"]["current"])
        last = blended
    else:
        blended = None


    # Spatio-temporal fusion

    if "VLSR-VHTR-DTC" in datacubes.keys():
        dtc_fitted = add_DTC_fitted(dot, vlsr_vhtr, current=datacubes["VLSR-VHTR-DTC"]["current"])
    else:
        dtc_fitted = None

    if "LSR-VHTR" in datacubes.keys():
        lsr_vhtr = add_LSR_VHTR(dot, dtc_fitted, lsr_htr, current=datacubes["LSR-VHTR"]["current"])
    else:
        lsr_vhtr = None

    if "MSR-VHTR" in datacubes.keys():
        msr_vhtr = add_MSR_VHTR(dot, msr_ltr, lsr_vhtr, current=datacubes["MSR-VHTR"]["current"])
    else:
        msr_vhtr = None

    if "FUSIONED" in datacubes.keys():
        fusioned = add_fusioned(dot, msr_vhtr, blended, current=datacubes["FUSIONED"]["current"])
        last = fusioned

    if "ENRICHED" in datacubes.keys():
        enriched = add_enriched(dot, last, current=datacubes["ENRICHED"]["current"])
        last = enriched

    if "TRAININGREADY" in datacubes.keys():
        training_ready = add_training_ready(dot, last, current=datacubes["TRAININGREADY"]["current"])
        last = training_ready

    if "DeThermIn" in datacubes.keys():
        DeThermIn = add_DeThermIn(dot, last, current=datacubes["DeThermIn"]["current"])

    return dot