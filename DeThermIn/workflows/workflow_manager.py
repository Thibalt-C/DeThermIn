from DeThermIn.workflows.metadata import metadata, available_areas
from DeThermIn.workflows import data_blending
from DeThermIn.data import read_datacube
from DeThermIn.workflows import fusion
from DeThermIn.workflows import SR


def pre_process(gdf, start, end, resolution):

    blended_dc = data_blending.launch(gdf, start, end, resolution)
    datacubes = blended_dc.find_attached_datacubes(mode="dict")
    msr_ltr_dc = read_datacube(datacubes["MSR-LTR"]["filepath"])

    return blended_dc, msr_ltr_dc


def spatio_temporal_fusion(blended_dc, lsr_htr_dc, msr_ltr_dc, vlsr_vhtr_dc, start, end):

    msr_vhtr_dc = fusion.launch(start, end, msr_ltr_dc.filepath, lsr_htr_dc.filepath, vlsr_vhtr_dc.filepath)
    blended_dc.fusion(msr_vhtr_dc,inplace=True)

    return blended_dc


def super_resolution(fusioned_dc, mode="dynamic"):

    fusioned_dc.enrich(inplace=True)

    DeThermIn = SR.launch(fusioned_dc.filepath, mode=mode)

    return DeThermIn


def DeThermIn_workflow(gdf, start, end, resolution, small_time_window, mode="dynamic"):

    blended_dc, lsr_htr_dc, msr_ltr_dc, vlsr_vhtr_dc = pre_process(gdf, start, end, resolution)

    small_start, small_end = small_time_window

    fusioned_dc = spatio_temporal_fusion(blended_dc, lsr_htr_dc, msr_ltr_dc, vlsr_vhtr_dc, small_start, small_end)

    DeThermIn = super_resolution(fusioned_dc, mode=mode)

    return DeThermIn