from dataManipulation.MymeekerStructure import humanHandStruct, shadowHandStruct, shadow4HandStruct, shadow3HandStruct


STRUCT_LIST = {
    'Human': humanHandStruct,
    'Shadow5': shadowHandStruct,
    'Shadow4': shadow4HandStruct,
    'Shadow3': shadow3HandStruct
}

BAD_JOINTS = {
    'Human': ['rh_WRJ1', 'rh_WRJ2'],
    'Shadow5': ['rh_WRJ1', 'rh_WRJ2'],
    'Shadow4': None,
    'Shadow3': None
}