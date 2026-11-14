import pandas as pd
from typing import Union

def ShadowSimplify(data:Union[pd.DataFrame, pd.Series], average:bool) -> Union[pd.DataFrame, pd.Series]:
    '''Simplifies the shadow hand
    Finger tips are coupled together (i.e. rh_FFJ1 + rh_FFJ2, rh_MFJ1 + rh_MFJ2, rh_RFJ1 + rh_RFJ2, rh_LFJ1 + rh_LFJ2)
    
    Args:
        data (pd.DataFrame): the data to simplify
        average (bool): whether to average the finger tips (True) or sum them (False)
        
    Returns:
        simplified (pd.DataFrame): the simplified data
    '''
    simplified = data.copy()

    if isinstance(data, pd.DataFrame):
        def update(ind1, ind2):
            if ind1 in simplified.columns:
                simplified[ind1] += simplified[ind2]
                if average:
                    simplified[ind1] /= 2
                simplified.drop(columns=[ind2], inplace=True)
    else:
        def update(ind1, ind2):
            if ind1 in simplified.index:
                simplified[ind1] += simplified[ind2]
                if average:
                    simplified[ind1] /= 2
                simplified.drop(index=[ind2], inplace=True)

    update('rh_FFJ1', 'rh_FFJ2')
    update('rh_MFJ1', 'rh_MFJ2')
    update('rh_RFJ1', 'rh_RFJ2')
    update('rh_LFJ1', 'rh_LFJ2')

    return simplified

def ShadowDeCouple(data:Union[pd.DataFrame, pd.Series], ratio:float, average:bool) -> Union[pd.DataFrame, pd.Series]:
    '''Decouples the shadow hand
    Finger tips are decoupled (i.e. rh_FFJ1 - rh_FFJ2, rh_MFJ1 - rh_MFJ2, rh_RFJ1 - rh_RFJ2, rh_LFJ1 - rh_LFJ2)
    
    Args:
        data (pd.DataFrame): the data to decouple
        ratio (float): the ratio of J2 to J1 (i.e. J2/J1 = ratio)
        average (bool): whether the total is the average of the finger tips (True) or the sum(False)
        
    Returns:
        decoupled (pd.DataFrame): the decoupled data
    '''
    decoupled = data.copy()

    if isinstance(data, pd.DataFrame):
        if 'rh_FFJ1' in decoupled.columns:
            JT = decoupled['rh_FFJ1']
            if average:
                JT *= 2
            decoupled['rh_FFJ1'] = JT / (1 + ratio)
            decoupled['rh_FFJ2'] = (ratio * JT) / (1 + ratio)
        if 'rh_MFJ1' in decoupled.columns:
            JT = decoupled['rh_MFJ1']
            if average:
                JT *= 2
            decoupled['rh_MFJ1'] = JT / (1 + ratio)
            decoupled['rh_MFJ2'] = (ratio * JT) / (1 + ratio)
        if 'rh_RFJ1' in decoupled.columns:
            JT = decoupled['rh_RFJ1']
            if average:
                JT *= 2
            decoupled['rh_RFJ1'] = JT / (1 + ratio)
            decoupled['rh_RFJ2'] = (ratio * JT) / (1 + ratio)
        if 'rh_LFJ1' in decoupled.columns:
            JT = decoupled['rh_LFJ1']
            if average:
                JT *= 2
            decoupled['rh_LFJ1'] = JT / (1 + ratio)
            decoupled['rh_LFJ2'] = (ratio * JT) / (1 + ratio)
    elif isinstance(data, pd.Series):
        if 'rh_FFJ1' in decoupled.index:
            JT = decoupled['rh_FFJ1']
            if average:
                JT *= 2
            decoupled['rh_FFJ1'] = JT / (1 + ratio)
            decoupled['rh_FFJ2'] = (ratio * JT) / (1 + ratio)
        if 'rh_MFJ1' in decoupled.index:
            JT = decoupled['rh_MFJ1']
            if average:
                JT *= 2
            decoupled['rh_MFJ1'] = JT / (1 + ratio)
            decoupled['rh_MFJ2'] = (ratio * JT) / (1 + ratio)
        if 'rh_RFJ1' in decoupled.index:
            JT = decoupled['rh_RFJ1']
            if average:
                JT *= 2
            decoupled['rh_RFJ1'] = JT / (1 + ratio)
            decoupled['rh_RFJ2'] = (ratio * JT) / (1 + ratio)
        if 'rh_LFJ1' in decoupled.index:
            JT = decoupled['rh_LFJ1']
            if average:
                JT *= 2
            decoupled['rh_LFJ1'] = JT / (1 + ratio)
            decoupled['rh_LFJ2'] = (ratio * JT) / (1 + ratio)
    else:
        raise TypeError("Data is expected to be either a DataFrame or a Series.")

    return decoupled
