from tdc.multi_pred import DTI

def load_data(threshold=20):
    data = DTI(name='BindingDB_Kd')

    # Step 1: harmonize
    data.harmonize_affinities(mode='mean')

    # Step 2: binarize
    data.binarize(threshold=threshold, order='descending')

    df = data.get_data()
    return df