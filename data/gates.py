
def nor_gate(VDD, VTH, VA, VB):
    # assert(VDD.all() == 1.0)
    assert(VDD == 1.0)
    on = (VA < VTH) * (VB < VTH)
    ret = on * VDD
    return ret
    
def not_gate(VDD, VTH, VA):
    # assert(VDD.all() == 1.0)
    assert(VDD == 1.0)
    on = (VA < VTH)
    ret = on * VDD
    return ret
    
def or_gate(VDD, VTH, VA, VB):        
    # assert(VDD.all() == 1.0)
    assert(VDD == 1.0)
    on = (VA > VTH) + (VB > VTH)
    ret = on * VDD
    return ret


    
