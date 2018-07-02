
def nor_gate(VDD, VTH, VA, VB):
    assert(VDD == 1.0)
    if (VA < VTH and VB < VTH):
        return VDD
    else:
        return 0.0
    
def not_gate(VDD, VTH, VA):
    assert(VDD == 1.0)
    if (VA < VTH):
        return VDD
    else:
        return 0.0
    
def or_gate(VDD, VTH, VA, VB):
    assert(VDD == 1.0)
    if (VA > VTH or VB > VTH):
        return VDD
    else:
        return 0.0


    
