
import compose
import component
import primitives


# OPTION1

@component
def foo():
    if this is true:
        primitives.1
        primitives.2
        
    else:
        primitives.3
        

@component
def something_else():
    primitives.3
    primtives.4
    
    
    
    
@compose
def main_composite():
    something_else(
    )
    foo()
    
    
agent.run(main_composite)



# OPTION 2
@compose
def foo():
    if this is true:
        primitives.1
        primitives.2
        
    else:
        primitives.3
        

@compose
def something_else():
    primitives.3
    primtives.4
    
    
    
    
@compose
def main_composite():
    something_else()
    foo()
    

agent.run(main_composite)